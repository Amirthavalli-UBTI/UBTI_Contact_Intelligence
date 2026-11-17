from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from html import escape as esc
from typing import Iterator
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xesc

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Theme set in code (replaces .streamlit/config.toml). Must run before set_page_config.
for _k, _v in {
    "theme.base": "light", "theme.primaryColor": "#6A41E0", "theme.backgroundColor": "#FFFFFF",
    "theme.secondaryBackgroundColor": "#F1F5F9", "theme.textColor": "#1F1B3A",
    "browser.gatherUsageStats": False,
}.items():
    try:
        st._config.set_option(_k, _v)
    except Exception:
        pass

log = logging.getLogger("ubti.lookup")
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="UBTI Contact Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================================
# SECTION 1 — Constants, settings, caches
# ======================================================================================

FREE_MAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com", "icloud.com",
    "aol.com", "proton.me", "protonmail.com", "zoho.com", "rediffmail.com", "yahoo.co.in",
}
PEOPLE_SITES = [
    "linkedin.com", "zoominfo.com", "rocketreach.co", "theorg.com",
    "contactout.com", "signalhire.com", "crunchbase.com", "apollo.io",
]
CONFIDENCE = ("high", "medium", "low", "not_found")


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str = field(default="", repr=False)
    azure_key: str = field(default="", repr=False)
    deployment: str = "gpt-4.1"
    tavily_key: str = field(default="", repr=False)
    tavily_depth: str = "basic"
    search_timeout: int = 20
    websearch_budget: int = 30
    synth_timeout: int = 60
    company_wait: int = 90
    io_workers: int = 12
    max_contacts: int = 25

    @property
    def has_azure(self) -> bool:
        return bool(self.azure_endpoint and self.azure_key and self.deployment)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_key)

    @property
    def ready(self) -> bool:
        return self.has_azure and self.has_tavily

    @property
    def engine_label(self) -> str:
        return "Tavily + Azure web search" if self.has_tavily else "Azure web search"


def load_settings() -> Settings:
    def i(k: str, d: int) -> int:
        try:
            return max(1, int(os.getenv(k, str(d)) or d))
        except ValueError:
            return d

    depth = (os.getenv("TAVILY_SEARCH_DEPTH", "basic") or "basic").strip().lower()
    return Settings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
        azure_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
        deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1").strip(),
        tavily_key=os.getenv("TAVILY_API_KEY", "").strip(),
        tavily_depth=depth if depth in ("basic", "advanced") else "basic",
        websearch_budget=i("WEBSEARCH_BUDGET_SECONDS", 30),
        io_workers=i("IO_WORKERS", 12),
        max_contacts=i("MAX_CONTACTS", 25),
    )


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return load_settings()


S = get_settings()


class TTLCache:
    def __init__(self, ttl: int = 86400, maxsize: int = 3000):
        self.ttl, self.maxsize = ttl, maxsize
        self._d: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._d.get(key)
            if not hit:
                return None
            if hit[0] < time.time():
                self._d.pop(key, None)
                return None
            return copy.deepcopy(hit[1])

    def set(self, key, value) -> None:
        with self._lock:
            if len(self._d) >= self.maxsize:
                self._d.pop(next(iter(self._d)))
            self._d[key] = (time.time() + self.ttl, copy.deepcopy(value))

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


@st.cache_resource(show_spinner=False)
def _caches() -> tuple[TTLCache, TTLCache]:
    return TTLCache(), TTLCache()


SEARCH_CACHE, PROFILE_CACHE = _caches()


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


# ======================================================================================
# SECTION 2 — Contact parsing (text + image)
# ======================================================================================

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
HEADER_RE = re.compile(r"^\s*(from|sent|to|cc|bcc|subject|date)\s*:\s*(.*)$", re.I)


@dataclass
class Contact:
    email: str
    display_name: str = ""
    local: str = ""
    domain: str = ""
    name_guess: str = ""
    company_hint: str = ""
    apollo: dict = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.domain.split(".")[0]


def guess_name_from_local(local: str) -> str:
    parts = [p for p in re.split(r"[._\-+]", local) if p and not p.isdigit()]
    return " ".join(p.capitalize() for p in parts)


def _chunks_to_contacts(text: str, hint: str, seen: set[str]) -> list[Contact]:
    out: list[Contact] = []
    for chunk in re.split(r"[;\n]|,(?![^<]*>)", text):
        chunk = chunk.strip()
        m = EMAIL_RE.search(chunk)
        if not m:
            continue
        email = m.group(0).lower().strip("'")
        if email in seen:
            continue
        seen.add(email)
        display = re.sub(r"\s+", " ", chunk[: m.start()].strip().strip("<>\"' ,"))
        local, domain = email.split("@", 1)
        out.append(Contact(email=email, display_name=display, local=local, domain=domain,
                           name_guess=display or guess_name_from_local(local),
                           company_hint=hint.strip()))
    return out


def parse_contacts(raw: str, company_hint: str = "", include_sender: bool = False) -> list[Contact]:
    text = re.sub(r"\[([^\]]+)\]\(mailto:([^)?\s]+)[^)]*\)", r"\1 <\2>", raw or "")
    lines = text.splitlines()
    if not any(HEADER_RE.match(l) and EMAIL_RE.search(l) for l in lines):
        return _chunks_to_contacts(text, company_hint, set())

    wanted = {"to", "cc", "bcc"} | ({"from"} if include_sender else set())
    seen: set[str] = set()
    out: list[Contact] = []
    current: str | None = None
    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            current, body = m.group(1).lower(), m.group(2)
        else:
            body = line
            if not line.strip():
                current = None
        if current in wanted:
            out.extend(_chunks_to_contacts(body, company_hint, seen))
    return out


IMAGE_EXTRACT_PROMPT = """You are an assistant that extracts contact information from images of emails, meeting invites,
business cards, chat screenshots, or any document that contains people's names and email addresses.

Extract ALL people mentioned with their:
- full_name: the person's display name
- email: their email address (if visible)
- company_hint: the company name if you can determine it from the domain, signature, or context

Return a JSON object with this exact shape:
{
  "contacts": [
    {"full_name": "...", "email": "...", "company_hint": "..."}
  ],
  "company_context": "any overall company or meeting context you can extract",
  "raw_text": "the full text you can read from the image, preserving structure"
}

Rules:
- Extract EVERY person, including From/sender
- If you can see email addresses, include them exactly as shown
- If you can only see names without emails, still include them with email as ""
- For company_hint, derive from email domain or any company names visible
- In raw_text, include everything you can read — headers, subject lines, signatures, body text
- Be thorough — do not skip anyone
- Return ONLY the JSON object, no markdown or explanation"""

SMART_PARSE_PROMPT = """You are an assistant that extracts contact information from messy, unstructured text.

Extract ALL people you can identify with their:
- full_name: the person's name
- email: their email address (if present)
- company_hint: company name if you can determine it

Return a JSON object:
{
  "contacts": [
    {"full_name": "...", "email": "...", "company_hint": "..."}
  ],
  "company_context": "overall company or context you can determine"
}

Rules:
- Include EVERY person mentioned
- If names appear without emails, include them (email: "")
- Look for company names ANYWHERE in the text
- Handle Indian name patterns: initials like "Ganesh R", "Venkatesh M"
- An email domain like @ak.com.sa means the company might be "Al Abdul Karim" or similar
- Return ONLY the JSON, no markdown"""


def _llm_entries_to_contacts(data: dict, hint: str, seen: set[str]) -> list[Contact]:
    out: list[Contact] = []
    for entry in data.get("contacts") or []:
        if not isinstance(entry, dict):
            continue
        email = str(entry.get("email") or "").strip().lower()
        name = str(entry.get("full_name") or "").strip()
        c_hint = str(entry.get("company_hint") or hint).strip()
        if email and EMAIL_RE.fullmatch(email) and email not in seen:
            seen.add(email)
            local, domain = email.split("@", 1)
            out.append(Contact(email=email, display_name=name, local=local, domain=domain,
                               name_guess=name or guess_name_from_local(local), company_hint=c_hint))
    return out


def contacts_from_image(image_bytes: bytes, s: Settings, mime: str = "image/png") -> list[Contact]:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    client = azure_client(s, s.synth_timeout + 30, retries=2)
    resp = client.chat.completions.create(
        model=s.deployment,
        messages=[
            {"role": "system", "content": "You extract contact information from images. Return only valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": IMAGE_EXTRACT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
            ]},
        ],
        response_format={"type": "json_object"},
    )
    data = extract_json(resp.choices[0].message.content)
    hint = str(data.get("company_context") or "").strip()
    seen: set[str] = set()
    contacts = _llm_entries_to_contacts(data, hint, seen)
    raw_text = str(data.get("raw_text") or "")
    if raw_text:
        contacts.extend(_chunks_to_contacts(raw_text, hint, seen))
    return contacts


def smart_parse(raw: str, s: Settings, company_hint: str = "", include_sender: bool = False) -> list[Contact]:
    fast = parse_contacts(raw, company_hint, include_sender)
    if fast:
        return fast
    prompt = f"Extract contacts from this text:\n\n{raw}"
    if company_hint:
        prompt += f"\n\nAdditional context from the user: {company_hint}"
    data = llm_json(s, SMART_PARSE_PROMPT, prompt)
    hint = str(data.get("company_context") or company_hint).strip()
    return _llm_entries_to_contacts(data, hint, set())


# ======================================================================================
# SECTION 3 — Search backends
# ======================================================================================

def tavily_search(s: Settings, query: str, include_domains: tuple[str, ...] | None = None,
                  max_results: int = 5) -> list[dict]:
    key = ("tav", query, s.tavily_depth, include_domains or (), max_results)
    hit = SEARCH_CACHE.get(key)
    if hit is not None:
        return hit
    payload: dict = {"query": query, "search_depth": s.tavily_depth, "max_results": max_results}
    if include_domains:
        payload["include_domains"] = list(include_domains)
    r = None
    for attempt in range(3):
        r = requests.post("https://api.tavily.com/search",
                          headers={"Authorization": f"Bearer {s.tavily_key}"},
                          json=payload, timeout=s.search_timeout)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            try:
                wait_s = float(r.headers.get("Retry-After", ""))
            except ValueError:
                wait_s = 1.5 * (attempt + 1)
            time.sleep(min(wait_s, 8))
            continue
        break
    r.raise_for_status()
    out = [{"title": x.get("title", ""), "url": x.get("url", ""), "content": (x.get("content") or "")[:700]}
           for x in r.json().get("results", []) if x.get("url")]
    SEARCH_CACHE.set(key, out)
    return out


def azure_client(s: Settings, timeout: int, retries: int) -> OpenAI:
    m = re.match(r"(https?://[^/\s]+)", s.azure_endpoint.strip())
    base = (m.group(1) if m else s.azure_endpoint.rstrip("/")) + "/openai/v1/"
    return OpenAI(base_url=base, api_key=s.azure_key, timeout=timeout, max_retries=retries)


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else {"_parse_error": text[:300]}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, dict):
                return val
        except Exception:
            pass
    return {"_parse_error": text[:300]}


def llm_json(s: Settings, system: str, user: str) -> dict:
    client = azure_client(s, s.synth_timeout, retries=2)
    data: dict = {}
    for _ in range(2):
        resp = client.chat.completions.create(
            model=s.deployment,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        data = extract_json(resp.choices[0].message.content)
        if "_parse_error" not in data:
            return data
        user += "\n\nReturn ONLY one valid JSON object."
    return data


def azure_web_search(s: Settings, prompt: str) -> tuple[dict, list[str]]:
    key = ("ws", _h(prompt, s.deployment))
    hit = SEARCH_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1]
    client = azure_client(s, s.websearch_budget + 15, retries=0)
    resp = client.responses.create(model=s.deployment, tools=[{"type": "web_search"}], input=prompt)
    urls: list[str] = []
    for item in getattr(resp, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            for ann in getattr(part, "annotations", None) or []:
                u = getattr(ann, "url", None)
                if u and u not in urls:
                    urls.append(u)
    data = extract_json(getattr(resp, "output_text", "") or "")
    SEARCH_CACHE.set(key, [data, urls])
    return data, urls


# ======================================================================================
# SECTION 4 — Research prompts
# ======================================================================================

COMPANY_SCHEMA = """{
  "company_name": str, "domain": str, "headquarters": str, "other_offices": [str],
  "industry": str, "what_they_do": str, "products": [str], "size": str,
  "parent_or_group": str, "key_leadership": [{"name": str, "title": str}], "sources": [url]
}"""

PERSON_SCHEMA = """{
  "full_name": str, "current_title": str, "current_company": str, "location": str,
  "department": str, "ai_relevance": "Very High" | "High" | "Medium" | "Low",
  "notes": str, "previous_roles": [{"title": str, "company": str, "period": str}],
  "summary": str, "linkedin_url": str, "confidence": "high" | "medium" | "low" | "not_found",
  "match_reasoning": str, "sources": [url]
}"""

SYSTEM_RULES = (
    "You are a B2B research analyst. Use ONLY the evidence provided. "
    "Never invent names, titles or URLs; use \"\" or [] for anything you cannot support. "
    "Matching rules: display name and email local part must fit the person. "
    "A profile at a related group company is a valid match; explain the link in match_reasoning. "
    "Sources must be URLs from the evidence. Return a single JSON object only.\n\n"
    "DEPARTMENT & AI RELEVANCE rules:\n"
    "- department: infer from title (CTO -> IT, CFO -> Finance, CHRO -> HR, COO -> Operations, "
    "CMO -> Marketing, CEO/MD -> Executive Leadership).\n"
    "- ai_relevance: 'Very High' for CIO/CTO/CDO/Head of AI; 'High' for VP/Director Eng/Analytics; "
    "'Medium' for CFO/COO/CHRO; 'Low' otherwise.\n"
    "- notes: exactly 2 lines separated by newline, each under 80 chars."
)

PLAYBOOK = """Research playbook:
1. Search "<display name>" + company; "<display name>" + email domain.
2. If surname is an initial, find full surname via first name + company + senior title.
3. Check linkedin, zoominfo, rocketreach, theorg, contactout, crunchbase, apollo.
4. Check related companies (subsidiary, parent, sister brands, previous employers).
5. Look for press, interviews, event-speaker pages.
Stop once title and company are corroborated by at least one source."""


def company_task(domain: str, hint: str) -> str:
    return (f"Build a company profile for the organisation that owns the email domain '{domain}'. "
            "Include parent company, subsidiaries and sister brands if any (in parent_or_group)."
            + (f" Context from the user: {hint}." if hint else "")
            + f"\nReturn JSON with this shape:\n{COMPANY_SCHEMA}")


def person_task(c: Contact, company: dict, all_companies: dict) -> str:
    co_name = company.get("company_name") or c.stem
    related = [f"{d}: {v.get('company_name', '')} (group: {v.get('parent_or_group', '')})"
               for d, v in all_companies.items()]
    return (
        "Identify this business contact: full name, CURRENT title and company, location, previous roles.\n"
        f"Email: {c.email}\n"
        f"Display name from email header: {c.display_name or '(none)'}\n"
        f"Email domain: {c.domain} (company: {co_name}; group/parent: {company.get('parent_or_group') or 'unknown'})\n"
        f"Other companies on the same email thread: {'; '.join(related) or 'none'}\n"
        + (f"User context: {c.company_hint}\n" if c.company_hint else "")
        + (f"Structured data from Apollo:\n{json.dumps(c.apollo)}\n" if c.apollo else "")
        + f"\nReturn JSON with this shape:\n{PERSON_SCHEMA}"
    )


# ======================================================================================
# SECTION 5 — Evidence handling + normalisation
# ======================================================================================

def _norm_url(u: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", (u or "").strip().lower()).rstrip("/")


def _rank(results: list[dict], tokens: list[str], keep: int = 14) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in results:
        seen.setdefault(_norm_url(r["url"]), r)
    toks = [t.lower() for t in tokens if len(t) > 1]

    def score(r: dict) -> int:
        blob = f"{r.get('title', '')} {r.get('content', '')} {r['url']}".lower()
        return sum(blob.count(t) for t in toks)

    return sorted(seen.values(), key=score, reverse=True)[:keep]


def _s(v) -> str:
    return str(v).strip() if isinstance(v, (str, int, float)) else ""


def _urls(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [u for u in v if isinstance(u, str) and u.startswith(("http://", "https://"))]


def normalize_company(d: dict, domain: str, allowed: set[str]) -> dict:
    out = {k: _s(d.get(k)) for k in ("company_name", "domain", "headquarters", "industry",
                                     "what_they_do", "size", "parent_or_group")}
    out["company_name"] = out["company_name"] or domain
    out["domain"] = out["domain"] or domain
    for k in ("other_offices", "products"):
        v = d.get(k)
        out[k] = [_s(x) for x in v if _s(x)] if isinstance(v, list) else []
    out["key_leadership"] = [{"name": _s(x.get("name")), "title": _s(x.get("title"))}
                             for x in d.get("key_leadership") or []
                             if isinstance(x, dict) and _s(x.get("name"))]
    out["sources"] = list(dict.fromkeys(u for u in _urls(d.get("sources")) if _norm_url(u) in allowed))
    return out


def normalize_person(d: dict, c: Contact, allowed: set[str]) -> dict:
    out = {k: _s(d.get(k)) for k in ("full_name", "current_title", "current_company", "location",
                                     "department", "ai_relevance", "notes",
                                     "summary", "linkedin_url", "match_reasoning")}
    if out["ai_relevance"] not in ("Very High", "High", "Medium", "Low"):
        out["ai_relevance"] = "Medium" if out["current_title"] else ""
    out["previous_roles"] = [{k: _s(r.get(k)) for k in ("title", "company", "period")}
                             for r in d.get("previous_roles") or []
                             if isinstance(r, dict) and (_s(r.get("title")) or _s(r.get("company")))]
    ap = c.apollo or {}
    if ap:
        out["current_title"] = out["current_title"] or _s(ap.get("title"))
        out["current_company"] = out["current_company"] or _s(ap.get("organization"))
        out["linkedin_url"] = out["linkedin_url"] or _s(ap.get("linkedin_url"))
        out["location"] = out["location"] or ", ".join(
            x for x in (_s(ap.get("city")), _s(ap.get("country"))) if x)
        if _s(ap.get("linkedin_url")):
            allowed = allowed | {_norm_url(ap["linkedin_url"])}
    if out["linkedin_url"] and _norm_url(out["linkedin_url"]) not in allowed:
        out["linkedin_url"] = ""
    conf = _s(d.get("confidence")).lower().replace(" ", "_")
    if conf not in CONFIDENCE:
        conf = "low" if out["current_title"] else "not_found"
    if conf == "high" and not (out["current_title"] and out["current_company"]):
        conf = "medium"
    out["confidence"] = conf
    out["full_name"] = out["full_name"] or c.name_guess
    out["sources"] = list(dict.fromkeys(u for u in _urls(d.get("sources")) if _norm_url(u) in allowed))
    return out


def _incomplete(p: dict) -> bool:
    return (not p.get("current_title") or not p.get("current_company")
            or p.get("confidence") in ("low", "not_found"))


# ======================================================================================
# SECTION 6 — Research jobs + orchestrator
# ======================================================================================

def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw), None
    except Exception as e:
        log.warning("%s failed: %s: %s", getattr(fn, "__name__", "call"), type(e).__name__, str(e)[:200])
        return None, getattr(fn, "__name__", "call")


def _collect(futs: list[Future], timeout: float) -> tuple[list[dict], list[str]]:
    wait(futs, timeout=timeout)
    results, errors = [], []
    for f in futs:
        if not f.done():
            errors.append("timeout")
            continue
        val, err = f.result()
        if err:
            errors.append(err)
        elif val:
            results.extend(val)
    return results, errors


def _ws_result(f: Future | None, deadline: float, notes: list[str]) -> tuple[dict, list[str]]:
    if f is None:
        return {}, []
    wait([f], timeout=max(0.0, deadline - time.time()))
    if not f.done():
        notes.append("Azure web search was still running; results were built from the faster sources.")
        return {}, []
    val, err = f.result()
    if err or not val:
        notes.append("Azure web search was unavailable for part of this lookup.")
        return {}, []
    return val


def _public(c: Contact) -> dict:
    return {"email": c.email, "domain": c.domain, "name_guess": c.name_guess,
            "display_name": c.display_name, "used_apollo": bool(c.apollo)}


def _fallback_profile(c: Contact, msg: str) -> dict:
    return {"full_name": c.name_guess, "current_title": "", "current_company": "", "location": "",
            "department": "", "ai_relevance": "", "notes": "", "summary": "", "linkedin_url": "",
            "match_reasoning": "", "confidence": "not_found", "previous_roles": [], "sources": [],
            "_error": msg}


def _company_job(domain: str, hint: str, s: Settings, io_pool: ThreadPoolExecutor, force: bool) -> dict:
    ckey = ("company", domain, hint.lower())
    if not force:
        hit = PROFILE_CACHE.get(ckey)
        if hit:
            return hit
    t0, notes = time.time(), []
    stem = domain.split(".")[0]
    tav = [io_pool.submit(_safe, tavily_search, s, q) for q in
           (f"{domain} company", f"{stem} company headquarters offices", f"{stem} CEO leadership team")]
    ws = (io_pool.submit(_safe, azure_web_search, s,
                         SYSTEM_RULES + "\n\n" + PLAYBOOK + "\n\n" + company_task(domain, hint))
          if s.has_azure else None)
    ev, _ = _collect(tav, s.search_timeout * 2)
    ws_data, ws_urls = _ws_result(ws, t0 + s.websearch_budget, notes)
    ev = _rank(ev, [stem, domain])
    allowed = {_norm_url(r["url"]) for r in ev} | {_norm_url(u) for u in ws_urls}
    user = company_task(domain, hint) + "\n\nEVIDENCE (web results):\n" + json.dumps(ev, ensure_ascii=False)
    if ws_data:
        user += "\n\nWEB-SEARCH FINDINGS:\n" + json.dumps({"findings": ws_data, "urls": ws_urls}, ensure_ascii=False)
    try:
        raw = llm_json(s, SYSTEM_RULES, user)
    except Exception as e:
        log.warning("company synth failed: %s", type(e).__name__)
        return {"company_name": domain, "domain": domain,
                "_error": "Company profile could not be built.", "_notes": notes}
    out = normalize_company(raw, domain, allowed)
    out["_notes"] = notes
    PROFILE_CACHE.set(ckey, out)
    return out


def _gap_queries(c: Contact, company: dict) -> list[tuple[str, tuple[str, ...] | None]]:
    name = c.name_guess or c.local
    first = name.split()[0] if name.split() else c.local
    co = company.get("company_name") or c.stem
    group = company.get("parent_or_group") or ""
    qs: list[tuple[str, tuple[str, ...] | None]] = [
        (f'"{name}" "{co}"', None),
        (f'"{name}" {co}', tuple(PEOPLE_SITES)),
        (f'{first} {co} CEO OR CTO OR CIO OR "Vice President" OR Director OR Head OR Manager', None),
        (f'"{name}" {co} joined OR appointed OR role', None),
    ]
    if group and group.lower() not in co.lower():
        qs.append((f'"{name}" {group}', None))
    return qs


def _person_job(c: Contact, company_futs: dict[str, Future], s: Settings,
                io_pool: ThreadPoolExecutor, force: bool, hint: str) -> dict:
    pkey = ("person", c.email, hint.lower())
    if not force:
        hit = PROFILE_CACHE.get(pkey)
        if hit:
            return hit
    t0, notes = time.time(), []
    name, stem = c.name_guess or c.local, c.stem
    hint_q = ([(f'"{name}" {c.company_hint}', None)]
              if c.company_hint and c.company_hint.lower() not in stem else [])
    queries: list[tuple[str, tuple[str, ...] | None]] = [
        (f'"{name}" {stem}', None),
        (f'"{name}" {stem} linkedin', None),
        (f"{name} {c.domain}", None),
        (f'"{name}" {stem}', tuple(PEOPLE_SITES)),
    ] + hint_q
    tav = [io_pool.submit(_safe, tavily_search, s, q, dom) for q, dom in queries]
    ws = (io_pool.submit(_safe, azure_web_search, s,
                         SYSTEM_RULES + "\n\n" + PLAYBOOK + "\n\n" + person_task(c, {}, {}))
          if s.has_azure else None)

    ev, _ = _collect(tav, s.search_timeout * 2)
    companies: dict[str, dict] = {}
    for d, f in company_futs.items():
        try:
            companies[d] = f.result(timeout=s.company_wait)
        except Exception:
            companies[d] = {}
    company = companies.get(c.domain, {})

    ws_data, ws_urls = _ws_result(ws, t0 + s.websearch_budget, notes)
    tokens = [*name.split(), c.local, stem]

    def synth(evidence: list[dict]) -> tuple[dict, set[str]]:
        ranked = _rank(evidence, tokens)
        allow = {_norm_url(r["url"]) for r in ranked} | {_norm_url(u) for u in ws_urls}
        user = (person_task(c, company, companies) + "\n\nEVIDENCE (web results):\n"
                + json.dumps(ranked, ensure_ascii=False))
        if ws_data:
            user += ("\n\nWEB-SEARCH FINDINGS:\n"
                     + json.dumps({"findings": ws_data, "urls": ws_urls}, ensure_ascii=False))
        return llm_json(s, SYSTEM_RULES, user), allow

    try:
        raw, allowed = synth(ev)
        profile = normalize_person(raw, c, allowed)

        if _incomplete(profile):
            extra, _ = _collect([io_pool.submit(_safe, tavily_search, s, q, dom)
                                 for q, dom in _gap_queries(c, company)], s.search_timeout * 2)
            if extra:
                ev = ev + extra
                raw2, allowed2 = synth(ev)
                better = normalize_person(raw2, c, allowed2)
                fields = ("current_title", "current_company", "location", "linkedin_url")
                if (not _incomplete(better)) or \
                        sum(bool(better[k]) for k in fields) >= sum(bool(profile[k]) for k in fields):
                    profile = better
    except Exception as e:
        log.warning("person synth failed for %s: %s", c.email, type(e).__name__)
        return {"contact": _public(c),
                "profile": _fallback_profile(c, "Lookup failed for this contact. Please retry."),
                "_notes": notes}

    out = {"contact": _public(c), "profile": profile, "_notes": notes}
    if profile["confidence"] != "not_found":
        PROFILE_CACHE.set(pkey, out)
    return out


def run_lookup(contacts: list[Contact], hint: str, s: Settings, force: bool = False) -> Iterator[dict]:
    t0 = time.time()
    domains = sorted({c.domain for c in contacts if c.domain not in FREE_MAIL_DOMAINS})
    io_pool = ThreadPoolExecutor(max_workers=s.io_workers, thread_name_prefix="io")
    jobs = ThreadPoolExecutor(max_workers=len(domains) + len(contacts) + 1, thread_name_prefix="job")
    try:
        company_f = {d: jobs.submit(_company_job, d, hint, s, io_pool, force) for d in domains}
        person_f = {c.email: jobs.submit(_person_job, c, company_f, s, io_pool, force, hint)
                    for c in contacts}
        label = {f: ("company", d) for d, f in company_f.items()}
        label.update({f: ("person", e) for e, f in person_f.items()})
        names = {c.email: c.name_guess for c in contacts}
        total, done = len(label), 0
        yield {"type": "progress", "done": 0, "total": total, "message": "Searching the web in parallel…"}
        for f in as_completed(label):
            done += 1
            kind, ident = label[f]
            try:
                res = f.result()
            except Exception:
                res = {}
            if kind == "company":
                msg = f"Company profile ready: {(res or {}).get('company_name') or ident}"
            else:
                conf = ((res or {}).get("profile") or {}).get("confidence", "")
                msg = (f"Profile ready: {names.get(ident, ident)}"
                       + (f" ({conf.replace('_', ' ')})" if conf else ""))
            yield {"type": "progress", "done": done, "total": total, "message": msg}

        companies, notes = {}, []
        for d, f in company_f.items():
            r = (f.result() if not f.exception()
                 else {"company_name": d, "domain": d, "_error": "Company profile could not be built."})
            notes += r.pop("_notes", [])
            companies[d] = r
        people = []
        for c in contacts:
            f = person_f[c.email]
            r = (f.result() if not f.exception()
                 else {"contact": _public(c),
                       "profile": _fallback_profile(c, "Lookup failed for this contact. Please retry.")})
            notes += r.pop("_notes", [])
            people.append(r)
        yield {"type": "result", "data": {
            "companies": companies, "people": people,
            "notes": sorted(set(notes)), "elapsed": round(time.time() - t0, 1),
            "engine": s.engine_label,
        }}
    finally:
        io_pool.shutdown(wait=False, cancel_futures=True)
        jobs.shutdown(wait=False, cancel_futures=True)


# ======================================================================================
# SECTION 7 — Tabular model + exports (PDF / Word / Excel)
# ======================================================================================

NAVY, BLUE, SLATE, LINE, ZEBRA = "1E1747", "5B3FD6", "5B5880", "E6E3F5", "FBFAFF"

PEOPLE_COLS = ["S.No", "Name", "Title", "Company", "Department", "AI Relevance",
               "Confidence", "Notes", "Source"]
UI_PEOPLE_COLS = [c for c in PEOPLE_COLS if c != "S.No"]
COMPANY_COLS = ["Company", "Domain", "Industry", "Headquarters", "Size", "Parent / Group", "What they do"]


def _best_source(item: dict) -> str:
    p = item.get("profile", {})
    for u in p.get("sources") or []:
        if u and u.startswith("http"):
            return u
    li = p.get("linkedin_url", "")
    return li if li.startswith("http") else ""


def _conf_label(c: str) -> str:
    return (c or "").replace("_", " ").title()


def people_rows(people: list[dict]) -> list[dict]:
    rows = []
    for i, x in enumerate(people, 1):
        p, c = x["profile"], x["contact"]
        rows.append({
            "S.No": str(i),
            "Name": p.get("full_name") or c["name_guess"],
            "Title": p.get("current_title", ""),
            "Company": p.get("current_company", ""),
            "Department": p.get("department", ""),
            "AI Relevance": p.get("ai_relevance", ""),
            "Confidence": _conf_label(p.get("confidence", "")),
            "Notes": p.get("notes", ""),
            "Source": _best_source(x),
        })
    return rows


def company_rows(companies: dict[str, dict]) -> list[dict]:
    return [{
        "Company": co.get("company_name", d), "Domain": d,
        "Industry": co.get("industry", ""), "Headquarters": co.get("headquarters", ""),
        "Size": co.get("size", ""), "Parent / Group": co.get("parent_or_group", ""),
        "What they do": co.get("what_they_do", ""),
    } for d, co in companies.items()]


def _domain_of(url: str) -> str:
    return (urlparse(url).netloc or url).removeprefix("www.")


def to_xlsx(people: list[dict], companies: dict[str, dict]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    thin = Side(style="thin", color=LINE)
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor=NAVY)
    zebra = PatternFill("solid", fgColor=ZEBRA)
    conf_fill = {"High": "DCFCE7", "Medium": "FEF3C7", "Low": "FEE2E2", "Not Found": "E2E8F0"}

    def sheet(ws, cols, rows, widths):
        ws.append(cols)
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
            c.fill, c.border = head_fill, border
            c.alignment = Alignment(vertical="center", horizontal="left", wrap_text=True)
        ws.row_dimensions[1].height = 24
        for r_i, r in enumerate(rows, 2):
            ws.append([r[c] for c in cols])
            for c_i, col in enumerate(cols, 1):
                cell = ws.cell(r_i, c_i)
                cell.border = border
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.font = Font(name="Calibri", size=10)
                if r_i % 2 == 1:
                    cell.fill = zebra
                if col == "Source" and str(cell.value).startswith("http"):
                    cell.hyperlink = cell.value
                    cell.value = _domain_of(cell.value)
                    cell.font = Font(name="Calibri", size=10, color=BLUE, underline="single")
                if col == "Confidence" and cell.value in conf_fill:
                    cell.fill = PatternFill("solid", fgColor=conf_fill[cell.value])
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    wb = Workbook()
    ws = wb.active
    ws.title = "People"
    sheet(ws, PEOPLE_COLS, people_rows(people), [7, 24, 32, 26, 18, 14, 13, 44, 24])
    if companies:
        sheet(wb.create_sheet("Companies"), COMPANY_COLS, company_rows(companies),
              [26, 22, 22, 24, 16, 26, 60])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def to_docx(people: list[dict], companies: dict[str, dict], elapsed: float = 0) -> bytes:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Mm(297), Mm(210)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Mm(12))
    st_ = doc.styles["Normal"]
    st_.font.name, st_.font.size = "Calibri", Pt(9)

    def shade(cell, fill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        tcPr.append(shd)

    def link(par, url, text):
        r_id = par.part.relate_to(url, RT.HYPERLINK, is_external=True)
        h = OxmlElement("w:hyperlink")
        h.set(qn("r:id"), r_id)
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        col = OxmlElement("w:color")
        col.set(qn("w:val"), BLUE)
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rpr.append(col)
        rpr.append(u)
        r.append(rpr)
        t = OxmlElement("w:t")
        t.text = text
        r.append(t)
        h.append(r)
        par._p.append(h)

    def table(cols, rows, widths_mm):
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = "Table Grid"
        t.autofit = False
        for i, h in enumerate(cols):
            c = t.rows[0].cells[i]
            c.width = Mm(widths_mm[i])
            c.text = ""
            run = c.paragraphs[0].add_run(h)
            run.bold, run.font.size = True, Pt(9)
            run.font.color.rgb = RGBColor(255, 255, 255)
            shade(c, NAVY)
        for r_i, r in enumerate(rows):
            cells = t.add_row().cells
            for i, col in enumerate(cols):
                cells[i].width = Mm(widths_mm[i])
                v = str(r[col])
                par = cells[i].paragraphs[0]
                if col == "Source" and v.startswith("http"):
                    link(par, v, _domain_of(v))
                else:
                    par.add_run(v).font.size = Pt(8.5)
                if r_i % 2 == 1:
                    shade(cells[i], ZEBRA)
        return t

    h = doc.add_heading("Contact Intelligence Report", level=0)
    for run in h.runs:
        run.font.size, run.font.color.rgb = Pt(22), RGBColor.from_string(BLUE)
    n_high = sum(1 for x in people if x["profile"].get("confidence") == "high")
    doc.add_paragraph(f"{len(people)} contacts · {n_high} high confidence · "
                      f"{len(companies)} companies · {elapsed}s research time")
    doc.add_heading("People", level=1)
    table(PEOPLE_COLS, people_rows(people), [9, 36, 48, 38, 28, 22, 20, 52, 24])
    if companies:
        doc.add_heading("Companies", level=1)
        table(COMPANY_COLS, company_rows(companies), [40, 36, 34, 38, 24, 36, 69])
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def to_pdf(people: list[dict], companies: dict[str, dict], elapsed: float = 0) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, blue = colors.HexColor("#" + NAVY), colors.HexColor("#" + BLUE)
    ss = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontName="Helvetica", fontSize=7.5, leading=9.5,
                          textColor=colors.HexColor("#1E293B"))
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=blue, fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=navy, fontSize=13, spaceBefore=12, spaceAfter=6)
    sub = ParagraphStyle("sub", parent=cell, fontSize=9, textColor=colors.HexColor("#" + SLATE))
    conf_bg = {"High": "#DCFCE7", "Medium": "#FEF3C7", "Low": "#FEE2E2", "Not Found": "#E2E8F0"}

    def p(txt):
        return Paragraph(xesc(str(txt)).replace("\n", "<br/>"), cell)

    def tbl(cols, rows, widths):
        data = [[Paragraph(xesc(c), head) for c in cols]]
        for r in rows:
            line = []
            for c in cols:
                v = str(r[c])
                if c == "Source" and v.startswith("http"):
                    line.append(Paragraph(f'<link href="{xesc(v, {chr(34): "&quot;"})}" color="#{BLUE}">'
                                          f'{xesc(_domain_of(v))}</link>', cell))
                else:
                    line.append(p(v))
            data.append(line)
        t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), navy), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#" + LINE)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#" + ZEBRA)]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if "Confidence" in cols:
            ci = cols.index("Confidence")
            for i, r in enumerate(rows, 1):
                if r["Confidence"] in conf_bg:
                    style.append(("BACKGROUND", (ci, i), (ci, i), colors.HexColor(conf_bg[r["Confidence"]])))
        t.setStyle(TableStyle(style))
        return t

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#" + SLATE))
        canvas.drawString(12 * mm, 7 * mm, "UBTI Contact Intelligence")
        canvas.drawRightString(285 * mm, 7 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    n_high = sum(1 for x in people if x["profile"].get("confidence") == "high")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=14 * mm, title="Contact Intelligence Report")
    story = [
        Paragraph("Contact Intelligence Report", h1),
        Paragraph(f"{len(people)} contacts · {n_high} high confidence · {len(companies)} companies · "
                  f"{elapsed}s research time", sub),
        Paragraph("People", h2),
        tbl(PEOPLE_COLS, people_rows(people), [9, 32, 42, 34, 24, 20, 18, 58, 36]),
    ]
    if companies:
        story += [Spacer(1, 6), Paragraph("Companies", h2),
                  tbl(COMPANY_COLS, company_rows(companies), [38, 34, 32, 36, 22, 34, 77])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def build_files(result: dict) -> dict[str, bytes | None]:
    people, companies, el = result["people"], result["companies"], result.get("elapsed", 0)
    out: dict[str, bytes | None] = {}
    for key, fn in (("pdf", lambda: to_pdf(people, companies, el)),
                    ("docx", lambda: to_docx(people, companies, el)),
                    ("xlsx", lambda: to_xlsx(people, companies))):
        try:
            out[key] = fn()
        except Exception as e:
            log.warning("export %s failed: %s", key, e)
            out[key] = None
    return out


# ======================================================================================
# SECTION 8 — Theme + generative UI components
# ======================================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--deep:#150F33;--deep2:#1E1747;--ink:#1F1B3A;--muted:#6B6890;--line:#E6E3F5;--bg:#F7F5FF;
  --accent:#6A41E0;--accent2:#8B5CF6;--accent-soft:#F0EBFF;}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#F4F1FF 0%,#FFFFFF 380px);}
.block-container{max-width:1100px;padding-top:2rem;padding-bottom:8rem;}

/* ---------- sidebar: deep UBTI indigo ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--deep) 0%,#0E0A24 100%);border-right:0;}
[data-testid="stSidebarCollapseButton"] svg,[data-testid="stSidebarCollapsedControl"] svg{color:#B9B1E6;}
.logo{display:block;width:190px;margin:.4rem 0 1.6rem .3rem;}
[data-testid="stSidebar"] .stButton>button{background:transparent;border:1px solid transparent;color:#D8D2F5;
  justify-content:flex-start;text-align:left;border-radius:10px;padding:.55rem .8rem;font-weight:500;font-size:.9rem;box-shadow:none;}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.07);color:#fff;border-color:transparent;}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:rgba(139,92,246,.20);color:#fff;border-color:rgba(139,92,246,.40);}
.st-key-new_chat button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#fff!important;
  font-weight:600!important;justify-content:center!important;border:0!important;box-shadow:0 6px 18px rgba(106,65,224,.35)!important;}
.st-key-new_chat button:hover{filter:brightness(1.08);}
.side-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#8E86C2;font-weight:600;margin:1.6rem 0 .4rem .4rem;}
.side-empty{color:#8E86C2;font-size:.84rem;padding:.2rem .5rem;}
.side-foot{position:fixed;bottom:16px;left:22px;font-size:.78rem;}
.side-foot a{color:#8E86C2;text-decoration:none;} .side-foot a:hover{color:#fff;}

/* ---------- chat input ---------- */
[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:linear-gradient(to top,#fff 65%,rgba(255,255,255,0));}
[data-testid="stChatInput"],[data-testid="stChatInput"]>div{border:1px solid var(--line)!important;border-radius:20px!important;
  background:#fff!important;box-shadow:0 8px 30px rgba(32,20,90,.10)!important;}
[data-testid="stChatInput"]:focus-within,[data-testid="stChatInput"]>div:focus-within{border-color:var(--accent)!important;
  box-shadow:0 8px 32px rgba(106,65,224,.20)!important;}
[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--ink);}
[data-testid="stChatInputSubmitButton"]{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#fff!important;border-radius:12px;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2));}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--accent);}
[data-baseweb="tab-highlight"]{background-color:var(--accent)!important;}

/* ---------- conversation ---------- */
[data-testid="stChatMessage"]{background:transparent;padding:.7rem 0;gap:.9rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#fff;border:1px solid var(--line);
  border-radius:16px;padding:.8rem 1.1rem;box-shadow:0 2px 8px rgba(32,20,90,.04);}
[data-testid="stChatMessageAvatarUser"]{background:var(--deep2);color:#fff;}
[data-testid="stChatMessageAvatarAssistant"]{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fff;}
[data-testid="stStatus"]{border:1px solid var(--line);border-radius:14px;background:#fff;}

/* ---------- welcome ---------- */
.hero{text-align:center;margin:7vh 0 1.6rem;}
.pill{display:inline-block;font-size:.7rem;font-weight:600;letter-spacing:.12em;color:var(--accent);background:var(--accent-soft);
  border:1px solid #DDD3FF;padding:.3rem .8rem;border-radius:999px;margin-bottom:1.1rem;}
.hero h1{font-size:2.5rem;font-weight:700;color:var(--deep)!important;margin:0;letter-spacing:-.8px;line-height:1.15;}
.hero h1 span{-webkit-text-fill-color:transparent;background:linear-gradient(90deg,var(--accent),#A855F7);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:var(--muted);margin:.9rem auto 0;font-size:1.02rem;max-width:560px;line-height:1.55;}
.feats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1.8rem 0 1.2rem;}
.feat{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.15rem;box-shadow:0 4px 16px rgba(32,20,90,.05);}
.feat .ic{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;
  align-items:center;justify-content:center;margin-bottom:.8rem;}
.feat .ic svg{width:20px;height:20px;stroke:#fff;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
.feat b{display:block;color:var(--deep);font-size:.95rem;margin-bottom:.25rem;}
.feat span{color:var(--muted);font-size:.84rem;line-height:1.45;}
.try{text-align:center;color:var(--muted);font-size:.8rem;margin:1.4rem 0 .5rem;}
.main .stButton>button{border:1px solid var(--line);border-radius:12px;background:#fff;color:var(--ink);padding:.7rem 1rem;
  font-weight:500;font-size:.85rem;box-shadow:0 1px 2px rgba(32,20,90,.04);}
.main .stButton>button:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-soft);}

/* ---------- KPI cards ---------- */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:.4rem 0 1rem;}
.kpi{border:1px solid var(--line);border-radius:14px;padding:.75rem .95rem;background:#fff;}
.kpi .v{font-size:1.5rem;font-weight:700;color:var(--deep);line-height:1.1;}
.kpi .l{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-top:.25rem;}
.kpi.accent{background:linear-gradient(135deg,var(--accent),var(--accent2));border:0;}
.kpi.accent .v,.kpi.accent .l{color:#fff;}

/* ---------- generated tables ---------- */
.gt-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:#fff;box-shadow:0 4px 16px rgba(32,20,90,.05);}
table.gt{width:100%;border-collapse:collapse;font-size:.82rem;}
table.gt th{background:var(--deep2);color:#fff;text-align:left;padding:.65rem .75rem;font-weight:600;white-space:nowrap;position:sticky;top:0;}
table.gt td{padding:.6rem .75rem;border-top:1px solid var(--line);vertical-align:top;color:var(--ink);}
table.gt tr:nth-child(even) td{background:#FBFAFF;}
table.gt tr:hover td{background:var(--accent-soft);}
table.gt a{color:var(--accent);text-decoration:none;font-weight:500;}
table.gt a:hover{text-decoration:underline;}
.bd{display:inline-block;padding:.12rem .55rem;border-radius:999px;font-size:.72rem;font-weight:600;white-space:nowrap;}
.bd-ok{background:#DCFCE7;color:#166534;} .bd-warn{background:#FEF3C7;color:#92400E;}
.bd-bad{background:#FEE2E2;color:#991B1B;} .bd-na{background:#E8E6F4;color:#4B4872;}
.bd-vh{background:var(--deep2);color:#fff;} .bd-h{background:#E4DBFF;color:#4C2BB8;}
.bd-m{background:#E8E6F4;color:#3D3A63;} .bd-l{background:#F3F2FA;color:#6B6890;}
.muted{color:var(--muted);} .dl-title{font-size:.8rem;color:var(--muted);margin:1rem 0 .3rem;}
.stDownloadButton>button{color:var(--accent);border:1px solid var(--accent);border-radius:10px;font-weight:600;background:#fff;}
.stDownloadButton>button:hover{background:var(--accent);color:#fff;border-color:var(--accent);}
.detail p{margin:.15rem 0;font-size:.88rem;}
</style>
"""

LOGO_B64 = "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"
COMPANY_URL = "https://ubtiinc.com/"

ICON_DOC = '<div class="ic"><svg viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg></div>'
ICON_IMG = '<div class="ic"><svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="M21 16l-5-5-8 8"/></svg></div>'
ICON_DL = '<div class="ic"><svg viewBox="0 0 24 24"><path d="M12 4v11m0 0l-4-4m4 4l4-4M5 20h14"/></svg></div>'

CONF_CLS = {"High": "ok", "Medium": "warn", "Low": "bad", "Not Found": "na"}
AI_CLS = {"Very High": "vh", "High": "h", "Medium": "m", "Low": "l"}


def _cell_html(col: str, v: str) -> str:
    if not v:
        return '<span class="muted">—</span>'
    if col == "Confidence":
        return f'<span class="bd bd-{CONF_CLS.get(v, "na")}">{esc(v)}</span>'
    if col == "AI Relevance":
        return f'<span class="bd bd-{AI_CLS.get(v, "m")}">{esc(v)}</span>'
    if col == "Source":
        if v.startswith("http"):
            return f'<a href="{esc(v)}" target="_blank" rel="noopener">{esc(_domain_of(v))} ↗</a>'
        return '<span class="muted">—</span>'
    if col == "Name":
        return f"<b>{esc(v)}</b>"
    return esc(v).replace("\n", "<br>")


def table_html(cols: list[str], rows: list[dict]) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_cell_html(c, str(r[c]))}</td>" for c in cols) + "</tr>"
                   for r in rows)
    return f'<div class="gt-wrap"><table class="gt"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def kpi_html(people: list[dict], companies: dict, elapsed: float) -> str:
    n_high = sum(1 for x in people if x["profile"].get("confidence") == "high")
    items = [(len(people), "Contacts", True), (n_high, "High confidence", False),
             (len(companies), "Companies", False), (f"{elapsed}s", "Research time", False)]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi{" accent" if a else ""}"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l, a in items) + "</div>"


def detail_html(x: dict) -> str:
    p = x["profile"]
    parts = []
    for label, key in (("Summary", "summary"), ("Location", "location"), ("Why this match", "match_reasoning")):
        if p.get(key):
            parts.append(f"<p><b>{label}:</b> {esc(p[key])}</p>")
    if p.get("previous_roles"):
        roles = "; ".join(" · ".join(filter(None, (r["title"], r["company"], r["period"])))
                          for r in p["previous_roles"])
        parts.append(f"<p><b>Previous roles:</b> {esc(roles)}</p>")
    if p.get("linkedin_url", "").startswith("http"):
        parts.append(f'<p><b>LinkedIn:</b> <a href="{esc(p["linkedin_url"])}" target="_blank">'
                     f'{esc(_domain_of(p["linkedin_url"]))} ↗</a></p>')
    if p.get("sources"):
        links = " · ".join(f'<a href="{esc(u)}" target="_blank">{esc(_domain_of(u))}</a>' for u in p["sources"][:6])
        parts.append(f"<p><b>Sources:</b> {links}</p>")
    if p.get("_error"):
        parts.append(f"<p>⚠ {esc(p['_error'])}</p>")
    return '<div class="detail">' + ("".join(parts) or '<p class="muted">No further details found.</p>') + "</div>"


def render_result(msg: dict, mid: str) -> None:
    res = msg["result"]
    people, companies = res["people"], res["companies"]
    st.markdown(kpi_html(people, companies, res.get("elapsed", 0)), unsafe_allow_html=True)

    tabs = st.tabs(["People", "Companies", "Details"] if companies else ["People", "Details"])
    with tabs[0]:
        st.markdown(table_html(UI_PEOPLE_COLS, people_rows(people)), unsafe_allow_html=True)
    if companies:
        with tabs[1]:
            st.markdown(table_html(COMPANY_COLS, company_rows(companies)), unsafe_allow_html=True)
    with tabs[-1]:
        for x in people:
            p = x["profile"]
            title = " — ".join(filter(None, (p.get("full_name") or x["contact"]["name_guess"],
                                             p.get("current_title"))))
            with st.expander(title or x["contact"]["email"]):
                st.markdown(detail_html(x), unsafe_allow_html=True)

    for n in res.get("notes") or []:
        st.caption(f"ℹ {n}")

    files = msg.get("files") or {}
    st.markdown('<div class="dl-title">Download report</div>', unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([1, 1, 1, 2])
    specs = [(c1, "pdf", "PDF", "contact_intelligence.pdf", "application/pdf"),
             (c2, "docx", "Word", "contact_intelligence.docx",
              "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
             (c3, "xlsx", "Excel", "contact_intelligence.xlsx",
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")]
    for col, key, label, fname, mime in specs:
        with col:
            if files.get(key):
                st.download_button(f"⬇ {label}", files[key], file_name=fname, mime=mime,
                                   key=f"dl_{mid}_{key}", use_container_width=True)
            else:
                st.caption(f"{label} unavailable")


# ======================================================================================
# SECTION 9 — Chat state + app
# ======================================================================================

SUGGESTIONS = [
    "Ganesh R <ganesh.r@ak.com.sa>\nVenkatesh M <venkatesh.m@ak.com.sa>",
    "Jane Doe <jane.doe@acme.com>; John Smith <john.smith@acme.com>",
]


def _new_chat() -> None:
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.order.insert(0, cid)
    st.session_state.current = cid


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("chats", {})
    ss.setdefault("order", [])
    ss.setdefault("pending", None)
    if not ss.get("current") or ss.current not in ss.chats:
        _new_chat()


def _set_pending(text: str) -> None:
    st.session_state.pending = text


def _select(cid: str) -> None:
    st.session_state.current = cid


def render_message(m: dict, mid: str) -> None:
    avatar = ":material/person:" if m["role"] == "user" else ":material/hub:"
    with st.chat_message(m["role"], avatar=avatar):
        if m.get("text"):
            st.markdown(m["text"])
        for img in m.get("images") or []:
            st.image(img, width=200)
        if m.get("result"):
            render_result(m, mid)


def process(text: str, files: list, hint: str, include_sender: bool, force: bool) -> dict:
    contacts: list[Contact] = []
    seen: set[str] = set()

    def add(items: list[Contact]) -> None:
        for c in items:
            if c.email not in seen:
                seen.add(c.email)
                contacts.append(c)

    with st.status("Reading your input…", expanded=True) as status:
        try:
            if text.strip():
                add(smart_parse(text, S, hint, include_sender))
            for f in files:
                status.update(label=f"Reading image {getattr(f, 'name', '')}…")
                add(contacts_from_image(f.getvalue(), S, getattr(f, "type", None) or "image/png"))
        except Exception as e:
            log.warning("parse failed: %s", e)
            status.update(label="Could not read the input", state="error")
            return {"role": "assistant", "text": "I couldn't read that input. Please check your Azure settings and try again."}

        if not contacts:
            status.update(label="No contacts found", state="error")
            return {"role": "assistant", "text": (
                "I couldn't find any email addresses. Send a pasted email header, a list like "
                "`Name <email@company.com>`, or a screenshot that shows email addresses.")}

        truncated = len(contacts) > S.max_contacts
        contacts = contacts[:S.max_contacts]
        n_dom = len({c.domain for c in contacts})
        status.update(label=f"Researching {len(contacts)} contact(s) across {n_dom} domain(s)…")
        bar = st.progress(0.0)
        result = None
        for ev in run_lookup(contacts, hint, S, force=force):
            if ev["type"] == "progress":
                bar.progress(ev["done"] / max(ev["total"], 1))
                st.write(f"• {ev['message']}")
            else:
                result = ev["data"]
        status.update(label=f"Research complete in {result['elapsed']}s", state="complete", expanded=False)

    confs = [x["profile"]["confidence"] for x in result["people"]]
    summary = (f"I researched **{len(contacts)}** contact(s) across **{len(result['companies'])}** "
               f"company profile(s): **{confs.count('high')}** high, **{confs.count('medium')}** medium, "
               f"**{confs.count('low') + confs.count('not_found')}** need review.")
    if truncated:
        summary += f" Only the first {S.max_contacts} contacts were processed."
    return {"role": "assistant", "text": summary, "result": result, "files": build_files(result)}


# ======================================================================================
# SECTION 9b — Follow-up conversation (streamed answers grounded in earlier results)
# ======================================================================================

FOLLOWUP_SYSTEM = (
    "You are a B2B research assistant inside a contact-intelligence app. The user is chatting about "
    "contacts that were researched earlier in this conversation. Answer using the CONTEXT (earlier research "
    "results) and the FRESH WEB EVIDENCE. Never invent facts, titles, employers or URLs; if something is not "
    "supported by the evidence, say it was not found. Cite sources as markdown links [site](url). "
    "Be concise and well structured: short paragraphs or bullet points, bold key facts. "
    "When asked for more details about a person, cover: current role, company, location, career history, "
    "expertise/skills, public activity, and a confidence note. Reply in the user's language."
)


def _chat_context(history: list[dict]) -> tuple[str, list[dict]]:
    """Compact JSON of earlier results, plus the person the question most likely refers to."""
    people, companies = [], {}
    for m in history:
        res = m.get("result")
        if not res:
            continue
        companies.update(res.get("companies") or {})
        for x in res["people"]:
            p = x["profile"]
            people.append({
                "name": p.get("full_name") or x["contact"]["name_guess"], "email": x["contact"]["email"],
                **{k: p.get(k) for k in ("current_title", "current_company", "department", "location", "summary",
                                         "previous_roles", "notes", "confidence", "linkedin_url", "sources")}})
    ctx = json.dumps({"people": people, "companies": list(companies.values())}, ensure_ascii=False)[:14000]
    return ctx, people


def _find_target(question: str, people: list[dict]) -> dict | None:
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    best, best_n = None, 0
    for p in people:
        toks = {t for t in re.findall(r"[a-z0-9]+", f"{p['name']} {p['email'].split('@')[0]}".lower()) if len(t) > 2}
        n = len(toks & q)
        if n > best_n:
            best, best_n = p, n
    return best


def build_followup_messages(question: str, history: list[dict], s: Settings) -> list[dict]:
    ctx, people = _chat_context(history)
    target = _find_target(question, people)
    if target:
        who = f'{target["name"]} {target.get("current_company") or ""}'.strip()
        queries = [(f'"{target["name"]}" {target.get("current_company") or ""}', None),
                   (f"{who} experience career background", None),
                   (f'"{target["name"]}" {target.get("current_company") or ""}', tuple(PEOPLE_SITES))]
    else:
        queries = [(question, None), (f"{question} {people[0].get('current_company') or ''}".strip(), None)] \
            if people else [(question, None)]
    ev: list[dict] = []
    if s.has_tavily:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(_safe, tavily_search, s, q, d, 6) for q, d in queries]
            ev, _ = _collect(futs, s.search_timeout * 2)
        ev = _rank(ev, [t for t in re.findall(r"\w+", question + " " + (target["name"] if target else ""))], keep=12)
    turns = [{"role": m["role"], "content": m["text"]} for m in history[-6:] if m.get("text")]
    user = (f"CONTEXT (earlier research results):\n{ctx or 'none'}\n\n"
            f"FRESH WEB EVIDENCE:\n{json.dumps(ev, ensure_ascii=False)}\n\nQUESTION: {question}")
    return [{"role": "system", "content": FOLLOWUP_SYSTEM}, *turns, {"role": "user", "content": user}]


def stream_chat(messages: list[dict], s: Settings) -> Iterator[str]:
    client = azure_client(s, s.synth_timeout * 2, retries=2)
    for chunk in client.chat.completions.create(model=s.deployment, messages=messages, stream=True):
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()
    ss = st.session_state

    # ---------------- Sidebar ----------------
    hint, include_sender, force = "", False, False
    with st.sidebar:
        st.markdown(f'<a href="{COMPANY_URL}" target="_blank"><img class="logo" alt="Unlimited Innovations" '
                    f'src="data:image/png;base64,{LOGO_B64}"></a>', unsafe_allow_html=True)
        st.markdown('<div class="new-chat">', unsafe_allow_html=True)
        if st.button("＋  New chat", use_container_width=True, key="new_chat"):
            if ss.chats[ss.current]["messages"]:
                _new_chat()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)
        history = [c for c in ss.order if ss.chats[c]["messages"]]
        if not history:
            st.markdown('<div class="side-empty">Your lookups will appear here.</div>', unsafe_allow_html=True)
        for cid in history:
            st.button(ss.chats[cid]["title"][:30], key=f"chat_{cid}", use_container_width=True,
                      type="primary" if cid == ss.current else "secondary",
                      on_click=_select, args=(cid,))
        st.markdown(f'<div class="side-foot"><a href="{COMPANY_URL}" target="_blank">ubtiinc.com</a></div>',
                    unsafe_allow_html=True)

    chat = ss.chats[ss.current]
    msgs = chat["messages"]

    # ---------------- Conversation ----------------
    if not msgs:
        st.markdown(
            '<div class="hero"><div class="pill">CONTACT INTELLIGENCE</div>'
            '<h1>Know exactly who you are <span>talking to</span></h1>'
            '<p>Paste an email thread or contact list, or drop a screenshot. Get verified roles, '
            'companies and a report you can share.</p></div>'
            '<div class="feats">'
            '<div class="feat">' + ICON_DOC + '<b>Paste anything</b><span>Email headers, or a list of Name &lt;email&gt; contacts.</span></div>'
            '<div class="feat">' + ICON_IMG + '<b>Drop a screenshot</b><span>Invites, signatures and business cards. Ctrl+V works.</span></div>'
            '<div class="feat">' + ICON_DL + '<b>Export in one click</b><span>Download results as PDF, Word or Excel.</span></div>'
            '</div><div class="try">Or try a sample</div>', unsafe_allow_html=True)
        _, c1, c2, _ = st.columns([1, 2, 2, 1])
        for col, s_ in zip((c1, c2), SUGGESTIONS):
            with col:
                st.button(", ".join(EMAIL_RE.findall(s_))[:34] + "…", key=f"sug_{hash(s_)}",
                          use_container_width=True, on_click=_set_pending, args=(s_,))
    for i, m in enumerate(msgs):
        render_message(m, f"{ss.current}_{i}")

    # ---------------- Input ----------------
    fallback_files: list = []
    try:
        val = st.chat_input("Paste contacts or a screenshot, or ask a follow-up question…",
                            accept_file="multiple", file_type=["png", "jpg", "jpeg", "webp"])
    except TypeError:  # Streamlit < 1.43
        with st.sidebar:
            fallback_files = st.file_uploader("Screenshots", type=["png", "jpg", "jpeg", "webp"],
                                              accept_multiple_files=True) or []
        val = st.chat_input("Paste contacts or ask a follow-up…")

    text, files = "", list(fallback_files)
    if isinstance(val, str):
        text = val
    elif val:
        text = getattr(val, "text", "") or ""
        files = list(getattr(val, "files", None) or [])
    if ss.pending:
        text, ss.pending = ss.pending, None
    if not (text.strip() or (files and val)):
        return
    if not S.ready:
        st.error("Configure API keys in .env first.")
        return

    is_lookup = bool(files) or bool(EMAIL_RE.search(text))
    history = list(msgs)
    images = [f.getvalue() for f in files]
    body = text.strip()
    msgs.append({"role": "user", "images": images,
                 "text": f"```\n{body}\n```" if (body and is_lookup) else body})
    if chat["title"] == "New chat":
        m = EMAIL_RE.search(text)
        chat["title"] = (m.group(0) if m else body.splitlines()[0][:30]) if body else "Screenshot lookup"
    render_message(msgs[-1], "tmp")
    with st.chat_message("assistant", avatar=":material/hub:"):
        if is_lookup:
            reply = process(text, files, hint, include_sender, force)
        else:
            try:
                with st.spinner("Researching…"):
                    messages = build_followup_messages(body, history, S)
                answer = st.write_stream(stream_chat(messages, S))
            except Exception as e:
                log.warning("follow-up failed: %s", e)
                answer = "Sorry, I couldn't complete that request. Please try again."
                st.markdown(answer)
            reply = {"role": "assistant", "text": answer}
    msgs.append(reply)
    st.rerun()

if __name__ == "__main__":
    main()
