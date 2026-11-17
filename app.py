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
    "theme.base": "light", "theme.primaryColor": "#1F4FD8", "theme.backgroundColor": "#FFFFFF",
    "theme.secondaryBackgroundColor": "#F1F5F9", "theme.textColor": "#0F172A",
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

NAVY, BLUE, SLATE, LINE, ZEBRA = "0F172A", "1F4FD8", "475569", "E2E8F0", "F8FAFC"

PEOPLE_COLS = ["S.No", "Name", "Title", "Company", "Department", "AI Relevance",
               "Confidence", "Notes", "Source"]
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
:root{--navy:#0F172A;--ink:#1E293B;--muted:#64748B;--line:#E2E8F0;--bg:#F8FAFC;--accent:#1F4FD8;}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',system-ui,sans-serif;}
#MainMenu,footer{visibility:hidden;}
.block-container{max-width:1080px;padding-top:2rem;padding-bottom:7rem;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--navy);border-right:1px solid #1E293B;}
[data-testid="stSidebar"] *{color:#CBD5E1;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#F8FAFC;}
[data-testid="stSidebar"] .stButton>button{background:transparent;border:1px solid transparent;
  color:#E2E8F0;justify-content:flex-start;text-align:left;border-radius:8px;padding:.45rem .7rem;font-weight:400;}
[data-testid="stSidebar"] .stButton>button:hover{background:#1E293B;border-color:#1E293B;}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:#1E293B;border-color:#334155;}
[data-testid="stSidebar"] .stButton>button.newchat,[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea{background:#1E293B!important;color:#F1F5F9!important;border-color:#334155!important;}
[data-testid="stSidebar"] hr{border-color:#1E293B;}
.brand{font-weight:700;font-size:1.05rem;color:#F8FAFC;letter-spacing:.2px;margin-bottom:.2rem;}
.brand b{color:#60A5FA;}
.side-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#64748B;margin:.9rem 0 .3rem;}

/* accent fallback if theme options are ignored */
.stProgress>div>div>div>div{background:var(--accent);}
.main .stButton>button[kind="primary"],.stDownloadButton>button{border-color:var(--accent);}
.stDownloadButton>button{color:var(--accent);font-weight:600;}
.stDownloadButton>button:hover{background:var(--accent);color:#fff;border-color:var(--accent);}

/* chat */
[data-testid="stChatMessage"]{background:transparent;padding:.6rem 0;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#F1F5F9;border-radius:14px;padding:.7rem 1rem;}
[data-testid="stChatInput"]{border:1px solid var(--line);border-radius:16px;box-shadow:0 2px 12px rgba(15,23,42,.08);}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent);}

/* welcome */
.hero{text-align:center;margin:12vh 0 1.5rem;}
.hero h1{font-size:2rem;font-weight:700;color:var(--navy);margin:0;}
.hero p{color:var(--muted);margin-top:.5rem;}

/* KPI cards */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:.4rem 0 1rem;}
.kpi{border:1px solid var(--line);border-radius:12px;padding:.7rem .9rem;background:#fff;}
.kpi .v{font-size:1.5rem;font-weight:700;color:var(--navy);line-height:1.1;}
.kpi .l{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem;}
.kpi.accent{border-left:4px solid var(--accent);}

/* generated tables */
.gt-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:#fff;}
table.gt{width:100%;border-collapse:collapse;font-size:.84rem;}
table.gt th{background:var(--navy);color:#fff;text-align:left;padding:.6rem .7rem;font-weight:600;white-space:nowrap;position:sticky;top:0;}
table.gt td{padding:.55rem .7rem;border-top:1px solid var(--line);vertical-align:top;color:var(--ink);}
table.gt tr:nth-child(even) td{background:var(--bg);}
table.gt tr:hover td{background:#EFF6FF;}
table.gt a{color:var(--accent);text-decoration:none;font-weight:500;}
table.gt a:hover{text-decoration:underline;}
.bd{display:inline-block;padding:.12rem .55rem;border-radius:999px;font-size:.72rem;font-weight:600;white-space:nowrap;}
.bd-ok{background:#DCFCE7;color:#166534;} .bd-warn{background:#FEF3C7;color:#92400E;}
.bd-bad{background:#FEE2E2;color:#991B1B;} .bd-na{background:#E2E8F0;color:#475569;}
.bd-vh{background:#1E3A8A;color:#fff;} .bd-h{background:#DBEAFE;color:#1E40AF;}
.bd-m{background:#E2E8F0;color:#334155;} .bd-l{background:#F1F5F9;color:#64748B;}
.muted{color:var(--muted);} .dl-title{font-size:.8rem;color:var(--muted);margin:1rem 0 .3rem;}
.detail p{margin:.15rem 0;font-size:.88rem;}
</style>
"""

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
        st.markdown(table_html(PEOPLE_COLS, people_rows(people)), unsafe_allow_html=True)
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


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()
    ss = st.session_state

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown('<div class="brand">◆ UBTI <b>Contact Intelligence</b></div>', unsafe_allow_html=True)
        if st.button("＋  New chat", use_container_width=True, key="new_chat"):
            _new_chat()
            st.rerun()

        st.markdown('<div class="side-label">Chats</div>', unsafe_allow_html=True)
        for cid in ss.order:
            st.button(ss.chats[cid]["title"][:30], key=f"chat_{cid}", use_container_width=True,
                      type="primary" if cid == ss.current else "secondary",
                      on_click=_select, args=(cid,))

        st.markdown('<div class="side-label">Settings</div>', unsafe_allow_html=True)
        hint = st.text_input("Company / context hint", placeholder="e.g. Al Abdul Karim, Saudi Arabia")
        include_sender = st.checkbox("Include email sender (From)", value=False)
        force = st.checkbox("Bypass cache (re-research)", value=False)

        st.markdown('<div class="side-label">Status</div>', unsafe_allow_html=True)
        if S.ready:
            st.caption("● APIs configured")
        else:
            miss = ([] if S.has_azure else ["AZURE_OPENAI_ENDPOINT / API_KEY / DEPLOYMENT"]) + \
                   ([] if S.has_tavily else ["TAVILY_API_KEY"])
            st.error("Missing: " + ", ".join(miss))
        st.caption(f"Engine: {S.engine_label}")
        if st.button("Clear caches", use_container_width=True):
            SEARCH_CACHE.clear()
            PROFILE_CACHE.clear()
            st.toast("Caches cleared")

    chat = ss.chats[ss.current]
    msgs = chat["messages"]

    # ---------------- Conversation ----------------
    if not msgs:
        st.markdown('<div class="hero"><h1>Who would you like to research?</h1>'
                    '<p>Paste an email header, a list of contacts, or attach a screenshot.</p></div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(SUGGESTIONS))
        for col, s_ in zip(cols, SUGGESTIONS):
            with col:
                st.button(s_.replace("\n", " · ")[:70], key=f"sug_{hash(s_)}", use_container_width=True,
                          on_click=_set_pending, args=(s_,))
    for i, m in enumerate(msgs):
        render_message(m, f"{ss.current}_{i}")

    # ---------------- Input ----------------
    fallback_files: list = []
    try:
        val = st.chat_input("Paste contacts or a screenshot (Ctrl+V), or attach images…",
                            accept_file="multiple", file_type=["png", "jpg", "jpeg", "webp"])
    except TypeError:  # Streamlit < 1.43
        with st.sidebar:
            fallback_files = st.file_uploader("Screenshots", type=["png", "jpg", "jpeg", "webp"],
                                              accept_multiple_files=True) or []
        val = st.chat_input("Paste contacts here…")

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

    images = [f.getvalue() for f in files]
    msgs.append({"role": "user", "text": f"```\n{text.strip()}\n```" if text.strip() else "", "images": images})
    if chat["title"] == "New chat":
        m = EMAIL_RE.search(text)
        chat["title"] = (m.group(0) if m else text.strip().splitlines()[0][:30]) if text.strip() else "Screenshot lookup"
    render_message(msgs[-1], "tmp")
    with st.chat_message("assistant", avatar=":material/hub:"):
        reply = process(text, files, hint, include_sender, force)
    msgs.append(reply)
    st.rerun()


if __name__ == "__main__":
    main()
