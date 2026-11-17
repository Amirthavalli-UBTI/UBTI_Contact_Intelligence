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
from urllib.parse import quote_plus, urlparse
from xml.sax.saxutils import escape as xesc

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Theme set in code (replaces .streamlit/config.toml). Must run before set_page_config.
for _k, _v in {
    "theme.base": "light", "theme.primaryColor": "#2B6EA8", "theme.backgroundColor": "#FFFFFF",
    "theme.secondaryBackgroundColor": "#F1F5F9", "theme.textColor": "#0F1B33",
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
    out = [{"title": x.get("title", ""), "url": x.get("url", ""), "content": (x.get("content") or "")[:1200 if "linkedin.com/in/" in x.get("url", "") else 700]}
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
    "CURRENT ROLE rules (critical):\n"
    "- current_title and current_company must be the role the person holds TODAY. LinkedIn is the highest-priority "
    "source: a LinkedIn headline or an Experience entry marked 'Present' beats every other source.\n"
    "- If several roles are listed, pick the one marked Present or with the latest start date. Roles with an end "
    "date, or described in the past tense, go ONLY in previous_roles - never in current_title.\n"
    "- Never guess a title from seniority or tenure. If the evidence does not show a current title, return \"\" "
    "and confidence 'low'.\n"
    "- linkedin_url must be a linkedin.com/in/ URL that appears in the evidence for THIS person.\n"
    "- When sources disagree, trust LinkedIn, then the most recent dated source.\n\n"
    "DEPARTMENT & AI RELEVANCE rules:\n"
    "- department: infer from title (CTO -> IT, CFO -> Finance, CHRO -> HR, COO -> Operations, "
    "CMO -> Marketing, CEO/MD -> Executive Leadership).\n"
    "- ai_relevance: 'Very High' for CIO/CTO/CDO/Head of AI; 'High' for VP/Director Eng/Analytics; "
    "'Medium' for CFO/COO/CHRO; 'Low' otherwise.\n"
    "- notes: exactly 2 lines separated by newline, each under 80 chars."
)

PLAYBOOK = """Research playbook:
1. LinkedIn FIRST: search "<display name>" + company on linkedin.com and read the headline and current position.
2. Then search "<display name>" + company; "<display name>" + email domain.
3. If surname is an initial, find full surname via first name + company + senior title.
4. Check zoominfo, rocketreach, theorg, contactout, crunchbase, apollo to corroborate.
5. Check related companies (subsidiary, parent, sister brands, previous employers).
6. Look for press, interviews, event-speaker pages.
Stop once the CURRENT title and company are corroborated, preferably by LinkedIn."""


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

def clean_linkedin(u: str) -> str:
    """Canonical https://www.linkedin.com/in/<slug>, or '' if not a profile URL (fixes country sub-domains)."""
    m = re.match(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/([^/?#\s]+)", (u or "").strip(), re.I)
    return f"https://www.linkedin.com/in/{m.group(1)}" if m else ""


def linkedin_search_url(name: str, company: str = "") -> str:
    return "https://www.linkedin.com/search/results/people/?keywords=" + quote_plus(f"{name} {company}".strip())


def _norm_url(u: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", (u or "").strip().lower()).rstrip("/")


def _rank(results: list[dict], tokens: list[str], keep: int = 14) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in results:
        seen.setdefault(_norm_url(r["url"]), r)
    toks = [t.lower() for t in tokens if len(t) > 1]

    def score(r: dict) -> int:
        blob = f"{r.get('title', '')} {r.get('content', '')} {r['url']}".lower()
        return sum(blob.count(t) for t in toks) + (40 if "linkedin.com/in/" in r["url"] else 0)

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
    allowed_li = {_norm_url(clean_linkedin("https://" + a)) for a in allowed if "linkedin.com/in/" in a}
    li = clean_linkedin(out["linkedin_url"])
    out["linkedin_url"] = li if li and _norm_url(li) in allowed_li else ""
    conf = _s(d.get("confidence")).lower().replace(" ", "_")
    if conf not in CONFIDENCE:
        conf = "low" if out["current_title"] else "not_found"
    if conf == "high" and not (out["current_title"] and out["current_company"]):
        conf = "medium"
    srcs: list[str] = []
    for u in _urls(d.get("sources")):
        lu = clean_linkedin(u)
        if lu and _norm_url(lu) in allowed_li:
            srcs.append(lu)
        elif _norm_url(u) in allowed:
            srcs.append(u)
    if out["linkedin_url"]:
        srcs.insert(0, out["linkedin_url"])
    out["sources"] = sorted(dict.fromkeys(srcs), key=lambda u: "linkedin.com/in/" not in u)
    if conf == "high" and not any("linkedin.com/in/" in u for u in out["sources"]):
        conf = "medium"  # current title not corroborated by LinkedIn
    out["confidence"] = conf
    out["full_name"] = out["full_name"] or c.name_guess
    return out


def _ensure_linkedin(profile: dict, ev: list[dict], name: str) -> dict:
    """If the model returned no LinkedIn URL, take the best matching LinkedIn result from the evidence."""
    if profile.get("linkedin_url"):
        return profile
    toks = [t for t in re.findall(r"[a-z]+", name.lower()) if len(t) > 2][:2]
    for r in ev if toks else []:
        li = clean_linkedin(r["url"])
        blob = f'{r.get("title", "")} {r.get("content", "")} {r["url"]}'.lower()
        if li and all(t in blob for t in toks):
            profile["linkedin_url"] = li
            profile["sources"] = list(dict.fromkeys([li, *profile.get("sources", [])]))
            break
    return profile


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
        (f'"{name}" {co}', ("linkedin.com",)),
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
    li_only = ("linkedin.com",)
    queries: list[tuple[str, tuple[str, ...] | None]] = [
        (f'"{name}" {stem}', li_only),
        (f"{name} {stem} current position experience", li_only),
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

    profile = _ensure_linkedin(profile, ev, name)
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

NAVY, BLUE, SLATE, LINE, ZEBRA = "1F5382", "2B6EA8", "5B6B82", "DCE6F1", "F5F9FD"

PEOPLE_COLS = ["S.No", "Name", "Title", "Company", "Department", "AI Relevance",
               "Confidence", "Notes", "Source"]
UI_PEOPLE_COLS = [c for c in PEOPLE_COLS if c != "S.No"]
COMPANY_COLS = ["Company", "Domain", "Industry", "Headquarters", "Size", "Parent / Group", "What they do"]


def _best_source(item: dict) -> str:
    """LinkedIn profile first, then any other source."""
    p = item.get("profile", {})
    li = p.get("linkedin_url", "")
    if li.startswith("http"):
        return li
    srcs = [u for u in p.get("sources") or [] if u and u.startswith("http")]
    return next((u for u in srcs if "linkedin.com/in/" in u), srcs[0] if srcs else "")


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
    host = (urlparse(url).netloc or url).removeprefix("www.")
    return "LinkedIn" if host.endswith("linkedin.com") else host


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
:root{--deep:#1F5382;--deep2:#245E8F;--ink:#0F1B33;--muted:#5B6B82;--line:#DCE6F1;--bg:#F4F8FC;
  --accent:#2B6EA8;--accent2:#4A8CC4;--accent-soft:#EAF2FA;}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stAppViewContainer"]{background:#FFFFFF;}
.block-container{max-width:1100px;padding-top:2rem;padding-bottom:8rem;}

/* ---------- sidebar: deep UBTI indigo ---------- */
[data-testid="stSidebar"]{background:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(165deg,#326898 0%,#286598 50%,#1C486E 100%);background-size:34px 34px,34px 34px,auto;border-right:0;}
[data-testid="stSidebarCollapseButton"] svg,[data-testid="stSidebarCollapsedControl"] svg{color:#DCEAF6;}
[data-testid="stSidebarHeader"]{height:0;min-height:0;padding:0;overflow:visible;}
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"]{position:absolute;top:.5rem;right:.5rem;z-index:5;}
[data-testid="stSidebarUserContent"]{padding-top:1.1rem!important;}
.logo-card{display:block;background:#fff;border-radius:14px;padding:12px 16px;margin:0 0 1.3rem 0;box-shadow:0 8px 22px rgba(10,35,66,.28);}
.logo{display:block;width:100%;}
[data-testid="stSidebar"] .stButton>button{background:transparent;border:1px solid transparent;color:#EAF3FB;
  justify-content:flex-start;text-align:left;border-radius:10px;padding:.55rem .8rem;font-weight:500;font-size:.9rem;box-shadow:none;}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.10);color:#fff;border-color:transparent;}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:rgba(255,255,255,.16);color:#fff;border-color:rgba(255,255,255,.30);}
.st-key-new_chat button{background:rgba(255,255,255,.12)!important;color:#fff!important;font-weight:600!important;justify-content:center!important;border:1px solid rgba(255,255,255,.30)!important;box-shadow:none!important;}
.st-key-new_chat button:hover{background:rgba(255,255,255,.22)!important;border-color:rgba(255,255,255,.5)!important;}
.side-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#BFD6EC;font-weight:600;margin:1.6rem 0 .4rem .4rem;}
.side-empty{color:#BFD6EC;font-size:.84rem;padding:.2rem .5rem;}
.side-foot{position:fixed;bottom:16px;left:22px;font-size:.78rem;}
.side-foot a{color:#BFD6EC;text-decoration:none;} .side-foot a:hover{color:#fff;}

/* ---------- chat input ---------- */
[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:linear-gradient(to top,#fff 65%,rgba(255,255,255,0));}
[data-testid="stChatInput"],[data-testid="stChatInput"]>div{border:1px solid var(--line)!important;border-radius:20px!important;
  background:#fff!important;box-shadow:0 8px 30px rgba(31,83,130,.10)!important;}
[data-testid="stChatInput"]:focus-within,[data-testid="stChatInput"]>div:focus-within{border-color:var(--accent)!important;
  box-shadow:0 8px 32px rgba(43,110,168,.20)!important;}
[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--ink);}
[data-testid="stChatInputSubmitButton"]{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#fff!important;border-radius:12px;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2));}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--accent);}
[data-baseweb="tab-highlight"]{display:none!important;}
[data-baseweb="tab-list"] button[aria-selected="true"]{border-bottom:2px solid var(--accent);}

/* ---------- conversation ---------- */
[data-testid="stChatMessage"]{background:transparent;padding:.7rem 0;gap:.9rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#fff;border:1px solid var(--line);
  border-radius:16px;padding:.8rem 1.1rem;box-shadow:0 2px 8px rgba(31,83,130,.04);}
[data-testid="stChatMessageAvatarUser"]{background:var(--deep2);color:#fff;}
[data-testid="stChatMessageAvatarAssistant"]{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fff;}
[data-testid="stStatus"]{border:1px solid var(--line);border-radius:14px;background:#fff;}

/* ---------- welcome ---------- */
.hero{text-align:center;margin:13vh 0 1.4rem;}
.pill{display:inline-block;font-size:.7rem;font-weight:600;letter-spacing:.12em;color:var(--accent);background:var(--accent-soft);
  border:1px solid #CFE1F1;padding:.3rem .8rem;border-radius:999px;margin-bottom:1.1rem;}
.hero h1{font-size:2.5rem;font-weight:700;color:var(--deep)!important;margin:0;letter-spacing:-.8px;line-height:1.15;}
.hero h1 span{-webkit-text-fill-color:transparent;background:linear-gradient(90deg,var(--accent),#7FB0DA);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:var(--muted);margin:.9rem auto 0;font-size:1.02rem;max-width:560px;line-height:1.55;}
.feats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1.8rem 0 1.2rem;}
.feat{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.15rem;box-shadow:0 4px 16px rgba(31,83,130,.05);}
.feat .ic{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;
  align-items:center;justify-content:center;margin-bottom:.8rem;}
.feat .ic svg{width:20px;height:20px;stroke:#fff;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
.feat b{display:block;color:var(--deep);font-size:.95rem;margin-bottom:.25rem;}
.feat span{color:var(--muted);font-size:.84rem;line-height:1.45;}
.try{text-align:center;color:var(--muted);font-size:.8rem;margin:1.4rem 0 .5rem;}
.main .stButton>button{border:1px solid var(--line);border-radius:12px;background:#fff;color:var(--ink);padding:.7rem 1rem;
  font-weight:500;font-size:.85rem;box-shadow:0 1px 2px rgba(31,83,130,.04);}
.main .stButton>button:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-soft);}

/* ---------- KPI cards ---------- */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:.4rem 0 1rem;}
.kpi{border:1px solid var(--line);border-radius:14px;padding:.75rem .95rem;background:#fff;}
.kpi .v{font-size:1.5rem;font-weight:700;color:var(--deep);line-height:1.1;}
.kpi .l{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-top:.25rem;}
.kpi.accent{background:linear-gradient(135deg,var(--accent),var(--accent2));border:0;}
.kpi.accent .v,.kpi.accent .l{color:#fff;}

/* ---------- generated tables ---------- */
.gt-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:#fff;box-shadow:0 4px 16px rgba(31,83,130,.05);}
table.gt{width:100%;border-collapse:collapse;font-size:.82rem;}
table.gt th{background:var(--deep2);color:#fff;text-align:left;padding:.65rem .75rem;font-weight:600;white-space:nowrap;position:sticky;top:0;}
table.gt td{padding:.6rem .75rem;border-top:1px solid var(--line);vertical-align:top;color:var(--ink);}
table.gt tr:nth-child(even) td{background:#F7FAFD;}
table.gt tr:hover td{background:var(--accent-soft);}
table.gt a{color:var(--accent);text-decoration:none;font-weight:500;}
table.gt a:hover{text-decoration:underline;}
.bd{display:inline-block;padding:.12rem .55rem;border-radius:999px;font-size:.72rem;font-weight:600;white-space:nowrap;}
.bd-ok{background:#DCFCE7;color:#166534;} .bd-warn{background:#FEF3C7;color:#92400E;}
.bd-bad{background:#FEE2E2;color:#991B1B;} .bd-na{background:#E6ECF7;color:#475569;}
.bd-vh{background:var(--deep2);color:#fff;} .bd-h{background:#DCEAF7;color:#245E8F;}
.bd-m{background:#E6ECF7;color:#334155;} .bd-l{background:#F1F5FB;color:#5B6B82;}
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
             (len(companies), "Companies", False), ("Cached" if elapsed < 0.2 else f"{elapsed}s", "Research time", False)]
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
    find = linkedin_search_url(p.get("full_name") or x["contact"]["name_guess"], p.get("current_company", ""))
    li = p.get("linkedin_url", "")
    parts.append("<p><b>LinkedIn:</b> " + (f'<a href="{esc(li)}" target="_blank">Open profile ↗</a> · ' if li else "")
                 + f'<a href="{esc(find)}" target="_blank">Search on LinkedIn ↗</a> '
                   '<span class="muted">(use this if a profile link shows an error)</span></p>')
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
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.6])
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
    with c4:
        st.button("↻ Re-research (fresh data)", key=f"rr_{mid}", use_container_width=True,
                  on_click=lambda m=mid: st.session_state.__setitem__("refresh", m))


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


def process(text: str, files: list, hint: str, include_sender: bool, force: bool,
            preset: list[Contact] | None = None) -> dict:
    contacts: list[Contact] = []
    seen: set[str] = set()

    def add(items: list[Contact]) -> None:
        for c in items:
            if c.email not in seen:
                seen.add(c.email)
                contacts.append(c)

    with st.status("Reading your input…", expanded=True) as status:
        try:
            if preset:
                add(preset)
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
    "Only cite URLs that appear in CONTEXT or FRESH WEB EVIDENCE; never create or guess a URL, and never say a page "
    "shows something unless the evidence snippet does. If the user reports a broken link, explain that search-index "
    "links can be outdated or private, quote what the evidence snippet says, and suggest searching the name and company "
    "on LinkedIn. Treat LinkedIn as the primary source for a person\'s CURRENT title. "
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
        queries = [(f'"{target["name"]}" {target.get("current_company") or ""}', ("linkedin.com",)),
                   (f'"{target["name"]}" {target.get("current_company") or ""}', None),
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


PLACEHOLDER = "Paste contacts or a screenshot, or ask a follow-up question…"


def read_input(key: str) -> tuple[bool, str, list]:
    """Chat box. Inside a column (new chat) it renders inline/centred; at top level it pins to the bottom."""
    fallback: list = []
    try:
        val = st.chat_input(PLACEHOLDER, accept_file="multiple",
                            file_type=["png", "jpg", "jpeg", "webp"], key=key)
    except TypeError:  # Streamlit < 1.43
        with st.sidebar:
            fallback = st.file_uploader("Screenshots", type=["png", "jpg", "jpeg", "webp"],
                                        accept_multiple_files=True, key=f"up_{key}") or []
        val = st.chat_input(PLACEHOLDER, key=key)
    text, files = "", list(fallback)
    if isinstance(val, str):
        text = val
    elif val:
        text = getattr(val, "text", "") or ""
        files = list(getattr(val, "files", None) or [])
    return bool(val), text, files


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()
    ss = st.session_state

    # ---------------- Sidebar ----------------
    hint, include_sender, force = "", False, False
    with st.sidebar:
        st.markdown(f'<a class="logo-card" href="{COMPANY_URL}" target="_blank"><img class="logo" alt="Unlimited Innovations" '
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

    # ---------------- Conversation / input ----------------
    if not msgs:  # new chat: ChatGPT-style centred prompt box
        st.markdown(
            '<div class="hero"><div class="pill">CONTACT INTELLIGENCE</div>'
            '<h1>Know exactly who you are <span>talking to</span></h1>'
            '<p>Paste an email thread or contact list, or drop a screenshot. Get verified roles, '
            'companies and a report you can share.</p></div>', unsafe_allow_html=True)
        _, mid_col, _ = st.columns([1, 6, 1])
        with mid_col:
            got, text, files = read_input("chat_in_center")
        st.markdown('<div class="try">Or try a sample</div>', unsafe_allow_html=True)
        _, c1, c2, _ = st.columns([1, 2, 2, 1])
        for col, s_ in zip((c1, c2), SUGGESTIONS):
            with col:
                st.button(", ".join(EMAIL_RE.findall(s_))[:34] + "…", key=f"sug_{hash(s_)}",
                          use_container_width=True, on_click=_set_pending, args=(s_,))
        st.markdown(
            '<div class="feats">'
            '<div class="feat">' + ICON_DOC + '<b>Paste anything</b><span>Email headers, or a list of Name &lt;email&gt; contacts.</span></div>'
            '<div class="feat">' + ICON_IMG + '<b>Drop a screenshot</b><span>Invites, signatures and business cards. Ctrl+V works.</span></div>'
            '<div class="feat">' + ICON_DL + '<b>Export in one click</b><span>Download results as PDF, Word or Excel.</span></div>'
            '</div>', unsafe_allow_html=True)
    else:
        for i, m in enumerate(msgs):
            render_message(m, f"{ss.current}_{i}")

        if ss.get("refresh"):  # "Re-research" button on a result
            idx = int(str(ss.refresh).rsplit("_", 1)[1])
            ss.refresh = None
            if S.ready and 0 <= idx < len(msgs) and msgs[idx].get("result"):
                cts = [Contact(email=x["contact"]["email"], display_name=x["contact"]["display_name"],
                               local=x["contact"]["email"].split("@")[0], domain=x["contact"]["domain"],
                               name_guess=x["contact"]["name_guess"]) for x in msgs[idx]["result"]["people"]]
                msgs.append({"role": "user", "text": "↻ Re-research with fresh data", "images": []})
                render_message(msgs[-1], "tmp")
                with st.chat_message("assistant", avatar=":material/hub:"):
                    reply = process("", [], "", False, True, preset=cts)
                msgs.append(reply)
                st.rerun()
        got, text, files = read_input("chat_in_bottom")

    if ss.pending:
        text, ss.pending = ss.pending, None
    if not (text.strip() or files):
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
