"""
UBTI Contact Intelligence — single-file Streamlit application.

Everything is in this one file:
  1. Settings + thread-safe TTL cache
  2. Contact parsing (regex, AI vision for images, AI smart-parse for messy text)
  3. Search backends (Tavily, Apollo, Azure OpenAI web_search)
  4. Research prompts + anti-hallucination normalisation
  5. Parallel research orchestrator
  6. Exports (CSV, Excel, Word, PDF)
  7. Streamlit UI (table view + detail cards + 3 input modes)

Credentials come from Streamlit secrets or environment variables:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT (default gpt-4.1)
  TAVILY_API_KEY           (required)
  APOLLO_API_KEY           (optional, improves accuracy)
  APP_PASSWORD             (optional, adds a shared access gate)

Run:  streamlit run app.py
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from html import escape as esc
from typing import Callable, Iterator
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass

log = logging.getLogger("ubti.lookup")

st.set_page_config(
    page_title="UBTI Contact Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================================================================================
# SECTION 1 — Constants & settings
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
    apollo_key: str = field(default="", repr=False)
    tavily_depth: str = "basic"
    search_timeout: int = 20          # seconds per Tavily call
    websearch_budget: int = 30        # seconds we are willing to wait for Azure web_search
    synth_timeout: int = 60           # seconds per synthesis call
    company_wait: int = 90            # max seconds a person waits for the company profile
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


def load_settings(get: Callable[[str, str], str]) -> Settings:
    def g(k: str, d: str = "") -> str:
        return str(get(k, d) or d).strip()

    def i(k: str, d: int) -> int:
        try:
            return max(1, int(g(k, str(d))))
        except ValueError:
            return d

    depth = g("TAVILY_SEARCH_DEPTH", "basic").lower()
    return Settings(
        azure_endpoint=g("AZURE_OPENAI_ENDPOINT"),
        azure_key=g("AZURE_OPENAI_API_KEY"),
        deployment=g("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
        tavily_key=g("TAVILY_API_KEY"),
        apollo_key=g("APOLLO_API_KEY"),
        tavily_depth=depth if depth in ("basic", "advanced") else "basic",
        websearch_budget=i("WEBSEARCH_BUDGET_SECONDS", 30),
        io_workers=i("IO_WORKERS", 12),
        max_contacts=i("MAX_CONTACTS", 25),
    )


def cfg(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file present
        pass
    return os.getenv(name, default)


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return load_settings(cfg)


S = get_settings()


class TTLCache:
    """Small thread-safe TTL cache — survives Streamlit reruns."""

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


@st.cache_resource(show_spinner=False)
def _caches() -> tuple[TTLCache, TTLCache]:
    return TTLCache(), TTLCache()


SEARCH_CACHE, PROFILE_CACHE = _caches()


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


# ======================================================================================
# SECTION 2 — Contact parsing
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
    """
    Accepts: 'Name <a@b.com>; Name2 <c@d.com>', markdown mailto links, one email per line,
    or a whole Outlook header block (From/Sent/To/Cc/Subject). For a header block only
    To/Cc/Bcc are read (plus From when include_sender is True); Sent/Subject are ignored.
    """
    # Rewrite markdown mailto links as "Name <addr>" — keeping the address, which a
    # bare "\1" replacement would throw away for links whose label is just the name.
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
            current = m.group(1).lower()
            body = m.group(2)
        else:
            body = line  # wrapped continuation of the previous header
            if not line.strip():
                current = None
        if current in wanted:
            out.extend(_chunks_to_contacts(body, company_hint, seen))
    return out


# ── Image-based extraction (Azure OpenAI vision) ──────────────────────────────

IMAGE_EXTRACT_PROMPT = """You are an assistant that extracts contact information from images of emails, meeting invites,
business cards, chat screenshots, or any document that contains people's names and email addresses.

Extract ALL people mentioned with their:
- full_name: the person's display name
- email: their email address (if visible)
- company_hint: the company name if you can determine it from the domain, signature, or context

Return a JSON object with this exact shape:
{
  "contacts": [
    {"full_name": "...", "email": "...", "company_hint": "..."},
    ...
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


def extract_contacts_from_image(image_bytes: bytes, s: Settings, mime: str = "image/png") -> dict:
    """Send an image to Azure OpenAI vision and extract contacts from it."""
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
    return extract_json(resp.choices[0].message.content)


def contacts_from_image(image_bytes: bytes, s: Settings, mime: str = "image/png",
                        include_sender: bool = True) -> tuple[list[Contact], str]:
    """Extract contacts from an image and return (contacts, company_hint)."""
    data = extract_contacts_from_image(image_bytes, s, mime)
    contacts: list[Contact] = []
    seen: set[str] = set()
    hint = str(data.get("company_context") or "").strip()
    raw_text = str(data.get("raw_text") or "")

    for entry in data.get("contacts") or []:
        email = str(entry.get("email") or "").strip().lower()
        name = str(entry.get("full_name") or "").strip()
        c_hint = str(entry.get("company_hint") or hint).strip()
        if email and EMAIL_RE.fullmatch(email) and email not in seen:
            seen.add(email)
            local, domain = email.split("@", 1)
            contacts.append(Contact(
                email=email, display_name=name, local=local, domain=domain,
                name_guess=name or guess_name_from_local(local), company_hint=c_hint,
            ))

    # If the vision model could not get emails, also try regex on the raw text it read
    if raw_text:
        contacts.extend(_chunks_to_contacts(raw_text, hint, seen))

    return contacts, hint


# ── Smart parsing for messy / unstructured text ───────────────────────────────

SMART_PARSE_PROMPT = """You are an assistant that extracts contact information from messy, unstructured text.

The text might be:
- A copy-paste from an email (with or without headers)
- Names and emails without proper separators
- A list of people with the company name mentioned separately (maybe at the top or bottom)
- A meeting invite pasted as text
- A mix of names, emails, company info in no particular format
- Text with line breaks, tabs, or inconsistent formatting

Extract ALL people you can identify with their:
- full_name: the person's name
- email: their email address (if present)
- company_hint: company name if you can determine it

Return a JSON object:
{
  "contacts": [
    {"full_name": "...", "email": "...", "company_hint": "..."},
    ...
  ],
  "company_context": "overall company or context you can determine"
}

Rules:
- Include EVERY person mentioned
- If names appear without emails, include them (email: "")
- Look for company names ANYWHERE in the text — they might be at the bottom, in a subject line, or in a signature
- Handle Indian name patterns: initials like "Ganesh R", "Venkatesh M", etc.
- An email domain like @ak.com.sa means the company might be "Al Abdul Karim" or similar — use context
- If you see "Subject:" lines, extract meeting context
- Return ONLY the JSON, no markdown"""


def smart_parse(raw: str, s: Settings, company_hint: str = "",
                include_sender: bool = False) -> tuple[list[Contact], str]:
    """Use the LLM to parse messy/unstructured text into contacts."""
    fast = parse_contacts(raw, company_hint, include_sender)
    if fast:
        return fast, company_hint

    client = azure_client(s, s.synth_timeout, retries=2)
    prompt = f"Extract contacts from this text:\n\n{raw}"
    if company_hint:
        prompt += f"\n\nAdditional context from the user: {company_hint}"

    resp = client.chat.completions.create(
        model=s.deployment,
        messages=[
            {"role": "system", "content": SMART_PARSE_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    data = extract_json(resp.choices[0].message.content)
    contacts: list[Contact] = []
    seen: set[str] = set()
    llm_hint = str(data.get("company_context") or company_hint).strip()

    for entry in data.get("contacts") or []:
        email = str(entry.get("email") or "").strip().lower()
        name = str(entry.get("full_name") or "").strip()
        c_hint = str(entry.get("company_hint") or llm_hint).strip()
        if email and EMAIL_RE.fullmatch(email) and email not in seen:
            seen.add(email)
            local, domain = email.split("@", 1)
            contacts.append(Contact(
                email=email, display_name=name, local=local, domain=domain,
                name_guess=name or guess_name_from_local(local), company_hint=c_hint,
            ))
        elif name and not email:
            log.info("Skipping name-only contact (no email): %s", name)

    return contacts, llm_hint


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


def apollo_match(s: Settings, email: str) -> dict:
    key = ("apollo", email)
    hit = SEARCH_CACHE.get(key)
    if hit is not None:
        return hit
    r = requests.post(
        "https://api.apollo.io/api/v1/people/match",
        headers={"x-api-key": s.apollo_key, "Content-Type": "application/json",
                 "Cache-Control": "no-cache"},
        params={"email": email, "reveal_personal_emails": "false", "reveal_phone_number": "false"},
        timeout=s.search_timeout,
    )
    if r.status_code != 200:
        return {}
    p = r.json().get("person") or {}
    org = p.get("organization") or {}
    out = {
        "name": p.get("name"), "title": p.get("title"), "headline": p.get("headline"),
        "organization": org.get("name"), "linkedin_url": p.get("linkedin_url"),
        "city": p.get("city"), "country": p.get("country"),
        "employment_history": [
            {"title": e.get("title"), "organization": e.get("organization_name"),
             "start": e.get("start_date"), "end": e.get("end_date"), "current": e.get("current")}
            for e in (p.get("employment_history") or [])[:6]
        ],
    } if p else {}
    SEARCH_CACHE.set(key, out)
    return out


# ── Azure OpenAI helpers ──────────────────────────────────────────────────────

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
    """Azure OpenAI Responses API with the built-in web_search tool. Returns (json, cited urls)."""
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
  "industry": str, "what_they_do": str (2-3 sentences), "products": [str], "size": str,
  "parent_or_group": str, "key_leadership": [{"name": str, "title": str}], "sources": [url]
}"""

PERSON_SCHEMA = """{
  "full_name": str, "current_title": str, "current_company": str, "location": str,
  "department": str (e.g. "IT", "Finance", "HR", "Operations", "Sales", "Engineering", "Legal", "Marketing", "Executive Leadership", "Supply Chain"),
  "ai_relevance": "Very High" | "High" | "Medium" | "Low" (how relevant this person is to AI/digital transformation initiatives based on their role),
  "notes": str (exactly 2 concise lines: line 1 = what this person likely does day-to-day based on their role; line 2 = why they matter for a B2B engagement or AI initiative),
  "previous_roles": [{"title": str, "company": str, "period": str}],
  "summary": str (1-2 sentences on who they are and their likely role in a deal),
  "linkedin_url": str or "",
  "confidence": "high" | "medium" | "low" | "not_found",
  "match_reasoning": str (why this is the same person as the email),
  "sources": [url]
}"""

SYSTEM_RULES = (
    "You are a B2B research analyst. Use ONLY the evidence provided (or that you find via search). "
    "Never invent names, titles or URLs; use \"\" or [] for anything you cannot support. If evidence is weak or "
    "conflicting, say so and lower confidence. Matching rules: the display name and the email local part "
    "(e.g. 'paulr' = Paul R...) must fit the person. The company does NOT have to equal the email domain exactly: "
    "people often use a parent/group company's email while working for a subsidiary or sister company, or have "
    "moved between group companies. A profile at a related company (same group, subsidiary, parent, or a company "
    "that appears alongside the email domain company in the evidence) is a valid match; explain the link in "
    "match_reasoning. Many people share common Indian names, so require at least one link to the company or group. "
    "Prefer facts confirmed by two or more sources. Fill every field the evidence supports. "
    "Sources must be URLs from the evidence. Return a single JSON object only.\n\n"
    "DEPARTMENT & AI RELEVANCE rules:\n"
    "- department: infer from the person's title (e.g. CTO/VP Engineering -> 'IT', CFO -> 'Finance', "
    "CHRO -> 'HR', COO -> 'Operations', CMO -> 'Marketing', CEO/MD -> 'Executive Leadership'). "
    "Use short standard labels.\n"
    "- ai_relevance: rate how relevant this person is for AI/digital transformation engagement. "
    "'Very High' = CIO, CTO, CDO, Head of AI/Data, IT Director; 'High' = VP/Director of Engineering, "
    "Analytics, Digital; 'Medium' = functional heads (CFO, COO, CHRO) who are stakeholders; "
    "'Low' = roles with limited AI touchpoints.\n"
    "- notes: exactly 2 lines separated by a newline. Line 1: what this person likely does day-to-day. "
    "Line 2: why they are relevant for a B2B/AI engagement. Keep each line under 80 chars."
)

PLAYBOOK = """Research playbook:
1. Search "<display name>" + company; "<display name>" + email domain.
2. If the surname is only an initial (e.g. "Ganesh R"), find the full surname: search the first name with the
   company and senior-title words (CEO, CTO, CIO, VP, Director, Head, Manager).
3. Check people-data / org-chart sites (linkedin, zoominfo, rocketreach, theorg, contactout, crunchbase, apollo).
4. Check related companies (subsidiary, parent, sister brands, previous employers). If the person moved, search the new company.
5. Look for press, interviews and event-speaker pages naming the person with a title.
Be quick: stop once title and company are corroborated by at least one source."""


def company_task(domain: str, hint: str) -> str:
    return (f"Build a company profile for the organisation that owns the email domain '{domain}'. "
            "Include its parent company, subsidiaries and sister brands if any (in parent_or_group)."
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
        + (f"Structured data from Apollo (strong evidence if it matches):\n{json.dumps(c.apollo)}\n" if c.apollo else "")
        + f"\nReturn JSON with this shape:\n{PERSON_SCHEMA}"
    )


# ======================================================================================
# SECTION 5 — Evidence handling + normalisation (anti-hallucination)
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
    out["other_offices"] = ([_s(x) for x in (d.get("other_offices") or []) if _s(x)]
                            if isinstance(d.get("other_offices"), list) else [])
    out["products"] = ([_s(x) for x in (d.get("products") or []) if _s(x)]
                       if isinstance(d.get("products"), list) else [])
    lead = []
    for x in d.get("key_leadership") or []:
        if isinstance(x, dict) and _s(x.get("name")):
            lead.append({"name": _s(x.get("name")), "title": _s(x.get("title"))})
    out["key_leadership"] = lead
    out["sources"] = list(dict.fromkeys(u for u in _urls(d.get("sources")) if _norm_url(u) in allowed))
    return out


def normalize_person(d: dict, c: Contact, allowed: set[str]) -> dict:
    out = {k: _s(d.get(k)) for k in ("full_name", "current_title", "current_company", "location",
                                     "department", "ai_relevance", "notes",
                                     "summary", "linkedin_url", "match_reasoning")}
    if out["ai_relevance"] not in ("Very High", "High", "Medium", "Low"):
        out["ai_relevance"] = "Medium" if out["current_title"] else ""
    roles = []
    for r in d.get("previous_roles") or []:
        if isinstance(r, dict) and (_s(r.get("title")) or _s(r.get("company"))):
            roles.append({k: _s(r.get(k)) for k in ("title", "company", "period")})
    out["previous_roles"] = roles
    ap = c.apollo or {}
    if ap:  # Apollo is structured data: use it to fill gaps the model left empty
        out["current_title"] = out["current_title"] or _s(ap.get("title"))
        out["current_company"] = out["current_company"] or _s(ap.get("organization"))
        out["linkedin_url"] = out["linkedin_url"] or _s(ap.get("linkedin_url"))
        out["location"] = out["location"] or ", ".join(
            x for x in (_s(ap.get("city")), _s(ap.get("country"))) if x)
        if _s(ap.get("linkedin_url")):
            allowed = allowed | {_norm_url(ap["linkedin_url"])}
    if out["linkedin_url"] and _norm_url(out["linkedin_url"]) not in allowed:
        out["linkedin_url"] = ""        # never show a LinkedIn URL we did not actually retrieve
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
    """Run a leaf call; never raise. Returns (value, error_label)."""
    try:
        return fn(*a, **kw), None
    except Exception as e:  # noqa: BLE001 - leaf failures must not kill the whole lookup
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
        user += "\n\nWEB-SEARCH FINDINGS (independent second search engine):\n" + json.dumps(
            {"findings": ws_data, "urls": ws_urls}, ensure_ascii=False)
    try:
        raw = llm_json(s, SYSTEM_RULES, user)
    except Exception as e:  # noqa: BLE001
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
    apollo_f = io_pool.submit(_safe, apollo_match, s, c.email) if s.apollo_key else None

    ev, _ = _collect(tav, s.search_timeout * 2)
    if apollo_f is not None:
        wait([apollo_f], timeout=s.search_timeout)
        if apollo_f.done() and apollo_f.result()[0]:
            c = Contact(**{**c.__dict__, "apollo": apollo_f.result()[0]})

    companies: dict[str, dict] = {}
    for d, f in company_futs.items():
        try:
            companies[d] = f.result(timeout=s.company_wait)
        except Exception:  # noqa: BLE001
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
            user += ("\n\nWEB-SEARCH FINDINGS (independent second search engine; merge with the evidence above):\n"
                     + json.dumps({"findings": ws_data, "urls": ws_urls}, ensure_ascii=False))
        return llm_json(s, SYSTEM_RULES, user), allow

    try:
        raw, allowed = synth(ev)
        profile = normalize_person(raw, c, allowed)

        if _incomplete(profile):   # one targeted gap-fill round, then re-synthesise
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
    except Exception as e:  # noqa: BLE001
        log.warning("person synth failed for %s: %s", c.email, type(e).__name__)
        return {"contact": _public(c),
                "profile": _fallback_profile(c, "Lookup failed for this contact. Please retry."),
                "_notes": notes}

    out = {"contact": _public(c), "profile": profile, "_notes": notes}
    if profile["confidence"] != "not_found":
        PROFILE_CACHE.set(pkey, out)
    return out


def run_lookup(contacts: list[Contact], hint: str, s: Settings, force: bool = False) -> Iterator[dict]:
    """
    Generator so the UI can render progress from the main Streamlit thread.

    Per company : 3 Tavily queries + 1 Azure web_search -> 1 synthesis call
    Per person  : 5 Tavily queries + 1 Azure web_search (+ Apollo) -> 1 synthesis call
    """
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
            except Exception:  # noqa: BLE001
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
# SECTION 7 — Exports:  S.No | Name | Title | Department | AI Relevance | Notes | Source
# ======================================================================================

def _best_source(item: dict) -> str:
    """First usable URL from the person's sources list, falling back to LinkedIn."""
    for u in item.get("profile", {}).get("sources") or []:
        if u and u.startswith("http"):
            return u
    li = item.get("profile", {}).get("linkedin_url", "")
    return li if li and li.startswith("http") else ""


def _build_df(people: list[dict]) -> pd.DataFrame:
    rows = []
    for idx, x in enumerate(people, 1):
        p, c = x["profile"], x["contact"]
        rows.append({
            "S.No": idx,
            "Name": p.get("full_name") or c["name_guess"],
            "Title": p.get("current_title", ""),
            "Department": p.get("department", ""),
            "AI Relevance / Interest": p.get("ai_relevance", ""),
            "Notes": p.get("notes", ""),
            "Source": _best_source(x),
        })
    return pd.DataFrame(rows)


def _build_company_df(companies: dict[str, dict]) -> pd.DataFrame:
    rows = [{
        "Company": co.get("company_name", d),
        "Domain": d,
        "Industry": co.get("industry", ""),
        "Headquarters": co.get("headquarters", ""),
        "Size": co.get("size", ""),
        "Parent / Group": co.get("parent_or_group", ""),
        "What they do": co.get("what_they_do", ""),
    } for d, co in companies.items()]
    return pd.DataFrame(rows)


def to_csv(people: list[dict], companies: dict[str, dict]) -> bytes:
    return _build_df(people).to_csv(index=False).encode("utf-8-sig")


def to_xlsx(people: list[dict], companies: dict[str, dict]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        _build_df(people).to_excel(w, sheet_name="People", index=False)
        if companies:
            _build_company_df(companies).to_excel(w, sheet_name="Companies", index=False)
        for sheet_name in w.sheets:                      # auto-fit column widths
            ws = w.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                header_len = len(str(col[0].value or ""))
                ws.column_dimensions[col[0].column_letter].width = min(max(max_len, header_len) + 3, 55)
    return buf.getvalue()


def to_docx(people: list[dict], companies: dict[str, dict], elapsed: float = 0) -> bytes:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    style.paragraph_format.space_after = Pt(4)

    t = doc.add_heading("Contact Intelligence Report", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = t.runs[0]
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0x1F, 0x4F, 0xD8)

    n_high = sum(1 for x in people if x["profile"].get("confidence") == "high")
    doc.add_paragraph(
        f"{len(people)} contacts · {n_high} high confidence · "
        f"{len(companies)} companies · {elapsed}s research time"
    ).runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    doc.add_heading("People", level=1)
    headers = ["S.No", "Name", "Title", "Department", "AI Relevance", "Notes", "Source"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(9)

    for idx, x in enumerate(people, 1):
        p, c = x["profile"], x["contact"]
        vals = [str(idx), p.get("full_name") or c["name_guess"], p.get("current_title", ""),
                p.get("department", ""), p.get("ai_relevance", ""), p.get("notes", ""), _best_source(x)]
        row = table.add_row()
        for i, v in enumerate(vals):
            row.cells[i].text = v
            for r in row.cells[i].paragraphs[0].runs:
                r.font.size = Pt(9)

    if companies:
        doc.add_heading("Companies", level=1)
        co_headers = ["Company", "Domain", "Industry", "HQ", "Size"]
        co_table = doc.add_table(rows=1, cols=len(co_headers))
        co_table.style = "Light Grid Accent 1"
        for i, h in enumerate(co_headers):
            cell = co_table.rows[0].cells[i]
            cell.text = h
            for r in cell.paragraphs[0].runs:
                r.bold = True
                r.font.size = Pt(9)
        for d, co in companies.items():
            vals = [co.get("company_name", d), d, co.get("industry", ""),
                    co.get("headquarters", ""), co.get("size", "")]
            row = co_table.add_row()
            for i, v in enumerate(vals):
                row.cells[i].text = v
                for r in row.cells[i].paragraphs[0].runs:
                    r.font.size = Pt(9)

    doc.add_paragraph("")
    footer = doc.add_paragraph("AI-assembled from public web data — verify before relying on it.")
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def to_pdf(people: list[dict], companies: dict[str, dict], elapsed: float = 0) -> bytes:
    from fpdf import FPDF

    def _latin(t: str) -> str:
        """Replace non-latin-1 chars so built-in Helvetica does not choke."""
        return (t.replace("—", "-").replace("–", "-").replace("‘", "'").replace("’", "'")
                 .replace("“", '"').replace("”", '"').replace("…", "..."))

    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(31, 79, 216)
    pdf.cell(0, 12, _latin("Contact Intelligence Report"), new_x="LMARGIN", new_y="NEXT")

    n_high = sum(1 for x in people if x["profile"].get("confidence") == "high")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, _latin(f"{len(people)} contacts  |  {n_high} high confidence  |  "
                          f"{len(companies)} companies  |  {elapsed}s research time"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "People", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    headers = ["S.No", "Name", "Title", "Dept", "AI Relevance", "Notes", "Source"]
    widths = [14, 38, 42, 30, 28, 72, 53]

    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(234, 240, 255)
    pdf.set_text_color(31, 79, 216)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(15, 23, 42)
    for idx, x in enumerate(people, 1):
        p, c = x["profile"], x["contact"]
        src = _best_source(x)
        src_disp = src.replace("https://", "").replace("http://", "")[:35] if src else ""
        notes_text = (p.get("notes") or "").replace("\n", " | ")
        vals = [str(idx),
                (p.get("full_name") or c["name_guess"])[:24],
                (p.get("current_title") or "")[:26],
                (p.get("department") or "")[:18],
                (p.get("ai_relevance") or "")[:12],
                notes_text[:48],
                src_disp]
        for v, w in zip(vals, widths):
            pdf.cell(w, 6, _latin(v), border=1)
        pdf.ln()

    if companies:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "Companies", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        co_h = ["Company", "Domain", "Industry", "HQ", "Size"]
        co_w = [60, 50, 55, 55, 47]
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(234, 240, 255)
        pdf.set_text_color(31, 79, 216)
        for h, w in zip(co_h, co_w):
            pdf.cell(w, 7, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(15, 23, 42)
        for d, co in companies.items():
            vals = [(co.get("company_name") or d)[:38], d[:32], (co.get("industry") or "")[:35],
                    (co.get("headquarters") or "")[:35], (co.get("size") or "")[:28]]
            for v, w in zip(vals, co_w):
                pdf.cell(w, 6, _latin(v), border=1)
            pdf.ln()

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, _latin("AI-assembled from public web data - verify before relying on it."))
    return bytes(pdf.output())


# ======================================================================================
# SECTION 8 — Styling
# ======================================================================================

CSS = """
<style>
:root{--ink:#0f172a;--ink2:#334155;--muted:#64748b;--line:#e5e9f0;--bg:#f6f8fb;--card:#fff;
--accent:#1f4fd8;--accent-soft:#eaf0ff;--ok:#0f9d6b;--warn:#b7791f;--bad:#b42318;}
.stApp,.stApp p,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp li,.stApp summary,.stApp h1,.stApp h2,.stApp h3{font-family:"Segoe UI",Inter,-apple-system,system-ui,Roboto,"Helvetica Neue",sans-serif;}
[data-testid="stIconMaterial"],.material-symbols-rounded{font-family:"Material Symbols Rounded"!important;}
.stApp{background:var(--bg);color:var(--ink);}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none!important;}
.block-container{max-width:1120px;padding:2.2rem 1.5rem 4rem;}

.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.6rem;}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:600;letter-spacing:.01em;color:var(--ink);}
.logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#1f4fd8,#5b8cff);display:grid;place-items:center;color:#fff;font-size:13px;}
.tag{font-size:.74rem;color:var(--muted);border:1px solid var(--line);background:#fff;border-radius:999px;padding:.22rem .7rem;}
.hero h1{font-size:2.05rem;line-height:1.15;margin:0 0 .35rem;font-weight:650;letter-spacing:-.02em;color:var(--ink);}
.hero p{margin:0 0 1.4rem;color:var(--muted);font-size:1.02rem;max-width:640px;}

[data-testid="stTabs"] [role="tablist"]{gap:1.1rem;border-bottom:1px solid var(--line);}
[data-testid="stTabs"] button[role="tab"]{padding:.55rem 1rem;color:var(--muted);font-weight:500;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent);}
[data-testid="stForm"]{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem 1.4rem;box-shadow:0 1px 2px rgba(15,23,42,.04);}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{border-radius:10px;border:1px solid var(--line);background:#fff;}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);}
label p{font-size:.82rem!important;font-weight:550!important;color:var(--ink2)!important;}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:10px;font-weight:600;padding:.55rem 1.3rem;border:1px solid var(--line);transition:all .15s ease;}
[data-testid="stFormSubmitButton"]>button[kind="primary"],.stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent);color:#fff;}
[data-testid="stFormSubmitButton"]>button[kind="primary"]:hover{background:#1941b3;box-shadow:0 4px 14px rgba(31,79,216,.28);transform:translateY(-1px);}
[data-testid="stStatusWidget"],[data-testid="stExpander"]{border-radius:12px;border:1px solid var(--line);background:var(--card);}
[data-testid="stExpander"] summary{font-size:.86rem;color:var(--muted);}

.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin:1.6rem 0 1.1rem;}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.9rem 1.1rem;}
.kpi b{display:block;font-size:1.55rem;font-weight:650;letter-spacing:-.02em;}
.kpi span{font-size:.76rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
.sec{font-size:.78rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:650;margin:1.8rem 0 .8rem;}

.pc{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.15rem;margin-bottom:.4rem;
box-shadow:0 1px 2px rgba(15,23,42,.04);transition:box-shadow .18s ease,transform .18s ease,border-color .18s ease;}
.pc:hover{box-shadow:0 8px 24px rgba(15,23,42,.08);transform:translateY(-1px);border-color:#d3dbe8;}
.pc-top{display:flex;gap:.8rem;align-items:center;}
.av{flex:0 0 42px;height:42px;border-radius:12px;background:var(--accent-soft);color:var(--accent);display:grid;place-items:center;font-weight:650;font-size:.92rem;}
.av.co{background:#eef2f7;color:var(--ink2);}
.pc-id{flex:1;min-width:0;}
.pc-name{font-weight:650;font-size:1.02rem;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pc-role{font-size:.88rem;color:var(--ink2);margin-top:.1rem;}
.pill{font-size:.7rem;font-weight:650;padding:.2rem .6rem;border-radius:999px;white-space:nowrap;}
.pill.hi{background:#e7f7f0;color:var(--ok);}.pill.md{background:#fff5e0;color:var(--warn);}
.pill.lo{background:#fdeceb;color:var(--bad);}.pill.nf{background:#eef1f5;color:var(--muted);}
.meta{display:flex;flex-wrap:wrap;gap:.35rem 1rem;margin-top:.8rem;font-size:.82rem;color:var(--muted);}
.meta b{color:var(--ink2);font-weight:550;}
.sum{margin-top:.7rem;font-size:.88rem;color:var(--ink2);line-height:1.5;}
.links{margin-top:.8rem;display:flex;gap:.5rem;flex-wrap:wrap;}
.lnk{font-size:.78rem;font-weight:600;color:var(--accent)!important;background:var(--accent-soft);border-radius:8px;padding:.28rem .65rem;text-decoration:none!important;}
.lnk:hover{filter:brightness(.96);}
.err{margin-top:.7rem;font-size:.82rem;color:var(--bad);}
.det h5{font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:.9rem 0 .35rem;}
.det ul{margin:0;padding-left:1.1rem;color:var(--ink2);font-size:.87rem;line-height:1.55;}
.det a{color:var(--accent);text-decoration:none;word-break:break-all;}
.det p{font-size:.86rem;color:var(--ink2);margin:0;line-height:1.5;}
.note{font-size:.78rem;color:var(--muted);margin-top:.9rem;}

.res-tbl{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.88rem;}
.res-tbl th{background:var(--accent-soft);color:var(--accent);font-weight:650;text-transform:uppercase;font-size:.72rem;
letter-spacing:.06em;padding:.65rem .7rem;border-bottom:2px solid var(--accent);text-align:left;}
.res-tbl td{padding:.6rem .7rem;border-bottom:1px solid var(--line);color:var(--ink2);vertical-align:top;}
.res-tbl tr:hover td{background:#f8fafd;}
.res-tbl .sno{width:3.5rem;text-align:center;color:var(--muted);font-weight:600;}
.res-tbl .name-cell{font-weight:600;color:var(--ink);}
.res-tbl .notes-cell{font-size:.82rem;line-height:1.45;max-width:280px;}
.res-tbl .src-cell a{color:var(--accent);text-decoration:none;font-size:.8rem;word-break:break-all;}
.res-tbl .src-cell a:hover{text-decoration:underline;}
.ai-pill{font-size:.68rem;font-weight:650;padding:.18rem .5rem;border-radius:999px;white-space:nowrap;display:inline-block;}
.ai-pill.vh{background:#e0f2fe;color:#0369a1;}.ai-pill.h{background:#e7f7f0;color:var(--ok);}
.ai-pill.m{background:#fff5e0;color:var(--warn);}.ai-pill.l{background:#eef1f5;color:var(--muted);}

.foot{margin-top:2.4rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.78rem;color:var(--muted);}
.gate{max-width:380px;margin:12vh auto 0;}
.upload-hint{font-size:.82rem;color:var(--muted);margin-top:.3rem;}
.export-row{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;}
@media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr);}.hero h1{font-size:1.6rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ======================================================================================
# SECTION 9 — UI render helpers
# ======================================================================================

def safe_href(u: str) -> str:
    return esc(u, quote=True) if u.startswith(("http://", "https://")) else ""


def initials(name: str) -> str:
    parts = [p for p in name.replace(".", " ").split() if p]
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper() if parts else "?"


def host(u: str) -> str:
    return (urlparse(u).netloc or u).removeprefix("www.")


CONF = {"high": ("hi", "High confidence"), "medium": ("md", "Medium"),
        "low": ("lo", "Low"), "not_found": ("nf", "Not found")}


def _ai_pill(level: str) -> str:
    cls = {"Very High": "vh", "High": "h", "Medium": "m", "Low": "l"}.get(level, "m")
    return f'<span class="ai-pill {cls}">{esc(level)}</span>' if level else ""


def people_table(people: list[dict]) -> str:
    """S.No | Name | Title | Department | AI Relevance / Interest | Notes | Source"""
    h = ('<table class="res-tbl"><thead><tr>'
         '<th class="sno">S.No</th><th>Name</th><th>Title</th><th>Department</th>'
         '<th>AI Relevance / Interest</th><th>Notes</th><th>Source</th>'
         '</tr></thead><tbody>')
    for idx, item in enumerate(people, 1):
        p, c = item["profile"], item["contact"]
        name = p.get("full_name") or c["name_guess"]
        notes = esc(p.get("notes") or "").replace("\n", "<br>")
        src = _best_source(item)
        src_html = (f'<a href="{safe_href(src)}" target="_blank" rel="noopener noreferrer">'
                    f'{esc(host(src))}</a>') if safe_href(src) else ""
        h += (f'<tr><td class="sno">{idx}</td>'
              f'<td class="name-cell">{esc(name)}</td>'
              f'<td>{esc(p.get("current_title") or "")}</td>'
              f'<td>{esc(p.get("department") or "")}</td>'
              f'<td>{_ai_pill(p.get("ai_relevance") or "")}</td>'
              f'<td class="notes-cell">{notes}</td>'
              f'<td class="src-cell">{src_html}</td></tr>')
    return h + "</tbody></table>"


def person_card(item: dict) -> str:
    c, p = item["contact"], item["profile"]
    name = p.get("full_name") or c["name_guess"]
    role = " · ".join(x for x in (p.get("current_title"), p.get("current_company")) if x) or "Role not found"
    cls, label = CONF.get(p.get("confidence", "not_found"), CONF["not_found"])
    meta = f'<span><b>Email</b> {esc(c["email"])}</span>'
    if p.get("location"):
        meta += f'<span><b>Location</b> {esc(p["location"])}</span>'
    links = ""
    if safe_href(p.get("linkedin_url", "")):
        links = (f'<div class="links"><a class="lnk" href="{safe_href(p["linkedin_url"])}" '
                 f'target="_blank" rel="noopener noreferrer">LinkedIn ↗</a></div>')
    summary = f'<div class="sum">{esc(p["summary"])}</div>' if p.get("summary") else ""
    err = f'<div class="err">{esc(p["_error"])}</div>' if p.get("_error") else ""
    return (f'<div class="pc"><div class="pc-top"><div class="av">{esc(initials(name))}</div>'
            f'<div class="pc-id"><div class="pc-name">{esc(name)}</div><div class="pc-role">{esc(role)}</div></div>'
            f'<span class="pill {cls}">{label}</span></div>'
            f'<div class="meta">{meta}</div>{summary}{links}{err}</div>')


def person_details(item: dict) -> str:
    p = item["profile"]
    h = '<div class="det">'
    if p.get("previous_roles"):
        h += "<h5>Previous roles</h5><ul>" + "".join(
            f'<li>{esc(r.get("title", ""))}{" — " + esc(r["company"]) if r.get("company") else ""}'
            f'{" (" + esc(r["period"]) + ")" if r.get("period") else ""}</li>'
            for r in p["previous_roles"]) + "</ul>"
    if p.get("match_reasoning"):
        h += f'<h5>Why this match</h5><p>{esc(p["match_reasoning"])}</p>'
    if p.get("sources"):
        h += "<h5>Sources</h5><ul>" + "".join(
            f'<li><a href="{safe_href(u)}" target="_blank" rel="noopener noreferrer">{esc(host(u))}</a></li>'
            for u in p["sources"] if safe_href(u)) + "</ul>"
    if item["contact"].get("used_apollo"):
        h += '<div class="note">Verified against Apollo structured data.</div>'
    return h + "</div>" if len(h) > 22 else '<div class="det"><p>No additional details found.</p></div>'


def company_card(d: str, co: dict) -> str:
    name = co.get("company_name") or d
    meta = "".join(f"<span><b>{k}</b> {esc(co[f])}</span>" for k, f in
                   (("HQ", "headquarters"), ("Industry", "industry"), ("Size", "size")) if co.get(f))
    desc = f'<div class="sum">{esc(co["what_they_do"])}</div>' if co.get("what_they_do") else ""
    grp = (f'<div class="meta"><span><b>Group</b> {esc(co["parent_or_group"])}</span></div>'
           if co.get("parent_or_group") else "")
    err = f'<div class="err">{esc(co["_error"])}</div>' if co.get("_error") else ""
    return (f'<div class="pc"><div class="pc-top"><div class="av co">{esc(initials(name))}</div>'
            f'<div class="pc-id"><div class="pc-name">{esc(name)}</div><div class="pc-role">{esc(d)}</div></div></div>'
            f'<div class="meta">{meta}</div>{grp}{desc}{err}</div>')


def company_details(co: dict) -> str:
    h = '<div class="det">'
    if co.get("products"):
        h += "<h5>Products &amp; services</h5><p>" + esc(", ".join(co["products"])) + "</p>"
    if co.get("other_offices"):
        h += "<h5>Other offices</h5><p>" + esc(", ".join(co["other_offices"])) + "</p>"
    if co.get("key_leadership"):
        h += "<h5>Leadership</h5><ul>" + "".join(
            f'<li>{esc(x["name"])}{" — " + esc(x["title"]) if x.get("title") else ""}</li>'
            for x in co["key_leadership"]) + "</ul>"
    if co.get("sources"):
        h += "<h5>Sources</h5><ul>" + "".join(
            f'<li><a href="{safe_href(u)}" target="_blank" rel="noopener noreferrer">{esc(host(u))}</a></li>'
            for u in co["sources"] if safe_href(u)) + "</ul>"
    return h + "</div>" if len(h) > 22 else '<div class="det"><p>No additional details found.</p></div>'


# ======================================================================================
# SECTION 10 — Access gate (optional: set APP_PASSWORD)
# ======================================================================================

APP_PASSWORD = cfg("APP_PASSWORD")
if APP_PASSWORD and not st.session_state.get("authed"):
    st.markdown('<div class="gate"><div class="brand"><div class="logo">◆</div>UBTI Contact Intelligence</div>'
                '<p style="color:#64748b;margin:.8rem 0 0">Enter your access code to continue.</p></div>',
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid, st.form("gate"):
        code = st.text_input("Access code", type="password")
        if st.form_submit_button("Continue", type="primary"):
            if hmac.compare_digest(code.encode(), APP_PASSWORD.encode()):
                st.session_state["authed"] = True
                st.rerun()
            st.error("Incorrect access code.")
    st.stop()


# ======================================================================================
# SECTION 11 — Header + inputs
# ======================================================================================

st.markdown(
    '<div class="topbar"><div class="brand"><div class="logo">◆</div>UBTI Contact Intelligence</div>'
    '<div class="tag">Public-web research · sources included</div></div>'
    '<div class="hero"><h1>Know who you are talking to.</h1>'
    '<p>Enter a name and email, paste an Outlook header, drop a screenshot of a meeting invite, '
    'or just paste messy text — we will figure it out.</p></div>',
    unsafe_allow_html=True,
)

if not S.ready:
    st.error("This service is not configured yet. Please contact your administrator.")
    st.stop()


def execute(contacts: list[Contact], hint: str, force: bool) -> None:
    if not contacts:
        st.warning("No valid email address found. Please check the input.")
        return
    if len(contacts) > S.max_contacts:
        st.info(f"Processing the first {S.max_contacts} of {len(contacts)} contacts.")
        contacts = contacts[: S.max_contacts]
    final = None
    with st.status("Researching…", expanded=True) as status:
        bar = st.progress(0.0)
        for ev in run_lookup(contacts, hint, S, force):
            if ev["type"] == "progress":
                frac = ev["done"] / ev["total"] if ev["total"] else 0.0
                bar.progress(min(frac, 1.0))
                status.update(label=f"Researching · {ev['done']} of {ev['total']} complete")
                st.write(ev["message"])
            else:
                final = ev["data"]
        status.update(label=f"Completed in {final['elapsed']}s", state="complete", expanded=False)
    st.session_state["results"] = final


tab_one, tab_bulk, tab_image = st.tabs(["Single lookup", "Bulk · Outlook paste", "Screenshot / Image"])

with tab_one:
    with st.form("single", border=True):
        c1, c2 = st.columns(2)
        in_name = c1.text_input("Full name", placeholder="Paul Raj")
        in_email = c2.text_input("Office email", placeholder="paulr@company.com")
        in_ctx = st.text_input("Company or context (optional)", placeholder="e.g. Ducont — Dubai IT services")
        f1, f2 = st.columns([3, 1.2])
        fresh1 = f1.checkbox("Ignore saved results (fresh search)", key="fresh1")
        go_one = f2.form_submit_button("Find contact", type="primary", use_container_width=True)
    if go_one:
        m = EMAIL_RE.fullmatch(in_email.strip().strip("<>"))
        if not m:
            st.warning("Please enter a valid office email address.")
        else:
            em = m.group(0).lower()
            local, dom = em.split("@", 1)
            nm = " ".join(in_name.split())
            execute([Contact(email=em, display_name=nm, local=local, domain=dom,
                             name_guess=nm or guess_name_from_local(local),
                             company_hint=in_ctx.strip())],
                    in_ctx.strip(), fresh1)

with tab_bulk:
    with st.form("bulk", border=True):
        raw = st.text_area("Emails, Outlook header, or any text with contacts", height=150,
                           placeholder="Paste anything — a To/Cc line, a meeting invite, names and emails in any "
                                       "format, or even a messy copy-paste. We'll extract the contacts.")
        in_ctx2 = st.text_input("Company or context (optional)", key="ctx2")
        b1, b2, b3 = st.columns([1.6, 1.8, 1.2])
        inc_sender = b1.checkbox("Include sender (From)")
        fresh2 = b2.checkbox("Ignore saved results", key="fresh2")
        go_bulk = b3.form_submit_button("Find contacts", type="primary", use_container_width=True)
    if go_bulk:
        contacts = parse_contacts(raw, in_ctx2, include_sender=inc_sender)   # fast regex pass
        all_emails = {e.lower() for e in EMAIL_RE.findall(raw or "")}
        found_emails = {c.email for c in contacts}
        missed = all_emails - found_emails
        hint_bulk = in_ctx2.strip()

        if not contacts or missed:       # fall back to the LLM smart parser
            with st.spinner("Extracting contacts from unstructured text…"):
                try:
                    smart_contacts, llm_hint = smart_parse(raw, S, in_ctx2, include_sender=inc_sender)
                except Exception:
                    smart_contacts, llm_hint = [], ""
            added = 0
            for sc in smart_contacts:
                if sc.email not in found_emails:
                    contacts.append(sc)
                    found_emails.add(sc.email)
                    added += 1
            hint_bulk = hint_bulk or (llm_hint or "").strip()
            if added:
                st.success(f"AI extracted {added} additional contact(s) from the text.")
        execute(contacts, hint_bulk, fresh2)

with tab_image:
    uploaded = st.file_uploader(
        "Upload a screenshot or image",
        type=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
        help="Screenshot of an email, meeting invite, business card, or any image with contact information.",
    )
    st.markdown('<p class="upload-hint">Supported: email screenshots, meeting invites, business cards, '
                'chat screenshots — any image with names and email addresses.</p>', unsafe_allow_html=True)
    img_ctx = st.text_input("Company or context (optional)", key="ctx_img")
    ic1, ic2, ic3 = st.columns([1.6, 1.8, 1.2])
    img_sender = ic1.checkbox("Include sender (From)", key="img_sender", value=True)
    img_fresh = ic2.checkbox("Ignore saved results", key="fresh_img")
    go_img = ic3.button("Extract & find contacts", type="primary", use_container_width=True)

    if go_img:
        if not uploaded:
            st.warning("Please upload an image first.")
        else:
            image_bytes = uploaded.read()
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp"}
            ext = (uploaded.name or "image.png").rsplit(".", 1)[-1].lower()
            mime = mime_map.get(ext, "image/png")

            with st.spinner("Reading image with AI vision…"):
                try:
                    contacts, hint = contacts_from_image(image_bytes, S, mime, include_sender=img_sender)
                except Exception as e:
                    st.error(f"Could not read the image. Please try a clearer screenshot. ({type(e).__name__})")
                    contacts, hint = [], img_ctx

            if contacts:
                st.success(f"Extracted {len(contacts)} contacts from the image.")
                execute(contacts, (hint or img_ctx).strip(), img_fresh)
            else:
                st.warning("No email addresses could be extracted from this image. "
                           "Try a clearer screenshot or use the Bulk tab to paste the text manually.")


# ======================================================================================
# SECTION 12 — Results
# ======================================================================================

res = st.session_state.get("results")
if res:
    people, companies = res["people"], res["companies"]
    n_high = sum(1 for x in people if x["profile"].get("confidence") == "high")
    st.markdown(
        '<div class="kpis">'
        f'<div class="kpi"><b>{len(people)}</b><span>Contacts</span></div>'
        f'<div class="kpi"><b>{n_high}</b><span>High confidence</span></div>'
        f'<div class="kpi"><b>{len(companies)}</b><span>Companies</span></div>'
        f'<div class="kpi"><b>{res["elapsed"]}s</b><span>Research time</span></div></div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sec">People</div>', unsafe_allow_html=True)
    st.markdown(people_table(people), unsafe_allow_html=True)

    for i in range(0, len(people), 2):          # expandable detail cards below the table
        cols = st.columns(2, gap="medium")
        for col, item in zip(cols, people[i:i + 2]):
            with col:
                p, c = item["profile"], item["contact"]
                nm = p.get("full_name") or c["name_guess"]
                with st.expander(f"🔍 {nm} — details & sources"):
                    st.markdown(person_card(item), unsafe_allow_html=True)
                    st.markdown(person_details(item), unsafe_allow_html=True)

    if companies:
        st.markdown('<div class="sec">Companies</div>', unsafe_allow_html=True)
        items = list(companies.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2, gap="medium")
            for col, (d, co) in zip(cols, items[i:i + 2]):
                with col:
                    st.markdown(company_card(d, co), unsafe_allow_html=True)
                    with st.expander("Details & sources"):
                        st.markdown(company_details(co), unsafe_allow_html=True)

    st.markdown('<div class="sec">Export results</div>', unsafe_allow_html=True)
    ex1, ex2, ex3, ex4, ex5, _ = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 3])
    elapsed = res.get("elapsed", 0)

    ex1.download_button("📄 CSV", to_csv(people, companies), "contacts.csv", "text/csv",
                        use_container_width=True)
    ex2.download_button("📊 Excel", to_xlsx(people, companies), "contacts.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)
    ex3.download_button("📝 Word", to_docx(people, companies, elapsed), "contacts.docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True)
    ex4.download_button("📕 PDF", to_pdf(people, companies, elapsed), "contacts.pdf",
                        "application/pdf", use_container_width=True)

    if ex5.button("Clear results", use_container_width=True):
        st.session_state.pop("results", None)
        st.rerun()

    if res.get("notes"):
        st.markdown(f'<div class="note">{esc(" ".join(res["notes"]))}</div>', unsafe_allow_html=True)

st.markdown('<div class="foot">AI-assembled from public web data — verify before relying on it. '
            'Use for legitimate B2B purposes only.</div>', unsafe_allow_html=True)
