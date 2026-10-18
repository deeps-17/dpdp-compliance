"""
Policy scraper: downloads each company's privacy policy, cleans it to plain text,
saves it under data/raw/, and records the outcome in the database.

Run from the project root:
    python src/collection/scrape_policies.py --limit 10          # test on 10 companies first
    python src/collection/scrape_policies.py                     # all companies not yet collected
    python src/collection/scrape_policies.py --retry             # also retry failed/blocked ones
    python src/collection/scrape_policies.py --sector "Fintech"  # only one sector
    python src/collection/scrape_policies.py --dry-run           # just list what would be scraped

Safe to stop (Ctrl+C) and run again: finished companies are skipped.

Good-citizen rules built in: checks robots.txt, waits between requests, and identifies itself
honestly. Optionally set a contact address with:  $env:SCRAPER_CONTACT = "project-email@example.com"
"""
import argparse
import csv
import hashlib
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "db" / "dpdp.db"
DEFAULT_RAW = ROOT / "data" / "raw"

ROBOT_TOKEN = "DPDP-FYP-ResearchBot"
CONTACT = os.environ.get("SCRAPER_CONTACT", "").strip()
USER_AGENT = f"{ROBOT_TOKEN}/1.0 (AKGEC final year project; academic use)"
if CONTACT:
    USER_AGENT += f" contact: {CONTACT}"

TIMEOUT = 20
PRIVACY_WORDS = ("privacy", "personal data", "personal information", "data protection")
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "br", "tr", "td", "th", "table", "section",
              "article", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]
# Page furniture that is never policy text. (We do NOT remove <form>: some old sites wrap
# the whole page in one.)
JUNK_TAGS = ["script", "style", "noscript", "svg", "iframe", "nav", "header", "footer", "aside"]
JUNK_ATTR = re.compile(r"cookie|consent-banner|navbar|breadcrumb|site-header|site-footer|"
                       r"newsletter|popup|modal|sidebar|menu", re.I)


# ---------------------------------------------------------------- helpers
def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "unknown"


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


_robots_cache = {}


def robots_allows(session, url):
    """True if robots.txt allows us to fetch this URL (or there is no usable robots.txt)."""
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in _robots_cache:
        rp = robotparser.RobotFileParser()
        try:
            r = session.get(origin + "/robots.txt", timeout=10)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except requests.RequestException:
            rp.parse([])
        _robots_cache[origin] = rp
    return _robots_cache[origin].can_fetch(ROBOT_TOKEN, url)


def fetch(session, url):
    """GET with a couple of retries. Returns (response, error_message)."""
    last_error = None
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        except requests.RequestException as exc:
            last_error = str(exc)[:200]
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(5 * (attempt + 1))
            continue
        return resp, None
    return None, last_error


def html_to_text(html):
    """Strip page furniture and return the readable text, one block per line."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(JUNK_TAGS):
        tag.decompose()

    body = soup.body or soup
    total = len(body.get_text(" ", strip=True)) or 1

    # Remove cookie banners, menus, popups, but never a wrapper that holds most of the page.
    for tag in soup.find_all(attrs={"class": JUNK_ATTR}) + soup.find_all(attrs={"id": JUNK_ATTR}):
        try:
            if getattr(tag, "decomposed", False):
                continue
            if len(tag.get_text(" ", strip=True)) <= 0.5 * total:
                tag.decompose()
        except Exception:
            continue

    body = soup.body or soup
    total_after = len(body.get_text(" ", strip=True)) or 1
    main = soup.find("main") or soup.find("article")
    container = main if main and len(main.get_text(" ", strip=True)) >= 0.4 * total_after else body

    # Put a line break around block elements so sentences containing links stay on one line.
    for tag in container.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = container.get_text(separator="").replace("\xa0", " ")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def scrape_one(session, url, min_chars):
    """Returns (status, note, text). status is collected / blocked / failed."""
    if not robots_allows(session, url):
        return "blocked", "robots.txt does not allow automated access", ""

    resp, error = fetch(session, url)
    if resp is None:
        return "failed", f"request error: {error}", ""
    if resp.status_code in (401, 403):
        return "blocked", f"HTTP {resp.status_code} (site refuses automated access)", ""
    if resp.status_code in (404, 410):
        return "failed", f"HTTP {resp.status_code}: page not found, URL may be outdated", ""
    if resp.status_code != 200:
        return "failed", f"HTTP {resp.status_code}", ""

    ctype = resp.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        return "failed", "this is a PDF: copy the text by hand", ""
    if "html" not in ctype and "text" not in ctype:
        return "failed", f"unexpected content type: {ctype[:60]}", ""

    # requests guesses Latin-1 when no charset is given, which garbles characters like the rupee sign.
    if resp.encoding is None or (resp.encoding.lower() == "iso-8859-1" and "charset" not in ctype):
        resp.encoding = resp.apparent_encoding or "utf-8"

    text = html_to_text(resp.text)
    low = text.lower()

    if len(text) < min_chars:
        hint = ""
        if any(w in low for w in ("enable javascript", "captcha", "access denied", "verify you are human")):
            hint = " (looks like a bot check)"
        return "failed", f"only {len(text)} characters of text: probably needs JavaScript or the URL is wrong{hint}", ""
    if not any(w in low for w in PRIVACY_WORDS):
        return "failed", "text never mentions privacy or personal data: check the URL", ""

    note = ""
    if resp.url.rstrip("/") != url.rstrip("/"):
        note = f"redirected to {resp.url[:150]}"
    return "collected", note, text


# ---------------------------------------------------------------- database
def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def pick_targets(conn, args):
    statuses = ["not_started"] + (["failed", "blocked"] if args.retry else [])
    marks = ",".join("?" * len(statuses))
    sql = (f"SELECT policy_id, company_name, sector, source_url FROM policies "
           f"WHERE source_url IS NOT NULL AND status IN ({marks})")
    params = list(statuses)
    if args.sector:
        sql += " AND sector LIKE ?"
        params.append(f"%{args.sector}%")
    if args.company:
        sql += " AND company_name LIKE ?"
        params.append(f"%{args.company}%")
    sql += " ORDER BY policy_id"
    if args.limit:
        sql += " LIMIT ?"
        params.append(args.limit)
    return conn.execute(sql, params).fetchall()


def save_result(conn, policy_id, status, note, raw_path, content_hash):
    conn.execute(
        "UPDATE policies SET status=?, status_note=?, date_accessed=?, content_hash=?, raw_path=? "
        "WHERE policy_id=?",
        (status, note or None, now_utc(), content_hash, raw_path, policy_id),
    )
    conn.commit()


def twin_of(conn, policy_id, content_hash):
    row = conn.execute(
        "SELECT company_name FROM policies WHERE content_hash=? AND policy_id<>? LIMIT 1",
        (content_hash, policy_id),
    ).fetchone()
    return row["company_name"] if row else None


# ---------------------------------------------------------------- main
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Download and clean privacy policies listed in the database.")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    ap.add_argument("--limit", type=int, help="only process the first N companies")
    ap.add_argument("--retry", action="store_true", help="also retry failed and blocked companies")
    ap.add_argument("--sector", help="only companies whose sector contains this text")
    ap.add_argument("--company", help="only companies whose name contains this text")
    ap.add_argument("--min-delay", type=float, default=1.5)
    ap.add_argument("--max-delay", type=float, default=2.5)
    ap.add_argument("--min-chars", type=int, default=1500,
                    help="pages with less text than this count as failed")
    ap.add_argument("--dry-run", action="store_true", help="list what would be scraped, then stop")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    conn = connect(args.db)
    targets = pick_targets(conn, args)
    print(f"{len(targets)} companies to scrape.")
    if args.dry_run or not targets:
        for t in targets:
            print(f"  {t['policy_id']:>4}  {t['company_name']}  ->  {t['source_url']}")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"})

    results = []
    try:
        for i, t in enumerate(targets, start=1):
            pid, company, sector, url = t["policy_id"], t["company_name"], t["sector"], t["source_url"]
            status, note, text = scrape_one(session, url, args.min_chars)

            raw_path = content_hash = None
            if status == "collected":
                folder = raw_dir / slugify(sector)
                folder.mkdir(parents=True, exist_ok=True)
                file_path = folder / f"{pid:04d}-{slugify(company)}.txt"
                file_path.write_text(text, encoding="utf-8")
                try:
                    raw_path = file_path.relative_to(ROOT).as_posix()
                except ValueError:
                    raw_path = file_path.as_posix()
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                twin = twin_of(conn, pid, content_hash)
                if twin:
                    note = (note + "; " if note else "") + f"identical text to {twin}"

            save_result(conn, pid, status, note, raw_path, content_hash)
            size = f"{len(text):,} chars" if status == "collected" else note[:70]
            print(f"[{i}/{len(targets)}] {company}: {status} ({size})")
            results.append((pid, company, sector, url, status, note))

            if status != "blocked" or "robots" not in note:      # we did contact the site
                time.sleep(random.uniform(args.min_delay, args.max_delay))
    except KeyboardInterrupt:
        print("\nStopped by you. Progress is saved; run the same command again to continue.")

    # ---- summary and report
    counts = {}
    for r in results:
        counts[r[4]] = counts.get(r[4], 0) + 1
    print("\nThis run:", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "nothing done")
    overall = conn.execute("SELECT status, COUNT(*) AS n FROM policies GROUP BY status ORDER BY status").fetchall()
    print("Whole database:", ", ".join(f"{r['status']}: {r['n']}" for r in overall))

    problems = [r for r in results if r[4] != "collected" or "identical text" in (r[5] or "")]
    if problems:
        raw_dir.mkdir(parents=True, exist_ok=True)
        report = raw_dir / "scrape_report.csv"
        with open(report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["policy_id", "company", "sector", "url", "status", "note"])
            w.writerows(problems)
        print(f"Companies needing a look: {len(problems)}. Details in {report}")
    conn.close()


if __name__ == "__main__":
    main()
