"""
Create (or rebuild) the SQLite database and optionally import the company list.

Usage (from the project root):
    python db/init_db.py                      # create db/dpdp.db if missing
    python db/init_db.py --reset              # delete and rebuild from scratch
    python db/init_db.py --seed data/seeds/company_list.csv

The CSV needs the columns: Sector, Company, Policy URL (extra columns are ignored).
Uses only the Python standard library.
"""
import argparse
import csv
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "db" / "schema.sql"
DEFAULT_DB = ROOT / "db" / "dpdp.db"

NO_URL_VALUES = {"", "not available", "n/a", "na", "none", "-"}
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}


def clean_url(url):
    """Remove tracking parameters such as ?utm_source=chatgpt.com."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in TRACKING_PARAMS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def create_db(db_path, reset):
    db_path = Path(db_path)
    if reset and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def seed_companies(conn, csv_path):
    added = skipped = no_url = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            company = (row.get("Company") or "").strip()
            if not company:
                skipped += 1
                continue
            sector = (row.get("Sector") or "").strip() or None
            raw_url = (row.get("Policy URL") or "").strip()
            if raw_url.lower() in NO_URL_VALUES:
                url, status, note = None, "excluded", "No policy URL found"
                no_url += 1
            else:
                url, status, note = clean_url(raw_url), "not_started", None
            cur = conn.execute(
                "INSERT OR IGNORE INTO policies (company_name, sector, source_url, status, status_note) "
                "VALUES (?, ?, ?, ?, ?)",
                (company, sector, url, status, note),
            )
            added += cur.rowcount
    conn.commit()
    return added, skipped, no_url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--reset", action="store_true", help="delete the database and rebuild it")
    ap.add_argument("--seed", help="path to company_list.csv to import")
    args = ap.parse_args()

    conn = create_db(args.db, args.reset)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    print(f"Database ready: {args.db}")
    print("Tables:", ", ".join(tables))

    if args.seed:
        added, skipped, no_url = seed_companies(conn, args.seed)
        total = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        print(f"Imported {added} companies ({no_url} without a URL, {skipped} blank rows skipped).")
        print(f"Policies in database: {total}")
    conn.close()


if __name__ == "__main__":
    main()
