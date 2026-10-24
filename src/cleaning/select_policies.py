"""
Choose the policies that make up the dataset, and set the train/test split.

Step A:  python src/cleaning/select_policies.py --export
         Writes data/seeds/candidates.csv: every collected policy, with a text preview.
Step B:  Open candidates.csv in Excel, put 1 in the 'selected' column for the policies you want
         (select 12 per sector for 5 sectors = 60 total), and save it as data/seeds/selected_policies.csv (CSV).
Step C:  python src/cleaning/select_policies.py
         Checks your choice, puts 20% of the policies into TEST (fixed random seed),
         the rest into TRAIN, and saves this in the database (policies.split).

The split is made on whole policies, never on clauses. Once clauses exist in the database the
script refuses to change the split (use --force only if you really mean it).
"""
import argparse
import csv
import math
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "db" / "dpdp.db"
CANDIDATES = ROOT / "data" / "seeds" / "candidates.csv"
SELECTED = ROOT / "data" / "seeds" / "selected_policies.csv"
TRUE_VALUES = {"1", "y", "yes", "true", "x", "selected"}


def connect(db):
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def raw_file(raw_path):
    if not raw_path:
        return None
    p = Path(raw_path)
    return p if p.is_absolute() else ROOT / p


def export_candidates(conn, out_path):
    rows = conn.execute(
        "SELECT policy_id, company_name, sector, status_note, raw_path, content_hash "
        "FROM policies WHERE status='collected' ORDER BY sector, policy_id"
    ).fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["policy_id", "company", "sector", "chars", "preview", "same_text_as", "note", "selected"])
        for r in rows:
            path = raw_file(r["raw_path"])
            text = path.read_text(encoding="utf-8") if path and path.exists() else ""
            preview = " | ".join(text.splitlines()[:3])[:120]
            twins = [
                t["company_name"]
                for t in conn.execute(
                    "SELECT company_name FROM policies WHERE content_hash=? AND policy_id<>?",
                    (r["content_hash"], r["policy_id"]),
                )
            ] if r["content_hash"] else []
            w.writerow([
                r["policy_id"],
                r["company_name"],
                r["sector"],
                len(text),
                preview,
                "; ".join(twins),
                r["status_note"] or "",
                "",
            ])
    print(f"Wrote {len(rows)} collected policies to {out_path}")
    print("Next: open it in Excel, put 1 in the 'selected' column for 12 per sector (60 total),")
    print(f"      and save it as {SELECTED}")


def read_selection(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or "policy_id" not in rows[0]:
        sys.exit("The selection file needs a 'policy_id' column.")
    if "selected" in rows[0]:
        rows = [r for r in rows if (r.get("selected") or "").strip().lower() in TRUE_VALUES]
    ids = []
    for r in rows:
        try:
            ids.append(int(float(r["policy_id"])))
        except (TypeError, ValueError):
            sys.exit(f"Bad policy_id value: {r['policy_id']!r}")
    if len(ids) != len(set(ids)):
        sys.exit("The same policy_id is selected more than once.")
    return ids


def allocate_test_counts(sizes, fraction, rng):
    """How many test policies per sector: proportional, total = round(fraction * N)."""
    total_test = round(fraction * sum(sizes.values()))
    counts = {s: math.floor(fraction * n) for s, n in sizes.items()}
    for s, n in sizes.items():
        if n >= 3 and counts[s] == 0:
            counts[s] = 1
    leftovers = sorted(sizes, key=lambda s: (-(fraction * sizes[s] - math.floor(fraction * sizes[s])), rng.random()))
    i = 0
    while sum(counts.values()) < total_test and i < 10 * len(leftovers):
        s = leftovers[i % len(leftovers)]
        if counts[s] < sizes[s] - 1:  # always keep at least one training policy per sector
            counts[s] += 1
        i += 1
    return counts


def assign_split(conn, ids, fraction, seed, force):
    # Verify clauses table exists before checking count
    clauses_exist = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='clauses'"
    ).fetchone()
    if clauses_exist and not force and conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0] > 0:
        sys.exit("Clauses already exist, so the split is locked. Use --force only if you are sure.")

    marks = ",".join("?" * len(ids))
    found = {
        r["policy_id"]: r
        for r in conn.execute(
            f"SELECT policy_id, company_name, sector, status, content_hash FROM policies WHERE policy_id IN ({marks})",
            ids,
        )
    }
    missing = [i for i in ids if i not in found]
    if missing:
        sys.exit(f"These policy_ids are not in the database: {missing}")
    not_ok = [f"{found[i]['company_name']} ({found[i]['status']})" for i in ids if found[i]["status"] != "collected"]
    if not_ok:
        sys.exit("These are not 'collected': " + ", ".join(not_ok))

    by_hash = {}
    for i in ids:
        h = found[i]["content_hash"]
        if h:
            by_hash.setdefault(h, []).append(found[i]["company_name"])
    for names in by_hash.values():
        if len(names) > 1:
            print("WARNING: identical text selected twice:", ", ".join(names), "(keep only one)")

    by_sector = {}
    for i in sorted(ids):
        by_sector.setdefault(found[i]["sector"], []).append(i)
    rng = random.Random(seed)
    counts = allocate_test_counts({s: len(v) for s, v in by_sector.items()}, fraction, rng)

    conn.execute("UPDATE policies SET split=NULL")
    test_names = []
    print(f"\n{'Sector':<36}{'selected':>9}{'train':>7}{'test':>6}")
    for sector, members in sorted(by_sector.items()):
        members = members[:]
        rng.shuffle(members)
        test_ids, train_ids = members[: counts[sector]], members[counts[sector] :]
        conn.executemany("UPDATE policies SET split='test' WHERE policy_id=?", [(i,) for i in test_ids])
        conn.executemany("UPDATE policies SET split='train' WHERE policy_id=?", [(i,) for i in train_ids])
        test_names += [found[i]["company_name"] for i in test_ids]
        print(f"{str(sector):<36}{len(members):>9}{len(train_ids):>7}{len(test_ids):>6}")
    conn.commit()
    n_test = len(test_names)
    print(f"{'TOTAL':<36}{len(ids):>9}{len(ids) - n_test:>7}{n_test:>6}")
    print("\nTest policies (locked away until the final evaluation):")
    print("  " + ", ".join(sorted(test_names)))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Select dataset policies and assign the train/test split.")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--export", action="store_true", help="write candidates.csv and stop")
    ap.add_argument("--selection", default=str(SELECTED))
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="change the split even though clauses exist")
    args = ap.parse_args()

    conn = connect(args.db)
    if args.export:
        export_candidates(conn, CANDIDATES)
        return
    sel = Path(args.selection)
    if not sel.exists():
        sys.exit(f"Selection file not found: {sel}\nRun with --export first, then edit and save the file.")
    assign_split(conn, read_selection(sel), args.test_fraction, args.seed, args.force)
    conn.close()


if __name__ == "__main__":
    main()
