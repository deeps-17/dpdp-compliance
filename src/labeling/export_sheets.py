"""
Make the labelling sheets from the clauses in the database.

Run from the project root:
    python src/labeling/export_sheets.py

What it does (everything is random with a fixed seed, so it is repeatable):
  1. Picks about 15 random clauses from each of the 60 policies.
  2. Training sample -> data/labels/suggest_batch_01.csv, _02.csv ...   (these go to Claude for suggestions)
  3. A random 100 of the training sample are marked 'blind check': the team labels these WITHOUT seeing
     any suggestion -> data/labels/train_blind_check.xlsx
  4. Test sample -> data/labels/test_for_humans.xlsx   (humans only, never shown to Claude)
  5. data/labels/sample_manifest.csv records exactly which clauses were picked.

Clauses that are too short or too long are skipped. A test clause that also appears word-for-word in a
training policy is skipped, so the models cannot score points by remembering.
It refuses to run again if the manifest exists (re-sampling after labelling started would break things).
"""
import argparse
import csv
import hashlib
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "db" / "dpdp.db"
OUT_DIR = ROOT / "data" / "labels"
STATUSES = ["satisfied", "not_satisfied"]


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def thash(text):
    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()


def words(text):
    return len(text.split())


def clip(text, n=400):
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def load_clauses(conn):
    rows = conn.execute(
        "SELECT c.clause_id, c.policy_id, c.clause_index, c.clause_text, p.company_name, p.split "
        "FROM clauses c JOIN policies p ON p.policy_id = c.policy_id "
        "WHERE p.split IN ('train','test') ORDER BY c.policy_id, c.clause_index").fetchall()
    by_policy = {}
    for r in rows:
        by_policy.setdefault(r["policy_id"], []).append(dict(r))
    return by_policy


def obligation_names(conn):
    names = [r["name"] for r in conn.execute(
        "SELECT name FROM obligations WHERE obligation_id > 0 ORDER BY obligation_id")]
    return names + ["None"]


def pick_sample(by_policy, per_policy, min_w, max_w, rng):
    train_hashes = {thash(c["clause_text"]) for cl in by_policy.values() for c in cl if c["split"] == "train"}
    chosen, seen = [], set()
    for pid in sorted(by_policy):
        clauses = by_policy[pid]
        split = clauses[0]["split"]
        eligible = [c for c in clauses if min_w <= words(c["clause_text"]) <= max_w]
        rng.shuffle(eligible)
        taken = 0
        for c in eligible:
            h = thash(c["clause_text"])
            if h in seen or (split == "test" and h in train_hashes):
                continue
            seen.add(h)
            chosen.append(c)
            taken += 1
            if taken >= per_policy:
                break
    return chosen


def with_context(c, by_policy):
    siblings = by_policy[c["policy_id"]]
    i = c["clause_index"]
    prev_c = next((s["clause_text"] for s in siblings if s["clause_index"] == i - 1), "")
    next_c = next((s["clause_text"] for s in siblings if s["clause_index"] == i + 1), "")
    return clip(prev_c), clip(next_c)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_human_sheet(path, rows, names):
    """Excel sheet with dropdowns. Falls back to CSV if openpyxl is not installed."""
    header = ["clause_id", "previous_clause", "clause_text", "next_clause", "obligation", "status", "note"]
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        csv_path = path.with_suffix(".csv")
        write_csv(csv_path, header, rows)
        print(f"  (openpyxl not installed, wrote {csv_path.name} instead: run  pip install openpyxl  for dropdowns)")
        return csv_path
    wb = Workbook()
    ws = wb.active
    ws.title = "Label"
    ws.append(header)
    for r in rows:
        ws.append(r)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color="1F2A44")
    for col, width in zip("ABCDEFG", (10, 45, 70, 45, 24, 16, 30)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        row[2].font = Font(bold=True)
    last = len(rows) + 1
    dv_ob = DataValidation(type="list", formula1='"' + ",".join(names) + '"', allow_blank=True)
    dv_st = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"', allow_blank=True)
    ws.add_data_validation(dv_ob)
    ws.add_data_validation(dv_st)
    dv_ob.add(f"E2:E{last}")
    dv_st.add(f"F2:F{last}")
    ws.freeze_panes = "D2"
    guide = wb.create_sheet("How to label")
    for line in [
        "For each clause (bold column) choose ONE obligation it is mainly about. Pick None if no rule fits.",
        "If you did not pick None, also choose a status: satisfied or not_satisfied. Leave status empty for None.",
        "satisfied = clear and specific (named officer, stated retention period, described way to withdraw).",
        "not_satisfied = about the rule but vague, missing a required part, or against the Act.",
        "The previous/next clauses are only for context. Label the bold clause.",
        "Use dpdp_obligation_rubric.xlsx, not gut feeling. Label on your own. If unsure, write a note and move on.",
        "Save as: <yourname>_<sheetname>.xlsx",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 120
    wb.save(path)
    return path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Create the labelling sheets.")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--per-policy", type=int, default=15)
    ap.add_argument("--blind-check", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="re-sample even though a manifest exists")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "sample_manifest.csv"
    if manifest.exists() and not args.force:
        sys.exit(f"{manifest} already exists, so the sample is locked. Use --force only if nobody has labelled yet.")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    by_policy = load_clauses(conn)
    if not by_policy:
        sys.exit("No clauses found for policies with a train/test split. Run the clause splitter first.")
    names = obligation_names(conn)

    rng = random.Random(args.seed)
    sample = pick_sample(by_policy, args.per_policy, args.min_words, args.max_words, rng)
    train = [c for c in sample if c["split"] == "train"]
    test = [c for c in sample if c["split"] == "test"]
    rng.shuffle(train)
    rng.shuffle(test)
    blind_ids = {c["clause_id"] for c in rng.sample(train, min(args.blind_check, len(train)))}

    # 1) batches for Claude (training sample)
    batches = [train[i:i + args.batch_size] for i in range(0, len(train), args.batch_size)]
    for n, batch in enumerate(batches, start=1):
        rows = []
        for c in batch:
            prev_c, next_c = with_context(c, by_policy)
            rows.append([c["clause_id"], prev_c, c["clause_text"], next_c, "", "", ""])
        write_csv(out / f"suggest_batch_{n:02d}.csv",
                  ["clause_id", "previous_clause", "clause_text", "next_clause",
                   "suggested_obligation", "suggested_status", "reason"], rows)
    batch_of = {c["clause_id"]: n for n, b in enumerate(batches, start=1) for c in b}

    # 2) blind check sheet (training clauses, no suggestions)
    blind_rows = []
    for c in train:
        if c["clause_id"] in blind_ids:
            prev_c, next_c = with_context(c, by_policy)
            blind_rows.append([c["clause_id"], prev_c, c["clause_text"], next_c, "", "", ""])
    blind_file = write_human_sheet(out / "train_blind_check.xlsx", blind_rows, names)

    # 3) test sheet (humans only)
    test_rows = []
    for c in test:
        prev_c, next_c = with_context(c, by_policy)
        test_rows.append([c["clause_id"], prev_c, c["clause_text"], next_c, "", "", ""])
    test_file = write_human_sheet(out / "test_for_humans.xlsx", test_rows, names)

    # 4) manifest
    write_csv(manifest, ["clause_id", "policy_id", "company", "split", "blind_check", "batch"],
              [[c["clause_id"], c["policy_id"], c["company_name"], c["split"],
                1 if c["clause_id"] in blind_ids else 0, batch_of.get(c["clause_id"], "")]
               for c in sorted(sample, key=lambda x: x["clause_id"])])

    print(f"Policies with clauses: {len(by_policy)}")
    print(f"Training sample: {len(train)} clauses in {len(batches)} batch file(s) (for Claude)")
    print(f"Blind check:     {len(blind_rows)} training clauses -> {blind_file.name} (humans only)")
    print(f"Test sample:     {len(test)} clauses -> {test_file.name} (humans only, never shown to Claude)")
    short = [pid for pid, cl in by_policy.items()
             if sum(1 for c in sample if c["policy_id"] == pid) < args.per_policy]
    if short:
        print(f"Note: {len(short)} policies had fewer than {args.per_policy} usable clauses.")
    print(f"Files are in {out}")
    conn.close()


if __name__ == "__main__":
    main()
