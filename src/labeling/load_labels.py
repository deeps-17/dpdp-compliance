"""Load reviewed labels into SQLite, compute agreement, list disagreements.

Commands (run from the project root, venv active):
  python src/labeling/load_labels.py train  --dir data/labels/reviewed
  python src/labeling/load_labels.py blind  --file data/labels/blind_done.xlsx
  python src/labeling/load_labels.py test   --a data/labels/test_A.xlsx --b data/labels/test_B.xlsx
  python src/labeling/load_labels.py adjudicate --file data/labels/disagreements.csv
  python src/labeling/load_labels.py report
"""
import argparse, csv, glob, os, sqlite3, sys
from collections import Counter

DB = "db/dpdp.db"
OBL = {0: "None", 1: "Notice", 2: "Consent", 3: "Withdrawal of Consent",
       4: "Purpose Limitation", 5: "Security Safeguards", 6: "Breach Notification",
       7: "Retention and Erasure", 8: "Grievance Redressal", 9: "Childrens Data",
       10: "Cross-Border Transfer"}


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


NAME2ID = {norm(v): k for k, v in OBL.items()}
NAME2ID[norm("Children's Data")] = 9
NAME2ID[norm("Cross Border Transfer")] = 10
STATUSES = {"satisfied", "not_satisfied"}


def die(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def connect(path):
    if not os.path.exists(path):
        die("database not found: " + path + " (use --db to point to it)")
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "nat") else s


def read_table(path):
    """Return list of dicts from csv or xlsx."""
    if not os.path.exists(path):
        die("file not found: " + path)
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if path.lower().endswith(".xlsx"):
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        head = [clean(h) for h in rows[0]]
        return [dict(zip(head, r)) for r in rows[1:]]
    die("unsupported file type: " + path)


def find_col(rows, path, must, avoid=()):
    cols = list(rows[0].keys()) if rows else []
    hits = [c for c in cols if must in norm(c) and not any(a in norm(c) for a in avoid)]
    if len(hits) != 1:
        die("in %s I could not find exactly one column containing '%s'. Columns found: %s"
            % (path, must, cols))
    return hits[0]


def parse_label(obl, st, where):
    """Return (obligation_id, status) in DB form. Exits on bad values."""
    obl, st = clean(obl), clean(st)
    if norm(obl) not in NAME2ID:
        die("%s: unknown obligation '%s'" % (where, obl))
    oid = NAME2ID[norm(obl)]
    if oid == 0:
        return 0, "not_applicable"
    st = st.lower().replace(" ", "_")
    if st not in STATUSES:
        die("%s: status must be satisfied or not_satisfied for '%s', got '%s'" % (where, obl, st))
    return oid, st


def put_annotation(con, cid, who, oid, st):
    con.execute("INSERT OR REPLACE INTO annotations (clause_id, annotator, obligation_id, status) "
                "VALUES (?,?,?,?)", (cid, who, oid, st))


def put_final(con, cid, oid, st, method):
    con.execute("INSERT OR REPLACE INTO final_labels (clause_id, obligation_id, status, method) "
                "VALUES (?,?,?,?)", (cid, oid, st, method))


def check_clause_ids(con, ids, where):
    have = {r[0] for r in con.execute("SELECT clause_id FROM clauses")}
    missing = [i for i in ids if i not in have]
    if missing:
        die("%s: %d clause_id values are not in the clauses table, e.g. %s" % (where, len(missing), missing[:5]))


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cmd_train(a):
    files = sorted(glob.glob(os.path.join(a.dir, "suggest_batch_*.csv")))
    if not files:
        die("no suggest_batch_*.csv files in " + a.dir)
    con = connect(a.db)
    total = changed = 0
    for p in files:
        rows = read_table(p)
        for need in ("clause_id", "suggested_obligation", "suggested_status",
                     "reviewer_obligation", "reviewer_status"):
            if rows and need not in rows[0]:
                die("%s is missing column '%s'. Columns found: %s" % (p, need, list(rows[0].keys())))
        check_clause_ids(con, [int(r["clause_id"]) for r in rows], p)
        for r in rows:
            cid = int(r["clause_id"])
            where = "%s clause %d" % (os.path.basename(p), cid)
            s_o, s_s = parse_label(r["suggested_obligation"], r["suggested_status"], where + " (suggestion)")
            ro, rs = clean(r["reviewer_obligation"]), clean(r["reviewer_status"])
            if ro == "" and rs == "":
                f_o, f_s, method = s_o, s_s, "agreed"
            else:
                if ro == "":
                    ro = OBL[s_o]
                if rs == "" and norm(ro) != norm("None"):
                    rs = s_s if s_s != "not_applicable" else ""
                f_o, f_s = parse_label(ro, rs, where + " (reviewer)")
                method = "agreed" if (f_o, f_s) == (s_o, s_s) else "adjudicated"
            put_annotation(con, cid, "llm_suggest", s_o, s_s)
            put_annotation(con, cid, "team_review", f_o, f_s)
            put_final(con, cid, f_o, f_s, method)
            total += 1
            changed += method == "adjudicated"
    con.commit()
    print("Loaded %d training clauses from %d files." % (total, len(files)))
    print("Reviewer changed the suggestion on %d clauses (%.1f%%)." % (changed, 100.0 * changed / total))
    print("Rows with both columns blank were counted as 'agreed'. Check no one skipped rows by accident.")


def read_labels(path):
    rows = read_table(path)
    if not rows:
        die(path + " is empty")
    ccol = find_col(rows, path, "clauseid")
    ocol = find_col(rows, path, "obligation", avoid=("suggest",))
    scol = find_col(rows, path, "status", avoid=("suggest",))
    out = {}
    for r in rows:
        if clean(r[ocol]) == "":
            continue
        cid = int(float(clean(r[ccol])))
        out[cid] = parse_label(r[ocol], r[scol], "%s clause %d" % (os.path.basename(path), cid))
    return out


def cmd_blind(a):
    con = connect(a.db)
    lab = read_labels(a.file)
    check_clause_ids(con, list(lab), a.file)
    for cid, (o, s) in lab.items():
        put_annotation(con, cid, "blind_check", o, s)
    con.commit()
    sug = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT clause_id, obligation_id, status FROM annotations WHERE annotator='llm_suggest'")}
    both = [cid for cid in lab if cid in sug]
    if len(both) < len(lab):
        print("WARNING: %d blind clauses have no llm_suggest row. Run the train command first." % (len(lab) - len(both)))
    full = [(lab[c], sug[c]) for c in both]
    only = [(lab[c][0], sug[c][0]) for c in both]
    print("Blind-check clauses compared: %d" % len(both))
    print("Exact agreement (obligation + status): %.1f%%" % (100.0 * sum(x == y for x, y in full) / max(1, len(full))))
    print("Agreement on obligation only:         %.1f%%" % (100.0 * sum(x == y for x, y in only) / max(1, len(only))))
    print("Cohen kappa, obligation only:         %.3f" % kappa(only))
    print("Cohen kappa, obligation + status:     %.3f" % kappa([("%d|%s" % x, "%d|%s" % y) for x, y in full]))


def cmd_test(a):
    out = os.path.join(os.path.dirname(a.a) or ".", "disagreements.csv")
    if os.path.exists(out) and not a.force:
        die(out + " already exists. Running test again would overwrite your adjudication work. "
            "Move it away or add --force if you really want to redo it.")
    con = connect(a.db)
    A, B = read_labels(a.a), read_labels(a.b)
    ids = sorted(set(A) | set(B))
    check_clause_ids(con, ids, "test files")
    split_bad = con.execute("SELECT COUNT(*) FROM clauses c JOIN policies p ON p.policy_id=c.policy_id "
                            "WHERE c.clause_id IN (%s) AND p.split != 'test'" % ",".join(map(str, ids))).fetchone()[0]
    if split_bad:
        die("%d clauses in the test files belong to TRAIN policies. Wrong file?" % split_bad)
    miss = [c for c in ids if c not in A or c not in B]
    if miss:
        print("WARNING: %d clauses are labelled by only one person (first: %s). They are skipped." % (len(miss), miss[:5]))
    common = [c for c in ids if c in A and c in B]
    dis = []
    for c in common:
        put_annotation(con, c, "test_A", *A[c])
        put_annotation(con, c, "test_B", *B[c])
        if A[c] == B[c]:
            put_final(con, c, A[c][0], A[c][1], "agreed")
        else:
            dis.append(c)
    con.commit()
    only = [(A[c][0], B[c][0]) for c in common]
    full = [("%d|%s" % A[c], "%d|%s" % B[c]) for c in common]
    print("Test clauses labelled by both: %d" % len(common))
    print("Exact agreement: %.1f%%" % (100.0 * (len(common) - len(dis)) / max(1, len(common))))
    print("Cohen kappa, obligation only:     %.3f   (target 0.70 or higher)" % kappa(only))
    print("Cohen kappa, obligation + status: %.3f" % kappa(full))
    texts = {r[0]: r[1] for r in con.execute("SELECT clause_id, clause_text FROM clauses")}
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clause_id", "clause_text", "A_obligation", "A_status", "B_obligation", "B_status",
                    "adjudicated_obligation", "adjudicated_status"])
        for c in dis:
            w.writerow([c, texts[c], OBL[A[c][0]], A[c][1], OBL[B[c][0]], B[c][1], "", ""])
    print("%d disagreements written to %s. Final labels for them are NOT set yet." % (len(dis), out))


def cmd_test_single(a):
    """One person labelled the whole test set. No agreement can be measured."""
    con = connect(a.db)
    lab = read_labels(a.file)
    ids = sorted(lab)
    check_clause_ids(con, ids, a.file)
    bad = con.execute("SELECT COUNT(*) FROM clauses c JOIN policies p ON p.policy_id=c.policy_id "
                      "WHERE c.clause_id IN (%s) AND p.split != 'test'" % ",".join(map(str, ids))).fetchone()[0]
    if bad:
        die("%d clauses in this file belong to TRAIN policies. Wrong file?" % bad)
    for cid in ids:
        put_annotation(con, cid, "test_single", *lab[cid])
        put_final(con, cid, lab[cid][0], lab[cid][1], "agreed")
    con.commit()
    print("Loaded %d test clauses labelled by one person (annotator test_single)." % len(ids))
    print("No second annotator, so NO agreement (kappa) can be reported. Say this in the paper.")
    c = Counter(OBL[lab[i][0]] for i in ids)
    for k, n in sorted(c.items(), key=lambda x: -x[1]):
        print("   %-24s %d" % (k, n))


def cmd_adjudicate(a):
    con = connect(a.db)
    rows = read_table(a.file)
    n = 0
    for r in rows:
        if clean(r.get("adjudicated_obligation")) == "":
            continue
        cid = int(r["clause_id"])
        o, s = parse_label(r["adjudicated_obligation"], r.get("adjudicated_status"), "clause %d" % cid)
        put_final(con, cid, o, s, "adjudicated")
        n += 1
    con.commit()
    left = len(rows) - n
    print("Adjudicated %d clauses. Still empty: %d" % (n, left))


def cmd_report(a):
    con = connect(a.db)
    print("Final labels per obligation, by split (train / test):")
    q = ("SELECT f.obligation_id, p.split, COUNT(*) FROM final_labels f "
         "JOIN clauses c ON c.clause_id=f.clause_id JOIN policies p ON p.policy_id=c.policy_id "
         "GROUP BY 1,2")
    t = {}
    for oid, sp, n in con.execute(q):
        t.setdefault(oid, {})[sp] = n
    for oid in sorted(OBL):
        d = t.get(oid, {})
        tr = d.get("train", 0)
        flag = "  <-- under 40 training examples" if oid != 0 and tr < 40 else ""
        print("  %2d %-22s train %4d   test %4d%s" % (oid, OBL[oid], tr, d.get("test", 0), flag))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("train"); s.add_argument("--dir", default="data/labels/reviewed")
    s = sub.add_parser("blind"); s.add_argument("--file", required=True)
    s = sub.add_parser("test"); s.add_argument("--a", required=True); s.add_argument("--b", required=True); s.add_argument("--force", action="store_true")
    s = sub.add_parser("adjudicate"); s.add_argument("--file", required=True)
    s = sub.add_parser("test-single"); s.add_argument("--file", required=True)
    sub.add_parser("report")
    a = p.parse_args()
    {"train": cmd_train, "blind": cmd_blind, "test": cmd_test,
     "adjudicate": cmd_adjudicate, "report": cmd_report, "test-single": cmd_test_single}[a.cmd](a)


if __name__ == "__main__":
    main()