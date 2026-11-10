"""Copy Deepakshi's own test labels into a copy of the test sheet.

  python src/labeling/fill_test_sheet.py --in data/labels/test_for_humans.xlsx --out data/labels/test_Deepakshi.xlsx

This only transcribes labels a human decided. It checks the sheet against the list and refuses to
save if anything does not match. The input file is never changed.
"""
import argparse, sys
import openpyxl

NONE = [3787, 10754, 8781, 2527, 6939, 5377, 2882, 8623, 7087, 6127, 2557, 8598, 8795, 10614, 10627, 3782, 2512, 8584, 2798, 8853, 10650, 3744, 5630, 8519, 3812, 5681, 6129, 8834, 7085, 5687, 8648, 8626, 5648, 8488, 2875, 2860, 2516, 5278, 6159, 2839, 8879, 10821, 10640, 8686, 6158, 10521, 10531, 2519, 10587, 7006, 8522, 8534, 8688, 3783, 2862, 8493, 2813, 2507, 7099, 5631, 6106, 10598, 10747, 8621, 5633, 2535, 6921, 7125, 8602, 3742, 8496, 6289, 8873, 10569, 8561, 8687, 6110, 10530, 8489, 8871, 5611, 3743, 10797, 10529, 2505]

# (obligation, status) -> clause ids
LISTS = [
    ("None", "", NONE),
    ("Notice", "not_satisfied", [6294, 8828, 2533, 10840, 10812, 8791, 5350, 10844, 6275, 5665, 2901, 5714, 8605, 5669, 10813, 7045, 5205, 5201, 6094, 2539]),
    ("Consent", "not_satisfied", [8535, 3790, 10542, 5458, 2530, 2830, 8533, 8782, 3811, 8800, 3791, 10823, 5215, 5294, 6169, 8606, 8600]),
    ("Security Safeguards", "satisfied", [5486, 8557]),
    ("Security Safeguards", "not_satisfied", [10793, 8674, 5688, 2863, 10791, 3772, 2803, 3773, 2866, 5482, 10638, 10792]),
    ("Cross-Border Transfer", "not_satisfied", [6211, 8815, 8835, 2846]),
    ("Purpose Limitation", "satisfied", [6957, 7151, 10572, 6154, 3754, 2574, 5330, 5614, 2510, 5240, 10596, 5269, 2581]),
    ("Purpose Limitation", "not_satisfied", [6907, 7107, 3747, 7028, 8619, 2580, 2856]),
    ("Retention and Erasure", "satisfied", [5627, 10859]),
    ("Retention and Erasure", "not_satisfied", [3771, 5522, 7038, 8831, 10834, 8832]),
    ("Grievance Redressal", "not_satisfied", [7054, 5679, 8590, 8691, 6349, 8587, 2897, 5533]),
    ("Withdrawal of Consent", "not_satisfied", [8570, 5662, 6174, 10778]),
]


def die(m):
    print("ERROR: " + m); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.inp == a.out:
        die("--out must be a different file from --in")

    # 1. check the list itself
    seen, dup = {}, []
    for o, s, ids in LISTS:
        for i in ids:
            if i in seen:
                dup.append((i, seen[i], (o, s)))
            seen[i] = (o, s)
    print("Ids in your list: %d total, %d unique" % (sum(len(l[2]) for l in LISTS), len(seen)))
    if dup:
        print("These ids appear more than once in your list:")
        for i, first, second in dup:
            print("   %d : %s / %s   and   %s / %s" % (i, first[0], first[1] or "-", second[0], second[1] or "-"))
        die("Fix the duplicates first. Tell me which label you want for each id.")

    # 2. check the sheet against the list
    wb = openpyxl.load_workbook(a.inp)
    if "Label" not in wb.sheetnames:
        die("no sheet called Label. Sheets found: %s" % wb.sheetnames)
    ws = wb["Label"]
    head = {str(c.value).strip(): c.column for c in ws[1] if c.value is not None}
    for need in ("clause_id", "obligation", "status"):
        if need not in head:
            die("column '%s' not found. Columns: %s" % (need, list(head)))
    rows = {}
    for r in range(2, ws.max_row + 1):
        v = ws.cell(r, head["clause_id"]).value
        if v is not None and str(v).strip() != "":
            rows[int(float(v))] = r
    sheet_ids, list_ids = set(rows), set(seen)
    print("Clauses in the sheet: %d" % len(sheet_ids))
    miss, extra = sorted(sheet_ids - list_ids), sorted(list_ids - sheet_ids)
    if miss:
        print("In the sheet but NOT in your list (%d): %s" % (len(miss), miss))
    if extra:
        print("In your list but NOT in the sheet (%d): %s" % (len(extra), extra))
    if miss or extra:
        die("The list and the sheet do not match, so nothing was saved.")

    # 3. write
    changed = 0
    for cid, (o, s) in seen.items():
        r = rows[cid]
        old = (ws.cell(r, head["obligation"]).value, ws.cell(r, head["status"]).value)
        if (old[0] not in (None, "")) and (old[0] != o or (old[1] or "") != s):
            changed += 1
        ws.cell(r, head["obligation"]).value = o
        ws.cell(r, head["status"]).value = s if s else None
    wb.save(a.out)
    print("Saved %s. Rows that already had a different label and were overwritten: %d" % (a.out, changed))
    from collections import Counter
    for (o, s), n in sorted(Counter(seen.values()).items()):
        print("   %-24s %-14s %d" % (o, s or "-", n))


if __name__ == "__main__":
    main()
