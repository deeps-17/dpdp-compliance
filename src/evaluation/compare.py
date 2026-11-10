"""Join classical-model predictions and Gemini answers, then make the result tables.

  python src/evaluation/compare.py --classical runs/classical --gemini runs/gemini/final --out runs/report
  python src/evaluation/compare.py --classical runs/dev --gemini runs/gemini/dev --out runs/report_dev   (practice)

Reads predictions_<target>.csv from the classical folder and run1.csv, run2.csv ... from the Gemini folder.
Writes tables as CSV and one markdown file. Every number comes from those files.
"""
import argparse, glob, os, sys, time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
TARGETS = ["obligation", "status", "joint"]
CLASSICAL = [("majority", "pred_majority"), ("keyword", "pred_keyword"), ("svm", "pred_svm"), ("random_forest", "pred_rf")]


def die(m):
    print("ERROR: " + m); sys.exit(1)


def mf1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def boot(y, p, pol, reps=2000):
    rng = np.random.default_rng(SEED)
    ids = np.unique(pol)
    idx = {i: np.where(pol == i)[0] for i in ids}
    v = []
    for _ in range(reps):
        ix = np.concatenate([idx[i] for i in rng.choice(ids, size=len(ids), replace=True)])
        v.append(mf1(y[ix], p[ix]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def label_of(target, o, s, valid):
    if str(valid) != "1":
        return "INVALID"
    return o if target == "obligation" else s if target == "status" else o + "|" + s


def vote(stack):
    """Majority answer across runs. If there is no majority, use run 1."""
    out = []
    for i in range(stack.shape[1]):
        label, n = Counter(stack[:, i]).most_common(1)[0]
        out.append(label if n > 1 or stack.shape[0] == 1 else stack[0, i])
    return np.array(out)


def row(system, y, p, pol):
    lo, hi = boot(y, p, pol)
    return {"system": system, "macro_f1": mf1(y, p), "ci_low": lo, "ci_high": hi, "accuracy": accuracy_score(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", required=True)
    ap.add_argument("--gemini", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gem = {}
    if a.gemini:
        for f in sorted(glob.glob(os.path.join(a.gemini, "run*.csv"))):
            g = pd.read_csv(f, keep_default_na=False, dtype=str)
            g["clause_id"] = g.clause_id.astype(int)
            gem[os.path.basename(f)[:-4]] = g.set_index("clause_id")
        if not gem:
            die("no run*.csv files in " + a.gemini)
    md = ["# Results (generated %s)\n" % time.strftime("%Y-%m-%d %H:%M"),
          "Classical folder: `%s`. Gemini folder: `%s`.\n" % (a.classical, a.gemini or "none")]
    found = False
    for t in TARGETS:
        pf = os.path.join(a.classical, "predictions_%s.csv" % t)
        if not os.path.exists(pf):
            continue
        found = True
        d = pd.read_csv(pf, keep_default_na=False, dtype=str)
        d["clause_id"] = d.clause_id.astype(int)
        y, pol = d.true.values, d.policy_id.values
        rows = [row(n, y, d[c].values, pol) for n, c in CLASSICAL]
        votes = None
        if gem:
            per_run = {}
            for rn, g in gem.items():
                absent = [c for c in d.clause_id if c not in g.index]
                if absent:
                    die("%s is missing %d clauses (e.g. %s). Finish the Gemini run first." % (rn, len(absent), absent[:3]))
                per_run[rn] = np.array([label_of(t, g.at[c, "obligation"], g.at[c, "status"], g.at[c, "valid"]) for c in d.clause_id])
                rows.append(row("gemini_" + rn, y, per_run[rn], pol))
            rows.append({"system": "gemini_mean_of_runs", "macro_f1": float(np.mean([mf1(y, p) for p in per_run.values()])),
                         "ci_low": np.nan, "ci_high": np.nan,
                         "accuracy": float(np.mean([accuracy_score(y, p) for p in per_run.values()]))})
            votes = vote(np.vstack(list(per_run.values())))
            rows.append(row("gemini_vote", y, votes, pol))
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(a.out, "table2_%s.csv" % t), index=False)
        md.append("## Table 2 (%s): macro-F1 and accuracy\n" % t)
        md.append(res.round(3).to_markdown(index=False) + "\n")
        print("\n== %s ==" % t)
        print(res.round(3).to_string(index=False))
        if t == "obligation":
            labels = sorted(set(y))
            pc = {"n_test": [int((y == l).sum()) for l in labels]}
            sysmap = {"svm": d.pred_svm.values, "random_forest": d.pred_rf.values}
            if votes is not None:
                sysmap["gemini_vote"] = votes
            for n, p in sysmap.items():
                pc[n] = f1_score(y, p, labels=labels, average=None, zero_division=0)
            tab = pd.DataFrame(pc, index=labels).round(3)
            tab.to_csv(os.path.join(a.out, "table3_perclass_obligation.csv"))
            md.append("## Table 3: F1 per obligation (test)\n")
            md.append(tab.to_markdown() + "\n")
            print("\nPer-class F1:\n" + tab.to_string())
    if not found:
        die("no predictions_*.csv in " + a.classical)
    if gem:
        t4 = []
        for rn, g in gem.items():
            tin = pd.to_numeric(g.input_tokens, errors="coerce")
            tout = pd.to_numeric(g.output_tokens, errors="coerce")
            lat = pd.to_numeric(g.latency_ms, errors="coerce")
            cost = (tin.sum() * a.price_in + tout.sum() * a.price_out) / 1e6
            t4.append({"run": rn, "clauses": len(g), "invalid": int((g.valid != "1").sum()),
                       "input_tokens": int(tin.sum()), "output_tokens": int(tout.sum()),
                       "cost_per_1000_clauses_usd": cost / len(g) * 1000, "mean_latency_ms": float(lat.mean())})
        t4 = pd.DataFrame(t4)
        ids = sorted(set.intersection(*[set(g.index) for g in gem.values()]))
        same = float(np.mean([len({(g.at[c, "obligation"], g.at[c, "status"]) for g in gem.values()}) == 1 for c in ids])) \
            if len(gem) > 1 else float("nan")
        t4.to_csv(os.path.join(a.out, "table4_cost_speed.csv"), index=False)
        md.append("## Table 4: Gemini cost, speed, consistency\n")
        md.append(t4.round(4).to_markdown(index=False) + "\n")
        md.append("Share of clauses where all runs gave the same obligation and status: %.3f\n" % same)
        md.append("Prices used (USD per 1M tokens): input %s, output %s. Classical models cost no API money.\n" % (a.price_in, a.price_out))
        print("\n" + t4.round(4).to_string(index=False))
        print("Same answer in all runs: %.3f" % same)
        if a.price_in == 0 and a.price_out == 0:
            print("NOTE: prices are 0. Pass --price-in and --price-out to get a cost.")
    open(os.path.join(a.out, "results.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\nSaved tables and results.md to " + a.out)


if __name__ == "__main__":
    main()
