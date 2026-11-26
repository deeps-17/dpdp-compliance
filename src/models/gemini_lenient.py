"""
Strict vs lenient scoring of the saved Gemini answers (Route B test set).
Makes NO API calls. Reads only files already on disk. Prompt and answers are untouched.

Strict  : answer counts only if the pool script marked it valid (exact label match).
Lenient : label is normalised before matching: lowercase, apostrophes removed,
          a trailing "(s5)"-style suffix stripped. Anything still unmatched stays invalid.

Run from the repo root:
    python src\\models\\gemini_lenient.py
    python src\\models\\gemini_lenient.py --answers runs\\gemini\\routeB\\all_answers.csv --test data\\test.csv
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\u2019\u2018'`]", "", s)               # straight/curly apostrophes
    s = re.sub(r"\s*\(\s*s\d+\s*\)\s*$", "", s)         # trailing "(s5)"
    return re.sub(r"\s+", " ", s).strip()


def raw_label(row):
    """The label Gemini actually wrote, from the saved raw reply (fallback: obligation column)."""
    raw = row.get("raw")
    if isinstance(raw, str) and raw.strip():
        try:
            d = json.loads(raw)
            if isinstance(d, dict) and d.get("obligation") is not None:
                return str(d["obligation"])
        except Exception:
            m = re.search(r'"obligation"\s*:\s*"([^"]*)"', raw)
            if m:
                return m.group(1)
    ob = row.get("gem_obligation")
    return "" if pd.isna(ob) else str(ob)


def macro_f1(y_true, y_pred, drop_breach=False):
    classes = sorted(set(y_true))
    if drop_breach:
        classes = [c for c in classes if not c.startswith("breach")]
    return f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)


def bootstrap_ci(df, col, n_boot=2000, seed=42, drop_breach=False):
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in df.groupby("policy_id")]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(macro_f1(df.loc[idx, "true"].tolist(), df.loc[idx, col].tolist(), drop_breach))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarise(df, col):
    y, p = df["true"].tolist(), df[col].tolist()
    return {
        "macro_f1": float(macro_f1(y, p)),
        "macro_f1_ci95_policy_bootstrap": bootstrap_ci(df, col),
        "macro_f1_without_breach": float(macro_f1(y, p, drop_breach=True)),
        "accuracy": float(accuracy_score(y, p)),
        "invalid_answers": int((df[col] == "invalid").sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(Path("runs")/"gemini"/"routeB"/"all_answers.csv"))
    ap.add_argument("--test", default=str(Path("data")/"test.csv"))
    ap.add_argument("--out", default=str(Path("runs")/"gemini"/"routeB"))
    a = ap.parse_args()

    ans = pd.read_csv(a.answers, dtype={"clause_id": str})
    test = pd.read_csv(a.test, dtype={"clause_id": str, "policy_id": str})
    ans = ans.drop_duplicates("clause_id", keep="last").rename(columns={"obligation": "gem_obligation"})

    df = test[["clause_id", "policy_id", "obligation"]].rename(columns={"obligation": "true"}).merge(
        ans, on="clause_id", how="left")
    missing = df["raw"].isna() & df["gem_obligation"].isna()
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} test clauses have no Gemini answer. Finish the run first.")
    df["true"] = df["true"].astype(str).str.strip().str.lower()
    df["gemini_label"] = df.apply(raw_label, axis=1)

    # strict: pool script's own validity flag, exact (lowercased) match
    valid = df["valid"].astype(float).fillna(0).astype(int) == 1
    df["strict_pred"] = np.where(valid, df["gemini_label"].str.strip().str.lower(), "invalid")

    # lenient: normalised match against the test label set
    label_map = {norm(t): t for t in df["true"].unique()}
    df["lenient_pred"] = df["gemini_label"].map(lambda s: label_map.get(norm(s), "invalid"))
    df["rescued"] = (df["strict_pred"] == "invalid") & (df["lenient_pred"] != "invalid")

    res = {
        "model": str(ans["model"].iloc[0]) if "model" in ans else "gemini-3.6-flash",
        "n": int(len(df)),
        "strict": summarise(df, "strict_pred"),
        "lenient": summarise(df, "lenient_pred"),
        "rescued_by_lenient_rule": int(df["rescued"].sum()),
    }

    print(json.dumps(res, indent=2))
    print("\nRescued by the lenient rule (what Gemini wrote -> counted as):")
    r = df[df["rescued"]]
    print(r.groupby(["gemini_label", "lenient_pred"]).size().to_string() if len(r) else "  none")
    still = df[df["lenient_pred"] == "invalid"]
    print("\nStill invalid after the lenient rule:")
    print(still[["clause_id", "gemini_label"]].to_string(index=False) if len(still) else "  none")
    print("\nLenient per-class report:")
    print(classification_report(df["true"], df["lenient_pred"], zero_division=0, digits=2))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df[["clause_id", "policy_id", "true", "gemini_label", "strict_pred", "lenient_pred", "rescued"]].to_csv(
        out / "gemini_routeB_lenient.csv", index=False)
    (out / "gemini_routeB_strict_vs_lenient.json").write_text(json.dumps(res, indent=2))
    print(f"saved gemini_routeB_lenient.csv and gemini_routeB_strict_vs_lenient.json in {out}")


if __name__ == "__main__":
    main()
