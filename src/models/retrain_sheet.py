"""
Retrain the DPDP clause classifiers (linear SVM + Random Forest on TF-IDF)
from privacy_clauses_dataset.csv (or .xlsx).

Put this file in F:\\dpdp-classifier\\src\\models\\ and run from the repo root:

  # 1) Cross-validation only (never touches the locked test set)
  python src\\models\\retrain_sheet.py --sheet data\\privacy_clauses_dataset.csv

  # 2) Mix in old 'none' clauses (CSV columns: clause_text, obligation, policy_id)
  python src\\models\\retrain_sheet.py --sheet data\\privacy_clauses_dataset.csv ^
      --old-train data\\old_train.csv --old-none-n 150

  # 3) Final fit + score the locked test set (SECOND ROUND, do this once)
  python src\\models\\retrain_sheet.py --sheet data\\privacy_clauses_dataset.csv ^
      --old-train data\\old_train.csv --old-none-n 150 ^
      --test data\\test.csv --score-test

Synthetic clauses (clause_origin == Synthetic / ids starting SYN-) are used for TRAINING only.
Cross-validation always validates on REAL clauses, grouped by company. Use --no-synthetic to
train without them. Test CSV columns: clause_text, obligation (optional: status).
"""
import argparse, json, re, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
STATUS_OK = {"satisfied", "not_satisfied"}


def norm_text(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def read_any(p):
    p = Path(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, keep_default_na=False, na_values=[""])   # keeps 'n/a' as text
    return pd.read_excel(p, sheet_name=0, keep_default_na=False, na_values=[""])


def load_sheet(path, target, keep_testbook, use_synthetic):
    df = read_any(path).rename(columns={"clause": "clause_text"})
    cid = df["clause_id"].astype(str)
    origin = df["clause_origin"].astype(str).str.lower() if "clause_origin" in df.columns else ""
    df["is_synth"] = (origin == "synthetic") | cid.str.startswith("SYN-")
    df["company"] = None
    real = ~df["is_synth"]
    df.loc[real, "company"] = df.loc[real, "company_name"].ffill()   # name only on first row
    df.loc[df["is_synth"], "company"] = "Synthetic"
    bad = df[df["clause_text"].isna() | df["label"].isna()]
    if len(bad):
        print(f"dropping {len(bad)} rows with no clause text or no label: "
              f"{[str(x) for x in bad['clause_id'].tolist()]}  <- fix these in the sheet")
    df = df.dropna(subset=["clause_text", "label"])
    df["clause_text"] = df["clause_text"].map(norm_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower().replace(
        {"puprose limitation": "purpose limitation"})
    if "dpdp_status" in df.columns:
        df["dpdp_status"] = df["dpdp_status"].astype(str).str.strip().str.lower().replace(
            {"satiesfied": "satisfied"})
    if not keep_testbook:
        df = df[df["company"].astype(str).str.lower() != "testbook"]   # paraphrases, not verbatim
    if not use_synthetic:
        df = df[~df["is_synth"]]
    if target == "status":
        odd = df[~df["dpdp_status"].isin(STATUS_OK | {"n/a"})]
        if len(odd):
            print("unrecognised status values (rows dropped):", odd["dpdp_status"].unique().tolist())
        df = df[df["dpdp_status"].isin(STATUS_OK)]
        df["y"] = df["dpdp_status"]
    else:
        df["y"] = df["label"]
    n0 = len(df)
    df = df.drop_duplicates(subset="clause_text")
    df["group"] = df["company"].astype(str)
    print(f"rows used: {len(df)} ({int(df['is_synth'].sum())} synthetic); dropped {n0 - len(df)} exact duplicates")
    return df[["clause_text", "y", "group", "is_synth"]].reset_index(drop=True)


def load_old(path, n_none, target):
    old = pd.read_csv(path, keep_default_na=False, na_values=[""])
    old["clause_text"] = old["clause_text"].map(norm_text)
    col = "obligation" if target == "obligation" else "status"
    if col not in old.columns:
        raise SystemExit(f"old-train CSV needs a '{col}' column")
    old["y"] = old[col].astype(str).str.strip().str.lower()
    old["group"] = "old_" + (old["policy_id"].astype(str) if "policy_id" in old.columns else "x")
    old["is_synth"] = False
    if target == "obligation":
        none = old[old["y"] == "none"]
        keep = none.sample(n=min(n_none, len(none)), random_state=SEED)
        print(f"old training clauses mixed in: {len(keep)} 'none' clauses")
        return keep[["clause_text", "y", "group", "is_synth"]]
    return old[["clause_text", "y", "group", "is_synth"]]


def models():
    tf = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                                 max_features=8000, lowercase=True)
    return {
        "svm": make_pipeline(tf(), LinearSVC(C=0.3, class_weight="balanced")),
        "rf": make_pipeline(tf(), RandomForestClassifier(
            n_estimators=300, max_depth=30, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
    }


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)


def cross_validate(df, splits=5):
    """Validate on REAL clauses only (grouped by company); synthetic clauses join every training fold."""
    X, y, g = df["clause_text"].values, df["y"].values, df["group"].values
    syn_idx = np.where(df["is_synth"].values)[0]
    real_idx = np.where(~df["is_synth"].values)[0]
    gkf = GroupKFold(n_splits=min(splits, len(set(g[real_idx]))))
    res = {}
    for name in models():
        f1s, accs = [], []
        for tr, te in gkf.split(real_idx, y[real_idx], g[real_idx]):
            tr_idx = np.concatenate([real_idx[tr], syn_idx]); te_idx = real_idx[te]
            m = models()[name].fit(X[tr_idx], y[tr_idx])
            p = m.predict(X[te_idx])
            f1s.append(macro_f1(y[te_idx], p)); accs.append(accuracy_score(y[te_idx], p))
        res[name] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)),
                     "acc_mean": float(np.mean(accs))}
        print(f"CV (real clauses, grouped by company) {name}: macro-F1 {np.mean(f1s):.3f} "
              f"+/- {np.std(f1s):.3f}, acc {np.mean(accs):.3f}")
    return res


def drop_near_dups(train, test_texts, thr=0.9):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(
        list(train["clause_text"]) + list(test_texts))
    sim = cosine_similarity(vec.transform(train["clause_text"]), vec.transform(test_texts))
    leak = sim.max(axis=1) >= thr
    print(f"near-duplicates of test clauses removed from training: {int(leak.sum())}")
    return train[~leak]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--target", default="obligation", choices=["obligation", "status"])
    ap.add_argument("--old-train"); ap.add_argument("--old-none-n", type=int, default=150)
    ap.add_argument("--keep-testbook", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--test"); ap.add_argument("--score-test", action="store_true")
    ap.add_argument("--out", default="runs/retrain_sheet")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train = load_sheet(a.sheet, a.target, a.keep_testbook, not a.no_synthetic)
    if a.old_train:
        train = pd.concat([train, load_old(a.old_train, a.old_none_n, a.target)], ignore_index=True)
    tab = pd.crosstab(train["y"], np.where(train["is_synth"], "synthetic", "real"))
    print("\nclass counts (real vs synthetic):\n" + tab.to_string())
    few = tab[tab.get("real", 0) < 10] if "real" in tab.columns else tab
    if len(few):
        print("\nWARNING: fewer than 10 REAL examples (cross-validation and test scores for these "
              "cannot be trusted): " + ", ".join(f"{k}={v}" for k, v in few["real"].items()))

    print("\n--- cross-validation (never uses the test set) ---")
    cv = cross_validate(train)
    json.dump({"class_counts": tab.to_dict(), "cv": cv, "target": a.target},
              open(f"{a.out}/cv_{a.target}.json", "w"), indent=2)

    if a.test and a.score_test:
        test = pd.read_csv(a.test, keep_default_na=False, na_values=[""])
        test["clause_text"] = test["clause_text"].map(norm_text)
        col = "obligation" if a.target == "obligation" else "status"
        ytest = test[col].astype(str).str.strip().str.lower().values
        train2 = drop_near_dups(train, test["clause_text"].tolist())
        print("\n--- SECOND-ROUND scoring on the locked test set ---")
        print("Report these next to the first-round numbers; do not tune on them.")
        out = {}
        for name, m in models().items():
            m.fit(train2["clause_text"].values, train2["y"].values)
            p = m.predict(test["clause_text"].values)
            out[name] = {"macro_f1": float(macro_f1(ytest, p)), "acc": float(accuracy_score(ytest, p))}
            print(f"{name}: macro-F1 {out[name]['macro_f1']:.3f}, acc {out[name]['acc']:.3f}")
            print(classification_report(ytest, p, zero_division=0))
            joblib.dump(m, f"{a.out}/{name}_{a.target}_round2.joblib")
        json.dump(out, open(f"{a.out}/test_round2_{a.target}.json", "w"), indent=2)
    elif a.test:
        print("\n(test file given but --score-test not set, so the test set was NOT scored)")


if __name__ == "__main__":
    main()
