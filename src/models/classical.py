"""Classical models for DPDP clause classification: baselines, LinearSVC, Random Forest.

Commands (run from the project root, venv active):
  python src/models/classical.py cv    --target obligation   (tune with 5-fold GroupKFold on TRAIN only)
  python src/models/classical.py final --target obligation   (fit on all TRAIN, score the locked TEST once)

Targets: obligation (11 classes), status (satisfied / not_satisfied), joint (obligation|status).
The 'final' command refuses to run until data/labels/LOCKED exists, and refuses a second run.
"""
import argparse, itertools, json, os, platform, re, sqlite3, sys, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
OBL = {0: "None", 1: "Notice", 2: "Consent", 3: "Withdrawal of Consent",
       4: "Purpose Limitation", 5: "Security Safeguards", 6: "Breach Notification",
       7: "Retention and Erasure", 8: "Grievance Redressal", 9: "Childrens Data",
       10: "Cross-Border Transfer"}

# Hand-written from the rubric and the Act's wording. Not tuned on any data.
KEYWORDS = {
    "Notice": ["we collect", "information we collect", "personal data we", "data protection board", "your rights"],
    "Consent": ["consent", "agree to", "you agree", "by using", "accept"],
    "Withdrawal of Consent": ["withdraw", "opt out", "opt-out", "unsubscribe", "revoke"],
    "Purpose Limitation": ["purpose", "solely for", "only for", "only use", "used to"],
    "Security Safeguards": ["encrypt", "security", "secure", "firewall", "access control", "safeguard", "ssl"],
    "Breach Notification": ["breach", "unauthorised access", "unauthorized access", "notify you of"],
    "Retention and Erasure": ["retain", "retention", "delete", "erase", "erasure", "stored for", "as long as"],
    "Grievance Redressal": ["grievance", "data protection officer", "dpo", "complaint", "redressal"],
    "Childrens Data": ["child", "children", "minor", "under 18", "parent", "guardian"],
    "Cross-Border Transfer": ["outside india", "transfer", "cross-border", "other countries", "overseas", "abroad"],
}


def die(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def load(db, split):
    if not os.path.exists(db):
        die("database not found: " + db + " (use --db)")
    con = sqlite3.connect(db)
    q = ("SELECT c.clause_id, c.policy_id, c.clause_text, f.obligation_id, f.status "
         "FROM clauses c JOIN policies p ON p.policy_id = c.policy_id "
         "JOIN final_labels f ON f.clause_id = c.clause_id WHERE p.split = ?")
    df = pd.read_sql_query(q, con, params=(split,))
    con.close()
    if df.empty:
        die("no labelled clauses for split '%s'. Have final labels been loaded?" % split)
    df["obligation"] = df.obligation_id.map(OBL)
    df.loc[df.obligation == "None", "status"] = "not_applicable"
    df["joint"] = df.obligation + "|" + df.status
    return df


def target_frame(df, target):
    """Return rows and label column used for this target."""
    if target == "status":
        df = df[df.obligation != "None"]
    return df, ("obligation" if target == "obligation" else target)


def svm_pipe(C):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=2)),
                     ("clf", LinearSVC(C=C, class_weight="balanced", random_state=SEED, max_iter=20000))])


def rf_pipe(n, depth, leaf):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=2)),
                     ("clf", RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=leaf,
                                                    class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])


GRID_SVM = [{"C": c} for c in (0.1, 0.3, 1, 3, 10)]
GRID_RF = [{"n": 300, "depth": d, "leaf": l} for d in (None, 30) for l in (1, 2)]


def keyword_predict(texts, majority_status):
    out = []
    for t in texts:
        t = t.lower()
        scores = {o: sum(t.count(k) for k in ks) for o, ks in KEYWORDS.items()}
        best = max(scores, key=scores.get)
        out.append(best if scores[best] > 0 else "None")
    return out


def to_target_label(obl_pred, target, status_majority):
    if target == "obligation":
        return list(obl_pred)
    if target == "status":
        return [status_majority] * len(obl_pred)
    return [o + "|" + ("not_applicable" if o == "None" else status_majority) for o in obl_pred]


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def cmd_cv(a):
    df, col = target_frame(load(a.db, "train"), a.target)
    y, X, g = df[col].values, df.clause_text.values, df.policy_id.values
    n_pol = len(set(g))
    k = min(5, n_pol)
    if k < 3:
        die("need at least 3 training policies for cross-validation")
    folds = list(GroupKFold(n_splits=k).split(X, y, g))
    rows = []
    cands = [("svm", p, svm_pipe(p["C"])) for p in GRID_SVM] + \
            [("rf", p, rf_pipe(p["n"], p["depth"], p["leaf"])) for p in GRID_RF]
    for name, params, pipe in cands:
        sc = []
        for tr, va in folds:
            pipe.fit(X[tr], y[tr])
            sc.append(macro_f1(y[va], pipe.predict(X[va])))
        rows.append({"model": name, "params": json.dumps(params), "cv_macro_f1_mean": np.mean(sc),
                     "cv_macro_f1_std": np.std(sc)})
        print("%-4s %-40s macro-F1 %.3f +- %.3f" % (name, json.dumps(params), np.mean(sc), np.std(sc)))
    res = pd.DataFrame(rows)
    # baselines under the same folds
    maj = []
    kw = []
    for tr, va in folds:
        m = pd.Series(y[tr]).mode()[0]
        maj.append(macro_f1(y[va], [m] * len(va)))
        sm = pd.Series(df.status.values[tr]).mode()[0]
        kw.append(macro_f1(y[va], to_target_label(keyword_predict(X[va], sm), a.target, sm)))
    print("majority baseline macro-F1 %.3f   keyword baseline %.3f" % (np.mean(maj), np.mean(kw)))
    best = {}
    for name in ("svm", "rf"):
        r = res[res.model == name].sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
        best[name] = json.loads(r.params)
    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    res.to_csv(os.path.join(a.out, "cv_%s_%s.csv" % (a.target, stamp)), index=False)
    json.dump({"target": a.target, "best_params": best, "folds": k, "train_policies": n_pol,
               "train_clauses": int(len(df)), "seed": SEED, "sklearn": sklearn.__version__,
               "python": platform.python_version()},
              open(os.path.join(a.out, "best_params_%s.json" % a.target), "w"), indent=2)
    print("Saved tuning results and best params to " + a.out)
    small = [c for c, n in pd.Series(y).value_counts().items() if n < 10]
    if small:
        print("WARNING: classes with fewer than 10 training clauses: %s" % small)


def boot_ci(y, p, pol, reps=2000):
    rng = np.random.default_rng(SEED)
    ids = np.unique(pol)
    idx = {i: np.where(pol == i)[0] for i in ids}
    vals = []
    for _ in range(reps):
        pick = rng.choice(ids, size=len(ids), replace=True)
        ix = np.concatenate([idx[i] for i in pick])
        vals.append(macro_f1(y[ix], p[ix]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cmd_final(a):
    if a.dev:
        if os.path.basename(a.db).lower() == "dpdp.db":
            die("--dev is only for a dev copy of the database (made by make_dev_db.py), never db/dpdp.db")
        print("DEV RUN on a practice split. These numbers are NOT results for the paper.")
    elif not os.path.exists(a.lock):
        die("%s does not exist. Freeze the labels first, then create it (see instructions)." % a.lock)
    done = os.path.join(a.out, "final_%s.json" % a.target)
    if os.path.exists(done) and not a.force:
        die(done + " exists. The test set is scored once. Add --force only if you must redo it and will say so in the paper.")
    bp = os.path.join(a.out, "best_params_%s.json" % a.target)
    if not os.path.exists(bp):
        die("run the cv command first: " + bp + " is missing")
    best = json.load(open(bp))["best_params"]
    tr, col = target_frame(load(a.db, "train"), a.target)
    te_all = load(a.db, "test")
    te, _ = target_frame(te_all, a.target)
    ytr, yte = tr[col].values, te[col].values
    preds = {}
    s = best["svm"]
    m = svm_pipe(s["C"]).fit(tr.clause_text.values, ytr)
    preds["svm"] = m.predict(te.clause_text.values)
    r = best["rf"]
    m = rf_pipe(r["n"], r["depth"], r["leaf"]).fit(tr.clause_text.values, ytr)
    preds["rf"] = m.predict(te.clause_text.values)
    sm = tr.status.mode()[0]
    preds["majority"] = np.array([pd.Series(ytr).mode()[0]] * len(te))
    preds["keyword"] = np.array(to_target_label(keyword_predict(te.clause_text.values, sm), a.target, sm))
    metrics, labels = {}, sorted(set(ytr) | set(yte))
    os.makedirs(a.out, exist_ok=True)
    pol = te.policy_id.values
    for name, p in preds.items():
        lo, hi = boot_ci(yte, p, pol)
        metrics[name] = {"macro_f1": macro_f1(yte, p), "ci95": [lo, hi],
                         "accuracy": accuracy_score(yte, p)}
        rep = pd.DataFrame(classification_report(yte, p, labels=labels, zero_division=0, output_dict=True)).T
        rep.to_csv(os.path.join(a.out, "perclass_%s_%s.csv" % (a.target, name)))
        cm = pd.DataFrame(confusion_matrix(yte, p, labels=labels), index=labels, columns=labels)
        cm.to_csv(os.path.join(a.out, "confusion_%s_%s.csv" % (a.target, name)))
        print("%-9s macro-F1 %.3f  (95%% CI %.3f to %.3f)  accuracy %.3f" %
              (name, metrics[name]["macro_f1"], lo, hi, metrics[name]["accuracy"]))
    out = te[["clause_id", "policy_id"]].copy()
    out["true"] = yte
    for name, p in preds.items():
        out["pred_" + name] = p
    out.to_csv(os.path.join(a.out, "predictions_%s.csv" % a.target), index=False)
    json.dump({"target": a.target, "metrics": metrics, "best_params": best, "test_clauses": int(len(te)),
               "test_policies": int(len(set(pol))), "seed": SEED, "sklearn": sklearn.__version__,
               "date": time.strftime("%Y-%m-%d %H:%M:%S")}, open(done, "w"), indent=2)
    print("Saved to " + a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="db/dpdp.db")
    p.add_argument("--out", default="runs/classical")
    p.add_argument("--lock", default="data/labels/LOCKED")
    sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("cv", "final"):
        s = sub.add_parser(n)
        s.add_argument("--target", choices=["obligation", "status", "joint"], default="obligation")
        if n == "final":
            s.add_argument("--force", action="store_true")
            s.add_argument("--dev", action="store_true", help="practice run on a dev copy of the database")
    a = p.parse_args()
    {"cv": cmd_cv, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
