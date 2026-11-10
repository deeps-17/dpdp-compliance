"""Make a practice copy of the database for building and testing the pipeline.

It copies db/dpdp.db to db/dev.db and, in the COPY only, moves a few labelled TRAINING
policies to the 'test' split. So the models can be trained, scored and compared end to end
without touching the real test set. The real database is never changed.

  python src/models/make_dev_db.py            (default: hold out 8 training policies)
  python src/models/make_dev_db.py --n 10
"""
import argparse, os, random, shutil, sqlite3, sys

SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="db/dpdp.db")
    p.add_argument("--dst", default="db/dev.db")
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        print("ERROR: source and destination are the same file"); sys.exit(1)
    if not os.path.exists(a.src):
        print("ERROR: not found: " + a.src); sys.exit(1)
    shutil.copyfile(a.src, a.dst)
    con = sqlite3.connect(a.dst)
    rows = con.execute("SELECT c.policy_id, COUNT(*) FROM clauses c JOIN policies p ON p.policy_id=c.policy_id "
                       "JOIN final_labels f ON f.clause_id=c.clause_id WHERE p.split='train' "
                       "GROUP BY c.policy_id ORDER BY c.policy_id").fetchall()
    if len(rows) <= a.n + 3:
        print("ERROR: only %d labelled training policies; cannot hold out %d" % (len(rows), a.n)); sys.exit(1)
    random.Random(SEED).shuffle(rows)
    held = sorted(r[0] for r in rows[:a.n])
    con.executemany("UPDATE policies SET split='test' WHERE policy_id=?", [(i,) for i in held])
    con.commit()
    print("Wrote %s (practice copy). Held out %d training policies as a practice test: %s" % (a.dst, a.n, held))
    for sp in ("train", "test"):
        n = con.execute("SELECT COUNT(*) FROM final_labels f JOIN clauses c ON c.clause_id=f.clause_id "
                        "JOIN policies p ON p.policy_id=c.policy_id WHERE p.split=?", (sp,)).fetchone()[0]
        print("  %s labelled clauses: %d" % (sp, n))
    print("Use it with: --db db/dev.db --out runs/dev")


if __name__ == "__main__":
    main()
