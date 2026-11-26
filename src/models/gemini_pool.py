"""
Run Gemini on the Route B test set with a pool of API keys, ONE run per clause.

The script uses key 1 until Gemini says its quota is used up, then moves to key 2, then key 3, and so on.
When every key is used up it stops. Run it again the next day: it reads what is already answered and
carries on with the clauses that are still missing. Same prompt, parser and settings as gemini_runner.py
(put this file next to it in src/models/).

Keys go in a .env file in the repo root (never in code, never in git):
    GEMINI_KEY_1=...
    GEMINI_KEY_2=...        (any number of keys, GEMINI_KEY_1 .. GEMINI_KEY_30)

  python src/models/gemini_pool.py ping                 # tests every key with one tiny call
  python src/models/gemini_pool.py run                  # answer all clauses, rotating keys
  python src/models/gemini_pool.py status
  python src/models/gemini_pool.py merge                # score, once all clauses are answered
  Options: --test data/test.csv  --out runs/gemini/routeB  --model gemini-3.5-flash  --sleep 7
           --max-per-key N (rotate after N calls per key per run)  --mock  --mock-limit N

Free-tier daily limits reset at midnight Pacific time (about 1:30 pm in India in winter).
"""
import argparse, csv, hashlib, json, os, platform, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_runner import PROMPT, PROMPT_VERSION, parse, mock_reply   # same prompt and parser as before

FIELDS = ["clause_id", "key_slot", "obligation", "status", "reason", "valid", "error",
          "latency_ms", "input_tokens", "output_tokens", "raw"]
NAME_MAP = {"childrens data": "children's data"}


def die(m):
    print("ERROR: " + m); sys.exit(1)


def pt_date():
    """Today's date in Pacific time, because Google's daily quota resets at midnight there."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


def load_keys():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    keys = {}
    for i in range(1, 31):
        k = os.environ.get(f"GEMINI_KEY_{i}")
        if k:
            keys[i] = k
    return keys


def kind(e):
    """Sort an API error into: day (daily quota), rate (per-minute), badkey, other."""
    m = str(e).lower()
    quota = any(x in m for x in ("429", "resource_exhausted", "quota"))
    if quota and any(x in m for x in ("perday", "per day", "daily", "requestsperday")):
        return "day"
    if quota or "rate limit" in m or "rate_limit" in m:
        return "rate"
    if any(x in m for x in ("api key not valid", "api_key_invalid", "permission_denied", "unauthenticated", "status': 403", "status': 401", "code': 403", "code': 401")):
        return "badkey"
    if any(x in m for x in ("504", "503", "502", "500", "deadline_exceeded", "deadline expired", "unavailable",
                            "internal", "timed out", "timeout", "connection", "overloaded")):
        return "transient"
    return "other"


def make_real_caller(key, model, timeout):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")

    def call(prompt):
        t0 = time.time()
        r = client.models.generate_content(model=model, contents=prompt, config=cfg)
        ms = (time.time() - t0) * 1000
        u = getattr(r, "usage_metadata", None)
        out_tok = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        return (r.text or ""), ms, getattr(u, "prompt_token_count", None), out_tok
    return call


def make_mock_caller(limit):
    n = {"c": 0}

    def call(prompt):
        n["c"] += 1
        if limit and n["c"] > limit:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        return mock_reply(prompt.split("CLAUSE\n")[-1])
    return call


class Pool:
    def __init__(self, keys, out, model, timeout, mock, mock_limit, mock_keys):
        self.out = Path(out); self.state_path = self.out / "key_state.json"
        self.state = json.load(open(self.state_path)) if self.state_path.exists() else {}
        self.today = pt_date()
        slots = list(keys) if not mock else list(range(1, mock_keys + 1))
        self.callers = {s: (make_mock_caller(mock_limit) if mock else make_real_caller(keys[s], model, timeout)) for s in slots}
        self.slots = slots; self.cur = None; self.waited = False; self.tries = 0; self.used = {s: 0 for s in slots}
        self.pick()

    def dead(self, s):
        st = self.state.get(str(s))
        return bool(st) and st["date_pt"] == self.today

    def alive(self):
        return [s for s in self.slots if not self.dead(s)]

    def pick(self):
        a = self.alive()
        self.cur = a[0] if a else None
        self.waited = False

    def mark(self, s, reason):
        self.state[str(s)] = {"date_pt": self.today, "reason": reason}
        json.dump(self.state, open(self.state_path, "w"), indent=2)
        print(f"  key {s}: stopped for today ({reason}).")
        self.pick()


def check_meta(out, model):
    sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
    mp = Path(out) / "meta.json"
    if mp.exists():
        old = json.load(open(mp))
        if (old["model"], old["prompt_sha256_12"]) != (model, sha):
            die("this output folder was started with a different model or prompt. Keep them the same, or use a new --out.")
    else:
        json.dump({"model": model, "prompt_version": PROMPT_VERSION, "prompt_sha256_12": sha, "temperature": 0,
                   "runs_per_clause": 1, "first_started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "python": platform.python_version()}, open(mp, "w"), indent=2)


def read_test(path):
    return pd.read_csv(path, dtype={"clause_id": str}, keep_default_na=False, na_values=[""])


def done_ids(path):
    if not Path(path).exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["clause_id"] for r in csv.DictReader(f)}


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = "MOCK" if a.mock else a.model
    check_meta(out, model)
    df = read_test(a.test)
    path = out / "all_answers.csv"
    done = done_ids(path)
    order = np.random.default_rng(42).permutation(len(df))          # fixed random order: a partial run stays representative
    todo = [(df.loc[i, "clause_id"], df.loc[i, "clause_text"]) for i in order if df.loc[i, "clause_id"] not in done]
    keys = {} if a.mock else load_keys()
    if not a.mock and not keys:
        die("no keys found. Put GEMINI_KEY_1=..., GEMINI_KEY_2=... in the .env file.")
    pool = Pool(keys, out, model, a.timeout, a.mock, a.mock_limit, a.mock_keys)
    print(f"clauses: {len(df)} | already answered: {len(done)} | to do: {len(todo)}")
    print(f"keys: {sorted(pool.slots)} | usable today: {pool.alive()} (Pacific date {pool.today})")
    if not todo:
        print("Nothing left to do. Run merge."); return
    if not pool.alive():
        print("All keys are used up for today. Run again after midnight Pacific time (about 1:30 pm in India)."); return
    new = not path.exists(); n = 0; skips = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for cid, text in todo:
            prompt = PROMPT.replace("{clause_text}", text)
            res = None
            while pool.cur is not None:
                s = pool.cur
                if a.max_per_key and pool.used[s] >= a.max_per_key:
                    pool.mark(s, f"reached --max-per-key {a.max_per_key}"); continue
                try:
                    res = (s, pool.callers[s](prompt)); pool.waited = False; pool.tries = 0; pool.used[s] += 1; break
                except Exception as e:
                    k = kind(e)
                    if k == "day":
                        pool.mark(s, "daily quota used up")
                    elif k == "badkey":
                        print(f"  key {s} error: {str(e)[:200]}")
                        pool.mark(s, "key rejected (invalid or no permission)")
                    elif k == "transient":
                        pool.tries += 1
                        if pool.tries > a.retries:
                            print(f"  Gemini keeps timing out on {cid}; skipping it for now (next run will retry it).")
                            pool.tries = 0; res = "skip"; break
                        wait = 15
                        print(f"  key {s}: Gemini server timeout ({str(e)[:60]}); retry {pool.tries}/{a.retries} in {wait}s (key is NOT stopped)")
                        time.sleep(wait)
                    elif not pool.waited:
                        wait = 65 if k == "rate" else 20
                        print(f"  key {s}: {k} problem ({str(e)[:120]}); waiting {wait}s and trying once more")
                        time.sleep(wait); pool.waited = True
                    else:
                        pool.mark(s, "limit or error persisted" if k == "rate" else "errors persisted")
            if res is None:
                print("All keys are used up for today. Run the same command again after midnight Pacific time.")
                break
            if res == "skip":
                skips += 1
                if skips >= 3:
                    print("Gemini's servers are timing out repeatedly. Stop now and run again in an hour. No key was stopped.")
                    break
                continue
            skips = 0
            s, (raw, ms, tin, tout) = res
            o, st, r, ok, err = parse(raw)
            w.writerow({"clause_id": cid, "key_slot": s, "obligation": o, "status": st, "reason": r, "valid": ok,
                        "error": err, "latency_ms": ms, "input_tokens": tin, "output_tokens": tout, "raw": raw})
            f.flush(); n += 1
            print(f"[{len(done) + n}/{len(df)}] key {s} {cid} -> {o or 'INVALID'}")
            if not a.mock:
                time.sleep(a.sleep)
    print(f"this run: {n} new answers | total answered: {len(done) + n} / {len(df)}")
    if len(done) + n == len(df):
        print("All clauses answered. Run merge to score.")


def status(a):
    df = read_test(a.test); done = done_ids(Path(a.out) / "all_answers.csv")
    print(f"answered {len(done)} / {len(df)}")
    sp = Path(a.out) / "key_state.json"
    if sp.exists():
        print("key state:", json.load(open(sp)), "| Pacific date now:", pt_date())


def ping(a):
    keys = load_keys()
    if not keys:
        die("no keys found in .env")
    want = {int(x) for x in a.keys.split(",") if x.strip()}
    for s, k in keys.items():
        if want and s not in want:
            continue
        try:
            raw, ms, tin, tout = make_real_caller(k, a.model, a.timeout)('Return only this JSON: {"ok": true}')
            print(f"key {s}: works ({ms:.0f} ms) {raw!r}")
        except Exception as e:
            print(f"key {s}: FAILED [{kind(e)}] {str(e)[:160]}")


def macro_f1(y, p):
    return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)


def merge(a):
    g = pd.read_csv(Path(a.out) / "all_answers.csv", dtype={"clause_id": str}, keep_default_na=False, na_values=[""])
    ans = read_test(a.test)
    ans["true"] = ans["obligation"].astype(str).str.strip().str.lower().replace(NAME_MAP)
    g["pred"] = g["obligation"].astype(str).str.strip().str.lower().replace(NAME_MAP)
    g.loc[g["valid"] != 1, "pred"] = "invalid"
    m = ans.merge(g[["clause_id", "key_slot", "pred", "valid", "reason", "latency_ms", "input_tokens", "output_tokens"]],
                  on="clause_id", how="left")
    missing = m[m["pred"].isna()]["clause_id"].tolist()
    print(f"clauses: {len(ans)} | answered: {len(ans) - len(missing)} | missing: {len(missing)}")
    if missing and not a.allow_missing:
        die(f"{len(missing)} clauses are not answered yet. Finish the run, or use --allow-missing (not for the paper).")
    m = m[m["pred"].notna()]
    res = {"model": a.model, "n": int(len(m)), "invalid_answers": int((m["pred"] == "invalid").sum()),
           "macro_f1": float(macro_f1(m["true"], m["pred"])), "accuracy": float(accuracy_score(m["true"], m["pred"]))}
    if "policy_id" in m.columns:
        rng = np.random.default_rng(42); pol = m["policy_id"].unique(); by = {p: m[m["policy_id"] == p] for p in pol}
        bs = [macro_f1(*(lambda s: (s["true"], s["pred"]))(pd.concat([by[p] for p in rng.choice(pol, len(pol), replace=True)])))
              for _ in range(2000)]
        res["macro_f1_ci95_policy_bootstrap"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    if m["input_tokens"].notna().any():
        res["mean_latency_ms"] = float(m["latency_ms"].astype(float).mean())
        res["total_input_tokens"] = int(m["input_tokens"].fillna(0).sum())
        res["total_output_tokens"] = int(m["output_tokens"].fillna(0).sum())
    print(json.dumps(res, indent=2)); print(classification_report(m["true"], m["pred"], zero_division=0))
    m.to_csv(Path(a.out) / "gemini_routeB_scored.csv", index=False)
    json.dump(res, open(Path(a.out) / "gemini_routeB_results.json", "w"), indent=2)
    print("saved gemini_routeB_scored.csv and gemini_routeB_results.json in", a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "ping", "status", "merge"])
    p.add_argument("--test", default="data/test.csv", help="CSV with clause_id, clause_text, obligation (and policy_id)")
    p.add_argument("--out", default="runs/gemini/routeB")
    p.add_argument("--model", default="gemini-3.6-flash")
    p.add_argument("--sleep", type=float, default=7.0, help="seconds between calls (free-tier per-minute limits)")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--retries", type=int, default=1, help="extra tries after a server timeout (503/504) before skipping the clause")
    p.add_argument("--keys", default="", help="ping only: comma list of key numbers, e.g. 2,4")
    p.add_argument("--max-per-key", type=int, default=0, help="rotate after N calls per key in one run (0 = no cap)")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--mock", action="store_true", help="fake replies, no API (pipeline test only)")
    p.add_argument("--mock-limit", type=int, default=0, help="with --mock: each fake key runs out after N calls")
    p.add_argument("--mock-keys", type=int, default=3)
    a = p.parse_args()
    {"run": run, "ping": ping, "status": status, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()