"""Run Gemini on clauses with the fixed prompt. One CSV per run, plus a meta.json.

  python src/models/gemini_runner.py --db db/dev.db --dev --out runs/gemini/dev --limit 20
  python src/models/gemini_runner.py --db db/dpdp.db --out runs/gemini/final     (needs data/labels/LOCKED)

The model name comes from --model or config.yaml (gemini.model_name). The API key comes from
the environment variable GEMINI_API_KEY (a .env file is read if python-dotenv is installed).
--mock answers with a fake rule-based reply so you can test the pipeline without the API.
"""
import argparse, csv, hashlib, json, os, platform, re, sqlite3, sys, time

PROMPT_VERSION = "v1"
OBLIGATIONS = ["Notice", "Consent", "Withdrawal of Consent", "Purpose Limitation", "Security Safeguards",
               "Breach Notification", "Retention and Erasure", "Grievance Redressal", "Childrens Data",
               "Cross-Border Transfer", "None"]
PROMPT = """You are labelling clauses from Indian company privacy policies against India's Digital Personal Data Protection (DPDP) Act, 2023.

TASK
Read ONE clause. Return the single obligation it mainly addresses, and whether the clause satisfies that obligation as written.

OBLIGATIONS (use these exact names)
1. Notice (s5)
   Satisfied if the clause states what personal data is collected, the purpose, how to exercise rights, and how to complain to the Data Protection Board.
   Not satisfied if it misses the purpose, the rights, or the complaint route.
2. Consent (s6(1)-(3))
   Satisfied if it describes consent as free, specific, informed, unconditional, given by clear affirmative action, limited to what is necessary, in plain language.
   Not satisfied if it uses pre-ticked boxes, bundled or unrelated permissions, or vague wording like "by using this site you agree".
3. Withdrawal of Consent (s6(4)-(6))
   Satisfied if it describes a withdrawal method roughly as easy as giving consent, and what happens after withdrawal.
   Not satisfied if no withdrawal method is mentioned, or it is buried or unclear.
4. Purpose Limitation (s4, s7)
   Satisfied if it states specific purpose(s) and limits processing to them.
   Not satisfied if it uses open-ended language like "and for other purposes we deem fit".
5. Security Safeguards (s8(5))
   Satisfied if it describes concrete technical or organizational measures (for example encryption, access control).
   Not satisfied if it is generic, like "we take your security seriously".
6. Breach Notification (s8(6))
   Satisfied if it states it will notify the Board and affected users of a breach.
   Not satisfied if there is no breach notification commitment.
7. Retention and Erasure (s8(7)-(8))
   Satisfied if it states a specific retention period or deletion trigger.
   Not satisfied if it says "as long as necessary" with no period or trigger.
8. Grievance Redressal (s8(9)-(10), s13)
   Satisfied if it names a Grievance Officer or DPO with contact details and a response timeframe.
   Not satisfied if it is a generic "contact us" with no named officer or timeframe.
9. Childrens Data (s9)
   Satisfied if it states verifiable parental or guardian consent for under-18 users, and no targeted ads or tracking of children.
   Not satisfied if minors are not mentioned, or there is no verifiable consent process.
10. Cross-Border Transfer (s16)
   Satisfied if it says data may be transferred outside India and on what basis.
   Not satisfied if transfers are mentioned with no basis, or not disclosed.
0. None
   Use for boilerplate, definitions, jurisdiction, general cookie descriptions, headings, fragments, or anything no obligation fits.

RULES
1. Judge only the text of the clause. Do not assume anything that is not written in it.
2. Choose exactly one obligation. If two fit, choose the one the clause is mainly about.
3. Status is "satisfied" only if the clause meets that obligation's "Satisfied if" test. Vague, generic or incomplete wording is "not_satisfied".
4. If the obligation is None, the status must be "not_applicable".
5. You are describing what the clause discloses, not whether the company obeys the law.

OUTPUT
Return JSON only, with no other text:
{"obligation": "<one of: Notice, Consent, Withdrawal of Consent, Purpose Limitation, Security Safeguards, Breach Notification, Retention and Erasure, Grievance Redressal, Childrens Data, Cross-Border Transfer, None>", "status": "<satisfied | not_satisfied | not_applicable>", "reason": "<under 20 words>"}

CLAUSE
{clause_text}
"""
FIELDS = ["clause_id", "run", "obligation", "status", "reason", "valid", "error", "latency_ms",
          "input_tokens", "output_tokens", "raw"]


def die(m):
    print("ERROR: " + m); sys.exit(1)


def parse(text):
    """Return (obligation, status, reason, valid, error)."""
    try:
        t = text.strip()
        t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
        d = json.loads(t)
        o, s, r = str(d.get("obligation", "")).strip(), str(d.get("status", "")).strip(), str(d.get("reason", ""))
    except Exception as e:
        return "", "", "", 0, "bad_json: %s" % e
    if o not in OBLIGATIONS:
        return o, s, r, 0, "unknown_obligation"
    if o == "None":
        return o, "not_applicable", r, 1, ""
    if s not in ("satisfied", "not_satisfied"):
        return o, s, r, 0, "bad_status"
    return o, s, r, 1, ""


def mock_reply(clause):
    """Fake reply for testing the pipeline only. NOT a model."""
    t = clause.lower()
    for key, o in (("breach", "Breach Notification"), ("withdraw", "Withdrawal of Consent"),
                   ("retain", "Retention and Erasure"), ("child", "Childrens Data"),
                   ("security", "Security Safeguards"), ("grievance", "Grievance Redressal")):
        if key in t:
            return json.dumps({"obligation": o, "status": "not_satisfied", "reason": "mock"}), 5.0, 100, 20
    if "so bad" in t:
        return "not json at all", 5.0, 100, 5
    return json.dumps({"obligation": "None", "status": "not_applicable", "reason": "mock"}), 5.0, 100, 20


def make_caller(model, timeout):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        die("no API key. Set the environment variable GEMINI_API_KEY (or put it in a .env file).")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        die("package missing. Run: pip install google-genai")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")

    def call(prompt):
        last = None
        for attempt in range(4):
            try:
                t0 = time.time()
                r = client.models.generate_content(model=model, contents=prompt, config=cfg)
                ms = (time.time() - t0) * 1000
                u = getattr(r, "usage_metadata", None)
                # thinking models bill their reasoning tokens as output, so count both
                out_tok = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
                return (r.text or ""), ms, getattr(u, "prompt_token_count", None), out_tok
            except Exception as e:  # network or rate limit: wait and try again
                last = e
                print("  API problem (try %d of 4): %s: %s" % (attempt + 1, type(e).__name__, str(e)[:300]))
                time.sleep(2 ** attempt * 2)
        raise RuntimeError("API failed after 4 tries: %s" % last)
    return call


def load_clauses(db, split, limit):
    con = sqlite3.connect(db)
    q = ("SELECT c.clause_id, c.clause_text FROM clauses c JOIN policies p ON p.policy_id=c.policy_id "
         "JOIN final_labels f ON f.clause_id=c.clause_id WHERE p.split=? ORDER BY c.clause_id")
    rows = con.execute(q, (split,)).fetchall()
    con.close()
    return rows[:limit] if limit else rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="db/dpdp.db")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--out", default="")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="only the first N clauses (for trying it out)")
    p.add_argument("--model", default="")
    p.add_argument("--sleep", type=float, default=0.5, help="seconds between calls")
    p.add_argument("--lock", default="data/labels/LOCKED")
    p.add_argument("--dev", action="store_true", help="practice run on db/dev.db, skips the lock check")
    p.add_argument("--mock", action="store_true", help="fake replies, no API")
    p.add_argument("--timeout", type=float, default=60, help="seconds to wait for one reply")
    p.add_argument("--only-run", type=int, default=0, help="do just this run number (1, 2 or 3), so you can run several windows at once")
    p.add_argument("--ping", action="store_true", help="send ONE tiny request, print the result, and stop")
    a = p.parse_args()

    if not a.ping and not a.out:
        die("--out is required (the folder to save answers in)")
    if a.ping:
        a.dev = True
    if a.dev:
        if os.path.basename(a.db).lower() == "dpdp.db" and not a.ping:
            die("--dev is only for a dev copy of the database, never db/dpdp.db")
        print("DEV RUN. These numbers are NOT results for the paper.")
    elif a.split == "test" and not os.path.exists(a.lock):
        die("%s does not exist. The test set is not locked yet." % a.lock)
    if a.split == "test" and not a.dev and not a.mock and a.limit:
        die("do not use --limit on the real test set")

    model = a.model
    if not model and not a.mock:
        try:
            import yaml
            model = (yaml.safe_load(open("config.yaml")).get("gemini") or {}).get("model_name", "")
        except Exception:
            model = ""
        if not model:
            die("no model name. Use --model NAME or fill gemini.model_name in config.yaml")
    if a.mock:
        model = "MOCK"
    if a.ping:
        if a.mock:
            die("--ping needs the real API, not --mock")
        call = make_caller(model, a.timeout)
        t0 = time.time()
        raw, ms, tin, tout = call('Return only this JSON: {"ok": true}')
        print("Reply: %r" % raw)
        print("Seconds: %.1f   input tokens: %s   output tokens (incl. thinking): %s   model: %s" % (time.time() - t0, tin, tout, model))
        return
    clauses = load_clauses(a.db, a.split, a.limit)
    if not clauses:
        die("no labelled clauses in split '%s' of %s" % (a.split, a.db))
    os.makedirs(a.out, exist_ok=True)
    call = None if a.mock else make_caller(model, a.timeout)
    sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
    meta = {"model": model, "prompt_version": PROMPT_VERSION, "prompt_sha256_12": sha, "temperature": 0,
            "runs": a.runs, "n_clauses": len(clauses), "split": a.split, "db": a.db, "mock": a.mock,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version()}
    mp = os.path.join(a.out, "meta.json")
    if os.path.exists(mp):
        old = json.load(open(mp))
        if (old["model"], old["prompt_sha256_12"]) != (model, sha):
            die("%s was made with a different model or prompt. Use a new --out folder." % a.out)
        meta["first_started"] = old.get("first_started", old["date"])
    if not os.path.exists(mp):
        json.dump(meta, open(mp, "w"), indent=2)

    to_do = [a.only_run] if a.only_run else list(range(1, a.runs + 1))
    for run in to_do:
        path = os.path.join(a.out, "run%d.csv" % run)
        done = set()
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                done = {int(r["clause_id"]) for r in csv.DictReader(f)}
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            todo = [c for c in clauses if c[0] not in done]
            print("run %d: %d clauses to do (%d already done)" % (run, len(todo), len(done)))
            for i, (cid, text) in enumerate(todo, 1):
                prompt = PROMPT.replace("{clause_text}", text)
                try:
                    raw, ms, tin, tout = mock_reply(text) if a.mock else call(prompt)
                    o, s, r, ok, err = parse(raw)
                except Exception as e:
                    raw, ms, tin, tout, o, s, r, ok, err = "", "", None, None, "", "", "", 0, str(e)
                w.writerow({"clause_id": cid, "run": run, "obligation": o, "status": s, "reason": r, "valid": ok,
                            "error": err, "latency_ms": ms, "input_tokens": tin, "output_tokens": tout, "raw": raw})
                f.flush()
                if not a.mock:
                    time.sleep(a.sleep)
                if i % 25 == 0:
                    print("  run %d: %d / %d" % (run, i, len(todo)))
    bad = 0
    for run in to_do:
        with open(os.path.join(a.out, "run%d.csv" % run), newline="", encoding="utf-8") as f:
            bad += sum(1 for r in csv.DictReader(f) if r["valid"] != "1")
    print("Done. Invalid or failed answers across all runs: %d. Saved to %s" % (bad, a.out))


if __name__ == "__main__":
    main()