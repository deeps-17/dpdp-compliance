"""
Cut each collected privacy policy into clauses and store them in the `clauses` table.

Typical use (always look first, then write):

  1. Pilot on 12 training policies, spread across sectors, short and long mixed:
       python src/cleaning/split_clauses.py --pilot 12
     This only REPORTS and writes data/processed/split_review.csv. Nothing is saved.
  2. Read ~30 clauses per policy in the CSV (or use --sample 30), fix the rules, run again.
  3. When the pilot looks clean, save it:
       python src/cleaning/split_clauses.py --pilot 12 --write
  4. Later, for everything you selected:
       python src/cleaning/split_clauses.py --all --write

  Or name the policies yourself:   --ids 3,7,12

What it does to each policy text, in order:
  - removes menu lines at the top and "Back to top" / copyright lines at the bottom
  - drops bare "Last updated: ..." lines
  - finds headings (short line, no full stop, followed by longer text); a heading is not a clause,
    it is stored as a label on the clauses under it
  - cuts paragraphs into sentences, but not at "Pvt. Ltd.", "No.", "Mr.", "e.g." and similar
  - turns "(i) ... (ii) ..." lists and bullet lists into one clause per item, with the opening words
    ("We may share your data with:") copied in front of each item
  - keeps an address / grievance-officer contact block together as one clause
  - attaches fragments of fewer than 5 words to the sentence before
  - flags sentences of more than 80 words (flag = 'long') and very short leftovers (flag = 'short')
  - drops exact repeats inside the same policy

Two columns are added to `clauses` the first time you use --write: heading and flag.
(Add them to db/schema.sql too, so --reset keeps them. See the end of the chat message.)

Re-running --write on a policy replaces its clauses, unless labels or predictions already exist for
them; then it refuses (use --force only if you really mean to delete those labels).
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
REVIEW_CSV = ROOT / "data" / "processed" / "split_review.csv"

MIN_WORDS = 5            # fragments shorter than this are attached to the sentence before
LONG_WORDS = 80          # clauses longer than this are flagged
HEADING_MAX_WORDS = 12   # a heading is at most this long
CONTACT_MAX_WORDS = 14   # a line of a contact block is at most this long

# ---- Words that mark page furniture. Extend these lists when the pilot shows new junk. ----
NAV_RE = re.compile(
    r"^(login|log in|sign ?in|sign ?up|register|cart|my cart|menu|home|search|help|help ?cent(er|re)|"
    r"contact us|categories|download( the)? app|become a seller|skip to (main )?content|wishlist|"
    r"my account|account|orders|my orders|notifications|track order|language|english|hindi|more|"
    r"about us|careers|blog|stores?|offers?)$", re.I)
FOOTER_STOP_RE = re.compile(r"(back to top|scroll to top|all rights reserved|\u00a9|copyright\s*\d{4})", re.I)
FOOTER_RE = re.compile(
    r"^(follow us.*|download( the)? app.*|facebook|twitter|instagram|linkedin|youtube|"
    r"terms( of (use|service))?|privacy( policy)?|sitemap|help|contact us|about us|careers)$", re.I)
YEAR_ONLY_RE = re.compile(r"^\W*\d{4}\W*$")
# a copyright / "all rights reserved" line anywhere in the text is page furniture, never a clause
JUNK_LINE_RE = re.compile(r"^(?:.*(?:\{\{|\}\}).*|processing your payment\.?|please do not close this tab\.?|\u00a9.{0,80}|copyright\b(?=.*\d{4}).{0,60}|all rights reserved\.?|desktop site|mobile site)$", re.I)
NUMBER_ONLY_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3})*\.?$")                 # a line holding just "4." or "5.1.1."
SECTION_NUM_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3})*[.)]\s+")               # "2.2. Text..." at the start of a line
LEAD_NUM_RE = SECTION_NUM_RE
SECNUM_IN_TEXT_RE = re.compile(r"(?:^|(?<=[.!?;:] ))\d{1,3}(?:\.\d{1,3})*\.(?=\s)")
META_RE = re.compile(
    r"^(last (updated|modified|revised)|effective( date)?|version|updated on|date of last revision)\b.{0,40}$", re.I)

ABBREVIATIONS = ["Pvt", "Ltd", "Inc", "Co", "Corp", "No", "Nos", "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr",
                 "Jr", "vs", "Rs", "Sec", "Art", "Approx", "Dept", "Govt", "Est", "Fig", "Ref"]
ABBR_RE = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\.")
ABBR_TAIL_RE = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\.$")
INITIALS_RE = re.compile(r"(?:\b[A-Za-z]\.){2,}")          # e.g.  i.e.  U.S.  a.m.
SENT_END_RE = re.compile(r"([.!?][\"\u201d')\]]*)\s+(?=[\"\u201c\u2018'(\[]?[A-Z0-9])")

BULLET_RE = re.compile(
    r"^\s*(?:[\u2022\u25cf\u25aa\u2023\u25e6*\-\u2013\u2014]|"
    r"\((?:[ivxlc]+|[a-z]|\d{1,2})\)|(?:[ivxlc]+|[a-z]|\d{1,2})[.)])\s+(?=\S)", re.I)
NUMBER_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")
ENUM_RE = re.compile(r"(?<!\w)\(\s*((?:i{1,3}|iv|v|vi{1,3}|ix|x)|[a-h]|[1-9])\s*\)", re.I)

CONTACT_KEY_RE = re.compile(
    r"^(?:name|designation|e-?mail|address|phone|tel|telephone|mobile|contact(?: (?:no|number|details))?|"
    r"time|timings?|fax|website|url|grievance officer|nodal officer|officer)"
    r"(?:\s+(?:of|id)\b[^:]{0,40})?\s*[:\-\u2013]", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")
PIN_RE = re.compile(r"\b\d{6}\b")


SENTENCE_STARTERS = {"we", "you", "your", "our", "please", "this", "these", "those", "that", "it", "they", "he",
                     "she", "i", "the", "there", "if", "no", "yes", "see", "note", "kindly", "thank", "do",
                     "does", "all", "any"}


def words(s):
    return len(s.split())


def normalise(s):
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"[ \t]+", " ", s).strip()


def ends_terminal(s):
    return s.rstrip().endswith((".", "!", "?", ":", "\u201d", '"'))


def text_hash(s):
    return hashlib.sha1(re.sub(r"\s+", " ", s.lower()).strip().encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------
# Step 1: tidy the edges
# ------------------------------------------------------------------------------------------
def strip_edges(lines, stats, removed):
    start = 0
    while start < len(lines):
        t = lines[start]
        if not t:
            start += 1
        elif NAV_RE.match(t):
            removed.append("TOP:    " + t)
            stats["nav_top"] += 1
            start += 1
        else:
            break
    lines = lines[start:]

    window = max(0, len(lines) - 15)
    for i in range(window, len(lines)):
        if lines[i] and words(lines[i]) <= 15 and FOOTER_STOP_RE.search(lines[i]):
            after = [x for x in lines[i + 1:] if x]
            if all(words(x) <= 10 for x in after):
                # everything after the marker is short page furniture: cut it all
                for gone in lines[i:]:
                    if gone:
                        removed.append("BOTTOM: " + gone)
                        stats["nav_bottom"] += 1
                lines = lines[:i]
            else:
                # real text follows (e.g. a copyright line in the middle): remove only the marker
                # line, plus a year left on a line of its own
                drop = [i]
                if i + 1 < len(lines) and YEAR_ONLY_RE.match(lines[i + 1]):
                    drop.append(i + 1)
                for k in drop:
                    removed.append("BOTTOM: " + lines[k])
                    stats["nav_bottom"] += 1
                lines = [x for k, x in enumerate(lines) if k not in drop]
            break
    while lines and (not lines[-1] or FOOTER_RE.match(lines[-1])):
        if lines[-1]:
            removed.append("BOTTOM: " + lines[-1])
            stats["nav_bottom"] += 1
        lines.pop()

    kept = []
    for t in lines:
        if t and (META_RE.match(t) or JUNK_LINE_RE.match(t)):
            removed.append("META:   " + t)
            stats["meta"] += 1
        else:
            kept.append(t)
    return kept


# ------------------------------------------------------------------------------------------
# Step 2: group lines into blocks (heading / paragraph / bullet item / contact block)
# ------------------------------------------------------------------------------------------
def preprocess_numbers(lines):
    """Section numbers are not text. A number alone on a line, or in front of a full sentence, just marks
    the start of a new paragraph; short numbered titles ("1. Information we collect") are left for the
    heading rule."""
    out = []
    for t in lines:
        if NUMBER_ONLY_RE.match(t):
            out.append("")
            continue
        m = SECTION_NUM_RE.match(t)
        if m:
            rest = t[m.end():]
            if rest.rstrip().endswith((".", "!", "?", ";")) or words(rest) >= 14:
                out.append("")
                out.append(rest)
                continue
        out.append(t)
    return out


def next_nonblank(lines, i):
    while i < len(lines):
        if lines[i]:
            return lines[i]
        i += 1
    return None


def is_strong_contact(line):
    return bool(CONTACT_KEY_RE.match(line) or "@" in line or PHONE_RE.search(line) or PIN_RE.search(line))


def is_contact_start(line):
    if words(line) > CONTACT_MAX_WORDS or not is_strong_contact(line):
        return False
    if CONTACT_KEY_RE.match(line) or words(line) <= 6:
        return True
    looks_like_sentence = line.rstrip().endswith((".", "!", "?")) and line.count(",") < 3
    return not looks_like_sentence


def contact_extends(line):
    if words(line) > CONTACT_MAX_WORDS:
        return False
    if is_strong_contact(line):
        return True
    return not line.rstrip().endswith((".", "!", "?")) or bool(ABBR_TAIL_RE.search(line))


def is_heading(line, nxt):
    t = line.strip()
    if not nxt or not t or words(t) > HEADING_MAX_WORDS:
        return False
    if t.endswith((".", "!", ";", ",", ":")) or "@" in t:
        return False
    if t.endswith("?") and words(t) > 8:
        return False
    if CONTACT_KEY_RE.match(t) or t.startswith("("):
        return False
    body = NUMBER_PREFIX_RE.sub("", t)
    if not body or not (body[0].isupper() or body[0].isdigit()):
        return False
    if t.isupper():
        return True
    return words(nxt) > words(t) or is_strong_contact(nxt) or (
        words(nxt) <= HEADING_MAX_WORDS and nxt[0].isupper() and not nxt.endswith((".", "!", "?")))


def parse_blocks(lines):
    blocks = []                    # (kind, text, heading)
    heading = None
    buf = None                     # lines of the paragraph being built
    bullet = None                  # lines of the bullet item being built

    def flush():
        nonlocal buf, bullet
        if buf:
            blocks.append(("para", " ".join(buf), heading))
        if bullet:
            blocks.append(("bullet", " ".join(bullet), heading))
        buf = None
        bullet = None

    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if not line:
            flush()
            i += 1
            continue
        nxt = next_nonblank(lines, i + 1)

        if is_contact_start(line):
            flush()
            run = [line]
            j = i + 1
            while j < n and lines[j] and contact_extends(lines[j]):
                if not is_strong_contact(lines[j]) and is_heading(lines[j], next_nonblank(lines, j + 1)):
                    break                        # a heading right after the block is not part of it
                run.append(lines[j])
                j += 1
            blocks.append(("contact", " | ".join(run), heading))
            i = j
            continue

        if is_heading(line, nxt):
            flush()
            heading = line
            i += 1
            continue

        if BULLET_RE.match(line):
            flush()
            bullet = [BULLET_RE.sub("", line, count=1)]
            i += 1
            continue

        if bullet is not None:
            if line[0].islower() and not ends_terminal(" ".join(bullet)):
                bullet.append(line)          # wrapped bullet line
                i += 1
                continue
            flush()

        if buf and not ends_terminal(buf[-1]):
            buf.append(line)                 # hard-wrapped paragraph line
        else:
            flush()
            buf = [line]
        i += 1
    flush()
    return blocks


# ------------------------------------------------------------------------------------------
# Step 3: sentences, lists, fragments
# ------------------------------------------------------------------------------------------
def split_sentences(text):
    s = ABBR_RE.sub(lambda m: m.group(1) + "\x00", text)
    s = INITIALS_RE.sub(lambda m: m.group(0).replace(".", "\x00"), s)
    s = SECNUM_IN_TEXT_RE.sub(lambda m: m.group(0).replace(".", "\x00"), s)
    s = SENT_END_RE.sub(lambda m: m.group(1) + "\x01", s)
    return [p.replace("\x00", ".").strip() for p in s.split("\x01") if p.strip()]


def clean_item(item):
    item = item.strip()
    item = re.sub(r"(?:[;,]\s*(?:and|or)|[;,]|\s+(?:and|or))\s*$", "", item).strip()
    return item


def expand_enumeration(sentence, intro=None):
    """'We use data to: (i) A; (ii) B; and (iii) C.'  ->  three clauses with the opening words copied in."""
    marks = list(ENUM_RE.finditer(sentence))
    if len(marks) >= 2 and marks[0].group(1).lower() in {"i", "a", "1"}:
        stem = sentence[:marks[0].start()].strip()
        if not stem and intro:
            stem = intro
        out = []
        for k, m in enumerate(marks):
            end = marks[k + 1].start() if k + 1 < len(marks) else len(sentence)
            item = clean_item(sentence[m.end():end])
            if item:
                out.append(((stem + " " + item).strip(), "list"))
        if out:
            return out
    return [(sentence, "sent")]


def build_clauses(blocks):
    out = []                       # dicts: text, heading, kind
    intro = None
    for idx, (kind, text, head) in enumerate(blocks):
        nxt_kind = blocks[idx + 1][0] if idx + 1 < len(blocks) else None
        if kind == "para":
            if text.rstrip().endswith(":") and nxt_kind == "bullet":
                intro = text.rstrip()          # copied in front of each bullet below
                continue
            intro = None
            for sent in split_sentences(text):
                for piece, k in expand_enumeration(sent):
                    if k == "sent" and sent.rstrip().endswith(":"):
                        k = "intro"
                    out.append({"text": piece, "heading": head, "kind": k})
        elif kind == "bullet":
            out.append({"text": ((intro + " " + text) if intro else text).strip(), "heading": head, "kind": "list"})
        else:                                   # contact block
            intro = None
            out.append({"text": text, "heading": head, "kind": "contact"})
    return out


def finish(clauses, stats):
    merged = []
    pending = None                 # a run-in title ("Opting out.") waiting for the clause it belongs to

    def log(before, frag):
        stats["merged"] += 1
        stats["merged_log"].append((before[-50:], frag))

    def settle(piece):
        """A short piece that found no partner after it: attach it to the clause before, else keep it."""
        if merged and merged[-1]["heading"] == piece["heading"]:
            log(merged[-1]["text"], piece["text"])
            merged[-1]["text"] += " " + piece["text"]
        else:
            merged.append(piece)

    for c in clauses:
        c["text"] = LEAD_NUM_RE.sub("", normalise(c["text"]))
        if not c["text"]:
            continue
        if pending is not None:
            if c["heading"] == pending["heading"] and c["kind"] in ("sent", "list"):
                log(pending["text"], c["text"])
                c["text"] = pending["text"] + " " + c["text"]
            else:
                settle(pending)
            pending = None
        frag = c["kind"] == "sent" and words(c["text"]) < MIN_WORDS
        toks = c["text"].split()
        title_case = len(toks) >= 2 and all(t[0].isupper() for t in toks)
        if frag and c["text"][0].isupper() and not c["text"].endswith((";", ",")) \
                and (title_case or toks[0].lower().strip(".,:;?") not in SENTENCE_STARTERS):
            pending = c                      # title-like piece: belongs with what FOLLOWS it
            continue
        if frag and not c["text"][0].islower():
            settle(c)                        # tail of a sentence cut too early, or a very short sentence
            continue
        merged.append(c)
    if pending is not None:
        settle(pending)

    seen, final = set(), []
    for c in merged:
        h = text_hash(c["text"])
        if h in seen:
            stats["dups"] += 1
            continue
        seen.add(h)
        c["hash"] = h
        n = words(c["text"])
        c["words"] = n
        c["flag"] = "long" if n > LONG_WORDS else ("short" if n < MIN_WORDS else "")
        final.append(c)
    return final


def split_policy(text):
    stats = dict.fromkeys(["nav_top", "nav_bottom", "meta", "merged", "dups"], 0)
    stats["merged_log"] = []
    removed = []
    lines = [normalise(x) for x in text.replace("\r", "").split("\n")]
    lines = preprocess_numbers(strip_edges(lines, stats, removed))
    clauses = finish(build_clauses(parse_blocks(lines)), stats)
    stats["clauses"] = len(clauses)
    stats["short"] = sum(1 for c in clauses if c["flag"] == "short")
    stats["long"] = sum(1 for c in clauses if c["flag"] == "long")
    return clauses, stats, removed


# ------------------------------------------------------------------------------------------
# Database and command line
# ------------------------------------------------------------------------------------------
def connect(db):
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def raw_file(raw_path):
    if not raw_path:
        return None
    p = Path(raw_path)
    return p if p.is_absolute() else ROOT / p


def ensure_columns(conn):
    have = {r["name"] for r in conn.execute("PRAGMA table_info(clauses)")}
    for col in ("heading", "flag"):
        if col not in have:
            conn.execute(f"ALTER TABLE clauses ADD COLUMN {col} TEXT")
    conn.commit()


def pick_pilot(conn, n, seed):
    rows = conn.execute("SELECT policy_id, sector, raw_path FROM policies "
                        "WHERE split='train' AND status='collected'").fetchall()
    by_sector = {}
    for r in rows:
        p = raw_file(r["raw_path"])
        if p and p.exists() and p.stat().st_size > 0:
            by_sector.setdefault(str(r["sector"]), []).append((p.stat().st_size, r["policy_id"]))
    if not by_sector:
        sys.exit("No training policies found. Run select_policies.py first (it sets the split).")
    rng = random.Random(seed)
    sectors = sorted(by_sector)
    rng.shuffle(sectors)
    for s in sectors:
        by_sector[s].sort()
    picks, toggle = [], 0
    while len(picks) < n and any(by_sector.values()):
        for s in sectors:
            if len(picks) >= n:
                break
            if by_sector[s]:
                picks.append(by_sector[s].pop(0 if toggle % 2 == 0 else -1)[1])   # shortest, longest, ...
                toggle += 1
    return picks


def resolve_ids(conn, args):
    if args.ids:
        try:
            return [int(x) for x in args.ids.split(",") if x.strip()]
        except ValueError:
            sys.exit("--ids must be numbers separated by commas, e.g. --ids 3,7,12")
    if args.pilot:
        return pick_pilot(conn, args.pilot, args.seed)
    if args.all:
        return [r["policy_id"] for r in conn.execute(
            "SELECT policy_id FROM policies WHERE split IS NOT NULL AND status='collected' ORDER BY policy_id")]
    sys.exit("Say which policies: --pilot 12, --ids 3,7,12 or --all")


def has_labels(conn, policy_id):
    q = ("SELECT (SELECT COUNT(*) FROM annotations a JOIN clauses c ON a.clause_id=c.clause_id WHERE c.policy_id=?)"
         " + (SELECT COUNT(*) FROM final_labels f JOIN clauses c ON f.clause_id=c.clause_id WHERE c.policy_id=?)"
         " + (SELECT COUNT(*) FROM predictions p JOIN clauses c ON p.clause_id=c.clause_id WHERE c.policy_id=?)")
    return conn.execute(q, (policy_id,) * 3).fetchone()[0] > 0


def save_clauses(conn, policy_id, clauses, force):
    if has_labels(conn, policy_id):
        if not force:
            return False
        ids = "SELECT clause_id FROM clauses WHERE policy_id=?"
        for table in ("annotations", "final_labels", "predictions"):
            conn.execute(f"DELETE FROM {table} WHERE clause_id IN ({ids})", (policy_id,))
    conn.execute("DELETE FROM clauses WHERE policy_id=?", (policy_id,))
    conn.executemany(
        "INSERT INTO clauses (policy_id, clause_index, clause_text, text_hash, heading, flag) VALUES (?,?,?,?,?,?)",
        [(policy_id, i, c["text"], c["hash"], c["heading"], c["flag"]) for i, c in enumerate(clauses, 1)])
    conn.commit()
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Split policies into clauses.")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--pilot", type=int, metavar="N", help="N training policies spread over sectors and lengths")
    ap.add_argument("--ids", help="comma-separated policy_ids")
    ap.add_argument("--all", action="store_true", help="every selected policy (those with a train/test split)")
    ap.add_argument("--write", action="store_true", help="save to the database (default: report only)")
    ap.add_argument("--force", action="store_true", help="replace clauses even if labels exist (deletes the labels)")
    ap.add_argument("--sample", type=int, metavar="N", help="review CSV gets only N spread-out clauses per policy")
    ap.add_argument("--show-removed", action="store_true", help="print the menu/footer lines that were removed")
    ap.add_argument("--show-merged", action="store_true", help="print the short fragments that were attached to the sentence before")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conn = connect(args.db)
    ids = resolve_ids(conn, args)
    if args.write:
        ensure_columns(conn)

    marks = ",".join("?" * len(ids))
    info = {r["policy_id"]: r for r in conn.execute(
        f"SELECT policy_id, company_name, sector, status, raw_path, split FROM policies WHERE policy_id IN ({marks})", ids)}
    rng = random.Random(args.seed)

    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    review = []
    print(f"\n{'id':>4}  {'company':<24}{'sector':<22}{'chars':>8}{'clauses':>8}{'short':>6}{'long':>5}"
          f"{'merged':>7}{'dups':>5}{'menu':>5}  saved")
    totals = 0
    for pid in ids:
        r = info.get(pid)
        if r is None:
            print(f"{pid:>4}  not in the database, skipped")
            continue
        path = raw_file(r["raw_path"])
        if r["status"] != "collected" or not path or not path.exists():
            print(f"{pid:>4}  {r['company_name']:<24}no collected text, skipped")
            continue
        if r["split"] == "test":
            print(f"      note: {r['company_name']} is a TEST policy - split it if you must, but do not tune rules on it")
        text = path.read_text(encoding="utf-8", errors="replace")
        clauses, st, removed = split_policy(text)
        totals += len(clauses)

        saved = "-"
        if args.write:
            saved = "yes" if save_clauses(conn, pid, clauses, args.force) else "NO (labels exist, use --force)"
        print(f"{pid:>4}  {str(r['company_name'])[:23]:<24}{str(r['sector'])[:21]:<22}{len(text):>8}{st['clauses']:>8}"
              f"{st['short']:>6}{st['long']:>5}{st['merged']:>7}{st['dups']:>5}{st['nav_top'] + st['nav_bottom'] + st['meta']:>5}  {saved}")
        if args.show_removed:
            for line in removed:
                print("        removed ->", line[:100])

        if args.show_merged:
            for before, frag in st["merged_log"][:25]:
                print(f"        merged -> ...{before!r} + {frag!r}")

        shown = list(enumerate(clauses, 1))
        if args.sample and len(shown) > args.sample:
            shown = sorted(rng.sample(shown, args.sample))
        for i, c in shown:
            review.append([pid, r["company_name"], i, c["heading"] or "", c["kind"], c["flag"], c["words"], c["text"]])

    with open(REVIEW_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["policy_id", "company", "clause_index", "heading", "kind", "flag", "words", "clause_text"])
        w.writerows(review)
    print(f"\n{totals} clauses in total. Review file: {REVIEW_CSV}")
    if not args.write:
        print("Nothing was saved to the database (add --write when the clauses look right).")
    conn.close()


if __name__ == "__main__":
    main()