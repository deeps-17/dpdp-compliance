-- DPDP Compliance Classifier: SQLite schema
-- Rebuild the database any time with:  python db/init_db.py --reset
PRAGMA foreign_keys = ON;

-- One row per company privacy policy. The train/test split lives HERE (per policy),
-- so clauses from one policy never end up on both sides.
CREATE TABLE IF NOT EXISTS policies (
    policy_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name   TEXT NOT NULL,
    sector         TEXT,
    source_url     TEXT,
    status         TEXT NOT NULL DEFAULT 'not_started'
                   CHECK (status IN ('not_started','collected','blocked','failed','excluded')),
    status_note    TEXT,
    date_accessed  TEXT,
    content_hash   TEXT,
    raw_path       TEXT,
    split          TEXT CHECK (split IN ('train','test') OR split IS NULL),
    UNIQUE (company_name, sector)
);

-- Reference table: the DPDP obligations used in the rubric.
CREATE TABLE IF NOT EXISTS obligations (
    obligation_id  INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    act_section    TEXT
);

-- One row per clause cut from a policy.
CREATE TABLE IF NOT EXISTS clauses (
    clause_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id     INTEGER NOT NULL REFERENCES policies(policy_id),
    clause_index  INTEGER NOT NULL,
    clause_text   TEXT NOT NULL,
    text_hash     TEXT,
    UNIQUE (policy_id, clause_index)
);

-- One row per (clause, annotator). Needed to compute Cohen's kappa.
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    clause_id      INTEGER NOT NULL REFERENCES clauses(clause_id),
    annotator      TEXT NOT NULL,
    obligation_id  INTEGER NOT NULL REFERENCES obligations(obligation_id),
    status         TEXT NOT NULL
                   CHECK (status IN ('satisfied','not_satisfied','not_stated','not_applicable')),
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (clause_id, annotator)
);

-- The agreed label after disagreements are resolved. Models train and test on this table.
CREATE TABLE IF NOT EXISTS final_labels (
    clause_id      INTEGER PRIMARY KEY REFERENCES clauses(clause_id),
    obligation_id  INTEGER NOT NULL REFERENCES obligations(obligation_id),
    status         TEXT NOT NULL
                   CHECK (status IN ('satisfied','not_satisfied','not_stated','not_applicable')),
    method         TEXT NOT NULL DEFAULT 'agreed'
                   CHECK (method IN ('agreed','adjudicated'))
);

-- One row per training or prompting run, so every result is reproducible.
CREATE TABLE IF NOT EXISTS model_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL
                    CHECK (model_name IN ('svm','random_forest','gemini','keyword_baseline')),
    model_version   TEXT,
    prompt_version  TEXT,
    params_json     TEXT,
    random_seed     INTEGER,
    started_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  INTEGER NOT NULL REFERENCES model_runs(run_id),
    clause_id               INTEGER NOT NULL REFERENCES clauses(clause_id),
    predicted_obligation_id INTEGER REFERENCES obligations(obligation_id),
    predicted_status        TEXT
                            CHECK (predicted_status IN ('satisfied','not_satisfied','not_stated','not_applicable')
                                   OR predicted_status IS NULL),
    confidence              REAL,
    latency_ms              REAL,
    raw_response            TEXT,
    UNIQUE (run_id, clause_id)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    eval_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES model_runs(run_id),
    split            TEXT NOT NULL CHECK (split IN ('train','test','cv')),
    accuracy         REAL,
    precision_macro  REAL,
    recall_macro     REAL,
    f1_macro         REAL,
    n_clauses        INTEGER,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clauses_policy      ON clauses(policy_id);
CREATE INDEX IF NOT EXISTS idx_clauses_hash        ON clauses(text_hash);
CREATE INDEX IF NOT EXISTS idx_annotations_clause  ON annotations(clause_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run     ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_clause  ON predictions(clause_id);

-- Seed the obligations (from the rubric).
INSERT OR IGNORE INTO obligations (obligation_id, name, act_section) VALUES
 (1,  'Notice',                 'Section 5'),
 (2,  'Consent',                'Section 6(1)-(3)'),
 (3,  'Withdrawal of Consent',  'Section 6(4)-(6)'),
 (4,  'Purpose Limitation',     'Sections 4, 7'),
 (5,  'Security Safeguards',    'Section 8(5)'),
 (6,  'Breach Notification',    'Section 8(6)'),
 (7,  'Retention and Erasure',  'Section 8(7)-(8)'),
 (8,  'Grievance Redressal',    'Sections 8(9)-(10), 13'),
 (9,  'Childrens Data',         'Section 9'),
 (10, 'Cross-Border Transfer',  'Section 16'),
 (0,  'None',                   NULL);
