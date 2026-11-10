# Results (generated 2026-11-10 13:19)

Classical folder: `runs/classical`. Gemini folder: `none`.

## Table 2 (obligation): macro-F1 and accuracy

| system        |   macro_f1 |   ci_low |   ci_high |   accuracy |
|:--------------|-----------:|---------:|----------:|-----------:|
| majority      |      0.071 |    0.062 |     0.081 |      0.472 |
| keyword       |      0.588 |    0.471 |     0.739 |      0.683 |
| svm           |      0.623 |    0.535 |     0.686 |      0.606 |
| random_forest |      0.557 |    0.459 |     0.645 |      0.533 |

## Table 3: F1 per obligation (test)

|                       |   n_test |   svm |   random_forest |
|:----------------------|---------:|------:|----------------:|
| Consent               |       17 | 0.452 |           0.552 |
| Cross-Border Transfer |        4 | 0.889 |           0.75  |
| Grievance Redressal   |        8 | 0.615 |           0.5   |
| None                  |       85 | 0.642 |           0.562 |
| Notice                |       20 | 0.48  |           0.423 |
| Purpose Limitation    |       20 | 0.609 |           0.464 |
| Retention and Erasure |        8 | 0.5   |           0.429 |
| Security Safeguards   |       14 | 0.692 |           0.667 |
| Withdrawal of Consent |        4 | 0.727 |           0.667 |

## Table 2 (status): macro-F1 and accuracy

| system        |   macro_f1 |   ci_low |   ci_high |   accuracy |
|:--------------|-----------:|---------:|----------:|-----------:|
| majority      |      0.451 |    0.425 |     0.473 |      0.821 |
| keyword       |      0.451 |    0.425 |     0.473 |      0.821 |
| svm           |      0.451 |    0.397 |     0.529 |      0.716 |
| random_forest |      0.491 |    0.426 |     0.599 |      0.8   |

## Table 2 (joint): macro-F1 and accuracy

| system        |   macro_f1 |   ci_low |   ci_high |   accuracy |
|:--------------|-----------:|---------:|----------:|-----------:|
| majority      |      0.053 |    0.047 |     0.065 |      0.472 |
| keyword       |      0.036 |    0.034 |     0.044 |      0.389 |
| svm           |      0.297 |    0.214 |     0.409 |      0.489 |
| random_forest |      0.26  |    0.209 |     0.364 |      0.417 |
