# Model card: VA Mail Sorter

## Model details

- **Name:** VA Mail Sorter triage classifier
- **Type:** TF-IDF (word unigrams and bigrams) + L2 logistic regression, sigmoid-calibrated with `CalibratedClassifierCV` (`method="sigmoid"`, `ensemble=False`, 5-fold on the training split)
- **Deployed C:** 0.25 (chosen by training-split stratified CV, maximizing F2 of ACTION_NEEDED, then recall)
- **Decision threshold:** 0.07 on calibrated P(ACTION_NEEDED)
- **Threshold policy:** max_f2_fallback, selected on the **validation** split only
- **Seed:** 42
- **Comparisons (not deployed):** keyword rules, calibrated linear SVM (C=0.5), calibrated multinomial naive Bayes (alpha=0.1)
- **Suspicious override:** rule score, cutoff 3.5 chosen on validation (max_f1_then_recall). When it fires, the triage label is SUSPICIOUS and replaces the class label in the report.

Probabilities used at inference come from the sigmoid calibrator. Term explanations come from the single logistic regression fit on the full training split (the `ensemble=False` estimator).

## Intended use

A local triage aid for a veteran sorting mail that *resembles* VA, DFAS, and VGLI (Prudential/Securian-administered life insurance) notices into ACTION_NEEDED or INFORMATIONAL, with a calibrated probability and the top reasons.

The tool never replies, deletes, or moves mail.

## Out of scope

This is not legal advice, benefits advice, or an official VA, DFAS, Prudential, or Securian system. It does not decide claims, debts, premiums, or appeals. A score is not a determination. Verify the contents of a real letter on VA.gov, myPay, or the insurer's own site. Do not click links in mail that looks suspicious.

## Data

- Public corpus: synthetic subjects and bodies only (`data/synthetic/emails.csv`)
- Rows: 192 train / 80 validation / 32 test
- Labels: ACTION_NEEDED and INFORMATIONAL, plus a SUSPICIOUS flag that overrides the class at triage time
- Split: whole synthetic template families are assigned before rendering; families do not cross partitions (seed 42)
- The test split is not used to choose C, the class threshold, or the suspicious cutoff
- Labeling rules: `docs/labeling_guide.md`

## Evaluation procedure

1. Stratified 5-fold CV on the training split compares the keyword baseline, logistic regression, linear SVM, and multinomial naive Bayes. CV predictions use each model's default decision rule (probability 0.5, or the SVM margin).
2. The deployed model is logistic regression, because its coefficients are the explanation. Its C is the CV winner inside the logistic grid only.
3. `CalibratedClassifierCV` fits the sigmoid map from out-of-fold training scores.
4. The class threshold and the suspicious cutoff are chosen on validation only.
5. The figures below are a single pass over the held-out test split.

Recall of ACTION_NEEDED is the priority: a missed deadline costs more than an extra review. The validation rule keeps the highest recall that still holds precision of at least 0.70. Ties break toward the threshold closest to 0.5. If no threshold clears the floor, the rule falls back to F2.

## Held-out test metrics (deployed model)

| | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| ACTION_NEEDED | 0.800 | 1.000 | 0.889 | 16 |
| INFORMATIONAL | 1.000 | 0.750 | 0.857 | 16 |

- F2 (ACTION_NEEDED): 0.952
- Brier score: 0.027661
- Expected calibration error (10 equal-width bins): 0.115
- Confusion matrix (rows true, columns predicted; order ACTION_NEEDED, INFORMATIONAL): [[16, 0], [4, 12]]
- Validation operating point at this same threshold: precision 0.608, recall 1.000, F2 0.886

![Held-out confusion matrix](../reports/confusion_matrix.png)

![Held-out reliability diagram](../reports/calibration_curve.png)

### Suspicious override on the held-out split

| Precision | Recall | F1 | Suspicious support |
| ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 16 |

## Training-split cross-validation

Default decision rule, not the validation threshold. Used to pick logistic C and to report baselines.

| Model | F2 mean | F2 std | Precision mean | Recall mean |
| --- | ---: | ---: | ---: | ---: |
| keyword_rules | 0.980 | 0.007 | 0.906 | 1.000 |
| tfidf_logreg_C0.25 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_logreg_C1.0 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_logreg_C4.0 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_linear_svc_C0.5 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_linear_svc_C1.0 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_linear_svc_C4.0 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_multinomial_nb_alpha0.1 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_multinomial_nb_alpha0.5 | 1.000 | 0.000 | 1.000 | 1.000 |
| tfidf_multinomial_nb_alpha1.0 | 1.000 | 0.000 | 1.000 | 1.000 |

## Held-out comparison at each model's own validation threshold

The deployed model stays logistic regression even if another row is higher. Test numbers were not used to choose it.

| Model | Precision ACTION | Recall ACTION | F2 ACTION | Brier |
| --- | ---: | ---: | ---: | ---: |
| keyword_rules | 0.500 | 1.000 | 0.833 | 0.249615 |
| tfidf_linear_svc | 0.842 | 1.000 | 0.964 | 0.036750 |
| tfidf_multinomial_nb | 1.000 | 1.000 | 1.000 | 0.050594 |
| tfidf_logreg | 0.800 | 1.000 | 0.952 | 0.027661 |

## Error analysis

Class mismatches on the held-out split, up to 15 rows. The full table, including hard cases that were scored correctly, is `reports/error_analysis.csv`.

| Id | True | Predicted | P(action) | Note |
| --- | --- | --- | ---: | --- |
| vams-0289 | INFORMATIONAL | ACTION_NEEDED | 0.098 | Suspicious cues were below the validation cutoff. |
| vams-0299 | INFORMATIONAL | ACTION_NEEDED | 0.070 | Suspicious cues were below the validation cutoff. |
| vams-0302 | INFORMATIONAL | ACTION_NEEDED | 0.170 | Suspicious cues were below the validation cutoff. |
| vams-0303 | INFORMATIONAL | ACTION_NEEDED | 0.084 | Suspicious cues were below the validation cutoff. |

Suspicious false alarms (benign mail at or above the validation cutoff). The cue list matches words such as "password" even when the sentence says the office will not ask for one.

No suspicious false alarms on the held-out test split.

## Explanations

Each prediction lists up to eight terms by contribution (TF-IDF value times the logistic coefficient) in the direction of the predicted class, and any rule hits for a deadline date, a dollar amount, or the words debt, due, action required, and decision.

## Ethical considerations

- A false informational label can hide a deadline. The threshold policy spends precision to avoid that, and still will miss some polite notices.
- A false action label creates review work. That is the cheaper error for this aid, not a reason to treat the score as a demand.
- Phishing look-alikes are a separate flag. The flag is a heuristic. It is not a determination that a real message is safe.
- The corpus has no real veterans, claim numbers, Social Security numbers, or account numbers. Do not point this training code at a real mailbox export and then publish the result.

## Limitations

- Every public email is synthetic. Performance on real VA, DFAS, or VGLI mail is unmeasured.
- The test split is small, so a single letter moves precision and recall.
- Calibration is estimated on that same small test split; the reliability diagram is noisy.
- Linear TF-IDF explanations miss negation scope and document layout that a scanned letter would have.
- The suspicious rules look for a fixed cue list (passwords, payment scams, suspension threats, verify-your-account language, look-alike links). A phish that avoids those words will not be flagged.
- Optional Gmail access is read-only, off by default, and not part of the reported metrics.

## Privacy

`data/real/` and `secrets/` are gitignored. Training and the numbers in this card use `data/synthetic/` only.
