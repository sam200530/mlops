# Leakage-Safe Fraud Detection Platform

A fraud detection system on the [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection)
dataset, built around one question: **how do you know your validation number is real?**

The answer, measured in this repository:

> Random stratified 5-fold cross-validation scored **0.8512 PR-AUC**.
> Purged forward-chaining temporal CV — same model, same data, same code — scored
> **0.5583**.
>
> The random estimate was inflated by **+0.2929 (52% relative)** and reported a
> **five times smaller** standard deviation while doing it. It looked more
> trustworthy exactly where it was more wrong.

Everything else here exists to make that finding trustworthy and to ship the
resulting model.

> **Every number below was produced by code in this repository.** Model metrics
> from `scripts/train.py` (also in `reports/model_comparison.csv` and MLflow);
> dataset facts from `scripts/inspect_dataset.py`; drift figures from
> `scripts/monitor.py`. Where something was not executed, it says so.

---

## Contents

[Problem](#problem) · [Dataset](#dataset) · [Why Random CV Was Misleading](#why-random-cv-was-misleading) ·
[Temporal Validation](#temporal-validation) · [Leakage-Safe Feature Engineering](#leakage-safe-feature-engineering) ·
[Model Comparison](#model-comparison) · [Final Model](#final-model) · [Evaluation](#evaluation) ·
[SHAP Explainability](#shap-explainability) · [FastAPI](#fastapi) · [Docker](#docker) ·
[Drift Monitoring](#drift-monitoring) · [Testing / CI](#testing--ci) ·
[Project Architecture](#project-architecture) · [How to Run](#how-to-run) ·
[Limitations](#limitations) · [Future Improvements](#future-improvements)

---

## Problem

Given a card-not-present transaction, return a calibrated fraud probability fast
enough to sit inside an authorisation flow, plus the reasons behind it.

The binding constraint is **cost asymmetry**, not accuracy. Missing fraud means a
chargeback; flagging a legitimate customer means a declined sale. At 3.4993%
prevalence there are 27.58 legitimate transactions per fraudulent one, so even a
small false-positive *rate* produces an alert queue no review team can clear. The
system therefore optimises **ranking quality at low alert volume** and
**calibrated probabilities**.

Three properties of fraud shaped every decision:

- **Extreme imbalance.** Predicting "never fraud" scores **96.5007% accuracy**.
  Accuracy cannot separate a useful model from a constant, so it is reported
  nowhere in this project.
- **Non-stationarity.** Measured prevalence moves between **2.4762% and 4.1795%**
  across 30-day blocks. Validation assuming a fixed distribution measures the
  wrong thing.
- **Delayed labels.** Chargebacks arrive weeks later, so live PR-AUC is not
  observable. Degradation must be caught from input and score *distributions* —
  which is why drift monitoring here is not decoration.

## Dataset

Four CSVs, profiled by `scripts/inspect_dataset.py` (full output:
[`docs/01_dataset_audit.md`](docs/01_dataset_audit.md)):

| file | size | rows | columns | duplicate rows |
|---|---|---|---|---|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 | 0 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 | 0 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 | 0 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 | 0 |

**Target:** `isFraud` — 20,663 fraud / 569,877 legitimate = **3.4993%**.

**Column families:** `V1`–`V339` (mean 43.04% missing), `C1`–`C14` (**0%
missing**), `D1`–`D15` (timedeltas, 58.15% mean missing), `M1`–`M9`,
`card1`–`card6`, `addr1/2`, `dist1/2`, email domains, identity block.

### Three audit findings that changed the design

**1. The Kaggle test set is unlabeled** — 393 columns to train's 394, the only
difference being `isFraud`. No metric can be computed on it, so the final holdout
is carved from the *training* period by time, and the test files serve only as an
unlabeled distribution for drift analysis.

**2. Identity covers only 24.42% of train rows** (28.01% of test), strictly 1:1.
An inner join would discard 75.6% of the data, and since coverage itself differs
between periods it would bias the sample too. `LEFT JOIN`, with
`identity_present` promoted to a feature.

**3. `test_identity.csv` uses `id-01`…`id-38`; train uses `id_01`…`id_38`.** All
38 names differ. Unfixed, a model trained on one and scored on the other sees 38
all-null columns and **degrades silently instead of crashing**. Renamed once at
the load boundary, then asserted absent.

## Why Random CV Was Misleading

Three measured facts make random K-fold invalid here:

| fact | measurement |
|---|---|
| Train and test are disjoint in time | train days 1–183, test days 213–396, gap **exactly 30.0 days**, **0** shared timestamps |
| Prevalence is non-stationary | 2.4762% → 4.0373% → 4.0319% → 3.9265% → 3.4723% → 3.4013% → 4.1795% per 30-day block |
| Entities recur | the same card transacts repeatedly, so random folds train and validate on the same entity |

Deployment is **forward extrapolation across a gap**. Random CV measures
*interpolation inside a window the model will never operate in*.

So it was measured rather than asserted (`scripts/train.py --random-cv-control`):

| CV scheme | PR-AUC | ROC-AUC | P@top 1% | std dev |
|---|---|---|---|---|
| Random stratified 5-fold | **0.8512** | 0.9657 | 0.9975 | ± 0.0044 |
| Purged forward-chaining 5-fold | **0.5583** | 0.8838 | 0.9268 | ± 0.0225 |
| **Optimism** | **+0.2929** | +0.0819 | +0.0707 | — |

The holdout later scored **0.5669** — within one standard deviation of the
temporal estimate and nowhere near the random one. That is the practical
confirmation that the temporal scheme was the honest choice.

Logged to MLflow as `random_cv_optimism_pr_auc`.

## Temporal Validation

A chronological split was **verified feasible before adoption**
(`src/data/validation.py::validate_temporal_order`, asserted on every load):

| check | result |
|---|---|
| `TransactionDT` nulls / negatives | 0 / 0 |
| Already monotonic in raw file order | yes |
| `corr(TransactionID, TransactionDT)` | 0.99828 → the ID is a disguised time index, hard-excluded |
| Days covered | 1–182, **182 distinct days, none missing** |
| Largest gap between timestamps | 4,138 s (1.15 h) |
| Rows sharing a timestamp | 33,932 (**5.746%**), max 8 |
| Rows exactly on the 80th-percentile cut | **0** |

Ties matter: boundaries snap to timestamp edges so a tie group is never split
across partitions, and `_assert_disjoint_in_time` verifies no timestamp appears in
two partitions.

**Three-way split** (`data/processed/split_metadata.json`):

| partition | rows | days | fraud rate | role |
|---|---|---|---|---|
| train | 377,945 | 1–109 | 3.4119% | model fitting |
| validation | 94,487 | 109–141 | 3.9201% | threshold + calibration |
| **holdout** | **118,108** | **141–182** | **3.4409%** | **scored once, at the end** |

**Purged forward-chaining folds** over the modelling period: each trains on the
past and validates on the next contiguous block, with a **7-day purge gap** — at
least as wide as the longest velocity look-back (168 h), because otherwise a
trailing aggregate computed at the start of a validation block reaches back into
training rows even though the *rows* are separated.

| fold | train rows | train fraud | val rows | val fraud |
|---|---|---|---|---|
| 0 | 54,466 | — | 78,739 | — |
| 1 | 133,979 | 2.5288% | 78,739 | 4.1974% |
| 2 | 214,078 | 3.0517% | 78,738 | 3.9396% |
| 3 | 293,305 | 3.3481% | 78,739 | 3.7085% |
| 4 | 372,880 | 3.4030% | 78,739 | 3.8342% |

Validation fraud rate exceeds training fraud rate in *every* fold — a structural
property random folds would average away.

## Leakage-Safe Feature Engineering

The pipeline has a three-phase shape, and **the shape is the leakage control**
(`src/features/pipeline.py`):

| phase | what it does | safe on the full frame? |
|---|---|---|
| `prepare(df)` | row-local transforms + past-only velocity | **yes** — nothing fitted, nothing reads a future row |
| `fit(train_df)` | learns frequency counts, per-entity baselines, categorical vocabularies | **training partition only** |
| `transform(df)` | pure lookup | applied identically to validation, holdout, serving |

The only method that learns anything takes the training frame as its argument, so
fitting an encoder on validation data is structurally impossible rather than
merely discouraged.

**Encoders are refitted inside every CV fold** (`src/models/training.py::_fit_one`).
Frequency counts and per-entity means are *population statistics*; fitting them
once on the whole modelling period and then cross-validating would leak each
fold's validation rows into its own training features.

| family | features | mechanism |
|---|---|---|
| Amount | `log_amount`, `amount_cents`, `is_round_amount` | Right-skewed (mean 135.03, max 31,937). Class means are close (149.24 vs 134.51), so absolute amount is weak — the decimal part fingerprints card testing and currency conversion. |
| Per-entity amount | `*_amt_mean_hist`, `*_amt_diff_from_mean`, `*_amt_zscore` | Fraud is anomalous *for that account*, not absolutely. |
| Velocity | `*_txn_count_{1,24,168}h`, `*_amt_sum_*`, `*_seconds_since_prev` | Card testing and ATO are burst behaviours invisible in one transaction. |
| Time (cyclical) | `hour_of_day`, `day_of_week`, `is_night`, `is_weekend` | Fraud rate peaks where volume troughs — automation at off-peak hours. Cyclical, so it transfers across the 30-day gap. |
| Email | provider/TLD split, `email_domains_match` | Purchaser/recipient mismatch is a classic mule indicator. |
| Device | `device_vendor`, `os_family`, `browser_*`, screen dims | `DeviceInfo` has >1,000 distinct values and would overfit verbatim. |
| Missingness | `n_missing_total`, per-family counts, `identity_present` | Missingness is structural, not random, and correlates with fraud. |
| Frequency encoding | `card1_freq`, `card2_freq`, `addr1_freq`, … | Rare identifiers are disproportionately fraudulent; uses no label, so it cannot leak the target. |

**Velocity is causal, not leakage.** Each value uses only rows *strictly earlier
in time*, which at inference have already happened; the current row is excluded by
construction. Four tests in `tests/test_features.py::TestVelocityIsCausal` assert
this, including that five sequential transactions produce counts `[0, 1, 2, 3, 4]`.

**One deliberate score sacrifice.** Fitting frequency encodings on train ∪ test is
a well-known Kaggle booster and it is **transductive leakage** — it assumes the
scoring population is known at training time, which a live API cannot. Counts come
from the training partition only. This costs leaderboard points and is right for a
serving system.

**Hard exclusions:** `isFraud`; `TransactionID` (corr 0.998 with time); raw
`TransactionDT` (disjoint ranges, so a tree routes every test row to one leaf);
`V107` (constant in test, not train).

## Model Comparison

All three models on **identical persisted folds**, from a full run
(`python scripts/train.py --random-cv-control`). Every figure below is retained in
MLflow; `reports/model_comparison.csv` is regenerated on each run and therefore
reflects whatever was executed last, so the table here is the canonical record:

| model | CV PR-AUC | lift | ROC-AUC | precision | recall | F1 | Brier | P@top 1% | train time |
|---|---|---|---|---|---|---|---|---|---|
| **LightGBM (tuned)** | **0.5728 ± 0.0248** | **15.82×** | 0.8921 | 0.7161 | 0.4623 | 0.5613 | 0.0239 | 0.9283 | 1223.7 s |
| LightGBM (baseline) | 0.5583 ± 0.0225 | 15.45× | 0.8838 | 0.6766 | 0.4692 | 0.5533 | 0.0236 | 0.9268 | 1031.6 s |
| Random Forest | 0.4698 ± 0.0440 | 12.86× | 0.8822 | 0.5476 | 0.4051 | 0.4648 | 0.0920 | 0.8280 | 532.4 s |
| Logistic Regression | 0.3546 ± 0.0690 | 9.59× | 0.8317 | 0.4538 | 0.3581 | 0.3992 | 0.1386 | 0.7174 | 421.3 s |

**Boosting is measured to be better, not assumed** — LightGBM beats Random Forest
by +0.103 PR-AUC and Logistic Regression by +0.218.

**ROC-AUC hides most of that gap.** Random Forest reaches 0.8822 against
LightGBM's 0.8921 — a 0.010 difference — while the PR-AUC gap is ten times larger.
With 569,877 negatives in the FPR denominator, ROC-AUC barely registers the
false-positive volume separating these models. Selecting on ROC-AUC would have
called it near a tie.

**Calibration separates them further:** Brier 0.0239 (LightGBM) vs 0.0920 (RF) vs
0.1386 (LogReg).

Two caveats stated rather than hidden: Logistic Regression and Random Forest were
fitted on **150,000 rows** (all positives kept, negatives downsampled) because
their one-hot matrices are 1,296 features wide and a 472k × 1,296 dense float32
matrix does not fit this machine; LightGBM used every row with 530 native
features. Tuning gained **+0.0145 PR-AUC** from 5 Optuna trials in 1928.6 s — the
wall-clock cap stopped the search, not convergence.

## Final Model

**LightGBM**, Optuna-tuned, isotonic-calibrated, trained on all 472,432 modelling
rows. Hyperparameters: `learning_rate` 0.0276, `num_leaves` 230,
`min_child_samples` 144, `feature_fraction` 0.659, `bagging_fraction` 0.662.

Imbalance is handled by **reweighting, not resampling**. SMOTE would interpolate
between fraud rows across a ~530-column space that is largely categorical and
heavily missing; the interpolants would not be plausible transactions.
`scale_pos_weight` leaves the data honest and only changes the loss — and because
reweighting distorts probabilities, isotonic calibration follows.

The saved artifact bundles **model + feature pipeline + calibrator + threshold**
together, which removes the most common production failure in ML systems: a
preprocessing step drifting out of sync with the model.

## Evaluation

The holdout was scored **exactly once**, using the threshold (0.3827) chosen on
validation and applied unchanged.

| metric | validation (last fold) | **holdout (final)** |
|---|---|---|
| PR-AUC | 0.6008 | **0.5669** |
| PR-AUC lift over prevalence | 15.67× | **16.48×** |
| ROC-AUC | 0.9123 | **0.9091** |
| Precision | 0.7437 | **0.7279** |
| Recall | 0.5114 | **0.4727** |
| F1 | 0.6061 | **0.5732** |
| Brier | 0.0211 | **0.0202** |
| Precision @ top 0.1% | — | **0.9492** |
| Precision @ top 1% | 0.9098 | **0.9086** |
| Recall @ top 1% | — | **0.2640** |
| Rows | 78,739 | 118,108 |

**Confusion matrix** at threshold 0.3827 (prevalence 3.4409%):

|  | predicted legit | predicted fraud |
|---|---|---|
| **actually legit** | 113,326 | 718 |
| **actually fraud** | 2,143 | 1,921 |

What this means operationally:

- **PR-AUC 0.5669 is 16.48× the no-skill floor** of 0.0344. The absolute number
  looks unimpressive only if the floor is forgotten.
- **At a 1% alert budget, 90.9% of alerts are genuine fraud**, catching 26.4% of
  all fraud — nine in ten investigations are productive.
- **At the operating point: 718 false positives against 1,921 caught frauds** —
  roughly one false alarm per 2.7 detections, at the cost of missing 2,143. That
  trade is a business choice, which is why the threshold is configuration.
- **Calibration is real**: isotonic cut expected calibration error from 0.01477 to
  ~0 on validation, and the holdout Brier came in *below* validation.

**Holdout (0.5669) sits below validation (0.6008) and within one SD of the
temporal CV mean (0.5728 ± 0.0248).** That small drop is the expected, honest
pattern: validation informed the threshold and calibration, so it is mildly
optimistic; the holdout was untouched. A holdout scoring *above* validation would
be a reason to suspect the split, not to celebrate.

Figures in `reports/figures/`: PR curve with the prevalence baseline drawn on it,
ROC, reliability diagram, confusion matrix, per-class score distribution.

## SHAP Explainability

`src/explainability/shap_explainer.py`, using `TreeExplainer` — exact for tree
ensembles and needing no background dataset, which is what makes a per-request
explanation viable at all.

| rank | feature | mean \|SHAP\| |
|---|---|---|
| 1 | `C13` | 0.4902 |
| 2 | `C1` | 0.3284 |
| 3 | **`D1_anchored`** (engineered) | 0.2640 |
| 4 | `C14` | 0.2519 |
| 5 | `V70` | 0.2187 |
| 6 | `C11` | 0.1934 |
| 7 | `C2` | 0.1847 |
| 8 | **`entity_card_amt_mean_hist`** (engineered) | 0.1835 |
| 9 | `card1` | 0.1770 |
| 10 | `P_emaildomain` | 0.1715 |

Ten of the top 30 are engineered here. Mean |SHAP| is preferred over LightGBM's
split-count importance because it is in units of model output and is consistent
between the global ranking and the per-transaction explanation the API returns —
the same number explains both.

⚠️ See [Limitations](#limitations) on the `*_anchored` features: their high
ranking is real but does **not** imply they generalise.

## FastAPI

Three routes (`api/routes.py`). The model bundle is loaded **once** at startup and
reused for every request.

| endpoint | purpose |
|---|---|
| `GET /health` | Liveness plus model provenance: name, trained-at, feature count, calibration status, threshold, holdout metrics. Returns 200 with `status="degraded"` when no model is loaded — "up but modelless" and "down" are different facts. |
| `POST /predict` | Calibrated probability, risk band, threshold, latency, feature-completeness count. |
| `POST /explain` | Prediction plus ranked SHAP contributors (`?top_n=`). |

Provenance is folded into `/health` rather than given a separate `/model-info`
route, and batch scoring was removed — endpoints that exist to look complete are
not endpoints.

**Input contract.** Requiring all ~430 raw columns would make the API unusable, so
the schema names the high-signal fields and accepts the long tail through one
`extra_features` map. Anything omitted becomes NaN — a genuine capability, since
the model is a LightGBM trained on data that is 43% missing across the V block.
`extra="forbid"` catches typo'd fields, validation failures return a structured
422, and a missing model yields 503 rather than 500.

## Docker

```bash
docker build -t fraud-api .
```

```bash
docker run -p 8000:8000 -v "$(pwd)/models:/app/models:ro" fraud-api
```

Multi-stage build: wheels compile in a builder stage so gcc stays out of the
shipped image; `libgomp1` is installed for LightGBM's OpenMP threading; the
service runs as a non-root user (uid 10001) with a `HEALTHCHECK` on `/health`.

The model artifact is **bind-mounted, not baked in** — otherwise every retrain
forces an image rebuild, and image contents depend on training output. Mounted
read-only, because a service must never modify its own model.

There is no `docker-compose.yml`: with no database or cache to orchestrate, a
single container is the whole system.

## Drift Monitoring

```bash
python scripts/monitor.py --current test
```

> The traffic is not production traffic — this project has no users, and the
> report says so in its own payload. What *is* real is the distribution shift: the
> comparison is the training period against the **real, unlabeled IEEE-CIS test
> period beginning 30 days later**, not a synthetic perturbation.

**PSI** is the trigger (interpretable on a fixed scale, insensitive to sample
size); **KS** is reported alongside (catches shape changes PSI's binning smooths
over, but its p-value goes to zero for any difference once n is large).
Thresholds are stated rather than implied: **< 0.10 stable, 0.10–0.25 moderate,
&gt; 0.25 significant** (`configs/config.yaml`). Bin edges come from the reference
distribution and are reused, since re-binning per window would compare two
different binnings. Per-feature missing-rate deltas fall out of the same pass.

Measured over 507 features, 60,000 rows per period:

| result | value |
|---|---|
| Significantly drifted (PSI > 0.25) | **181** |
| Moderately drifted | 6 |
| Stable | 320 |
| **Prediction drift (model output)** | **PSI 0.0329 — stable** |
| Mean predicted probability | 0.0435 → 0.0312 |

**Heavy input drift, stable output.** The drifted inputs are dominated by device
and identity metadata — `id_31` browser strings (PSI 13.21), `id_30` OS (12.26),
`id_33` resolution (11.92) — which change naturally as browsers and handsets
update over a month. The model does not lean on them enough for its output to
move. Score drift is the only signal available without labels, which is why it is
computed.

### The finding that changed a design decision

Monitoring caught a real defect in this project's **own** feature engineering.

All 15 `D*_anchored` features drift significantly, while the raw `D` columns they
derive from are stable:

| feature | PSI | KS |
|---|---|---|
| `D9_anchored` | 12.447 | **1.000** |
| `D13_anchored` | 5.499 | 0.965 |
| all 15 anchored | 3.5 – 12.4 | 0.84 – 1.00 |
| `D15` (raw) | 0.108 | 0.140 |
| `D1` (raw) | 0.0069 | 0.041 |

A KS statistic of **1.000** means the distributions do not overlap at all.

The cause is my own transformation: `D_n_anchored = day_index − D_n` was meant to
turn a moving delta into a fixed calendar anchor, but `day_index` is **absolute
time**, and the test period sits at days 213–396 against training's 1–182. It
reintroduced through the back door exactly the risk the audit had documented — and
`D1_anchored` still ranks 3rd by SHAP, because the holdout is adjacent to training
and hides the problem.

That a monitoring system built for this project found a genuine flaw in the
project's own features, rather than reporting a reassuring all-clear, is the
strongest evidence that the monitoring is real.

## Testing / CI

```bash
pytest -q
```

**88 tests, all passing.** The suite runs **without the dataset**, on
schema-faithful synthetic fixtures (`tests/conftest.py`) reproducing the real
column families, dtypes, missingness patterns, chronological ordering with ties,
and ~3.5% prevalence with genuine signal. That is deliberate: CI exercises real
code paths — including training a small LightGBM and scoring it through the API —
on a machine with no access to a 1.3 GB Kaggle download.

| file | focus |
|---|---|
| `tests/test_data_validation.py` | schema invariants, temporal split, purged folds, tie handling |
| `tests/test_features.py` | feature builders, **velocity causality**, encoder leakage safety |
| `tests/test_evaluation.py` | metrics, alert budgets, calibration, comparison table |
| `tests/test_monitoring.py` | PSI, KS, drift verdicts |
| `tests/test_api.py` | all three endpoints, invalid input, cold-start velocity, degraded mode |

Several tests exist because they caught real bugs: a nullable-string comparison
that crashed email features; predictions misaligned with request order after the
frame is sorted chronologically; and an over-strict `card1` bound that rejected
the real test split, where `card1` reaches 18,397 against training's 18,396.

**GitHub Actions** (`.github/workflows/ci.yml`): `lint → test → docker build → API
smoke test`. The smoke test starts the image **without a model** and asserts
`/health` reports degraded, all three routes appear in the OpenAPI spec,
`/predict` returns **503** (not 500), and an invalid body returns **422**.

## Project Architecture

```
IEEE-CIS CSVs
     ↓  src/data/loading.py      streamed CSV→Parquet, id-NN→id_NN, LEFT JOIN
     ↓  src/data/validation.py   fail-loud schema + chronology invariants
     ↓  src/features/            prepare → fit → transform (the leakage boundary)
     ↓  src/data/splitting.py    20% chronological holdout + purged folds
     ↓  src/models/              LogReg · RandomForest · LightGBM + Optuna
     ↓  src/evaluation/          PR-AUC, ROC-AUC, alert budgets, calibration
     ↓  src/explainability/      SHAP global + per-transaction
     ↓  models/model_artifact.pkl   model + pipeline + calibrator + threshold
     ↓  api/                     FastAPI: /health /predict /explain
     ↓  Dockerfile               single container
     ↓  src/monitoring/drift.py  PSI + KS vs the real test period
```

```
├── configs/config.yaml          split, features, training, serving, monitoring
├── data/README.md               how to obtain the dataset (never committed)
├── docs/                        01_dataset_audit.md · 02_leakage_analysis.md
├── notebooks/01_eda.ipynb       analysis only; imports from src/
├── src/
│   ├── data/                    schema, loading, validation, preprocessing, splitting
│   ├── features/                builders, velocity, aggregations, pipeline
│   ├── models/                  estimators, training, tuning, artifact
│   ├── evaluation/              metrics, calibration, compare, plots
│   ├── explainability/          shap_explainer
│   ├── monitoring/              drift
│   └── utils/                   paths, config, logging, seed
├── api/                         main, routes, schemas, dependencies, settings
├── scripts/                     inspect_dataset, build_dataset, train, evaluate, monitor
├── tests/                       5 modules, 88 tests
├── Dockerfile · requirements.txt · pyproject.toml
└── .github/workflows/ci.yml
```

Every path derives from the repository root via `src/utils/paths.py` (overridable
with `FRAUD_PROJECT_ROOT`, which is how the container points at `/app`). There are
no absolute paths in the codebase.

## How to Run

**Prerequisites:** Python 3.12, ~4 GB free RAM for training, ~2 GB disk.

```bash
pip install -r requirements-dev.txt
```

Download the four CSVs from the [competition page](https://www.kaggle.com/c/ieee-fraud-detection/data)
into `data/raw/` (see [`data/README.md`](data/README.md); raw data is gitignored).

```bash
python scripts/inspect_dataset.py
```

```bash
python scripts/build_dataset.py --with-test
```

```bash
python scripts/train.py --random-cv-control
```

```bash
python scripts/evaluate.py --partition holdout
```

```bash
uvicorn api.main:app --reload --port 8000
```

Then open http://localhost:8000/docs, or:

```bash
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"transaction_amt\": 149.99, \"product_cd\": \"W\", \"card1\": 13926, \"card4\": \"visa\", \"p_emaildomain\": \"gmail.com\"}"
```

```bash
python scripts/monitor.py --current test
```

Experiment tracking writes to a local `./mlruns` store; `mlflow ui` shows the runs.
No MLflow server is required.

## Limitations

1. **The `D*_anchored` features are a known defect in the shipped model.** All 15
   drift severely against the true test period (KS up to 1.000) because the anchor
   derives from absolute time. They rank highly by SHAP and the holdout metrics
   include them, because the holdout is adjacent in time to training and hides the
   problem. **Holdout performance should be read as optimistic for a
   30-day-forward deployment.** Documented rather than quietly patched, because a
   measurement — not a guess — identified it.

2. **Velocity features are cold-started at serving time.** The API keeps no
   transaction history, so trailing-window counts are computed from the request
   alone: a single transaction correctly yields 0, meaning "no prior activity known
   to this service". Honest, but weaker than training, where full history was
   available. A production deployment needs a feature store.

3. **No production traffic.** Drift monitoring uses real dataset periods, not live
   users; there is no label feedback loop and no alerting integration.

4. **The Kaggle test set has no labels**, so the final metric is a chronological
   holdout from the training period — an honest near-future estimate, not a
   30-day-forward measurement.

5. **Entity keys are a proxy.** No account identifier exists, so
   `card1 + addr1 + card2` approximates one. Cards sharing those values merge; a
   card whose `addr1` changes splits.

6. **Dense baselines are subsampled** to 150,000 rows, recorded in
   `model_comparison.csv` rather than hidden. The comparison is not perfectly
   equal, and saying so is the point.

7. **Bounded hyperparameter search** — 5 Optuna trials under a 1800 s cap; the
   timeout stopped it, not convergence.

8. **Single-node, single-worker.** No horizontal scaling, A/B routing, or shadow
   deployment.

9. **Docker was not built in the development environment**, because Docker is not
   installed there. The Dockerfile is written and the CI workflow builds and
   smoke-tests it, but the build has not been executed locally.

## Future Improvements

Roughly in order of value per unit of effort:

1. **Remove or reformulate `D*_anchored` and retrain.** They measurably harm
   temporal generalisation. Either drop them (the raw `D` columns are stable and
   already present) or re-express the anchor relative to the transaction rather
   than to an absolute day index. The comparison must be rerun, since this changes
   the feature set behind the reported metrics.
2. **A feature store for velocity**, so serving and training share one definition
   and cold starts do not degrade the first requests.
3. **Adversarial validation to prune shift-heavy features** — 181 of 507 already
   drift significantly.
4. **Cost-sensitive thresholding**: minimise expected monetary loss given
   chargeback and review costs, rather than optimising F1, which weights precision
   and recall equally as no fraud team does.
5. **Revisit the 339 V columns on SHAP evidence** — kept deliberately for the
   baseline; correlation-clustering would shrink inference cost if PR-AUC holds.
6. **Scheduled retraining** gated on holdout PR-AUC and drift.

---

## License and data use

Code is provided for portfolio and educational purposes. The IEEE-CIS dataset is
subject to Kaggle competition rules and is not redistributed here.
