"""Generate the pipeline / code-walkthrough companion PDF.

Shares the styling helpers of build_prep_pdf.py by importing them, so the two
documents stay visually consistent. Reportlab only; no arrows (U+2192) or
U+2212, neither of which exists in Helvetica's WinAnsi encoding.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Reuse the styling layer from the prep document without re-executing its story.
_src = Path("build_prep_pdf.py").read_text(encoding="utf-8")
_head = _src.split("# ============================== COVER ==============================")[0]
_mod = {}
exec(compile(_head, "prep_styles", "exec"), _mod)

P, H1, H2, H3, SP = _mod["P"], _mod["H1"], _mod["H2"], _mod["H3"], _mod["SP"]
bullets, callout, table = _mod["bullets"], _mod["callout"], _mod["table"]
mm, A4 = _mod["mm"], _mod["A4"]
colors, MUTED, RULE = _mod["colors"], _mod["MUTED"], _mod["RULE"]
BaseDocTemplate = _mod["BaseDocTemplate"]
Frame, PageTemplate, PageBreak = _mod["Frame"], _mod["PageTemplate"], _mod["PageBreak"]

OUT = "Fraud_Detection_MLOps_Pipeline_Walkthrough.pdf"
story: list = []

# ============================== COVER ==============================
story += [
    SP(34),
    P("Pipeline &amp; Code Walkthrough", "title"),
    P("What runs, in what order, from which file, and why", "sub"),
    SP(10),
]
story += callout("What this document is",
    "The companion to the interview-prep PDF. That one covers <b>reasoning</b>; this one covers "
    "<b>mechanics</b> &mdash; the five commands that produce everything, which module each one calls, "
    "what lands on disk at each stage, and why the boundaries fall where they do.<br/><br/>"
    "Every path, function name and artifact listed here was read out of the repository.", "key")

story += table([
    ["Command", "Reads", "Writes", "Runtime"],
    ["<font face='Courier' size='7.5'>scripts/inspect_dataset.py</font>", "4 raw CSVs",
     "dataset_audit.json + .md", "~3 min"],
    ["<font face='Courier' size='7.5'>scripts/build_dataset.py</font>", "4 raw CSVs",
     "interim + processed parquet, folds, split", "~25 min"],
    ["<font face='Courier' size='7.5'>scripts/train.py</font>", "processed parquet + folds",
     "model_artifact.pkl, comparison, SHAP", "~50 min"],
    ["<font face='Courier' size='7.5'>scripts/evaluate.py</font>", "artifact + holdout",
     "evaluation_holdout.json, 5 figures", "~2 min"],
    ["<font face='Courier' size='7.5'>scripts/monitor.py</font>", "artifact + both periods",
     "feature_drift.csv, monitoring_summary.json", "~1 min"],
], [56 * mm, 34 * mm, 52 * mm, 21 * mm])

story.append(P("Runtimes are from this machine and are dominated by memory pressure rather than "
                "compute; see Step 3.", "small"))

story.append(PageBreak())

# ============================== LAYOUT ==============================
story.append(H1("Repository layout &mdash; what lives where"))
story.append(P("Eight packages under <font face='Courier' size='8'>src/</font>, five entry-point "
                "scripts, an API package and a test suite. The rule throughout: "
                "<b>scripts orchestrate, src implements, nothing in src knows about the filesystem "
                "except through <font face='Courier' size='8'>utils/paths.py</font></b>."))

story += table([
    ["Path", "Lines", "Responsibility"],
    ["<b>scripts/</b>", "1,735", "<b>Entry points.</b> Each is a thin main() that wires src modules together"],
    ["&nbsp;&nbsp;inspect_dataset.py", "446", "EDA and the pre-implementation audit"],
    ["&nbsp;&nbsp;build_dataset.py", "328", "Join, validate, split, compute velocity, prepare partitions"],
    ["&nbsp;&nbsp;train.py", "544", "CV, tuning, calibration, final fit, SHAP, holdout, artifact"],
    ["&nbsp;&nbsp;evaluate.py", "140", "Score a partition, bootstrap CIs, write figures"],
    ["&nbsp;&nbsp;monitor.py", "159", "Feature and prediction drift"],
    ["&nbsp;&nbsp;run_ablation.py", "118", "One fold per process &mdash; the memory workaround"],
    ["<b>src/data/</b>", "1,162", "<b>Loading and splitting.</b> schema, loading, validation, splitting, preprocessing"],
    ["<b>src/features/</b>", "992", "<b>Feature engineering.</b> builders, velocity, aggregations, pipeline"],
    ["<b>src/models/</b>", "1,110", "<b>Modelling.</b> estimators, training, tuning, artifact"],
    ["<b>src/evaluation/</b>", "706", "<b>Metrics.</b> metrics, calibration, plots, compare"],
    ["<b>src/explainability/</b>", "186", "SHAP wrapper (TreeExplainer)"],
    ["<b>src/monitoring/</b>", "225", "PSI, KS, drift summaries"],
    ["<b>src/utils/</b>", "228", "config, paths, logging, seed &mdash; no ML logic"],
    ["<b>api/</b>", "621", "FastAPI app: main, routes, schemas, dependencies, settings"],
    ["<b>tests/</b>", "1,219", "104 tests on synthetic fixtures (conftest.py is 314 lines of fixtures)"],
], [45 * mm, 16 * mm, 102 * mm])

story += callout("Why the tests are bigger than most of the src packages",
    "<font face='Courier' size='8'>tests/conftest.py</font> is 314 lines because CI has no dataset "
    "&mdash; the real data is 1.3 GB and under Kaggle competition terms. The fixtures build a "
    "<b>schema-faithful synthetic frame</b> instead, which is what lets CI train a real (small) "
    "LightGBM model and score it through the actual API on every push. Tests that need no data are "
    "cheap; tests that exercise the real code path are worth the fixture cost.", "good")

story.append(PageBreak())

# ============================== STEP 1 ==============================
story.append(H1("Step 1 &mdash; EDA and the dataset audit"))
story.append(P("<font face='Courier' size='8.5'>python scripts/inspect_dataset.py</font>", "code"))

story.append(H2("Why this exists as a script, not a notebook"))
story += bullets([
    "It runs in CI-able, re-runnable form &mdash; a notebook's output is a screenshot of a state you "
    "cannot reproduce.",
    "It writes <b>machine-readable</b> output (<font face='Courier' size='8'>dataset_audit.json</font>) "
    "alongside human-readable (<font face='Courier' size='8'>dataset_audit.md</font>), so later stages "
    "can assert against audit facts instead of re-deriving them.",
    "The whole project's premise is that facts must be measured. Starting with an unversioned notebook "
    "would contradict that on page one.",
])

story.append(H2("What it computes, and why each answer mattered"))
story += table([
    ["Function", "What it answers", "Why the answer changed a decision"],
    ["<font face='Courier' size='7.5'>profile_csv()</font>",
     "Per-column dtype, null rate, cardinality, target rate. Chunked so a 1.3 GB CSV never fully loads",
     "Found the V block at ~43% missing &mdash; which is why NaN routing, not imputation, drives model choice"],
    ["<font face='Courier' size='7.5'>analyse_join()</font>",
     "How transaction and identity relate on TransactionID",
     "Identity covers only ~24% of rows. <b>Forced LEFT JOIN</b> &mdash; an INNER JOIN would have "
     "silently discarded ~76% of the data"],
    ["<font face='Courier' size='7.5'>temporal_summary()</font>",
     "Is TransactionDT usable as a time axis?",
     "<b>This is the gate for the entire project.</b> Non-null, non-negative, already monotonic in file "
     "order, 182 contiguous days, no missing day, and <b>zero rows on the 80th-percentile cut</b>"],
    ["<font face='Courier' size='7.5'>flagged_columns()</font>",
     "Columns above a 90% missing threshold, and constants",
     "Produced the do-not-use list, including V107 (constant in test but not train)"],
], [30 * mm, 55 * mm, 78 * mm])

story += callout("The single most important thing this step produced",
    "<b>corr(TransactionID, TransactionDT) = 0.99828.</b><br/><br/>"
    "The row id is a disguised timestamp. Feeding it to the model would have let a tree split on "
    "&quot;transaction number greater than X&quot; and learn time directly &mdash; and it would have "
    "looked like a great feature in random CV. It is hard-excluded in "
    "<font face='Courier' size='8'>src/data/schema.py</font>.<br/><br/>"
    "This is the answer to <i>&quot;what did EDA actually buy you?&quot;</i> Not a histogram &mdash; a "
    "leakage vector caught before a single model was trained.", "key")

story.append(H2("The rule this step established"))
story.append(P("The audit was written <b>before</b> any implementation, and the decisions it forced "
                "&mdash; LEFT JOIN, exclude TransactionID, chronological split is feasible, keep all "
                "339 V columns for the baseline &mdash; are cited by later modules rather than "
                "re-argued. When an interviewer asks &quot;how did you decide X?&quot;, the answer is "
                "a measurement in "
                "<font face='Courier' size='8'>reports/dataset_audit.json</font>, not a preference."))

story.append(PageBreak())

# ============================== STEP 2 ==============================
story.append(H1("Step 2 &mdash; Building the dataset"))
story.append(P("<font face='Courier' size='8.5'>python scripts/build_dataset.py</font>", "code"))
story.append(P("Four stages. This is the only script that touches raw CSVs, and it is where every "
                "leakage boundary is physically established."))

story += table([
    ["Stage", "Calls", "Produces", "The decision embedded here"],
    ["<b>1. Join + validate</b>",
     "<font face='Courier' size='7.5'>loading.build_interim()</font>, "
     "<font face='Courier' size='7.5'>validation.validate_frame()</font>",
     "<font face='Courier' size='7.5'>data/interim/train_joined.parquet</font>",
     "LEFT JOIN on TransactionID. Streamed CSV to Parquet via ParquetWriter so the 1.3 GB never fully "
     "materialises. Schema and temporal-order assertions run here and on every later load"],
    ["<b>2. Global velocity</b>",
     "<font face='Courier' size='7.5'>velocity.compute_velocity_frame()</font>",
     "<font face='Courier' size='7.5'>data/interim/train_velocity.parquet</font>",
     "Computed <b>once over the whole timeline</b>, because a 168-hour look-back needs continuous "
     "history. Only ~30 columns, so it is cached and joined per partition"],
    ["<b>3. Split + folds</b>",
     "<font face='Courier' size='7.5'>splitting.temporal_split()</font>, "
     "<font face='Courier' size='7.5'>save_split()</font>, "
     "<font face='Courier' size='7.5'>save_folds()</font>",
     "<font face='Courier' size='7.5'>split_indices.npz</font>, "
     "<font face='Courier' size='7.5'>folds_temporal.npz</font>, "
     "<font face='Courier' size='7.5'>folds_random.npz</font>",
     "Cut on <b>timestamp edges</b> so a tie group never straddles partitions. Folds are persisted so "
     "every model is scored on byte-identical splits. The random folds exist <i>only</i> as the control "
     "experiment"],
    ["<b>4. Prepare partitions</b>",
     "<font face='Courier' size='7.5'>FeaturePipeline.prepare()</font> per chunk",
     "<font face='Courier' size='7.5'>modelling_prepared.parquet</font>, "
     "<font face='Courier' size='7.5'>holdout_prepared.parquet</font>, "
     "<font face='Courier' size='7.5'>test_prepared.parquet</font>",
     "Only <b>prepare</b> runs here &mdash; row-local and past-only features, nothing fitted. Chunked "
     "because the full frame is ~970 MB"],
], [26 * mm, 38 * mm, 42 * mm, 57 * mm])

story += callout("Why velocity is computed globally but encoders are not",
    "This looks inconsistent until you see the distinction. <b>Velocity is causal by construction</b> "
    "&mdash; an offset-band searchsorted can only ever count rows strictly earlier in time, so "
    "computing it over the whole timeline cannot pull the future into a row. It is a per-row lookup "
    "backwards.<br/><br/>"
    "<b>Encoders are population statistics.</b> A frequency count over the whole timeline genuinely "
    "does contain future information. So velocity is global and cached; encoders are fitted inside "
    "every fold.<br/><br/>"
    "The purge gap exists because velocity is global: a 168-hour window at the <i>start</i> of a "
    "validation block would reach back into training rows, even though the rows are disjoint. Hence "
    "7 days.", "key")

story.append(H2("What ends up on disk"))
story += table([
    ["File", "Contents"],
    ["<font face='Courier' size='7.5'>interim/train_joined.parquet</font>", "590,540 x 434, joined and validated"],
    ["<font face='Courier' size='7.5'>interim/train_velocity.parquet</font>", "~30 velocity columns indexed by TransactionID"],
    ["<font face='Courier' size='7.5'>processed/modelling_prepared.parquet</font>", "472,432 x 514 &mdash; days 1-141, 3.5135% fraud"],
    ["<font face='Courier' size='7.5'>processed/holdout_prepared.parquet</font>", "118,108 x 514 &mdash; days 141-182, 3.4409% fraud"],
    ["<font face='Courier' size='7.5'>processed/test_prepared.parquet</font>", "506,691 x 513 &mdash; unlabelled Kaggle test"],
    ["<font face='Courier' size='7.5'>processed/folds_temporal.npz</font>", "5 purged forward-chaining index pairs"],
    ["<font face='Courier' size='7.5'>processed/folds_random.npz</font>", "5 stratified random pairs &mdash; the control"],
    ["<font face='Courier' size='7.5'>processed/split_metadata.json</font>", "Row counts, day ranges, fraud rates per partition"],
], [66 * mm, 97 * mm])

story.append(PageBreak())

# ============================== FEATURE PIPELINE ==============================
story.append(H1("The FeaturePipeline &mdash; the leakage boundary itself"))
story.append(P("<font face='Courier' size='8.5'>src/features/pipeline.py</font> (255 lines). This is "
                "the most important file in the repository. Three methods, and what separates them is "
                "<b>what each is permitted to know</b>."))

story += table([
    ["Method", "Runs when", "May see", "May learn", "Calls"],
    ["<b>prepare(df)</b>", "Dataset build (Step 2)", "Current row + strictly earlier rows",
     "<b>Nothing</b>",
     "<font face='Courier' size='7'>build_stateless_features()</font> then "
     "<font face='Courier' size='7'>add_velocity_features()</font> or a cached velocity join"],
    ["<b>fit(train_df)</b>", "Inside each CV fold, and once for the final model",
     "The training partition only", "Population statistics",
     "<font face='Courier' size='7'>FrequencyEncoder.fit()</font>, "
     "<font face='Courier' size='7'>EntityAmountAggregator.fit()</font>, "
     "<font face='Courier' size='7'>v_nan_groups()</font>, "
     "<font face='Courier' size='7'>CategoricalCodeEncoder.fit()</font>"],
    ["<b>transform(df)</b>", "Every scoring path, including the API", "Any frame",
     "<b>Nothing</b> &mdash; pure lookup",
     "<font face='Courier' size='7'>_apply_fitted()</font>, then column selection"],
], [22 * mm, 32 * mm, 32 * mm, 24 * mm, 53 * mm])

story.append(H2("What build_stateless_features() chains, in order"))
story += table([
    ["Builder", "Adds", "Why the order matters"],
    ["<font face='Courier' size='7.5'>add_time_features()</font>", "_day_index, hour, day-of-week",
     "<b>Must be first</b> &mdash; three later builders need _day_index"],
    ["<font face='Courier' size='7.5'>add_amount_features()</font>", "log amount, cents, amount bands", "Independent"],
    ["<font face='Courier' size='7.5'>add_email_features()</font>", "domain, suffix, P/R match flag", "Independent"],
    ["<font face='Courier' size='7.5'>add_device_features()</font>", "os_family, browser, screen dims", "Independent"],
    ["<font face='Courier' size='7.5'>add_missingness_features()</font>", "Null-pattern indicators",
     "Missingness is signal here, not noise"],
    ["<font face='Courier' size='7.5'>add_anchored_d_features()</font>", "15 D*_anchored columns",
     "<b>Computed but excluded from the model</b> &mdash; kept so drift monitoring can keep measuring them"],
    ["<font face='Courier' size='7.5'>add_entity_keys()</font>", "3 integer entity keys",
     "Deterministic integer arithmetic, not factorize codes, so keys are stable across partitions"],
    ["<font face='Courier' size='7.5'>add_uid_entity_key()</font>", "_entity_uid",
     "Needs _day_index and D1. Hashed deterministically; 194,519 groups"],
], [46 * mm, 41 * mm, 76 * mm])

story += callout("Why an integer key rather than pandas factorize",
    "<font face='Courier' size='8'>factorize()</font> assigns codes by <b>order of appearance</b>, so "
    "the same card gets a different code in the training partition than in the holdout. Every "
    "cross-partition lookup would silently mismatch.<br/><br/>"
    "The keys are therefore built by positional arithmetic &mdash; "
    "<font face='Courier' size='8'>card1 * 1e6 + addr1 * 1e3 + card2</font> &mdash; with an assertion "
    "that addr1 and card2 stay below 1000 so they cannot overflow into the next slot and merge two "
    "different entities. The uid uses a deterministic hash for the same reason.", "key")

story.append(PageBreak())

# ============================== STEP 3 ==============================
story.append(H1("Step 3 &mdash; Training"))
story.append(P("<font face='Courier' size='8.5'>python scripts/train.py --models lightgbm "
                "--random-cv-control</font>", "code"))
story.append(P("Nine phases inside one MLflow run. Line numbers are from "
                "<font face='Courier' size='8'>scripts/train.py</font>."))

story += table([
    ["Phase", "Line", "What happens", "Why it is here and not elsewhere"],
    ["1. Load", "~210", "modelling + holdout parquet, persisted folds",
     "Folds are <b>loaded, never recomputed</b>, so every model shares byte-identical splits"],
    ["2. CV per model", "229", "<font face='Courier' size='7'>train_cv()</font> over the 5 temporal folds",
     "Encoders refit per fold inside <font face='Courier' size='7'>_fit_one()</font>"],
    ["3. Random-CV control", "246", "The same model on <font face='Courier' size='7'>folds_random</font>",
     "<b>Optional flag.</b> Produces the +0.2929 headline. Never used for selection"],
    ["4. Select winner", "273", "argmax of mean CV PR-AUC", "Selection on the primary metric only"],
    ["5. Tune", "286", "<font face='Courier' size='7'>tune_model()</font> &mdash; Optuna on the last 2 folds",
     "Only the winner is tuned. Early folds train on as little as 46k rows and are least representative"],
    ["6. Calibrate + threshold", "343", "Isotonic on the last fold's validation slice",
     "Never on training (overconfident) and never on the holdout (would spend it)"],
    ["7. Final fit", "404", "<font face='Courier' size='7'>train_final()</font> on all 472,432 rows",
     "Early stopping uses an <b>inner</b> temporal tail, so the holdout stays untouched"],
    ["8. SHAP", "449", "TreeExplainer, global importance + plots", "Only for tree models"],
    ["9. Holdout", "475", "<b>Scored exactly once</b>, threshold applied unchanged",
     "Re-optimising the threshold here would turn the holdout into a validation set"],
    ["10. Persist", "511", "Artifact, comparison CSV, MLflow", "Artifact bundles four objects &mdash; see below"],
], [30 * mm, 12 * mm, 47 * mm, 74 * mm])

story.append(H2("Inside train_cv() &mdash; the loop that matters"))
story.append(P("<font face='Courier' size='8'>src/models/training.py</font>. For each of the 5 folds:"))
story += bullets([
    "<b>Slice</b> the prepared frame by the persisted fold indices.",
    "<b>Fit a fresh FeaturePipeline</b> on that fold's training rows only &mdash; this is the "
    "per-fold refit, and it is why the log prints "
    "<font face='Courier' size='8'>FrequencyEncoder fitted on 15 columns</font> five times with "
    "growing group counts.",
    "<b>Branch on model family.</b> Dense models (LogReg, RF) go through "
    "<font face='Courier' size='8'>LinearPreprocessor</font> for imputation, scaling and one-hot, and "
    "are subsampled to 100,000 rows. Boosters get the frame directly, downcast to float32.",
    "<b>Fit with early stopping</b> against an inner temporal tail &mdash; never the outer validation fold.",
    "<b>Predict</b>, store out-of-fold probabilities, compute metrics, free memory, next fold.",
])

story += callout("The float32 downcast &mdash; why it is correct, not a hack",
    "Boosted trees <b>bin</b> their inputs before choosing a split, so float64 carries more precision "
    "than the histogram can use. The extra 4 bytes per value buy nothing and cost 1.29 GB on the "
    "largest fold.<br/><br/>"
    "It was verified rather than assumed: the same fold before and after produced "
    "<b>identical results to six decimal places</b> across PR-AUC, ROC-AUC, precision, recall and F1. "
    "It is scoped to the booster branch only, because the dense path already emits float32.", "good")

story += callout("run_ablation.py &mdash; and why it exists",
    "The 5-fold loop holds the modelling frame plus per-fold copies for the whole run. This machine's "
    "commit limit fell from 31.3 GB to 24.5 GB mid-project, and the loop stopped fitting.<br/><br/>"
    "<font face='Courier' size='8'>scripts/run_ablation.py</font> scores <b>one fold per process</b>, "
    "so every allocation returns to the OS between folds, and appends the result to a CSV. Paired with "
    "<font face='Courier' size='8'>train.py --skip-cv --oof-npz</font> it can produce a full shipped "
    "model in small pieces &mdash; only the last fold's validation slice is needed for calibration and "
    "thresholding, so one isolated fold replaces the whole loop.<br/><br/>"
    "This is worth mentioning in an interview: it is a real engineering constraint solved without "
    "weakening the method.", "warn")

story.append(PageBreak())

# ============================== ARTIFACT ==============================
story.append(H1("The artifact &mdash; the training/serving boundary"))
story.append(P("<font face='Courier' size='8.5'>src/models/artifact.py</font> &mdash; "
                "<font face='Courier' size='8'>models/model_artifact.pkl</font>, 16.4 MB."))

story += table([
    ["Bundled object", "Its job", "What breaks if it goes stale"],
    ["<b>model</b>", "The fitted LightGBM booster", "&mdash;"],
    ["<b>feature_pipeline</b>", "All fitted encoders + the exact column order",
     "A retrained model with an old pipeline sees columns in a different order. Tree models score that "
     "<b>silently and wrongly</b> &mdash; no exception is raised"],
    ["<b>calibrator</b>", "Isotonic mapping to real probabilities",
     "Raw scores map through an old distribution, so every risk band shifts quietly"],
    ["<b>decision_threshold</b>", "0.2988, chosen on validation",
     "The precision/recall balance the business signed off on changes without anyone deciding to change it"],
], [32 * mm, 45 * mm, 86 * mm])

story += callout("The one-line version for an interview",
    "&quot;They are saved together because they are one object logically &mdash; the model's inputs are "
    "defined by the pipeline, its outputs only mean anything after the calibrator, and its decisions "
    "only mean anything at the threshold. Every way they can drift apart fails <b>silently</b>, which "
    "is what makes training/serving skew the most common serious production failure in ML.&quot;", "key")

story.append(H2("Steps 4 and 5 &mdash; evaluation and monitoring"))
story += table([
    ["Command", "Chain", "Output"],
    ["<font face='Courier' size='7.5'>evaluate.py --partition holdout</font>",
     "<font face='Courier' size='7'>ModelArtifact.load()</font> then "
     "<font face='Courier' size='7'>predict_proba()</font> then "
     "<font face='Courier' size='7'>compute_metrics()</font> + "
     "<font face='Courier' size='7'>bootstrap_metric_ci()</font> + "
     "<font face='Courier' size='7'>all_evaluation_plots()</font>",
     "<font face='Courier' size='7'>evaluation_holdout.json</font> and 5 PNGs. The bootstrap is "
     "<b>stratified by class</b> so prevalence is fixed across draws &mdash; pooled resampling would "
     "inflate the interval with an artefact, since PR-AUC moves with prevalence"],
    ["<font face='Courier' size='7.5'>monitor.py</font>",
     "<font face='Courier' size='7'>feature_drift()</font> + "
     "<font face='Courier' size='7'>prediction_drift()</font> + "
     "<font face='Courier' size='7'>summarise()</font>, on 60,000-row samples from each period",
     "<font face='Courier' size='7'>feature_drift.csv</font>, "
     "<font face='Courier' size='7'>monitoring_summary.json</font>. Compares 507 features: 181 "
     "significant, 6 moderate, 320 stable. This is the script that caught the D*_anchored defect"],
], [40 * mm, 63 * mm, 60 * mm])

story.append(H2("The API request path"))
story.append(P("<font face='Courier' size='8'>api/main.py</font> loads the artifact <b>once at "
                "startup</b> into app state. Per request:"))
story += table([
    ["Order", "What happens", "Note"],
    ["1", "Pydantic validates the body against "
     "<font face='Courier' size='7'>schemas.py</font>", "Malformed payloads never reach the model"],
    ["2", "<font face='Courier' size='7'>Depends(require_artifact)</font> resolves",
     "<b>Runs before body validation</b> &mdash; which is why a missing artifact returns 503, not 422. "
     "A CI smoke test asserts exactly this"],
    ["3", "<font face='Courier' size='7'>build_prepared_frame()</font> sorts the batch chronologically "
     "and computes velocity from it", "Single transactions honestly report count = 0 rather than "
     "inventing history"],
    ["4", "<font face='Courier' size='7'>artifact.predict_proba()</font> then risk banding", "Pipeline, "
     "then model, then calibrator &mdash; all from the one bundle"],
    ["5", "Responses re-mapped via <font face='Courier' size='7'>original_positions</font>",
     "The frame was re-sorted for velocity, so output order must be restored. A regression test guards "
     "this after it was once wrong"],
], [12 * mm, 78 * mm, 73 * mm])

story.append(PageBreak())

# ============================== ORDER + WHY ==============================
story.append(H1("The whole thing, in one page"))
story += table([
    ["#", "Command", "What it establishes"],
    ["0", "<font face='Courier' size='7.5'>inspect_dataset.py</font>",
     "<b>Facts before code.</b> LEFT JOIN needed (24% identity coverage), TransactionID is a disguised "
     "clock (r = 0.99828), chronological split is feasible (182 contiguous days, 0 rows on the cut)"],
    ["1", "<font face='Courier' size='7.5'>build_dataset.py</font>",
     "<b>Boundaries made physical.</b> Join, validate, split on timestamp edges, persist folds, compute "
     "global causal velocity, run prepare() only"],
    ["2", "<font face='Courier' size='7.5'>train.py</font>",
     "<b>Fitting confined to folds.</b> CV, random-CV control, tuning, calibration, final fit, SHAP, "
     "holdout once, bundled artifact"],
    ["3", "<font face='Courier' size='7.5'>evaluate.py</font>",
     "<b>Uncertainty quantified.</b> Metrics, alert budgets, stratified bootstrap CIs, figures"],
    ["4", "<font face='Courier' size='7.5'>monitor.py</font>",
     "<b>Forward-looking check.</b> Feature and prediction drift &mdash; distribution, not accuracy"],
    ["5", "<font face='Courier' size='7.5'>uvicorn api.main:app</font>",
     "<b>Served.</b> /health, /predict, /explain from the one artifact"],
    ["6", "<font face='Courier' size='7.5'>kaggle/make_submission.py</font>",
     "Optional. Uncalibrated scores, train-only encoders &mdash; both costing rank deliberately"],
], [8 * mm, 42 * mm, 113 * mm])

story.append(H2("If you are asked to walk through the code"))
story += bullets([
    "Start at <font face='Courier' size='8'>src/features/pipeline.py</font>, not at the model. The "
    "three-phase split is the design; everything else is consequence.",
    "Then <font face='Courier' size='8'>src/models/training.py::_fit_one()</font> &mdash; it shows the "
    "per-fold refit and the dense-versus-booster branch in about 40 lines.",
    "Then <font face='Courier' size='8'>src/models/artifact.py</font> &mdash; the training/serving boundary.",
    "Mention <font face='Courier' size='8'>src/utils/config.py</font> if asked about configurability: "
    "no threshold, ratio or seed is hardcoded in src; config resolves defaults, then YAML, then "
    "environment variables.",
])

story += callout("The sentence that ties it together",
    "&quot;The pipeline is ordered by <b>what each stage is allowed to know</b>, not by what it "
    "computes. prepare() may look backwards but learn nothing. fit() may learn, but only from the "
    "training partition. transform() may run anywhere but has no mechanism to learn. That turns "
    "&lsquo;could this leak?&rsquo; into one call site to audit instead of five hundred columns to "
    "reason about.&quot;", "good")


# ============================== BUILD ==============================
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(23 * mm, 12 * mm, "Fraud Detection MLOps – Pipeline & Code Walkthrough")
    canvas.drawRightString(187 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(23 * mm, 15.5 * mm, 187 * mm, 15.5 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=23 * mm, rightMargin=24 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm,
                      title="Fraud Detection MLOps - Pipeline & Code Walkthrough",
                      author="sam200530")
frame = Frame(doc.leftMargin, doc.bottomMargin, 163 * mm,
              A4[1] - doc.topMargin - doc.bottomMargin, id="body")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_footer)])
doc.build(story)
print(f"wrote {OUT}")
