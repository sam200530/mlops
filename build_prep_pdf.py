"""Generate the interview-preparation PDF.

Reportlab only: weasyprint/wkhtmltopdf/pandoc are unavailable and the
environment must not be modified. Arrows (U+2192) are avoided because
Helvetica's WinAnsi encoding lacks the glyph and renders a black box.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUT = "Fraud_Detection_MLOps_Interview_Prep.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6470")
ACCENT = colors.HexColor("#0b5fa5")
WARN_BG = colors.HexColor("#fdf3e3")
WARN_ED = colors.HexColor("#d98324")
KEY_BG = colors.HexColor("#eaf2fb")
KEY_ED = colors.HexColor("#0b5fa5")
TRAP_BG = colors.HexColor("#fdeceb")
TRAP_ED = colors.HexColor("#c0392b")
GOOD_BG = colors.HexColor("#eaf6ec")
GOOD_ED = colors.HexColor("#2e7d४4") if False else colors.HexColor("#2e7d44")
GREY_BG = colors.HexColor("#f4f5f7")
RULE = colors.HexColor("#d5d9e0")

ss = getSampleStyleSheet()


def _s(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.4, leading=13.6, textColor=INK,
                alignment=TA_LEFT, spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "title": _s("title", fontName="Helvetica-Bold", fontSize=25, leading=30,
                textColor=ACCENT, spaceAfter=10),
    "sub": _s("sub", fontSize=12, leading=17, textColor=MUTED, spaceAfter=6),
    "h1": _s("h1", fontName="Helvetica-Bold", fontSize=16.5, leading=21,
             textColor=ACCENT, spaceBefore=16, spaceAfter=9),
    "h2": _s("h2", fontName="Helvetica-Bold", fontSize=12, leading=16,
             textColor=INK, spaceBefore=12, spaceAfter=6),
    "h3": _s("h3", fontName="Helvetica-BoldOblique", fontSize=10, leading=14,
             textColor=MUTED, spaceBefore=9, spaceAfter=4),
    "p": _s("p"),
    "small": _s("small", fontSize=8.4, leading=12, textColor=MUTED),
    "code": _s("code", fontName="Courier", fontSize=8.2, leading=11.6),
    "cell": _s("cell", fontSize=8.5, leading=11.8, spaceAfter=0),
    "cellb": _s("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11.8, spaceAfter=0),
    "q": _s("q", fontName="Helvetica-Bold", fontSize=10, leading=14,
            textColor=ACCENT, spaceBefore=9, spaceAfter=3),
}


def P(text, style="p"):
    return Paragraph(text, S[style])


def H1(text):
    return Paragraph(text, S["h1"])


def H2(text):
    return Paragraph(text, S["h2"])


def H3(text):
    return Paragraph(text, S["h3"])


def SP(h=4):
    return Spacer(1, h)


def bullets(items, style="p"):
    out = []
    for it in items:
        out.append(Table(
            [[Paragraph("&bull;", S[style]), Paragraph(it, S[style])]],
            colWidths=[5 * mm, 158 * mm],
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]),
        ))
    return out


def callout(title, body, kind="key"):
    bg, ed = {"key": (KEY_BG, KEY_ED), "warn": (WARN_BG, WARN_ED),
              "trap": (TRAP_BG, TRAP_ED), "good": (GOOD_BG, GOOD_ED)}[kind]
    inner = [Paragraph(f"<b>{title}</b>", S["cellb"])] if title else []
    inner.append(Paragraph(body, S["cell"]))
    t = Table([[inner]], colWidths=[163 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ed),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return [t, SP(7)]


def table(rows, widths, header=True, zebra=True, align_right=()):
    data = []
    for r_i, row in enumerate(rows):
        cells = []
        for c_i, cell in enumerate(row):
            st = "cellb" if (header and r_i == 0) else "cell"
            cells.append(Paragraph(str(cell), S[st]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), GREY_BG))
        cmds.append(("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT))
    if zebra:
        for i in range(1 if header else 0, len(data)):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafbfc")))
    for c in align_right:
        cmds.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    return [t, SP(8)]


def qa(question, short, deeper, trap):
    parts = [Paragraph(f"Q. {question}", S["q"]),
             Paragraph(f"<b>SHORT:</b> {short}", S["p"]),
             Paragraph(f"<b>DEEPER:</b> {deeper}", S["p"])]
    parts += callout("TRAP", trap, "trap")
    return KeepTogether(parts)


def weak_strong(weak, strong):
    return KeepTogether([
        Table([[Paragraph(f"<b>WEAK</b>&nbsp;&nbsp; {weak}", S["cell"])]],
              colWidths=[163 * mm],
              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), TRAP_BG),
                                ("LINEBEFORE", (0, 0), (0, -1), 2.2, TRAP_ED),
                                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                                ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])),
        Table([[Paragraph(f"<b>STRONG</b>&nbsp;&nbsp; {strong}", S["cell"])]],
              colWidths=[163 * mm],
              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), GOOD_BG),
                                ("LINEBEFORE", (0, 0), (0, -1), 2.2, GOOD_ED),
                                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                                ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])),
        SP(9),
    ])


story: list = []

# ============================== COVER ==============================
story += [
    SP(38),
    P("Fraud Detection MLOps", "title"),
    P("Interview Preparation &amp; Deep Understanding", "title"),
    SP(6),
    P("A defence manual for the IEEE-CIS leakage-safe fraud detection platform. "
      "Every figure in this document was read out of the repository, not recalled.", "sub"),
    SP(14),
]
story += table([
    ["Shipped model", "LightGBM, 547 features, untuned, isotonic-calibrated"],
    ["Holdout PR-AUC", "<b>0.5538</b> &nbsp; 95% CI [0.5384, 0.5688] &nbsp; 16.09x the 3.44% floor"],
    ["Temporal CV PR-AUC", "0.5591 +/- 0.0195 (5 purged forward-chaining folds)"],
    ["Holdout ROC-AUC", "0.8999 &nbsp; 95% CI [0.8943, 0.9057]"],
    ["Headline finding", "Random CV said 0.8512. Temporal CV said 0.5583. Optimism <b>+0.2929</b>."],
    ["Second finding", "Drift monitoring caught 15 self-built features at KS up to 1.000"],
    ["Tests / CI", "104 passing, ruff clean, green CI"],
], [38 * mm, 125 * mm], header=False, zebra=True)

story += callout(
    "How to use this document",
    "Part 1 is the story you tell. Parts 2-11 are the reasoning behind each decision. "
    "Part 12 is 50+ drilled questions. Part 13 is the hostile interviewer. Part 16 is the "
    "30-minutes-before cheat sheet. If you read only two things, read Part 1 and Part 16.", "key")

story += callout(
    "The rule that keeps you safe",
    "Never claim more than the repository proves. This project's credibility comes from having "
    "measured its own mistakes -- overclaiming throws that away. Where evidence is partial, this "
    "document says so, and so should you.", "warn")

story.append(PageBreak())

# ============================== PART 1 ==============================
story.append(H1("Part 1 &mdash; The Project Story"))
story.append(P("One story, four lengths. Same spine every time: <b>a number I trusted turned out "
                "to be wrong, and everything else followed from fixing that.</b>"))

story.append(H2("30-second explanation"))
story += callout("", 
    "\"It's a fraud detection system on the IEEE-CIS dataset, but the point isn't the model &mdash; "
    "it's that the project checks whether its own validation is honest. When I cross-validated "
    "the normal way, random 5-fold, I got 0.85 PR-AUC. When I switched to time-based validation, "
    "the same model on the same data scored 0.56. That 0.29 gap was pure self-deception, and "
    "everything in the repo exists to make sure the number I report is the real one. Final model "
    "is LightGBM at 0.554 holdout PR-AUC, about 16 times the fraud base rate, served behind "
    "FastAPI with drift monitoring.\"", "good")

story.append(H2("60-second explanation"))
story.append(P("Add the <i>why</i> of the gap and the second finding:"))
story += callout("",
    "\"...The reason random CV lies here is that the same card appears many times in the dataset. "
    "Shuffle the rows and a card's January and March transactions land in training while its "
    "February one lands in validation &mdash; so the model is interpolating between things it has "
    "already seen instead of predicting forward. Real deployment is always forward. So I built "
    "purged forward-chaining CV: train on the past, leave a 7-day gap, validate on the future.<br/><br/>"
    "Then the monitoring caught something I didn't expect. I'd engineered 15 features that anchored "
    "the D columns to a calendar date. SHAP ranked one of them third most important. But drift "
    "monitoring showed all 15 had a KS statistic up to 1.000 against the real test period &mdash; "
    "the distributions didn't overlap at all. I'd anchored to <i>absolute</i> time, and the test "
    "period is 30 days later. So I removed them, took the accuracy hit, and then recovered it by "
    "reusing the same quantity as an account identifier instead of a number.\"", "good")

story.append(H2("2-minute explanation"))
story.append(P("Now you add the numbers and the shipping story. Structure it as five beats:"))
story += table([
    ["Beat", "What you say", "Numbers to land"],
    ["1. The setup",
     "IEEE-CIS, 590,540 transactions, 434 raw columns, 3.4993% fraud. Severe imbalance means "
     "accuracy is useless &mdash; predicting 'never fraud' scores 96.5%.",
     "590,540 rows / 3.4993%"],
    ["2. The discovery",
     "Random CV 0.8512, temporal CV 0.5583, same model and code. Random also reported a five "
     "times smaller standard deviation &mdash; it looked more trustworthy exactly where it was "
     "more wrong.",
     "+0.2929 optimism"],
    ["3. The rebuild",
     "Three-phase pipeline: prepare (causal, no fitting), fit (train partition only), transform "
     "(pure lookup). Encoders refitted inside every fold. 7-day purge because my longest velocity "
     "window is 168 hours.",
     "5 folds, 7-day gap"],
    ["4. The self-caught bug",
     "Drift monitoring flagged my own D*_anchored features at KS 1.000. Ablation priced them at "
     "+0.0115 in-period. I removed them, then rebuilt the same information as an entity key.",
     "0.5583 / 0.5468 / 0.5591"],
    ["5. The ship",
     "LightGBM beat XGBoost, CatBoost, RF and LogReg on identical folds. Bundled artifact behind "
     "FastAPI, Docker, PSI/KS drift, 104 tests, green CI.",
     "0.5538 holdout"],
], [22 * mm, 100 * mm, 41 * mm])

story.append(H2("5-minute deep dive &mdash; structure, not a script"))
story.append(P("Spend the time in proportion. If you rush the first two minutes to reach the "
                "architecture, you have thrown away your best material."))
story += table([
    ["Time", "Section", "The point you are making"],
    ["0:00-0:45", "Problem framing", "Imbalance makes accuracy meaningless; PR-AUC with a stated floor"],
    ["0:45-2:00", "<b>The central experiment</b>", "Random vs temporal CV, why entity recurrence causes it"],
    ["2:00-3:00", "Leakage architecture", "prepare/fit/transform; per-fold refitting; the purge gap"],
    ["3:00-4:00", "<b>The D*_anchored case study</b>", "Monitoring caught it, ablation priced it, uid recovered it"],
    ["4:00-4:30", "Model comparison", "LightGBM won; XGBoost and CatBoost lost and are reported as losses"],
    ["4:30-5:00", "Shipping and honesty", "Artifact bundling, API, CI &mdash; then volunteer the limitations"],
], [24 * mm, 47 * mm, 92 * mm])

story += callout("Delivery note",
    "End on a limitation, not a triumph. \"The model is untuned &mdash; I refused to reuse "
    "hyperparameters searched on a feature set that no longer exists, so I'm leaving about 0.017 "
    "on the table.\" Volunteering that reads as senior. Being caught not knowing it reads as the "
    "opposite.", "key")

story.append(PageBreak())

# ============================== PART 2 ==============================
story.append(H1("Part 2 &mdash; Why Each Major Decision Was Made"))
story.append(P("The column that matters in an interview is <b>&quot;why the alternative is worse&quot;</b>. "
                "Anyone can name a technique; being able to say what it beats is what separates a "
                "practitioner from someone reciting a tutorial."))

story += table([
    ["Decision", "Why", "Alternative", "Why the alternative is worse", "Evidence"],
    ["<b>PR-AUC</b> as selection metric", "3.4993% prevalence &mdash; only performance on the rare class matters",
     "Accuracy; ROC-AUC", "&quot;Never fraud&quot; scores 96.5% accuracy. ROC-AUC has ~570k negatives in the FPR "
     "denominator so it barely moves on false-positive volume",
     "RF 0.8819 ROC vs LGBM 0.8838 (gap 0.002) while the PR-AUC gap is 0.091 &mdash; 45x wider"],
    ["<b>Temporal CV</b>", "Deployment always predicts forward in time",
     "Random stratified K-fold", "Same card appears in train and validation; the model interpolates "
     "rather than extrapolates", "0.8512 vs 0.5583, optimism +0.2929"],
    ["<b>Purged CV</b>", "Trailing aggregates straddle the boundary even when rows do not",
     "Plain forward-chaining", "A velocity feature at the start of validation reaches back into "
     "training rows", "All 5 folds verified at exactly 7.0-day gaps"],
    ["<b>7-day purge</b>", "Must be at least the longest look-back window", "A shorter gap, or none",
     "Longest velocity window is 168h = 7d; anything less leaks by construction",
     "Config-enforced; gap measured off the persisted folds"],
    ["<b>Chronological holdout</b>", "The last 20% by time, scored exactly once",
     "Random holdout", "A random holdout inherits the same interpolation problem being corrected for",
     "118,108 rows, days 141-182, 3.4409% fraud"],
    ["<b>LEFT JOIN identity</b>", "Only ~24% of transactions have identity records",
     "INNER JOIN", "Would discard ~76% of the data and bias toward transactions that happen to "
     "carry device info", "Coverage measured in the dataset audit"],
    ["<b>Train-only encoder fitting</b>", "Frequency counts and per-entity means are population statistics",
     "Fit on all data before splitting", "Validation rows contribute to their own encoding &mdash; "
     "the metric flatters itself", "Encoders refit inside every fold; visible in run logs"],
    ["<b>Frequency encoding</b>", "High-cardinality ids where &quot;how common is this&quot; beats the id itself",
     "One-hot; target encoding", "One-hot explodes to thousands of columns; target encoding leaks the "
     "label unless heavily regularised", "card1_freq ranks 9th by SHAP"],
    ["<b>Past-only velocity</b>", "Counts must use only transactions that already happened",
     "Whole-window counts", "A count over the full window includes the future &mdash; textbook target leakage",
     "Offset-band searchsorted; 1/24/168h windows"],
    ["<b>No SMOTE</b>", "Reweighting keeps the data honest and only changes the loss",
     "SMOTE / oversampling", "Interpolating between fraud rows across a ~550-column mostly-categorical, "
     "heavily-missing space produces transactions that could not exist", "Stated design decision in estimators.py"],
    ["<b>scale_pos_weight</b>", "Corrects the loss for imbalance without inventing rows",
     "Leaving imbalance uncorrected", "The model under-predicts the positive class at 3.5% prevalence",
     "Applied to all boosters"],
    ["<b>Isotonic calibration</b>", "The API returns a probability driving a risk band, so it must mean something",
     "Serving raw scores", "Reweighting distorts probabilities; a 0.7 would not mean 70%",
     "ECE 0.06201 to 0.00000 on the calibration fold"],
    ["<b>LightGBM</b>", "Native NaN routing, native categoricals, fast enough to tune",
     "XGBoost / CatBoost / RF / LogReg", "All four measured lower on identical folds; see Part 6",
     "0.5583 vs 0.5370 / 0.5368 / 0.4677 / 0.3560"],
    ["<b>Artifact bundling</b>", "model + pipeline + calibrator + threshold saved as one object",
     "Saving the model alone", "Preprocessing drifts out of sync with the model &mdash; the most "
     "common production ML failure", "ModelArtifact carries all four"],
    ["<b>SHAP</b>", "Per-prediction reasons; declining a customer needs a stated reason",
     "Gain/split importance", "Not per-prediction, not in output units, and inconsistent between "
     "global and local views", "/explain endpoint; TreeExplainer"],
    ["<b>FastAPI</b>", "Typed request validation and a 3-endpoint surface",
     "Flask; a larger API", "Pydantic rejects malformed payloads before they reach the model; more "
     "endpoints is more to keep correct", "/health, /predict, /explain"],
    ["<b>Docker</b>", "Reproducible runtime, verified in CI", "&quot;Works on my machine&quot;",
     "Dependency drift between dev and prod stays invisible until it breaks", "Built and smoke-tested on every push"],
    ["<b>PSI + KS</b>", "PSI triggers alerts; KS reported beside it",
     "One statistic alone", "PSI is binned and can be dominated by an epsilon floor on rare "
     "categoricals; KS is distribution-free and catches what PSI blurs", "D9_anchored: PSI 12.447, KS 1.000"],
    ["<b>Removing D*_anchored</b>", "Proven to fail under the exact shift the model faces",
     "Keeping them for the +0.0115", "The in-period gain is the size of the trap, not a benefit",
     "KS up to 1.000 vs raw D1 at 0.041"],
    ["<b>Reusing it as an entity key</b>", "Identity survives distribution shift; magnitude does not",
     "Discarding the information entirely", "Throws away a genuine account signal along with the "
     "drift problem", "CV 0.5468 to 0.5591; entity_uid_freq 2nd by SHAP"],
], [26 * mm, 33 * mm, 24 * mm, 45 * mm, 35 * mm])

story.append(PageBreak())

# ============================== PART 3 ==============================
story.append(H1("Part 3 &mdash; The Central Experiment"))
story += callout("The one result to know cold",
    "Random stratified 5-fold: <b>0.8512</b> PR-AUC (+/- 0.0044).<br/>"
    "Purged forward-chaining temporal CV: <b>0.5583</b> PR-AUC (+/- 0.0225).<br/>"
    "Same model, same data, same code. Optimism <b>+0.2929</b>, which is <b>52% relative</b>.<br/><br/>"
    "Both arms used the 530-feature configuration, so the comparison is internally consistent. "
    "This is <u>not</u> the shipped model's number &mdash; that is 0.5591 CV / 0.5538 holdout.", "key")

story.append(P("The detail people miss, and the one that makes the story land: <b>random CV also "
                "reported a five times smaller standard deviation</b> (+/- 0.0044 against +/- 0.0225). "
                "It was not merely optimistic, it was <i>confidently</i> optimistic &mdash; it looked "
                "more trustworthy exactly where it was more wrong."))

story.append(H2("Why random CV is wrong here"))
story.append(P("<b>Simple:</b> shuffling puts the future into the training set."))
story.append(P("<b>Technical:</b> the dataset is not a collection of independent rows. It is a set of "
                "<i>entities</i> (cards) observed repeatedly across 182 days. Random splitting assumes "
                "rows are exchangeable, and rows sharing an entity are strongly correlated."))
story.append(P("<b>In this project:</b> a card with transactions on days 10, 45 and 120 gets days 10 "
                "and 120 into training and day 45 into validation. The model has seen that card's "
                "behaviour on <i>both sides</i> of the row it is being asked to predict."))

story.append(H2("How entity recurrence creates leakage &mdash; mechanically"))
story += bullets([
    "<b>Direct memorisation.</b> The model learns &quot;card 13926 is fraudulent&quot; rather than what "
    "fraud looks like. At validation the card is already known, so the answer is recalled, not inferred.",
    "<b>Encoder contamination.</b> Frequency counts and per-entity amount means computed over a shuffled "
    "split include the validation rows themselves, so a row's encoding is partly a function of that row.",
    "<b>Fraud clustering.</b> Fraud arrives in bursts on a compromised card. Split a burst across the "
    "fold boundary and the training half all but announces the validation half.",
    "<b>Interpolation instead of extrapolation.</b> Filling a gap between two known points is a "
    "strictly easier problem than predicting past the end of what the model has seen.",
])

story.append(H2("Why stratification does not fix it"))
story += callout("A favourite follow-up",
    "Stratification balances the <b>class ratio</b> across folds. It says nothing about <b>which "
    "entities</b> or <b>which time periods</b> land where. A stratified fold still contains the same "
    "card on both sides of the split, and still puts February in validation with January and March in "
    "training. Stratification fixes a prevalence problem; this is a dependence and ordering problem. "
    "Solving the wrong problem well changes nothing.", "warn")

story.append(H2("Why the drop is so large"))
story.append(P("0.29 is a big number and an interviewer may probe whether something else explains it. "
                "Four effects stack, and naming them separately shows you understand the mechanism:"))
story += table([
    ["Effect", "What it contributes"],
    ["Entity recurrence", "The same cards appear on both sides of a random split"],
    ["Encoder contamination", "Population statistics computed over the rows they later encode"],
    ["Prevalence shift over time", "Validation fraud rate exceeds training fraud rate in <b>every</b> "
     "temporal fold &mdash; a structural property random folds average away"],
    ["Forward extrapolation", "Temporal CV asks the genuinely harder question: predict a period never seen"],
], [45 * mm, 118 * mm])

story.append(H2("Forward extrapolation, in one line"))
story.append(P("<b>Interpolation</b> fills a gap inside the range you have seen. <b>Extrapolation</b> "
                "predicts beyond the end of it. Production is always extrapolation &mdash; today's model "
                "scores tomorrow's transactions. Random CV measures interpolation skill and reports it as "
                "though it were extrapolation skill. Those are different quantities, and only one of them "
                "is what you get paid for."))

story.append(H2("How I proved temporal CV was the honest one"))
story += bullets([
    "<b>Ran both and kept both.</b> The random arm is a deliberate control "
    "(<font face='Courier' size='8'>--random-cv-control</font>), not a discarded mistake, and is logged "
    "to MLflow as <font face='Courier' size='8'>random_cv_optimism_pr_auc</font>.",
    "<b>The holdout adjudicated.</b> A chronological holdout, untouched until the end, scored 0.5639 "
    "under that configuration &mdash; within one standard deviation of the temporal estimate and nowhere "
    "near the random one. The temporal estimate predicted reality; the random one did not.",
    "<b>Fold temporality was verified, not asserted.</b> Boundaries were read back off the persisted folds: "
    "every validation start is later than its train end, index overlap is empty, and all 5 gaps measure "
    "exactly 7.0 days.",
    "<b>It reproduced.</b> Three separate process invocations returned 0.5583 to four decimal places.",
])

story += callout("&quot;What if you had deployed the random-CV model?&quot;",
    "Give both halves. <b>Commercially:</b> you promise the business 0.85 and deliver roughly 0.56 &mdash; "
    "the alert queue fills with false positives, analyst trust collapses, and the model gets switched off. "
    "<b>Technically:</b> the model would be genuinely worse, not merely mis-measured, because features and "
    "hyperparameters would have been selected against the wrong objective. Every downstream decision "
    "inherits the error.", "warn")

story.append(PageBreak())

# ============================== PART 4 ==============================
story.append(H1("Part 4 &mdash; Data Leakage, Through This Project"))
story.append(P("Six kinds appear in this codebase. For each: the concrete instance, why it counts as "
                "leakage, the defence, and how the defence was tested. Talk about them this way &mdash; "
                "an interviewer can tell instantly whether you have read about leakage or fought it."))

story += table([
    ["Type", "The instance here", "Why it is leakage", "Defence", "How it was tested"],
    ["<b>A. Target leakage</b>", "A velocity count computed over a whole window rather than only "
     "past rows", "The count includes transactions that had not happened at prediction time",
     "Offset-band searchsorted so every count uses strictly earlier rows",
     "Unit tests assert counts are 0 for a first-ever transaction"],
    ["<b>B. Temporal leakage</b>", "Random K-fold placing later transactions in training",
     "Training on the future to predict the past &mdash; impossible in deployment",
     "Purged forward-chaining CV, 7-day gap, chronological holdout",
     "Measured directly: +0.2929 optimism. Fold boundaries verified off persisted folds"],
    ["<b>C. Encoder leakage</b>", "Frequency counts and per-entity means fitted before splitting",
     "The encoding of a validation row is partly computed from that row",
     "Three-phase pipeline; encoders refit inside every fold",
     "Fold logs show FrequencyEncoder and EntityAmountAggregator refitting per fold"],
    ["<b>D. Transductive leakage</b>", "Fitting encoders over concat(train, test) &mdash; standard "
     "in Kaggle solutions", "Uses the test distribution to build the training representation; there "
     "is no test set in production", "Encoders fitted train-only, forfeiting leaderboard rank on purpose",
     "kaggle/README.md states the cost explicitly"],
    ["<b>E. Feature-engineering leakage</b>", "<b>D*_anchored</b>: day_index - D_n",
     "Anchors to <i>absolute</i> time, so the feature's meaning shifts as the calendar advances",
     "Removed as model inputs; the quantity reused as an entity key instead",
     "Drift monitoring: KS up to 1.000 vs raw D1 at 0.041. See Part 5"],
    ["<b>F. Entity leakage</b>", "The same card appearing across a fold boundary",
     "The model recalls the entity rather than generalising from behaviour",
     "Temporal splits keep an entity's later transactions strictly after its earlier ones",
     "Validation fraud rate exceeds training rate in every fold &mdash; the expected signature"],
], [24 * mm, 32 * mm, 34 * mm, 34 * mm, 39 * mm])

story.append(H2("The three-phase architecture, and why it makes leakage structurally harder"))
story.append(P("Most leakage is not a bug in a line of code. It is an <i>ordering</i> mistake &mdash; "
                "something learned before it was allowed to be. So the pipeline separates operations "
                "by <b>what they are permitted to know</b>, rather than by what they compute:"))

story += table([
    ["Phase", "May look at", "May learn", "Examples here"],
    ["<b>prepare(df)</b>", "The current row, and strictly earlier rows", "<b>Nothing.</b> No fitted state",
     "Time features, amount features, past-only velocity, entity keys, <font face='Courier' size='8'>_entity_uid</font>"],
    ["<b>fit(train_df)</b>", "The training partition only", "Population statistics",
     "Frequency counts, per-entity amount means, categorical vocabularies, V missingness blocks"],
    ["<b>transform(df)</b>", "Any frame", "<b>Nothing.</b> Pure lookup",
     "Maps learned statistics onto rows; unseen values map to a defined default"],
], [26 * mm, 36 * mm, 34 * mm, 67 * mm])

story += callout("Why this is stronger than being careful",
    "The guarantee is <b>structural</b>, not procedural. <font face='Courier' size='8'>transform()</font> "
    "has no mechanism to learn anything &mdash; it only reads fitted state. So the question "
    "&quot;could this leak?&quot; collapses to a much smaller one: &quot;was "
    "<font face='Courier' size='8'>fit()</font> handed anything but the training partition?&quot; "
    "That is one call site to audit instead of five hundred columns. Discipline fails under deadline "
    "pressure; an architecture that cannot express the mistake does not.", "key")

story.append(H2("Why encoders must be refitted inside every fold"))
story.append(P("This is the single most common thing candidates get wrong, and a strong answer here "
                "is disproportionately convincing."))
story += bullets([
    "Fold 3 trains on days 1-83.8. Its frequency counts must reflect <b>only</b> what was observable by "
    "day 83.8. Counts computed once over the whole modelling period would include days 84-141.",
    "Fitting encoders once outside the loop makes every fold's encoding partly a function of every other "
    "fold's data. The folds stop being independent estimates and the CV mean becomes optimistic.",
    "The cost is real &mdash; refitting per fold is slower, and unseen categories in later folds map to a "
    "default rather than a learned value. That cost is the point: it is what deployment will actually face.",
    "Visible in the run logs: <font face='Courier' size='8'>FrequencyEncoder fitted on 15 columns</font> and "
    "<font face='Courier' size='8'>EntityAmountAggregator fitted on 4 entity keys</font> print once per fold, "
    "with group counts growing as the training window expands (11,080 then 11,909 and so on).",
])

story += callout("If asked &quot;how do you know your pipeline does not leak?&quot;",
    "Do not answer &quot;I was careful.&quot; Answer with the architecture, then the measurement: "
    "&quot;Fitting is confined to one phase that only ever receives the training partition, and I "
    "measured what leakage would have looked like &mdash; random CV inflated PR-AUC by 0.2929. If my "
    "temporal pipeline were leaking, the holdout would not have landed within one standard deviation "
    "of the CV estimate.&quot;", "good")

story.append(PageBreak())

# ============================== PART 5 ==============================
story.append(H1("Part 5 &mdash; The D*_anchored Case Study"))
story.append(P("This is your best story. It is the one place where the project catches its own mistake, "
                "prices it, and then recovers from it. Tell it as a chain &mdash; each step forced the next."))

story += table([
    ["#", "Step", "What happened"],
    ["1", "I engineered the feature", "D*_anchored = day_index - D_n, meant to turn a moving "
     "day-delta into a fixed calendar anchor such as &quot;the day this card was first seen&quot;"],
    ["2", "It looked excellent", "D1_anchored ranked <b>3rd of 530</b> by mean |SHAP| (0.2640)"],
    ["3", "Drift monitoring disagreed", "Comparing the training period against the real future test "
     "period, all 15 anchored features drifted severely"],
    ["4", "The numbers were extreme", "D9_anchored PSI 12.447, <b>KS 1.000</b>; all 15 in KS 0.84-1.00. "
     "A KS of 1.000 means the distributions do not overlap <i>at all</i>"],
    ["5", "The raw columns were fine", "D15 KS 0.140, D1 KS <b>0.041</b> &mdash; so the instability was "
     "created by my transformation, not inherited from the data"],
    ["6", "Root cause", "day_index is <b>absolute</b> time. Training covers days 1-182; the test period "
     "sits at days 213-396. The anchor's numeric range simply moves"],
    ["7", "Ablation priced it", "Removing all 15: CV 0.5583 to 0.5468, a loss of <b>0.0115</b>, losing "
     "4 of 5 folds. Largest loss on fold 4 (-0.0390), the fold closest in time"],
    ["8", "Interpretation", "The in-period gain <i>is</i> the trap. Every fold and the holdout validate "
     "inside or adjacent to training, exactly where an absolute anchor still lines up"],
    ["9", "Removed and shipped", "515-feature model: holdout fell 0.5639 to 0.5279; prediction-drift PSI "
     "improved 0.0329 to 0.0100"],
    ["10", "Reformulated", "D1 - day_index is <b>constant per card</b>. Used as a grouping key it "
     "identifies an account; drift in the value is irrelevant to equality"],
    ["11", "entity_uid built", "Card fields + that anchor, hashed deterministically: <b>194,519 groups</b> "
     "over 472,432 rows (2.43 txns each) vs the old key's 37,859"],
    ["12", "The model agreed", "entity_uid_freq entered SHAP at <b>#2</b> (0.2277), behind only C13"],
    ["13", "Performance recovered", "547-feature model: CV 0.5591, holdout 0.5538 &mdash; CV now slightly "
     "<i>above</i> the original leaky configuration"],
], [8 * mm, 34 * mm, 121 * mm])

story.append(H2("The numbers, and exactly what they do and do not prove"))
story += table([
    ["Configuration", "Features", "CV PR-AUC", "Holdout PR-AUC", "Prediction PSI"],
    ["Original (leaky anchor)", "530", "0.5583 +/- 0.0251", "0.5639 [0.5488, 0.5786]", "0.0329"],
    ["Stripped", "515", "0.5468 +/- 0.0120", "0.5279 [0.5126, 0.5433]", "<b>0.0100</b>"],
    ["<b>Shipped hybrid</b>", "<b>547</b>", "<b>0.5591 +/- 0.0195</b>", "<b>0.5538 [0.5384, 0.5688]</b>", "0.0455"],
], [40 * mm, 18 * mm, 33 * mm, 42 * mm, 30 * mm])

story += callout("What these numbers DO prove",
    "&bull; Removing the features cost real in-period accuracy: 0.0360 on the holdout, on "
    "non-overlapping intervals.<br/>"
    "&bull; The uid recovered 0.0259 of that, improving <b>5 folds out of 5</b>.<br/>"
    "&bull; The hybrid's holdout interval [0.5384, 0.5688] <b>overlaps</b> the leaky model's "
    "[0.5488, 0.5786], so the two are not statistically distinguishable on this holdout.<br/>"
    "&bull; The anchored features were genuinely unstable across the train/test boundary "
    "(KS up to 1.000, against raw D1 at 0.041).", "good")

story += callout("What these numbers DO NOT prove",
    "&bull; They do <b>not</b> prove the hybrid generalises better to days 213-396. <b>That period has "
    "no labels.</b> No accuracy claim about it can be made from this repository.<br/>"
    "&bull; Prediction PSI 0.0455 is a <b>distribution</b> statistic, not a performance one. It went "
    "<i>up</i> from the stripped model's 0.0100, and it is the highest of the three.<br/>"
    "&bull; &quot;Statistically indistinguishable&quot; means the intervals overlap. It does <b>not</b> mean "
    "the models are equal, and it does not mean the hybrid is better.<br/>"
    "&bull; Part of the stripped model's drop is the missing hyperparameter search, not feature removal. "
    "The two causes are <b>not separable</b> from these runs.", "warn")

story.append(H2("The prepared answers"))

story.append(qa(
    "How did you discover the leakage?",
    "Drift monitoring, not intuition. I was comparing the training period against the real test period "
    "with PSI and KS, and all 15 features I had built showed KS up to 1.000 while the raw D columns they "
    "derive from stayed at 0.041. That pattern &mdash; my derived features unstable, their inputs stable "
    "&mdash; pointed straight at my own transformation.",
    "The monitoring was built as an ops component, not a debugging tool; it found the defect as a side "
    "effect. That is the strongest kind of evidence that the monitoring is real rather than decorative "
    "&mdash; it reported a problem instead of an all-clear.",
    "Do not say you &quot;noticed it looked odd.&quot; The value of the story is that a system caught it, "
    "not that you had a hunch."))

story.append(qa(
    "Why didn't SHAP catch the problem?",
    "SHAP measures how much the model <i>relies</i> on a feature, not whether that reliance will survive "
    "into the future. It is computed on data the model was fitted and validated against &mdash; all of it "
    "in or adjacent to the training period, which is exactly where the anchor still works. SHAP ranked it "
    "3rd because within that window it genuinely was the 3rd most useful feature.",
    "SHAP and drift answer different questions. SHAP: &quot;what is this model doing?&quot; Drift: &quot;will "
    "the inputs still look like this tomorrow?&quot; No amount of in-sample attribution can answer the second. "
    "This is a concrete argument for why explainability does not replace monitoring.",
    "Never say SHAP was wrong or broken. It answered its question correctly; it was simply the wrong "
    "question for this failure mode."))

story.append(qa(
    "If the feature was so predictive, why remove it?",
    "Because its predictiveness was an artefact of where I was measuring. Every CV fold and the holdout sit "
    "inside or next to the training period, which is precisely where an absolute-time anchor still lines up. "
    "The +0.0115 it contributed is the size of the trap, not a benefit &mdash; it would not survive to days "
    "213-396, where KS is 1.000.",
    "This is the difference between a feature that is predictive and one that is <i>reliably</i> predictive. "
    "Any feature that improves every metric you can measure before deployment and fails after it is the most "
    "dangerous class of defect, because nothing in the standard workflow catches it.",
    "Do not claim the feature &quot;did not work.&quot; It worked. Saying otherwise makes the story incoherent "
    "&mdash; if it did not work, removing it would not have cost 0.036 holdout PR-AUC."))

story.append(qa(
    "Why did removing it reduce performance?",
    "For the same reason it was ranked highly: in-period, the anchor is genuinely informative. The holdout "
    "covers days 141-182, immediately adjacent to training, so it is measured in the regime where the feature "
    "still works. The holdout <i>structurally cannot</i> show the benefit of removing it.",
    "This is why the decision could not be judged on the holdout alone. The only forward-looking evidence "
    "available without labels is the drift statistic, which is why I reported it alongside &mdash; while being "
    "clear it is not a performance measure.",
    "Do not present the drop as unimportant. It was 0.036 on non-overlapping intervals. Acknowledge it, then "
    "explain why the metric could not see the upside."))

story.append(qa(
    "Why does using the same information as an entity key make sense?",
    "Because the two uses depend on different properties. As a number, the model learns thresholds on its "
    "<i>magnitude</i>, and the magnitude shifts as the calendar advances. As a grouping key it is compared only "
    "for <b>equality</b> &mdash; two transactions from the same card fall in the same group in any period, "
    "whatever the absolute value happens to be. Identity survives distribution shift; magnitude does not.",
    "Concretely, D1 - day_index is constant for a given card because it encodes the account-open date. "
    "Combined with the card attributes it approximates an account id: 194,519 groups at 2.43 transactions each. "
    "The aggregates computed <i>over</i> those groups &mdash; counts and amount statistics &mdash; are fitted on "
    "the training partition only.",
    "Do not say the drift &quot;stopped mattering.&quot; It stopped mattering <i>for this use</i>. The underlying "
    "value drifts exactly as much as before."))

story.append(qa(
    "Isn't the entity key also leakage?",
    "No, on two counts. It is computed row-locally &mdash; only from values in the same row, with no future "
    "information. And every statistic aggregated over it is fitted on the training partition only, so an "
    "account unseen in training gets a count of 0 rather than a value borrowed from the test period.",
    "The leakage-shaped version of this would be fitting uid aggregates over concat(train, test), which is "
    "exactly what the reference Kaggle solution does and exactly what was refused. That choice costs "
    "leaderboard rank and is documented as costing it.",
    "Do not claim it is &quot;proven leakage-free under all conditions.&quot; What is verified is that it is "
    "row-local and train-only-fitted. That is a strong claim; it is not an unconditional one."))

story.append(qa(
    "How do you know the new entity feature is safe?",
    "Three things are verified and I would state exactly these. It is row-local by construction. Its encoders "
    "are fitted on the training partition only. And the resulting model's prediction drift against the test "
    "period is PSI 0.0455 &mdash; inside the documented stable band, though notably the highest of the three "
    "configurations.",
    "The honest boundary: I have no labels for days 213-396, so I cannot demonstrate predictive performance "
    "there. What I can show is that the construction has no mechanism for future information to enter, and "
    "that the output distribution has not collapsed.",
    "Do not offer PSI 0.0455 as evidence of accuracy. It is a distribution statistic. Conflating the two is "
    "the exact error this project exists to avoid."))

story.append(qa(
    "What is the trade-off?",
    "The uid buys accuracy and spends distributional stability. It recovered 0.0259 holdout PR-AUC, but "
    "prediction drift rose to PSI 0.0455 &mdash; higher than both the stripped model (0.0100) and the leaky "
    "one (0.0329). Most test-period accounts were never seen in training, so entity_uid_freq is 0 for them.",
    "That zero is correct behaviour: an unseen account genuinely has no history. The alternative &mdash; "
    "fitting the encoder over test data so unseen accounts get a plausible count &mdash; is the transductive "
    "shortcut the whole project refuses. So the drift is the honest price of the honest choice.",
    "Do not hide the PSI increase. An interviewer reading the drift table will see it, and volunteering it "
    "first is worth more than the 0.0355 it costs you."))

story.append(PageBreak())

# ============================== PART 6 ==============================
story.append(H1("Part 6 &mdash; Model Selection"))
story += callout("The framing that wins this section",
    "Do not defend LightGBM as a preference. Defend it as the <b>outcome of a controlled comparison</b>: "
    "five models, identical persisted folds, identical encoders, mirrored search spaces. Two challengers "
    "were run and lost, and both are reported as losses. A comparison in which your favourite always wins "
    "proves nothing about the methodology.", "key")

story += table([
    ["Model", "CV PR-AUC", "ROC-AUC", "Brier", "Train time", "Verdict"],
    ["<b>LightGBM</b>", "<b>0.5583 +/- 0.0225</b>", "0.8838", "0.0236", "639 s", "<b>Shipped</b>"],
    ["XGBoost", "0.5370 +/- 0.0211", "0.8754", "0.0242", "1580 s", "Lost: -0.021 at 2.1x cost"],
    ["CatBoost", "0.5368 (fold 4)", "0.8867", "&mdash;", "1431 s", "Lost: -0.048 at 4.6x cost"],
    ["Random Forest", "0.4677 +/- 0.0404", "0.8819", "0.0966", "185 s", "Lost: ROC close, PR-AUC far"],
    ["Logistic Regression", "0.3560 +/- 0.0703", "0.8328", "0.1400", "115 s", "The floor"],
], [32 * mm, 32 * mm, 20 * mm, 18 * mm, 22 * mm, 39 * mm])
story.append(P("CatBoost is a single fold, measured against LightGBM's 0.5845 on that same fold with the "
                "shipped feature set. The other rows are 5-fold means on the 530-feature configuration.", "small"))

story.append(H2("Why LightGBM"))
story += bullets([
    "<b>Native NaN routing.</b> The V block is heavily missing. LightGBM learns a default direction per "
    "split rather than requiring imputation, so missingness stays informative instead of being averaged away.",
    "<b>Native categoricals.</b> 38 categorical columns with thousands of levels are consumed directly, "
    "avoiding a 929-column one-hot matrix.",
    "<b>Speed.</b> It is the only model fast enough at this scale to tune properly &mdash; 639 s against "
    "XGBoost's 1580 s and CatBoost's 1431 s per fold.",
    "<b>It won on the metric that matters</b>, on identical folds, by 0.021 over the nearest challenger.",
])

story.append(H2("Why not each of the others"))
story += table([
    ["Model", "The honest answer"],
    ["<b>XGBoost</b>", "Measured 0.5370 against 0.5583, winning only 1 of 5 folds, at 2.1x the training "
     "cost. It was given a <i>mirrored</i> search space, the same native categorical handling, the same "
     "folds and the same early-stopping rule, so the comparison reflects the algorithms rather than a "
     "luckier configuration. The margin (0.021) is close to the fold-to-fold spread (+/- 0.021), so I would "
     "call it a consistent but modest loss &mdash; the 4-of-5 fold record is what makes it credible, not "
     "the mean alone."],
    ["<b>CatBoost</b>", "0.5368 against LightGBM's 0.5845 on fold 4, at 1431 s vs 310 s. Added deliberately "
     "because its ordered target statistics handle high-cardinality categoricals differently and this "
     "dataset is largely high-cardinality categorical &mdash; so the difference was measured, not assumed. "
     "Reported from one fold because at ~24 min/fold a full run is two hours and the gap is more than twice "
     "the fold-to-fold spread. Worth adding <i>why</i> it lost here when it won in the reference solution: "
     "that run used a GPU with 5000 trees and transductive encodings. A model's reported strength is a "
     "property of its setup as much as its algorithm."],
    ["<b>Random Forest</b>", "0.4677 against 0.5583. It cannot route NaN structurally, so it needs an "
     "imputed dense matrix, and imputation destroys the informativeness of missingness. Its Brier of 0.0966 "
     "is <b>4x worse</b> than LightGBM's 0.0236 &mdash; averaged vote proportions are badly scaled at 3.5% "
     "prevalence, which matters because the API serves probabilities."],
    ["<b>Logistic Regression</b>", "0.3560. Its job is to be the floor: if a complex model cannot clearly "
     "beat a regularised linear one, the complexity is not earning its keep. It recovers only about 64% of "
     "LightGBM's PR-AUC, and it is also <b>3x more volatile</b> across folds (+/- 0.0703 vs +/- 0.0225) "
     "&mdash; the weaker models are also the less trustworthy ones."],
], [30 * mm, 133 * mm])

story.append(H2("Why ROC-AUC made the models look closer than PR-AUC"))
story += callout("The single best metric argument in this project",
    "Random Forest reaches <b>0.8819</b> ROC-AUC against LightGBM's <b>0.8838</b> &mdash; a gap of "
    "<b>0.002</b>. On PR-AUC the same two models differ by <b>0.091</b>, roughly <b>45 times wider</b>.<br/><br/>"
    "The cause: ROC-AUC's false-positive rate has about <b>570,000 negatives</b> in its denominator. Adding "
    "a few thousand false positives barely moves it. Precision's denominator is only the rows you actually "
    "flagged, so the same false positives move it a great deal. At 3.4993% prevalence, false-positive volume "
    "<i>is</i> the operational cost &mdash; every one is an analyst hour and possibly a declined customer.<br/><br/>"
    "<b>Selecting on ROC-AUC would have called these two models equivalent.</b> This is the in-repo "
    "demonstration of why PR-AUC is the selection metric, not a claim borrowed from a blog post.", "key")

story.append(H2("The row-count confound &mdash; and how it was controlled"))
story.append(P("An interviewer who is paying attention will spot this before you mention it, so mention it "
                "first. Logistic Regression and Random Forest were fitted on <b>100,000 rows</b> (all "
                "positives kept, negatives downsampled) because their one-hot matrices are 929 features "
                "wide, while the boosters used all 472,432. That is a genuine confound: the baselines could "
                "be losing on data volume rather than model class."))
story.append(P("<b>So the boosters were rerun at the baselines' row count</b> "
                "(<font face='Courier' size='8'>--force-subsample 100000</font>, same folds, same seed):"))
story += table([
    ["Measurement", "Value"],
    ["LightGBM at 100k rows", "0.5556 (against 0.5583 at full data)"],
    ["Cost of subsampling to LightGBM", "<b>0.0027</b>"],
    ["LightGBM margin over Random Forest", "<b>0.0879</b> &mdash; <b>33x larger</b> than the subsampling cost"],
    ["LightGBM margin over Logistic Regression", "0.1996 &mdash; 74x larger"],
], [70 * mm, 93 * mm])
story.append(P("<b>The ranking is not an artefact of training-set size.</b> A nice detail to offer: fold 0 "
                "is byte-identical in both runs, because its training window holds only 46,274 rows &mdash; "
                "below the cap. That confirms the flag does nothing when it should do nothing, which is how "
                "you know the control itself is correct."))
story += callout("If asked &quot;why not just use a sparse matrix and fit on all 472k rows?&quot;",
    "Because sparse is <b>worse</b> here, and this is a good place to show you checked rather than assumed. "
    "After median-fill the 492 numeric columns have almost no zeros, so CSR pays 8 bytes per nonzero "
    "(4-byte value + 4-byte index) where dense pays 4: <b>1.98 GB sparse against 1.62 GB dense</b>. Sparse "
    "storage only wins when the numeric block itself is sparse; here only the one-hot block is. The honest "
    "fix is more RAM, and the equal-data control answers the question without needing it.", "good")

story.append(PageBreak())

# ============================== PART 7 ==============================
story.append(H1("Part 7 &mdash; Calibration"))
story.append(P("<b>Simple:</b> a calibrated model that says 0.7 is right about 70% of the time. An "
                "uncalibrated one might say 0.7 and be right 30% of the time &mdash; the <i>ordering</i> can "
                "still be perfect while the numbers mean nothing."))

story.append(H2("Why ranking is not enough here"))
story += bullets([
    "The API returns a probability that drives a <b>risk band</b> (thresholds 0.30 and 0.70 in config). A "
    "band boundary at 0.70 is meaningless unless 0.70 corresponds to something real.",
    "Fraud decisions are <b>cost-weighted</b>. Deciding whether to block depends on expected loss = "
    "probability x amount. Multiply an amount by an uncalibrated score and you get a meaningless number.",
    "The decision threshold is <b>configuration</b>, chosen for a business trade-off. Business owners "
    "cannot reason about a threshold on an arbitrary score.",
    "<b>But</b>: calibration is deliberately <i>not</i> used for the Kaggle submission. Isotonic regression "
    "is a monotone step function &mdash; it cannot improve a ranking, and its flat segments create ties that "
    "can only hurt a rank metric like ROC-AUC.",
])

story.append(H2("Why reweighting makes calibration necessary"))
story.append(P("<font face='Courier' size='8'>scale_pos_weight</font> multiplies the loss contribution of "
                "positives by the negative/positive ratio &mdash; about 27.5 at this prevalence. That is the "
                "right way to handle imbalance because it changes only the loss, leaving the data honest. "
                "But it deliberately distorts the model's output: it now predicts as if fraud were far more "
                "common than it is. <b>Reweighting and calibration are a package</b> &mdash; if you reweight "
                "and then serve a probability, you must calibrate, or the probability is systematically "
                "inflated."))

story.append(H2("Isotonic, and why not Platt"))
story += table([
    ["", "Isotonic regression", "Platt scaling"],
    ["Form", "Monotone step function, non-parametric", "Sigmoid, two parameters"],
    ["Assumes", "Only that the mapping is monotone", "That the distortion is sigmoid-shaped"],
    ["Data needed", "More &mdash; can overfit on small folds", "Very little"],
    ["Why chosen here", "The calibration fold has <b>78,739 rows</b>, ample for a non-parametric fit, and "
     "the distortion from scale_pos_weight is not sigmoid-shaped", "Would impose a shape the distortion "
     "does not have"],
], [24 * mm, 72 * mm, 67 * mm])

story.append(H2("How it was fitted, and the result"))
story += bullets([
    "Fitted on the <b>last CV fold's validation slice</b> &mdash; never on training data (the model is "
    "overconfident there) and never on the holdout (which would spend it).",
    "On the calibration fold: <b>ECE 0.06201 to 0.00000</b>, Brier 0.03535 to 0.02325.",
    "On the holdout: <b>Brier 0.0210</b>, which is <i>better</i> than validation's 0.0221 &mdash; the "
    "calibration transferred.",
    "The calibrator is bundled into the artifact alongside the model, pipeline and threshold, so serving "
    "cannot accidentally skip it.",
])

story += callout("The limitation to volunteer, not hide",
    "<b>Holdout ECE is 0.01116.</b> The earlier 530-feature configuration achieved <b>0.00338</b> &mdash; "
    "roughly 3x better. Calibration is fitted on a single fold, and this one generalised less well. The "
    "served probabilities are usable, and the Brier score actually improved, but they are less sharp than "
    "the previous configuration's. Say this before you are asked; the drift and calibration tables are the "
    "two places a careful reader looks for something you glossed over.", "warn")

story.append(qa(
    "Did calibration improve things on the holdout?",
    "Partly, and I would be precise about which part. Brier improved to 0.0210, better than validation's "
    "0.0221, so the calibration transferred out of sample. But holdout ECE is 0.01116 against 0.00338 for "
    "the earlier configuration &mdash; the calibrator generalised less well this time.",
    "ECE and Brier can move in different directions because they measure different things: Brier is a proper "
    "scoring rule combining calibration and sharpness, while ECE measures only the gap between predicted and "
    "observed frequency within bins. A model can be better overall and still bin-wise less well aligned.",
    "Do not claim calibration was a clean win. Quoting only the Brier improvement while omitting the ECE "
    "regression is the kind of selective reporting this project is otherwise built to avoid."))

story.append(PageBreak())

# ============================== PART 8 ==============================
story.append(H1("Part 8 &mdash; Numbers to Memorise"))
story += callout("Correction against your draft list",
    "Two figures in the list you gave me were stale, and I have used the repository values throughout: "
    "<b>tests are 104, not 88</b>, and <b>holdout ECE is 0.01116, not 0.01402</b> (0.01402 belonged to the "
    "515-feature model, not the shipped 547-feature one).", "warn")

story.append(H2("MUST MEMORISE &mdash; you will be asked these directly"))
story += table([
    ["Quantity", "Value", "Why it matters in the room"],
    ["Fraud prevalence", "<b>3.4993%</b>", "The floor every metric is judged against"],
    ["Random CV PR-AUC", "<b>0.8512</b>", "The lie"],
    ["Temporal CV PR-AUC", "<b>0.5583</b>", "The truth (530-feature config)"],
    ["Optimism", "<b>+0.2929</b> (52% relative)", "Your headline finding"],
    ["Shipped CV PR-AUC", "<b>0.5591 +/- 0.0195</b>", "What you actually ship"],
    ["Shipped holdout PR-AUC", "<b>0.5538</b>", "The number that counts"],
    ["95% CI", "<b>[0.5384, 0.5688]</b>", "Shows you quantify uncertainty"],
    ["PR-AUC lift", "<b>16.09x</b> the floor", "Reframes a &quot;low&quot; 0.55 instantly"],
    ["Holdout ROC-AUC", "<b>0.8999</b>", "The friendlier-sounding number"],
    ["Precision @ 0.1% budget", "<b>0.9831</b>", "The operational headline"],
    ["Precision @ 1% budget", "<b>0.9094</b>", "Nine in ten alerts are real fraud"],
    ["Final prediction PSI", "<b>0.0455</b> (stable)", "Distribution, <u>not</u> accuracy"],
    ["Anchored-feature KS", "up to <b>1.000</b>", "Zero distribution overlap"],
    ["Features removed / final", "<b>15</b> removed / <b>547</b> final", "The case study in two numbers"],
    ["Tests", "<b>104</b> passing", "Engineering credibility"],
], [42 * mm, 40 * mm, 81 * mm])

story.append(H2("GOOD TO KNOW &mdash; reach for these on follow-ups"))
story += table([
    ["Quantity", "Value"],
    ["Dataset size", "590,540 transactions, 434 raw columns, 20,663 fraud / 569,877 legitimate"],
    ["Modelling / holdout split", "472,432 rows (days 1-141) / 118,108 rows (days 141-182, 3.4409%)"],
    ["CV structure", "5 purged forward-chaining folds, 7-day gap, windows 46,274 to 372,880 rows"],
    ["Three configurations", "530 leaky / 515 stripped / <b>547 shipped</b>"],
    ["Holdout trajectory", "0.5639 leaky, 0.5279 stripped, <b>0.5538 shipped</b>"],
    ["Prediction PSI trajectory", "0.0329 leaky, 0.0100 stripped, <b>0.0455 shipped</b>"],
    ["Ablation cost", "-0.0115 CV in-period, losing 4 of 5 folds"],
    ["uid groups", "194,519 over 472,432 rows (2.43 txns each) vs the old key's 37,859"],
    ["Model comparison", "LGBM 0.5583 / XGB 0.5370 / CatBoost 0.5368 / RF 0.4677 / LogReg 0.3560"],
    ["ROC vs PR gap (RF)", "ROC 0.002 apart, PR-AUC 0.091 apart &mdash; 45x"],
    ["Subsampling control", "Costs LightGBM 0.0027 against a 0.0879 margin over RF"],
    ["Holdout confusion", "TN 112,424 / FP 1,620 / FN 1,896 / TP 2,168 at threshold 0.2988"],
    ["Calibration", "Fold ECE 0.06201 to 0.00000; holdout ECE 0.01116, Brier 0.0210"],
    ["SHAP top 3", "C13 (0.4900), <b>entity_uid_freq (0.2277)</b>, P_emaildomain (0.2254)"],
    ["Feature drift counts", "181 significant / 6 moderate / 320 stable of 507 compared"],
], [42 * mm, 121 * mm])

story += callout("Two numbers people confuse &mdash; keep them straight",
    "<b>0.5583</b> is the temporal CV score of the <b>530-feature leaky</b> configuration, and it is the "
    "right-hand side of your headline experiment.<br/>"
    "<b>0.5591 / 0.5538</b> are the CV and holdout of the <b>547-feature shipped</b> model.<br/><br/>"
    "If you quote 0.5583 as &quot;my model&quot;, a careful interviewer who has read your README will catch "
    "the mismatch.", "warn")

story.append(PageBreak())

# ============================== PART 9 ==============================
story.append(H1("Part 9 &mdash; System Design / MLOps"))
story.append(P("The question is almost always some version of: <b>&quot;how does your model get from "
                "training to production?&quot;</b> Walk the chain, and at each hop name the failure it "
                "prevents. Naming failures is what makes it sound like experience rather than a diagram."))

story += table([
    ["Stage", "What happens", "The failure it prevents"],
    ["1. Raw data", "Two CSVs, LEFT JOIN on TransactionID, streamed to Parquet via ParquetWriter",
     "INNER JOIN would silently drop ~76% of rows"],
    ["2. Validation", "Schema and temporal-order assertions on every load "
     "(<font face='Courier' size='8'>validate_temporal_order</font>)", "Silent corruption reaching training"],
    ["3. Split", "Chronological: 80% modelling / 20% holdout, cut on timestamp edges, persisted",
     "Tie groups split across partitions; models scored on different splits"],
    ["4. prepare()", "Row-local and past-only features. No fitting", "Future information entering features"],
    ["5. fit()", "Encoders learn from the training partition only, refit per fold", "Encoder leakage"],
    ["6. transform()", "Pure lookup of fitted state", "Learning at inference time"],
    ["7. Train", "LightGBM with early stopping on an inner temporal tail", "Overfitting the outer fold"],
    ["8. Calibrate", "Isotonic on the last fold's validation slice", "Probabilities that do not mean anything"],
    ["9. Threshold", "Chosen on validation, applied unchanged to the holdout", "Turning the holdout into a validation set"],
    ["10. Artifact", "model + pipeline + calibrator + threshold pickled together", "<b>Training/serving skew</b>"],
    ["11. API", "FastAPI loads the artifact once at startup", "Per-request load latency; version drift mid-run"],
    ["12. Docker", "Reproducible image, model mounted read-only", "&quot;Works on my machine&quot;"],
    ["13. Monitor", "PSI/KS on features and predictions", "Silent model rot"],
    ["14. CI", "lint, tests, docker build, API smoke test on every push", "Broken main"],
], [22 * mm, 78 * mm, 63 * mm])

story.append(H2("Why the artifact bundles four things"))
story += callout("The most valuable 30 seconds in an MLOps interview",
    "The artifact contains <b>model + feature pipeline + calibrator + threshold</b>. They are saved and "
    "loaded as a single object because they are a single object logically &mdash; the model's inputs are "
    "defined by the pipeline, its outputs are only meaningful after the calibrator, and its decisions are "
    "only meaningful at the threshold.<br/><br/>"
    "<b>If they become inconsistent:</b> a retrained model with a stale pipeline sees columns in a different "
    "order &mdash; and tree models will happily score that, silently and wrongly, with no exception raised. "
    "A stale calibrator maps new scores through an old distribution, so risk bands quietly shift. A stale "
    "threshold changes the precision/recall balance the business signed off on. Every one of these fails "
    "<i>silently</i>, which is what makes training/serving skew the most common serious production failure "
    "in ML systems.", "key")

story.append(H2("The API surface"))
story += table([
    ["Endpoint", "Purpose", "Design note"],
    ["<font face='Courier' size='8'>/health</font>", "Liveness plus provenance",
     "Returns model name, training timestamp and feature count &mdash; so you can tell <i>which</i> model "
     "is live, not merely that something is"],
    ["<font face='Courier' size='8'>/predict</font>", "Score a batch",
     "Velocity is computed from the request batch; a single transaction honestly reports count = 0 rather "
     "than inventing history. Responses are re-mapped to input order via original_positions, with a "
     "regression test guarding it"],
    ["<font face='Courier' size='8'>/explain</font>", "Per-prediction SHAP",
     "Regulatory need: declining a customer requires a stated reason. Same TreeExplainer as the global "
     "analysis, so global and local explanations are consistent"],
], [26 * mm, 32 * mm, 105 * mm])
story.append(P("Three endpoints, deliberately. Every additional endpoint is more surface to keep correct, "
                "test and secure &mdash; and the project was explicitly simplified down to these from a "
                "larger surface.", "small"))

story.append(H2("Docker: why the model is mounted, not baked in"))
story += bullets([
    "<b>Image size and build time.</b> The artifact is tens of MB; baking it in means rebuilding and "
    "repushing the whole image for every retrain.",
    "<b>Separation of concerns.</b> The image is <i>code</i>, which changes on release cadence. The model is "
    "<i>data</i>, which changes on retrain cadence. Coupling them forces the slower cadence on both.",
    "<b>Read-only mount.</b> The container cannot modify the artifact, so a compromised or buggy process "
    "cannot silently alter the deployed model &mdash; and the same image can serve a rollback by pointing at "
    "a previous artifact.",
    "<b>Rollback speed.</b> Reverting a bad model becomes a mount change, not an image rebuild.",
    "The honest caveat: <b>Docker is verified in CI, not on the development machine</b>, which has no Docker "
    "installed. Say so rather than implying local verification.",
])

story.append(H2("CI, and why it is ordered that way"))
story += table([
    ["Stage", "What it catches", "Why here"],
    ["1. Lint (ruff check + format)", "Style and simple errors", "Cheapest &mdash; fail in seconds, not minutes"],
    ["2. Tests (104, with coverage)", "Logic and leakage regressions", "Runs on synthetic fixtures; the "
     "1.3 GB dataset is not in CI, by design"],
    ["3. Docker build", "Packaging and dependency-resolution errors", "Only worth building once the code is known good"],
    ["4. API smoke test", "The container actually serves", "End-to-end proof that the image runs, not just builds"],
], [40 * mm, 45 * mm, 78 * mm])
story += callout("A detail worth mentioning if CI comes up",
    "The ruff version is read out of requirements-dev.txt with grep rather than hardcoded in the workflow. "
    "A hardcoded pin previously disagreed with the locally installed version and failed the build on pure "
    "formatting differences. Single source of truth for a tool version is a small thing that prevents a "
    "recurring, confusing class of failure.", "good")

story.append(PageBreak())

# ============================== PART 10 ==============================
story.append(H1("Part 10 &mdash; Drift Monitoring"))
story += callout("Burn this into memory before anything else in this section",
    "<b>DRIFT IS NOT PERFORMANCE.</b><br/><br/>"
    "Drift says the <i>inputs or outputs look different</i> than they used to. It says nothing about whether "
    "predictions are <i>correct</i>. A model can drift heavily and stay accurate; it can show zero drift and "
    "be completely wrong. They are measured differently and they answer different questions.<br/><br/>"
    "Confusing the two is the single most common error in monitoring discussions &mdash; and given that this "
    "whole project is about not fooling yourself with a metric, getting this wrong in the room would be "
    "especially costly.", "warn")

story.append(H2("The two statistics, and why both"))
story += table([
    ["", "PSI (Population Stability Index)", "KS (Kolmogorov-Smirnov)"],
    ["What it measures", "Weighted divergence between binned distributions", "Largest gap between two "
     "cumulative distributions"],
    ["Range", "0 upward, unbounded", "0 to 1, bounded"],
    ["Role here", "<b>Triggers</b> the alert", "<b>Reported alongside</b> for interpretation"],
    ["Weakness", "Binned; on rare categoricals the epsilon floor can dominate and inflate the value",
     "Less sensitive to changes in the tails"],
    ["Why both", "PSI above ~10 in this project comes from high-cardinality categoricals where a level "
     "appears in one period and not the other &mdash; so categorical PSI is treated as a <i>ranking</i>, "
     "not a distance. KS is bounded and distribution-free, so it disambiguates",
     "KS = 1.000 has an unambiguous meaning: the distributions do not overlap at all"],
], [26 * mm, 69 * mm, 68 * mm])

story.append(H2("Thresholds, and where they come from"))
story += table([
    ["Band", "Verdict", "Provenance"],
    ["PSI &lt; 0.10", "Stable", "Conventional credit/fraud risk practice"],
    ["0.10 &ndash; 0.25", "Moderate shift &mdash; investigate", "Same"],
    ["PSI &gt; 0.25", "Significant shift &mdash; act", "Same"],
], [30 * mm, 55 * mm, 78 * mm])
story += callout("Say this out loud when asked about thresholds",
    "&quot;These are <b>heuristics from industry practice, not properties of this dataset.</b> I keep them "
    "in config rather than code so tightening an alert is a config change, not a code change.&quot; "
    "Admitting a threshold is conventional is far stronger than implying you derived it.", "good")

story.append(H2("What the monitoring actually found"))
story += bullets([
    "<b>507 features compared</b> between the modelling and test periods: 181 significant, 6 moderate, "
    "320 stable.",
    "<b>The defect:</b> all 15 D*_anchored features drifted severely &mdash; D9_anchored at PSI 12.447 and "
    "<b>KS 1.000</b> &mdash; while the raw D columns they derive from stayed stable (D1 at KS 0.041).",
    "<b>Prediction drift</b> for the shipped model: PSI <b>0.0455</b>, verdict stable, reference mean 0.0588 "
    "against current 0.0423.",
    "Most-drifted overall are identity columns (id_23, id_27, id_31 at PSI 13.2-13.7) &mdash; device and "
    "browser strings genuinely change between periods, which is expected rather than alarming.",
])

story.append(qa(
    "Does low prediction PSI prove your model is accurate?",
    "No. PSI compares the <i>distribution</i> of predicted scores between two periods. It would be perfectly "
    "possible to output the same distribution of scores while assigning them to entirely the wrong "
    "transactions. Accuracy requires labels, and I have none for the test period.",
    "What low prediction PSI does provide is a cheap, label-free early warning. If the score distribution "
    "collapses or shifts sharply, something has broken upstream &mdash; a feature pipeline change, a data "
    "source change, or genuine population shift. It is a smoke alarm, not a thermometer.",
    "Never present 0.0455 as evidence the model performs well on the future period. That is precisely the "
    "conflation this project was built to avoid, and an interviewer who catches it will discount everything "
    "else you said."))

story.append(qa(
    "Your shipped model's prediction PSI went up from 0.0100 to 0.0455. Doesn't that make it worse?",
    "It makes it <i>less distributionally stable</i>, and I would not dress that up. The cause is known: the "
    "uid means most test-period accounts are unseen in training, so entity_uid_freq is 0 for them. That is "
    "correct behaviour &mdash; an unseen account genuinely has no history.",
    "The alternative would be fitting the uid encoder over train and test together so unseen accounts receive "
    "a plausible count. That is transductive and undeployable. So the drift is the honest price of the honest "
    "choice, and 0.0455 is still well inside the &lt;0.10 stable band.",
    "Do not argue the increase is unimportant. Concede it clearly, explain the mechanism, and note it remains "
    "inside the documented band."))

story.append(PageBreak())

# ============================== PART 11 ==============================
story.append(H1("Part 11 &mdash; SHAP"))
story.append(P("<b>Simple:</b> SHAP splits a single prediction into per-feature contributions that sum to "
                "the prediction. <b>Technical:</b> Shapley values from cooperative game theory &mdash; the "
                "average marginal contribution of a feature across all orderings in which it could be added "
                "to the model."))

story.append(H2("Why TreeExplainer specifically"))
story += bullets([
    "<b>Exact for tree ensembles</b>, not sampled &mdash; the general KernelExplainer approximates and is "
    "far slower.",
    "<b>No background dataset required</b>, which is what makes a <i>per-request</i> explanation viable at "
    "serving time. A method needing a reference sample per call would not fit in an API.",
    "Fast enough that <font face='Courier' size='8'>/explain</font> is a real endpoint rather than an "
    "offline batch job.",
])

story.append(H2("Why mean |SHAP| rather than gain or split importance"))
story += table([
    ["", "mean |SHAP|", "Gain / split importance"],
    ["Units", "Model output units &mdash; directly interpretable", "Arbitrary internal quantity"],
    ["Per-prediction?", "Yes &mdash; same method for global and local", "No &mdash; global only"],
    ["Consistency", "Global ranking and the API's per-row explanation are the <b>same number</b>",
     "Global importance and any local story can disagree"],
    ["Bias", "Unbiased with respect to cardinality", "Split count favours high-cardinality features"],
], [26 * mm, 72 * mm, 65 * mm])
story.append(P("&quot;Mean |SHAP|&quot; is simply the average absolute contribution of a feature across "
                "rows &mdash; how much it moves predictions on average, regardless of direction.", "small"))

story.append(H2("The shipped model's top features"))
story += table([
    ["Rank", "Feature", "mean |SHAP|", "Note"],
    ["1", "C13", "0.4900", "Anonymised count column; consistently dominant"],
    ["2", "<b>entity_uid_freq</b>", "<b>0.2277</b>", "<b>Engineered.</b> Transactions per synthetic account"],
    ["3", "P_emaildomain", "0.2254", "Purchaser email domain"],
    ["4", "C1", "0.1872", "Anonymised count column"],
    ["5", "dist1", "0.1832", "Distance feature"],
    ["6", "V70", "0.1697", "From the anonymised V block"],
    ["7", "card1", "0.1684", "Card identifier"],
    ["8", "C14", "0.1677", "Anonymised count column"],
    ["9", "<b>card1_freq</b>", "0.1636", "<b>Engineered.</b> Frequency encoding"],
    ["10", "card6", "0.1607", "Card type"],
], [14 * mm, 44 * mm, 26 * mm, 79 * mm])

story += callout("Why entity_uid_freq at #2 matters so much",
    "The uid was built on a <b>theory</b>: that D1 - day_index is useless as a drifting number but "
    "valuable as a stable identity. Nothing guaranteed the model would agree.<br/><br/>"
    "It entering at rank 2 of 547, behind only C13, is <b>independent confirmation of the reframing</b> "
    "&mdash; the model found real signal in account identity. And it is a satisfying bookend: D1_anchored "
    "previously held rank 3 as a raw number, and the same underlying information now enters at rank 2 "
    "through a key that survives distribution shift.", "good")

story.append(H2("Why SHAP did not reveal the D*_anchored problem"))
story += callout("Prepare this one carefully &mdash; it is a favourite",
    "SHAP is computed on data the model was fitted and evaluated against, all of which lies inside or "
    "immediately adjacent to the training period. That is <b>exactly the regime where the absolute-time "
    "anchor still works</b>. So SHAP correctly reported that D1_anchored was the 3rd most useful feature "
    "&mdash; within that window it genuinely was.<br/><br/>"
    "SHAP answers <i>&quot;what is this model doing?&quot;</i> Drift answers <i>&quot;will the inputs still "
    "look like this tomorrow?&quot;</i> No amount of in-sample attribution can answer the second question. "
    "This project is a concrete argument that <b>explainability does not substitute for monitoring</b> "
    "&mdash; you need both, and they catch different failures.", "key")

story.append(PageBreak())

# ============================== PART 12 ==============================
story.append(H1("Part 12 &mdash; Hard Interview Questions"))
story.append(P("Fifty-four questions, grouped. Each has a <b>SHORT</b> answer to say out loud, a "
                "<b>DEEPER</b> layer for when they keep pulling, and a <b>TRAP</b> &mdash; the tempting "
                "wrong thing to say."))

story.append(H2("ML fundamentals"))
story.append(qa("Why is accuracy the wrong metric here?",
    "At 3.4993% fraud, predicting &quot;never fraud&quot; scores 96.5% accuracy while catching nothing. "
    "Accuracy is dominated by the majority class, so it cannot distinguish a useful model from a useless one.",
    "Accuracy weights all errors equally, but a false negative is a chargeback and a false positive is a "
    "declined customer &mdash; different costs entirely. It also depends on an arbitrary 0.5 threshold, "
    "whereas PR-AUC integrates over all thresholds.",
    "Do not just say &quot;the data is imbalanced.&quot; Give the 96.5% number &mdash; it does the work."))
story.append(qa("What is PR-AUC and why is 0.55 good?",
    "Area under the precision-recall curve. The no-skill baseline equals prevalence, 0.0344 on the holdout, "
    "so 0.5538 is <b>16.09x</b> the floor. PR-AUC always looks small at low prevalence; the absolute number "
    "is meaningless without the floor.",
    "PR-AUC ignores true negatives entirely, which is what makes it appropriate when the negative class is "
    "overwhelming and uninteresting. Unlike ROC-AUC its baseline moves with prevalence, so it must always be "
    "quoted against that baseline.",
    "Never quote 0.5538 alone. Without the floor it sounds like a coin flip."))
story.append(qa("What is the difference between PR-AUC and ROC-AUC?",
    "ROC-AUC uses TPR against FPR; FPR's denominator is all negatives &mdash; about 570,000 here &mdash; so "
    "false-positive volume barely moves it. PR-AUC uses precision, whose denominator is only what you "
    "flagged, so it responds sharply to false positives.",
    "In this repo Random Forest and LightGBM differ by 0.002 ROC-AUC but 0.091 PR-AUC &mdash; 45x wider. "
    "Selecting on ROC-AUC would have called them equivalent. That is a measured demonstration, not a "
    "theoretical claim.",
    "Do not say ROC-AUC is &quot;bad&quot;. It is a reasonable ranking metric; it is simply insensitive to "
    "the cost that dominates here."))
story.append(qa("Why not just optimise F1?",
    "F1 is computed at a single threshold, so it bakes in one precision/recall trade-off. That trade-off is "
    "a business decision, not a modelling one, so it belongs in config &mdash; which is where the threshold "
    "lives in this project.",
    "F1 also weights precision and recall equally, which fraud economics rarely does. PR-AUC evaluates the "
    "whole ranking, letting the threshold be chosen afterwards for whatever alert budget the team has.",
    "Do not dismiss F1 &mdash; it is reported. Argue about <i>selection</i> versus <i>reporting</i>."))
story.append(qa("What is an alert budget and why report precision at one?",
    "It is the fraction of transactions a review team can actually investigate. At a 1% budget, 90.9% of my "
    "flagged transactions are genuine fraud; at 0.1%, 98.3%. That is the number an operations manager can "
    "act on.",
    "It converts a curve into a staffing decision. PR-AUC summarises the whole ranking; precision-at-budget "
    "answers &quot;if I can review 1,181 transactions a day, how many are worth reviewing?&quot;",
    "Do not quote precision at a budget without the recall beside it &mdash; at 1% you catch 26.4% of fraud, "
    "and omitting that oversells."))
story.append(qa("Why did you keep all 339 V columns instead of reducing them first?",
    "Deliberate sequencing: establish a leakage-safe baseline with everything, then prune on evidence from "
    "SHAP and validation rather than on a prior guess about which anonymised columns matter.",
    "Correlation-clustering before a baseline risks removing signal you cannot yet measure the loss of. The "
    "V block was later summarised by missingness structure &mdash; 13 blocks, 26 row-wise features &mdash; "
    "which is a measured reduction rather than an assumed one.",
    "Do not claim you did feature selection. You explicitly deferred it, and that was the decision."))

story.append(H2("Time-series validation"))
story.append(qa("Why temporal CV instead of random?",
    "Because deployment always predicts forward. I measured the difference: random 5-fold gave 0.8512 and "
    "purged temporal CV gave 0.5583 on the same model, data and code &mdash; 0.2929 of pure optimism.",
    "The mechanism is entity recurrence. The same card appears many times over 182 days, so shuffling puts "
    "its later transactions in training and an earlier one in validation. The model interpolates between "
    "known points rather than extrapolating past them.",
    "Do not answer &quot;because it is time-series data.&quot; That is a label, not a reason."))
story.append(qa("What is purged cross-validation and why did you need it?",
    "A gap between the end of training and the start of validation. My velocity features look back up to 168 "
    "hours, so without a 7-day gap an aggregate computed at the start of validation would reach back into "
    "training rows even though the rows themselves are separated.",
    "The purge width is not arbitrary &mdash; it must be at least the longest look-back in the feature set. "
    "Tie it to the feature definition and it stays correct when features change; hardcode 7 and it silently "
    "breaks the day someone adds a 14-day window.",
    "Do not say the gap is &quot;to be safe.&quot; Give the 168-hour justification."))
story.append(qa("Why forward-chaining rather than a sliding window?",
    "Expanding windows use all history available at each point, which is what a production retrain would do. "
    "My folds grow from 46,274 to 372,880 rows.",
    "A sliding window would test robustness to a fixed training size and deliberately discard old data. That "
    "is the right choice when you suspect old data is harmful; here the data spans only 182 days and "
    "discarding it would weaken early folds further.",
    "Do not imply forward-chaining is universally correct. It matches the retraining strategy, and you should "
    "say so."))
story.append(qa("How do you know your folds are actually time-ordered?",
    "I verified it off the persisted folds rather than trusting the code that wrote them. Every validation "
    "start is later than its train end, index overlap is empty, rows are ordered within each window, and all "
    "five gaps measure exactly 7.0 days.",
    "Folds are cut on timestamp edges, persisted once to folds_temporal.npz and reused byte-identically by "
    "every model, so no two models can be scored on different splits. A separate assertion fails the run if "
    "a timestamp appears in two partitions.",
    "&quot;I used TimeSeriesSplit&quot; is not evidence. Naming a class is not the same as verifying output."))
story.append(qa("Why is the validation fraud rate higher than training in every fold?",
    "It is a structural property of the data over time, and it is precisely what random folds average away. "
    "Seeing it in every fold is a positive signal that the temporal structure is intact.",
    "It also means each fold is slightly harder than its training distribution suggests &mdash; a mild "
    "pessimism that is more honest than the alternative. If it appeared in only some folds I would suspect "
    "the split.",
    "Do not describe it as a problem to correct. It is the signature of a correct temporal split."))
story.append(qa("Your holdout is only days 141-182. Isn't that too short?",
    "It is 118,108 transactions, which is ample statistically &mdash; the bootstrap CI on holdout PR-AUC is "
    "only about +/- 0.015 wide. The real limitation is not size, it is <i>adjacency</i>.",
    "Being immediately after training means it cannot exhibit the longer-horizon shift the true test period "
    "(days 213-396) would. That is exactly why the D*_anchored defect was invisible to it, and why I needed "
    "drift monitoring as a separate signal.",
    "Do not defend the holdout as sufficient. Concede the adjacency limit &mdash; it is the honest answer and "
    "it sets up your best story."))

story.append(H2("Leakage"))
story.append(qa("What kinds of leakage did you have to prevent?",
    "Six, and I would name them concretely: target leakage in velocity counts, temporal leakage from random "
    "splitting, encoder leakage from fitting before splitting, transductive leakage from train+test encoding, "
    "feature-engineering leakage in my own D*_anchored features, and entity leakage across fold boundaries.",
    "The architectural defence is a three-phase pipeline where only one phase can learn anything, and it only "
    "ever receives the training partition. That turns &quot;could this leak?&quot; into a single call site to "
    "audit.",
    "Do not recite textbook categories. Name the instance in your own codebase for each."))
story.append(qa("Why must encoders be refitted inside every fold?",
    "Because frequency counts and per-entity means are population statistics. Fold 3 trains on days 1-83.8, so "
    "its counts must reflect only what was observable by day 83.8. Fitting once outside the loop would include "
    "days 84-141.",
    "Fitting outside makes every fold's encoding a function of every other fold's data, so folds stop being "
    "independent and the CV mean becomes optimistic. The cost &mdash; slower, and unseen categories mapping to "
    "a default &mdash; is exactly what deployment faces.",
    "Do not say &quot;to avoid leakage&quot; and stop. Explain <i>which</i> statistic and <i>which</i> rows."))
story.append(qa("What is transductive leakage and why does it matter for Kaggle?",
    "Fitting encoders over train and test together. It is legal in competitions and worth real leaderboard "
    "rank, but it uses the test distribution to build the training representation &mdash; and in production "
    "there is no test set to look at.",
    "The reference 17th-place solution does this throughout: frequency encodings over concat(train, test), and "
    "a feature that is literally TransactionAmt.isin(test.TransactionAmt). It scores 0.952 ROC-AUC on my exact "
    "split against my 0.900, and most of that gap is technique I refused.",
    "Do not pretend your score is competitive with theirs. Explain precisely which techniques account for the "
    "gap &mdash; that is far more impressive than the score would have been."))
story.append(qa("How would you detect leakage in someone else's pipeline?",
    "First, run the control: compare random CV against a proper temporal split. A large gap is the signature. "
    "Second, check where encoders are fitted relative to the split. Third, look for features that are "
    "suspiciously dominant in SHAP but unstable under drift.",
    "The third is the subtle one and it is what caught my own bug &mdash; a feature can be legitimate in "
    "construction yet still encode information that will not transfer. In-sample attribution alone cannot "
    "reveal it.",
    "Do not claim a single test proves absence of leakage. These are detectors, not proofs."))
story.append(qa("Your velocity features count past transactions. How do you know they are causal?",
    "They are built with an offset-band searchsorted over a globally time-sorted frame, so a count can only "
    "include rows strictly earlier in time. There are unit tests asserting a first-ever transaction gets a "
    "count of 0.",
    "Velocity is computed once over the whole timeline during dataset build rather than per partition, because "
    "the look-back needs continuous history. That is safe precisely because it is past-only by construction "
    "&mdash; but it is also why the purge gap exists for the fold boundaries.",
    "Do not say &quot;I used a rolling window.&quot; Rolling windows are trivially easy to get wrong at "
    "boundaries; explain the causality guarantee."))

story.append(H2("Feature engineering"))
story.append(qa("Walk me through your most interesting feature.",
    "entity_uid. It combines the card attributes with D1 - day_index, which is constant per card because "
    "it encodes when the account was first seen. That gives 194,519 synthetic accounts over 472,432 rows, "
    "against 37,859 for my previous key. It ranks 2nd by SHAP.",
    "The interesting part is its history: that same quantity was previously a numeric feature that drifted at "
    "KS 1.000 and had to be removed. As a grouping key the drift is irrelevant, because keys are compared for "
    "equality, not magnitude.",
    "Do not present it as a clever trick you thought of first. Credit the IEEE-CIS 17th-place solution for the "
    "idea &mdash; and note you reimplemented it train-only."))
story.append(qa("Why frequency encoding rather than one-hot or target encoding?",
    "For high-cardinality identifiers, &quot;how common is this value&quot; is more useful to a tree than the "
    "value itself. One-hot would explode to thousands of columns; target encoding leaks the label unless "
    "heavily regularised.",
    "The counts are fitted on the training partition only, so an unseen value maps to 0 rather than borrowing a "
    "count from the future. That forfeits some accuracy relative to fitting over everything, and that forfeit "
    "is the point.",
    "Do not claim frequency encoding is leakage-free by nature. It is leakage-free <i>because of where it is "
    "fitted</i>."))
story.append(qa("How do you handle missing values?",
    "Mostly by not handling them. LightGBM and the other boosters route NaN natively, learning a default "
    "direction per split, so missingness stays informative. Only the dense baselines get imputation, because "
    "they cannot accept NaN.",
    "Missingness is itself signal here &mdash; the V block is ~43% missing and the pattern is structured. I "
    "also built explicit missingness indicators, deduplicated to 66 distinct patterns from 364 candidates, and "
    "summarised the V block by shared null count into 13 blocks.",
    "Do not say &quot;I imputed with the median.&quot; That is what you did for two baseline models only, and "
    "it is the weaker path."))

story.append(H2("Imbalanced classification"))
story.append(qa("Why not SMOTE?",
    "SMOTE interpolates between minority examples. Across a ~550-column space that is largely categorical and "
    "heavily missing, the interpolants would not be plausible transactions &mdash; you would get a card type "
    "halfway between credit and debit.",
    "scale_pos_weight achieves the same rebalancing by changing the loss rather than the data, which leaves "
    "the data honest and adds no synthetic rows to explain. The cost is distorted probabilities, which is why "
    "calibration follows.",
    "Do not say SMOTE &quot;does not work.&quot; It works in dense continuous spaces; explain why <i>this</i> "
    "space is wrong for it."))
story.append(qa("What does scale_pos_weight actually do?",
    "It multiplies the loss contribution of positive examples by the negative-to-positive ratio, about 27.5 at "
    "this prevalence, so the model stops treating the minority class as noise.",
    "It shifts the decision boundary without touching the data, but it deliberately makes the model predict as "
    "though fraud were far more common than it is. That is why reweighting and calibration are a package &mdash; "
    "reweight and serve a raw probability and it is systematically inflated.",
    "Do not describe it as &quot;handling imbalance&quot; and stop. Name the probability distortion; it shows "
    "you understand the consequence."))
story.append(qa("Would undersampling the majority class have worked?",
    "It is what I do for the two dense baselines, capped at 100,000 rows with all positives kept &mdash; but "
    "purely as a memory accommodation, not a modelling choice.",
    "Undersampling discards real data, which is a genuine loss when the negative class carries information "
    "about normal behaviour. I measured the cost: subsampling to 100k costs LightGBM 0.0027 PR-AUC.",
    "Do not present the subsampling as a deliberate imbalance strategy. It was a constraint, and I ran a "
    "control to prove it did not change the model ranking."))

story.append(H2("Model selection"))
story.append(qa("Why LightGBM over XGBoost?",
    "I ran both on identical folds with mirrored search spaces. LightGBM scored 0.5583 against 0.5370, winning "
    "4 of 5 folds, at less than half the training time.",
    "The margin (0.021) is close to the fold-to-fold spread (+/- 0.021), so I call it consistent but modest "
    "&mdash; the 4-of-5 fold record is what makes it credible, not the mean alone. Being precise about the "
    "strength of your own evidence matters more than the result.",
    "Do not claim LightGBM is inherently better. It won this comparison on this data."))
story.append(qa("Why did CatBoost lose here when it often wins on categorical data?",
    "Because setup matters as much as algorithm. It scored 0.5368 against LightGBM's 0.5845 on the same fold, "
    "at 4.6x the runtime &mdash; on CPU, 2000 trees, with train-only encoding.",
    "In the reference solution CatBoost beat LightGBM, but that run used a GPU with 5000 trees and transductive "
    "encodings. I could not reproduce those conditions and would not want the transductive part. So the honest "
    "statement is that it lost <i>under my constraints</i>.",
    "Do not generalise to &quot;CatBoost is worse.&quot; And do not hide that yours is a single-fold result."))
story.append(qa("You only ran CatBoost on one fold. Isn't that weak evidence?",
    "It is weaker than the others and I label it as such. I ran it because the gap &mdash; 0.048 &mdash; is more "
    "than twice the fold-to-fold spread of about 0.02, so additional folds were unlikely to reverse the "
    "conclusion, and a full run is two hours at ~24 minutes per fold.",
    "If the gap had been within the spread I would have run all five or reported nothing. Deciding how much "
    "evidence a conclusion needs is part of the work, and I would rather state a bounded result honestly than "
    "either overstate one fold or omit the model.",
    "Do not present the single fold as equivalent to the 5-fold results. Say which it is."))

story.append(H2("Calibration"))
story.append(qa("What is calibration and why did you need it?",
    "A calibrated model that outputs 0.7 is right about 70% of the time. I needed it because the API returns a "
    "probability that drives risk bands, and because scale_pos_weight deliberately distorts probabilities.",
    "Fitted isotonic on the last fold's validation slice &mdash; never training data, where the model is "
    "overconfident, and never the holdout, which would spend it. Fold ECE went 0.06201 to 0.00000 and holdout "
    "Brier came in at 0.0210, better than validation's 0.0221.",
    "Do not claim calibration improves ranking. It cannot &mdash; isotonic is monotone. That is exactly why the "
    "Kaggle submission uses uncalibrated scores."))
story.append(qa("Why isotonic rather than Platt scaling?",
    "Platt fits a sigmoid, which assumes the distortion has a sigmoid shape. The distortion from "
    "scale_pos_weight does not. Isotonic assumes only monotonicity, and my calibration fold has 78,739 rows "
    "&mdash; ample for a non-parametric fit.",
    "Isotonic's risk is overfitting on small calibration sets and creating tied predictions from its flat "
    "segments. Neither is a problem at this size, and the ties only matter for rank metrics, which is why they "
    "are avoided for the Kaggle submission.",
    "Do not say isotonic is &quot;more flexible&quot; and stop. Name the assumption each one makes."))

story.append(H2("Drift monitoring"))
story.append(qa("What is PSI and how do you interpret it?",
    "Population Stability Index &mdash; a weighted divergence between two binned distributions. Under 0.10 is "
    "stable, 0.10 to 0.25 moderate, above 0.25 significant. Those bands are conventional credit-risk "
    "heuristics, not properties of this dataset, and they live in config.",
    "PSI's weakness is that it is binned, so on high-cardinality categoricals where a level appears in one "
    "period and not the other, an epsilon floor can dominate and produce values above 10. I therefore treat "
    "categorical PSI as a ranking rather than a distance, and report KS alongside.",
    "Do not present the thresholds as derived. Say they are conventions &mdash; it is more credible, not less."))
story.append(qa("Why report KS as well as PSI?",
    "KS is bounded between 0 and 1 and distribution-free, so it disambiguates the cases where PSI's binning "
    "inflates the number. KS = 1.000 has one unambiguous meaning: the distributions do not overlap at all.",
    "That is exactly what happened with D9_anchored &mdash; PSI 12.447 could have been an artefact of binning, "
    "but KS 1.000 confirmed the distributions were genuinely disjoint. One statistic alone would have left the "
    "diagnosis ambiguous.",
    "Do not treat them as interchangeable. Each covers the other's blind spot."))
story.append(qa("What did your monitoring actually discover?",
    "A defect in my own feature engineering. All 15 D*_anchored features I built drifted severely &mdash; up to "
    "PSI 12.447 and KS 1.000 &mdash; while the raw D columns they derive from stayed stable at KS 0.041.",
    "That pattern is diagnostic: derived features unstable while their inputs are stable means the "
    "transformation is the problem. Root cause was anchoring to absolute day index while the test period sits "
    "30+ days later.",
    "Do not describe monitoring as something you built and never used. Its value here is precisely that it "
    "reported a problem rather than an all-clear."))

story.append(H2("MLOps and deployment"))
story.append(qa("Why bundle the pipeline with the model?",
    "Because they are one object logically. The model's inputs are defined by the pipeline, its outputs only "
    "mean anything after the calibrator, and its decisions only mean anything at the threshold. Saving them "
    "separately invites them to drift apart.",
    "The failure mode is silent: a retrained model with a stale pipeline sees columns in a different order, and "
    "a tree model will score that without raising anything. Training/serving skew is the most common serious "
    "production failure in ML and it almost never announces itself.",
    "Do not describe it as convenience. It is a correctness guarantee."))
story.append(qa("How does your API handle a single-transaction request when features need history?",
    "Velocity is computed from the request batch, so a single transaction honestly reports count = 0. It does "
    "not invent history and it does not fail.",
    "The alternative would be a state store like Redis holding per-entity counters &mdash; which was "
    "deliberately removed to keep the system single-node and honest about what it knows. The trade-off is that "
    "single-transaction requests get weaker velocity features, and that is documented rather than hidden.",
    "Do not claim the API has full production velocity. Cold-start is a real limitation and it is in the "
    "limitations section."))
story.append(qa("What happens if a request contains a category the model never saw?",
    "It maps to a defined default &mdash; frequency 0 for unseen values, and an explicit unknown level for "
    "categoricals. Nothing raises, and nothing is silently invented.",
    "This is a direct consequence of fitting encoders train-only: unseen values are an expected condition, not "
    "an error, because production will constantly produce them. A pipeline fitted over train+test would have "
    "hidden this until deployment.",
    "Do not say &quot;it would error.&quot; Being able to state the default behaviour is the point."))
story.append(qa("How would you retrain this in production?",
    "Rerun the pipeline on an extended window, revalidate with the same purged temporal CV, score the new "
    "holdout once, and compare against the incumbent before promoting. The artifact bundling means promotion is "
    "swapping one file.",
    "The gate should be holdout PR-AUC plus drift, not accuracy alone, and the retraining cadence should be "
    "driven by observed drift rather than a fixed calendar. That is listed as future work &mdash; it is not "
    "implemented.",
    "Do not describe scheduled retraining as though it exists. It is in Future Improvements, not in the repo."))
story.append(qa("Why is Docker verified only in CI?",
    "Because Docker is not installed on the development machine. Rather than claim local verification I did not "
    "do, the README states that the build and API smoke test run in CI on every push.",
    "The smoke test does more than build &mdash; it starts the container and exercises the endpoints, so it "
    "proves the image serves rather than merely compiles.",
    "Do not imply you tested it locally. The claim you can defend is the CI one."))

story.append(H2("Project-specific challenges"))
story.append(qa("What was the hardest engineering problem?",
    "Memory. The full pipeline repeatedly exhausted the machine &mdash; at one point a 37 MiB allocation failed "
    "because the system commit charge was at 82% of its limit. I had to restructure training to run one fold per "
    "process.",
    "Two of the fixes were genuine inefficiencies I had introduced: np.nan_to_num allocating a full-size boolean "
    "mask, and np.hstack holding both the parts and the result. A third was not my code at all &mdash; the "
    "machine's commit limit had fallen from 31.3 GB to 24.5 GB. Diagnosing which was which mattered more than "
    "any individual fix.",
    "Do not present it as purely an environment problem. Two of the three causes were mine."))
story.append(qa("Tell me about a bug you found in your own code.",
    "The V missingness-block grouping was derived per frame rather than fitted. That gives 14 groups on the full "
    "training partition but 13 on a 30,000-row slice &mdash; so fit and transform would have disagreed on the "
    "feature set and the pipeline would have raised mid-run.",
    "It was also a leakage bug, not only a crash: deriving group membership from whichever frame is passed lets "
    "validation rows influence their own encoding. Making it a fitted parameter fixed both problems at once, "
    "which is usually the sign you have found the real issue rather than a symptom.",
    "Do not pick a trivial bug. Pick one where the fix reveals you understood the underlying principle."))
story.append(qa("What would you do differently if you started again?",
    "Build drift monitoring before feature engineering, not after. My D*_anchored features survived into a "
    "shipped model precisely because I had no drift signal at the time I created them.",
    "I would also set the memory budget as a design constraint up front rather than discovering it at fold 4. "
    "And I would have run the equal-data control for the dense baselines immediately, instead of leaving the "
    "row-count confound as an unstated asterisk until someone could have challenged it.",
    "Do not answer &quot;nothing.&quot; And do not pick something cosmetic &mdash; pick a sequencing decision "
    "that actually cost you."))
story.append(qa("What is the single most important thing you learned?",
    "That a metric can be confidently wrong. Random CV did not just overstate performance by 0.29 &mdash; it "
    "reported a five times smaller standard deviation while doing it. It looked more trustworthy exactly where "
    "it was more wrong.",
    "The generalisable lesson is that validation design is a modelling decision, not boilerplate, and it "
    "deserves the same scrutiny as the model. Everything else in this project follows from taking that "
    "seriously once.",
    "Do not give a generic lesson about &quot;the importance of clean data.&quot; Use the number."))

story.append(PageBreak())

# ============================== PART 13 ==============================
story.append(H1("Part 13 &mdash; The Interviewer Attacks Your Project"))
story += callout("How to hold your nerve here",
    "Every one of these attacks is <b>fair</b>, and most are already acknowledged in your README. The winning "
    "move is never to defend the weakness &mdash; it is to show you already knew about it, measured its size, "
    "and can say what you would do next. <b>Concede, quantify, redirect.</b>", "key")

story.append(qa("Your final model is untuned. Why should I trust it?",
    "Trust it because it is honestly measured, not because it is optimal. I refused to reuse hyperparameters "
    "searched on the 530-feature space because the feature set changed to 547 &mdash; carrying them over would "
    "make any result unattributable between the feature change and mismatched settings.",
    "Based on the one completed search, tuning is worth roughly +0.017. So the shipped 0.5538 is a floor, not a "
    "ceiling. The blocker is memory, not method: Optuna refits the two largest folds repeatedly in one process "
    "and the machine's commit limit fell from 31.3 GB to 24.5 GB mid-project.",
    "Do not claim untuned is somehow better. It is a real gap; the defensible part is <i>why</i> you did not "
    "paper over it."))

story.append(qa("Your holdout is only days 141-182. How do you know it generalises to days 213-396?",
    "I do not, and I would not claim to. Days 213-396 have <b>no labels</b>. Nothing in this repository can "
    "make an accuracy claim about that period.",
    "What I can say is narrower and I would keep it narrow: the model's <i>output distribution</i> against that "
    "period is stable at PSI 0.0455, and the features that were provably unstable there &mdash; KS up to 1.000 "
    "&mdash; have been removed as model inputs. Those are construction and distribution arguments, not "
    "performance evidence. Closing this properly needs labelled future data or a live shadow deployment.",
    "Never let PSI stand in for accuracy. This is the trap the whole project exists to avoid, and falling into "
    "it here would undo your credibility."))

story.append(qa("Your final PSI is 0.0455, which is not predictive performance. So why do you think the model is robust?",
    "You are right that it is not performance, and I would not argue otherwise. My robustness argument rests on "
    "three things, none of which is PSI: the validation scheme is temporal and purged, the features proven "
    "unstable against the future were removed, and the holdout was scored exactly once.",
    "PSI 0.0455 is a label-free smoke alarm &mdash; it would tell me if the score distribution had collapsed. "
    "It is also worth noting it is the <i>highest</i> of my three configurations, up from 0.0100, because "
    "unseen accounts make entity_uid_freq zero. So I would present it as reassurance that nothing broke, not "
    "as evidence anything works.",
    "Do not overclaim robustness. &quot;Nothing indicates it broke&quot; is honest; &quot;it is robust&quot; is not."))

story.append(qa("You removed a feature and reused the same information as an entity key. Isn't that just hiding leakage?",
    "It would be if the information were leakage &mdash; but it never was. D1 - day_index contains no "
    "future information; it is computed from the current row alone. Its problem was <i>instability</i>, not "
    "leakage: as a number its range shifts with the calendar.",
    "The two uses depend on different properties. A numeric feature is used via thresholds on its magnitude, "
    "which moves. A key is used via equality, which does not. And every statistic aggregated over the key is "
    "fitted on the training partition only, so unseen accounts get 0 rather than a value borrowed from the "
    "test period.",
    "Do not claim it is &quot;proven leakage-free under all conditions.&quot; What is verified is row-locality "
    "and train-only fitting. State that, not more."))

story.append(qa("Your final model has lower holdout PR-AUC than your original. Isn't that a regression?",
    "On the holdout, yes: 0.5538 against 0.5639. I would not soften that. But the original relied on features "
    "with KS up to 1.000 against the deployment period, so its holdout score was partly earned by something "
    "that would not survive deployment.",
    "Two further points. The intervals overlap &mdash; [0.5384, 0.5688] against [0.5488, 0.5786] &mdash; so the "
    "two are not statistically distinguishable on this holdout. And in cross-validation the new model is "
    "slightly <i>ahead</i>, 0.5591 against 0.5583. The holdout also cannot show the upside, because it sits "
    "adjacent to training where the old features still worked.",
    "Do not claim the new model is better on the holdout. It is not. Argue about what the holdout can and "
    "cannot measure."))

story.append(qa("You say the leakage-free model is statistically indistinguishable from the leaky one. Isn't that overstating it?",
    "It is a fair challenge and I would tighten the language. What I can defend precisely is that the 95% "
    "bootstrap intervals overlap. That means I cannot reject the hypothesis that they perform equally &mdash; "
    "it does not demonstrate they are equal.",
    "Overlapping intervals is weaker evidence than a formal test of the difference, which I did not run. The "
    "correct phrasing is &quot;not distinguishable at this sample size,&quot; not &quot;equivalent.&quot; "
    "Absence of a detectable difference is not evidence of no difference.",
    "Do not defend the stronger phrasing. Conceding the precise statistical limit is more impressive than the "
    "claim was."))

story.append(qa("You never ran a labelled future test. So none of this is validated in the way that matters.",
    "Correct, and it is stated in the limitations. The Kaggle test set has no public labels, so a genuine "
    "forward test was not available to me.",
    "What I substituted is explicit: a chronological holdout for performance, and drift statistics for the "
    "unlabelled future period &mdash; with a clear line drawn that the second measures distribution, not "
    "accuracy. If I could extend the project one way, it would be a shadow deployment collecting labels with "
    "the real chargeback delay, which is a matter of weeks to months.",
    "Do not offer holdout or PSI as a substitute for a labelled future test. Name the gap."))

story.append(qa("Your CatBoost comparison is incomplete.",
    "It is &mdash; one fold, not five, and I label it that way rather than presenting it as equivalent evidence.",
    "The reasoning: the gap is 0.048 against a fold-to-fold spread of about 0.02, so more folds were unlikely to "
    "reverse it, and at ~24 minutes per fold a full run is two hours on hardware that was already the binding "
    "constraint. Had the gap been within the spread I would have run all five or reported nothing.",
    "Do not defend it as sufficient. Defend the <i>decision</i> about how much evidence the conclusion needed."))

story.append(qa("Random Forest and Logistic Regression used fewer rows. Your comparison is confounded.",
    "It would have been, so I controlled for it. I reran the boosters at the baselines' 100,000-row count: "
    "subsampling costs LightGBM 0.0027 while its margin over Random Forest is 0.0879 &mdash; 33 times larger.",
    "A detail that shows the control itself is sound: fold 0 is byte-identical in both runs, because its "
    "training window holds only 46,274 rows, below the cap. The flag correctly does nothing when it should do "
    "nothing. I also checked whether sparse matrices would remove the cap entirely &mdash; they would not, "
    "because after median-fill the numeric block is dense, so CSR costs 1.98 GB against 1.62 GB.",
    "Do not simply assert the confound does not matter. Give the control."))

story.append(qa("Why should I care about PR-AUC? Most teams report ROC-AUC.",
    "Because at 3.4993% prevalence ROC-AUC cannot see the cost that dominates. Its FPR denominator holds about "
    "570,000 negatives, so thousands of false positives barely move it &mdash; but every false positive is an "
    "analyst hour and possibly a declined customer.",
    "I have the demonstration in-repo: Random Forest and LightGBM differ by 0.002 ROC-AUC and 0.091 PR-AUC. "
    "Selecting on ROC-AUC would have called them equivalent and shipped the worse model. I still report ROC-AUC "
    "&mdash; it is comparable across studies &mdash; but I select on PR-AUC.",
    "Do not dismiss ROC-AUC entirely. You report it. The argument is about <i>selection</i>."))

story.append(qa("Why not just use the Kaggle leaderboard as your benchmark?",
    "Because the leaderboard rewards techniques that are undeployable. Competitive solutions fit encoders over "
    "train and test together; one feature in the reference solution is literally "
    "TransactionAmt.isin(test.TransactionAmt).",
    "I do generate a submission, deliberately leakage-free and with uncalibrated scores since the metric is "
    "ROC-AUC and isotonic ties can only hurt a ranking. I expect a mid-tier score and the folder documents why. "
    "The 17th-place reference reaches 0.952 ROC-AUC on my exact split against my 0.900 &mdash; and I can name "
    "precisely which techniques account for the gap.",
    "Do not pretend your score is competitive. Being able to decompose the gap is the stronger position."))

story.append(qa("Why didn't you use SMOTE? It's standard for imbalanced problems.",
    "It is standard in dense continuous spaces. This space is ~550 columns, largely categorical, and heavily "
    "missing &mdash; interpolating between two fraud rows yields a card type halfway between credit and debit, "
    "which is not a transaction that could exist.",
    "scale_pos_weight achieves the rebalancing by changing the loss instead of the data, leaving the data "
    "honest. The cost is distorted probabilities, which is exactly why isotonic calibration follows &mdash; "
    "reweighting and calibration are a package.",
    "Do not say SMOTE is bad. Explain why this feature space is wrong for it."))

story.append(qa("Prediction PSI is your only forward signal, but fraud labels are delayed by weeks. Is that enough?",
    "No, and it is not meant to be. PSI is the signal available on <i>day one</i>, before any label exists. It "
    "catches pipeline breakage and sharp population shifts &mdash; it cannot catch a model that is quietly "
    "getting the wrong answers.",
    "A real production setup needs both loops: distribution monitoring at request time, and a delayed "
    "performance loop that scores predictions once chargebacks settle, typically 30 to 90 days later. This "
    "project implements the first and does not implement the second. Retraining gated on holdout PR-AUC and "
    "drift is listed as future work.",
    "Do not imply drift monitoring is a complete monitoring story. Naming the missing second loop is what shows "
    "production judgement."))

story.append(PageBreak())

# ============================== PART 14 ==============================
story.append(H1("Part 14 &mdash; Weak Answer to Strong Answer"))
story.append(P("Same question, two answers. The weak ones are not wrong &mdash; they are <i>unsupported</i>, "
                "and that is what makes them forgettable. Every strong version adds a mechanism or a number."))

pairs = [
    ("I used temporal CV because time series are different.",
     "Deployment always predicts forward, and in this dataset the same card recurs across 182 days &mdash; so a "
     "random split puts a card's later transactions in training and an earlier one in validation, letting the "
     "model interpolate rather than extrapolate. I measured the cost: random CV gave 0.8512, purged temporal CV "
     "gave 0.5583, an optimism of +0.2929."),
    ("I used PR-AUC because the data is imbalanced.",
     "At 3.4993% prevalence, predicting &quot;never fraud&quot; scores 96.5% accuracy and ROC-AUC has ~570k "
     "negatives in its denominator, so neither responds to false-positive volume &mdash; which is the "
     "operational cost. In my comparison RF and LightGBM differ by 0.002 ROC-AUC but 0.091 PR-AUC."),
    ("I removed features that were drifting.",
     "Drift monitoring showed all 15 of my D*_anchored features at KS up to 1.000 against the test period, while "
     "the raw D columns they derive from stayed at 0.041 &mdash; so my transformation caused it. An ablation "
     "priced them at +0.0115 in-period, which is the size of the trap rather than a benefit, because every fold "
     "validates where the anchor still works."),
    ("I used SHAP for explainability.",
     "TreeExplainer, because it is exact for tree ensembles and needs no background dataset, which is what makes "
     "a per-request /explain endpoint viable. Mean |SHAP| is in model output units and is the same number "
     "globally and per-row, so the API's explanation and the global ranking cannot disagree."),
    ("I calibrated the model.",
     "scale_pos_weight deliberately distorts probabilities, and the API serves a probability that drives risk "
     "bands, so calibration is not optional. Isotonic on the last fold's validation slice: fold ECE 0.06201 to "
     "0.00000, holdout Brier 0.0210. I deliberately do <i>not</i> calibrate the Kaggle submission, because "
     "isotonic ties can only hurt a rank metric."),
    ("LightGBM performed best.",
     "On identical persisted folds with mirrored search spaces: LightGBM 0.5583, XGBoost 0.5370, CatBoost 0.5368, "
     "Random Forest 0.4677, Logistic Regression 0.3560. Two challengers were run and lost, and both are reported "
     "as losses &mdash; a comparison where the favourite always wins proves nothing."),
    ("I split the data chronologically to avoid leakage.",
     "The last 20% by time, cut on timestamp edges so tie groups cannot straddle partitions, and scored exactly "
     "once at the end using a threshold chosen on validation. 118,108 rows, days 141-182. Re-optimising the "
     "threshold on it would have turned it into a second validation set."),
    ("My model gets 0.55 PR-AUC.",
     "0.5538 on the holdout, 95% CI [0.5384, 0.5688], which is 16.09x the 3.44% no-skill floor. At a 1% alert "
     "budget 90.9% of flagged transactions are genuine fraud; tightening to 0.1% raises that to 98.3%."),
    ("I handled missing values.",
     "Mostly by not handling them &mdash; the boosters route NaN natively and learn a default direction per "
     "split, so missingness stays informative. Only the dense baselines get median imputation. I also "
     "deduplicated 364 missingness indicators into 66 distinct patterns and summarised the V block into 13 "
     "missingness groups."),
    ("I used Docker for deployment.",
     "The image carries code and the artifact is mounted read-only, so retraining does not require an image "
     "rebuild and a rollback is a mount change. Build and API smoke test run in CI on every push &mdash; I "
     "should be clear that Docker is verified in CI, not on my development machine, which has no Docker installed."),
    ("I monitored for drift.",
     "PSI as the trigger and KS reported alongside, over 507 features between the modelling and test periods: "
     "181 significant, 6 moderate, 320 stable. PSI is binned and can be dominated by an epsilon floor on rare "
     "categoricals, so I treat categorical PSI as a ranking; KS is bounded and disambiguates it."),
    ("My entity key identifies accounts.",
     "Card attributes plus D1 - day_index, which is constant per card because it encodes the account-open "
     "date. 194,519 groups over 472,432 rows at 2.43 transactions each, against 37,859 for my previous "
     "card1+addr1+card2 key. It entered SHAP at rank 2."),
    ("I didn't use SMOTE because it doesn't work well.",
     "It works well in dense continuous spaces. This one is ~550 columns, largely categorical and heavily "
     "missing, so an interpolated row is a card type halfway between credit and debit &mdash; not a transaction "
     "that could exist. scale_pos_weight rebalances the loss instead and leaves the data honest."),
    ("The model is good enough for production.",
     "It is a defensible baseline with an honest evaluation, and I would want three things before production: "
     "a completed hyperparameter search, a shadow deployment collecting delayed labels, and a retraining trigger "
     "gated on holdout PR-AUC and drift. None of those is implemented."),
    ("I wrote tests for the pipeline.",
     "104 tests covering leakage boundaries specifically &mdash; that velocity counts are 0 for a first-ever "
     "transaction, that boosters never receive the imputed dense matrix, that the API returns responses in input "
     "order, and that CatBoost's categorical fill is applied identically at fit and predict."),
    ("Random CV gave better results.",
     "Random CV gave a <i>higher number</i>, which is not the same thing &mdash; 0.8512 against 0.5583. It also "
     "reported a five times smaller standard deviation, +/- 0.0044 against +/- 0.0225. It was confidently wrong, "
     "and the chronological holdout landed within one standard deviation of the temporal estimate, not the random one."),
    ("I did feature engineering on the transaction data.",
     "Past-only velocity over 1, 24 and 168-hour windows using an offset-band searchsorted so counts can only "
     "include strictly earlier rows; per-entity amount deviations; frequency encodings on high-cardinality "
     "identifiers; and a synthetic account key. All fitted train-only, all causal by construction."),
    ("The holdout confirmed my results.",
     "The holdout scored 0.5538 against a CV mean of 0.5591 &mdash; slightly <i>below</i>, which is the correct "
     "direction. Validation informed the threshold and calibration so it is mildly optimistic; the holdout was "
     "untouched. A holdout scoring above validation would be a reason to suspect the split, not to celebrate."),
    ("I removed the leaky features and retrained.",
     "I removed them, measured the cost &mdash; holdout fell 0.5639 to 0.5279 on non-overlapping intervals "
     "&mdash; and then recovered 0.0259 of it by reusing the same quantity as an entity key rather than a "
     "numeric feature. The shipped interval now overlaps the original's."),
    ("My project shows I can build an end-to-end ML system.",
     "It shows the system catches its own errors. It measured its own validation as inflated by 0.2929, and its "
     "own monitoring found a defect in its own feature engineering that SHAP had ranked 3rd most important. "
     "Both are in the repository as findings, not as things I fixed quietly."),
]
for w, s in pairs:
    story.append(weak_strong(w, s))

story.append(PageBreak())

# ============================== PART 15 ==============================
story.append(H1("Part 15 &mdash; Memorise vs Understand vs Skip"))
story.append(P("Prepare in this order. Anything in the third column that you try to memorise is time taken "
                "from the second column, which is where interviews are actually won or lost."))

story.append(H2("MEMORISE EXACTLY &mdash; these must be instant"))
story += bullets([
    "<b>0.8512 / 0.5583 / +0.2929</b> &mdash; random CV, temporal CV, optimism. The headline.",
    "<b>0.5591 +/- 0.0195</b> CV and <b>0.5538 [0.5384, 0.5688]</b> holdout &mdash; the shipped model.",
    "<b>3.4993%</b> prevalence and <b>16.09x</b> lift. Never quote the PR-AUC without the floor.",
    "<b>530 / 515 / 547</b> features and which is which. Confusing these is the easiest way to look careless.",
    "<b>KS 1.000</b> for D*_anchored against <b>0.041</b> for raw D1.",
    "<b>0.9831</b> and <b>0.9094</b> precision at the 0.1% and 1% alert budgets.",
    "<b>PSI 0.0455</b>, and that it is <u>distribution, not accuracy</u>.",
    "<b>104</b> tests. <b>7-day</b> purge. <b>5</b> folds.",
])

story.append(H2("UNDERSTAND DEEPLY &mdash; you will be pushed on the mechanism"))
story += bullets([
    "<b>Why entity recurrence breaks random CV</b>, and why stratification does not help.",
    "<b>Interpolation versus extrapolation</b>, and why production is always the latter.",
    "<b>The three-phase pipeline</b>, and why a structural guarantee beats being careful.",
    "<b>Why encoders must be refitted per fold</b> &mdash; which statistic, which rows, and what it costs.",
    "<b>Why the purge gap is 7 days</b> and not an arbitrary buffer.",
    "<b>Identity versus magnitude</b> &mdash; the whole uid argument rests on this one distinction.",
    "<b>Why calibration is required once you reweight</b>, and why isotonic over Platt.",
    "<b>Drift is not performance.</b> If you internalise one thing from Part 10, this is it.",
    "<b>Why PR-AUC over ROC-AUC</b> at low prevalence, with the 0.002-vs-0.091 example.",
    "<b>What your evidence does not prove</b> &mdash; especially about days 213-396.",
])

story.append(H2("DO NOT WASTE TIME MEMORISING"))
story += bullets([
    "Exact hyperparameter values. Know that the model is untuned and roughly what tuning is worth.",
    "Fold-by-fold PR-AUC tables. Know the mean and the spread.",
    "Exact PSI values for individual features, other than D9_anchored's 12.447 as an illustration.",
    "Library versions, file paths, function signatures.",
    "The confusion matrix cell counts &mdash; know the shape (roughly 1.3 false positives per detection), not the digits.",
    "Precise SHAP values below rank 2. Know C13 leads and entity_uid_freq is second.",
])

story += callout("If you have 30 minutes to prepare",
    "Read Part 1 (the story) and Part 16 (the cheat sheet). Then re-read the four &quot;what these numbers do "
    "not prove&quot; bullets in Part 5. That last one is what stops you overclaiming under pressure, which is "
    "the only way this project can be made to look bad.", "key")

story.append(PageBreak())

# ============================== PART 16 ==============================
story.append(H1("Part 16 &mdash; Final Cheat Sheet"))
story.append(P("Read this last, in the corridor.", "small"))

story.append(H2("1. The story, in five beats"))
story += table([
    ["1", "Random 5-fold CV said <b>0.8512</b> PR-AUC. It looked great."],
    ["2", "Purged temporal CV said <b>0.5583</b> on the same model and code. Optimism <b>+0.2929</b>, and random "
     "CV reported a 5x <i>smaller</i> standard deviation while being wrong."],
    ["3", "So I rebuilt around a three-phase leakage-safe pipeline: purged forward-chaining CV, 7-day gap, "
     "encoders refitted per fold, chronological holdout scored once."],
    ["4", "Drift monitoring then caught <b>my own</b> features: 15 D*_anchored at KS up to 1.000. Ablation priced "
     "them at 0.0115. I removed them, then recovered the loss by reusing the same quantity as an entity key."],
    ["5", "Shipped LightGBM at <b>0.5538</b> holdout (16.09x the floor) behind FastAPI, Docker, PSI/KS drift, "
     "104 tests and green CI."],
], [8 * mm, 155 * mm], header=False)

story.append(H2("2. Ten numbers"))
story += table([
    ["0.8512", "random CV", "0.5583", "temporal CV"],
    ["+0.2929", "optimism (52%)", "0.5591", "shipped CV +/- 0.0195"],
    ["0.5538", "shipped holdout", "[0.5384, 0.5688]", "its 95% CI"],
    ["16.09x", "lift over 3.44% floor", "0.9831 / 0.9094", "precision @ 0.1% / 1%"],
    ["KS 1.000", "anchored-feature drift", "0.0455", "prediction PSI (not accuracy)"],
], [22 * mm, 45 * mm, 32 * mm, 64 * mm], header=False)

story.append(H2("3. Ten decisions, one line each"))
story += bullets([
    "<b>PR-AUC</b> not accuracy &mdash; 96.5% accuracy is achievable by predicting nothing.",
    "<b>Temporal CV</b> not random &mdash; measured 0.2929 of optimism.",
    "<b>7-day purge</b> &mdash; matches the 168-hour velocity look-back.",
    "<b>Encoders fitted train-only</b>, refitted per fold &mdash; population statistics must not see the future.",
    "<b>scale_pos_weight not SMOTE</b> &mdash; interpolated categorical rows are not real transactions.",
    "<b>Isotonic calibration</b> &mdash; reweighting distorts probabilities and the API serves one.",
    "<b>LightGBM</b> &mdash; won on identical folds; native NaN and categoricals.",
    "<b>Artifact bundles four things</b> &mdash; model, pipeline, calibrator, threshold. Prevents silent skew.",
    "<b>PSI triggers, KS interprets</b> &mdash; PSI is binned and can be inflated on rare categoricals.",
    "<b>D*_anchored removed as a feature, reused as a key</b> &mdash; identity survives drift, magnitude does not.",
])

story.append(H2("4. The three questions you will definitely get"))
story += table([
    ["Question", "Your answer in one breath"],
    ["Why is your PR-AUC only 0.55?",
     "It is <b>16.09x</b> the 3.44% no-skill floor, and at a 0.1% alert budget 98.3% of flags are genuine fraud. "
     "Random CV on the same data would have let me report 0.85 &mdash; I measured that as 0.29 of self-deception."],
    ["How do you know there is no leakage?",
     "Structurally, fitting happens in one phase that only ever sees the training partition. Empirically, I "
     "measured what leakage looks like &mdash; +0.2929 &mdash; and my holdout landed within one standard "
     "deviation of the temporal CV estimate."],
    ["What is wrong with your project?",
     "The model is untuned, worth about +0.017. My holdout sits adjacent to training so it cannot show "
     "longer-horizon shift. And I have no labels for the real test period, so I can make distribution claims "
     "about it but not accuracy claims."],
], [42 * mm, 121 * mm])

story.append(H2("5. Limitations to volunteer before you are asked"))
story += bullets([
    "Shipped model is <b>untuned</b> &mdash; roughly +0.017 left on the table, blocked by memory not method.",
    "<b>No labelled future data.</b> Days 213-396 have no labels; only distribution claims are available.",
    "Holdout is <b>adjacent</b> to training, so it understates longer-horizon shift.",
    "The uid <b>raises prediction drift</b> to 0.0455, the highest of the three configurations.",
    "Holdout <b>ECE regressed</b> to 0.01116 from 0.00338 in the earlier configuration.",
    "Dense baselines used <b>100k rows</b> &mdash; controlled for, but not perfectly equal.",
    "CatBoost is a <b>single fold</b>.",
    "<b>Single-node</b>: no scaling, A/B routing or shadow deployment. Docker verified in CI only.",
])

story += callout("6. The one lesson",
    "<b>A metric can be confidently wrong.</b> Random CV did not merely overstate performance by 0.29 &mdash; it "
    "reported a five times smaller standard deviation while doing it. It looked more trustworthy exactly where "
    "it was more wrong.<br/><br/>"
    "Validation design is a modelling decision, not boilerplate. Everything else in this project follows from "
    "taking that seriously once.", "key")

story += callout("7. Final reminder before you walk in",
    "Your strongest material is not the model &mdash; 0.5538 is respectable, not remarkable. It is that the "
    "project <b>measured its own mistakes twice</b>: it proved its validation was inflated, and its monitoring "
    "found a defect in its own features that SHAP had ranked 3rd most important.<br/><br/>"
    "That is rare, and it is entirely undone by a single overclaim. When you do not know, say so. When the "
    "evidence is partial, say which part. <b>The honesty is the product.</b>", "good")

# ============================== BUILD ==============================
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(23 * mm, 12 * mm, "Fraud Detection MLOps – Interview Preparation")
    canvas.drawRightString(187 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(23 * mm, 15.5 * mm, 187 * mm, 15.5 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=23 * mm, rightMargin=24 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm,
                      title="Fraud Detection MLOps - Interview Preparation & Deep Understanding",
                      author="sam200530")
frame = Frame(doc.leftMargin, doc.bottomMargin, 163 * mm,
              A4[1] - doc.topMargin - doc.bottomMargin, id="body")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_footer)])
doc.build(story)
print(f"wrote {OUT}")
