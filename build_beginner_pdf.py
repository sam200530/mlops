"""Generate the beginner's explainer PDF.

Reuses the styling layer from build_prep_pdf.py, then overrides the body styles
to be larger and airier -- this document is meant to be read slowly by someone
new to ML, not scanned by someone revising.
"""

from __future__ import annotations

from pathlib import Path

_src = Path("build_prep_pdf.py").read_text(encoding="utf-8")
_head = _src.split("# ============================== COVER ==============================")[0]
_mod = {}
exec(compile(_head, "prep_styles", "exec"), _mod)

S = _mod["S"]
Paragraph, Table, TableStyle = _mod["Paragraph"], _mod["Table"], _mod["TableStyle"]
KeepTogether = _mod["KeepTogether"]
mm, A4, colors = _mod["mm"], _mod["A4"], _mod["colors"]
MUTED, RULE, ACCENT, INK = _mod["MUTED"], _mod["RULE"], _mod["ACCENT"], _mod["INK"]
GREY_BG = _mod["GREY_BG"]
BaseDocTemplate, Frame = _mod["BaseDocTemplate"], _mod["Frame"]
PageTemplate, PageBreak, Spacer = _mod["PageTemplate"], _mod["PageBreak"], _mod["Spacer"]

# Beginner-friendly: bigger body text, more leading.
S["p"].fontSize = 10.6
S["p"].leading = 16.2
S["p"].spaceAfter = 8
S["cell"].fontSize = 9.4
S["cell"].leading = 13.4
S["cellb"].fontSize = 9.4
S["cellb"].leading = 13.4
S["h1"].fontSize = 18
S["h1"].leading = 23
S["h1"].spaceBefore = 18
S["h2"].fontSize = 13
S["h2"].leading = 17

P, H1, H2, SP = _mod["P"], _mod["H1"], _mod["H2"], _mod["SP"]
bullets, callout, table = _mod["bullets"], _mod["callout"], _mod["table"]

OUT = "Fraud_Detection_Explained_From_Zero.pdf"
story: list = []


def big_number(value, label, note=""):
    """A single headline figure, for when one number carries the point."""
    inner = [Paragraph(f"<font size='24' color='#0b5fa5'><b>{value}</b></font>", S["cell"]),
             Paragraph(f"<b>{label}</b>", S["cell"])]
    if note:
        inner.append(Paragraph(f"<font color='#5b6470'>{note}</font>", S["cell"]))
    t = Table([[inner]], colWidths=[163 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f8fd")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#bcd4ec")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return [t, SP(10)]


def diagram(lines, caption=""):
    """Monospace ASCII figure in a soft box."""
    body = "<br/>".join(l.replace(" ", "&nbsp;") for l in lines)
    t = Table([[Paragraph(f"<font face='Courier' size='8.6'>{body}</font>", S["cell"])]],
              colWidths=[163 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREY_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    out = [t]
    if caption:
        out.append(Paragraph(caption, S["small"]))
    out.append(SP(10))
    return out


# ============================== COVER ==============================
story += [
    SP(46),
    P("Your Fraud Detection Project,", "title"),
    P("Explained From Zero", "title"),
    SP(8),
    P("No prior machine learning knowledge assumed. Every idea is built up one step at a time, "
      "in the order you actually did the work.", "sub"),
    SP(20),
]
story += callout("How to read this",
    "Read it slowly, in order. Each section only uses ideas from the sections before it, so if "
    "something does not make sense, the explanation is probably a page or two earlier rather than "
    "later.<br/><br/>"
    "Every number in here is a real number from your own project.", "key")
story += callout("The short version, if you read nothing else",
    "You built a system that spots stolen-credit-card purchases. Its <b>most valuable feature is not "
    "the model</b> &mdash; it is that the system caught <b>two of its own mistakes</b>, measured how "
    "bad they were, and fixed them.<br/><br/>"
    "That is unusual, and it is the thing worth talking about.", "good")

story.append(PageBreak())

# ============================== 1 ==============================
story.append(H1("1. The problem you are solving"))
story.append(P("Someone steals a credit card. The thief buys something online. In a split second, "
                "software has to decide: <b>allow this purchase, or block it?</b>"))
story.append(P("Get it wrong one way, and a thief walks off with the goods and the shop loses money. "
                "Get it wrong the other way, and a real customer standing at a checkout gets declined, "
                "is embarrassed, and might never come back."))
story.append(P("Your project is the software that makes that decision. It looks at a purchase and "
                "outputs something like <i>&quot;I think this is 87% likely to be fraud.&quot;</i>"))

story.append(H2("Why a computer and not a person"))
story.append(P("A human reviewer could look at a purchase and form a good opinion. But there are "
                "hundreds of thousands of purchases, and each decision has to happen in well under a "
                "second, while the customer waits. So the reviewing has to be automated &mdash; and the "
                "humans get pointed only at the purchases the software finds suspicious."))
story += callout("Keep this in mind throughout",
    "The software does not replace the fraud team. It <b>ranks</b> purchases so the team spends its "
    "limited hours on the most suspicious ones. That is why, later on, you will see a measurement "
    "called <i>&quot;precision at a 1% alert budget&quot;</i> &mdash; it literally means &quot;if my "
    "team can only check 1 in 100, how many of those are worth checking?&quot;", "key")

story.append(PageBreak())

# ============================== 2 ==============================
story.append(H1("2. The data you were given"))
story.append(P("A public dataset from Kaggle, released by a company called IEEE-CIS. Think of it as an "
                "enormous spreadsheet:"))
story += big_number("590,540", "purchases (rows)", "each one a real transaction that already happened")
story += big_number("434", "pieces of information per purchase (columns)",
                    "amount, card, email domain, device, and many anonymised ones")

story.append(P("Some columns are obvious: how much was spent, what kind of card, which email domain. "
                "Many others are just called <font face='Courier' size='9'>V1</font>, "
                "<font face='Courier' size='9'>V2</font>, all the way to "
                "<font face='Courier' size='9'>V339</font> &mdash; the company deliberately hid what "
                "they mean, for privacy. You have to use them without knowing what they are."))

story.append(H2("The answer key"))
story.append(P("One column is special: <font face='Courier' size='9'>isFraud</font>. It is "
                "<b>1</b> if that purchase turned out to be fraud, and <b>0</b> if it was fine. This is "
                "how the computer learns &mdash; and how you check whether it learned anything useful."))

story.append(H2("It arrived in two pieces"))
story += diagram([
    "train_transaction.csv        train_identity.csv",
    "  (what was bought)            (device, browser)",
    "        |                              |",
    "        +--------- glued on ----------+",
    "                TransactionID",
], "Two files, joined on a shared purchase number.")

story.append(P("Here is your first real decision. Only about <b>24%</b> of purchases have a matching "
                "row in the device file. Most purchases have no device information at all."))
story += callout("The decision, and why it mattered",
    "There are two ways to glue two tables together.<br/><br/>"
    "<b>INNER JOIN</b> keeps only rows that appear in <i>both</i> files. That would have thrown away "
    "<b>76% of your data</b> &mdash; and worse, the 24% kept would all be purchases that happened to "
    "carry device info, which is not a fair sample.<br/><br/>"
    "<b>LEFT JOIN</b> keeps every purchase and simply leaves the device columns blank where there is "
    "no match. That is what you used.<br/><br/>"
    "You only knew to make this choice because you measured the coverage first.", "good")

story.append(PageBreak())

# ============================== 3 ==============================
story.append(H1("3. What &quot;training a model&quot; actually means"))
story.append(P("Strip away the jargon and it is simple. A model is a pattern-finding machine."))
story += diagram([
    "STEP 1   Show it 400,000 purchases WITH the answers.",
    "         It notices patterns on its own, like:",
    '           "3am + brand-new device + big amount = often fraud"',
    "",
    "STEP 2   Show it purchases it has NEVER seen. Hide the answers.",
    "",
    "STEP 3   Compare its guesses to the real answers. Score it.",
])
story.append(P("That is the whole idea. Everything else in your project is about making sure "
                "<b>step 2 and step 3 are honest</b> &mdash; because it is astonishingly easy to "
                "accidentally cheat, and then be very pleased with a score that means nothing."))

story.append(H2("You never tell it the rules"))
story.append(P("This is the part beginners often miss. You do <b>not</b> write rules like "
                "<i>&quot;if the amount is over 500 and the device is new, flag it.&quot;</i> You show "
                "the machine examples, and it works out the rules itself &mdash; including "
                "combinations far too complicated for a person to write down."))
story.append(P("That is powerful, and it is also exactly why it can go wrong silently. The machine "
                "will latch onto <i>any</i> pattern that helps it score well, including patterns that "
                "will not exist tomorrow. Most of your project is about that problem."))

story.append(PageBreak())

# ============================== 4 ==============================
story.append(H1("4. Why this problem is hard: almost nothing is fraud"))
story += big_number("3.4993%", "of purchases are fraud",
                    "about 3 or 4 out of every 100 -- this single fact causes most of the difficulty")

story.append(H2("The trap that catches everyone"))
story.append(P("Suppose I write a fraud detector that is one line long:"))
story += diagram(['always answer: "NOT FRAUD"'])
story.append(P("How accurate is it? It is right on every legitimate purchase, and there are a lot of "
                "those. It scores <b>96.5% accuracy</b>."))
story += callout("Stop and take this in",
    "A model that does <b>literally nothing</b> and catches <b>zero fraud</b> scores 96.5%.<br/><br/>"
    "So &quot;accuracy&quot; is worse than useless here &mdash; it actively misleads you. If someone "
    "shows you a fraud model and quotes accuracy, they have not understood their own problem.", "warn")

story.append(H2("So what do you measure instead?"))
story.append(P("You need a score that <b>only cares about the rare thing</b> you are hunting. Yours is "
                "called <b>PR-AUC</b>. You do not need the mathematics; you need what it means:"))
story += table([
    ["Score", "What it means"],
    ["<b>0.034</b>", "What you would get by flagging purchases <b>completely at random</b>. This is the "
     "floor &mdash; pure luck."],
    ["<b>0.554</b>", "<b>What your model gets.</b>"],
    ["<b>1.000</b>", "Perfect. Never achieved by anything real."],
], [24 * mm, 139 * mm])

story += big_number("16x", "better than guessing",
                    "0.554 against a random-chance floor of 0.034")

story += callout("Why 0.55 is not a bad score",
    "0.55 out of 1.0 <i>looks</i> like a fail, because we are all trained by school exams to read 55% "
    "as poor. It is not a percentage.<br/><br/>"
    "PR-AUC always looks small when the thing you are hunting is rare &mdash; that is just how the "
    "arithmetic works. <b>The number is meaningless without the floor next to it.</b> Always say "
    "&quot;0.554, which is 16 times the 0.034 baseline.&quot; Never say 0.554 on its own.", "key")

story.append(H2("The number a business actually cares about"))
story.append(P("Here is a more concrete way to say the same thing. Suppose your fraud team can only "
                "investigate the <b>1 in every 1,000</b> purchases the model finds most suspicious."))
story += big_number("98.3%", "of those alerts are genuine fraud",
                    "if the team can check 1 in 100 instead, it is still 90.9%")
story.append(P("That is a sentence a manager can act on. It is the same information as &quot;PR-AUC "
                "0.554&quot;, translated into staffing."))

story.append(PageBreak())

# ============================== 5 ==============================
story.append(H1("5. The big mistake &mdash; the heart of your project"))
story.append(P("This is the most important section in this document. Read it twice."))

story.append(H2("How testing is supposed to work"))
story.append(P("To find out if a model is any good, you hide some data from it during training, then "
                "test it on the hidden part. The hidden part acts like an exam it has not seen."))
story.append(P("<b>The standard way everyone is taught:</b> shuffle all your rows like a deck of cards, "
                "then hide a random 20%."))
story.append(P("You did exactly that. You scored <b>0.8512</b>. That is a very good score."))

story += callout("It was fake",
    "And the reason is subtle enough that most people never notice it.", "warn")

story.append(H2("Why shuffling breaks everything here"))
story.append(P("Your data is not 590,540 unrelated purchases. It is a much smaller number of "
                "<b>credit cards</b>, each making many purchases over six months."))
story += diagram([
    "Card #13926 appears many times across the 6 months:",
    "",
    "   Day 10        Day 45        Day 120",
    "     |             |             |",
    "   TRAIN         TEST          TRAIN      <-- after shuffling",
    "",
    "The model studies day 10 and day 120 of this card,",
    "then is tested on day 45 -- which sits BETWEEN them.",
], "After shuffling, the model has seen the same card on both sides of the answer.")

story += callout("The exam analogy",
    "It is like giving a student the answers to questions 1 and 3 while they revise, then testing them "
    "on question 2 &mdash; and being impressed when they do well.<br/><br/>"
    "They are not solving the problem. They are <b>remembering</b>, and filling in a small gap between "
    "two things they were told.", "key")

story.append(H2("Why that never happens in real life"))
story.append(P("Tomorrow morning, your model has to judge purchases that <b>have not happened yet</b>. "
                "There is no possible way to have seen the future. Real life is always: learn from the "
                "past, guess about what comes next."))
story.append(P("So a shuffled test measures the wrong skill entirely. It measures <b>filling in gaps</b>, "
                "and then reports that number as though it measured <b>predicting forward</b>."))

story.append(PageBreak())

# ============================== 6 ==============================
story.append(H1("6. The fix, and what it cost"))
story.append(P("Instead of shuffling, you split by <b>date</b>. Always train on earlier days, always "
                "test on later days:"))
story += diagram([
    "Round 1:  train on days 1-13    ->  test on days 20-38",
    "Round 2:  train on days 1-31    ->  test on days 38-65",
    "Round 3:  train on days 1-58    ->  test on days 65-91",
    "Round 4:  train on days 1-84    ->  test on days 91-115",
    "Round 5:  train on days 1-108   ->  test on days 115-141",
], "Five rounds. The training window always grows, and always stops before the test window starts.")

story.append(H2("Why there is a gap between them"))
story.append(P("You may notice training stops at day 13 but testing starts at day 20 &mdash; a "
                "<b>7-day gap</b>. That is deliberate, and the reason is neat."))
story.append(P("Some of your clues are things like <i>&quot;how many times was this card used in the "
                "last 7 days?&quot;</i> If testing started the instant training ended, then a purchase "
                "on the first test day would look back 7 days &mdash; straight into the training "
                "period. The rows would be separated, but the information would still cross over."))
story += callout("So the gap is not a safety margin, it is a measurement",
    "The gap is <b>7 days</b> because your longest look-back window is <b>7 days</b> (168 hours). If "
    "you ever added a 14-day clue, the gap would have to become 14 days. It is tied to the features, "
    "not picked for comfort.", "good")

story.append(H2("The result &mdash; your headline finding"))
story += table([
    ["Testing method", "Score", "Honest?"],
    ["Shuffled (the standard way)", "<b>0.8512</b>", "No &mdash; the model had seen the future"],
    ["Split by time (your way)", "<b>0.5583</b>", "Yes"],
    ["<b>The difference</b>", "<b>0.2929</b>", "<b>This is how much you were fooling yourself</b>"],
], [58 * mm, 30 * mm, 75 * mm])

story += callout("The detail that makes this finding really good",
    "The shuffled test did not just give a higher score. It also reported that it was <b>five times "
    "more consistent</b> across its five rounds.<br/><br/>"
    "So it looked <i>more trustworthy</i> at exactly the moment it was <i>more wrong</i>. That is what "
    "makes this kind of mistake so dangerous &mdash; everything about it feels reassuring.", "key")

story.append(P("Most people never run this check. They publish the 0.85, deploy it, and are baffled "
                "when the real-world performance is nowhere near. <b>You measured your own "
                "self-deception and wrote the number down.</b>"))

story.append(PageBreak())

# ============================== 7 ==============================
story.append(H1("7. &quot;Leakage&quot; &mdash; the one word to really understand"))
story += callout("Definition",
    "<b>Leakage is when your model accidentally sees information it would not have in real life.</b>", "key")

story.append(H2("The clearest example"))
story.append(P("Imagine you are predicting tomorrow's weather, and one of your input columns is "
                "<b>tomorrow's temperature</b>."))
story.append(P("Your model would be 99% accurate. It would also be completely worthless, because "
                "tomorrow morning that column will be empty &mdash; that is the whole thing you are "
                "trying to predict."))
story.append(P("The shuffling problem in section 5 was leakage. So was a bug you found later. Leakage "
                "is the central enemy of this entire project."))

story.append(H2("How your project makes leakage hard"))
story.append(P("Rather than trying to be careful (people get tired, deadlines happen), you built the "
                "code so the mistake is <b>difficult to express</b>. Every operation belongs to one of "
                "three steps, and each step is only allowed to know certain things:"))
story += table([
    ["Step", "What it is allowed to do", "Example"],
    ["<b>1. prepare</b>", "Look at the current purchase and <b>earlier</b> ones. Learn nothing.",
     "&quot;How many times was this card used in the past hour?&quot;"],
    ["<b>2. fit</b>", "<b>Learn</b> &mdash; but only from training data, never from test data.",
     "&quot;How common is gmail.com?&quot; counted over training purchases only"],
    ["<b>3. transform</b>", "Look up what step 2 learned. Learn nothing new.",
     "&quot;This is gmail.com, so use the number learned earlier&quot;"],
], [26 * mm, 62 * mm, 75 * mm])

story += callout("Why this is stronger than just being careful",
    "Learning can <b>only</b> happen in step 2. Step 2 is <b>only ever handed training data</b>.<br/><br/>"
    "So the question &quot;could this leak?&quot; stops being &quot;let me check 547 features&quot; and "
    "becomes &quot;let me check one place in the code.&quot; The safety comes from the shape of the "
    "system, not from remembering to be good.", "good")

story.append(PageBreak())

# ============================== 8 ==============================
story.append(H1("8. Features &mdash; the clues you built"))
story.append(P("Raw data on its own is weak. The real skill is <b>inventing useful clues</b> from it. "
                "You built your 434 raw columns up to <b>547</b> clues. Some examples:"))
story += table([
    ["Clue you built", "Why it helps"],
    ["How many times has this card been used in the last <b>1 hour / 24 hours / 7 days</b>?",
     "A stolen card gets used in a burst. Normal cards do not."],
    ["How does this amount compare to this card's <b>usual</b> spending?",
     "A card that always buys coffee suddenly buying a laptop is worth a look."],
    ["How <b>common</b> is this email domain, this address, this device?",
     "Rare and throwaway domains behave differently from ordinary ones."],
    ["Which pieces of information are <b>missing</b>?",
     "What is absent is itself a clue &mdash; fraudsters often supply less."],
], [72 * mm, 91 * mm])

story += callout("The rule every single clue had to obey",
    "Each clue may only use information that existed <b>before that purchase happened</b>.<br/><br/>"
    "&quot;How many times was this card used in the last hour&quot; is fine.<br/>"
    "&quot;How many times was this card used in total&quot; is <b>not</b> &mdash; the total includes "
    "the future.<br/><br/>"
    "It sounds obvious written down. It is very easy to get wrong in code, which is why there are "
    "tests specifically checking that a card's very first purchase gets a count of zero.", "warn")

story.append(PageBreak())

# ============================== 9 ==============================
story.append(H1("9. Choosing which machine to use"))
story.append(P("There are many kinds of pattern-finding machine. Rather than picking a fashionable one, "
                "you tried <b>five</b> on exactly the same data, with exactly the same rules, and let "
                "the measurements decide."))
story += table([
    ["Machine", "Score", "Result"],
    ["<b>LightGBM</b>", "<b>0.5583</b>", "<b>Winner &mdash; this is what you shipped</b>"],
    ["XGBoost", "0.5370", "Lost, and took twice as long"],
    ["CatBoost", "0.5368", "Lost, and took over four times as long"],
    ["Random Forest", "0.4677", "Clearly behind"],
    ["Logistic Regression", "0.3560", "The simplest option, kept as a floor"],
], [45 * mm, 28 * mm, 90 * mm])

story.append(H2("Why keeping the losers matters"))
story += callout("This is worth saying in an interview",
    "Two of these &mdash; XGBoost and CatBoost &mdash; are famous, well-respected tools. You ran them "
    "properly and <b>they lost</b>. You wrote that in your README rather than quietly deleting them."
    "<br/><br/>"
    "A comparison where your favourite always wins tells nobody anything. A comparison where you "
    "report a result you did not want is evidence that the other results can be trusted too.", "good")

story.append(H2("Why the simplest model is still in the list"))
story.append(P("Logistic Regression scores 0.3560, far behind. That is precisely its job. It is the "
                "<b>floor</b>: if a complicated machine cannot clearly beat a simple one, the "
                "complication is not earning its place. LightGBM beats it by a wide margin, so the "
                "complexity is justified &mdash; and now you can prove that rather than assume it."))

story.append(PageBreak())

# ============================== 10 ==============================
story.append(H1("10. The bug you caught in your own work"))
story.append(P("This is the best story in your project. Take your time with it."))

story.append(H2("What happened, step by step"))
story += table([
    ["1", "You invented 15 clues, called the <b>D*_anchored</b> features."],
    ["2", "The model <b>loved</b> them. One ranked as the <b>3rd most useful clue out of 530</b>."],
    ["3", "But you had also built a <b>monitor</b> &mdash; a separate program that asks: "
          "&quot;does the future data still look like the data I trained on?&quot;"],
    ["4", "The monitor raised an alarm. Those 15 clues had <b>zero overlap</b> with future data. "
          "Not &quot;a bit different&quot; &mdash; <b>no overlap at all</b>."],
    ["5", "The raw columns those clues were built from were <b>completely fine</b>. So the problem was "
          "your transformation, not the data."],
], [8 * mm, 155 * mm], header=False)

story.append(H2("Why it broke &mdash; the sea-level analogy"))
story.append(P("Those clues measured time from a <b>fixed calendar starting point</b>. Your training "
                "data covers days 1 to 182. The real future data is days 213 to 396."))
story += callout("The analogy",
    "It is like measuring everyone's height <b>from sea level</b>.<br/><br/>"
    "In your home town that works fine. Then you move the whole operation to a mountain village, and "
    "suddenly everyone is &quot;2,000 metres tall.&quot; The people did not change. Your reference "
    "point moved.<br/><br/>"
    "Your clues had the same flaw: perfectly sensible during training, meaningless a month later.", "key")

story.append(H2("What you did about it"))
story.append(P("You deleted all 15. Your score got <b>worse</b> &mdash; from 0.5639 down to 0.5279."))
story += callout("Why deleting them was still right",
    "The score those clues earned was <b>borrowed from a future that would not arrive</b>. They looked "
    "good on every test you could run before deployment, and would have failed after it.<br/><br/>"
    "That is the most dangerous kind of bug there is: one that <i>improves</i> every number you can "
    "check.", "warn")

story.append(PageBreak())

# ============================== 11 ==============================
story.append(H1("11. The clever save"))
story.append(P("Then you noticed something, and this is the cleverest idea in the whole project."))

story.append(P("That same calculation &mdash; <b>D1 minus the day number</b> &mdash; gives the "
                "<b>same value for every purchase made on one card</b>. It is effectively the card's "
                "birthday."))

story += diagram([
    "As a MEASUREMENT (a number the model does maths on):",
    "    the value slides as the calendar moves        -> BROKEN",
    "",
    "As a NAME TAG (something you only check for a match):",
    "    two purchases from the same card share a tag",
    "    ... in ANY month, whatever the number is      -> FINE",
])

story += callout("The one-sentence idea",
    "<b>Identity survives change. Measurement does not.</b><br/><br/>"
    "If you ask &quot;how big is this number?&quot;, a shifting number ruins you. If you only ask "
    "&quot;is this the same as that?&quot;, the shift does not matter at all.", "key")

story.append(P("So you rebuilt it as an <b>account ID</b> instead of a number. It groups your 472,432 "
                "purchases into <b>194,519 accounts</b> &mdash; a much sharper picture than the "
                "37,859 groups you had before."))

story.append(H2("Did it work?"))
story += table([
    ["Version", "Score", "What it was"],
    ["Original", "0.5639", "Included the broken clues &mdash; score partly borrowed"],
    ["After deleting them", "0.5279", "Honest, but weaker"],
    ["<b>After the rebuild</b>", "<b>0.5538</b>", "<b>Honest, and nearly all the score recovered</b>"],
], [40 * mm, 25 * mm, 98 * mm])

story += callout("The model agreed with you",
    "Out of <b>547</b> clues, the new account-ID clue came in at <b>number 2</b> &mdash; second most "
    "useful of everything you built.<br/><br/>"
    "You had a theory. You built it. The model independently confirmed it. That is about as satisfying "
    "as this work gets.", "good")

story.append(PageBreak())

# ============================== 12 ==============================
story.append(H1("12. The finishing touches"))

story.append(H2("Calibration &mdash; making the numbers honest"))
story.append(P("If your model says <b>&quot;70% likely to be fraud&quot;</b>, then out of 100 such "
                "purchases, roughly 70 should really be fraud. If only 30 are, the number is a lie "
                "&mdash; even if the model is still good at <i>ranking</i> which is worse than which."))
story.append(P("Calibration is a correction step that fixes the numbers so they mean what they say. "
                "You need it because the service hands out probabilities that other systems make "
                "decisions with."))

story.append(H2("Serving it &mdash; turning code into a service"))
story += table([
    ["Piece", "In plain terms"],
    ["<b>FastAPI</b>", "Turns your model into something other software can ask questions over the "
     "internet &mdash; send a purchase, get back a risk score"],
    ["<b>Docker</b>", "A sealed box containing your program and everything it needs, so it runs "
     "identically on any computer. Solves &quot;but it works on my machine&quot;"],
    ["<b>SHAP</b>", "Explains <i>why</i> each individual decision was made. Legally necessary &mdash; "
     "you cannot decline a customer with &quot;the computer said so&quot;"],
    ["<b>Monitoring</b>", "Keeps watching whether the world has drifted away from what the model "
     "learned. <b>This is what caught your bug</b>"],
    ["<b>104 tests + CI</b>", "Every time you change the code, a robot re-checks that nothing broke, "
     "automatically"],
], [32 * mm, 131 * mm])

story += callout("One important warning about monitoring",
    "Monitoring tells you whether the data <b>looks different</b>. It does <b>not</b> tell you whether "
    "the model is <b>still right</b>.<br/><br/>"
    "Those are two different things and it matters a lot. To know whether it is still right, you need "
    "to know what actually turned out to be fraud &mdash; and in the real world that information "
    "arrives weeks later, when customers dispute charges.<br/><br/>"
    "If you ever say &quot;my drift is low so my model is accurate&quot;, that is wrong, and it is "
    "exactly the kind of mistake this project exists to avoid.", "warn")

story.append(PageBreak())

# ============================== 13 ==============================
story.append(H1("13. The whole story, in order"))
story += table([
    ["1", "<b>Looked at the data first.</b> Found only 24% had device info (so: keep everything, do "
          "not throw 76% away) and found that the row number secretly encoded the time (so: never let "
          "the model see it)."],
    ["2", "<b>Built the clues</b> &mdash; 547 of them, every one using only information from before "
          "the purchase."],
    ["3", "<b>Tested the standard way</b> and scored 0.8512."],
    ["4", "<b>Realised it was cheating</b>, because shuffling let the model see the same card's future."],
    ["5", "<b>Rebuilt the testing honestly</b>, splitting by date with a 7-day gap. The real score was "
          "0.5583. The gap between those two numbers, <b>0.2929</b>, is your headline finding."],
    ["6", "<b>Compared five machines</b> on identical data. LightGBM won; two famous alternatives lost "
          "and are reported as losses."],
    ["7", "<b>Monitoring caught your own bug</b> &mdash; 15 clues that would break in the real world, "
          "one of which the model ranked 3rd most important."],
    ["8", "<b>Deleted them and took the hit</b>, dropping to 0.5279."],
    ["9", "<b>Rebuilt the same information as an account ID</b> instead of a number, recovering to "
          "<b>0.5538</b> &mdash; and the model ranked that new clue <b>2nd out of 547</b>."],
    ["10", "<b>Shipped it</b> behind a web service, in a sealed box, with explanations, monitoring, "
           "104 tests and automatic checking."],
], [8 * mm, 155 * mm], header=False)

story.append(H2("If someone asks what you built, say this"))
story += callout("",
    "&quot;I built a fraud detector on a public dataset of 590,000 card purchases. My first score was "
    "0.85, but I found my testing method was cheating &mdash; it was letting the model see the future. "
    "Tested honestly it was 0.56, so I rebuilt the whole thing to be honest.<br/><br/>"
    "Then my monitoring caught that 15 features I had built myself would break in the real world, even "
    "though the model rated one of them third most important. I removed them, lost accuracy, and got "
    "most of it back by using the same information a smarter way.<br/><br/>"
    "The final model is 16 times better than guessing, and it runs as a real service with monitoring "
    "and automated tests.&quot;", "good")

story += callout("And the thing to remember",
    "<b>The point of your project is not the model.</b> 0.554 is respectable, not remarkable.<br/><br/>"
    "The point is that the system <b>caught its own mistakes twice</b> &mdash; it proved its own "
    "testing was lying, and its own monitoring found a flaw in its own features.<br/><br/>"
    "Most projects cannot do that. That is what makes yours worth talking about.", "key")


# ============================== BUILD ==============================
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(23 * mm, 12 * mm, "Fraud Detection Explained From Zero")
    canvas.drawRightString(187 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(23 * mm, 15.5 * mm, 187 * mm, 15.5 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=23 * mm, rightMargin=24 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm,
                      title="Your Fraud Detection Project, Explained From Zero",
                      author="sam200530")
frame = Frame(doc.leftMargin, doc.bottomMargin, 163 * mm,
              A4[1] - doc.topMargin - doc.bottomMargin, id="body")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_footer)])
doc.build(story)
print(f"wrote {OUT}")
