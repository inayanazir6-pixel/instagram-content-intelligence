"""Comment triage: turn model scores into a recommended action for every comment.

No single model is trusted on its own (they are imperfect). Strong actions need
agreement between models, unsure comments go to a human, and every decision
comes with the reason it was made. Thresholds can be changed by the user.
"""
import csv
import io
import re
from dataclasses import dataclass

from src.scoring import analyze_comment, clean_text

# action key -> (queue order, label shown to the user)
ACTIONS = {
    "hide_report": (1, "🚨 Hide / report"),
    "spam": (2, "🗑️ Spam"),
    "review": (3, "⚠️ Review"),
    "question": (4, "❓ Answer question"),
    "reply": (5, "💬 Reply calmly"),
    "thank": (6, "💚 Thank / pin"),
    "unsure": (7, "🔍 Human check"),
    "ok": (8, "✅ No action"),
}
# Starting points for replies (edit the wording before posting)
REPLY_TEMPLATES = {
    "hide_report": "No reply. Hide the comment and report it to the platform.",
    "spam": "No reply. Delete it as spam.",
    "review": "Read it first. If it breaks your rules, hide it; do not argue.",
    "question": "Thanks for asking! [your answer]. Tell me if anything is still unclear.",
    "reply": "Thanks for sharing your view. Here is the data behind our post: [link]. Happy to keep talking respectfully.",
    "thank": "Thank you so much for your support! 💜",
    "unsure": "Read it and decide.",
    "ok": "",
}
STANCE_NAMES = {"favor": "Supportive", "none": "Neutral", "against": "Critical"}
TEXT_COLUMNS = ("comment", "comments", "text", "comment_text", "body", "message", "content")

# ---- Wording rules --------------------------------------------------------------
# The models are imperfect, so clear-cut cases are also caught by simple word rules
# (real moderation tools combine both). They run on the cleaned, lower-case text.
THREAT_PATTERN = re.compile(
    r"\b(?:kill|rape|murder|beat|hurt|stab|shoot|burn)\s+(?:you|her|them|women|girls)\b"
    r"|\bdeserve\s+to\s+(?:die|be\s+(?:raped|beaten|killed))\b"
)
SPAM_PATTERN = re.compile(
    r"\b(?:follow me|free followers|click (?:the )?link|link in (?:my )?bio|dm me|"
    r"check (?:out )?my (?:page|profile)|promo code|giveaway)\b"
)
INSULT_PATTERN = re.compile(
    r"\b(?:idiots?|stupid|dumb|morons?|ugly|trash|garbage|losers?|pathetic|disgusting|scum|brainless|shut up)\b"
)
SEXIST_PATTERN = re.compile(
    r"\bbelongs? in the kitchen\b|\bgo back (?:to the kitchen|home)\b|\bmake me a sandwich\b"
    r"|\btoo emotional\b|\btypical (?:woman|women|female|girl)\b"
    r"|\bonly because (?:she is|she's|she was) a (?:woman|girl)\b|\byou women\b|\bstop whining\b"
    r"|\bcan women even\b|\bwomen (?:can't|cannot) (?:drive|lead|think)\b"
    r"|\bwomen are (?:too |just |so )?(?:weak|emotional|stupid|inferior|irrational)\b"
    r"|\bknow your place\b"
)
POSITIVE_PATTERN = re.compile(
    r"\b(?:thank you|thanks for|thanks|congratulations|congrats|inspiring|proud of|great post|great work|"
    r"wonderful|well deserved|love this|so important|well said|couldn't agree more|gives me hope)\b"
)


@dataclass
class Thresholds:
    hate: float = 0.50            # hate-speech probability that triggers hide/report
    toxic_hide: float = 0.85      # toxic AND sexist together -> hide/report
    sexist_hide: float = 0.65
    sexist_review: float = 0.70   # sexism model alone -> review
    toxic_review: float = 0.90    # toxicity model alone -> review
    confidence: float = 0.42      # below this stance confidence -> human check
    reply_confidence: float = 0.60
    thank_confidence: float = 0.60


def decide(analysis: dict, text_clean: str, th: Thresholds, text_raw: str = "") -> dict:
    """Return the action, reaction and reasons for one analysed comment."""
    base = {"topics": analysis["topics"], "text_clean": text_clean}
    if analysis.get("too_short"):
        return {**base, "action": "ok", "reaction": "Too short", "reasons": ["too short to judge"],
                "p_toxic": 0.0, "p_hate": 0.0, "p_sexist": 0.0, "stance": "", "stance_conf": 0.0,
                "sexism_type": ""}

    tox = analysis["toxicity"]["probabilities"]
    p_hate = tox.get("hate_speech", 0.0)
    p_toxic = 1 - tox.get("neither", 0.0)
    p_sexist = analysis["sexism"]["probabilities"].get("sexist", 0.0)
    stance_key = analysis["stance"]["label"]
    stance_conf = analysis["stance"]["probabilities"][stance_key]
    sexism_type = analysis.get("sexism_type", {}).get("label", "")

    threat = bool(THREAT_PATTERN.search(text_clean))
    insult = bool(INSULT_PATTERN.search(text_clean))
    if threat:
        action, reasons = "hide_report", ["threatening wording"]
    elif SPAM_PATTERN.search(text_clean):
        action, reasons = "spam", ["looks like spam or self-promotion"]
    elif p_hate >= th.hate:
        action, reasons = "hide_report", [f"hate-speech score {p_hate:.0%}"]
    elif p_toxic >= th.toxic_hide and p_sexist >= th.sexist_hide:
        action, reasons = "hide_report", [f"toxic {p_toxic:.0%} and sexist {p_sexist:.0%} agree"]
    elif insult:
        action, reasons = "review", ["insulting or abusive wording"]
    elif SEXIST_PATTERN.search(text_clean):
        action, reasons = "review", ["sexist stereotype wording"]
    elif p_sexist >= th.sexist_review:
        action, reasons = "review", [f"sexism score {p_sexist:.0%}"]
    elif p_toxic >= th.toxic_review:
        action, reasons = "review", [f"toxicity score {p_toxic:.0%}"]
    elif "?" in (text_raw or ""):
        action, reasons = "question", ["contains a question"]
    elif POSITIVE_PATTERN.search(text_clean):
        action, reasons = "thank", ["positive wording"]
    elif stance_conf < th.confidence:
        action, reasons = "unsure", [f"models unsure ({stance_conf:.0%})"]
    elif stance_key == "against" and stance_conf >= th.reply_confidence:
        action, reasons = "reply", ["critical but not abusive: a calm, factual reply may help"]
    elif stance_key == "favor" and stance_conf >= th.thank_confidence:
        action, reasons = "thank", ["supportive comment"]
    else:
        action, reasons = "ok", ["nothing concerning"]

    toxic_reaction = threat or insult or p_hate >= th.hate or p_toxic >= th.toxic_review
    return {
        **base, "action": action, "reasons": reasons,
        "reaction": "Toxic" if toxic_reaction else STANCE_NAMES[stance_key],
        "p_toxic": p_toxic, "p_hate": p_hate, "p_sexist": p_sexist,
        "stance": STANCE_NAMES[stance_key], "stance_conf": stance_conf, "sexism_type": sexism_type,
    }


# ---- Reading and writing CSV files -------------------------------------------
def read_rows(csv_text: str):
    """Parse CSV text. Returns (column names, rows as dicts, name of the comment column)."""
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("﻿")))
    rows = list(reader)
    fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError("The file is empty.")
    lowered = {f.strip().lower(): f for f in fields}
    text_col = next((lowered[c] for c in TEXT_COLUMNS if c in lowered), fields[0])
    return fields, rows, text_col


RESULT_COLUMNS = ["action", "reaction", "reasons", "p_toxic", "p_hate", "p_sexist",
                  "stance", "stance_confidence", "sexism_type", "topics", "suggested_reply"]


def triage_rows(rows: list, text_col: str, th: Thresholds = None) -> list:
    """Analyse every row and return new row dicts (original columns + results), most urgent first."""
    th = th or Thresholds()
    out = []
    for row in rows:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        analysis = analyze_comment(text, top_k=0)
        d = decide(analysis, clean_text(text), th, text)
        out.append({
            **row,
            "action": d["action"], "reaction": d["reaction"], "reasons": "; ".join(d["reasons"]),
            "p_toxic": round(d["p_toxic"], 3), "p_hate": round(d["p_hate"], 3),
            "p_sexist": round(d["p_sexist"], 3), "stance": d["stance"],
            "stance_confidence": round(d["stance_conf"], 3), "sexism_type": d["sexism_type"],
            "topics": ", ".join(d["topics"]), "suggested_reply": REPLY_TEMPLATES[d["action"]],
        })
    out.sort(key=lambda r: ACTIONS[r["action"]][0])  # stable: keeps the original order inside a group
    return out


def rows_to_csv(fields: list, rows: list) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields + RESULT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
