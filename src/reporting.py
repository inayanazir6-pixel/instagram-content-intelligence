"""Turn triaged comments into a short written report (Markdown).

Everything here is plain counting and templates, so the same input always gives
the same report (no dates, no randomness) and it runs anywhere, including automatically.
"""
import csv
from collections import Counter, defaultdict

from src import config
from src.triage import ACTIONS

TOPIC_NAMES = {
    "gender_equality": "Gender equality", "womens_education": "Women's education",
    "employment": "Employment", "independence": "Independence",
    "harassment_violence": "Harassment & violence", "traditional_roles": "Traditional gender roles",
    "untagged": "No topic matched",
}
MIN_TOPIC_COMMENTS = 5  # smaller topics are too small to say anything about
FLAGGED = ("hide_report", "spam", "review")


def load_recommendations() -> dict:
    """The Stage-3 playbook made on Day 4, keyed by topic name (empty if the file is missing)."""
    path = config.GOLD_DIR / "recommendations.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["topic"]: r for r in csv.DictReader(f)}


def summarize(rows: list) -> dict:
    n = len(rows)
    actions = Counter(r["action"] for r in rows)
    reactions = Counter(r["reaction"] for r in rows)
    per_topic = defaultdict(lambda: {"n": 0, "flagged": 0})
    for r in rows:
        for t in [t.strip() for t in r["topics"].split(",") if t.strip()]:
            per_topic[t]["n"] += 1
            per_topic[t]["flagged"] += r["action"] in FLAGGED
    topics = [
        {"topic": t, "n": v["n"], "flagged": v["flagged"], "flagged_pct": 100 * v["flagged"] / v["n"]}
        for t, v in per_topic.items() if t != "untagged" and v["n"] >= MIN_TOPIC_COMMENTS
    ]
    topics.sort(key=lambda d: (-d["flagged_pct"], -d["n"]))
    flagged = sum(actions[a] for a in FLAGGED)
    return {
        "n": n, "actions": actions, "reactions": reactions, "topics": topics, "flagged": flagged,
        "health": 100 * (n - flagged) / n if n else 0.0,
    }


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "0%"


def build_report(rows: list, text_col: str, title: str = "Comment report") -> str:
    s = summarize(rows)
    n, actions = s["n"], s["actions"]
    if n == 0:
        return f"# {title}\n\nNo comments found."
    recs = load_recommendations()

    lines = [f"# {title}", "", f"**{n} comments analysed.** Conversation health: **{s['health']:.0f}%** "
             f"of comments need no moderation ({s['flagged']} flagged).", "", "## Recommended actions", "",
             "| Action | Comments | Share |", "|---|---|---|"]
    for key, (_, label) in sorted(ACTIONS.items(), key=lambda kv: kv[1][0]):
        lines.append(f"| {label} | {actions[key]} | {_pct(actions[key], n)} |")

    lines += ["", "## What stands out", ""]
    if actions["hide_report"]:
        lines.append(f"- **{actions['hide_report']} comment(s)** look abusive or threatening. Hide or report them first.")
    if actions["review"]:
        lines.append(f"- **{actions['review']} comment(s)** may be sexist or toxic. A person should look at them.")
    if actions["spam"]:
        lines.append(f"- **{actions['spam']} spam comment(s)**: delete them.")
    if actions["question"]:
        lines.append(f"- **{actions['question']} question(s)** are waiting for an answer.")
    if actions["reply"]:
        lines.append(f"- **{actions['reply']} critical but civil comment(s)** could be answered with a calm, factual reply.")
    if actions["thank"]:
        lines.append(f"- **{actions['thank']} supportive comment(s)**: thank the authors or pin the best ones.")
    if actions["unsure"]:
        lines.append(f"- **{actions['unsure']} comment(s)** were unclear to the models, so check them by eye.")
    if not any(actions[a] for a in ("hide_report", "spam", "review", "question", "reply", "thank", "unsure")):
        lines.append("- Nothing needs attention.")

    if s["topics"]:
        lines += ["", "## Topics", "", "| Topic | Comments | Flagged | Flagged share |", "|---|---|---|---|"]
        for t in s["topics"]:
            lines.append(f"| {TOPIC_NAMES.get(t['topic'], t['topic'])} | {t['n']} | {t['flagged']} | {t['flagged_pct']:.0f}% |")
        worst = s["topics"][0]
        if worst["flagged"]:
            name = TOPIC_NAMES.get(worst["topic"], worst["topic"])
            lines += ["", f"**Most problematic topic: {name}** ({worst['flagged_pct']:.0f}% of its comments flagged).", ""]
            rec = recs.get(worst["topic"])
            if rec:
                lines += [f"Idea to test for this topic: {rec['recommendation']}", "",
                          f"_Evidence from the research data: {rec['evidence']}_"]

    urgent = [r for r in rows if r["action"] in FLAGGED][:10]
    if urgent:
        lines += ["", "## Comments to look at first", ""]
        for r in urgent:
            text = r[text_col].replace("\n", " ")
            lines.append(f"- {ACTIONS[r['action']][1]}: \"{text[:140]}\" ({r['reasons']})")

    lines += ["", "---", "_Automatic triage with models trained on public tweets. The models make mistakes: "
              "use this as a first sort, and let a person make the final decision._"]
    return "\n".join(lines) + "\n"
