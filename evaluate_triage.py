"""How accurate is the triage? Test it on comments that humans already labelled.

EDOS has a human label (sexist / not sexist) for every comment. We use only its
official TEST split, which the models never saw while training, so the numbers are honest.
Run from the project root:
    python evaluate_triage.py            (about a minute)
"""
import argparse
from collections import defaultdict
from pathlib import Path

from src.triage import read_rows, triage_rows

FLAGGED_ACTIONS = ("hide_report", "spam", "review")

SOURCES = [("threatening", "threat rule"), ("spam", "spam rule"), ("hate-speech", "hate-speech model"),
           ("agree", "toxicity + sexism models agree"), ("insulting", "insult word rule"),
           ("stereotype", "sexist phrase rule"), ("sexism score", "sexism model alone"),
           ("toxicity score", "toxicity model alone")]


def source_of(reason: str) -> str:
    return next((name for key, name in SOURCES if key in reason), reason)


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("data/raw/edos.csv"))
    parser.add_argument("--max", type=int, default=3000, help="comments to test (keeps it quick)")
    parser.add_argument("--out", type=Path, default=Path("reports/triage_evaluation.md"))
    args = parser.parse_args()

    _, rows, text_col = read_rows(args.file.read_text(encoding="utf-8", errors="replace"))
    test = [r for r in rows if r.get("split") == "test"][: args.max]
    results = triage_rows(test, text_col)

    flagged = [r for r in results if r["action"] in FLAGGED_ACTIONS]
    sexist = [r for r in results if r["label_sexist"] == "sexist"]
    caught = [r for r in sexist if r["action"] in FLAGGED_ACTIONS]
    hit = [r for r in flagged if r["label_sexist"] == "sexist"]
    n = len(results)

    by_reason = defaultdict(lambda: [0, 0])  # reason -> [flagged, truly sexist]
    for r in flagged:
        key = source_of(r["reasons"])
        by_reason[key][0] += 1
        by_reason[key][1] += r["label_sexist"] == "sexist"

    lines = [
        "# Triage evaluation on the EDOS test split", "",
        f"{n} comments never seen in training. {len(sexist)} ({pct(len(sexist), n)}) are truly sexist.", "",
        "| Measure | Result |", "|---|---|",
        f"| Comments flagged (hide / spam / review) | {len(flagged)} ({pct(len(flagged), n)}) |",
        f"| Truly sexist comments that were flagged (recall) | {len(caught)} of {len(sexist)} ({pct(len(caught), len(sexist))}) |",
        f"| Flagged comments that are truly sexist (precision) | {len(hit)} of {len(flagged)} ({pct(len(hit), len(flagged))}) |",
        "", "## Which part of the tool flagged what", "", "| Part | Flagged | Truly sexist |", "|---|---|---|",
    ]
    for key, (flag_n, sexist_n) in sorted(by_reason.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| {key} | {flag_n} | {sexist_n} ({pct(sexist_n, flag_n)}) |")
    lines += ["", "_Flagged comments that are not labelled sexist may still be rude or hateful, "
              "so precision here is a strict, low estimate._"]
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())