"""Try many sensitivity settings on the EDOS test split and show which balance works best.

The comments are analysed once; then every combination of settings is re-scored
(this is quick). Run from the project root:
    python tune_triage.py
"""
import argparse
from itertools import product
from pathlib import Path

from src.scoring import analyze_comment, clean_text
from src.triage import Thresholds, decide, read_rows

FLAGGED = ("hide_report", "spam", "review")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("data/raw/edos.csv"))
    parser.add_argument("--max", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=Path("reports/triage_tuning.md"))
    args = parser.parse_args()

    _, rows, text_col = read_rows(args.file.read_text(encoding="utf-8", errors="replace"))
    test = [r for r in rows if r.get("split") == "test" and (r.get(text_col) or "").strip()][: args.max]
    cache = [(analyze_comment(r[text_col], top_k=0), clean_text(r[text_col]), r[text_col],
              r["label_sexist"] == "sexist") for r in test]
    total_sexist = sum(c[3] for c in cache)

    def score(th: Thresholds) -> dict:
        flagged = hits = 0
        for analysis, clean, raw, is_sexist in cache:
            if decide(analysis, clean, th, raw)["action"] in FLAGGED:
                flagged += 1
                hits += is_sexist
        recall = hits / total_sexist if total_sexist else 0
        precision = hits / flagged if flagged else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        return {"flagged": 100 * flagged / len(cache), "recall": 100 * recall,
                "precision": 100 * precision, "f1": 100 * f1}

    results = []
    for hate, sexist, toxic in product([0.5, 0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.7, 0.8], [0.9, 0.95, 0.98, 1.01]):
        results.append(((hate, sexist, toxic), score(Thresholds(hate=hate, sexist_review=sexist, toxic_review=toxic))))
    current = score(Thresholds())
    results.sort(key=lambda item: -item[1]["f1"])

    lines = ["# Triage settings tried on the EDOS test split", "",
             f"{len(cache)} comments, {total_sexist} truly sexist. Ranked by F1 (balance of recall and precision).", "",
             "| Hate score | Sexism score | Toxicity score | Flagged | Recall | Precision | F1 |", "|---|---|---|---|---|---|---|"]
    for (hate, sexist, toxic), m in results[:8]:
        tox = "off" if toxic > 1 else toxic
        lines.append(f"| {hate} | {sexist} | {tox} | {m['flagged']:.0f}% | {m['recall']:.0f}% | {m['precision']:.0f}% | {m['f1']:.0f} |")
    lines += ["", f"**Current settings** (0.5 / 0.7 / 0.9): flagged {current['flagged']:.0f}%, recall {current['recall']:.0f}%, "
              f"precision {current['precision']:.0f}%, F1 {current['f1']:.0f}.", "",
              "_EDOS labels only sexism, so this favours settings that find sexism. Hateful or abusive comments "
              "that are not sexist count as mistakes here._"]
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())