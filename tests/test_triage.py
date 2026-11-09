"""Tests for triage and reporting, using tiny fake models so results are predictable.
Run with:  python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config, scoring
from src.reporting import build_report, summarize
from src.triage import Thresholds, read_rows, rows_to_csv, triage_rows

VOCAB = ["idiot", "love", "kitchen", "equal", "wrong"]
STOP = ["this", "is", "and", "the", "in", "are", "will", "i", "you", "women", "belong"]


def fake_model(task, labels, coef, intercepts):
    return {"task": task, "labels": labels, "stop_words": STOP, "vocabulary": VOCAB,
            "idf": [1.0] * len(VOCAB), "coefficients": coef, "intercepts": intercepts}


MODELS = {
    "toxicity": fake_model("toxicity", ["neither", "offensive", "hate_speech"],
                           [[-8, 4, 0, 0, 0], [8, -4, 0, 0, 0], [0, 0, 0, 0, 0]], [3, 0, -6]),
    "sexism": fake_model("sexism", ["not sexist", "sexist"],
                         [[0, 0, -6, 0, 0], [0, 0, 6, 0, 0]], [3, 0]),
    "stance": fake_model("stance", ["against", "favor", "none"],
                         [[0, 0, 0, 0, 6], [0, 6, 0, 6, 0], [0, 0, 0, 0, 0]], [0, 0, 1]),
    "sexism_type": fake_model("sexism_type", ["derogation", "animosity"], [[0] * 5, [0] * 5], [1, 0]),
}


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for task, payload in MODELS.items():
            (Path(self.tmp.name) / f"{task}.json").write_text(json.dumps(payload))
        patcher = mock.patch.object(config, "EXPORT_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        scoring.load_model.cache_clear()
        self.addCleanup(scoring.load_model.cache_clear)

    def action(self, text):
        return triage_rows([{"comment": text}], "comment")[0]["action"]

    def test_each_action(self):
        self.assertEqual(self.action("i will kill you"), "hide_report")          # threat wording
        self.assertEqual(self.action("kitchen idiot"), "hide_report")            # toxic + sexist agree
        self.assertEqual(self.action("women belong in the kitchen"), "review")   # sexism alone
        self.assertEqual(self.action("you idiot"), "review")                     # toxicity alone
        self.assertEqual(self.action("this is wrong and unfair"), "reply")       # critical, civil
        self.assertEqual(self.action("equal rights are great"), "thank")         # supportive
        self.assertEqual(self.action("zzz qqq"), "ok")                           # neutral
        self.assertEqual(self.action("ok"), "ok")                                # too short

    def test_word_rules(self):
        self.assertEqual(self.action("follow me for free followers, click the link in my bio"), "spam")
        self.assertEqual(self.action("go back home please"), "review")           # sexist phrase
        self.assertEqual(self.action("what an idiot"), "review")                 # insult
        self.assertEqual(self.action("Is the event free to attend?"), "question")
        self.assertEqual(self.action("Thank you so much for sharing this"), "thank")

    def test_unsure_when_models_are_not_confident(self):
        rows = triage_rows([{"comment": "zzz qqq"}], "comment", Thresholds(confidence=0.9))
        self.assertEqual(rows[0]["action"], "unsure")

    def test_most_urgent_first_and_empty_rows_skipped(self):
        rows = triage_rows([{"comment": "equal rights are great"}, {"comment": ""},
                            {"comment": "i will kill you"}], "comment")
        self.assertEqual([r["action"] for r in rows], ["hide_report", "thank"])

    def test_read_rows_finds_text_column_and_keeps_other_columns(self):
        fields, rows, col = read_rows("user,Comment,likes\nann,great post,3\n")
        self.assertEqual((fields, col, rows[0]["likes"]), (["user", "Comment", "likes"], "Comment", "3"))

    def test_csv_and_report(self):
        fields, rows, col = read_rows("user,comment\na,women belong in the kitchen\nb,equal rights are great\n"
                                      "c,i will kill you\nd,this is wrong and unfair\n")
        result = triage_rows(rows, col)
        csv_text = rows_to_csv(fields, result)
        self.assertTrue(csv_text.startswith("user,comment,action,"))
        s = summarize(result)
        self.assertEqual((s["n"], s["flagged"]), (4, 2))
        report = build_report(result, col, "Test")
        for part in ("# Test", "4 comments analysed", "Hide / report", "Comments to look at first"):
            self.assertIn(part, report)
        self.assertEqual(report, build_report(result, col, "Test"))  # same input, same report


if __name__ == "__main__":
    unittest.main()
