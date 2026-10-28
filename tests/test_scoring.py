"""Small unit tests: run with   python -m unittest discover -s tests -v"""
import math
import unittest

from src.scoring import TextModel, clean_text, tag_topics


class CleaningTests(unittest.TestCase):
    def test_removes_noise(self):
        self.assertEqual(clean_text("RT @bob: Soooo GOOD!!! http://x.co #Feminism &amp; more"),
                         "soo good feminism more")

    def test_numeric_entities_removed(self):
        self.assertEqual(clean_text("so funny &#128514; ok"), "so funny ok")

    def test_empty(self):
        self.assertEqual(clean_text(None), "")


class TopicTests(unittest.TestCase):
    def test_topics(self):
        self.assertEqual(tag_topics("women need equal pay and education"),
                         ["gender_equality", "womens_education"])
        self.assertEqual(tag_topics("i like pizza"), [])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = TextModel({
            "task": "demo", "labels": ["pos", "neg"], "stop_words": ["the", "a"],
            "vocabulary": ["good", "bad", "good day"], "idf": [1.0, 2.0, 3.0],
            "coefficients": [[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5]], "intercepts": [0.1, -0.1],
        })

    def test_terms_drop_stopwords_and_add_pairs(self):
        self.assertEqual(self.model.terms("the good day"), ["good", "day", "good day"])

    def test_probabilities_sum_to_one_and_match_hand_calculation(self):
        out = self.model.predict("the good day")
        # features: good=1*1.0, "good day"=1*3.0 -> logit_pos = 0.1+1.0+1.5 = 2.6, logit_neg = -0.1-1.0-1.5 = -2.6
        expected = 1 / (1 + math.exp(-5.2))
        self.assertAlmostEqual(out["probabilities"]["pos"], expected, places=9)
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0)
        self.assertEqual(out["label"], "pos")
        self.assertEqual(out["top_terms"][0], "good day")

    def test_unknown_words_use_intercept_only(self):
        out = self.model.predict("zzz yyy")
        self.assertEqual(out["label"], "pos")


if __name__ == "__main__":
    unittest.main()