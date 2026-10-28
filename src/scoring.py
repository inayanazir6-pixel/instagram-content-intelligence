"""Score a comment with plain Python (no Spark, no Java).

The deployed web app cannot run Spark, so this module repeats - step by step -
what the Spark pipeline does, using the JSON files exported on Day 3:
  clean text -> tokenize -> drop stop words -> add word pairs -> count vocabulary
  terms -> multiply by IDF -> logistic regression -> softmax probabilities.
`check_parity.py` proves that the answers match Spark's.
"""
import json
import math
import re
from functools import lru_cache

from src import config
from src.lexicon import TOPIC_KEYWORDS

_A = re.ASCII  # make \w \s \b behave like Java's (the Spark side)

# ---- 1. Cleaning (mirror of src/text_cleaning.py) ----------------------------
_ENTITIES = [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')]


def clean_text(text: str) -> str:
    t = text or ""
    for entity, char in _ENTITIES:
        t = t.replace(entity, char)
    t = re.sub(r"&#?\w+;", " ", t, flags=_A)
    t = re.sub("[\u2018\u2019]", "'", t)
    t = re.sub(r"\bRT\b", " ", t, flags=_A)
    t = re.sub(r"https?://\S+|www\.\S+|\[URL\]", " ", t, flags=_A)
    t = re.sub(r"@\w+|\[USER\]", " ", t, flags=_A)
    t = re.sub(r"(?i)#semst\b", " ", t, flags=_A)
    t = re.sub(r"#(\w+)", r"\1", t, flags=_A)
    t = t.lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^a-z0-9'\s]", " ", t, flags=_A)
    t = re.sub(r"\s+", " ", t, flags=_A)
    return t.strip()


# ---- 2. Topics (mirror of src/topics.py) -------------------------------------
_TOPIC_PATTERNS = {
    topic: re.compile(r"\b(?:" + "|".join(words) + r")\b", _A)
    for topic, words in TOPIC_KEYWORDS.items()
}


def tag_topics(text_clean: str) -> list:
    return [t for t, p in _TOPIC_PATTERNS.items() if p.search(text_clean or "")]


# ---- 3. One exported model ---------------------------------------------------
class TextModel:
    def __init__(self, payload: dict):
        self.task = payload["task"]
        self.labels = payload["labels"]
        self.stop_words = {w.lower() for w in payload["stop_words"]}
        self.vocab_index = {term: i for i, term in enumerate(payload["vocabulary"])}
        self.idf = payload["idf"]
        self.coef = payload["coefficients"]      # classes x vocabulary
        self.intercepts = payload["intercepts"]

    def terms(self, text_clean: str) -> list:
        tokens = [t for t in text_clean.split() if t]
        kept = [t for t in tokens if t not in self.stop_words]
        bigrams = [f"{a} {b}" for a, b in zip(kept, kept[1:])]
        return kept + bigrams

    def features(self, text_clean: str) -> dict:
        """TF-IDF of the terms that are in the vocabulary: {vocab index: value}."""
        counts = {}
        for term in self.terms(text_clean):
            j = self.vocab_index.get(term)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        return {j: c * self.idf[j] for j, c in counts.items()}

    def predict(self, text_clean: str, top_k: int = 5) -> dict:
        x = self.features(text_clean)
        logits = [b + sum(row[j] * v for j, v in x.items())
                  for row, b in zip(self.coef, self.intercepts)]
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        best = max(range(len(probs)), key=probs.__getitem__)
        vocab_terms = {j: t for t, j in self.vocab_index.items()}
        pushes = sorted(((self.coef[best][j] * v, vocab_terms[j]) for j, v in x.items()), reverse=True)
        return {
            "label": self.labels[best],
            "probabilities": dict(zip(self.labels, probs)),
            "top_terms": [t for s, t in pushes[:top_k] if s > 0],
        }


@lru_cache(maxsize=None)
def load_model(task: str) -> TextModel:
    path = config.EXPORT_DIR / f"{task}.json"
    return TextModel(json.loads(path.read_text(encoding="utf-8")))


# ---- 4. Everything for one comment -------------------------------------------
def analyze_comment(text: str) -> dict:
    cleaned = clean_text(text)
    result = {"text_clean": cleaned, "topics": tag_topics(cleaned) or ["untagged"]}
    if len(cleaned.split()) < config.MIN_WORDS:
        result["too_short"] = True
        return result
    result["too_short"] = False
    for task in ("toxicity", "sexism", "stance"):
        result[task] = load_model(task).predict(cleaned)
    if result["sexism"]["label"] == "sexist":
        result["sexism_type"] = load_model("sexism_type").predict(cleaned)
    return result