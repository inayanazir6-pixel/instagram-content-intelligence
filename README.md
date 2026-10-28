# Instagram Content Intelligence

A data-driven system that analyses how audiences react to **gender and women's-rights content**:
it classifies reactions, shows which topics attract the most negative or sexist responses, and turns those
patterns into messaging ideas to test. The data pipeline is built with **PySpark**; the results are served
in a **Streamlit** web app.

**Live app:** _paste your Streamlit link here_

## What it does

| Stage | Question | How |
|---|---|---|
| 1. Reactions | Is a comment supportive, neutral, critical or toxic? | Stance, toxicity and sexism classifiers (Spark ML) |
| 2. Why negative? | Which topics and argument types dominate negative reactions? | Topic tagging (UDF) + aggregation by topic and sexism type |
| 3. Recommendations | How could a post be worded? | Rule-based playbook linked to the observed patterns (hypotheses to test) |

## Data

No public Instagram dataset exists for this topic, so three public English datasets are used (see `data/README.md`):
**Davidson** hate speech (24.8k tweets), **EDOS** sexism (20k comments), **TweetEval** feminism stance (~1k tweets).
Raw files are not committed; download instructions are in `data/README.md`.

## Pipeline (PySpark)

```
data/raw (CSV) -> bronze (typed Parquet) -> silver (cleaned, partitioned) -> gold (small CSV) + quarantine
```

| Day | Script | PySpark features shown |
|---|---|---|
| 1 | `run_day1.py` | explicit schemas, PERMISSIVE parsing, lineage columns, one-pass validation report (PASS/WARN/FAIL) |
| 2 | `run_day2.py` | regexp cleaning, window functions (dedupe), quarantine, **RDD** word counts with a broadcast variable, Spark SQL, partitioned Parquet |
| 3 | `run_day3.py` | ML Pipeline (Tokenizer, StopWordsRemover, NGram, CountVectorizer, IDF, LogisticRegression), CrossValidator, class weights, model persistence |
| 4 | `run_day4.py` | UDF topic tagging, explode, joins/broadcast joins, groupBy aggregations, batch scoring with saved models |
| 5 | `check_parity.py` | proves the app's plain-Python scorer matches Spark |

## Results (held-out test data)

| Model | Accuracy | Majority-class baseline | Macro-F1 |
|---|---|---|---|
| Toxicity (Davidson) | 0.840 | 0.770 | 0.685 |
| Sexism yes/no (EDOS) | 0.775 | 0.757 | 0.689 |
| Sexism type (EDOS) | 0.467 | 0.468 | 0.421 |
| Stance (TweetEval) | 0.491 | 0.647 | 0.442 |

Key pattern (observed labels): comments about **gender equality / feminism** are sexist 46% of the time versus
24% overall (lift 1.89); employment comments follow (lift 1.32). Derogation (insults) is the most common
type of sexism in most topics.

## Run it

```bash
# Web app only
pip install -r requirements.txt
streamlit run streamlit_app.py

# Full pipeline (needs Java; works in GitHub Codespaces)
pip install -r requirements-spark.txt
python run_day1.py && python run_day2.py && python run_day3.py && python run_day4.py
python check_parity.py
python -m unittest discover -s tests -v
```

## Project layout

```
src/            config, schemas, ingest, validate, text_cleaning, transform, rdd_analysis, eda,
                modeling, export, lexicon, topics, analysis, insights, scoring (Spark-free)
data/gold/      result tables read by the app
models/exports/ trained models as JSON (used by the app)
streamlit_app.py  the web app          tests/  unit tests          reports/  validation report
```

## Limitations

- **Not Instagram data.** All datasets are English tweets; results may not transfer to Instagram.
- **Topics are rule-based** (keyword lists), not human-labelled; most comments match no topic.
- Gender-equality keywords overlap with sexist comments, which inflates that topic's lift.
- Stance labels are noisy; the sexism-type and stance models are barely better than guessing.
- No dataset links posts to comments, so recommendations are **hypotheses to test**, not measured effects.