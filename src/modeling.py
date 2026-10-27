"""Spark ML: build, tune and evaluate one text classifier per task.

Pipeline (all stages are Spark ML):
  StringIndexer -> Tokenizer -> StopWordsRemover -> NGram(2) -> SQLTransformer
  -> CountVectorizer -> IDF -> LogisticRegression (class-weighted)
"""
import logging

import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    CountVectorizer,
    IDF,
    NGram,
    SQLTransformer,
    StopWordsRemover,
    StringIndexer,
    Tokenizer,
)
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config

log = logging.getLogger(__name__)

# Positions of the stages inside the fitted PipelineModel
IDX_INDEXER, IDX_STOPWORDS, IDX_COUNTVEC, IDX_IDF, IDX_LR = 0, 2, 5, 6, 7


# ---- Data --------------------------------------------------------------------
def load_task_data(spark: SparkSession, task: str):
    """Return (train, test) with columns comment_id, split, text_clean, label."""
    spec = config.TASKS[task]
    df = spark.read.parquet(str(config.SILVER_DIR / "comments"))
    df = df.filter(F.col("source") == spec["source"])
    if spec["filter_label"]:
        df = df.filter(F.col("label") == spec["filter_label"])
    df = df.select("comment_id", "split", "text_clean", F.col(spec["label_col"]).alias("label"))

    if spec["split"] == "hash":
        # No official split: send ~20% of comments to the test set, decided by a
        # hash of comment_id, so the split is identical on every run.
        is_test = F.pmod(F.hash("comment_id"), F.lit(10)) >= 8
    else:
        is_test = F.col("split") == "test"
    return df.filter(~is_test), df.filter(is_test)


def add_class_weights(train: DataFrame) -> DataFrame:
    """weight = n_rows / (n_classes * n_rows_of_that_class): rare classes count more."""
    counts = train.groupBy("label").count()
    total, n_classes = train.count(), counts.count()
    weights = counts.select(
        "label", (F.lit(total) / (F.lit(n_classes) * F.col("count"))).alias("weight")
    )
    return train.join(F.broadcast(weights), "label")  # broadcast join: tiny table


# ---- Model -------------------------------------------------------------------
def _stop_words() -> list:
    """Spark's English stop words, but keep negations ("not", "don't")."""
    default = StopWordsRemover.loadDefaultStopWords("english")
    return [w for w in default if not (w in {"no", "nor", "not"} or w.endswith("n't"))]


def build_pipeline():
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label_idx",
        weightCol="weight",
        family="multinomial",
        maxIter=40,
    )
    stages = [
        StringIndexer(inputCol="label", outputCol="label_idx", stringOrderType="frequencyDesc"),
        Tokenizer(inputCol="text_clean", outputCol="tokens"),
        StopWordsRemover(inputCol="tokens", outputCol="filtered", stopWords=_stop_words()),
        NGram(n=2, inputCol="filtered", outputCol="bigrams"),
        SQLTransformer(statement="SELECT *, concat(filtered, bigrams) AS terms FROM __THIS__"),
        CountVectorizer(
            inputCol="terms", outputCol="tf", vocabSize=config.VOCAB_SIZE, minDF=2.0
        ),
        IDF(inputCol="tf", outputCol="features"),
        lr,
    ]
    return Pipeline(stages=stages), lr


def train(train_df: DataFrame) -> CrossValidatorModel:
    """3-fold cross-validation over the regularisation strength."""
    pipeline, lr = build_pipeline()
    grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_idx", predictionCol="prediction", metricName="f1"
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        seed=config.RANDOM_SEED,
        parallelism=2,
    )
    return cv.fit(train_df)


# ---- Evaluation --------------------------------------------------------------
def class_names(model: PipelineModel) -> list:
    return list(model.stages[IDX_INDEXER].labelsArray[0])


def evaluate(model: PipelineModel, test_df: DataFrame):
    """Return (summary dict, per-class list, confusion list) measured on the test set."""
    pred = model.transform(test_df).cache()
    labels = class_names(model)

    def metric(name: str, **kwargs) -> float:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label_idx", predictionCol="prediction", metricName=name, **kwargs
        )
        return float(evaluator.evaluate(pred))

    per_class = []
    for i, name in enumerate(labels):
        per_class.append({
            "label": name,
            "precision": round(metric("precisionByLabel", metricLabel=float(i)), 4),
            "recall": round(metric("recallByLabel", metricLabel=float(i)), 4),
            "f1": round(metric("fMeasureByLabel", metricLabel=float(i)), 4),
            "support": pred.filter(F.col("label_idx") == float(i)).count(),
        })

    n_test = sum(c["support"] for c in per_class)
    summary = {
        "n_test": n_test,
        "accuracy": round(metric("accuracy"), 4),
        "baseline_accuracy": round(max(c["support"] for c in per_class) / n_test, 4),
        "macro_f1": round(float(np.mean([c["f1"] for c in per_class])), 4),
        "weighted_f1": round(metric("f1"), 4),
    }

    confusion = [
        {
            "actual": labels[int(r["label_idx"])],
            "predicted": labels[int(r["prediction"])],
            "n": int(r["count"]),
        }
        for r in pred.groupBy("label_idx", "prediction").count().collect()
    ]
    pred.unpersist()
    return summary, per_class, confusion


def top_features(model: PipelineModel, top_n: int = 15) -> list:
    """The words and word pairs that push a comment towards each class."""
    labels = class_names(model)
    vocab = model.stages[IDX_COUNTVEC].vocabulary
    coef = model.stages[IDX_LR].coefficientMatrix.toArray()  # shape: classes x vocabulary
    rows = []
    for i, name in enumerate(labels):
        best = np.argsort(coef[i])[::-1][:top_n]
        for rank, j in enumerate(best, start=1):
            rows.append({
                "label": name,
                "rank": rank,
                "term": vocab[int(j)],
                "weight": round(float(coef[i][j]), 4),
            })
    return rows