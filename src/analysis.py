"""Day 4 analysis: where do negative reactions happen, and what kind are they?

Two kinds of evidence are used, and they are kept apart on purpose:
  * OBSERVED  - the human labels that come with each dataset (no model involved).
  * PREDICTED - our Day 3 models applied to the stance comments (a dataset they
                were never trained on), to describe WHY critical comments are critical.
"""
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config, modeling
from src.topics import explode_topics

MIN_N = 30  # topic rows with fewer comments are flagged low_n (too small to trust)


def _is_negative(df: DataFrame) -> DataFrame:
    """is_negative = 1 for hateful/offensive (Davidson), sexist (EDOS), against (stance)."""
    label = F.col("label")
    cond = (
        F.when(F.col("source") == "davidson", label.isin("hate_speech", "offensive"))
        .when(F.col("source") == "edos", label == "sexist")
        .otherwise(label == "against")
    )
    return df.withColumn("is_negative", cond.cast("int"))


def topic_coverage(df: DataFrame) -> DataFrame:
    """How many comments mention each topic, per source?"""
    totals = df.groupBy("source").agg(F.count("*").alias("source_total"))
    return (
        explode_topics(df).groupBy("source", "topic").agg(F.count("*").alias("n_comments"))
        .join(F.broadcast(totals), "source")
        .withColumn("pct_of_source", F.round(100 * F.col("n_comments") / F.col("source_total"), 1))
        .drop("source_total")
        .orderBy("source", F.desc("n_comments"))
    )


def negative_rate_by_topic(df: DataFrame) -> DataFrame:
    """OBSERVED share of negative comments per topic, compared with the source average."""
    df = _is_negative(df)
    overall = df.groupBy("source").agg(F.round(100 * F.avg("is_negative"), 1).alias("pct_negative_overall"))
    return (
        explode_topics(df)
        .groupBy("source", "topic")
        .agg(F.count("*").alias("n"), F.round(100 * F.avg("is_negative"), 1).alias("pct_negative"))
        .join(F.broadcast(overall), "source")
        .withColumn("lift", F.round(F.col("pct_negative") / F.col("pct_negative_overall"), 2))
        .withColumn("low_n", F.col("n") < MIN_N)
        .orderBy("source", F.desc("lift"))
    )


def sexism_type_by_topic(df: DataFrame) -> DataFrame:
    """OBSERVED: among sexist EDOS comments, which kind of sexism appears in each topic?"""
    sexist = df.filter((F.col("source") == "edos") & (F.col("label") == "sexist"))
    counts = explode_topics(sexist).groupBy("topic", "sub_label").agg(F.count("*").alias("n"))
    totals = counts.groupBy("topic").agg(F.sum("n").alias("topic_total"))
    return (
        counts.join(totals, "topic")
        .withColumn("pct_in_topic", F.round(100 * F.col("n") / F.col("topic_total"), 1))
        .withColumn("low_n", F.col("topic_total") < MIN_N)
        .orderBy("topic", F.desc("n"))
    )


def stance_by_topic(df: DataFrame) -> DataFrame:
    """OBSERVED: favor / none / against split inside each topic (stance dataset)."""
    counts = explode_topics(df.filter(F.col("source") == "stance")).groupBy("topic", "label").agg(F.count("*").alias("n"))
    totals = counts.groupBy("topic").agg(F.sum("n").alias("topic_total"))
    return (
        counts.join(totals, "topic")
        .withColumn("pct_in_topic", F.round(100 * F.col("n") / F.col("topic_total"), 1))
        .withColumn("low_n", F.col("topic_total") < MIN_N)
        .orderBy("topic", F.desc("n"))
    )


# ---- Predicted: apply Day 3 models to stance comments -------------------------
def add_prediction(df: DataFrame, task: str, out_col: str) -> DataFrame:
    """Add a column with the predicted class name. `df` must have text_clean and no `label` column."""
    model = PipelineModel.load(str(config.MODELS_DIR / task))
    labels = modeling.class_names(model)
    keep = df.columns
    scored = model.transform(df.withColumn("label", F.lit(labels[0])))  # dummy label: the indexer stage needs the column
    name = F.element_at(F.array(*[F.lit(x) for x in labels]), (F.col("prediction") + 1).cast("int"))
    return scored.select(*keep, name.alias(out_col))


def score_stance_comments(df: DataFrame) -> DataFrame:
    """Predict toxicity, sexism and sexism type for every stance comment."""
    base = (
        df.filter(F.col("source") == "stance")
        .select("comment_id", "text_clean", "topics", F.col("label").alias("stance"))
    )
    scored = add_prediction(base, "toxicity", "pred_toxicity")
    scored = add_prediction(scored, "sexism", "pred_sexism")
    scored = add_prediction(scored, "sexism_type", "pred_type_raw")
    return scored.withColumn(
        "pred_type", F.when(F.col("pred_sexism") == "sexist", F.col("pred_type_raw"))
    ).drop("pred_type_raw")


def why_critical_by_topic(scored: DataFrame) -> DataFrame:
    """Per topic and stance: share predicted toxic / sexist (model-based, so read with care)."""
    return (
        explode_topics(scored)
        .groupBy("topic", "stance")
        .agg(
            F.count("*").alias("n"),
            F.round(100 * F.avg((F.col("pred_toxicity") != "neither").cast("int")), 1).alias("pct_pred_toxic"),
            F.round(100 * F.avg((F.col("pred_sexism") == "sexist").cast("int")), 1).alias("pct_pred_sexist"),
        )
        .withColumn("low_n", F.col("n") < MIN_N)
        .orderBy("topic", "stance")
    )


def pred_type_by_topic(scored: DataFrame) -> DataFrame:
    """Among critical (against) comments predicted sexist: which argument type, per topic?"""
    sexist = scored.filter((F.col("stance") == "against") & F.col("pred_type").isNotNull())
    counts = explode_topics(sexist).groupBy("topic", "pred_type").agg(F.count("*").alias("n"))
    totals = counts.groupBy("topic").agg(F.sum("n").alias("topic_total"))
    return (
        counts.join(totals, "topic")
        .withColumn("pct_in_topic", F.round(100 * F.col("n") / F.col("topic_total"), 1))
        .withColumn("low_n", F.col("topic_total") < MIN_N)
        .orderBy("topic", F.desc("n"))
    )