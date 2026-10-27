"""Bronze -> Silver: one common table, cleaned text, and a quarantine for bad rows.

Silver table `comments` (one row per usable comment):
  comment_id, source, task, split, text_raw, text_clean, label, sub_label,
  word_count, char_count, is_retweet, has_url, n_hashtags, n_mentions, ...

`task` says which question the label answers:
  davidson -> "toxicity", edos -> "sexism", stance -> "stance"
"""
from functools import reduce

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src import config
from src.text_cleaning import add_text_features

_COMMON_COLS = [
    "source", "task", "split", "text_raw", "label", "sub_label",
    "source_file", "ingested_at", "_corrupt_record",
]


def _map_labels(col: Column, mapping: dict) -> Column:
    """Turn codes into names (0 -> 'hate_speech'). Unknown codes become null."""
    expr = None
    for code, name in mapping.items():
        expr = F.when(col == code, name) if expr is None else expr.when(col == code, name)
    return expr


def read_bronze(spark: SparkSession) -> dict:
    return {
        name: spark.read.parquet(str(config.BRONZE_DIR / name))
        for name in ("davidson", "edos", "stance")
    }


# ---- Make the three datasets look the same -----------------------------------
def standardize_davidson(df: DataFrame) -> DataFrame:
    return df.select(
        "source",
        F.lit("toxicity").alias("task"),
        F.lit(None).cast("string").alias("split"),  # Davidson has no official split
        F.col("text").alias("text_raw"),
        _map_labels(F.col("label"), config.DAVIDSON_LABELS).alias("label"),
        F.lit(None).cast("string").alias("sub_label"),
        "source_file", "ingested_at", "_corrupt_record",
    )


def standardize_edos(df: DataFrame) -> DataFrame:
    label = (
        F.when(F.col("label_sexist") == "sexist", "sexist")
        .when(F.col("label_sexist") == "not sexist", "not_sexist")
    )
    # "2. derogation" -> "derogation"
    sub_label = F.regexp_replace(
        F.lower(F.regexp_replace("label_category", r"^\d+\.\s*", "")), "[^a-z]+", "_"
    )
    split = F.when(F.col("split") == "dev", "validation").otherwise(F.col("split"))
    return df.select(
        "source",
        F.lit("sexism").alias("task"),
        split.alias("split"),
        F.col("text").alias("text_raw"),
        label.alias("label"),
        sub_label.alias("sub_label"),
        "source_file", "ingested_at", "_corrupt_record",
    )


def standardize_stance(df: DataFrame) -> DataFrame:
    return df.select(
        "source",
        F.lit("stance").alias("task"),
        "split",
        F.col("text").alias("text_raw"),
        _map_labels(F.col("label"), config.STANCE_LABELS).alias("label"),
        F.lit(None).cast("string").alias("sub_label"),
        "source_file", "ingested_at", "_corrupt_record",
    )


def build_comments(bronze: dict) -> DataFrame:
    """Stack the three datasets, add an id, clean the text, add features."""
    df = reduce(
        DataFrame.unionByName,
        [
            standardize_davidson(bronze["davidson"]).select(_COMMON_COLS),
            standardize_edos(bronze["edos"]).select(_COMMON_COLS),
            standardize_stance(bronze["stance"]).select(_COMMON_COLS),
        ],
    )
    # Stable id: the same comment always gets the same id, on every run.
    key = F.concat_ws(
        "||", F.col("source"), F.coalesce("split", F.lit("")), F.coalesce("text_raw", F.lit(""))
    )
    df = df.withColumn("comment_id", F.substring(F.sha2(key, 256), 1, 16))
    return add_text_features(df)


# ---- Quarantine --------------------------------------------------------------
def split_quarantine(df: DataFrame):
    """Return (clean rows, quarantined rows). Quarantined rows keep a reject_reason."""
    reason = (
        F.when(F.col("_corrupt_record").isNotNull(), "corrupt_record")
        .when(F.col("text_raw").isNull(), "missing_text")
        .when(F.col("label").isNull(), "missing_or_invalid_label")
        .when(F.col("word_count") < config.MIN_WORDS, "too_short_after_cleaning")
    )
    flagged = df.withColumn("reject_reason", reason)
    rejected = flagged.filter(F.col("reject_reason").isNotNull())
    kept = flagged.filter(F.col("reject_reason").isNull())

    # Window function: if several rows have the same cleaned text inside one
    # source, keep the first and quarantine the rest (also prevents the same text
    # from landing in both train and test later).
    w = Window.partitionBy("source", "text_clean").orderBy("comment_id")
    ranked = kept.withColumn("dup_rank", F.row_number().over(w))
    duplicates = (
        ranked.filter("dup_rank > 1")
        .drop("dup_rank")
        .withColumn("reject_reason", F.lit("duplicate_after_cleaning"))
    )
    clean = ranked.filter("dup_rank = 1").drop("dup_rank", "reject_reason", "_corrupt_record")
    quarantine = rejected.unionByName(duplicates)
    return clean, quarantine


def write_silver(clean: DataFrame, quarantine: DataFrame) -> None:
    clean.write.mode("overwrite").partitionBy("source").parquet(str(config.SILVER_DIR / "comments"))
    quarantine.write.mode("overwrite").parquet(str(config.QUARANTINE_DIR / "comments"))