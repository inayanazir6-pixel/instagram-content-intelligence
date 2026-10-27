"""Exploratory analysis. Every function returns a small DataFrame."""
import csv

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src import config


def label_distribution(df: DataFrame) -> DataFrame:
    """Rows and percentage per label, inside each source."""
    w = Window.partitionBy("source", "task")
    return (
        df.groupBy("source", "task", "label")
        .agg(F.count("*").alias("n"))
        .withColumn("pct", F.round(100 * F.col("n") / F.sum("n").over(w), 1))
        .orderBy("source", F.desc("n"))
    )


def length_stats(df: DataFrame) -> DataFrame:
    """How long are comments of each label? (in words)"""
    return (
        df.groupBy("source", "label")
        .agg(
            F.count("*").alias("n"),
            F.round(F.avg("word_count"), 1).alias("avg_words"),
            F.percentile_approx("word_count", 0.5).alias("median_words"),
            F.max("word_count").alias("max_words"),
            F.round(100 * F.avg(F.col("is_retweet").cast("int")), 1).alias("pct_retweets"),
        )
        .orderBy("source", "label")
    )


def sexism_breakdown(spark: SparkSession, df: DataFrame) -> DataFrame:
    """Spark SQL: which types of sexism are most common in EDOS?"""
    df.createOrReplaceTempView("comments")
    return spark.sql(
        """
        SELECT sub_label,
               COUNT(*) AS n,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_sexist
        FROM comments
        WHERE source = 'edos' AND label = 'sexist'
        GROUP BY sub_label
        ORDER BY n DESC
        """
    )


def quarantine_summary(quarantine: DataFrame) -> DataFrame:
    return (
        quarantine.groupBy("source", "reject_reason")
        .agg(F.count("*").alias("n"))
        .orderBy("source", F.desc("n"))
    )


def sample_per_label(df: DataFrame, source: str, per_label: int = 3, seed: int = 7) -> DataFrame:
    """A few random comments per label, so we can read them and sanity-check labels."""
    w = Window.partitionBy("label").orderBy(F.rand(seed))
    return (
        df.filter(F.col("source") == source)
        .withColumn("r", F.row_number().over(w))
        .filter(F.col("r") <= per_label)
        .select("label", "text_raw")
        .orderBy("label")
    )


def save_gold(df: DataFrame, name: str):
    """Save a small result table to data/gold/<name>.csv (the app will read these)."""
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GOLD_DIR / f"{name}.csv"
    rows = df.collect()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(df.columns)
        writer.writerows([list(r) for r in rows])
    return path