"""Word-frequency analysis written with the low-level RDD API.

Pipeline:  rows -> flatMap -> ((source, label, word), 1) -> reduceByKey -> counts
A broadcast variable ships the stop-word set to every worker once.
"""
import logging
from operator import add

from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src import config

log = logging.getLogger(__name__)


def top_words_by_class(spark: SparkSession, df: DataFrame, top_n: int = 15, min_len: int = 3):
    """Return (DataFrame of top words per source+label, list of top-10 words overall)."""
    sc = spark.sparkContext
    stop_words = set(StopWordsRemover.loadDefaultStopWords("english")) | config.EXTRA_STOPWORDS
    stop_bc = sc.broadcast(stop_words)  # broadcast variable

    def to_pairs(row):
        """One comment -> many ((source, label, word), 1) pairs."""
        for word in row["text_clean"].split():
            if len(word) >= min_len and word not in stop_bc.value:
                yield ((row["source"], row["label"], word), 1)

    rows_rdd = df.select("source", "label", "text_clean").rdd
    log.info("RDD partitions: %d", rows_rdd.getNumPartitions())

    counts = rows_rdd.flatMap(to_pairs).reduceByKey(add).cache()  # transformations

    # Action 1: the 10 most frequent words across everything
    overall = (
        counts.map(lambda kv: (kv[0][2], kv[1]))
        .reduceByKey(add)
        .takeOrdered(10, key=lambda kv: -kv[1])
    )

    # RDD -> DataFrame, then rank the words inside each group with a window function
    words_df = spark.createDataFrame(
        counts.map(lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[1])),
        ["source", "label", "word", "n"],
    )
    w = Window.partitionBy("source", "label").orderBy(F.desc("n"), "word")
    top = (
        words_df.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= top_n)
        .orderBy("source", "label", "rank")
    )
    return top, overall