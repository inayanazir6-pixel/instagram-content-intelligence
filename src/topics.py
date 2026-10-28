"""Topic tagging with a Python UDF (user-defined function).

A UDF lets us run ordinary Python inside Spark. It is slower than built-in
functions, so we use it only here, where the logic (several keyword lists)
is easier to write in Python. One comment can have several topics.
"""
import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from src.lexicon import TOPIC_KEYWORDS

# Compile once; each pattern is wrapped in word boundaries.
_PATTERNS = {
    topic: re.compile(r"\b(?:" + "|".join(words) + r")\b")
    for topic, words in TOPIC_KEYWORDS.items()
}


def tag_topics(text):
    """Plain Python: return the list of topics found in one cleaned comment."""
    if not text:
        return []
    return [topic for topic, pattern in _PATTERNS.items() if pattern.search(text)]


tag_topics_udf = F.udf(tag_topics, ArrayType(StringType()))


def add_topics(df: DataFrame) -> DataFrame:
    """Add `topics` (array of topic names; ["untagged"] if nothing matched) and `n_topics`."""
    tagged = df.withColumn("topics", tag_topics_udf(F.col("text_clean")))
    tagged = tagged.withColumn(
        "topics", F.when(F.size("topics") == 0, F.array(F.lit("untagged"))).otherwise(F.col("topics"))
    )
    return tagged.withColumn("n_topics", F.size("topics"))


def explode_topics(df: DataFrame) -> DataFrame:
    """One row per (comment, topic): needed to count comments per topic."""
    return df.withColumn("topic", F.explode("topics"))