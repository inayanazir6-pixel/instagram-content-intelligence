"""Text cleaning with built-in Spark functions (faster than Python UDFs).

`text_raw` is never modified. We create a new column `text_clean` for analysis
and modelling, plus a few simple numeric features.
"""
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

_HTML_ENTITIES = [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')]


def clean_text(col: Column) -> Column:
    """Return a cleaned, lower-cased version of a text column."""
    t = col
    for entity, char in _HTML_ENTITIES:                          # "&amp;" -> "&"
        t = F.regexp_replace(t, entity, char)
    t = F.regexp_replace(t, r"&#?\w+;", " ")                     # leftover entities like &#128514;
    t = F.regexp_replace(t, "[\u2018\u2019]", "'")               # curly quotes -> '
    t = F.regexp_replace(t, r"\bRT\b", " ")                      # retweet marker
    t = F.regexp_replace(t, r"https?://\S+|www\.\S+|\[URL\]", " ")  # links
    t = F.regexp_replace(t, r"@\w+|\[USER\]", " ")               # @mentions
    t = F.regexp_replace(t, r"(?i)#semst\b", " ")                # dataset tag, not real text
    t = F.regexp_replace(t, r"#(\w+)", "$1")                     # #word -> word
    t = F.lower(t)
    t = F.regexp_replace(t, r"(.)\1{2,}", "$1$1")                # sooooo -> soo
    t = F.regexp_replace(t, r"[^a-z0-9'\s]", " ")                # punctuation, emojis
    t = F.regexp_replace(t, r"\s+", " ")                         # collapse spaces
    return F.trim(t)


def add_text_features(df: DataFrame) -> DataFrame:
    """Add text_clean and simple features computed from text_raw / text_clean."""
    return (
        df.withColumn("text_clean", clean_text(F.col("text_raw")))
        .withColumn("char_count", F.length("text_clean"))
        .withColumn(
            "word_count",
            F.when(F.col("text_clean") == "", 0).otherwise(F.size(F.split("text_clean", " "))),
        )
        .withColumn("is_retweet", F.col("text_raw").rlike(r"\bRT\b"))
        .withColumn("has_url", F.col("text_raw").rlike(r"https?://|www\.|\[URL\]"))
        .withColumn("n_hashtags", F.regexp_count("text_raw", F.lit(r"#\w+")))
        .withColumn("n_mentions", F.regexp_count("text_raw", F.lit(r"@\w+|\[USER\]")))
    )