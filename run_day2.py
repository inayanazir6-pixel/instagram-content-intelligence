"""Day 2 pipeline: bronze -> clean silver + quarantine, then EDA and RDD word analysis.

Run from the project root:
    python run_day2.py
"""
import logging

from pyspark.sql import functions as F

from src import eda, rdd_analysis, transform, validate
from src.spark_session import get_spark

log = logging.getLogger("day2")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    spark = get_spark()
    bronze = transform.read_bronze(spark)

    # The one EDOS row Day 1 flagged for broken characters
    log.info("EDOS rows flagged by the broken-character check:")
    bronze["edos"].filter(F.col("text").rlike(validate.MOJIBAKE_REGEX)).select("text").show(
        truncate=False
    )

    # ---- Bronze -> Silver ----------------------------------------------------
    comments = transform.build_comments(bronze)
    clean, quarantine = transform.split_quarantine(comments)
    clean, quarantine = clean.cache(), quarantine.cache()
    log.info("Silver rows kept: %d | quarantined: %d", clean.count(), quarantine.count())

    print("\nBEFORE / AFTER CLEANING (random sample)")
    clean.orderBy(F.rand(3)).select("source", "text_raw", "text_clean").show(8, truncate=55)

    transform.write_silver(clean, quarantine)
    log.info("Silver and quarantine written")

    # ---- Exploratory analysis ------------------------------------------------
    dist = eda.label_distribution(clean)
    lengths = eda.length_stats(clean)
    sexism = eda.sexism_breakdown(spark, clean)
    qsum = eda.quarantine_summary(quarantine)

    print("\nLABEL DISTRIBUTION")
    dist.show(truncate=False)
    print("COMMENT LENGTH BY LABEL")
    lengths.show(truncate=False)
    print("TYPES OF SEXISM IN EDOS (Spark SQL)")
    sexism.show(truncate=False)
    print("WHY ROWS WERE QUARANTINED")
    qsum.show(truncate=False)
    print("STANCE LABEL CHECK: 3 random tweets per label")
    eda.sample_per_label(clean, "stance").show(truncate=100)

    # ---- RDD word frequencies ------------------------------------------------
    top_words, overall = rdd_analysis.top_words_by_class(spark, clean)
    print("TOP 10 WORDS OVERALL (RDD takeOrdered):", overall)
    print("TOP 5 WORDS PER SOURCE AND LABEL (RDD reduceByKey + window rank)")
    top_words.filter("rank <= 5").show(60, truncate=False)

    # ---- Save small result tables for the app --------------------------------
    results = {
        "label_distribution": dist,
        "length_stats": lengths,
        "sexism_breakdown": sexism,
        "quarantine_summary": qsum,
        "top_words": top_words,
    }
    for name, frame in results.items():
        log.info("Saved %s", eda.save_gold(frame, name))

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())