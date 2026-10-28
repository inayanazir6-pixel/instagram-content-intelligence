"""Day 4: tag topics, find patterns in negative reactions, build recommendations.

Run from the project root:
    python run_day4.py
"""
import logging

from src import analysis, config, eda, insights
from src.spark_session import get_spark
from src.topics import add_topics

log = logging.getLogger("day4")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    spark = get_spark()

    comments = spark.read.parquet(str(config.SILVER_DIR / "comments"))
    comments = add_topics(comments).cache()
    log.info("Comments with topics: %d", comments.count())

    coverage = analysis.topic_coverage(comments)
    neg_rates = analysis.negative_rate_by_topic(comments)
    sex_types = analysis.sexism_type_by_topic(comments)
    stance_split = analysis.stance_by_topic(comments)

    print("\nTOPIC COVERAGE (how many comments mention each topic)")
    coverage.show(30, truncate=False)
    print("OBSERVED NEGATIVE RATE BY TOPIC (lift > 1 = worse than the source average)")
    neg_rates.show(30, truncate=False)
    print("OBSERVED: TYPE OF SEXISM BY TOPIC (EDOS, sexist comments only)")
    sex_types.show(40, truncate=False)
    print("OBSERVED: STANCE BY TOPIC")
    stance_split.show(30, truncate=False)

    log.info("Scoring stance comments with the Day 3 models...")
    scored = analysis.score_stance_comments(comments).cache()
    why = analysis.why_critical_by_topic(scored)
    pred_types = analysis.pred_type_by_topic(scored)
    print("PREDICTED: toxic / sexist share by topic and stance (model-based)")
    why.show(40, truncate=False)
    print("PREDICTED: argument type among critical comments")
    pred_types.show(40, truncate=False)

    recs = insights.build_recommendations(spark, neg_rates, sex_types, stance_split)
    print("RECOMMENDATIONS (hypotheses to test, not proven effects)")
    recs.select("topic", "sexism_lift", "dominant_sexism_type", "low_evidence", "recommendation").show(truncate=60)

    for name, frame in {
        "topic_coverage": coverage, "negative_rate_by_topic": neg_rates,
        "sexism_type_by_topic": sex_types, "stance_by_topic": stance_split,
        "why_critical_by_topic": why, "pred_type_by_topic": pred_types,
        "recommendations": recs,
    }.items():
        log.info("Saved %s", eda.save_gold(frame, name))

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())