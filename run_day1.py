"""Day 1 pipeline: ingest raw CSVs -> validate -> write the bronze layer.

Run from the project root:
    python run_day1.py
"""
import json
import logging

from src import config, ingest, validate
from src.spark_session import get_spark

log = logging.getLogger("day1")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    spark = get_spark()
    datasets = ingest.load_all(spark)

    all_checks, profiles = [], {}
    for name, df in datasets.items():
        spec = config.DATASET_SPECS[name]
        df = df.cache()  # keep in memory: we scan it several times

        log.info("=== %s ===", name.upper())
        df.printSchema()
        df.select(spec["text_col"], spec["label_col"]).show(3, truncate=70)

        checks, profile = validate.validate_dataset(df, name, **spec)
        all_checks.extend(checks)
        profiles[name] = profile

        out = ingest.write_bronze(df, name)
        log.info("Bronze Parquet written to %s", out)
        df.unpersist()

    # ---- Validation summary table -------------------------------------------
    rows = [
        (c["dataset"], c["check"], c["failed_rows"], c["pct_of_rows"], c["status"])
        for c in all_checks
    ]
    print("\nVALIDATION SUMMARY")
    spark.createDataFrame(
        rows, ["dataset", "check", "failed_rows", "pct_of_rows", "status"]
    ).show(len(rows), truncate=False)

    # ---- Dataset profiles (sizes, label balance, text length) ---------------
    print("DATASET PROFILES")
    print(json.dumps(profiles, indent=2, ensure_ascii=False))

    report_path = validate.save_report(all_checks, profiles)
    fails = sum(1 for c in all_checks if c["status"] == "FAIL")
    warns = sum(1 for c in all_checks if c["status"] == "WARN")
    log.info("Report saved to %s", report_path)
    log.info("Result: %d FAIL, %d WARN (Day 2 cleans these up)", fails, warns)

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())