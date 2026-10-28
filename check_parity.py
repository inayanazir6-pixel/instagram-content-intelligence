"""Parity check: does the plain-Python scorer (used by the web app) match Spark?

Run from the project root (needs Spark, ~1 minute):
    python check_parity.py
"""
import logging

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from src import config, modeling
from src.scoring import clean_text, load_model
from src.spark_session import get_spark

log = logging.getLogger("parity")
SAMPLE_ROWS = 1500


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    spark = get_spark()
    sample = (
        spark.read.parquet(str(config.SILVER_DIR / "comments"))
        .orderBy(F.rand(42)).limit(SAMPLE_ROWS)
        .select("text_raw", "text_clean").cache()
    )
    rows = sample.collect()
    clean_ok = sum(clean_text(r["text_raw"]) == r["text_clean"] for r in rows)
    print(f"\nCLEANING: Python == Spark for {clean_ok}/{len(rows)} comments "
          f"({100 * clean_ok / len(rows):.2f}%)")

    print("\nMODELS (same cleaned text given to both):")
    all_ok = clean_ok == len(rows)
    for task in config.TASKS:
        model = PipelineModel.load(str(config.MODELS_DIR / task))
        labels = modeling.class_names(model)
        scored = (
            model.transform(sample.withColumn("label", F.lit(labels[0])))
            .select("text_clean", "prediction", vector_to_array("probability").alias("p"))
            .collect()
        )
        py = load_model(task)
        same, worst = 0, 0.0
        for r in scored:
            out = py.predict(r["text_clean"])
            same += out["label"] == labels[int(r["prediction"])]
            diff = max(abs(out["probabilities"][lab] - r["p"][i]) for i, lab in enumerate(labels))
            worst = max(worst, diff)
        ok = same == len(scored) and worst < 1e-6
        all_ok &= ok
        print(f"  {task:<12} same label: {same}/{len(scored)} | largest probability difference: {worst:.2e} "
              f"| {'PASS' if ok else 'CHECK'}")

    print("\nOVERALL:", "PASS - the app scores exactly like Spark" if all_ok else "differences found - send me this output")
    spark.stop()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())