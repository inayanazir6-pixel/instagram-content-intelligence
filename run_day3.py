"""Day 3: train, tune and evaluate one Spark ML text classifier per task.

Run from the project root (takes several minutes):
    python run_day3.py
"""
import logging

from src import config, eda, export, modeling
from src.spark_session import get_spark

log = logging.getLogger("day3")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    spark = get_spark()

    summary_rows, class_rows, confusion_rows, feature_rows = [], [], [], []

    for task in config.TASKS:
        log.info("=== %s ===", task.upper())
        train, test = modeling.load_task_data(spark, task)
        train = modeling.add_class_weights(train).cache()
        test = test.cache()
        n_train = train.count()
        log.info("Training rows: %d | test rows: %d", n_train, test.count())

        log.info("Tuning with 3-fold cross-validation (a few minutes, no output until done)...")
        cv_model = modeling.train(train)
        model = cv_model.bestModel
        reg = float(model.stages[modeling.IDX_LR].getRegParam())
        log.info("Best regParam: %s | CV weighted-F1 per setting: %s",
                 reg, [round(m, 3) for m in cv_model.avgMetrics])

        summary, per_class, confusion = modeling.evaluate(model, test)
        log.info("TEST accuracy %.3f (always guessing the biggest class: %.3f) | macro-F1 %.3f",
                 summary["accuracy"], summary["baseline_accuracy"], summary["macro_f1"])

        summary_rows.append((
            task, n_train, summary["n_test"], summary["accuracy"],
            summary["baseline_accuracy"], summary["macro_f1"], summary["weighted_f1"], reg,
        ))
        class_rows += [
            (task, c["label"], c["precision"], c["recall"], c["f1"], c["support"])
            for c in per_class
        ]
        confusion_rows += [(task, c["actual"], c["predicted"], c["n"]) for c in confusion]
        feature_rows += [
            (task, f["label"], f["rank"], f["term"], f["weight"])
            for f in modeling.top_features(model)
        ]

        model.write().overwrite().save(str(config.MODELS_DIR / task))
        log.info("Spark model saved; app model: %s", export.export_lightweight(model, task))
        train.unpersist()
        test.unpersist()

    # ---- Results tables ------------------------------------------------------
    summary_df = spark.createDataFrame(
        summary_rows,
        ["task", "n_train", "n_test", "accuracy", "baseline_accuracy",
         "macro_f1", "weighted_f1", "best_reg_param"],
    )
    class_df = spark.createDataFrame(
        class_rows, ["task", "label", "precision", "recall", "f1", "support"]
    )
    confusion_df = spark.createDataFrame(confusion_rows, ["task", "actual", "predicted", "n"])
    features_df = spark.createDataFrame(
        feature_rows, ["task", "label", "rank", "term", "weight"]
    )

    print("\nMODEL SUMMARY (measured on the held-out test sets)")
    summary_df.show(truncate=False)
    print("PER-CLASS RESULTS")
    class_df.show(50, truncate=False)
    print("WORDS THAT PUSH TOWARDS EACH CLASS (top 8)")
    features_df.filter("rank <= 8").orderBy("task", "label", "rank").show(120, truncate=False)

    for name, frame in {
        "model_summary": summary_df,
        "class_metrics": class_df,
        "confusion_matrix": confusion_df,
        "top_features": features_df,
    }.items():
        log.info("Saved %s", eda.save_gold(frame, name))

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())