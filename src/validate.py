"""Data validation: quality checks that run in one pass and produce a report.

Severity levels
  FAIL = the row is unusable (missing text/label, invalid label, unparseable row)
  WARN = usable but suspicious (duplicates, empty text, broken characters)
"""
import json
from datetime import datetime

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src import config

# Typical garbage when UTF-8 text was decoded with the wrong encoding.
MOJIBAKE_REGEX = "Ã|ðŸ|â€"


def _flag(condition: Column) -> Column:
    """Count the rows where `condition` is true (nulls count as false)."""
    return F.sum(F.when(condition, 1).otherwise(0))


def _pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 2) if total else 0.0


def value_counts(df: DataFrame, col: str, names: dict = None) -> dict:
    """{value: row count}, biggest first. `names` optionally renames the values."""
    rows = df.groupBy(col).count().orderBy(F.desc("count")).collect()
    return {
        str(names.get(r[col], r[col]) if names else r[col]): int(r["count"])
        for r in rows
    }


def validate_dataset(
    df: DataFrame,
    name: str,
    text_col: str,
    label_col: str,
    allowed_labels: list,
    label_names: dict = None,
    distributions=(),
):
    """Run all checks on one dataset. Returns (list of checks, profile dict)."""
    text, label = F.col(text_col), F.col(label_col)

    # One Spark job computes every counter at once.
    stats = df.agg(
        F.count(F.lit(1)).alias("total_rows"),
        _flag(F.col("_corrupt_record").isNotNull()).alias("corrupt_rows"),
        _flag(text.isNull()).alias("null_text"),
        _flag(F.trim(text) == "").alias("empty_text"),
        _flag(label.isNull()).alias("null_label"),
        _flag(label.isNotNull() & ~label.isin(allowed_labels)).alias("invalid_label"),
        _flag(text.rlike(MOJIBAKE_REGEX)).alias("mojibake_text"),
        F.min(F.length(text)).alias("min_len"),
        F.round(F.avg(F.length(text)), 1).alias("avg_len"),
        F.max(F.length(text)).alias("max_len"),
    ).first()

    total = int(stats["total_rows"])
    duplicate_text = total - df.dropDuplicates([text_col]).count()

    rules = [  # (check name, failed rows, severity if failed)
        ("corrupt_rows", stats["corrupt_rows"], "FAIL"),
        ("null_text", stats["null_text"], "FAIL"),
        ("null_label", stats["null_label"], "FAIL"),
        ("invalid_label", stats["invalid_label"], "FAIL"),
        ("empty_text", stats["empty_text"], "WARN"),
        ("duplicate_text", duplicate_text, "WARN"),
        ("mojibake_text", stats["mojibake_text"], "WARN"),
    ]
    checks = []
    for check, failed, severity in rules:
        failed = int(failed or 0)
        checks.append({
            "dataset": name,
            "check": check,
            "failed_rows": failed,
            "pct_of_rows": _pct(failed, total),
            "status": "PASS" if failed == 0 else severity,
        })

    profile = {
        "total_rows": total,
        "text_length": {
            "min": stats["min_len"],
            "avg": stats["avg_len"],
            "max": stats["max_len"],
        },
        "distributions": {
            col: value_counts(df, col, label_names if col == label_col else None)
            for col in distributions
        },
    }
    return checks, profile


def save_report(checks: list, profiles: dict):
    """Write reports/validation_report.json and return its path."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "validation_report.json"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checks": checks,
        "profiles": profiles,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path