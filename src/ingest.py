"""Ingestion: read the raw CSV files with explicit schemas and save them as Parquet."""
import logging
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from src import config, schemas

log = logging.getLogger(__name__)


def _check_files_exist(paths) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {config.RAW_DIR}: {missing}. "
            "Upload them to data/raw/ using exactly these names."
        )


def _read_csv(spark: SparkSession, path, schema: StructType, source: str) -> DataFrame:
    """Read one CSV with the given schema and add lineage columns."""
    # Cheap safety check: does the file have the number of columns we expect?
    expected = len(schema.fields) - 1  # the last field is _corrupt_record
    found = len(spark.read.option("header", True).csv(str(path)).columns)
    if found != expected:
        raise ValueError(
            f"{path.name}: expected {expected} columns but found {found}. "
            "Is it the right file, and is it comma-separated?"
        )

    return (
        spark.read.option("header", True)
        .option("multiLine", True)   # tweets can contain line breaks inside quotes
        .option("quote", '"')
        .option("escape", '"')       # "" inside a quoted text means a literal quote
        .option("mode", "PERMISSIVE")  # keep bad rows instead of crashing
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(schema)
        .csv(str(path))
        .withColumn("source", F.lit(source))
        .withColumn("source_file", F.col("_metadata.file_name"))
        .withColumn("ingested_at", F.current_timestamp())
    )


def load_davidson(spark: SparkSession) -> DataFrame:
    _check_files_exist([config.DAVIDSON_FILE])
    return _read_csv(spark, config.DAVIDSON_FILE, schemas.DAVIDSON_SCHEMA, "davidson")


def load_edos(spark: SparkSession) -> DataFrame:
    _check_files_exist([config.EDOS_FILE])
    return _read_csv(spark, config.EDOS_FILE, schemas.EDOS_SCHEMA, "edos")


def load_stance(spark: SparkSession) -> DataFrame:
    """Read the three stance files and stack them into one DataFrame."""
    _check_files_exist(list(config.STANCE_FILES.values()))
    parts = [
        _read_csv(spark, path, schemas.STANCE_SCHEMA, "stance").withColumn("split", F.lit(split))
        for split, path in config.STANCE_FILES.items()
    ]
    return reduce(DataFrame.unionByName, parts)


def load_all(spark: SparkSession) -> dict:
    """Return {dataset name: DataFrame} for all three datasets."""
    return {
        "davidson": load_davidson(spark),
        "edos": load_edos(spark),
        "stance": load_stance(spark),
    }


def write_bronze(df: DataFrame, name: str):
    """Save a DataFrame to data/bronze/<name>/ as Parquet and return the path."""
    out = config.BRONZE_DIR / name
    df.write.mode("overwrite").parquet(str(out))
    return out