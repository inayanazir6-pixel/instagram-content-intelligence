"""One place to create the SparkSession so every script configures it the same way."""
from pyspark.sql import SparkSession


def get_spark(app_name: str = "InstagramContentIntelligence") -> SparkSession:
    """Return a local SparkSession that uses all CPU cores."""
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")  # small data -> few partitions
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")  # hide the noisy INFO/WARN logs
    return spark