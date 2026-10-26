"""Explicit schemas for every raw file.

Defining the schema by hand (instead of letting Spark guess with inferSchema)
makes the pipeline faster, predictable and safe: if a file changes shape, we
find out immediately.

Every schema ends with `_corrupt_record`: Spark puts the raw text of any row it
cannot parse there, so bad rows are captured instead of silently dropped.
"""
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

_CORRUPT = StructField("_corrupt_record", StringType(), True)

# davidson.csv : ,count,hate_speech,offensive_language,neither,class,tweet
DAVIDSON_SCHEMA = StructType([
    StructField("row_id", IntegerType(), True),
    StructField("vote_count", IntegerType(), True),      # annotators who voted
    StructField("hate_votes", IntegerType(), True),
    StructField("offensive_votes", IntegerType(), True),
    StructField("neither_votes", IntegerType(), True),
    StructField("label", IntegerType(), True),           # 0 hate, 1 offensive, 2 neither
    StructField("text", StringType(), True),
    _CORRUPT,
])

# edos.csv : rewire_id,text,label_sexist,label_category,label_vector,split
EDOS_SCHEMA = StructType([
    StructField("rewire_id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("label_sexist", StringType(), True),     # "sexist" / "not sexist"
    StructField("label_category", StringType(), True),   # type of sexism, or "none"
    StructField("label_vector", StringType(), True),     # fine-grained reason, or "none"
    StructField("split", StringType(), True),            # train / dev / test
    _CORRUPT,
])

# stance_train.csv, stance_val.csv, stance_test.csv : text,label
STANCE_SCHEMA = StructType([
    StructField("text", StringType(), True),
    StructField("label", IntegerType(), True),           # 0 none, 1 against, 2 favor
    _CORRUPT,
])