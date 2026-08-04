import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, lower, regexp_replace, trim

RAW_PATH = os.getenv("RAW_PATH", "data/raw")
PROCESSED_PATH = os.getenv("PROCESSED_PATH", "data/processed")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("SentimentPreprocess")
        .getOrCreate()
    )


def preprocess(df):
    return (
        df
        .dropna(subset=["text", "label"])
        .withColumn("text", lower(col("text")))
        .withColumn("text", regexp_replace(col("text"), r"http\S+", ""))
        .withColumn("text", regexp_replace(col("text"), r"[^a-z0-9\s]", ""))
        .withColumn("text", trim(col("text")))
        .withColumn("text_length", length(col("text")))
        .filter(col("text") != "")
        .filter(col("text_length") > 3)
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(RAW_PATH)
    print(f"[preprocess] Raw rows: {df.count()}")

    processed = preprocess(df)
    print(f"[preprocess] Processed rows: {processed.count()}")

    processed.write.mode("overwrite").parquet(PROCESSED_PATH)
    print(f"[preprocess] Written to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()