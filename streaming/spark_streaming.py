"""
Spark Structured Streaming — real-time feature computation.
Consumes GPS events from Kafka, computes 5-min and 1-hour aggregations,
and writes latest features to the online feature store (Redis).

Run: spark-submit streaming/spark_streaming.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, TimestampType,
)
from loguru import logger

from config import settings


# ── Schema ────────────────────────────────────────────────────────────────────
GPS_SCHEMA = StructType([
    StructField("timestamp",       TimestampType(), True),
    StructField("zone",            StringType(),    True),
    StructField("demand",          IntegerType(),   True),
    StructField("weather",         StringType(),    True),
    StructField("temperature_c",   DoubleType(),    True),
    StructField("precipitation_mm",DoubleType(),    True),
    StructField("wind_speed_kmh",  DoubleType(),    True),
    StructField("is_event",        BooleanType(),   True),
    StructField("lat",             DoubleType(),    True),
    StructField("lon",             DoubleType(),    True),
])


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("RideFlow-Streaming")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/rideflow_checkpoint")
        .getOrCreate()
    )


def read_kafka_stream(spark: SparkSession):
    """Read raw bytes from Kafka GPS topic."""
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", settings.KAFKA_TOPIC_GPS)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_messages(raw_df):
    """Parse JSON payload from Kafka value bytes."""
    return (
        raw_df
        .select(F.from_json(
            F.col("value").cast("string"), GPS_SCHEMA
        ).alias("data"))
        .select("data.*")
        .filter(F.col("zone").isNotNull())
        .withColumn("event_time", F.col("timestamp").cast(TimestampType()))
        .withWatermark("event_time", "10 minutes")  # allow 10min late arrivals
    )


def compute_5min_aggregations(parsed_df):
    """
    5-minute tumbling window: demand count, weather mode, avg temperature.
    Written to Redis for real-time API serving.
    """
    return (
        parsed_df
        .groupBy(
            F.window("event_time", "5 minutes"),
            F.col("zone"),
        )
        .agg(
            F.sum("demand").alias("demand_5min"),
            F.avg("temperature_c").alias("avg_temp_5min"),
            F.avg("precipitation_mm").alias("avg_precip_5min"),
            F.first("weather").alias("weather"),
            F.max("is_event").cast("int").alias("is_event"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "zone", "demand_5min", "avg_temp_5min",
            "avg_precip_5min", "weather", "is_event",
        )
    )


def compute_1hour_aggregations(parsed_df):
    """
    1-hour sliding window (step 5min): rolling demand stats.
    Used as lag features for model inference.
    """
    return (
        parsed_df
        .groupBy(
            F.window("event_time", "1 hour", "5 minutes"),  # sliding
            F.col("zone"),
        )
        .agg(
            F.sum("demand").alias("demand_1h"),
            F.avg("demand").alias("demand_1h_mean"),
            F.stddev("demand").alias("demand_1h_std"),
            F.max("demand").alias("demand_1h_max"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            "zone", "demand_1h", "demand_1h_mean",
            "demand_1h_std", "demand_1h_max",
        )
    )


def write_to_redis(batch_df, batch_id: int) -> None:
    """
    Micro-batch writer: push latest features to Redis online store.
    Called by Spark's foreachBatch.
    """
    import redis as redis_lib
    import json

    try:
        r = redis_lib.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_timeout=2,
        )
        rows = batch_df.collect()
        pipe = r.pipeline()

        for row in rows:
            zone    = str(row["zone"])
            features = {
                "demand_5min":   float(row.get("demand_5min", 0) or 0),
                "avg_temp_5min": float(row.get("avg_temp_5min", 12) or 12),
                "avg_precip_5min": float(row.get("avg_precip_5min", 0) or 0),
                "weather":       str(row.get("weather", "clear") or "clear"),
                "is_event":      int(row.get("is_event", 0) or 0),
            }
            key     = f"stream:features:{zone}"
            payload = json.dumps(features)
            pipe.setex(key, settings.REDIS_TTL_SECONDS, payload)

        pipe.execute()
        logger.debug(f"Batch {batch_id}: wrote {len(rows)} zone features to Redis")

    except Exception as e:
        logger.warning(f"Redis write failed (batch {batch_id}): {e}")


def write_to_parquet(batch_df, batch_id: int) -> None:
    """
    Write aggregated features to offline Parquet store for training.
    Partitioned by date for efficient reads.
    """
    if batch_df.isEmpty():
        return
    out = str(settings.FEATURE_STORE_PATH / "streaming")
    (
        batch_df
        .withColumn("date", F.to_date("window_start"))
        .write
        .partitionBy("date")
        .mode("append")
        .parquet(out)
    )
    logger.debug(f"Batch {batch_id}: written to Parquet offline store")


def run_streaming_pipeline():
    spark  = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Starting RideFlow Spark Streaming pipeline...")
    raw_df    = read_kafka_stream(spark)
    parsed_df = parse_messages(raw_df)
    agg_5min  = compute_5min_aggregations(parsed_df)

    # Stream 1: write to Redis (online store) every 30s
    q_redis = (
        agg_5min.writeStream
        .foreachBatch(write_to_redis)
        .outputMode("update")
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", "/tmp/rideflow_redis_ckpt")
        .start()
    )

    # Stream 2: write to Parquet (offline store) every 5min
    q_parquet = (
        agg_5min.writeStream
        .foreachBatch(write_to_parquet)
        .outputMode("update")
        .trigger(processingTime="5 minutes")
        .option("checkpointLocation", "/tmp/rideflow_parquet_ckpt")
        .start()
    )

    logger.success("Both streaming queries started. Awaiting termination...")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run_streaming_pipeline()
