import os
import sys
import time
from contextlib import contextmanager

from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(
    cores=os.cpu_count(),
    memory=os.environ.get("PYSPARK_DRIVER_MEMORY", "8g"),
    executor_memory=os.environ.get("PYSPARK_EXECUTOR_MEMORY", "1g"),
    local_dir="/mnt/data/tmp",
    app_name="plantclef",
    **kwargs,
):
    """Get a spark session for a single driver."""
    builder = (
        SparkSession.builder.config("spark.driver.memory", memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.cores", cores)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.local.dir", f"{local_dir}/{int(time.time())}")
    )
    for k, v in kwargs.items():
        builder = builder.config(k, v)
    return builder.appName(app_name).getOrCreate()


@contextmanager
def spark_resource(*args, **kwargs):
    """A context manager for a spark session."""
    spark = None
    try:
        spark = get_spark(*args, **kwargs)
        yield spark
    finally:
        if spark is not None:
            spark.stop()
