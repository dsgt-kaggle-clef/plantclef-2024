import argparse
import io

from PIL import Image
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import ArrayType, BinaryType

from plantclef.utils import get_spark


# Define the function to split image into grid
def _split_into_grid(image_binary, grid_size):
    image = Image.open(io.BytesIO(image_binary))
    w, h = image.size
    grid_w, grid_h = w // grid_size, h // grid_size
    patches = []
    for j in range(grid_size):
        for i in range(grid_size):
            left = i * grid_w
            upper = j * grid_h
            right = left + grid_w
            lower = upper + grid_h
            crop_image = image.crop((left, upper, right, lower))
            byte_arr = io.BytesIO()
            crop_image.save(byte_arr, format="PNG")
            patches.append(byte_arr.getvalue())
    return patches


def process_grid_df(df: DataFrame, grid_size: int = 3) -> DataFrame:
    # Register the UDF
    split_into_grid_udf = F.udf(
        lambda img: _split_into_grid(img, grid_size), ArrayType(BinaryType())
    )

    # Apply the UDF to the dataframe
    patches_df = df.withColumn("patches", split_into_grid_udf(F.col("data")))

    # Explode the dataframe to get each patch in a separate row
    exploded_df = patches_df.select(
        F.col("image_name"), F.posexplode("patches").alias("patch_number", "data")
    )

    # Create the final dataframe
    final_df = exploded_df.select("image_name", "patch_number", "data")
    return final_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images and metadata for a dataset stored on GCS."
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="8g",
        help="Amount of memory to use in Spark driver",
    )
    parser.add_argument(
        "--executor-memory",
        type=str,
        default="1g",
        help="Amount of memory to use in Spark executors",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=500,
        help="Number of partitions",
    )
    parser.add_argument(
        "--gcs-root-path",
        type=str,
        default="gs://dsgt-clef-plantclef-2024",
        help="Root directory for plantclef-2024 in GCS",
    )
    parser.add_argument(
        "--input-name-path",
        type=str,
        default="data/parquet_files/PlantCLEF2024_test",
        help="Root directory for test data in GCS",
    )
    parser.add_argument(
        "--output-name-path",
        type=str,
        default="/data/process/test_v2/grid_test_data",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=3,
        help="Size of the grid. Default is 3x3",
    )
    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    args = parse_args()
    # Input and output paths for training workflow
    input_path = f"{args.gcs_root_path}/{args.input_name_path}"
    output_path = f"{args.gcs_root_path}/{args.output_name_path}"

    # Initialize Spark with settings for using the big-disk-dev VM
    print(f"\ninput path: {input_path}")
    print(f"output path: {output_path}")
    print(f"memory: {args.memory}")
    print(f"num_partitions: {args.num_partitions}\n")
    spark = get_spark(
        memory=args.memory,
        executor_memory=args.executor_memory,
        **{"spark.sql.shuffle.partitions": args.num_partitions},
    )
    df = spark.read.parquet(input_path)

    # read data
    df = spark.read.parquet(input_path)
    # df = df.limit(100).cache()

    # get final dataframe
    final_df = process_grid_df(df, grid_size=args.grid_size)

    # Write the DataFrame to GCS in Parquet format
    final_df.repartition(args.num_partitions).write.mode("overwrite").parquet(
        output_path
    )


if __name__ == "__main__":
    main()
