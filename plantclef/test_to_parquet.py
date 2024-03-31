import argparse
from pathlib import Path

from pyspark.sql import functions as F

from plantclef.utils import get_spark

"""
Before running this script, make sure you have downloaded and extracted the test dataset into the data folder.
Use the bash file `download_extract_dataset.sh` in the scripts folder.
"""


def create_test_dataframe(spark, base_dir: Path):
    # Load all files from the base directory as binary data
    # Convert Path object to string when passing to PySpark
    image_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(base_dir.as_posix())
    )

    # Construct the string to be replaced - adjust this based on your actual base path
    to_remove = "file:" + str(base_dir.parents[0])

    # Remove "file:{base_dir.parents[0]" from path column
    image_df = image_df.withColumn("path", F.regexp_replace("path", to_remove, ""))

    # Split the path into an array of elements
    split_path = F.split(image_df["path"], "/")

    # Select and rename columns to fit the target schema, including renaming 'content' to 'data'
    image_final_df = image_df.select(
        "path",
        F.element_at(split_path, -1).alias("image_name"),
        F.col("content").alias("data"),
    ).repartition(500)

    return image_final_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images and metadata for a dataset stored on GCS."
    )
    parser.add_argument(
        "--cores", type=int, default=4, help="Number of cores used in Spark driver"
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="12g",
        help="Amount of memory to use in Spark driver",
    )
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=str(Path(".").resolve()),
        help="Base directory path for image data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="gs://dsgt-clef-plantclef-2024/data/parquet_files/PlantCLEF2024_test",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PlantCLEF2024test",
        help="Dataset name downloaded from tar file",
    )

    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    args = parse_args()

    # Initialize Spark
    spark = get_spark(cores=args.cores, memory=args.memory)

    # Convert raw-root-path to a Path object here
    base_dir = Path(args.image_root_path) / "data" / args.dataset_name

    # Create test image dataframe
    final_df = create_test_dataframe(spark=spark, base_dir=base_dir)

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(args.output_path)


if __name__ == "__main__":
    main()
