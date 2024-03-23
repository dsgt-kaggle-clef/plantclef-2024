import argparse
from pathlib import Path

from pyspark.sql import functions as F

from plantclef.utils import get_spark

"""
Before running this script, make sure you have downloaded and extracted the dataset into the data folder.
Use the bash file `download_extract_dataset.sh` in the scripts folder.
"""


def create_dataframe(spark, base_dir: Path, raw_root_path: str, meta_dataset_name: str):
    """Converts images into binary data and joins with a Metadata DataFrame"""
    # Load all files from the base directory as binary data
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
    )

    # Read the iNaturalist metadata CSV file
    meta_df = spark.read.csv(
        f"{raw_root_path}/{meta_dataset_name}.csv",
        header=True,
        inferSchema=True,
        sep=";",  # specify semicolon as delimiter
    )

    # Drop duplicate entries based on 'image_path' before the join
    meta_final_df = meta_df.dropDuplicates(["image_name"])

    # Perform an inner join on the 'image_path' column
    final_df = image_final_df.join(meta_final_df, "image_name", "inner").repartition(
        500, "species_id"
    )

    return final_df


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
        default="8g",
        help="Amount of memory to use in Spark driver",
    )
    parser.add_argument(
        "--raw-root-path",
        type=str,
        default="gs://dsgt-clef-plantclef-2024/raw/",
        help="Root directory path for metadata",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="gs://dsgt-clef-plantclef-2024/data/parquet_files/PlantCLEF-small_size",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PlantCLEF2024",
        help="Dataset name downloaded from tar file",
    )
    parser.add_argument(
        "--meta-dataset-name",
        type=str,
        default="PlantCLEF2024singleplanttrainingdata",
        help="Train Metadata CSV file",
    )

    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    args = parse_args()

    # Initialize Spark with settings for using the big-disk-dev VM
    spark = get_spark(cores=8, memory="28g", **{"spark.sql.shuffle.partitions": 500})

    # Convert image-root-path to a Path object here
    base_dir = Path(args.image_root_path) / "data" / args.dataset_name

    # Create image dataframe
    final_df = create_dataframe(
        spark=spark,
        base_dir=base_dir,
        raw_root_path=args.raw_root_path,
        meta_dataset_name=args.meta_dataset_name,
    )

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(args.output_path)


if __name__ == "__main__":
    main()
