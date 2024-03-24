import argparse

import cv2
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType

from plantclef.utils import get_spark


# Define a UDF to crop and resize images
def crop_resize_images(data, target_width=256, target_height=256):
    """
    Crop the center of the image and resize it to the specific size.

    Args:
        data (bytes): The binary data of the image.
        size (tuple): the target size of the image after cropping and resizing.
    Returns:
        bytes: The binary data of the cropped and resized image.
    """
    # Convert binary data to NumPy array, then to image
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Determine the size for cropping to a square
    height, width = image.shape[:2]
    crop_size = min(height, width)

    # Calculate crop coordinates to get the center square
    start_x = width // 2 - crop_size // 2
    start_y = height // 2 - crop_size // 2

    # Crop the center square
    image_cropped = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    # Resize the image
    target_size = target_width, target_height
    image_resized = cv2.resize(image_cropped, target_size, interpolation=cv2.INTER_AREA)

    # Convert the image back to binary data
    _, img_encoded = cv2.imencode(".jpg", image_resized)
    img_byte_arr = img_encoded.tobytes()

    return img_byte_arr


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
        default="gs://dsgt-clef-plantclef-2024/data/parquet_files/",
        help="Root directory path for dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PlantCLEF2024_training",
        help="Dataset name downloaded from tar file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="gs://dsgt-clef-plantclef-2024/data/parquet_files/PlantCLEF2024_training_cropped_resized",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=128,
        help="Size of the resized image. Default is 128x128",
    )

    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    args = parse_args()

    # Initialize Spark with settings for using the big-disk-dev VM
    spark = get_spark(
        cores=args.cores, memory=args.memory, **{"spark.sql.shuffle.partitions": 500}
    )

    # Define the GCS path to the Train parquet file
    train_gcs_path = f"{args.raw_root_path}{args.dataset_name}"

    # Read the Parquet file into a DataFrame
    train_df = spark.read.parquet(train_gcs_path)

    # Register the UDF with BinaryType return type
    crop_resize_udf = F.udf(crop_resize_images, BinaryType())

    # Select small batch size from DataFrame to test UDF
    image_df = train_df.limit(100)

    # Apply the UDF to crop and resize the images
    crop_df = image_df.withColumn(
        "cropped_image_data",
        crop_resize_udf(
            F.col("data"), F.lit(args.square_size), F.lit(args.square_size)
        ),
    )

    # Write the DataFrame to GCS in Parquet format
    crop_df.write.mode("overwrite").parquet(args.output_path)


if __name__ == "__main__":
    main()
