#!/bin/bash

# This script downloads and extracts a dataset from GCS.
# The dataset URL and destination directory are configurable.

# Usage:
# scripts/download_extract_dataset.sh [DATASET_URL] [DESTINATION_DIR]
#
# Example:
# scripts/download_extract_dataset.sh gs://dsgt-clef-plantclef-2024/raw/PlantCLEF2022_web_training_images_1.tar.gz /mnt/data
#
# This will download the dataset from the specified URL and extract it to the specified destination directory.
# If no arguments are provided, default values are used for both the dataset URL and destination directory.

set -e # Exit immediately if a command exits with a non-zero status.

# Default values
DEFAULT_DATASET_URL="gs://dsgt-clef-plantclef-2024/raw/PlantCLEF2022_web_training_images_1.tar.gz"
DEFAULT_DESTINATION_DIR="/mnt/data"

# Check if custom arguments are provided
if [ "$#" -ge 2 ]; then
    DATASET_URL="$1"
    DESTINATION_DIR="$2"
else
    DATASET_URL="$DEFAULT_DATASET_URL"
    DESTINATION_DIR="$DEFAULT_DESTINATION_DIR"
fi

DATASET_NAME=$(basename "$DATASET_URL")
# Remove the file extension to get the folder name
FOLDER_NAME="${DATASET_NAME%.*.*}"
DESTINATION_PATH="$DESTINATION_DIR/$DATASET_NAME"
IMAGES_DIR="$DESTINATION_DIR/images"

echo "Using dataset URL: $DATASET_URL"
echo "Downloading dataset to: $DESTINATION_DIR"

# Prepare the destination directory
sudo mount "$DESTINATION_DIR" || true # Proceed even if mount fails, assuming it's already mounted
sudo chmod -R 777 "$DESTINATION_DIR"
echo "Permissions set for $DESTINATION_DIR."

# Download the dataset
echo "Downloading dataset..."
gcloud storage cp "$DATASET_URL" "$DESTINATION_DIR" || {
    echo "Failed to download the dataset."
    exit 1
}

# Extract the dataset
echo "Extracting dataset..."
tar -xzf "$DESTINATION_PATH" -C "$DESTINATION_DIR"
echo "Dataset extracted to $DESTINATION_DIR."

# Rename the "images" directory if it exists
if [ -d "$IMAGES_DIR" ]; then
    echo "Renaming 'images' folder to '$FOLDER_NAME'"
    mv "$IMAGES_DIR" "$DESTINATION_DIR/$FOLDER_NAME"
else
    echo "'images' directory not found, skipping rename."
fi

# Final listing and disk usage report
echo "Final contents of $DESTINATION_DIR:"
ls "$DESTINATION_DIR"
echo "Disk usage and free space:"
df -h

echo "Script completed successfully."
