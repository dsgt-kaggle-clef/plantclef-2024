![banner](/figures/plantclef-banner.png)

# plantclef-2024

- https://www.imageclef.org/node/315

## Quickstart

Install the pre-commit hooks for formatting code:

```bash
pre-commit install
```

We are generally using a shared VM with limited space.
Install packages to the system using sudo:

```bash
sudo pip install -r requirements.txt
```

We can ignore the following message since we know what we are doing:

```
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

This should allow all users to use the packages without having to install them multiple times.
This is a problem with very large packages like `torch` and `spark`.

Then install the current package into your local user directory, so changes that you make are local to your own users.

```bash
pip install -e .
```

To run the `install_gcs_connector.sh` script, execute the following.

Before running the script, you need to make sure it's executable. You can do this with the chmod command:

```
chmod +x scripts/install_gcs_connector.sh
```

Now that the script is executable, you can run it:

```
sudo scripts/install_gcs_connector.sh
```

## Download and Extract Dataset Script

The `download_extract_dataset.sh` script automates the process of downloading and extracting a dataset from GCS. It allows for the customization of the dataset URL and the destination directory through command-line arguments.

### Usage

To use the script, navigate to your project directory and run the following command in your terminal:

```bash
scripts/download_extract_dataset.sh [DATASET_URL] [DESTINATION_DIR]
```

Replace **[DATASET_URL]** with the URL of the dataset you wish to download, and **[DESTINATION_DIR]** with the path where you want the dataset to be downloaded and extracted. If these arguments are not provided, the script will use its default settings.

### Example

Make sure the script is executable before running it:

```
chmod +x scripts/download_extract_dataset.sh
```

Run the script:

```
scripts/download_extract_dataset.sh gs://dsgt-clef-plantclef-2024/raw/PlantCLEF2022_web_training_images_1.tar.gz /mnt/data
```

This will download the `PlantCLEF2022_web_training_images_1.tar.gz` dataset from the specified Google Cloud Storage URL and extract it to `/mnt/data`.

## Create a dataframe from images and write it to GCS

The `images_to_parquet.py` script is designed to process image data and associated metadata for a specific dataset. It reads images and metadata from specified directories, performs necessary processing, and outputs the data in Parquet format to a GCS path. This script supports customizing paths for input data and output files through command-line arguments.

### Usage

Navigate to the project directory and run the script using the following command format:

```
python plantclef/images_to_parquet.py [OPTIONS]
```

**Options:**

- `--cores`: Number of cores used in Spark. Default is `4`.

- `--memory`: Amount of memory capacity to use in Spark driver. Default is `8g`.

- `--raw-root-path`: Root directory path for metadata. Default is `gs://dsgt-clef-plantclef-2024/raw/`.

- `--output-path`: GCS path for output Parquet files. Default is `gs://dsgt-clef-plantclef-2024/data/parquet_files/PlantCLEF2022_web_training_images_1`.

- `--dataset-name`: Dataset name downloaded from the tar file. Default is `PlantCLEF2022_web_training_images_1`.

- `--meta-dataset-name`: Train Metadata CSV file name. Default is `PlantCLEF2022_web_training_metadata`.

### Example Commands

Run the script with default settings:

```
python plantclef/images_to_parquet.py
```

Run the script with custom paths:

```
python plantclef/images_to_parquet.py --raw-root-path gs://my-custom-path/raw/ --output-path gs://my-custom-path/data/parquet_files/image_data --dataset-name MyDataset --meta-dataset-name MyMetadata
```

Here's an example for creating a parquet for the `PlantCLEF2022_web_training_images_1` dataset:

```
python plantclef/images_to_parquet.py --cores 4 --memory 8g --output-path gs://dsgt-clef-plantclef-2024/data/parquet_files/PlantCLEF2022_web_training_images_1 --dataset-name PlantCLEF2022_web_training_images_1 --meta-dataset-name PlantCLEF2022_web_training_metadata
```

Another example running the pipeline on the PlantCLEF 2024 datase:

```
python plantclef/images_to_parquet.py --cores 8 --memory 28g --output-path gs://dsgt-clef-plantclef-2024/data/parquet_files/PlantCLEF2024_training --dataset-name PlantCLEF2024 --meta-dataset-name PlantCLEF2024singleplanttrainingdata
```

For detailed help on command-line options, run `python plantclef/images_to_parquet.py --help`.
