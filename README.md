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
