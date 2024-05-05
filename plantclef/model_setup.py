import os
import subprocess


def setup_pretrained_model():
    """
    Downloads and unzips a model from Google Cloud Storage and returns the path to the specified model file.
    Checks if the model already exists and skips download and extraction if it does.

    :return: Absolute path to the model file.
    """
    local_directory = "/mnt/data/models"
    tar_filename = "PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar"
    relative_model_path = "pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar"
    full_model_path = os.path.join(local_directory, relative_model_path)

    # Check if the model file already exists
    if os.path.exists(full_model_path):
        print("Model already exists. Skipping download and extraction.")
        return full_model_path

    # Paths for the .tar file
    gcs_path = "gs://dsgt-clef-plantclef-2024/data/models/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar"
    tar_path = os.path.join(local_directory, tar_filename)

    # Ensure the directory exists
    os.makedirs(local_directory, exist_ok=True)

    # Download the .tar file from GCS
    subprocess.run(["gsutil", "cp", gcs_path, tar_path], check=True)

    # Unzip the .tar file
    subprocess.run(["tar", "-xvf", tar_path, "-C", local_directory], check=True)

    # Return the path to the model file
    return full_model_path


if __name__ == "__main__":
    # Get model
    model_path = setup_pretrained_model()
    print("Model path:", model_path)
