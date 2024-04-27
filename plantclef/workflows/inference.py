import csv
import os
from pathlib import Path

import luigi
import luigi.contrib.gcs
import numpy as np
import pandas as pd
import torch
from google.cloud import storage
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf

from plantclef.baseline.model import LinearClassifier
from plantclef.utils import spark_resource

"""
TODO
1. path_in_bucket in the _load_model_from_gcs method has weird folder name (u1p43hb2)
2. need to get the path_in_bucket in a dynamic way
3.
"""


class InferenceTask(luigi.Task):
    input_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    feature_col = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    limit_species = luigi.OptionalIntParameter(default=None)
    species_image_count = luigi.IntParameter(default=100)
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.IntParameter(default=32)
    experiment_run_path = "experiments/dsgt_run.csv"

    def output(self):
        # save the model run
        return luigi.contrib.gcs.GCSTarget(
            f"{self.default_root_dir}/{self.experiment_run_path}_SUCCESS"
        )

    def _load_model_from_gcs(
        self,
        num_features: int,
        num_classes: int,
    ):
        bucket_name = "dsgt-clef-plantclef-2024"
        path_in_bucket = (
            f"{self.default_root_dir}/plantclef-2024/u1p43hb2/checkpoints/last.ckpt"
        )
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path_in_bucket)
        blob.download_to_filename("last.ckpt")
        checkpoint = torch.load("last.ckpt", map_location=torch.device("cpu"))

        # Instantiate the model first
        model = LinearClassifier(num_features, num_classes)

        # Adjust the state_dict if necessary
        state_dict = checkpoint["state_dict"]
        adjusted_state_dict = {
            k.replace("layer.", "model."): v for k, v in state_dict.items()
        }

        # Load the state dictionary
        load_result = model.load_state_dict(adjusted_state_dict, strict=False)

        if load_result.missing_keys or load_result.unexpected_keys:
            print("Warning: There were missing or unexpected keys during model loading")
            print("Missing keys:", load_result.missing_keys)
            print("Unexpected keys:", load_result.unexpected_keys)

        return model

    def _remap_index_to_species_id(self, df):
        # Aggregate and filter species based on image count
        grouped_df = (
            df.groupBy("species_id")
            .agg(F.count("species_id").alias("n"))
            .filter(F.col("n") >= self.species_image_count)
            .orderBy(F.col("n").desc(), F.col("species_id"))
            .withColumn("index", F.monotonically_increasing_id())
        ).drop("n")

        # Use broadcast join to optimize smaller DataFrame joining
        filtered_df = df.join(F.broadcast(grouped_df), "species_id", "inner")
        return filtered_df

    def _prepare_dataframe_submission(self, result_df, filtered_df):
        # Join both dataframes to get species_id
        joined_df = result_df.join(
            filtered_df, result_df.predictions == filtered_df.index, "inner"
        ).select(result_df.image_name, "species_id", "index", "predictions")
        # Create columns for submission
        final_df = (
            joined_df.withColumn(
                "plot_id", F.regexp_replace("image_name", "\\.jpg$", "")
            )
            .groupBy("plot_id")
            .agg(F.collect_set("species_id").alias("species_ids"))
        )
        # Convert the set of species_ids to a formatted string enclosed in single square brackets
        final_df = final_df.withColumn(
            "species_ids",
            F.concat(F.lit("["), F.concat_ws(", ", "species_ids"), F.lit("]")),
        )

    def _write_cvs_to_gcs(self, final_df):
        # Convert Spark DataFrame to Pandas DataFrame
        final_pandas_df = final_df.toPandas()

        # Export to CSV with the specified format
        output_dir = f"{self.default_root_dir}/{self.experiment_run_path}"
        final_pandas_df.to_csv(output_dir, sep=";", index=False, quoting=csv.QUOTE_NONE)

    def run(self):
        with spark_resource() as spark:
            # get dataframes
            gcs_path = "gs://dsgt-clef-plantclef-2024"
            test_path = "data/process/test_v1/dino_dct/data"
            dct_emb_train = "data/process/training_cropped_resized_v2/dino_dct/data"
            # data module parameters
            test_path = f"{gcs_path}/{test_path}"
            dct_gcs_path = f"{gcs_path}/{dct_emb_train}"
            # read data
            test_df = spark.read.parquet(test_path)
            dct_df = spark.read.parquet(dct_gcs_path)

            # get parameters for the model
            num_features = int(
                len(dct_df.select(self.feature_col).first()[self.feature_col])
            )
            num_classes = int(dct_df.select("species_id").distinct().count())

            # Get model
            model = self._load_model_from_gcs(
                num_features=num_features,
                num_classes=num_classes,
            )
            # Broadcast the model to send to all executors
            sc = SparkContext.getOrCreate()
            broadcast_model = sc.broadcast(model)

            @pandas_udf("long")  # Adjust the return type based on your model's output
            def predict_udf(dct_embedding_series: pd.Series) -> pd.Series:
                local_model = broadcast_model.value  # Access the broadcast variable
                local_model.eval()  # Set the model to evaluation mode

                # Convert the list of numpy arrays to a single numpy array
                embeddings_array = np.array(list(dct_embedding_series))

                # Convert the numpy array to a PyTorch tensor
                embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)

                # Make predictions
                with torch.no_grad():
                    outputs = local_model(embeddings_tensor)
                    predicted_classes = outputs.argmax(
                        dim=1
                    ).numpy()  # Get all predicted classes at once

                return pd.Series(predicted_classes)

            # get predictions on test_df
            result_df = test_df.withColumn(
                "predictions", predict_udf(test_df["dct_embedding"])
            )

            # remap the indexes to species and get dataframe
            filtered_df = self._remap_index_to_species_id(dct_df)

            # prepare dataframe for submission
            final_df = self._prepare_dataframe_submission(
                result_df=result_df,
                filtered_df=filtered_df,
            )

            # write CSV file to bucket
            self._write_cvs_to_gcs(final_df=final_df)

            # write the output
            with self.output().open("w") as f:
                f.write("")
