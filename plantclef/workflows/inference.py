import csv

import luigi
import luigi.contrib.gcs
import numpy as np
import pandas as pd
import torch
from google.cloud import storage
from pyspark import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf

from plantclef.baseline.model import LinearClassifier
from plantclef.utils import spark_resource


class InferenceTask(luigi.Task):
    input_path = luigi.Parameter()
    test_path = luigi.Parameter()
    feature_col = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    limit_species = luigi.OptionalIntParameter(default=None)
    species_image_count = luigi.IntParameter(default=100)

    def output(self):
        # save the model run
        return luigi.contrib.gcs.GCSTarget(
            f"{self.default_root_dir}/experiments/_SUCCESS"
        )

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

        # Optionally limit the number of species
        if self.limit_species is not None:
            limited_grouped_df = (
                (
                    grouped_df.orderBy(F.rand(seed=42))
                    .limit(self.limit_species)
                    .withColumn("new_index", F.monotonically_increasing_id())
                )
                .drop("index")
                .withColumnRenamed("new_index", "index")
            )

            filtered_df = filtered_df.drop("index").join(
                F.broadcast(limited_grouped_df), "species_id", "inner"
            )

        return filtered_df

    def _load_model_from_gcs(
        self,
        num_features: int,
        num_classes: int,
    ):
        bucket_name = "dsgt-clef-plantclef-2024"
        relative_path = self.default_root_dir.split(f"{bucket_name}/")[-1]
        path_in_bucket = f"{relative_path}/checkpoints/last.ckpt"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path_in_bucket)
        blob.download_to_filename("last.ckpt")
        checkpoint = torch.load("last.ckpt", map_location=torch.device("cpu"))

        # Instantiate the model first
        model = LinearClassifier(num_features, num_classes)

        # Adjust the state_dict if necessary
        state_dict = checkpoint["state_dict"]

        # Load the state dictionary
        load_result = model.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys or load_result.unexpected_keys:
            print("Warning: There were missing or unexpected keys during model loading")
            print("Missing keys:", load_result.missing_keys)
            print("Unexpected keys:", load_result.unexpected_keys)

        return model

    def _prepare_dataframe_submission(self, result_df, filtered_df):
        # Join both dataframes to get species_id
        joined_df = result_df.join(
            F.broadcast(filtered_df),
            result_df.predictions == filtered_df.index,
            "inner",
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
        return final_df.cache()

    def _write_cvs_to_gcs(self, final_df):
        # Convert Spark DataFrame to Pandas DataFrame
        final_pandas_df = final_df.toPandas()

        # Export to CSV with the specified format
        output_dir = f"{self.default_root_dir}/experiments/dsgt_run.csv"
        final_pandas_df.to_csv(output_dir, sep=";", index=False, quoting=csv.QUOTE_NONE)

    def run(self):
        with spark_resource() as spark:
            # get dataframes
            gcs_path = "gs://dsgt-clef-plantclef-2024"
            test_path = f"data/process/{self.test_path}"
            emb_path = self.input_path.split(gcs_path)[-1]

            # paths to dataframe
            test_path = f"{gcs_path}/{test_path}"
            emb_gcs_path = f"{gcs_path}/{emb_path}"
            # read data
            test_df = spark.read.parquet(test_path)
            emb_df = spark.read.parquet(emb_gcs_path)

            # remap the indexes to species and get dataframe
            filtered_df = self._remap_index_to_species_id(emb_df)

            # get parameters for the model
            num_features = int(
                len(filtered_df.select(self.feature_col).first()[self.feature_col])
            )
            num_classes = int(filtered_df.select("species_id").distinct().count())

            # Get model
            model = self._load_model_from_gcs(
                num_features=num_features,
                num_classes=num_classes,
            )
            # Broadcast the model to send to all executors
            sc = SparkContext.getOrCreate()
            broadcast_model = sc.broadcast(model)

            @pandas_udf("long")  # Adjust the return type based on model's output
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
                "predictions", predict_udf(test_df[self.feature_col])
            ).cache()  # caching the dataframe with 1695 rows

            # prepare dataframe for submission
            final_df = self._prepare_dataframe_submission(
                result_df=result_df,
                filtered_df=filtered_df,
            )

            # write CSV file to GCS
            self._write_cvs_to_gcs(final_df=final_df)

            # write the output
            with self.output().open("w") as f:
                f.write("")


class PretrainedInferenceTask(luigi.Task):
    input_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    k = luigi.OptionalIntParameter(default=5)
    use_grid = luigi.OptionalBoolParameter(default=False)
    grid_size = luigi.OptionalIntParameter(default=3)

    def output(self):
        # save the model run
        output_path = f"{self.default_root_dir}/top_{self.k}_species/_SUCCESS"
        if self.use_grid:
            output_path = f"{self.default_root_dir}/top_{self.k}_species_grid_{self.grid_size}x{self.grid_size}/_SUCCESS"
        return luigi.contrib.gcs.GCSTarget(output_path)

    def _format_species_ids(self, species_ids: list) -> str:
        """Formats the species IDs in single square brackets, separated by commas."""
        formatted_ids = ", ".join(str(id) for id in species_ids)
        return f"[{formatted_ids}]"

    def _extract_top_k_species(self, logits: list) -> list:
        """Extracts the top k species from the logits list."""
        top_logits = [list(item.keys())[0] for item in logits[: self.k]]
        set_logits = sorted(set(top_logits), key=top_logits.index)
        return set_logits

    def _remove_extension(self, filename: str) -> str:
        """Removes the file extension from the filename."""
        return filename.rsplit(".", 1)[0]

    def _prepare_and_write_submission(self, spark_df: DataFrame) -> DataFrame:
        """Converts Spark DataFrame to Pandas, formats it, and writes to GCS."""
        records = []
        for row in spark_df.collect():
            image_name = self._remove_extension(row["image_name"])
            logits = row["dino_logits"]
            top_k_species = self._extract_top_k_species(logits)
            formatted_species = self._format_species_ids(top_k_species)
            records.append({"plot_id": image_name, "species_ids": formatted_species})

        pandas_df = pd.DataFrame(records)
        return pandas_df

    def _write_csv_to_gcs(self, df):
        """Writes the Pandas DataFrame to a CSV file in GCS."""
        folder_name = f"top_{self.k}_species"
        if self.use_grid:
            grid_name = f"grid_{self.grid_size}x{self.grid_size}"
            folder_name = f"{folder_name}_{grid_name}"
        file_name = f"dsgt_run_{folder_name}.csv"
        output_path = f"{self.default_root_dir}/{folder_name}/{file_name}"
        df.to_csv(output_path, sep=";", index=False, quoting=csv.QUOTE_NONE)

    def run(self):
        with spark_resource() as spark:
            # read data
            transformed_df = spark.read.parquet(self.input_path)
            transformed_df = transformed_df.orderBy("image_name")

            # get prepared dataframe
            pandas_df = self._prepare_and_write_submission(transformed_df)
            self._write_csv_to_gcs(pandas_df)

            # write the output
            with self.output().open("w") as f:
                f.write("")