from argparse import ArgumentParser

import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from plantclef.transforms import DCTN, WrappedDinoV2
from plantclef.utils import spark_resource


class ProcessBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_partitions = luigi.IntParameter(default=500)

    def output(self):
        # save both the model pipeline and the dataset
        return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/_SUCCESS")

    @property
    def feature_columns(self) -> list:
        raise NotImplementedError()

    def pipeline(self) -> Pipeline:
        raise NotImplementedError()

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)
        for c in features:
            # check if the feature is a vector and convert it to an array
            if "array" in transformed.schema[c].simpleString():
                continue
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

    def _get_subset(self, df):
        # Get subset of images to test pipeline
        subset_df = (
            df.where(F.col("species_id").isin([1361703, 1355927]))
            .orderBy(F.rand(1000))
            .limit(200)
            .cache()
        )
        return subset_df

    def run(self):
        with spark_resource(
            **{"spark.sql.shuffle.partitions": max(self.num_partitions, 200)}
        ) as spark:
            df = spark.read.parquet(self.input_path)

            # Get subset of data to test pipeline
            df = self._get_subset(df=df)

            model = self.pipeline().fit(df)
            model.write().overwrite().save(f"{self.output_path}/model")
            transformed = self.transform(model, df, self.feature_columns)
            transformed.repartition(self.num_partitions).write.mode(
                "overwrite"
            ).parquet(f"{self.output_path}/data")

        # now write the success file
        with self.output().open("w") as f:
            f.write("")


class ProcessDino(ProcessBase):
    @property
    def feature_columns(self):
        return ["dino_embedding"]

    def pipeline(self):
        dino = WrappedDinoV2(input_col="data", output_col="dino_embedding")
        return Pipeline(
            stages=[
                dino,
                SQLTransformer(
                    statement=f"SELECT image_name, species_id, dino_embedding FROM __THIS__"
                ),
            ]
        )


class ProcessDCT(ProcessBase):
    @property
    def feature_columns(self):
        return ["dct_embedding"]

    def pipeline(self):
        dct = DCTN(input_col="dino_embedding", output_col="dct_embedding")
        return Pipeline(
            stages=[
                dct,
                SQLTransformer(
                    statement=f"SELECT image_name, species_id, dct_embedding FROM __THIS__"
                ),
            ]
        )


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        feature_columns = {
            "dino": "data",
            "dct": "dino_embedding",
        }

        yield ProcessDino(
            input_path=self.input_path,
            output_path=f"{self.output_path}/dino",
        )
        yield ProcessDCT(
            input_path=f"{self.output_path}/dino",
            output_path=f"{self.output_path}/dino_dct",
        )


def parse_args():
    parser = ArgumentParser(description="Luigi pipeline")
    parser.add_argument(
        "--gcs-root-path",
        type=str,
        default="gs://dsgt-clef-plantclef-2024",
        help="Root directory for plantclef-2024 in GCS",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/parquet_files/PlantCLEF2024_training_cropped_resized_v2",
        help="Root directory for training data in GCS",
    )
    parser.add_argument(
        "--output-name-path",
        type=str,
        default="data/process/training_cropped_resized_v2",
        help="GCS path for output Parquet files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Input and output paths
    input_path = f"{args.gcs_root_path}/{args.train_data_path}"
    output_path = f"{args.gcs_root_path}/{args.output_name_path}"

    luigi.build(
        [Workflow(input_path=input_path, output_path=output_path)],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
