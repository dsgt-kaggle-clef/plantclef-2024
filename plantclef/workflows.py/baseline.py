import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F, DataFrame
from argparse import ArgumentParser

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
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

    def run(self):
        with spark_resource(
            **{"spark.sql.shuffle.partitions": max(self.num_partitions, 200)}
        ) as spark:
            df = spark.read.parquet(self.input_path)
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
        pass


class ProcessDFT(ProcessBase):
    @property
    def feature_columns(self):
        return ["dft_embedding"]

    def pipeline(self):
        pass


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        raise NotImplementedError()


def parse_args():
    parser = ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    luigi.build(
        [],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
