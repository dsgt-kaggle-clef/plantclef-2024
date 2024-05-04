import os
from pathlib import Path

import pytorch_lightning as pl
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql import functions as F


class PetastormDataModule(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_path,
        feature_col,
        limit_species=None,
        species_image_count=100,
        batch_size=32,
        num_partitions=32,
        workers_count=os.cpu_count(),
    ):
        super().__init__()
        cache_dir = "file:///mnt/data/tmp"
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path(cache_dir).as_posix()
        )
        self.spark = spark
        self.input_path = input_path
        self.feature_col = feature_col
        self.limit_species = limit_species
        self.species_image_count = species_image_count
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.workers_count = workers_count

    def _prepare_species_data(self):
        """
        Prepare species data by filtering, indexing, and joining.
        :return: DataFrame of filtered and indexed species data
        """
        # Read the Parquet file into a DataFrame
        df = self.spark.read.parquet(self.input_path).cache()

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

    def _prepare_dataframe(self, df, partitions=32):
        """Prepare the DataFrame for training by ensuring correct types and repartitioning"""
        return (
            df.withColumnRenamed(self.feature_col, "features")
            .withColumnRenamed("index", "label")
            .select(
                F.col("features").cast("array<float>").alias("features"),
                F.col("label").cast("long").alias("label"),
            )
            .repartition(partitions)
        )

    def _train_valid_split(self, df):
        """
        Perform train/valid random split
        :return: train_df, valid_df Spark DataFrames
        """
        train_df, valid_df = df.randomSplit([0.8, 0.2], seed=42)
        train_df = self._prepare_dataframe(train_df)
        valid_df = self._prepare_dataframe(valid_df)
        return train_df.cache(), valid_df.cache()

    def setup(self, stage=None):
        # Get prepared data
        prepared_df = self._prepare_species_data().cache()
        # train/valid Split
        self.train_data, self.valid_data = self._train_valid_split(df=prepared_df)

        # setup petastorm data conversion from Spark to PyTorch
        self.converter_train = make_spark_converter(self.train_data)
        self.converter_valid = make_spark_converter(self.valid_data)

    def _dataloader(self, converter):
        with converter.make_torch_dataloader(
            batch_size=self.batch_size,
            num_epochs=1,
            workers_count=self.workers_count,
        ) as dataloader:
            for batch in dataloader:
                yield batch

    def train_dataloader(self):
        for batch in self._dataloader(self.converter_train):
            yield batch

    def val_dataloader(self):
        for batch in self._dataloader(self.converter_valid):
            yield batch
