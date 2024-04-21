import wandb
import luigi
import luigi.contrib.gcs
import pytorch_lightning as pl
import torch
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from plantclef.baseline.data import PetastormDataModule
from plantclef.baseline.model import LinearClassifier
from plantclef.utils import spark_resource


class TrainDCTEmbeddingClassifier(luigi.Task):
    input_path = luigi.Parameter()
    feature_col = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    limit_species = luigi.OptionalIntParameter(default=None)
    species_image_count = luigi.IntParameter(default=100)
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.IntParameter(default=32)

    def output(self):
        # save the model run
        return luigi.contrib.gcs.GCSTarget(f"{self.default_root_dir}/_SUCCESS")

    def run(self):
        with spark_resource() as spark:
            # data module
            data_module = PetastormDataModule(
                spark,
                self.input_path,
                self.feature_col,
                self.limit_species,
                self.species_image_count,
                self.batch_size,
                self.num_partitions,
            )
            data_module.setup()

            # get parameters for the model
            num_features = int(
                len(
                    data_module.train_data.select(self.feature_col).first()[
                        self.feature_col
                    ]
                )
            )
            num_classes = int(
                data_module.train_data.select("species_id").distinct().count()
            )

            # model module
            model = LinearClassifier(
                num_features,
                num_classes,
            )

            # initialise the wandb logger and name your wandb project
            wandb_logger = WandbLogger(
                project='plantclef-2024', 
                name=Path(self.default_root_dir).name,
                save_dir=self.default_root_dir,
            )

            # add your batch size to the wandb config
            wandb_logger.experiment.config["batch_size"] = self.batch_size

            # trainer
            trainer = pl.Trainer(
                max_epochs=10,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.default_root_dir,
                logger=wandb_logger,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min"),
                    ModelCheckpoint(monitor="val_loss", save_last=True),
                ],
            )

            # fit model
            trainer.fit(model, data_module)

        # write the output
        with self.output().open("w") as f:
            f.write("")
