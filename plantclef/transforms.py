import io

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, MapType, StringType
from scipy.fftpack import dctn
from transformers import AutoImageProcessor, AutoModel


class WrappedDinoV2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for DinoV2 to add it to the pipeline
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        model_name: str = "facebook/dinov2-base",
        batch_size: int = 8,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        # Move model to GPU if available
        self.model.to(self.device)

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        def predict(inputs: np.ndarray) -> np.ndarray:
            images = [Image.open(io.BytesIO(input)) for input in inputs]
            model_inputs = self.processor(images=images, return_tensors="pt")
            # Move inputs to device
            model_inputs = {
                key: value.to(self.device) for key, value in model_inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**model_inputs)
                last_hidden_states = outputs.last_hidden_state

            numpy_array = last_hidden_states.cpu().numpy()
            new_shape = numpy_array.shape[:-2] + (-1,)
            numpy_array = numpy_array.reshape(new_shape)
            return numpy_array

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.batch_size,
            )(self.getInputCol()),
        )


class DCTN(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Run n-dimensional DCT on the input column
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        filter_size: int = 8,
        batch_size: int = 8,
        input_tensor_shapes=[[257, 768]],
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.input_tensor_shapes = input_tensor_shapes

    def _make_predict_fn(self):
        def dctn_filter(tile, k):
            coeff = dctn(tile)
            coeff_subset = coeff[:k, :k]
            return coeff_subset.flatten()

        def predict(inputs: np.ndarray) -> np.ndarray:
            # inputs is a 3D array of shape (batch_size, img_dim, img_dim)
            return np.array([dctn_filter(x, k=self.filter_size) for x in inputs])

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.batch_size,
                input_tensor_shapes=self.input_tensor_shapes,
            )(self.getInputCol()),
        )


class ExtractCLSToken(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Extract the CLS token (first 768 elements) from the Dino embedding column.
    """

    def __init__(
        self,
        input_col: str = "dino_embedding",
        output_col: str = "cls_embedding",
        token_dimension: int = 768,
    ):
        super(ExtractCLSToken, self).__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.token_dimension = token_dimension

    def _transform(self, df):
        # Extract the CLS token using slice function
        return df.withColumn(
            self.output_col, F.slice(F.col(self.input_col), 1, self.token_dimension)
        )


class PretrainedDinoV2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
        self,
        pretrained_path: str,
        input_col: str = "input",
        output_col: str = "output",
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
        grid_size: int = 3,
        use_grid: bool = False,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.model_name = model_name
        self.batch_size = batch_size
        self.pretrained_path = pretrained_path
        self.num_classes = 7806  # total number of plant species
        self.local_directory = "/mnt/data/models/pretrained_models"
        self.class_mapping_file = f"{self.local_directory}/class_mapping.txt"
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=self.pretrained_path,
        )
        self.model.to(self.device)
        self.model.eval()
        # Data transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        self.cid_to_spid = self._load_class_mapping()
        self.use_grid = use_grid
        self.grid_size = grid_size

    def _load_class_mapping(self):
        with open(self.class_mapping_file) as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def _split_into_grid(self, image):
        w, h = image.size
        grid_w, grid_h = w // self.grid_size, h // self.grid_size
        images = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = i * grid_w
                upper = j * grid_h
                right = left + grid_w
                lower = upper + grid_h
                crop_image = image.crop((left, upper, right, lower))
                images.append(crop_image)
        return images

    def _make_predict_fn(self):
        def predict(input_data):
            img = Image.open(io.BytesIO(input_data))
            top_k_proba = 20
            limit_logits = 20
            images = [img]
            # Use grid to get logits
            if self.use_grid:
                images = self._split_into_grid(img)
                top_k_proba = 10
                limit_logits = 5
            results = []
            for img in images:
                processed_image = self.transforms(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1) * 100
                    top_probs, top_indices = torch.topk(probabilities, k=top_k_proba)
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
                result = [
                    {self.cid_to_spid.get(index, "Unknown"): float(prob)}
                    for index, prob in zip(top_indices, top_probs)
                ]
                results.append(result)
            # Flatten the results from all grids, get top 5 probabilities
            flattened_results = [
                item for grid in results for item in grid[:limit_logits]
            ]
            # Sort by score in descending order
            sorted_results = sorted(
                flattened_results, key=lambda x: -list(x.values())[0]
            )
            return sorted_results

        return predict

    def _transform(self, df: DataFrame):
        predict_fn = self._make_predict_fn()
        predict_udf = F.udf(predict_fn, ArrayType(MapType(StringType(), FloatType())))
        return df.withColumn(
            self.getOutputCol(), predict_udf(F.col(self.getInputCol()))
        )


class WrappedPretrainedDinoV2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for DinoV2 to add it to the pipeline
    """

    def __init__(
        self,
        pretrained_path: str,
        input_col: str = "input",
        output_col: str = "output",
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.pretrained_path = pretrained_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_classes = 7806  # total number of plant species
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=self.pretrained_path,
        )
        # Data transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        # Move model to GPU if available
        self.model.to(self.device)

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        def predict(inputs: np.ndarray) -> np.ndarray:
            images = [Image.open(io.BytesIO(input)) for input in inputs]
            model_inputs = torch.stack(
                [self.transforms(img).to(self.device) for img in images]
            )

            with torch.no_grad():
                features = self.model.forward_features(model_inputs)
                cls_token = features[:, 0, :]

            numpy_array = cls_token.cpu().numpy()
            return numpy_array

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.batch_size,
            )(self.getInputCol()),
        )
