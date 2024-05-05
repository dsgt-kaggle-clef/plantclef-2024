import io

import numpy as np
import timm
import torch
from PIL import Image
from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
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
        model_name="facebook/dinov2-base",
        batch_size=8,
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
        input_col="dino_embedding",
        output_col="cls_embedding",
        token_dimension=768,
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
    """
    Wrapper for Pretrained DinoV2 to add it to the pipeline
    """

    def __init__(
        self,
        pretrained_path: str,
        input_col: str = "input",
        output_col: str = "output",
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size=8,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.model_name = model_name
        self.batch_size = batch_size
        self.pretrained_path = pretrained_path
        self.num_classes = 7806  # total number of plant species
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=self.pretrained_path,
        )
        self.model.to(self.device)

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        def predict(inputs: np.ndarray) -> np.ndarray:
            images = [Image.open(io.BytesIO(input)) for input in inputs]
            model_inputs = self.processor(images=images, return_tensors="pt").to(
                self.device
            )
            # # Move inputs to device
            # model_inputs = {
            #     key: value.to(self.device) for key, value in model_inputs.items()
            # }

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
