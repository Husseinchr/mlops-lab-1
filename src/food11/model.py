"""Shared preprocessing and MLflow probability adapter."""

import mlflow.pyfunc
import mlflow.pytorch
import numpy as np
import torch
from PIL import ImageOps
from torchvision import transforms


def image_transform():
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def preprocess(image):
    return image_transform()(ImageOps.exif_transpose(image).convert("RGB"))


class ProbabilityModel(mlflow.pyfunc.PythonModel):
    """Expose NCHW inputs and canonical Food-11 probabilities via pyfunc."""

    def load_context(self, context):
        self.network = mlflow.pytorch.load_model(
            context.artifacts["network"], map_location="cpu"
        ).eval()

    def predict(self, context, model_input, params=None):
        batch = torch.as_tensor(np.asarray(model_input), dtype=torch.float32)
        with torch.inference_mode():
            return torch.softmax(self.network(batch), dim=1).cpu().numpy()
