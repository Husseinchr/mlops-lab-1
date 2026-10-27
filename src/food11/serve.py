"""Food-11 inference API, loading the registry alias once at startup."""

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .data import CLASSES
from .model import preprocess

model = None
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    global model
    if not os.getenv("SKIP_MODEL"):
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        )
        attempts = int(os.getenv("MODEL_LOAD_ATTEMPTS", "30"))
        for attempt in range(attempts):
            try:
                model = await asyncio.to_thread(
                    mlflow.pyfunc.load_model,
                    os.getenv("MODEL_URI", "models:/food11@champion"),
                )
                break
            except Exception:
                if attempt == attempts - 1:
                    raise
                logger.warning("Model unavailable; retry %s/%s", attempt + 1, attempts)
                await asyncio.sleep(2)
    yield
    model = None


app = FastAPI(title="Food-11 classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile):
    if model is None:
        raise HTTPException(503, "Model is not loaded")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds 10 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            batch = preprocess(image).unsqueeze(0).numpy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Upload must be a valid image") from None
    probabilities = np.asarray(model.predict(batch), dtype=float)
    if (
        probabilities.shape != (1, len(CLASSES))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise HTTPException(500, "Model returned invalid probabilities")
    index = int(probabilities[0].argmax())
    return {"category": CLASSES[index], "confidence": float(probabilities[0, index])}
