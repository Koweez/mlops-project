import io
import os
from contextlib import asynccontextmanager

import mlflow
import onnx
import torch
from fastapi import FastAPI, File, Response, UploadFile
from mlflow.tracking import MlflowClient
from PIL import Image
from utils import (
    predict_mlflow,
    predict_onnx,
    predict_torch,
)

client = MlflowClient()

# env variables
model_path = os.getenv("MODEL_PATH", "models/")
mlflow_path = os.getenv("MLFLOW_PATH", "http://127.0.0.1:5000")

torch_model_path = model_path + "model.pt"
onnx_model_path = model_path + "onnx_model.onnx"

models = [
    {
        "name": "torch_segmentation_model",
        "path": torch_model_path,
        "framework": "torch",
    },
    {
        "name": "onnx_segmentation_model",
        "path": onnx_model_path,
        "framework": "onnx",
    },
]

loaded_models = {}  # to cache the models in memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server started")
    # before the server starts
    mlflow.set_tracking_uri(mlflow_path)
    for model in models:
        model_name = model["name"]
        model_path = model["path"]
        framework = model["framework"]

        if any(m.name == model_name for m in client.search_registered_models()):
            print(f"model '{model_name}' is already registered.")
        else:
            # signature = infer_signature(np.random.randn(1, 1, 256, 256))
            try:
                with mlflow.start_run(run_name=f"registering {model_name}"):
                    if framework == "torch":
                        model = torch.load(model_path)
                        mlflow.pytorch.log_model(model, "pytorch_model")
                    elif framework == "onnx":
                        model = onnx.load(model_path)
                        mlflow.onnx.log_model(model, "onnx_model")
                    else:
                        raise ValueError(f"unsupported framework: {framework}")
                print(f"model '{model_name}' registered successfully.")
            except Exception as e:
                print(f"error registering model '{model_name}': {e}")

        # Load the model from MLflow and cache it
        try:
            loaded_model = load_model_from_mlflow(model_name, stage="Production")
            loaded_models[model_name] = loaded_model
            print(f"model '{model_name}' loaded and cached successfully.")
        except Exception as e:
            print(f"error loading model '{model_name}': {e}")
    yield
    # after the server stops
    print("Server stopped")


app = FastAPI(lifespan=lifespan)

# load the models
# torch_model = load_torch_model("/app/models/model.pt")
# onnx_model = load_onnx_model("/app/models/onnx_model.onnx")


@app.get("/_stcore/health")
def health():
    return {"status": "healthy"}


@app.get("/get_available_models")
def get_available_models():
    models = client.search_registered_models()
    models_list = [
        {
            "name": model.name,
            "version": model.latest_versions[0].version,
            "stage": model.latest_versions[0].current_stage,
            "run_id": model.latest_versions[0].run_id,
        }
        for model in models
    ]

    return models_list


def load_model_from_mlflow(model_name: str, stage: str):
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)


@app.post("/predict")
def predict_api(
    file: UploadFile = File(...),
    model_name: str = "torch_segmentation_model",
    stage: str = "Production",
):
    image_bytes = file.file.read()
    model = load_model_from_mlflow(model_name, stage)

    preds = predict_mlflow(model, image_bytes)

    preds = preds.squeeze()

    preds = 255 - preds

    preds_image = Image.fromarray(preds.astype("uint8"))

    buffer = io.BytesIO()
    preds_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="image/png")


@app.post("/predict_torch")
def predict_torch_api(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    preds = predict_torch(torch_model, image_bytes)

    # squeeze the image (numpy array) to remove batch and channel dimensions
    preds = preds.squeeze()

    # black and white inversion
    preds = 255 - preds

    # convert to image
    preds_image = Image.fromarray(preds.astype("uint8"))

    # convert to bytes
    buffer = io.BytesIO()
    preds_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="image/png")


@app.post("/predict_onnx")
def predict_onnx_api(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    preds = predict_onnx(onnx_model, image_bytes)

    # squeeze the image (numpy array) to remove batch and channel dimensions
    preds = preds.squeeze()

    # black and white inversion
    preds = 255 - preds

    # convert to image
    preds_image = Image.fromarray(preds.astype("uint8"))

    # convert to bytes
    buffer = io.BytesIO()
    preds_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="image/png")
