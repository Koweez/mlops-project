import io

import mlflow
import numpy as np
from fastapi import FastAPI, File, Response, UploadFile
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from PIL import Image
from utils import (
    load_onnx_model,
    load_torch_model,
    predict_mlflow,
    predict_onnx,
    predict_torch,
)

app = FastAPI()

mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()


loaded_models = {}


@app.on_event("startup")
def register_and_load_models():
    models = [
        {
            "name": "torch_segmentation_model",
            "path": "/app/models/model.pt",
            "framework": "torch",
        },
        {
            "name": "onnx_segmentation_model",
            "path": "/app/models/onnx_model.onnx",
            "framework": "onnx",
        },
    ]

    for model in models:
        model_name = model["name"]
        model_path = model["path"]
        framework = model["framework"]

        if any(m.name == model_name for m in client.search_registered_models()):
            print(f"model '{model_name}' is already registered.")
        else:
            signature = infer_signature(np.random.randn(1, 1, 256, 256))
            try:
                with mlflow.start_run(run_name=f"registering {model_name}"):
                    if framework == "torch":
                        mlflow.pytorch.log_model(
                            model=mlflow.pytorch.load_model(model_path),
                            artifact_path="model",
                            signature=signature,
                        )
                    elif framework == "onnx":
                        mlflow.onnx.log_model(
                            onnx_model=model_path,
                            artifact_path="model",
                            signature=signature,
                        )
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


# load the models
torch_model = load_torch_model("/app/models/model.pt")
onnx_model = load_onnx_model("/app/models/onnx_model.onnx")


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
