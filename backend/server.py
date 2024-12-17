import io
from fastapi import FastAPI, File, Response, UploadFile

from PIL import Image
from utils import (
    predict_onnx,
    predict_torch,
    load_onnx_model,
    load_torch_model
)


app = FastAPI()

# load the models
torch_model = load_torch_model("/app/models/model.pt")
onnx_model = load_onnx_model("/app/models/onnx_model.onnx")


@app.get("/_stcore/health")
def health():
    return {"status": "healthy"}


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
