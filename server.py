from fastapi import FastAPI, UploadFile, File, Response
from utils import load_onnx_model, load_torch_model, predict_onnx, predict_torch
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load the models
torch_model = load_torch_model("models/model.pt")
onnx_model = load_onnx_model("models/onnx_model.onnx")

@app.get("/")
def read_root():
    return {"Hello": "World"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
