from fastapi import FastAPI, UploadFile, File
from utils import load_onnx_model, load_torch_model, predict_onnx, predict_torch
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
    # print(f"image bytes: {image_bytes}")
    preds = predict_torch(torch_model, image_bytes)
    print(f"preds: {preds}")
    
    # convert to bytes
    preds = io.BytesIO(preds)
    
    print(f"preds after converting to bytes: {preds}")
    
    return preds
    
@app.post("/predict_onnx")
def predict_onnx_api(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    # print(f"image bytes: {image_bytes}")
    preds = predict_onnx(onnx_model, image_bytes)
    print(f"preds: {preds}")
    return preds

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)