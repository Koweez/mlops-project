from fastapi import FastAPI, UploadFile
from utils import load_onnx_model, load_torch_model, predict_onnx, predict_torch

app = FastAPI()

# Load the models
torch_model = load_torch_model("models/model.pt")
onnx_model = load_onnx_model("models/onnx_model.onnx")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_torch")
def predict_torch_api(file: UploadFile):
    return predict_torch(torch_model, file)

@app.post("/predict_onnx")
def predict_onnx_api(file: UploadFile):
    return predict_onnx(onnx_model, file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
