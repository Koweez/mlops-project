import io

import numpy as np
import onnxruntime as rt
import torch
from fastapi import UploadFile
from PIL import Image

torch_model_path = "models/model.pt"
onnx_model_path = "models/onnx_model.onnx"


def file_to_img(file: UploadFile, out_shape=(1, 1, 256, 256)):
    file_content = io.BytesIO(file.file.read())
    img = Image.open(file_content).convert("L")
    img = img.resize((out_shape[3], out_shape[2]))
    img = torch.tensor(np.array(img)).float().unsqueeze(0).unsqueeze(0)
    return img


def bytes_to_img(image_bytes: bytes, out_shape=(1, 1, 256, 256)):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((out_shape[3], out_shape[2]))
    img = torch.tensor(np.array(img)).float().unsqueeze(0).unsqueeze(0)
    return img


def load_onnx_model(onnx_model_path):
    return rt.InferenceSession(onnx_model_path)


def load_torch_model(torch_model_path, device=torch.device("cpu")):
    model = torch.jit.load(torch_model_path, map_location=device)
    model.eval()
    return model


def predict_onnx(model, image_bytes: bytes):
    input = bytes_to_img(image_bytes)
    out = model.run(None, {"input.1": input.numpy()})
    return out[0]


def predict_torch(model, image_bytes: bytes):
    input = bytes_to_img(image_bytes)
    out = model(input)
    return out.cpu().detach().numpy()
