import onnxruntime as rt
import torch
from fastapi import UploadFile
import pandas as pd
from PIL import Image
import io
import numpy as np

torch_model_path = "models/model.pt"
onnx_model_path = "models/onnx_model.onnx"

def file_to_img(file: UploadFile, out_shape=(1, 1, 256, 256)):
    file_content = io.BytesIO(file.file.read())
    img = Image.open(file_content).convert("L")
    img = img.resize((out_shape[3], out_shape[2]))
    img = torch.tensor(np.array(img)).float().unsqueeze(0).unsqueeze(0)

def load_onnx_model(onnx_model_path):
    return rt.InferenceSession(onnx_model_path)

def load_torch_model(torch_model_path, device=torch.device("cpu")):
    model = torch.jit.load(torch_model_path, map_location=device)
    model.eval()
    return model

def predict_onnx(model, file):
    input = file_to_img(file).numpy()
    out = model.run(None, {"input.1": input})
    return out[0]

def predict_torch(model, file):
    input = file_to_img(file)
    out = model(input)
    return out.cpu().detach().numpy()

def pick_random_image(annotation_file):
    import random
    df = pd.read_csv(annotation_file)
    img_path = random.choice(df.iloc[:, 0])
    img = Image.open(img_path).convert("L")
    return img
    

img = pick_random_image("data/annotations_dataframe.csv")
