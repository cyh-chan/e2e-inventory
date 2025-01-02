import torch
from onnx2torch import convert

# Path to your ONNX model
onnx_model_path = "e2e_model.onnx"

# Load the ONNX model and convert to PyTorch
pytorch_model = convert(onnx_model_path)

# Check the structure of the PyTorch model
# print(pytorch_model)

