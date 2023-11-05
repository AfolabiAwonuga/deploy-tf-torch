import os
import pathlib

MODEL_ROOT = os.path.join(pathlib.Path(__file__).parent, "models")
TF_MODEL_PATH = os.path.join(MODEL_ROOT, "tf_model")
TORCH_MODEL_PATH = os.path.join(MODEL_ROOT, "torch_model.pth")

# print(TF_MODEL_PATH)