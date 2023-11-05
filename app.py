# UTIL
import io
import sys
import handler
import config
import base64


import numpy as np
from flask import Flask, request, Response, jsonify

# MODEL FRAMEWORK
import torch 
import tensorflow as tf
from handler import SimpleModel




# stop tensorflow from allocating all GPU memory
GPUS = tf.config.list_physical_devices("GPU")
if len(GPUS) > 0:
    tf.config.experimental.set_memory_growth(GPUS[0], True)

torch_arch = SimpleModel(5, 32, 1)
# load CV models
TF_MODEL = tf.keras.models.load_model(config.TF_MODEL_PATH)
TORCH_MODEL = torch.load(config.TORCH_MODEL_PATH)
TORCH_MODEL.eval()

# launch flask app
app = Flask(__name__)


@app.route("/health")
def health():
    return Response("{'health': 'ok'}", status=200)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    print(f"N GPUS available = {len(GPUS)}", file=sys.stderr)
    
    instances = request.get_json()["instances"]
    all_preds = list()
    for data in instances:
        tf_preds = tf.argmax(TF_MODEL.predict(data), axis=1).numpy()
        torch_preds = np.where(TORCH_MODEL(torch.tensor(data, dtype=torch.float32)).detach().numpy()[:,0]<0.5 , 0, 1)

        all_preds.append(
            {
                'TF':tf_preds.tolist(),
                'TORCH':torch_preds.tolist()
            }
        )

        return jsonify(all_preds)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

