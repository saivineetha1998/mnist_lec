from flask import Flask
from flask import request
import numpy as np
import pickle
app = Flask(__name__)

best_model_path = '/home/saivineetha/mnist_lec/mnist_lec/mnist_lec/better_models/model_0.01_0.2_0.1.sav'

def load(model_path):
    load_file = open(model_path, "rb")
    model = pickle.load(load_file)
    return model


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict', methods=['POST'])
def predict():
    clf = load(best_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    print("image:", image)
    print("prediction:", predicted[0])
    return str(predicted[0])
    # return "<p>Image obtained!</p>"