from flask import Flask
from flask import request
import numpy as np
import pickle
app = Flask(__name__)

svm_model_path = './mnist_lec/final_exam_models/'



def load(model_path):
    load_file = open(model_path, "rb")
    model = pickle.load(load_file)
    return model


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/svm_predict', methods=['POST'])
def svm_predict():
    clf = load(svm_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    print("image:", image)
    print("prediction:", predicted[0])
    return str(predicted[0])
    # return "<p>Image obtained!</p>"


@app.route('/decision_tree_predict', methods=['POST'])
def dt_predict():
    clf = load(dt_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    print("image:", image)
    print("prediction:", predicted[0])
    return str(predicted[0])
    # return "<p>Image obtained!</p>"


if __name__ == '__main__':
    app.run(host='0.0.0.0')