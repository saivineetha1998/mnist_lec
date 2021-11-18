from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict', methods=['POST'])
def predict():
    # clf = load(best_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    return None