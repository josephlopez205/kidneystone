import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("saved_steps.pkl", "rb"))
model = model["model"]

@app.route("/")
def Home():
    return render_template("index.html", prediction_text = "")

def text_renderer(array):
    if array == [0]:
        return "You are not at risk at developing kidney stones."
    elif array == [1]:
        return "You are at risk at developing kidney stones."
    else:
        return "Invalid"

def float_converter(array):
    return [float(x) for x in array]

def array_converter(i):
    return [np.array(i)]

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = float_converter(request.form.values())
    features = array_converter(float_features)
    prediction = model.predict(features)
    text = text_renderer(prediction)
    
    return render_template("index.html", prediction_text = text)

if __name__ == "__main__":
    app.run(debug=True)
