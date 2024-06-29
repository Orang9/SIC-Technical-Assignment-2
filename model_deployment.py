import joblib
from flask import Flask, request, jsonify

def save_model(model, file_name):
    joblib.dump(model, file_name)

def load_model(file_name):
    return joblib.load(file_name)

def create_app(model):
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        prediction = model.predict([data])
        return jsonify({'prediction': prediction[0]})

    return app
