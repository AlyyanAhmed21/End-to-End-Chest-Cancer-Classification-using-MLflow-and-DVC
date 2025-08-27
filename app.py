from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables for consistent encoding
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """Renders the main user interface."""
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    """Triggers the DVC pipeline to retrain the model."""
    # os.system("python main.py") # You can use this if you have a main orchestrator
    os.system("dvc repro")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    
    # The predict() method now returns just the index (0 or 1)
    prediction_value = clApp.classifier.predict()
    
    # This logic is confirmed by your class indices: {'adenocarcinoma': 0, 'normal': 1}
    if prediction_value == 1:
        prediction_text = "Normal"
    else: # The value was 0
        prediction_text = "Cancer"

    # The front-end expects the key "prediction"
    return jsonify([{"prediction": prediction_text}])


if __name__ == "__main__":
    clApp = ClientApp()
    # Run the app on all available interfaces (for Docker/deployment) and port 8080
    app.run(host='0.0.0.0', port=8080)