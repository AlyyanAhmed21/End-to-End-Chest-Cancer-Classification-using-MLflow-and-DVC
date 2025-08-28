# --- IMPORTS ---
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn 
import sys # <-- Add this import
from pathlib import Path # <-- Add this import

sys.path.append(str(Path(__file__).parent / "src"))

# Your existing ML components
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# --- CONFIGURATION ---
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# --- INITIALIZE FastAPI APP ---
app = FastAPI(
    title="Chest Cancer Classification API",
    description="An API to predict whether a chest CT scan shows signs of adenocarcinoma cancer."
)

# --- MIDDLEWARE (for CORS) ---
# This is the FastAPI equivalent of Flask-CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MOUNT STATIC FILES AND TEMPLATES ---
# This is how FastAPI serves your CSS, JS, and HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- LOAD THE PREDICTION PIPELINE ON STARTUP ---
# This ensures the model is loaded only once when the application starts.
classifier = PredictionPipeline(filename="inputImage.jpg")


# --- DEFINE THE REQUEST BODY STRUCTURE ---
# Pydantic model for automatic validation of the incoming JSON
class ImagePayload(BaseModel):
    image: str


# --- API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the main user interface (index.html).
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train")
async def trainRoute():
    """
    Triggers the DVC pipeline to retrain the model.
    NOTE: This is a blocking operation and not recommended for a real-world, high-traffic production server.
    It's suitable for this project's demonstration purposes.
    """
    os.system("dvc repro")
    return {"message": "Training done successfully!"}


@app.post("/predict")
async def predictRoute(payload: ImagePayload):
    """
    Accepts a base64 encoded image, saves it, runs prediction, and returns the result.
    """
    # 1. Decode the image and save it
    decodeImage(payload.image, "inputImage.jpg")

    # 2. Run the prediction pipeline
    prediction_value = classifier.predict()

    # 3. Translate the numeric prediction into a human-readable string
    # Based on your confirmed class indices: {'adenocarcinoma': 0, 'normal': 1}
    if prediction_value == 1:
        prediction_text = "Normal"
    else:  # The value was 0
        prediction_text = "Cancer"

    # 4. Return the result. FastAPI handles the JSON conversion.
    return [{"prediction": prediction_text}]


# --- RUN THE APP ---
# This block is for local development. Gunicorn/Uvicorn will run the app in production.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)