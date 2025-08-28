# --- IMPORTS ---
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# NEW, CRITICAL IMPORT for running behind a proxy
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

# --- ADD THIS BLOCK TO FIX 'ModuleNotFoundError' ---
# This adds the 'src' directory to the Python path
# so it can find the cnnClassifier package in the Docker container.
sys.path.append(str(Path(__file__).parent / "src"))
# ----------------------------------------------------

# Now we can import your custom ML components
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

# --- ADD PROXY MIDDLEWARE (FIXES HTTPS/MIXED CONTENT ERROR) ---
# This middleware is essential for running behind a reverse proxy like Hugging Face Spaces.
# It tells the app to trust the 'x-forwarded-proto' header from the proxy.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
# ------------------------------------------------------------

# --- MIDDLEWARE (for CORS) ---
# This should come AFTER the ProxyHeadersMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MOUNT STATIC FILES AND TEMPLATES ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- LOAD THE PREDICTION PIPELINE ON STARTUP ---
classifier = PredictionPipeline(filename="inputImage.jpg")

# --- DEFINE THE REQUEST BODY STRUCTURE ---
class ImagePayload(BaseModel):
    image: str

# --- API ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the main user interface (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/train")
async def trainRoute():
    """Triggers the DVC pipeline to retrain the model."""
    os.system("dvc repro")
    return {"message": "Training done successfully!"}


@app.post("/predict")
async def predictRoute(payload: ImagePayload):
    """
    Accepts a base64 encoded image, saves it to a temporary location,
    runs prediction, and returns the result.
    """
    # --- THIS IS THE FIX ---
    # Define a writable filename inside the /tmp directory.
    temp_image_path = "/tmp/inputImage.jpg"
    
    # 1. Decode the image and save it to the temporary path
    decodeImage(payload.image, temp_image_path)

    # 2. Update the classifier's filename to the new temporary path and predict
    classifier.filename = temp_image_path
    prediction_value = classifier.predict()
    # ----------------------

    # 3. Translate the numeric prediction into a human-readable string
    if prediction_value == 1:
        prediction_text = "Normal"
    else:
        prediction_text = "Cancer"

    # 4. Return the result
    return [{"prediction": prediction_text}]

# --- RUN THE APP ---
if __name__ == "__main__":
    # Note: Hugging Face uses port 7860 by default for its apps
    uvicorn.run(app, host="0.0.0.0", port=7860)