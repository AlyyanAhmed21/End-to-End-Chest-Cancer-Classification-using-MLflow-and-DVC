import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # --- FIX #1: LOAD THE CORRECT MODEL ---
        # Load the BEST model produced by your DVC pipeline.
        model_path = os.path.join("artifacts", "training", "best_model.h5")
        model = tf.keras.models.load_model(model_path)

        # --- Load and preprocess the image ---
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image_array = image.img_to_array(test_image)
        
        # --- FIX #2: THE CRITICAL RESCALING STEP ---
        # Scale the pixel values to be between 0 and 1, just like the training data.
        scaled_image_array = test_image_array / 255.0
        
        # Add the batch dimension
        input_data = np.expand_dims(scaled_image_array, axis=0)

        # --- Make the prediction on the CORRECTLY preprocessed image ---
        result_index = np.argmax(model.predict(input_data), axis=1)[0]
        print(f"Model predicted index: {result_index}")

        # --- FIX #3: RETURN THE CORRECT JSON STRUCTURE ---
        # The logic for translation should be in app.py to keep this pipeline clean,
        # but for now, we will just return the raw index.
        # app.py will handle translating 0/1 to "Cancer"/"Normal".
        return result_index