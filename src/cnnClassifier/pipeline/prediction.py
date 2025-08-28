import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        """
        Initializes the prediction pipeline.

        This is where we load the model ONCE when the application starts.
        This is much more efficient than loading it for every prediction.
        """
        self.filename = filename
        
        # --- THIS IS THE FIX ---
        # The Dockerfile places the model in a 'model' directory.
        # This is the correct path inside the container.
        model_path = os.path.join("model", "best_model.h5")
        self.model = tf.keras.models.load_model(model_path)
        # ----------------------

    def predict(self):
        """
        Performs the prediction on the image file.
        It uses the model that was already loaded in the constructor.
        """
        # Load and preprocess the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image_array = image.img_to_array(test_image)
        
        # Scale the pixel values to be between 0 and 1, just like the training data.
        scaled_image_array = test_image_array / 255.0
        
        # Add the batch dimension
        input_data = np.expand_dims(scaled_image_array, axis=0)

        # Make the prediction using the pre-loaded model
        prediction_probs = self.model.predict(input_data)
        result_index = np.argmax(prediction_probs, axis=1)

        # Return the raw index (e.g., [0] or [1])
        return result_index