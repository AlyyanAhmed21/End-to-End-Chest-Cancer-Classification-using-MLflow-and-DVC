import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

# --- NEW IMPORTS for advanced evaluation ---
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# -------------------------------------------

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None
        self.y_true = None
        self.y_pred = None

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def _get_predictions(self):
        """Gets ground truth labels and model's predicted labels."""
        self.y_true = self.valid_generator.classes
        y_pred_probs = self.model.predict(self.valid_generator)
        self.y_pred = np.argmax(y_pred_probs, axis=1)

    def evaluation(self):
        """Loads model, evaluates basic metrics, and gets detailed predictions."""
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self._get_predictions()
        self.save_score()

    # In your Evaluation component's save_score method

    def save_score(self):
        # If self.score is None or contains NaN, create a default file
        if self.score is None or np.isnan(self.score).any():
            print("⚠️ Warning: Invalid scores detected (NaN). Saving default scores file.")
            scores = {"loss": float('nan'), "accuracy": float('nan')}
        else:
            scores = {"loss": self.score[0], "accuracy": self.score[1]}
        
        # This will now always create the file
        save_json(path=Path("scores.json"), data=scores)
        print(f"Scores saved to scores.json: {scores}")

    def log_confusion_matrix(self):
        """Generates, saves, and logs the confusion matrix plot to MLflow."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        class_names = list(self.valid_generator.class_indices.keys())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        matrix_path = Path("confusion_matrix.png")
        plt.savefig(matrix_path)
        
        mlflow.log_artifact(matrix_path, "plots")
        print("Confusion Matrix plot saved and logged to MLflow.")
    
    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        
        with mlflow.start_run():
            print("Logging basic parameters and metrics to MLflow...")
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # --- Log detailed classification report metrics ---
            print("\n--- Classification Report ---")
            report = classification_report(self.y_true, self.y_pred, 
                                           target_names=list(self.valid_generator.class_indices.keys()),
                                           output_dict=True)
            print(classification_report(self.y_true, self.y_pred, 
                                        target_names=list(self.valid_generator.class_indices.keys())))

            for className, metrics in report.items():
                if isinstance(metrics, dict):
                    for metricName, value in metrics.items():
                        mlflow.log_metric(f"{className}_{metricName}", value)

            # --- Log the confusion matrix plot ---
            self.log_confusion_matrix()
            
            # --- Log the model as an artifact ---
            print("Logging model as an artifact...")
            mlflow.keras.log_model(self.model, "model")
            
            print("MLflow logging complete.")