import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

# --- NEW IMPORTS ---
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# --------------------

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20, # Reduced for stability
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
        # --- ADD THIS ---
        # Print class indices to be 100% sure of the mapping
        print(f"Discovered class indices: {self.train_generator.class_indices}")
        # --------------

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        # --- NEW: DEFINE CALLBACKS FOR SMART TRAINING ---
        # This will save the BEST model based on validation accuracy
        best_model_checkpoint = ModelCheckpoint(
            filepath=self.config.trained_model_path, # Saves the best model to your specified path
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

        # This will stop training if there's no improvement
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5, # Number of epochs with no improvement to wait
            restore_best_weights=True,
            verbose=1
        )
        
        callbacks_list = [best_model_checkpoint, early_stopping]
        # -----------------------------------------------

        # --- MODEL.FIT() IS NOW UPGRADED ---
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks_list # Pass the smart callbacks here
        )
        # -------------------------------------

        # --- NEW: SAVE TRAINING HISTORY FOR ANALYSIS ---
        history_df = pd.DataFrame(history.history)
        history_path = "training_history.csv" # Saved in the root directory
        history_df.to_csv(history_path, index=False)
        print(f"✅ Training history saved to {history_path}")
        # -----------------------------------------------
        
        # The save_model call is now handled by ModelCheckpoint,
        # so this is redundant but harmless. It will save the last epoch's model.
        # The BEST model is already saved by the callback.
        # self.save_model(
        #     path=self.config.trained_model_path,
        #     model=self.model
        # )