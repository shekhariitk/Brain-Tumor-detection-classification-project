import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from src.entity.config_entity import TrainingConfig
from src.components.callbacks import Callbacks, ClassWeightCalculator
from src.logger import logging

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.class_info = {}  # Dictionary to store class information

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",  # Use "binary" for binary classification, "categorical" for multi-class
            color_mode="grayscale",  # Set to "" for grayscale, "rgb" for color
            shuffle=True
        )
        
        logging.info("Creating training and validation generators...")
        if self.config.params_is_augmentation:
            logging.info("Data augmentation is enabled.")
            # If augmentation is enabled, add augmentation parameters
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                validation_split=0.2,  # Add validation split
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                validation_split=0.2,
                **datagenerator_kwargs
            )

        # Training generator
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "Training"),
            subset="training",
            **dataflow_kwargs
        )

        # Validation generator (from training data)
        self.valid_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "Training"),
            subset="validation",
            **dataflow_kwargs
        )


        # Store class information for confusion matrix
        self.class_info = {
            'class_indices': self.train_generator.class_indices,
            'class_names': list(self.train_generator.class_indices.keys()),
            'num_classes': len(self.train_generator.class_indices)
        }
        
        logging.info(f"Class information: {self.class_info}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def load_callbacks(self):
        logging.info("Loading callbacks for model training...")
        callbacks = Callbacks(
            model_dir=self.config.model_direct,
            patience=5,
            monitor='val_loss',
            mode='min'
        )
        return callbacks.get_callbacks()
    
    def compute_class_weights(self):
        logging.info("Calculating class weights for imbalanced dataset...")
        class_weight_calculator = ClassWeightCalculator(
            train_classes=self.train_generator.classes
        )
        return class_weight_calculator.compute_class_weights()

    def train(self):
        logging.info("Starting model training...")
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        logging.info(f"Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")


        logging.info("Fitting the model...")
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=self.load_callbacks(),
            verbose=1,
            class_weight=self.compute_class_weights()
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        logging.info(f"Model trained and saved to {self.config.trained_model_path}")
        return self.class_info
