import tensorflow as tf
from pathlib import Path
from src.entity.config_entity import EvaluationConfig
from src.utils.common import save_json
from src.logger import logging


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",  # Use "binary" for binary classification, "categorical" for multi-class
            color_mode="grayscale",  # Or "rgb" depending on your data
            shuffle=False
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            **dataflow_kwargs
        )

        logging.info("Validation generator created successfully.")

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        model = tf.keras.models.load_model(path)
        logging.info(f"Model loaded from: {path}")
        return model

    def evaluation(self):
        logging.info("Starting evaluation...")
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        logging.info(f"Evaluation completed. Loss: {self.score[0]}, Accuracy: {self.score[1]}")
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
        logging.info("Evaluation scores saved to scores.json.")
