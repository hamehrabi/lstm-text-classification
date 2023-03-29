import logging
import os

from app_entity.entity import DataPreprocessingEntity, DataIngestionEntity, ExperimentEntity
from app_configuration.configuration import AppConfiguration
import tensorflow as tf
from app_exception.exception import AppException
import sys
from app_entity.entity import TrainedModelEntity
from app_entity.config_entity import ModelTrainingConfig
from app_entity.entity import MetricInfoEntity
from app_utils.util import is_model_present

class ModelTrainer:

    def __init__(self, data_ingestion: DataIngestionEntity,
                 data_preprocessing: DataPreprocessingEntity,
                 experiment: ExperimentEntity,
                 app_config: AppConfiguration):
        try:

            self.data_ingestion = data_ingestion
            model_training_config = app_config.get_model_training_config(experiment_id=experiment.experiment_id)

            self.trained_model = TrainedModelEntity(experiment_id=experiment.experiment_id,
                                                    model_training_config=model_training_config
                                                    )
            self.trained_model.model_training_config = model_training_config
            self.data_preprocessing = data_preprocessing
            self.trained_model.status = True
            self.trained_model.message = f"{self.trained_model.message}\nModel trainer initialized"
        except Exception as e:
            if hasattr(self, "trained_model"):
                self.trained_model.status = False
                self.trained_model.message = f"{self.trained_model.message}\n{e}"
            raise AppException(e, sys) from e

    def load_trained_model_or_checkpoint_model(self):
        try:
            checkpoint_dir = self.trained_model.model_training_config.model_checkpoint_dir
            model_save_dir = self.trained_model.model_training_config.model_save_dir

            if is_model_present(model_save_dir):
                self.load_model(model_save_dir)
                if self.trained_model.model is not None:
                    self.trained_model.is_trained_model_loaded = True
            else:
                self.load_model(checkpoint_dir)
                if self.trained_model.model is not None:
                    self.trained_model.is_checkpoint_model_loaded = True

        except Exception as e:
            raise AppException(e, sys) from e

    def load_model(self, model_dir):
        try:
            if is_model_present(model_dir):
                model = tf.keras.models.load_model(model_dir)
                self.trained_model.model = model
                self.trained_model.message = f"{self.trained_model.message}\nModel loaded."
            else:
                self.trained_model.message = f"{self.trained_model.message}\nModel not loaded. " \
                                             f"Model dir[{model_dir} not exists.]"
        except Exception as e:
            raise AppException(e, sys) from e

    def get_model(self) -> TrainedModelEntity:
        try:
            self.load_trained_model_or_checkpoint_model()
            if not self.trained_model.is_trained_model_loaded and not self.trained_model.is_checkpoint_model_loaded:
                encoder = self.data_preprocessing.encoder
                self.trained_model.model_architecture = tf.keras.Sequential([
                    encoder,
                    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, )),

                    tf.keras.layers.Dense(64, activation='relu'),

                    tf.keras.layers.Dense(1)
                ])
                self.trained_model.model_architecture.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                                              optimizer=tf.keras.optimizers.Adam(1e-4),
                                                              metrics=['accuracy'])

                self.trained_model.model = self.trained_model.model_architecture
                self.trained_model.status = True
                self.trained_model.message = f"{self.trained_model.message}\nModel compiled successfully."
                self.trained_model.message = f"{self.trained_model.message}\nModel architecture created."
            self.trained_model.status = True

            return self.trained_model
        except Exception as e:
            self.trained_model.status = False
            self.trained_model.message = f"{self.trained_model.message}\n{e}"
            raise AppException(e, sys) from e

    def train_model(self) -> TrainedModelEntity:
        try:
            if self.trained_model.model is None:
                self.get_model()
            self.trained_model.message = f"{self.trained_model.message}\n Model training begin."

            checkpoint_callback_fn = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.trained_model.model_training_config.model_checkpoint_dir,
                save_best_only=True
            )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.trained_model.model_training_config.tensorboard_log_dir)

            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

            model_training_config = self.trained_model.model_training_config
            train_dataset = self.data_ingestion.train
            if not self.trained_model.is_trained_model_loaded:
                history = self.trained_model.model.fit(train_dataset,
                                                       epochs=model_training_config.epoch,
                                                       validation_data=self.data_ingestion.test,
                                                       validation_steps=model_training_config.validation_step,
                                                       callbacks=[stop_early,
                                                                  checkpoint_callback_fn,
                                                                  tensorboard_callback]
                                                       )
                self.trained_model.history = history
            test_dataset = self.data_ingestion.test
            train_loss, train_acc = self.trained_model.model.evaluate(train_dataset)
            test_loss, test_acc = self.trained_model.model.evaluate(test_dataset)
            metric_info = MetricInfoEntity(
                train_accuracy=train_acc,
                train_loss=train_loss,
                test_accuracy=test_acc,
                test_loss=test_loss
            )
            self.trained_model.metric_info = metric_info

            base_accuracy = self.trained_model.model_training_config.base_accuracy
            if base_accuracy >= test_acc:
                msg = f"Trained model accuracy: [{test_acc}] and model loss: [{test_loss}]."
                msg = f"{msg}\nAs trained model accuracy :  [{test_acc}] is less than base accuracy: [{base_accuracy}] " \
                      f"hence rejecting trained model. "
                logging.info(
                    msg
                )
                self.trained_model.message = f"{self.trained_model.message}\n{msg}"
                raise Exception(msg)
            self.trained_model.status = True
            self.trained_model.message = f"{self.trained_model.message}\nModel trained."
            return self.trained_model
        except Exception as e:
            self.trained_model.status = False
            self.trained_model.message = f"{self.trained_model.message}\n{e}"
            raise AppException(e, sys) from e

    def save_model(self) -> TrainedModelEntity:
        try:
            if self.trained_model.model is None:
                self.train_model()
            model_export_dir = self.trained_model.model_training_config.model_save_dir
            message = f"Saving model in dir: [{model_export_dir}]"
            logging.info(message)
            self.trained_model.message = f"{self.trained_model.message}\n{message}"
            self.trained_model.model.save(model_export_dir)
            message = f"Model saved in dir: [{model_export_dir}]"
            logging.info(message)
            self.trained_model.message = f"{self.trained_model.message}\n{message}"
            self.trained_model.status = True
            return self.trained_model
        except Exception as e:
            self.trained_model.status = False
            self.trained_model.message = f"{self.trained_model.message}\n{e}"
            raise AppException(e, sys) from e
