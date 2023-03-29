from app_exception.exception import AppException
from app_configuration.configuration import AppConfiguration
from app_logger.logger import logging, log_function_signature
from app_entity.entity import DataIngestionEntity, ExperimentEntity
import os, sys
import tensorflow_datasets as tfds
from collections import namedtuple
import tensorflow as tf

TRAIN_KEY = "train"
TEST_KEY = "test"


class DataLoader:

    @log_function_signature
    def __init__(self, experiment: ExperimentEntity, app_config: AppConfiguration):
        try:
            logging.info("Reading the dataset configuration.")
            data_set_config = app_config.get_dataset_configuration()
            logging.info(f"Dataset configuration :\n{data_set_config}\n read successfully")
            self.data_ingestion = DataIngestionEntity(experiment_id=experiment.experiment_id,
                                                      train=None,
                                                      test=None,
                                                      dataset_config=data_set_config
                                                      )
            self.data_ingestion.status = True
            self.data_ingestion.message = "Data Ingestion is initialized."
        except Exception as e:
            self.data_ingestion.message = f"{self.data_ingestion.message}\n{e}"
            self.data_ingestion.status = False
            raise AppException(e, sys) from e

    @log_function_signature
    def get_dataset(self) -> DataIngestionEntity:
        try:
            logging.info("Reading the dataset")

            dataset, dataset_info = tfds.load(self.data_ingestion.dataset_config.name, with_info=True,
                                              as_supervised=True)

            train_dataset, test_dataset = dataset[TRAIN_KEY], dataset[TEST_KEY]
            logging.info("Dataset read successfully")
            self.data_ingestion.train = train_dataset
            self.data_ingestion.test = test_dataset
            self.data_ingestion.message = f"{self.data_ingestion.message}\nData has been loaded from tensorflow " \
                                          f"dataset library {dataset_info} "
            self.data_ingestion.is_dataset_present=True
            self.data_ingestion.status = True
            return self.data_ingestion
        except Exception as e:
            self.data_ingestion.message = f"{self.data_ingestion.message}\n{e}"
            self.data_ingestion.status = False
            raise AppException(e, sys) from e

    @log_function_signature
    def get_batch_shuffle_dataset(self):
        try:
            if self.data_ingestion.is_dataset_present is None:
                logging.info("Dataset is available hence started loading dataset.")
                self.get_dataset()
            buffer_size = self.data_ingestion.dataset_config.buffer_size
            batch_size = self.data_ingestion.dataset_config.batch_size
            self.data_ingestion.train = self.data_ingestion.train.shuffle(buffer_size).batch(batch_size).prefetch(
                tf.data.AUTOTUNE)
            self.data_ingestion.test = self.data_ingestion.test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            logging.info("Dataset shuffled and batched successfully")
            self.data_ingestion.message = f"{self.data_ingestion.message}\nData shuffling compeleted with " \
                                          f"batch_size:{batch_size} and buffer_size:{buffer_size}"
            self.data_ingestion.status = True
            return self.data_ingestion
        except Exception as e:
            self.data_ingestion.message = f"{self.data_ingestion.message}\n{e}"
            self.data_ingestion.status = False
            logging.info(self.data_ingestion.message)
            raise AppException(e, sys) from e

    def __repr__(self):
        return f"DataLoader()"

    def __str__(self):
        return f"DataLoader()"
