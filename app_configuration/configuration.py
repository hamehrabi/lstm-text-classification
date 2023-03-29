import os, sys
from app_exception.exception import AppException
from app_utils.util import read_yaml_file
from collections import namedtuple
from app_logger import logging, log_function_signature
from app_entity.config_entity import DatasetConfig, PreprocessingConfig, TrainingPipelineConfig, ModelEvaluationConfig
from app_entity.config_entity import ModelTrainingConfig, ModelDeploymentConfig

# Initializing the default values for app configuration
ROOT_DIR = os.getcwd()
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_FILE_NAME)

SCHEMA_FILE_NAME = "dataset_schema.yaml"
SCHEMA_FILE_PATH = os.path.join(ROOT_DIR, SCHEMA_FILE_NAME)

# dataset keys
DATA_SET_KEY = "data_set"
DATA_SET_NAME_KEY = "name"
SCHEMA_KEY = "schema"
BUFFER_SIZE_KEY = "buffer_size"
BATCH_SIZE_KEY = "batch_size"


# preprocessing keys
PREPROCESSING_KEY = "preprocessing"
VOCAB_SIZE_KEY = "vocab_size"


# training configuration keys
TRAINING_CONFIG_KEY = "train_config"
TRAINING_MODEL_ROOT_DIR_KEY = "model_root_dir"
TRAINING_MODEL_SAVE_DIR_KEY = "model_save_dir"
TRAINING_MODEL_CHECKPOINT_DIR_KEY = "model_checkpoint_dir"
TRAINING_MODEL_EPOCH_KEY = "epoch"
TRAINING_MODEL_TENSORBOARD_LOG_DIR_KEY = "tensorboard_log_dir"
TRAINING_MODEL_BASE_ACCURACY_KEY = "base_accuracy"
TRAINING_MODEL_VALIDATION_STEP_KEY = "validation_step"


# Training pipeline config
TRAINING_PIPELINE_KEY = "training_pipeline_config"
ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_OBJ_DIR_KEY = "training_pipeline_obj_dir"
TRAINING_PIPELINE_OBJ_FILE_NAME_KAY = "training_pipeline_obj_file_name"
TRAINING_PIPELINE_EXECUTION_REPORT_DIR_KEY = "execution_report_dir"
TRAINING_PIPELINE_EXECUTION_REPORT_FILE_NAME_KEY = "execution_report_file_name"


# Model evaluation config
MODEL_EVALUATION_KEY = "model_eval_config"
MODEL_EVALUATION_CHANGE_THRESHOLD_CONFIG_KEY = "change_threshold"


# Model deployment config
MODEL_DEPLOYMENT_KEY = "model_deployment"
MODEL_DEPLOYMENT_SERVING_DIR_KEY = "model_serving_dir"


class AppConfiguration:
    """
    Reads the configuration file and returns the configuration object
    """

    @log_function_signature
    def __init__(self, ):
        try:

            logging.info("Reading the configuration file.")
            self.config_info = read_yaml_file(yaml_file_path=CONFIG_FILE_PATH)
            self.dataset_schema = read_yaml_file(yaml_file_path=SCHEMA_FILE_PATH)
            logging.info("Configuration file read successfully.")
        except Exception as e:
            raise AppException(e, sys) from e

    @log_function_signature
    def get_dataset_configuration(self) -> DatasetConfig:
        try:
            dataset_config = self.config_info[DATA_SET_KEY]
            logging.info(f"Dataset configuration :\n{dataset_config}\n read successfully.")
            response = DatasetConfig(name=dataset_config[DATA_SET_NAME_KEY],
                                     schema=self.dataset_schema[SCHEMA_KEY],
                                     batch_size=dataset_config[BATCH_SIZE_KEY],
                                     buffer_size=dataset_config[BUFFER_SIZE_KEY]
                                     )
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    @log_function_signature
    def get_preprocessing_configuration(self) -> PreprocessingConfig:
        try:
            preprocessing_config = self.config_info[PREPROCESSING_KEY]
            logging.info(f"Preprocessing configuration :\n{preprocessing_config}\n read successfully.")
            response = PreprocessingConfig(vocal_size=preprocessing_config[VOCAB_SIZE_KEY])
            logging.info(f"Preprocessing config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    @log_function_signature
    def get_model_training_config(self, experiment_id) -> ModelTrainingConfig:
        try:
            training_pipeline_config = self.get_training_pipeline_config(experiment_id=experiment_id)
            artifact_dir = training_pipeline_config.artifact_dir
            training_config = self.config_info[TRAINING_CONFIG_KEY]
            model_root_dir = os.path.join(artifact_dir, training_config[TRAINING_MODEL_ROOT_DIR_KEY])
            model_save_dir = os.path.join(model_root_dir, training_config[TRAINING_MODEL_SAVE_DIR_KEY])
            model_checkpoint_dir = os.path.join(model_root_dir, training_config[TRAINING_MODEL_CHECKPOINT_DIR_KEY])
            base_accuracy = training_config[TRAINING_MODEL_BASE_ACCURACY_KEY]
            tensorboard_log_dir = os.path.join(model_root_dir, training_config[TRAINING_MODEL_TENSORBOARD_LOG_DIR_KEY])
            epoch = training_config[TRAINING_MODEL_EPOCH_KEY]
            validation_step = training_config[TRAINING_MODEL_VALIDATION_STEP_KEY]

            # creating_directory

            dir_list = [model_save_dir, model_checkpoint_dir, tensorboard_log_dir, model_checkpoint_dir]
            for dir_name in dir_list:
                os.makedirs(dir_name, exist_ok=True)

            response = ModelTrainingConfig(model_save_dir=model_save_dir,
                                           model_root_dir=model_root_dir,
                                           model_checkpoint_dir=model_checkpoint_dir,
                                           base_accuracy=base_accuracy,
                                           tensorboard_log_dir=tensorboard_log_dir,
                                           epoch=epoch,
                                           validation_step=validation_step
                                           )
            logging.info(f"Model training config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys)

    @log_function_signature
    def get_training_pipeline_config(self, experiment_id) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_KEY]
            logging.info(f"Preprocessing configuration :\n{training_pipeline_config}\n read successfully.")
            artifact_dir = os.path.join(ROOT_DIR, training_pipeline_config[ARTIFACT_DIR_KEY], experiment_id)
            logging.info(f"Training pipeline artifact dir: {artifact_dir}")

            training_pipeline_obj_dir = os.path.join(artifact_dir,
                                                     training_pipeline_config[TRAINING_PIPELINE_OBJ_DIR_KEY])

            execution_report_dir = os.path.join(training_pipeline_config[TRAINING_PIPELINE_EXECUTION_REPORT_DIR_KEY])

            dir_to_create = [artifact_dir, training_pipeline_obj_dir, execution_report_dir]
            for dir_name in dir_to_create:
                os.makedirs(dir_name, exist_ok=True)

            execution_report_file_name = training_pipeline_config[TRAINING_PIPELINE_EXECUTION_REPORT_FILE_NAME_KEY]

            execution_report_file_path = os.path.join(ROOT_DIR, execution_report_dir, execution_report_file_name)

            training_pipeline_obj_file_name = training_pipeline_config[TRAINING_PIPELINE_OBJ_FILE_NAME_KAY]
            training_pipeline_obj_file_path = os.path.join(training_pipeline_obj_dir,
                                                           training_pipeline_obj_file_name)

            response = TrainingPipelineConfig(artifact_dir=artifact_dir,
                                              training_pipeline_obj_dir=training_pipeline_obj_dir,
                                              training_pipeline_obj_file_name=training_pipeline_obj_file_name,
                                              training_pipeline_obj_file_path=training_pipeline_obj_file_path,
                                              execution_report_dir=execution_report_dir,
                                              execution_report_file_name=execution_report_file_name,
                                              execution_report_file_path=execution_report_file_path
                                              )
            logging.info(f"Training pipeline config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    @log_function_signature
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_eval_config = self.config_info[MODEL_EVALUATION_KEY]
            response = ModelEvaluationConfig(
                change_threshold=model_eval_config[MODEL_EVALUATION_CHANGE_THRESHOLD_CONFIG_KEY])
            logging.info(f"Model evaluation config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    @log_function_signature
    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        try:

            model_deployment_config = self.config_info[MODEL_DEPLOYMENT_KEY]
            serving_dir = os.path.join(ROOT_DIR, model_deployment_config[MODEL_DEPLOYMENT_SERVING_DIR_KEY])
            os.makedirs(serving_dir, exist_ok=True)
            response = ModelDeploymentConfig(
                model_serving_dir=serving_dir)
            logging.info(f"Model evaluation config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def __repr__(self) -> str:
        return f"AppConfiguration()"

    def __str__(self) -> str:
        return f"AppConfiguration()"
