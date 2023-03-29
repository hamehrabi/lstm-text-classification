import json

import dill

from app_entity.entity import ExperimentEntity, TrainingPipelineEntity
from app_logger import logging, log_function_signature, EXPERIMENT_ID

from app_utils.util import write_json_file
import uuid
from datetime import datetime
from app_configuration.configuration import AppConfiguration
from app_exception import AppException
from src import DataLoader, DataPreprocessing, ModelTrainer, ModelEvaluation, ModelDeployment
import sys


class TrainingPipeline:

    @log_function_signature
    def __init__(self,
                 experiment_id=None,
                 experiment_name=None,
                 experiment_description=None,
                 execution_start_time_stamp=None,
                 executed_by_user=None,
                 executed_by_email=None,
                 execution_stop_time_stamp=None,
                 execution_status=None,
                 execution_description=None,
                 artifacts_dir=None
                 ):
        try:
            self.app_config = AppConfiguration()
            self.experiment = ExperimentEntity(
                experiment_id=EXPERIMENT_ID if experiment_id is None else experiment_id,
                experiment_name=experiment_name,
                config_info=self.app_config.config_info,
                experiment_description=experiment_description,
                execution_start_time_stamp=datetime.now(),
                executed_by_user=executed_by_user,
                executed_by_email=executed_by_email,
                execution_stop_time_stamp=execution_stop_time_stamp,
                execution_status=execution_status,
                execution_description=execution_description,
                artifacts_dir=artifacts_dir
            )
            training_pipeline_config = self.app_config.get_training_pipeline_config(
                experiment_id=self.experiment.experiment_id)
            self.training_pipeline = TrainingPipelineEntity(data_ingestion=None,
                                                            data_preprocessing=None,
                                                            model_trainer=None,
                                                            training_pipeline_config=training_pipeline_config
                                                            )

            self.training_pipeline.status = True
            self.training_pipeline.message = f"{self.training_pipeline.message}\n" \
                                             f"Training pipeline initialized"
        except Exception as e:
            self.training_pipeline.status = False
            self.training_pipeline.message = f"{self.training_pipeline.message}\n{e}"
            raise AppException(e, sys) from e

    @log_function_signature
    def start_training(self) -> TrainingPipelineEntity:
        """
        This function return an object of TrainingPipelineEntity

        """
        try:
            data_loader = DataLoader(experiment=self.experiment, app_config=self.app_config)

            data_ingestion = data_loader.get_batch_shuffle_dataset()
            self.training_pipeline.data_ingestion = data_ingestion
            data_preprocessor = DataPreprocessing(experiment=self.experiment,
                                                  app_config=self.app_config,
                                                  data_ingestion=data_ingestion
                                                  )
            data_preprocessing = data_preprocessor.get_text_encoder()
            self.training_pipeline.data_preprocessing = data_preprocessing
            model_trainer = ModelTrainer(data_ingestion=data_ingestion,
                                         data_preprocessing=data_preprocessing,
                                         experiment=self.experiment,
                                         app_config=self.app_config
                                         )

            self.training_pipeline.model_trainer = model_trainer.save_model()

            training_pipeline_config = self.training_pipeline.training_pipeline_config
            report_file_path = training_pipeline_config.execution_report_file_path
            model_evaluator = ModelEvaluation(
                execution_report_file_path=report_file_path,
                trained_model=self.training_pipeline.model_trainer,
                experiment=self.experiment,
                app_config=self.app_config,

            )
            model_eval_obj = model_evaluator.is_trained_model_acceptable()
            self.training_pipeline.model_evaluator = model_eval_obj
            model_deployment = ModelDeployment(
                experiment_id=self.experiment.experiment_id,
                app_config=self.app_config,
                model_eval_obj=model_eval_obj
            )
            if model_eval_obj.is_trained_model_accepted:
                model_deployer = model_deployment.export_model()
                self.training_pipeline.model_deployment = model_deployer
                # write code to deploy your model
            else:
                message = "Trained model is not acceptable hence stopping pipeline."
                logging.info(message)
                self.training_pipeline.message = f"{self.training_pipeline.message}\n{message}"

            self.training_pipeline.model_deployment = model_deployment.model_deployment_obj
            self.experiment.execution_stop_time_stamp = datetime.now()
            self.experiment.execution_status = True
            self.training_pipeline.status = True
            self.training_pipeline.message = f"{self.training_pipeline.message}\nModel Created."
            message = "Started saving training pipeline execution report."
            logging.info(message)
            self.training_pipeline.message = f"{self.training_pipeline.message}\n{message}"

            self.save_execution_report()
            return self.training_pipeline
        except Exception as e:
            raise AppException(e, sys) from e

    def save_execution_report(self):
        try:

            training_pipeline_config = self.training_pipeline.training_pipeline_config
            report_file_path = training_pipeline_config.execution_report_file_path

            data_ingestion = self.training_pipeline.data_ingestion
            data_preprocessing = self.training_pipeline.data_preprocessing
            model_training = self.training_pipeline.model_trainer

            execution_time_in_ms = self.experiment.execution_stop_time_stamp - self.experiment.execution_start_time_stamp

            model_eval = self.training_pipeline.model_evaluator
            model_deployment = self.training_pipeline.model_deployment

            report = {
                self.experiment.execution_start_time_stamp.strftime('%Y-%m-%d-%H-%M-%S'):
                {
                    "experiment_id": self.experiment.experiment_id,
                    "execution_start_time": self.experiment.execution_start_time_stamp.strftime(
                        '%Y-%m-%d-%H-%M-%S'),
                    "execution_stop_time": self.experiment.execution_stop_time_stamp.strftime('%Y-%m-%d-%H-%M-%S'),
                    "execution_time_in_ms": execution_time_in_ms.microseconds,
                    "config_information": self.experiment.config_info,
                    "executed_by_user": self.experiment.executed_by_user,
                    "executed_by_email_id": self.experiment.executed_by_email,
                    "execution_status": self.experiment.execution_status,
                    "data_ingestion": {
                        "status": data_ingestion.status,
                        "message": self.training_pipeline.message,
                        "data_config": data_ingestion.dataset_config._asdict()
                    },
                    "data_preprocessing": {
                        "status": data_preprocessing.status,
                        "message": data_preprocessing.message,
                        "preprocessing_config": data_preprocessing.preprocessing_config._asdict()
                    },
                    "model_training": {
                        "status": model_training.status,
                        "message": model_training.message,

                        "is_model_trained": True if model_training.model is not None else False,
                        "is_model_architecture_created": True if model_training.model_architecture is not None else False,
                        "model_training_accuracy": model_training.metric_info.train_accuracy,
                        "model_testing_accuracy": model_training.metric_info.test_accuracy,
                        "model_training_loss": model_training.metric_info.train_loss,
                        "model_testing_loss": model_training.metric_info.test_loss,
                        "training_config": model_training.model_training_config._asdict()

                    },
                    "model_evaluation": {
                        "status": model_eval.status,
                        "message": model_eval.message,
                        "is_trained_model_accepted": model_eval.is_trained_model_accepted,
                        "model_eval_config": model_eval.model_eval_config,
                        "improved_factor": model_eval.improved_factor,
                        "compared_model_path": model_eval.best_model_path,

                    },
                    "model_deployment":
                        {
                            "status": model_deployment.status,
                            "message": model_deployment.message,
                            "model_deployment_config": model_deployment.model_deployment_config,
                            "model_serving_dir": model_deployment.export_dir

                        }
                }
            }

            write_json_file(obj=report, file_path=report_file_path)
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.start_training()
    except Exception as e:
        logging.info(e)
        print(e)
