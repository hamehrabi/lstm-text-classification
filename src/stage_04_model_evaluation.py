from collections import namedtuple
from app_logger import logging, log_function_signature
from app_utils.util import read_json_file, is_model_present
from app_exception import AppException
import sys
from app_configuration.configuration import AppConfiguration
from app_entity.entity import ModelEvaluationEntity, ExperimentEntity, MetricInfoEntity, TrainedModelEntity, \
    BestModelEntity
import tensorflow as tf


class ModelEvaluation:

    def __init__(self,
                 execution_report_file_path: str,
                 trained_model: TrainedModelEntity,
                 experiment: ExperimentEntity,
                 app_config: AppConfiguration,
                 best_model: BestModelEntity = None):
        try:
            self.execution_report_file_path = execution_report_file_path
            self.best_model = best_model
            self.trained_model = trained_model

            model_eval_config = app_config.get_model_evaluation_config()
            self.model_eval_entity = ModelEvaluationEntity(
                experiment_id=experiment.experiment_id,
                model_eval_config=model_eval_config,
                trained_model=trained_model
            )
            self.model_eval_entity.status = True
            self.model_eval_entity.message = f"Model Evaluation object has been initialized."
        except Exception as e:
            raise AppException(e, sys) from e

    def get_best_model(self, ):
        try:
            execution_report = read_json_file(
                file_path=self.execution_report_file_path)
            best_execution_start_time = None
            best_score = 0
            for execution_start_time, execution_data in execution_report.items():
                current_execution_score = execution_data["model_training"]["model_testing_accuracy"]
                if current_execution_score > best_score:
                    best_score = current_execution_score
                    best_execution_start_time = execution_start_time

            if best_execution_start_time is not None:
                model_path = execution_report[best_execution_start_time]["model_training"]["training_config"][
                    "model_save_dir"]
                test_accuracy = execution_report[best_execution_start_time]["model_training"]["model_testing_accuracy"]
                train_accuracy = execution_report[best_execution_start_time]["model_training"][
                    "model_training_accuracy"]
                test_loss = execution_report[best_execution_start_time]["model_training"]["model_testing_loss"]
                train_loss = execution_report[best_execution_start_time]["model_training"]["model_training_loss"]
                metric_info = MetricInfoEntity(train_accuracy=train_accuracy,
                                               test_accuracy=test_accuracy,
                                               train_loss=train_loss,
                                               test_loss=test_loss)

                if is_model_present(model_path):
                    best_model = BestModelEntity(best_model=tf.keras.models.load_model(model_path),
                                                 model_path=model_path,
                                                 is_best_model_exists=True,
                                                 metric_info=metric_info
                                                 )
                    self.model_eval_entity.best_model_obj = best_model
                    self.model_eval_entity.status = True
                    message = f"Best model has been loaded from dir: [{model_path}] "
                    self.model_eval_entity.message = f"{self.model_eval_entity.message}\n{message}"
                    logging.info(message)
                else:
                    self.model_eval_entity.best_model = BestModelEntity(
                        best_model=None,
                        model_path=model_path,
                        is_best_model_exists=False
                    )
                    message = f"Best model dir: [{model_path}] is not available"
                    self.model_eval_entity.message = f"{self.model_eval_entity.message}\n{message}"
                    logging.info(message)
            return self.model_eval_entity
        except Exception as e:
            raise AppException(e, sys) from e

    def is_trained_model_acceptable(self) -> ModelEvaluationEntity:
        try:
            if self.model_eval_entity.best_model_obj.best_model is None:
                self.get_best_model()
            train_model_metrics = self.trained_model.metric_info
            best_model_metrics = self.model_eval_entity.best_model_obj.metric_info
            message = f"Training model metric: {train_model_metrics}"
            logging.info(message)
            self.model_eval_entity.message = f"{self.model_eval_entity.message}\n{message}"
            message = f"Best model metric: {best_model_metrics}"
            logging.info(message)
            self.model_eval_entity.message = f"{self.model_eval_entity.message}\n{message}"

            if best_model_metrics is None:
                message = "We have not found any of the existing model to compare with trained model"
                logging.info(message)
                self.model_eval_entity.message = f"{self.model_eval_entity.message}\n{message}"
                self.model_eval_entity.is_trained_model_accepted = True

                return self.model_eval_entity
            best_acc = best_model_metrics.test_accuracy
            trained_acc = train_model_metrics.test_accuracy
            if trained_acc >= best_acc:
                change_threshold = (trained_acc - best_acc) / best_acc
                self.model_eval_entity.improved_factor = change_threshold
                message = f"Trained model accuracy is better than best model by [{change_threshold}] factor."

                logging.info(message)
                self.model_eval_entity.message = f"{self.model_eval_entity.message}\n{message}"
                if self.model_eval_entity.model_eval_config.change_threshold <= change_threshold:
                    self.model_eval_entity.is_trained_model_accepted = True
                    message = f"Trained model accepted as it improved as compare to best model by" \
                              f" [{change_threshold}] factor."
                    logging.info(message)
                    self.model_eval_entity.message = f"{self.model_eval_entity.message}\n{message}"

            return self.model_eval_entity
        except Exception as e:
            raise AppException(e, sys) from e
