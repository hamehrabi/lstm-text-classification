from app_logger import logging
from app_configuration.configuration import AppConfiguration
from app_exception.exception import AppException
from app_entity.entity import ModelDeploymentEntity, ModelEvaluationEntity
import os, sys
from datetime import datetime
import tensorflow as tf


class ModelDeployment:

    def __init__(self,
                 experiment_id: str,
                 app_config: AppConfiguration,
                 model_eval_obj: ModelEvaluationEntity
                 ):
        """
        experiment_id: str,
        app_config: AppConfiguration,
        model_eval_obj: ModelEvaluationEntity

        """
        try:
            self.model_eval_obj = model_eval_obj
            model_deployment_config = app_config.get_model_deployment_config()
            self.model_deployment_obj = ModelDeploymentEntity(
                experiment_id=experiment_id,
                model_deployment_config=model_deployment_config,
                accepted_model=model_eval_obj.trained_model,
            )
            self.model_deployment_obj.status = True
            self.model_deployment_obj.message = f"Model deployment object initialized."
        except Exception as e:
            raise AppException(e, sys) from e

    def export_model(self) -> ModelDeploymentEntity:
        try:
            serving_dir = self.model_deployment_obj.model_deployment_config.model_serving_dir
            current_time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            serving_dir = os.path.join(serving_dir, current_time_stamp)
            os.makedirs(serving_dir, exist_ok=True)

            self.model_deployment_obj.export_dir = serving_dir
            message = f"Exporting model in dir: [{serving_dir}]"
            logging.info(message)
            self.model_deployment_obj.message = f"{self.model_deployment_obj.message}\n{message}"
            self.model_deployment_obj.accepted_model.model.save(serving_dir)

            message = f"Exported model in dir: [{serving_dir}]"
            logging.info(message)
            self.model_deployment_obj.message = f"{self.model_deployment_obj.message}\n{message}"
            return self.model_deployment_obj
        except Exception as e:
            raise AppException(e, sys) from e
