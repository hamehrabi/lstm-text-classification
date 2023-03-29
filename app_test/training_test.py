from pipeline.training_pipeline import TrainingPipeline
from app_exception.exception import AppException

import sys


def test_training_pipeline():
    try:
        TrainingPipeline().start_training()
    except Exception as e:
        raise AppException(e, sys) from e
