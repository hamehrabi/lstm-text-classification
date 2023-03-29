from app_test.training_test import test_training_pipeline

from app_logger import logging

if __name__ == "__main__":
    try:
        test_training_pipeline()
    except Exception as e:
        logging.error(e)
        print(e)
