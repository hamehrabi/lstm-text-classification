import logging
import os
from datetime import datetime
import uuid


EXPERIMENT_ID = str(uuid.uuid4())
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
FILE_NAME = f"log_{EXPERIMENT_ID}.log"


LOG_DIR = "logs"
LOG_DIR = os.path.join(os.getcwd(), LOG_DIR,CURRENT_TIME_STAMP)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w',
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def log_function_signature(func):
    def inner(*args, **kwargs):
        kw_args_text = ""
        for param_name, param_value in kwargs.items():
            kw_args_text = f"{kw_args_text},{param_name}= {param_value}"
        arg_text = list(args)
        arg_text = ",".join(map(str, arg_text))
        logging.info(f"Entering {func.__name__}({arg_text}{kw_args_text})")
        response = func(*args, **kwargs)
        logging.info(f"Exiting {func.__name__}({arg_text}{kw_args_text})")
        return response

    return inner

# @log_function_signature
# def print_text(msg):
#     print(msg)
#
# if __name__=="__main__":
#     print_text("Taha")