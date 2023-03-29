import logging
import os
from datetime import datetime
from app_exception import AppException
import uuid
import sys
import graphviz

parent_func_name = []

LOG_DIR = "logs"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

# experiment_id = "50a23d86-000b-4f7f-82d3-4d7044ed6e36"
experiment_id = str(uuid.uuid4())
EXPERIMENT_ID = experiment_id

file_name = f"log_{EXPERIMENT_ID}.log"

LOG_DIR = os.path.join(os.getcwd(), LOG_DIR, CURRENT_TIME_STAMP)
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, file_name)
logging.basicConfig(filename=log_file_path,
                    filemode='w',
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


def log_function_signature(func):
    def inner(*args, **kwargs):
        global parent_func_name
        global logging
        kw_args_text = ""
        arg_text = list(args)
        arg_text = ",".join(map(str, arg_text))
        for param_name, param_value in kwargs.items():
            kw_args_text = f"{kw_args_text},{param_name}= {param_value}"
        function_signature = f"{func.__name__}({arg_text}{kw_args_text})"
        logging.info(f"Entering {function_signature}")
        try:
            execution_start_timestamp = datetime.now()
            response = func(*args, **kwargs)
            execution_stop_timestamp = datetime.now()
            diff = execution_stop_timestamp - execution_start_timestamp
            logging.info(f"{func.__name__}() exec time {diff.microseconds} ms)")
            parent_func_name.append(f"{func.__name__}() exec time {diff.microseconds} ms")
        except Exception as e:
            raise e
        logging.info(f"Exiting {func.__name__}({arg_text}{kw_args_text})")
        return response

    return inner


def generate_graph():
    global parent_func_name
    g = graphviz.Digraph('G', filename=os.path.join(LOG_DIR, 'chart.gv'))
    for index in range(1, len(parent_func_name)):
        g.edge(parent_func_name[index], parent_func_name[index - 1])
    g.view()
    input("Press any key to exit")
