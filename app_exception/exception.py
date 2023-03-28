import os
import sys
from app_logger import log_function_signature


@log_function_signature
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred python script name [{file_name}]" \
                    f" line number [{exc_tb.tb_lineno}] error message [{error}] "

    return error_message


class AppException(Exception):


    def __init__(self, error_message: Exception, error_detail: sys):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __repr__(self):
        return AppException.__name__.__str__()

    def __str__(self):
        return self.error_message
