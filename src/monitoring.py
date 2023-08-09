import timeit
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def execution_time_runtime(func: Callable) -> Callable:
    """
    Usage: 
    -> import this function in script
    -> use as decorator for arbitrary function/method in order to time it
    """
    def wrapper_function(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        name = func.__name__
        stop = timeit.default_timer()
        execution_time = stop-start
        logger.info(f"execution time of {name} is {execution_time} seconds")
        return result
    return wrapper_function
