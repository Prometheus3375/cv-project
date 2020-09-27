import traceback
from functools import wraps
from time import sleep
from typing import Any, Callable


def except_errors(*, sleep_time: float = 1.) -> Callable[..., Any]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def temp(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                sleep(sleep_time)
                traces = ''.join(traceback.format_tb(e.__traceback__))
                print(f'Traceback (most recent call last):\n{traces}{e.__class__.__name__}: {e}')

        return temp

    return wrapper
