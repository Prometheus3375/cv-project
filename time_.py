from functools import wraps
from timeit import default_timer as get_time
from typing import Any, Callable, Tuple


def seconds2time(seconds: int) -> Tuple[int, int, int]:
    h = seconds // 3600
    m = (seconds - h * 3600) // 60
    s = seconds - h * 3600 - m * 60
    return h, m, s


def print_time_elapsed(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def temp(*args, **kwargs) -> Any:
        start = get_time()
        try:
            result = func(*args, **kwargs)
        except KeyboardInterrupt:
            result = None
        end = get_time()
        h, m, s = seconds2time(round(end - start))
        print(f'Time elapsed: {h}h {m}m {s}s')
        return result

    return temp
