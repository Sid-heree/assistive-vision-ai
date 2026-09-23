import time
from contextlib import contextmanager


@contextmanager
def timer():
    """
    Usage:
        with timer() as t:
            do_something()
        print(t["ms"])   # milliseconds elapsed, available AFTER the block
    """
    start = time.perf_counter()
    result = {"ms": 0.0}
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000