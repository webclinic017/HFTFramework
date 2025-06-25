import os
from functools import lru_cache


### decorator cache in memory max size =  how many recent return values to cache.
# memory_cache = lru_cache(maxsize=16192)
# cached_functions = []
# MAX_SIZE = 8192


def memory_cache(maxsize=128):
    def decorator(func):
        if os.environ.get("CACHE_DISABLED", False):
            return func
        func = lru_cache(maxsize=maxsize)(func)
        # cached_functions.append(func)
        return func

    return decorator
