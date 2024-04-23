#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from contextlib import contextmanager
from typing import ContextManager


@contextmanager
def context_env_vars(**env: str) -> ContextManager[None]:
    """
    Temporarily sets the environment variables within its context.

    Example usage:
    ```
    os.environ["X"] = 2
    with context_env_vars(X=3):
        print(os.environ["X"])  # prints 3
    print(os.environ["X"])  # prints 2
    ```
    """
    original_values = {}
    try:
        for key, value in env.items():
            original_values[key] = env.get(key, None)
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value


@contextmanager
def context_tensor_threads(num_cpu_threads: int) -> ContextManager[None]:
    """
    Temporarily, instructs numpy and torch to use @n cpu threads for processing tensors
    and arrays within the context.
    """
    num_cpu_threads = str(num_cpu_threads)
    with context_env_vars(
        MKL_NUM_THREADS=num_cpu_threads,
        OMP_NUM_THREADS=num_cpu_threads,
        NUMEXPR_NUM_THREADS=num_cpu_threads,
    ):
        yield
