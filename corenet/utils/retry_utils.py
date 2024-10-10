#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import random
import time
from typing import Any, Callable, Dict, List, Optional

from corenet.utils import logger


def run_with_retries(
    fn: Callable,
    max_retries: int,
    args: Optional[List] = None,
    kwargs: Optional[Dict] = None,
    function_name: Optional[str] = None,
) -> Any:
    """Runs a function with retries (using exponential backoff method) on failure.

    Args:
        fn: Function to run.
        max_retries: Maximum number of attempts before giving up.
        args: Args to be passed to the @fn as *args. Defaults to None that translates
            to empty args.
        kwargs: Kwargs to be passed to the @fn as **kwargs. Defaults to None that
            translates to empty kwargs.
        function_name: A label for the task that @fn performs, to be used in warning
            and error messages. Defaults to None that translates to "run {fn.__name__}".

    Returns:
        The value the @fn returns.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if function_name is None:
        function_name = f"run {fn.__name__}"

    for attempt in range(max_retries - 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait_duration = 2**attempt * random.uniform(0.5, 1.0)
            logger.warning(
                f"Failed to {function_name} at attempt {attempt+1}/"
                f"{max_retries} with error {e}; retrying in {wait_duration} seconds."
            )
            time.sleep(wait_duration)
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Unable to {function_name} after {max_retries} attempts with error {e}."
        ) from e
