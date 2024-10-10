#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import inspect
import sys
import types
from typing import Any, Union


def check(
    value: Any, on_failure: Union[str, Exception, types.FunctionType] = "Check failed"
) -> Any:
    """
    Checks if value is truthy and raises an exception if not.

    This is a replacement for assert, with the following advantages:
     - Cannot be disabled by the -O flag
     - Can raise any exception type
     - Returns the checked value for concise code

    on_failure can be:
     - A string, in which case a AssertionError is raised with that message.
     - A constructed exception to be raised.
     - A lambda returning any of the above, so that the message/exception
         doesn't need to be constructed if the check succeeds. If the lambda
         takes an argument it will be the value.
    """
    if value:
        return value

    if isinstance(on_failure, types.FunctionType):
        nparams = len(inspect.signature(on_failure).parameters)

        if nparams == 0:
            on_failure = on_failure()
        elif nparams == 1:
            on_failure = on_failure(value)
        else:
            raise ValueError("Expect at most 1 element lambda")

    if not isinstance(on_failure, Exception):
        on_failure = AssertionError(str(on_failure))

    # This used to pop the call stack from the exception traceback,
    # so that it would appear to come from the check() call itself,
    # but that seems to no longer work in python3.10
    check_failed = on_failure

    raise check_failed
