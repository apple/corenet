#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import sys
import time
import traceback
from typing import Optional, Union

text_colors = {
    "logs": "\033[34m",  # 033 is the escape code and 34 is the color code
    "info": "\033[32m",
    "warning": "\033[33m",
    "debug": "\033[93m",
    "error": "\033[31m",
    "bold": "\033[1m",
    "end_color": "\033[0m",
    "light_red": "\033[36m",
}


def get_curr_time_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def error(message: str) -> None:
    time_stamp = get_curr_time_stamp()
    error_str = (
        text_colors["error"]
        + text_colors["bold"]
        + "ERROR  "
        + text_colors["end_color"]
    )

    # exiting with code -1 does not tell any information about the error (e.g., NaN encountered in the loss).
    # For more descriptive error messages, we replace exit(-1) with sys.exit(ERROR_MESSAGE).
    # This allows us to handle specific exceptions in the tests.

    # print("{} - {} - {}".format(time_stamp, error_str, message), flush=True)
    # print("{} - {} - {}".format(time_stamp, error_str, "Exiting!!!"), flush=True)
    # exit(-1)

    if sys.exc_info()[0] is None:
        traceback.print_stack()
    else:
        traceback.print_exc()
    sys.exit("{} - {} - {}. Exiting!!!".format(time_stamp, error_str, message))


def color_text(in_text: str) -> str:
    return text_colors["light_red"] + in_text + text_colors["end_color"]


def log(message: str, end="\n") -> None:
    time_stamp = get_curr_time_stamp()
    log_str = (
        text_colors["logs"] + text_colors["bold"] + "LOGS   " + text_colors["end_color"]
    )
    print("{} - {} - {}".format(time_stamp, log_str, message), end=end)


def warning(message: Union[str, Warning]) -> None:
    if isinstance(message, Warning):
        message = f"{type(message).__name__}({','.join(map(repr, message.args))}"

    time_stamp = get_curr_time_stamp()
    warn_str = (
        text_colors["warning"]
        + text_colors["bold"]
        + "WARNING"
        + text_colors["end_color"]
    )
    print("{} - {} - {}".format(time_stamp, warn_str, message))


def ignore_exception_with_warning(message: str) -> None:
    """
    After catching a tolerable exception E1 (e.g. when Model.forward() fails during
    profiling with try-catch, it'll be helpful to log the exception for future
    investigation. But printing the error stack trace, as is, could be confusing
    when an uncaught (non-tolerable) exception "E2" raises down the road. Then, the log
    will contain two stack traces for E1, E2. When looking for errors in logs, users
    should look for E2, but they may find E1.

    This function appends "(WARNING)" at the end of all lines of the E1 traceback, so
    that the user can distinguish E1 from uncaught exception E2.

    Args:
        message: Extra explanation and context for debugging. (Note: the exception obj
    will be automatically fetched from python. No need to pass it as an argument or as
    message)
    """
    warning(f"{message}:\n{traceback.format_exc()}".replace("\n", "\n(WARNING)"))


def info(message: str, print_line: Optional[bool] = False) -> None:
    time_stamp = get_curr_time_stamp()
    info_str = (
        text_colors["info"] + text_colors["bold"] + "INFO   " + text_colors["end_color"]
    )
    print("{} - {} - {}".format(time_stamp, info_str, message))
    if print_line:
        double_dash_line(dashes=150)


def debug(message: str) -> None:
    time_stamp = get_curr_time_stamp()
    log_str = (
        text_colors["debug"]
        + text_colors["bold"]
        + "DEBUG   "
        + text_colors["end_color"]
    )
    print("{} - {} - {}".format(time_stamp, log_str, message))


def double_dash_line(dashes: Optional[int] = 75) -> None:
    print(text_colors["error"] + "=" * dashes + text_colors["end_color"])


def singe_dash_line(dashes: Optional[int] = 67) -> None:
    print("-" * dashes)


def print_header(header: str) -> None:
    double_dash_line()
    print(
        text_colors["info"]
        + text_colors["bold"]
        + "=" * 50
        + str(header)
        + text_colors["end_color"]
    )
    double_dash_line()


def print_header_minor(header: str) -> None:
    print(
        text_colors["warning"]
        + text_colors["bold"]
        + "=" * 25
        + str(header)
        + text_colors["end_color"]
    )


def disable_printing():
    sys.stdout = open(os.devnull, "w")


def enable_printing():
    sys.stdout = sys.__stdout__
