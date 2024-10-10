#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import sys

# NOTE: This environment variable should be set before any corenet import.
os.environ["CORENET_ENTRYPOINT"] = "pytest"

if sys.platform == "darwin":
    # Necessary to find sox when pytest is run in multiprocess mode. macOS
    # normally strips DYLD_LIBRARY_PATH when starting subprocesses, as a
    # security measure.
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib"

import random
import signal
from types import FrameType
from typing import Optional

import numpy as np
import pytest
import torch

from corenet import suppress_known_warnings

session_timed_out = False


def handle_timeout(signum: int, frame: Optional[FrameType] = None) -> None:
    global session_timed_out
    session_timed_out = True
    # Not calling pytest.exit() directly to capture the output of the test. See: https://stackoverflow.com/a/59234261/1139196
    pytest.fail("timeout")


def pytest_sessionstart():
    suppress_known_warnings()
    timeout = os.environ.get("PYTEST_GLOBAL_TIMEOUT", "")
    if not timeout:
        return
    if timeout.endswith("s"):
        timeout = int(timeout[:-1])
    elif timeout.endswith("m"):
        timeout = int(timeout[:-1]) * 60
    else:
        raise ValueError(
            f"Timeout value {timeout} should either end with 'm' (minutes) or 's' (seconds)."
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)


def pytest_runtest_logfinish(nodeid, location):
    if session_timed_out:
        pytest.exit("timeout")


@pytest.fixture(autouse=True)
def set_random_seed(request):
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
