#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

# LIBRARY_ROOT is the folder that contains `corenet/` module.
LIBRARY_ROOT = Path(__file__).parent.parent

MIN_TORCH_VERSION = "1.11.0"

SUPPORTED_IMAGE_EXTNS = [".png", ".jpg", ".jpeg"]  # Add image formats here
SUPPORTED_VIDEO_CLIP_VOTING_FN = ["sum", "max"]
SUPPORTED_VIDEO_READER = ["pyav", "decord"]

DEFAULT_IMAGE_WIDTH = DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_VIDEO_FRAMES = 8
DEFAULT_LOG_FREQ = 500

DEFAULT_ITERATIONS = 300000
DEFAULT_EPOCHS = 300
DEFAULT_MAX_ITERATIONS = DEFAULT_MAX_EPOCHS = 10000000

DEFAULT_RESULTS_DIR = "results"
DEFAULT_LOGS_DIR = "results/logs"

TMP_RES_FOLDER = "results_tmp"

TMP_CACHE_LOC = "/tmp/corenet"

Path(TMP_CACHE_LOC).mkdir(parents=True, exist_ok=True)

DATA_CACHE_DIR = ".corenet_data_cache"

Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def get_corenet_env() -> Literal["pytest", "default", "external"]:
    """
    Returns:
        One of the following values:
        * "pytest" iff corenet is loaded by pytest. For further details, please read
            the documentation of @is_test_env function.
        * "external" iff corenet is imported by modules other than the corenet itself
            and pytest.
        * "default" iff corenet is imported by one of its own entrypoints.

    Note: Please do not move this function to any module that has imports from corenet
        or imports for third-party non-builtin modules. The reason is that we invoke
        this function in corenet/__init__.py in order to suppress warnings *before*
        importing third-party libraries.
    """
    result = os.environ.get("CORENET_ENTRYPOINT")
    if result in (None, ""):
        # We fallback to "external", unless we detect corenet's default entrypoints
        # in sys.argv.
        result = "external"

        if len(sys.argv) > 0 and Path(sys.argv[0]).name.startswith("corenet-"):
            # This branch activates when running "corenet-train", etc. in commandline.
            os.environ["CORENET_ENTRYPOINT"] = "default"
            result = "default"
        elif len(sys.argv) > 1 and any(
            re.match(r"corenet[./](internal[./])?cli[./]", arg) for arg in sys.argv[1:]
        ):
            # This branch activates when running "python corenet/cli/main_train.py"
            # or running "python -m corenet.cli.main_train."
            os.environ["CORENET_ENTRYPOINT"] = "default"
            result = "default"

    if result not in ("pytest", "default", "external"):
        raise ValueError(
            f"Got invalid value for environment variable CORENET_ENTRYPOINT={result}."
        )
    return result


def is_test_env() -> bool:
    """
    Returns:
        True iff the corenet module is loaded by pytest.

    Note:
        - `CORENET_ENTRYPOINT=pytest` environment variable is set by `conftest.py` file.

        - Previously, we used to rely on the existence of "PYTEST_CURRENT_TEST" env var,
        which is set automatically by pytest, rather than CORENET_ENTRYPOINT=pytest.
        But the issue was that the `conftest.py` itself and some fixtures are run before
        "PYTEST_CURRENT_TEST" gets set.
    """
    return get_corenet_env() == "pytest"


def is_external_env() -> bool:
    """
    Returns:
        True iff the corenet module is loaded by modules other than the corenet itself
            and pytest
    """
    return get_corenet_env() == "external"


def if_test_env(then: Any, otherwise: Any) -> Any:
    return then if get_corenet_env() == "pytest" else otherwise
