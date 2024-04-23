#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys
from numbers import Number
from typing import Dict

from corenet.constants import is_test_env
from corenet.utils import logger


def is_iterable(x):
    return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))


def apply_recursively(x, cb, *args, **kwargs):
    if isinstance(x, dict):
        return {k: apply_recursively(v, cb, *args, **kwargs) for k, v in x.items()}
    elif is_iterable(x):
        x_type = type(x)
        return x_type([apply_recursively(y, cb, *args, **kwargs) for y in x])
    else:
        return cb(x, *args, **kwargs)


def flatten_to_dict(
    x, name: str, dict_sep: str = "/", list_sep: str = "_"
) -> Dict[str, Number]:
    if x is None:
        return {}
    elif isinstance(x, Number):
        return {name: x}
    elif isinstance(x, list):
        return {
            k: v
            for i, inner in enumerate(x)
            for k, v in flatten_to_dict(
                inner,
                name=name + list_sep + str(i),
                dict_sep=dict_sep,
                list_sep=list_sep,
            ).items()
        }
    elif isinstance(x, dict):
        return {
            k: v
            for iname, inner in x.items()
            for k, v in flatten_to_dict(
                inner,
                name=name + dict_sep + iname,
                dict_sep=dict_sep,
                list_sep=list_sep,
            ).items()
        }

    logger.error("This should never be reached!")
    return {}


def is_pytest_environment() -> bool:
    """Helper function to check if pytest environment or not"""
    logger.warning(
        DeprecationWarning(
            "utils.object_utils.is_pytest_environment is deprecated. Please use"
            " common.is_test_env instead."
        )
    )
    return is_test_env()
