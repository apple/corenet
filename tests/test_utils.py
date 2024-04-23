#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import re
from typing import Tuple

import pytest


def unset_pretrained_models_from_opts(opts: argparse.Namespace) -> None:
    """Unset the argument corresponding to pretrained model path in opts during tests"""
    opts_as_dict = vars(opts)
    for k, v in opts_as_dict.items():
        if is_pretrained_model_key(k):
            setattr(opts, k, None)


def is_pretrained_model_key(key_name: str) -> bool:
    """Check if arguments corresponding to model have a pretrained key or not."""
    return True if re.search(r".*model\..*\.pretrained$", key_name) else False


@pytest.mark.parametrize(
    "key_name_expected_output",
    [
        ("model.classification.pretrained", True),
        ("model.segmentation.pretrained", True),
        ("model.video_classification.pretrained", True),
        ("teacher.model.classification.pretrained", True),
        ("loss.classification.pretrained", False),
        ("model.classification.pretrained_dummy", False),
        ("model.classification.mypretrained", False),
        ("model.classification.my.pretrained", True),
    ],
)
def test_is_pretrained_model_key(key_name_expected_output: Tuple[str, bool]):
    key_name = key_name_expected_output[0]
    expected_output = key_name_expected_output[1]
    assert is_pretrained_model_key(key_name) == expected_output
