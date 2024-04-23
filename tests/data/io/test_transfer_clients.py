#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os

import pytest

from corenet.options.opts import get_training_arguments
from corenet.utils.download_utils import get_local_path


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "file_path",
    [
        # Downloading of below file has been tested on Oct 30, 2023.
        # To avoid CI/CD breaking, we skip these tests during CI/CD.
        "https://github.com/apple/ml-cvnets/blob/main/examples/range_augment/classification/mobilenet_v1.yaml",
        "http://farm4.staticflickr.com/3217/2975157083_4567dde5d5_z.jpg",
    ],
)
def test_client(file_path: str):
    opts = get_training_arguments(args=[])

    local_path = get_local_path(
        opts=opts,
        path=file_path,
        max_retries=1,
    )
    assert os.path.isfile(local_path)
