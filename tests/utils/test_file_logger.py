#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import tempfile

import pytest
import torch

from corenet.utils.file_logger import FileLogger


@pytest.mark.parametrize(
    "metric_name, epoch1, value1, epoch2, value2",
    [("metric", 0, 1.0, 1, 2.0), ("metric2", 5, 1.0, 6, 2.0)],
)
def test_file_logger(
    metric_name: str, epoch1: int, value1: float, epoch2: int, value2: float
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        # Case 1: The file doesn't exist.
        filename = os.path.join(tempdir, "stats.pt")
        logger = FileLogger(filename)

        logger.add_scalar(metric_name, value1, epoch1)
        logger.close()
        assert os.path.exists(filename)

        a = torch.load(filename)
        assert a == {"epochs": {epoch1: {"metrics": {metric_name: value1}}}}

        # Case 2: The file does exist.
        logger = FileLogger(filename)
        logger.add_scalar(metric_name, value2, epoch2)
        logger.close()

        a = torch.load(filename)
        assert a == {
            "epochs": {
                epoch1: {"metrics": {metric_name: value1}},
                epoch2: {"metrics": {metric_name: value2}},
            }
        }
