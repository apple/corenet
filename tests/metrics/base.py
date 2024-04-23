#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Tuple

import pytest
import torch


def sample_classification_outputs() -> Tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.tensor(
        [
            [0.02, 0.01, 0.91, 0.04, 0.01, 0.01, 0],
            [0.81, 0.03, 0.16, 0, 0, 0, 0],
            [0.4, 0.51, 0.05, 0.04, 0, 0, 0],
        ],
        dtype=torch.float,
    )

    predictions = (predictions + 1e-6).log()

    targets = torch.tensor([3, 2, 1], dtype=torch.long)
    return predictions, targets


@pytest.fixture(
    params=[
        {},
        {"pred": "pred_key"},
        {"target": "target_key"},
        {"pred": "pred_key", "target": "target_key"},
        {"target": "target_key", "pred": "pred_key"},
    ]
)
def transform_args(request):
    """
    Allows for testing all metrics with <metric>(pred=key1,target=key2) registry format.

    Tests the following combinations:
        <metric>
        <metric>(pred=pred_key)
        <metric>(target=target_key)
        <metric>(pred=pred_key, target=target_key)
        <metric>(target=target_key, pred=pred_key)
    """
    param_keys = request.param

    def encapsulate_with_keys(metric_names, predictions, targets, extras=None):
        if isinstance(metric_names, str):
            metric_names = [metric_names]

        pred_key = param_keys.get("pred", None)
        if pred_key:
            predictions = {pred_key: predictions}

        target_key = param_keys.get("target", None)
        if target_key:
            targets = {target_key: targets}

        # e.g. assuming param_keys = {"pred": "pred_key", "target": "target_key"} we will get:
        # {"pred": "pred_key", "target": "target_key"} -> ["pred=pred_key", "target=target_key"]
        # ["pred=pred_key", "target=target_key"] -> "(pred=pred_key,target=target_key)"
        params_str = ", ".join([f"{key}={value}" for key, value in param_keys.items()])
        if params_str:
            params_str = f"({params_str})"
            metric_names = [m + params_str for m in metric_names]

        return metric_names, (predictions, targets, extras)

    return encapsulate_with_keys
