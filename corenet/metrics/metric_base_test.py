#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, Union

import torch
from torch import Tensor

from corenet.metrics.metric_base import AverageMetric


class DummyMetric(AverageMetric):
    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        return prediction


def test_average_metric_distributed_batchsize(mocker):
    mocker.patch("torch.distributed.is_initialized", return_value=True)
    mocker.patch("torch.distributed.get_world_size", return_value=2)
    mocker.patch("torch.distributed.all_reduce", lambda x, *_, **__: x.add_(1))

    metric = DummyMetric(None, is_distributed=True)
    metric.update(torch.tensor([2.0]), None, batch_size=torch.tensor([2]))

    # Value is 2 and batch size is 2, but we're simulating the second device
    # having value 1 and batch size 1 by making sure all_reduce adds 1 to both
    # the value and the batch size. It's as if we have [2, 2] in GPU1 and [1]
    # in GPU 2. Therefore the expected average is 5/3.

    expected_value = (2 * 2 + 1 * 1) / 3
    assert metric.compute() == expected_value
