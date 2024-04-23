#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List, Optional

import torch
from torch.nn import functional as F

from corenet.loss_fn.classification.focal_loss import FocalLoss
from tests.configs import get_config


def get_opts(gamma: float, weights: Optional[List[float]] = None) -> argparse.Namespace:

    opts = get_config()
    setattr(
        opts,
        "loss.classification.focal_loss.gamma",
        gamma,
    )
    setattr(
        opts,
        "loss.classification.focal_loss.weights",
        weights,
    )

    return opts


def test_focal_loss_onehot_vs_probabilities() -> None:

    opts = get_opts(gamma=2)
    loss = FocalLoss(opts)

    N, num_classes = 128, 4

    logits = torch.randn([N, num_classes])
    targets = torch.randint(num_classes, (N,))

    observed_loss = loss(None, logits, targets)

    # Now, pass one-hot labels, and make sure they're the same.
    observed_loss_2 = loss(None, logits, F.one_hot(targets, num_classes))
    assert abs(observed_loss - observed_loss_2) / observed_loss < 1e-3


def test_focal_loss_tiny_gamma() -> None:
    for gamma in [0, 1e-3]:
        opts = get_opts(gamma=gamma)
        loss = FocalLoss(opts)

        N, num_classes = 128, 4

        logits = torch.randn([N, num_classes])
        targets = torch.randint(num_classes, (N,))

        observed_loss = loss(None, logits, targets)
        ce_loss = F.cross_entropy(logits, targets)

        assert abs(observed_loss - ce_loss) / ce_loss < 1e-3


def test_focal_loss_gamma() -> None:
    opts = get_opts(gamma=5)
    loss = FocalLoss(opts)

    N, num_classes = 128, 4

    logits = torch.randn([N, num_classes])
    targets = torch.randint(num_classes, (N,))

    observed_loss = loss(None, logits, targets)
    ce_loss = F.cross_entropy(logits, targets)

    assert abs(observed_loss - ce_loss) / ce_loss > 1e-3


def test_focal_loss_weights() -> None:

    opts = get_opts(gamma=1, weights=[1, 0, 0, 0])
    loss = FocalLoss(opts)

    N, num_classes = 128, 4

    logits = torch.randn([N, num_classes])
    targets = torch.randint(num_classes, (N,))

    observed_loss = loss(None, logits, targets)
    ce_loss = F.cross_entropy(logits, targets)

    assert observed_loss < ce_loss
