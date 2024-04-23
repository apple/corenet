#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch

from corenet.loss_fn.utils.class_weighting import compute_class_weights


def test_class_weighting():
    # test for checking the class weighting method
    targets = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.long)
    n_classes = 4
    norm_val = 1.0

    weights = compute_class_weights(
        target=targets, n_classes=n_classes, norm_val=norm_val
    )
    weights = torch.round(weights, decimals=2)

    expected_weights = torch.tensor([0.0, 2.47, 3.48, 6.49])

    torch.testing.assert_allclose(actual=weights, expected=expected_weights)
