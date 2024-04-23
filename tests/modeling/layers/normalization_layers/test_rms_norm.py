#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch

from corenet.modeling.layers.normalization.rms_norm import RMSNorm


def test_rms_norm() -> None:
    in_features = 16
    norm_layer = RMSNorm(num_features=in_features)

    inputs = [
        # 3D inputs (e.g., Transformers)
        torch.randn(size=(2, 4, in_features)),
        # 4D inputs (e.g., CNNs)
        torch.randn(size=(2, 4, 5, in_features)),
        # 2D inputs (e.g., Linear)
        torch.randn(size=(2, in_features)),
    ]
    for inp in inputs:
        out = norm_layer(inp)
        assert out.shape == inp.shape
        # check if there are any NaNs in the output.
        assert not torch.any(torch.isnan(out))
