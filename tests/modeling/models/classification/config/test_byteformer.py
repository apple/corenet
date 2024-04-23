#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import pytest

from corenet.modeling.models.classification.config import byteformer


@pytest.mark.parametrize("mode", ["tiny", "small", "base", "huge"])
def test_get_configuration(mode) -> None:
    opts = argparse.Namespace()
    setattr(opts, "model.classification.byteformer.mode", mode)
    setattr(opts, "model.classification.byteformer.dropout", 0.0)
    setattr(opts, "model.classification.byteformer.norm_layer", "layer_norm")
    byteformer.get_configuration(opts)
