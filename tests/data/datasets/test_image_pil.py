#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import pytest
import torch

from corenet.data.transforms import image_pil


def test_to_tensor() -> None:
    parser = argparse.ArgumentParser()
    parser = image_pil.ToTensor.add_arguments(parser)
    opts = parser.parse_args([])

    to_tensor = image_pil.ToTensor(opts=opts)

    H, W, C = 2, 2, 3
    num_masks = 2
    data = {
        "image": torch.rand([H, W, C]),
        "mask": torch.randint(0, 1, [num_masks, H, W]),
    }

    output = to_tensor(data)

    assert output["image"].shape == (H, W, C)
    assert output["mask"].shape == (num_masks, H, W)


def test_to_tensor_bad_mask() -> None:
    parser = argparse.ArgumentParser()
    parser = image_pil.ToTensor.add_arguments(parser)
    opts = parser.parse_args([])

    to_tensor = image_pil.ToTensor(opts=opts)

    H, W, C = 2, 2, 3
    num_categories = 2
    data = {
        "image": torch.rand([H, W, C]),
        "mask": torch.randint(0, 1, [num_categories, 1, H, W]),
    }

    with pytest.raises(SystemExit):
        to_tensor(data)
