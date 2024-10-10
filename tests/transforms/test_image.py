#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import random
from typing import Tuple, Union

import pytest
import torch
from PIL import Image

from corenet.data.transforms import image_pil as pil_transforms
from corenet.data.transforms import image_torch as torch_transforms


@pytest.mark.parametrize(
    "height, width, size",
    [
        (24, 24, 20),
        (24, 24, 40),
        (32, 32, [20, 50]),
    ],
)
def test_rrc_params(height: int, width: int, size: Union[int, Tuple[int, int]]) -> None:
    # this function tests the RandomResizedCrop Params
    parser = argparse.ArgumentParser()
    parser = pil_transforms.RandomResizedCrop.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(opts, "image_augmentation.random_resized_crop.interpolation", "bilinear")

    # create dummy image
    image = Image.new("RGB", (height, width))

    # This is adapted from torchvision
    # https://github.com/pytorch/vision/blob/78ffda7eb952571df728e2ae49c2aca788596138/test/test_transforms.py#L308
    epsilon = 0.05
    min_scale = 0.25
    scale_min = max(round(random.random(), 2), min_scale)
    scale = (scale_min, scale_min + round(random.random(), 2))
    aspect_min = max(round(random.random(), 2), epsilon)
    aspect_ratio = (aspect_min, aspect_min + round(random.random(), 2))

    setattr(opts, "image_augmentation.random_resized_crop.scale", scale)
    setattr(opts, "image_augmentation.random_resized_crop.aspect_ratio", aspect_ratio)
    random_resized_crop = pil_transforms.RandomResizedCrop(opts=opts, size=size)

    i, j, h, w = random_resized_crop.get_rrc_params(image)

    assert isinstance(i, int)
    assert isinstance(j, int)
    assert isinstance(h, int)
    assert isinstance(w, int)

    obtained_aspect_ratio = w / h
    # expected aspect ratio should satisfy either of following conditions (within certain margin):
    # 1. min(aspect_ratio) - e <= obtained_aspect_ratio <= max(aspect_ratio) + e
    # 2. obtained_aspect_ratio = 1.
    assert (
        min(aspect_ratio) - epsilon
        <= obtained_aspect_ratio
        <= max(aspect_ratio) + epsilon
    ) or (obtained_aspect_ratio == 1.0)


@pytest.mark.parametrize(
    "alpha, p, sample_key, target_key, batch_size",
    [
        (-1, -0.2, None, None, 0),
        (-1, 0, None, None, 1),
        # test for inputs in tensor as well as dict format
        (-1, 0, "dummy_key", "dummy_key", 2),
        (0.0, -0.2, None, None, 0),
        (0.0, 0, "dummy_key", "dummy_key", 1),
        (0.5, 0, None, None, 2),
    ],
)
def test_mixup_transform(
    alpha: float, p: float, sample_key: str, target_key: str, batch_size: int
) -> None:
    # this function tests the Mixup transforms
    parser = argparse.ArgumentParser()
    parser = torch_transforms.RandomMixup.add_arguments(parser)

    opts = parser.parse_args([])

    num_classes = 10

    setattr(opts, "image_augmentation.mixup.alpha", alpha)
    setattr(opts, "image_augmentation.mixup.p", p)
    setattr(opts, "image_augmentation.mixup.sample_key", sample_key)
    setattr(opts, "image_augmentation.mixup.target_key", target_key)

    # torch.roll works with batch size of 0. So, we expect to see the same shape as
    # input when applying mixup transforms
    img_tensor = torch.randn(size=(batch_size, 3, 8, 8))
    label_tensor = torch.randint(low=0, high=num_classes, size=(batch_size,))

    data = {
        "samples": img_tensor if sample_key is None else {sample_key: img_tensor},
        "targets": label_tensor if target_key is None else {target_key: label_tensor},
    }

    try:
        transform = torch_transforms.RandomMixup(opts, num_classes=num_classes)
        out = transform(data)
        assert {"samples", "targets"}.issubset(list(out.keys()))
        if sample_key is None:
            assert out["samples"].shape == img_tensor.shape
        else:
            assert out["samples"][sample_key].shape == img_tensor.shape
        if target_key is None:
            assert out["targets"].shape == label_tensor.shape
        else:
            assert out["targets"][target_key].shape == label_tensor.shape
    except AssertionError as e:
        if str(e) == "Alpha param can't be zero":
            pytest.skip(str(e))
        elif (
            str(e)
            == "Mixup probability should be between 0 and 1, where 1 is inclusive"
        ):
            pytest.skip(str(e))


@pytest.mark.parametrize(
    "alpha, p, sample_key, target_key, batch_size",
    [
        (-1, -0.2, None, None, 0),
        (-1, 0, None, None, 1),
        # test for inputs in tensor as well as dict format
        (-1, 0, "dummy_key", "dummy_key", 2),
        (0.0, -0.2, None, None, 0),
        (0.0, 0, "dummy_key", "dummy_key", 1),
        (0.5, 0, None, None, 2),
    ],
)
def test_cutmix_transform(
    alpha: float, p: float, sample_key: str, target_key: str, batch_size: int
) -> None:
    # this function tests the Cutmix transforms
    parser = argparse.ArgumentParser()
    parser = torch_transforms.RandomCutmix.add_arguments(parser)

    opts = parser.parse_args([])

    num_classes = 10

    setattr(opts, "image_augmentation.cutmix.alpha", alpha)
    setattr(opts, "image_augmentation.cutmix.p", p)
    setattr(opts, "image_augmentation.cutmix.sample_key", sample_key)
    setattr(opts, "image_augmentation.cutmix.target_key", target_key)

    # torch.roll works with batch size of 0. So, we expect to see the same shape as
    # input when applying mixup transforms
    img_tensor = torch.randn(size=(batch_size, 3, 8, 8))
    label_tensor = torch.randint(low=0, high=num_classes, size=(batch_size,))

    data = {
        "samples": img_tensor if sample_key is None else {sample_key: img_tensor},
        "targets": label_tensor if target_key is None else {target_key: label_tensor},
    }

    try:
        transform = torch_transforms.RandomCutmix(opts, num_classes=num_classes)
        out = transform(data)
        assert {"samples", "targets"}.issubset(list(out.keys()))
        if sample_key is None:
            assert out["samples"].shape == img_tensor.shape
        else:
            assert out["samples"][sample_key].shape == img_tensor.shape
        if target_key is None:
            assert out["targets"].shape == label_tensor.shape
        else:
            assert out["targets"][target_key].shape == label_tensor.shape
    except AssertionError as e:
        if str(e) == "Alpha param can't be zero":
            pytest.skip(str(e))
        elif (
            str(e)
            == "Cutmix probability should be between 0 and 1, where 1 is inclusive"
        ):
            pytest.skip(str(e))


@pytest.mark.parametrize("enable", [True, False])
@pytest.mark.parametrize("mean", [1, [1, 1, 1]])
@pytest.mark.parametrize("std", [1, [1, 1, 1]])
def test_to_tensor_normalization(
    enable: bool, mean: Union[int, list], std: Union[int, list]
) -> None:
    """
    This function tests the ToTensor transform with and without mean/std normalization when the mean and std
    are either floats or lists.
    """
    C, H, W = 3, 2, 2

    parser = argparse.ArgumentParser()
    parser = pil_transforms.ToTensor.add_arguments(parser)

    opts = parser.parse_args([])
    setattr(opts, "image_augmentation.to_tensor.mean_std_normalization.enable", enable)
    setattr(opts, "image_augmentation.to_tensor.mean_std_normalization.mean", mean)
    setattr(opts, "image_augmentation.to_tensor.mean_std_normalization.std", std)
    setattr(opts, "image_augmentation.to_tensor.dtype", "uint8")

    to_tensor = pil_transforms.ToTensor(opts=opts)

    x = torch.ones(1, C, H, W)
    x = x.float()
    data = {
        "image": x,
    }

    y = to_tensor(data)["image"]
    z = torch.zeros(1, C, H, W) if enable else x

    assert torch.equal(z, y)


def test_to_tensor_normalization_no_mean_std() -> None:
    """
    This function tests the ToTensor transform when men/std normalization is enabled, but the mean/std are not provided
    """
    parser = argparse.ArgumentParser()
    parser = pil_transforms.ToTensor.add_arguments(parser)

    opts = parser.parse_args([])
    setattr(opts, "image_augmentation.to_tensor.mean_std_normalization.enable", True)

    with pytest.raises(AssertionError):
        pil_transforms.ToTensor(opts=opts)
