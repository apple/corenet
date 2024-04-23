#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys

import pytest

from corenet.modeling.neural_augmentor.utils.neural_aug_utils import *


@pytest.mark.parametrize("noise_var", [0.0001, 0.01, 0.1])
def test_random_noise(noise_var):
    in_channels = 3
    in_height = 224
    in_width = 224
    x = torch.ones(size=(1, in_channels, in_width, in_height), dtype=torch.float)

    aug_out = random_noise(x, variance=torch.tensor(noise_var, dtype=torch.float))

    torch.testing.assert_allclose(actual=x.shape, expected=aug_out.shape)


@pytest.mark.parametrize("magnitude", [0.1, 1.0, 2.0])
def test_random_brightness(magnitude):
    in_channels = 3
    in_height = 224
    in_width = 224
    x = torch.ones(size=(1, in_channels, in_width, in_height), dtype=torch.float)

    aug_out = random_brightness(x, magnitude=torch.tensor(magnitude, dtype=torch.float))

    torch.testing.assert_allclose(actual=x.shape, expected=aug_out.shape)


@pytest.mark.parametrize("magnitude", [0.1, 1.0, 2.0])
def test_random_contrast(magnitude):
    in_channels = 3
    in_height = 224
    in_width = 224
    x = torch.ones(size=(1, in_channels, in_width, in_height), dtype=torch.float)

    aug_out = random_contrast(x, magnitude=torch.tensor(magnitude, dtype=torch.float))

    torch.testing.assert_allclose(actual=x.shape, expected=aug_out.shape)
