#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Any, Optional

import torch
from torch import Tensor, nn


class Clip(nn.Module):
    def __init__(
        self,
        min_val: float,
        max_val: float,
        hard_clip: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.hard_clip = hard_clip

    def forward(self, x: Any) -> Any:
        if self.hard_clip:
            with torch.no_grad():
                return x.clamp_(min=self.min_val, max=self.max_val)
        else:
            return (torch.sigmoid(x) * (self.max_val - self.min_val)) + self.min_val

    def __repr__(self):
        return "{}(min={}, max={}, clipping={})".format(
            self.__class__.__name__,
            self.min_val,
            self.max_val,
            "hard" if self.hard_clip else "soft",
        )


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Any) -> Any:
        return x


class FixedSampler(nn.Module):
    def __init__(
        self,
        value: float,
        clip_fn: Optional[nn.Module] = Identity(),
        *args,
        **kwargs,
    ):
        super().__init__()
        self._value = nn.Parameter(torch.FloatTensor(1, 3, 1, 1).fill_(value))
        self.clip_fn = clip_fn

    def forward(
        self, sample_shape=(), data_type=torch.float, device=torch.device("cpu")
    ) -> Tensor:
        # sample values from uniform distribution
        return self.clip_fn(self._value)

    def __repr__(self):
        return "{}(clip_fn={})".format(
            self.__class__.__name__,
            self.clip_fn,
        )


class UniformSampler(nn.Module):
    def __init__(
        self,
        low: float,
        high: float,
        min_fn: Optional[nn.Module] = Identity(),
        max_fn: Optional[nn.Module] = Identity(),
        *args,
        **kwargs,
    ):
        super().__init__()
        self._low = nn.Parameter(torch.tensor(low, dtype=torch.float))
        self._high = nn.Parameter(torch.tensor(high, dtype=torch.float))
        self.min_fn = min_fn
        self.max_fn = max_fn

    def forward(
        self, sample_shape=(), data_type=torch.float, device=torch.device("cpu")
    ) -> Tensor:
        # sample values from uniform distribution
        rand_tensor = torch.rand(sample_shape, dtype=data_type, device=device)
        return self.low + rand_tensor * (self.high - self.low)

    @property
    def high(self):
        return self.max_fn(self._high)

    @property
    def low(self):
        return self.min_fn(self._low)

    def __repr__(self):
        return "{}(min_fn={}, max_fn={})".format(
            self.__class__.__name__,
            self.min_fn,
            self.max_fn,
        )


def random_noise(x: Tensor, variance: Tensor, *args, **kwargs) -> Tensor:
    """Apply random noise sampled."""
    noise = torch.randn_like(x) * variance
    x = x + noise
    return x


def random_contrast(x: Tensor, magnitude: Tensor, *args, **kwargs) -> Tensor:
    # compute per-channel mean
    per_channel_mean = torch.mean(x, dim=[-1, -2], keepdim=True)

    # contrast can be written as
    # (1 - contrast_factor) * per_channel_mean + img * contrast_factor
    x = ((1.0 - magnitude) * per_channel_mean) + (x * magnitude)
    return x


def random_brightness(x: Tensor, magnitude: Tensor, *args, **kwargs) -> Tensor:
    """
    Brightness function.
    """
    x = x * magnitude
    return x


def identity(x: Tensor, *args, **kwargs) -> Tensor:
    """Identity function"""
    return x
