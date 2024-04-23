#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import random
from typing import List, Optional

import torch
from torch import Tensor, nn

from corenet.modeling.misc.common import parameter_list
from corenet.modeling.neural_augmentor.utils.neural_aug_utils import (
    Clip,
    FixedSampler,
    UniformSampler,
    random_brightness,
    random_contrast,
    random_noise,
)
from corenet.utils import logger

_distribution_tuple = (UniformSampler,)


class BaseNeuralAugmentor(nn.Module):
    """
    Base class for `neural (or range) augmentation <https://arxiv.org/abs/2212.10553>`_
    """

    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts

        self.lr_multiplier = getattr(
            opts, "model.learn_augmentation.lr_multiplier", 1.0
        )

        # Set variables corresponding to different transforms to None.
        # We will override them in child classes with learnable versions
        self.brightness = None
        self.contrast = None
        self.noise = None

        self.aug_fns = []

    def _is_valid_aug_fn_list(self, aug_fns):
        if self.training:
            if len(aug_fns) == 0:
                logger.error(
                    "{} needs at least one learnable function.".format(
                        self.__class__.__name__
                    )
                )

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        """Get trainable parameters"""
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [self.lr_multiplier] * len(param_list)

    def __repr__(self):
        aug_str = "{}(".format(self.__class__.__name__)

        if self.brightness is not None:
            aug_str += "\n\tBrightness={}, ".format(
                self.brightness.data.shape
                if isinstance(self.brightness, nn.Parameter)
                else self.brightness
            )

        if self.contrast is not None:
            aug_str += "\n\tContrast={}, ".format(
                self.contrast.data.shape
                if isinstance(self.contrast, nn.Parameter)
                else self.contrast
            )

        if self.noise is not None:
            aug_str += "\n\tNoise={}, ".format(
                self.noise.data.shape
                if isinstance(self.noise, nn.Parameter)
                else self.noise
            )

        aug_str += self.extra_repr()
        aug_str += ")"
        return aug_str

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--model.learn-augmentation.mode",
            type=str,
            default=None,
            choices=["basic", "distribution"],
            help="Neural augmentation mode",
        )

        group.add_argument(
            "--model.learn-augmentation.brightness",
            action="store_true",
            help="Learn parameters for brightness",
        )

        group.add_argument(
            "--model.learn-augmentation.contrast",
            action="store_true",
            help="Learn parameters for contrast",
        )

        group.add_argument(
            "--model.learn-augmentation.noise",
            action="store_true",
            help="Learn parameters for noise",
        )

        # LR multiplier
        group.add_argument(
            "--model.learn-augmentation.lr-multiplier",
            type=float,
            default=1.0,
            help="LR multiplier for neural aug parameters",
        )

        return parser

    def _build_aug_fns(self, opts) -> List:
        raise NotImplementedError

    def _apply_brightness(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Apply brightness augmentation function with learnable parameters.
        """
        # self._check_brightness_bounds()
        x_shape = [*x.shape]
        x_shape[1:] = [1] * (len(x_shape) - 1)
        if isinstance(self.brightness, nn.Parameter):
            # learning a fixed number of parameters
            magnitude = self.brightness
        elif isinstance(self.brightness, _distribution_tuple):
            # learning a distribution range from which parameter is sampled.
            magnitude = self.brightness(x_shape, device=x.device, data_type=x.dtype)
        else:
            raise NotImplementedError
        return random_brightness(x, magnitude, *args, **kwargs)

    def _apply_contrast(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Apply contrast augmentation function with learnable parameters.
        """
        # self._check_contrast_bounds()
        x_shape = [*x.shape]
        x_shape[1:] = [1] * (len(x_shape) - 1)

        if isinstance(self.contrast, nn.Parameter):
            # learning a fixed number of parameters
            magnitude = self.contrast
        elif isinstance(self.contrast, _distribution_tuple):
            # learning a distribution range from which parameter is sampled.
            magnitude = self.contrast(x_shape, device=x.device, data_type=x.dtype)
        else:
            raise NotImplementedError
        return random_contrast(x, magnitude, *args, *kwargs)

    def _apply_noise(self, x: Tensor, *args, **kwargs) -> Tensor:
        # self._check_noise_bounds()
        x_shape = [*x.shape]
        x_shape[1:] = [1] * (len(x_shape) - 1)

        if isinstance(self.noise, nn.Parameter):
            # learning a fixed number of parameters
            variance = self.noise
        elif isinstance(self.noise, _distribution_tuple):
            # learning a distribution range from which parameter is sampled.
            variance = self.noise(x_shape, device=x.device, data_type=x.dtype)
        else:
            raise NotImplementedError
        return random_noise(x, variance, *args, *kwargs)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        batch_size, in_channels, in_height, in_width = x.shape

        # Randomly apply augmentation to 50% of the samples
        n_aug_samples = max(1, (batch_size // 2))

        # shuffle the order of augmentations
        random.shuffle(self.aug_fns)

        for aug_fn in self.aug_fns:
            # select 50% samples for augmentation
            sample_ids = torch.randperm(
                n=batch_size, dtype=torch.long, device=x.device
            )[:n_aug_samples]
            x_aug = torch.index_select(x, dim=0, index=sample_ids)
            # apply augmentation
            x_aug = aug_fn(x=x_aug)
            # copy augmented samples to tensor
            x = torch.index_copy(x, dim=0, source=x_aug, index=sample_ids)

        # clip the values so that they are between 0 and 1
        x = torch.clip(x, min=0.0, max=1.0)
        return x


class BasicNeuralAugmentor(BaseNeuralAugmentor):
    """
    Basic neural augmentation. This class learns per-channel augmentation parameters
    and apply the same parameter to all images in a batch.

    See `neural (or range) augmentation <https://arxiv.org/abs/2212.10553>`_ paper for details.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        aug_fns = self._build_aug_fns(opts=opts)

        self._is_valid_aug_fn_list(aug_fns)

        self.aug_fns = aug_fns

    def _build_aug_fns(self, opts) -> List:
        aug_fns = []
        if getattr(opts, "model.learn_augmentation.brightness", False):
            self.brightness = FixedSampler(
                value=1.0, clip_fn=Clip(min_val=0.1, max_val=10.0)
            )
            aug_fns.append(self._apply_brightness)

        if getattr(opts, "model.learn_augmentation.contrast", False):
            self.contrast = FixedSampler(
                value=1.0, clip_fn=Clip(min_val=0.1, max_val=10.0)
            )
            aug_fns.append(self._apply_contrast)

        if getattr(opts, "model.learn_augmentation.noise", False):
            self.noise = FixedSampler(value=0.0, clip_fn=Clip(min_val=0.0, max_val=1.0))
            aug_fns.append(self._apply_noise)

        return aug_fns


class DistributionNeuralAugmentor(BaseNeuralAugmentor):
    """
    Distribution-based neural (or range) augmentation. This class samples the augmentation parameters
    from a specified distribution with learnable range.

    See `neural (or range) augmentation <https://arxiv.org/abs/2212.10553>`_ paper for details.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)

        aug_fns = self._build_aug_fns_with_uniform_dist(opts=opts)
        self._is_valid_aug_fn_list(aug_fns)
        self.aug_fns = aug_fns

    def _build_aug_fns_with_uniform_dist(self, opts) -> List:
        # need to define the learnable parameters in a way that are compatible with bucketing
        aug_fns = []
        if getattr(opts, "model.learn_augmentation.brightness", False):
            self.brightness = UniformSampler(
                low=0.5,
                high=1.5,
                min_fn=Clip(min_val=0.1, max_val=0.9),
                max_fn=Clip(min_val=1.1, max_val=10.0),
            )
            aug_fns.append(self._apply_brightness)

        if getattr(opts, "model.learn_augmentation.contrast", False):
            self.contrast = UniformSampler(
                low=0.5,
                high=1.5,
                min_fn=Clip(min_val=0.1, max_val=0.9),
                max_fn=Clip(min_val=1.1, max_val=10.0),
            )
            aug_fns.append(self._apply_contrast)

        if getattr(opts, "model.learn_augmentation.noise", False):
            self.noise = UniformSampler(
                low=0.0,
                high=0.1,
                min_fn=Clip(min_val=0.0, max_val=0.00005),
                max_fn=Clip(min_val=0.0001, max_val=1.0),
            )
            aug_fns.append(self._apply_noise)

        return aug_fns


def build_neural_augmentor(opts, *args, **kwargs):
    mode = getattr(opts, "model.learn_augmentation.mode", None)

    if mode is None:
        mode = "none"

    mode = mode.lower()
    if mode == "distribution":
        return DistributionNeuralAugmentor(opts=opts, *args, **kwargs)
    elif mode == "basic":
        return BasicNeuralAugmentor(opts=opts, *args, **kwargs)
    else:
        return None
