#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import math
import random
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F_torch
from torchvision.transforms import functional as F

from corenet.data.transforms import TRANSFORMATIONS_REGISTRY, BaseTransformation
from corenet.utils import logger


# Copied from PyTorch Torchvision
@TRANSFORMATIONS_REGISTRY.register(name="random_mixup", type="image_torch")
class RandomMixup(BaseTransformation):
    """
    Given a batch of input images and labels, this class randomly applies the
    `MixUp transformation <https://arxiv.org/abs/1710.09412>`_

    Args:
        opts (argparse.Namespace): Arguments
        num_classes (int): Number of classes in the dataset
    """

    def __init__(
        self, opts: argparse.Namespace, num_classes: int, *args, **kwargs
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        alpha = getattr(opts, "image_augmentation.mixup.alpha")
        assert (
            num_classes > 0
        ), "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = getattr(opts, "image_augmentation.mixup.p")
        assert (
            0.0 < self.p <= 1.0
        ), "MixUp probability should be between 0 and 1, where 1 is inclusive"
        self.alpha = alpha
        self.inplace = getattr(opts, "image_augmentation.mixup.inplace")
        self.sample_key = getattr(opts, "image_augmentation.mixup.sample_key")
        self.target_key = getattr(opts, "image_augmentation.mixup.target_key")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.mixup.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.mixup.alpha",
            type=float,
            default=0.2,
            help="Alpha for MixUp augmentation. Defaults to 0.2",
        )
        group.add_argument(
            "--image-augmentation.mixup.p",
            type=float,
            default=1.0,
            help="Probability for applying MixUp augmentation. Defaults to 1.0 ."
            " If both MixUp and CutMix are enabled,"
            " one is used with probability 0.5 per batch.",
        )
        group.add_argument(
            "--image-augmentation.mixup.inplace",
            action="store_true",
            default=False,
            help="Apply MixUp augmentation inplace. Defaults to False.",
        )

        group.add_argument(
            "--image-augmentation.mixup.sample-key",
            type=str,
            default=None,
            help="Name of the key if input is a dictionart. Defaults to None.",
        )

        group.add_argument(
            "--image-augmentation.mixup.target-key",
            type=str,
            default=None,
            help="Name of the key if target is a dictionary. Defaults to None.",
        )

        return parser

    def _apply_mixup_transform(
        self, image_tensor: Tensor, target_tensor: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if image_tensor.ndim != 4:
            logger.error(f"Batch ndim should be 4. Got {image_tensor.ndim}")
        if target_tensor.ndim != 1:
            logger.error(f"Target ndim should be 1. Got {target_tensor.ndim}")
        if not image_tensor.is_floating_point():
            logger.error(
                f"Batch datatype should be a float tensor. Got {image_tensor.dtype}."
            )
        if target_tensor.dtype != torch.int64:
            logger.error(
                f"Target datatype should be torch.int64. Got {target_tensor.dtype}"
            )

        if not self.inplace:
            image_tensor = image_tensor.clone()
            target_tensor = target_tensor.clone()

        if target_tensor.ndim == 1:
            target_tensor = F_torch.one_hot(
                target_tensor, num_classes=self.num_classes
            ).to(dtype=image_tensor.dtype)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image_tensor.roll(1, 0)
        target_rolled = target_tensor.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        image_tensor.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target_tensor.mul_(lambda_param).add_(target_rolled)
        return image_tensor, target_tensor

    def __call__(self, data: Dict) -> Dict:
        """
        Input data format:
            data: mapping of: {
                "samples": {"sample_key": Tensor of shape: [Batch, Channels, Height, Width]},
                "targets": {"target_key": IntTensor of shape: [Batch]}
            }

            OR
            data: mapping of: {
                "samples": {"sample_key": Tensor of shape: [Batch, Channels, Height, Width]},
                "targets": IntTensor of shape: [Batch]
            }

            OR
            data: mapping of: {
                "samples": Tensor of shape: [Batch, Channels, Height, Width],
                "targets": {"target_key": IntTensor of shape: [Batch]}
            }
            OR
            data: mapping of: {
                "samples": Tensor of shape: [Batch, Channels, Height, Width],
                "targets": IntTensor of shape: [Batch]
            }
        Output data format: Same as the input
        """

        if torch.rand(1).item() >= self.p:
            return data

        samples, targets = data.pop("samples"), data.pop("targets")

        if self.sample_key is not None:
            samples = samples[self.sample_key]
            if not isinstance(samples, Tensor):
                logger.error(
                    f"Samples need to be of type Tensor. Got: {type(samples)}. "
                    f"Maybe you want to check the value of --image-augmentation.mixup.sample-key"
                )

        if self.target_key is not None:
            targets = targets[self.target_key]
            if not isinstance(targets, Tensor):
                logger.error(
                    f"Targets need to be of type Tensor. Got: {type(targets)}. "
                    f"Maybe you want to check the value of --image-augmentation.mixup.target-key"
                )

        samples, targets = self._apply_mixup_transform(
            image_tensor=samples, target_tensor=targets
        )

        if self.sample_key is not None:
            if isinstance(samples, Tensor):
                samples = {self.sample_key: samples}
            else:
                samples[self.sample_key] = samples
        if self.target_key is not None:
            if isinstance(targets, Tensor):
                targets = {self.target_key: targets}
            else:
                targets[self.target_key] = targets

        data.update({"samples": samples, "targets": targets})

        return data

    def __repr__(self) -> str:
        return "{}(num_classes={}, p={}, alpha={}, inplace={})".format(
            self.__class__.__name__, self.num_classes, self.p, self.alpha, self.inplace
        )


@TRANSFORMATIONS_REGISTRY.register(name="random_cutmix", type="image_torch")
class RandomCutmix(BaseTransformation):
    """
    Given a batch of input images and labels, this class randomly applies the
    `CutMix transformation <https://arxiv.org/abs/1905.04899>`_

    Args:
        opts (argparse.Namespace): Arguments
        num_classes (int): Number of classes in the dataset
    """

    def __init__(
        self, opts: argparse.Namespace, num_classes: int, *args, **kwargs
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        alpha = getattr(opts, "image_augmentation.cutmix.alpha")
        assert (
            num_classes > 0
        ), "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = getattr(opts, "image_augmentation.cutmix.p")
        assert (
            0.0 < self.p <= 1.0
        ), "CutMix probability should be between 0 and 1, where 1 is inclusive"
        self.alpha = alpha
        self.inplace = getattr(opts, "image_augmentation.cutmix.inplace")
        self.sample_key = getattr(opts, "image_augmentation.cutmix.sample_key")
        self.target_key = getattr(opts, "image_augmentation.cutmix.target_key")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.cutmix.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )

        group.add_argument(
            "--image-augmentation.cutmix.alpha",
            type=float,
            default=1.0,
            help="Alpha for cutmix augmentation. Defaults to 1.0",
        )
        group.add_argument(
            "--image-augmentation.cutmix.p",
            type=float,
            default=1.0,
            help="Probability for applying cutmix augmentation. Defaults to 1.0"
            " If both MixUp and CutMix are enabled,"
            " one is used with probability 0.5 per batch.",
        )
        group.add_argument(
            "--image-augmentation.cutmix.inplace",
            action="store_true",
            default=False,
            help="Apply cutmix operation inplace. Defaults to False",
        )

        group.add_argument(
            "--image-augmentation.cutmix.sample-key",
            type=str,
            default=None,
            help="Name of the key if input is a dictionary. Defaults to None.",
        )

        group.add_argument(
            "--image-augmentation.cutmix.target-key",
            type=str,
            default=None,
            help="Name of the key if target is a dictionary. Defaults to None.",
        )
        return parser

    def _apply_cutmix_transform(
        self, image_tensor: Tensor, target_tensor: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if image_tensor.ndim != 4:
            logger.error(f"Batch ndim should be 4. Got {image_tensor.ndim}")
        if target_tensor.ndim != 1:
            logger.error(f"Target ndim should be 1. Got {target_tensor.ndim}")
        if not image_tensor.is_floating_point():
            logger.error(
                f"Batch dtype should be a float tensor. Got {image_tensor.dtype}."
            )
        if target_tensor.dtype != torch.int64:
            logger.error(
                f"Target dtype should be torch.int64. Got {target_tensor.dtype}"
            )

        if not self.inplace:
            image_tensor = image_tensor.clone()
            target_tensor = target_tensor.clone()

        if target_tensor.ndim == 1:
            target_tensor = F_torch.one_hot(
                target_tensor, num_classes=self.num_classes
            ).to(dtype=image_tensor.dtype)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image_tensor.roll(1, 0)
        target_rolled = target_tensor.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        W, H = F.get_image_size(image_tensor)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        image_tensor[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target_tensor.mul_(lambda_param).add_(target_rolled)
        return image_tensor, target_tensor

    def __call__(self, data: Dict) -> Dict:
        """
        Input data format:
            data: mapping of: {
                "samples": {"sample_key": Tensor of shape: [Batch, Channels, Height, Width]},
                "targets": {"target_key": IntTensor of shape: [Batch]}
            }

            OR
            data: mapping of: {
                "samples": {"sample_key": Tensor of shape: [Batch, Channels, Height, Width]},
                "targets": IntTensor of shape: [Batch]
            }

            OR
            data: mapping of: {
                "samples": Tensor of shape: [Batch, Channels, Height, Width],
                "targets": {"target_key": IntTensor of shape: [Batch]}
            }
            OR
            data: mapping of: {
                "samples": Tensor of shape: [Batch, Channels, Height, Width],
                "targets": IntTensor of shape: [Batch]
            }
        Output data format: Same as the input
        """

        if torch.rand(1).item() >= self.p:
            return data

        samples, targets = data.pop("samples"), data.pop("targets")

        if self.sample_key is not None:
            samples = samples[self.sample_key]
            if not isinstance(samples, Tensor):
                logger.error(
                    f"Samples need to be of type Tensor. Got: {type(samples)}. "
                    f"Maybe you want to check the value of --image-augmentation.cutmix.sample-key"
                )

        if self.target_key is not None:
            targets = targets[self.target_key]
            if not isinstance(targets, Tensor):
                logger.error(
                    f"Targets need to be of type Tensor. Got: {type(targets)}. "
                    f"Maybe you want to check the value of --image-augmentation.cutmix.target-key"
                )

        samples, targets = self._apply_cutmix_transform(
            image_tensor=samples, target_tensor=targets
        )

        if self.sample_key is not None:
            if isinstance(samples, Tensor):
                samples = {self.sample_key: samples}
            else:
                samples[self.sample_key] = samples
        if self.target_key is not None:
            if isinstance(targets, Tensor):
                targets = {self.target_key: targets}
            else:
                targets[self.target_key] = targets

        data.update({"samples": samples, "targets": targets})
        return data

    def __repr__(self) -> str:
        return "{}(num_classes={}, p={}, alpha={}, inplace={})".format(
            self.__class__.__name__, self.num_classes, self.p, self.alpha, self.inplace
        )


def apply_mixing_transforms(opts: argparse.Namespace, data: Dict) -> Dict:
    """
    Helper function to apply MixUp/CutMix transforms. If both MixUp and CutMix transforms
    are selected with 0.0 < p <= 1.0, then one of them is chosen randomly and applied.

    Input data format:
        data: mapping of: {
            "samples": {"sample_key": Tensor of shape: [Batch, Channels, Height, Width]},
            "targets": {"target_key": IntTensor of shape: [Batch]}
        }

        OR
        data: mapping of: {
            "samples": {"sample_key": Tensor of shape: [Batch, Channels, Height, Width},
            "targets": IntTensor of shape: [Batch]
        }

        OR
        data: mapping of: {
            "samples": Tensor of shape: [Batch, Channels, Height, Width],
            "targets": {"target_key": IntTensor of shape: [Batch]}
        }
        OR
        data: mapping of: {
            "samples": Tensor of shape: [Batch, Channels, Height, Width],
            "targets": IntTensor of shape: [Batch]
        }
    Output data format: Same as the input
    """

    mixup_transforms = []
    if getattr(opts, "image_augmentation.mixup.enable"):
        n_classes = getattr(opts, "model.classification.n_classes")
        if n_classes is None:
            logger.error("Please specify number of classes. Got None.")
        mixup_transforms.append(RandomMixup(opts=opts, num_classes=n_classes))

    if getattr(opts, "image_augmentation.cutmix.enable"):
        n_classes = getattr(opts, "model.classification.n_classes")
        if n_classes is None:
            logger.error("Please specify number of classes. Got None.")
        mixup_transforms.append(RandomCutmix(opts=opts, num_classes=n_classes))

    if len(mixup_transforms) > 0:
        _mixup_transform = random.choice(mixup_transforms)
        data = _mixup_transform(data)
    return data
