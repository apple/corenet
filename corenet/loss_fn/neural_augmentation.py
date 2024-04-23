#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import math
from typing import List, Mapping, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master


# NeuralAugmentation can be used with any task. Therefore, we register both name and type
# as the same.
@LOSS_REGISTRY.register(name="neural_augmentation", type="neural_augmentation")
class NeuralAugmentation(BaseCriteria):
    """Compute the augmentation loss, as described in the
    `RangeAugment <https://arxiv.org/abs/2212.10553>`_ paper.

    Args:
        opts: command line arguments
    """

    __supported_metrics = ["psnr"]

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        perceptual_metric = getattr(opts, "loss.neural_augmentation.perceptual_metric")
        is_master_node = is_master(opts)
        if perceptual_metric is None and is_master_node:
            logger.error(
                "Perceptual metric can't be none. "
                "Please specify perceptual metric using --loss.auxiliary.neural-augmentation.perceptual-metric argument"
            )
        if not isinstance(perceptual_metric, str) and is_master_node:
            logger.error(
                "The type of perceptual metric is not string. Got: {}".format(
                    type(perceptual_metric)
                )
            )
        perceptual_metric = perceptual_metric.lower()
        target_value = getattr(opts, "loss.neural_augmentation.target_value")

        self.curriculumn_learning = False
        self.iteration_based_training = getattr(
            opts, "scheduler.is_iteration_based", False
        )
        self.target_str = f"{target_value}"
        alpha = getattr(opts, "loss.neural_augmentation.alpha")
        if perceptual_metric == "psnr":
            if target_value is None and is_master_node:
                logger.error("Target PSNR value can not be None.")

            if isinstance(target_value, (int, float)):
                if target_value < 0:
                    if is_master_node:
                        logger.error(
                            "PSNR value should be >= 0 in {}. Got: {}".format(
                                self.__class__.__name__, target_value
                            )
                        )
                # compute target MSE using below equation
                # # PSNR = 20 log10(255) - 10 log10(MSE)
                target_mse = 10.0 ** ((20.0 * math.log10(255.0) - target_value) / 10.0)
                self.target_value = torch.ones(size=(1,), dtype=torch.float).fill_(
                    target_mse
                )
                self.target_str = f"{target_value}"
            elif isinstance(target_value, (list, tuple)) and len(target_value) == 2:
                start_target_value = target_value[0]
                end_target_value = target_value[1]

                if start_target_value < 0 or end_target_value < 0:
                    if is_master_node:
                        logger.error(
                            "PSNR value should be >= 0 in {}. Got: {}".format(
                                self.__class__.__name__, target_value
                            )
                        )

                # compute target MSE using below equation
                # # PSNR = 20 log10(255) - 10 log10(MSE)
                start_target_mse = 10.0 ** (
                    (20.0 * math.log10(255.0) - start_target_value) / 10.0
                )
                end_target_mse = 10.0 ** (
                    (20.0 * math.log10(255.0) - end_target_value) / 10.0
                )

                max_steps = (
                    getattr(opts, "scheduler.max_iterations")
                    if self.iteration_based_training
                    else getattr(opts, "scheduler.max_epochs")
                )

                if max_steps is None and is_master_node:
                    logger.error(
                        "Please specify {}. Got None.".format(
                            "--scheduler.max-iterations"
                            if self.iteration_based_training
                            else "--scheduler.max-epochs"
                        )
                    )

                curriculum_method = getattr(
                    opts, "loss.neural_augmentation.curriculum_method"
                )
                if curriculum_method in CURRICULUM_METHOD.keys():
                    self.target_value = CURRICULUM_METHOD[curriculum_method](
                        start=start_target_mse, end=end_target_mse, period=max_steps
                    )
                else:
                    raise NotImplementedError

                self.curriculumn_learning = True
                self.target_str = f"[{start_target_value}, {end_target_value}]"
            else:
                raise NotImplementedError

            # the maximum possible MSE error is computed as:
            # a = torch.ones((3, H, W)) * 255.0 # Max. input value is 255.0
            # b = torch.zeros((3, H, W)) # min. input value is 0.0
            # mse = torch.mean( (a -b) ** 2)
            # 65025 is the maximum mse
            self.alpha = alpha / 65025.0
        else:
            if is_master_node:
                logger.error(
                    "Supported perceptual metrics are: {}. Got: {}".format(
                        self.__supported_metrics, perceptual_metric
                    )
                )
        self.perceptual_metric = perceptual_metric
        self.device = getattr(opts, "dev.device", torch.device("cpu"))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--loss.neural-augmentation.perceptual-metric",
            type=str,
            default="psnr",
            choices=cls.__supported_metrics,
            help=f"Name of the perceptual metric to be used in {cls.__name__}.",
        )

        group.add_argument(
            "--loss.neural-augmentation.target-value",
            type=float,
            default=[40, 20],
            nargs="+",
            help=f"Target image similarity value in {cls.__name__}. Defaults to [40, 20]",
        )

        group.add_argument(
            "--loss.neural-augmentation.curriculum-method",
            type=str,
            default="cosine",
            choices=["linear", "cosine"],
            help=f"Curriculum for varying the target image similarity value in {cls.__name__}."
            f"Supported curriculums are {cls.__supported_metrics}. Defaults to cosine",
        )
        group.add_argument(
            "--loss.neural-augmentation.alpha",
            default=100.0,
            type=float,
            help="Scale loss value by alpha value. Defaults to 100. "
            "Note: When perceptual metric is PSNR, alpha value is divided by 65025",
        )
        return parser

    def _forward_psnr(
        self, input_tensor: Tensor, augmented_tensor: Tensor, *args, **kwargs
    ) -> Tensor:
        """Compute the MSE error between input and augmented image, and minimizes
        the distance between MSE error and target error.

        Args:
            input_tensor: Input image of shape [N, C, H, W]
            augmented_tensor: Augmented image of shape [N, C, H, W]

        Returns:
            A scalar loss value
        """

        squared_err = ((augmented_tensor - input_tensor) * 255.0) ** 2
        # [B, C, H, W] --> [B]
        pred_mse = torch.mean(squared_err, dim=[1, 2, 3])

        if self.curriculumn_learning:
            step = (
                kwargs.get("iterations", 0)
                if self.iteration_based_training
                else kwargs.get("epoch", 0)
            )
            if step >= len(self.target_value):
                step = -1
            target_mse = self.target_value[step]
        else:
            target_mse = self.target_value

        # compute L1 loss between target and current MSE
        smooth_l1_loss = F.smooth_l1_loss(
            input=pred_mse,
            target=target_mse.expand_as(pred_mse).to(
                device=pred_mse.device, dtype=pred_mse.dtype
            ),
            reduction="mean",
        )

        loss_na = smooth_l1_loss * self.alpha
        return loss_na

    def _compute_loss(
        self, input_tensor: Tensor, augmented_tensor: Tensor, *args, **kwargs
    ) -> Tensor:
        """Compute the neural augmentation loss.

        Args:
            input_tensor: Input image of shape [N, C, H, W]
            augmented_tensor: Augmented image of shape [N, C, H, W]

        Returns:
            A scalar value
        """
        if augmented_tensor is None:
            logger.error(
                f"Augmented tensor can't be None in {self.__class__.__name__} during training mode."
            )

        forward_loss_fn = getattr(self, f"_forward_{self.perceptual_metric}")
        loss_na = forward_loss_fn(
            input_tensor=input_tensor,
            augmented_tensor=augmented_tensor,
            *args,
            **kwargs,
        )
        return loss_na

    def forward(
        self,
        input_sample: Union[Tensor, Mapping[str, Union[Tensor, List[Tensor]]]],
        prediction: Mapping[str, Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        """Compute the loss between input and augmented image, as described in
        `RangeAugment <https://arxiv.org/abs/2212.10553>`_ paper.

        Args:
            input_sample: Input sample can either be a Tensor or a dictionary with mandatory key "image". In
                case of a dictionary, the values can be a Tensor or list of Tensors.
            prediction: Output of augmentation model. Mapping of (string: Tensor) with `augmented_tensor`
                as the required key.

        Shapes:
            input_sample:
                * Tensor: The shape of input tensor is [N, C, H, W]
                * Mapping[str, Tensor]: The shape of tensor is [N, C, H, W]
                * Mapping[str, List[Tensor]]: The length of List is N, and the shape of each tensor is [1, C, H, W]
            prediction: The shape of prediction["augmented_tensor"] is [N, C, H, W]

        Returns:
            A scalar loss value

        ...note:
            During validation or evaluation, neural augmentation loss is not computed and 0 is returned
        """
        if not self.training:
            return torch.tensor(0.0, device=self.device, dtype=torch.float)

        if not isinstance(prediction, Mapping):
            logger.error(
                "Prediction needs to be an instance of Mapping and must contain augmented_tensor"
                " as keys"
            )
        if isinstance(input_sample, Mapping):
            input_sample = input_sample["image"]

        if isinstance(input_sample, List):
            # if its a list of images, stack them
            input_sample = torch.stack(input_sample, dim=0)

        augmented_tensor = prediction["augmented_tensor"]
        loss_na = self._compute_loss(
            input_tensor=input_sample,
            augmented_tensor=augmented_tensor,
            *args,
            **kwargs,
        )

        return loss_na

    def extra_repr(self) -> str:
        return (
            "\n\t target_metric={}"
            "\n\t target_value={}"
            "\n\t curriculum_learning={}"
            "\n\t alpha={}".format(
                self.perceptual_metric,
                self.target_str,
                self.curriculumn_learning,
                self.alpha,
            )
        )


def linear_curriculum(start: int, end: int, period: int) -> Tensor:
    """This function implements linear curriculum

    Args:
        start: the starting value for the set of points
        end: the ending value for the set of points
        period: size of the constructed tensor

    Returns:
        A float tensor of length period
    """
    return torch.linspace(start=start, end=end, steps=period + 1, dtype=torch.float)


def cosine_curriculum(start: int, end: int, period: int) -> Tensor:
    """This function implements cosine curriculum
    Args:
        start: the starting value for the set of points
        end: the ending value for the set of points
        period: size of the constructed tensor

    Returns:
        A float tensor of length period
    """

    curr = [
        end + 0.5 * (start - end) * (1 + math.cos(math.pi * i / (period + 1)))
        for i in range(period + 1)
    ]

    curr = torch.tensor(curr, dtype=torch.float)
    return curr


CURRICULUM_METHOD = {
    "linear": linear_curriculum,
    "cosine": cosine_curriculum,
}
