#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Mapping, Optional, Tuple, Union

from torch import Tensor
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.segmentation.base_segmentation_criteria import (
    BaseSegmentationCriteria,
)
from corenet.loss_fn.utils.class_weighting import compute_class_weights
from corenet.utils import logger


@LOSS_REGISTRY.register(name="cross_entropy", type="segmentation")
class SegCrossEntropy(BaseSegmentationCriteria):
    """Cross entropy loss for the task of semantic segmentation.

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.ignore_idx = getattr(opts, "loss.segmentation.cross_entropy.ignore_index")
        self.use_class_wts = getattr(
            opts, "loss.segmentation.cross_entropy.class_weights"
        )
        self.aux_wt = getattr(opts, "loss.segmentation.cross_entropy.aux_weight")
        self.label_smoothing = getattr(
            opts, "loss.segmentation.cross_entropy.label_smoothing"
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != SegCrossEntropy:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.segmentation.cross-entropy.class-weights",
            action="store_true",
            default=False,
            help=f"Use class weights in {cls.__name__}. Defaults to False.",
        )
        group.add_argument(
            "--loss.segmentation.cross-entropy.ignore-index",
            type=int,
            default=-1,
            help=f"Target value that is ignored and does not contribute to "
            f"the input gradient in {cls.__name__}. Defaults to -1.",
        )
        group.add_argument(
            "--loss.segmentation.cross-entropy.aux-weight",
            type=float,
            # This is a typical value used in segmentation networks for auxiliary loss.
            # See PSPNet paper for instance: https://arxiv.org/abs/1612.01105
            default=0.4,
            help="Weight of auxiliary segmentation loss. Defaults to 0.4.",
        )
        group.add_argument(
            "--loss.segmentation.cross-entropy.label-smoothing",
            type=float,
            default=0.0,
            help=f"Specifies the amount of smoothing when computing the loss in {cls.__name__}, "
            f"where 0.0 means no smoothing. Defaults to 0.0.",
        )

        return parser

    def _compute_loss(
        self, pred_mask: Tensor, target_mask: Tensor, weight: Optional[Tensor] = None
    ) -> Tensor:
        """Computes the cross-entropy loss

        Args:
            pred_mask: Predicted segmentation mask
            target_mask: Target segmentation mask whose values are in the range `[0, C)`,
                where :math:`C` is the number of classes
            weight: class weights for handling class imbalancing.

        Shapes:
            pred_mask: Shape is [Batch size, Channels, Height, Width]
            target_mask: Shape is [Batch size, Height, Width]
            weight: Shape is [C]

        Returns:
            A scalar loss value
        """

        b, c, x_h, x_w = pred_mask.shape
        b, y_h, y_w = target_mask.shape

        # use label smoothing during training
        label_smoothing = self.label_smoothing if self.training else 0.0

        if x_h != y_h or x_w != y_w:
            # if predicting mask shape is not the same as target mask, resize it using
            # bilinear interpolation.
            pred_mask = F.interpolate(
                pred_mask, size=(y_h, y_w), mode="bilinear", align_corners=True
            )

        loss = F.cross_entropy(
            input=pred_mask,
            target=target_mask,
            weight=weight,
            ignore_index=self.ignore_idx,
            label_smoothing=label_smoothing,
        )

        return loss

    def _forward_seg(
        self,
        prediction: Union[Tensor, Tuple[Tensor, Tensor]],
        target: Tensor,
        *args,
        **kwargs,
    ) -> Mapping[str, Tensor]:
        """Computes the segmentation loss. If prediction is a Tuple[Tensor, Tensor], then weighted sum of CE losses is
        computed.

        Args:
            prediction: Output of segmentation model. If auxiliary branch is enabled, then prediction is
            a Tuple[Tensor, Tensor]. Otherwise, it is a Tensor.
            target: Ground truth segmentation mask.

        Shapes:
            prediction:
                * When prediction is a Tensor, then shape is [Batch size, Channels, Height, Width]
                * When prediction is a Tuple[Tensor, Tensor], then shape of one tensor is [Batch size, Channels, Height, Width]
                    while the other is [Batch size, Channels, Height / O, Width/ O]
                    where O is output stride of feature map (typically 4).
            target: Shape is [Batch size, Height, Width]

        Returns:
            Mapping of the form (string: scalar value) is returned with total_loss as mandatory and
            (seg_loss, aux_loss) as optional keys. total_loss is weighted sum of seg_loss and aux_loss (when applicable).

        ...note:
            When shape of prediction and target are not the same, prediction is resized using bilinear interpolation to
            match the size of target.
        """

        aux_out = None
        if isinstance(prediction, Tuple) and len(prediction) == 2:
            mask, aux_out = prediction
            assert isinstance(mask, Tensor)
            assert isinstance(aux_out, Tensor)
        elif isinstance(prediction, Tensor):
            mask = prediction
            assert isinstance(mask, Tensor)
        else:
            raise NotImplementedError(
                "For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor"
            )

        cls_wts = None
        if self.training:
            if self.use_class_wts:
                n_classes = mask.size(1)  # Mask is of shape B x C x H x W
                cls_wts = compute_class_weights(target=target, n_classes=n_classes)
            seg_loss = self._compute_loss(
                pred_mask=mask, target_mask=target, weight=cls_wts
            )

            if aux_out is not None:
                loss_aux = self._compute_loss(
                    pred_mask=aux_out, target_mask=target, weight=cls_wts
                )
                total_loss = seg_loss + (self.aux_wt * loss_aux)
                return {
                    "total_loss": total_loss,
                    "seg_loss": seg_loss,
                    "aux_loss": (self.aux_wt * loss_aux),
                }
            return {"total_loss": seg_loss}
        else:
            # during validation, we do not compute aux. loss
            seg_loss = self._compute_loss(
                pred_mask=mask, target_mask=target, weight=None
            )
            return {"total_loss": seg_loss}

    def forward(
        self,
        input_sample: Any,
        prediction: Union[
            Mapping[str, Union[Tensor, Tuple[Tensor, Tensor]]],
            Tensor,
            Tuple[Tensor, Tensor],
        ],
        target: Tensor,
        *args,
        **kwargs,
    ) -> Mapping[str, Tensor]:
        """Compute CE segmentation loss

        Args:
            input_sample: Input image tensor to model.
            prediction: Output of model. It can be a
                * Tensor
                * Tuple[Tensor, Tensor]
                * Mapping[segmentation_output, Tensor]
                * Mapping[segmentation_output, Tuple[Tensor, Tensor]], where segmentation_output is a required key.
            target: Target label tensor containing values in the range `[0, C)`, where :math:`C` is the number of classes

        Shapes:
            input_sample: This loss function does not care about this argument.
            prediction:
                * When prediction is a Tensor, then shape is [Batch size, C, Height, Width]
                * When prediction is a Tuple[Tensor, Tensor], then shape of one tensor is [Batch size, C, Height, Width]
                    while the other is [Batch size, C, Height / O, Width/ O]
                    where O is the output stride of feature map (typically 4).
                * When prediction is a dictionary, then the shape of prediction["segmentation_output"] should
                    be the same as described in above steps (depending on type).
            target: The shape of target tensor is [Batch size, Height, Width]

        Returns:
            Mapping of the form (string: scalar value) is returned with total_loss as mandatory and
            (seg_loss, aux_loss) as optional keys. total_loss is weighted sum of seg_loss and aux_loss (when applicable).
        """

        if isinstance(prediction, (Tuple, Tensor)):
            return self._forward_seg(
                prediction=prediction, target=target, *args, **kwargs
            )
        elif isinstance(prediction, Mapping):
            if "segmentation_output" not in prediction:
                logger.error(
                    f"segmentation_output is a mandatory key in prediction when"
                    f"type of prediction is Dict. Got: {prediction.keys()}"
                )

            seg_loss = self._forward_seg(
                prediction=prediction["segmentation_output"],
                target=target,
                *args,
                **kwargs,
            )

            return seg_loss
        else:
            logger.error(
                f"Prediction should be either a Tensor or Tuple[Tensor, Tensor] "
                f"or Dictionary[str, Tensor] in {self.__class__.__name__}. Got: {type(prediction)}"
            )

    def extra_repr(self) -> str:
        return (
            f"\n\t ignore_idx={self.ignore_idx}"
            f"\n\t class_weighting={self.use_class_wts}"
            f"\n\t label_smoothing={self.label_smoothing}"
            f"\n\t aux_weight={self.aux_wt}"
        )
