#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Optional, Tuple, Union

from torch import Tensor

from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel, get_model
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.segmentation.base_seg import (
    BaseSegmentation,
    set_model_specific_opts_before_model_building,
    unset_model_specific_opts_after_model_building,
)


@MODEL_REGISTRY.register(name="encoder_decoder", type="segmentation")
class SegEncoderDecoder(BaseSegmentation):
    """
    This class defines a encoder-decoder architecture for the task of semantic segmentation. Different segmentation
    heads (e.g., PSPNet and DeepLabv3) can be used

    Args:
        opts: command-line arguments
        encoder (BaseImageEncoder): Backbone network (e.g., MobileViT or ResNet)
    """

    def __init__(
        self, opts, encoder: BaseImageEncoder, seg_head, *args, **kwargs
    ) -> None:
        super().__init__(opts=opts, encoder=encoder)

        # delete layers that are not required in segmentation network
        self.encoder.classifier = None
        use_l5_exp = getattr(opts, "model.segmentation.use_level5_exp")
        if not use_l5_exp:
            self.encoder.conv_1x1_exp = None

        self.maybe_seg_norm_layer()
        self.seg_head = seg_head
        self.use_l5_exp = use_l5_exp
        self.set_default_norm_layer()

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        """This function separates the parameters for backbone and segmentation head, so that
        different learning rates can be used for backbone and segmentation head
        """
        if getattr(self.encoder, "enable_layer_wise_lr_decay"):
            encoder_params, enc_lr_mult = self.encoder.get_trainable_parameters(
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name="encoder.",
                *args,
                **kwargs,
            )
        else:
            encoder_params, enc_lr_mult = self.encoder.get_trainable_parameters(
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name="encoder.",
                *args,
                **kwargs,
            )
        decoder_params, dec_lr_mult = self.seg_head.get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name="seg_head.",
            *args,
            **kwargs,
        )

        total_params = sum([p.numel() for p in self.parameters()])
        encoder_params_count = sum([p.numel() for p in self.encoder.parameters()])
        decoder_params_count = sum([p.numel() for p in self.seg_head.parameters()])

        assert total_params == encoder_params_count + decoder_params_count, (
            "Total network parameters are not equal to "
            "the sum of encoder and decoder. "
            "{} != {} + {}".format(
                total_params, encoder_params_count, decoder_params_count
            )
        )

        return encoder_params + decoder_params, enc_lr_mult + dec_lr_mult

    def forward(
        self, x: Tensor, *args, **kwargs
    ) -> Union[Tuple[Tensor, Tensor], Tensor, Dict]:
        enc_end_points: Dict = self.encoder.extract_end_points_all(
            x, use_l5=True, use_l5_exp=self.use_l5_exp
        )

        if "augmented_tensor" in enc_end_points:
            output_dict = {
                "augmented_tensor": enc_end_points.pop("augmented_tensor"),
                "segmentation_output": self.seg_head(
                    enc_out=enc_end_points, *args, **kwargs
                ),
            }
            return output_dict
        else:
            return self.seg_head(enc_out=enc_end_points, *args, **kwargs)

    def update_classifier(self, opts, n_classes: int) -> None:
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        if hasattr(self.seg_head, "update_classifier"):
            self.seg_head.update_classifier(opts, n_classes)

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:

        output_stride = getattr(opts, "model.segmentation.output_stride", None)
        image_encoder = get_model(
            opts,
            category="classification",
            output_stride=output_stride,
            *args,
            **kwargs,
        )

        default_opt_info = set_model_specific_opts_before_model_building(opts)
        use_l5_exp = getattr(opts, "model.segmentation.use_level5_exp")

        seg_head = get_model(
            opts=opts,
            category="segmentation_head",
            model_name=getattr(opts, "model.segmentation.seg_head"),
            enc_conf=image_encoder.model_conf_dict,
            use_l5_exp=use_l5_exp,
            *args,
            **kwargs,
        )

        seg_model = cls(opts, encoder=image_encoder, seg_head=seg_head, *args, **kwargs)

        unset_model_specific_opts_after_model_building(
            opts, default_opts_info=default_opt_info
        )

        if getattr(opts, "model.segmentation.freeze_batch_norm"):
            cls.freeze_norm_layers(opts, model=seg_model)
        return seg_model
