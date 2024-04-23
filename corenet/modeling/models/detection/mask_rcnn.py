#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Faster and Mask-RCNN related imports
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops import MultiScaleRoIAlign

from corenet.modeling import parameter_list
from corenet.modeling.layers import ConvLayer2d, Identity
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.detection import DetectionPredTuple
from corenet.modeling.models.detection.base_detection import BaseDetection
from corenet.modeling.models.detection.utils.rcnn_utils import (
    FastRCNNConvFCHead,
    FastRCNNPredictor,
    MaskRCNNHeads,
    MaskRCNNPredictor,
    RPNHead,
)
from corenet.utils import logger


class MaskRCNNEncoder(nn.Module):
    def __init__(
        self,
        opts: argparse.Namespace,
        encoder: BaseImageEncoder,
        output_strides: List,
        projection_channels: int,
        encoder_lr_multiplier: Optional[float] = 1.0,
        *args,
        **kwargs,
    ) -> None:
        use_fpn = not getattr(opts, "model.detection.mask_rcnn.disable_fpn", False)
        super().__init__()
        # set classifier and exp layers to Identity
        encoder.conv_1x1_exp = Identity()
        encoder.classifier = Identity()

        # add projection layers that projects encoder feature maps to `projection_channels`
        backbone_proj_layers = nn.ModuleDict()
        self.backbone_output_strides = sorted(
            list({4, 8, 16, 32}.intersection(output_strides))
        )
        model_config = encoder.model_conf_dict
        self.backbone_map = {}
        fpn_proj_layers = nn.ModuleDict() if use_fpn else None
        for os in self.backbone_output_strides:
            if os == 4:
                in_channels = model_config["layer2"]["out"]
                backbone_os_str = "out_l2"
            elif os == 8:
                in_channels = model_config["layer3"]["out"]
                backbone_os_str = "out_l3"
            elif os == 16:
                in_channels = model_config["layer4"]["out"]
                backbone_os_str = "out_l4"
            elif os == 32:
                in_channels = model_config["layer5"]["out"]
                backbone_os_str = "out_l5"
            else:
                raise NotImplementedError

            conv_layer = ConvLayer2d(
                opts=opts,
                in_channels=in_channels,
                out_channels=projection_channels,
                kernel_size=1,
                use_norm=True,
                use_act=False,
            )
            backbone_proj_layers.add_module(str(os), conv_layer)
            self.backbone_map[os] = backbone_os_str

            if use_fpn:
                fpn_layer = ConvLayer2d(
                    opts=opts,
                    in_channels=projection_channels,
                    out_channels=projection_channels,
                    kernel_size=3,
                    use_norm=True,
                    use_act=False,
                )
                fpn_proj_layers.add_module(str(os), fpn_layer)

        # add extra layers if desired output stride is greater than 32.
        extra_layers = nn.ModuleDict()
        extra_layer_os = sorted(
            list((set(self.backbone_output_strides) ^ set(output_strides)))
        )
        for os in extra_layer_os:
            conv_layer = ConvLayer2d(
                opts=opts,
                in_channels=projection_channels,
                out_channels=projection_channels,
                kernel_size=3,
                stride=2,
                use_norm=True,
                use_act=False,
            )
            extra_layers.add_module(str(os), conv_layer)
        self.encoder = encoder
        self.backbone_proj_layers = backbone_proj_layers
        self.fpn_proj_layers = fpn_proj_layers
        self.use_fpn = use_fpn
        self.extra_layers = extra_layers
        self.out_channels = projection_channels
        self.augmented_tensor = None
        self.encoder_lr_multiplier = encoder_lr_multiplier

    def get_augmented_tensor(self) -> Tensor:
        return self.augmented_tensor

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # extract features from the backbone network
        enc_end_points: Dict = self.encoder.extract_end_points_all(x)

        self.augmented_tensor = enc_end_points.pop("augmented_tensor", None)

        outputs_backbone: Dict = {}
        # project backbone features
        for os, enc_key_name in self.backbone_map.items():
            x_proj = self.backbone_proj_layers[str(os)](
                enc_end_points.pop(enc_key_name)
            )
            outputs_backbone[f"{os}"] = x_proj

        if self.fpn_proj_layers:
            # FPN
            last_os = self.backbone_output_strides[-1]
            prev_fm = outputs_backbone[f"{last_os}"]
            prev_fm = self.fpn_proj_layers[f"{last_os}"](prev_fm)
            for os in self.backbone_output_strides[:-1][::-1]:
                curr_fm = outputs_backbone[f"{os}"]
                feat_shape = curr_fm.shape[-2:]
                inner_top_down = F.interpolate(prev_fm, size=feat_shape, mode="nearest")
                prev_fm = self.fpn_proj_layers[f"{os}"](curr_fm + inner_top_down)
                outputs_backbone[f"{os}"] = prev_fm

        if self.extra_layers:
            prev_os = self.backbone_output_strides[-1]
            for os, extra_layer in self.extra_layers.items():
                x_proj = extra_layer(outputs_backbone[f"{prev_os}"])
                outputs_backbone[f"{os}"] = x_proj
                prev_os = os
        return outputs_backbone

    def get_trainable_parameters(
        self,
        weight_decay: float = 0.0,
        no_decay_bn_filter_bias: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[List, List]:
        # We need to pop the module name. Otherwise, we may pass two
        # variables with the same name to get_trainable_parameters function
        module_name = kwargs.pop("module_name", "")

        """Returns a list of trainable parameters"""
        all_params = []
        all_params_lr = []

        # encoder parameters
        if (
            hasattr(self.encoder, "enable_layer_wise_lr_decay")
            and self.encoder.enable_layer_wise_lr_decay
        ):
            (
                backbone_param_list,
                backbone_lr_list,
            ) = self.encoder.get_trainable_parameters(
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "encoder.",
                *args,
                **kwargs,
            )

            all_params.extend(backbone_param_list)

            # Scale encoder LR, if applicable
            if self.encoder_lr_multiplier != 1.0:
                backbone_lr_list = [
                    (lr * self.encoder_lr_multiplier) for lr in backbone_lr_list
                ]

            all_params_lr.extend(backbone_lr_list)
        else:
            backbone_param_list = parameter_list(
                named_parameters=self.encoder.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "encoder.",
                *args,
                **kwargs,
            )

            all_params.extend(backbone_param_list)

            all_params_lr.extend(
                [self.encoder_lr_multiplier] * len(backbone_param_list)
            )

        if self.backbone_proj_layers:
            # projection layer parameters
            projection_param_list = parameter_list(
                named_parameters=self.backbone_proj_layers.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "backbone_proj_layers.",
                *args,
                **kwargs,
            )

            all_params.extend(projection_param_list)

            all_params_lr.extend([1.0] * len(projection_param_list))

        if self.fpn_proj_layers:
            # projection layer parameters
            fpn_projection_param_list = parameter_list(
                named_parameters=self.fpn_proj_layers.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "fpn_proj_layers.",
                *args,
                **kwargs,
            )

            all_params.extend(fpn_projection_param_list)

            all_params_lr.extend([1.0] * len(fpn_projection_param_list))

        if self.extra_layers:
            # extra layer parameters
            extra_layer_param_list = parameter_list(
                named_parameters=self.extra_layers.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "extra_layers.",
                *args,
                **kwargs,
            )

            all_params.extend(extra_layer_param_list)

            all_params_lr.extend([1.0] * len(extra_layer_param_list))
        return all_params, all_params_lr

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the activation checkpointing module in the encoder."""
        return self.encoder.get_activation_checkpoint_submodule_class()


@MODEL_REGISTRY.register(name="mask_rcnn", type="detection")
class MaskRCNNDetector(BaseDetection):
    """This class implements a `Mask RCNN style object detector <https://arxiv.org/abs/1703.06870>`

    Args:
        opts: command-line arguments
        encoder (BaseImageEncoder): Encoder network (e.g., ResNet or MobileViT)
    """

    def __init__(self, opts, encoder: BaseImageEncoder, *args, **kwargs) -> None:
        super().__init__(opts, encoder, *args, **kwargs)
        default_norm = self.set_norm_layer_opts()

        output_strides = getattr(
            opts, "model.detection.mask_rcnn.output_strides", [4, 8, 16, 32, 64]
        )
        if len(output_strides) == 0:
            logger.error(
                "Please specify output strides for extracting backbone feature maps "
                "using --model.detection.mask-rcnn.output-strides"
            )
        output_strides = sorted(output_strides)
        projection_channels = getattr(
            opts, "model.detection.mask_rcnn.backbone_projection_channels", 256
        )

        # anchor sizes and aspect ratios
        anchor_sizes = getattr(
            opts, "model.detection.mask_rcnn.anchor_sizes", [32, 64, 128, 256, 512]
        )
        # convert to a tuples
        if anchor_sizes is None:
            logger.error("Anchor sizes can't be None")
        elif len(anchor_sizes) != len(output_strides):
            logger.error(
                "Number of anchor sizes should be the same as the output stride. Got: {} and {}".format(
                    anchor_sizes, output_strides
                )
            )
        elif isinstance(anchor_sizes, List) and isinstance(anchor_sizes[0], List):
            # anchor sizes is a list of list. Convert to tuple
            anchor_sizes = tuple([tuple(a_size) for a_size in anchor_sizes])
        elif isinstance(anchor_sizes, List) and isinstance(anchor_sizes[0], int):
            # anchor sizes is a list of integers. Convert to tuple
            anchor_sizes = tuple([(a_size,) for a_size in anchor_sizes])
        else:
            raise NotImplementedError

        aspect_ratios = getattr(
            opts, "model.detection.mask_rcnn.aspect_ratio", [0.5, 1.0, 2.0]
        )  # ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        if aspect_ratios is None:
            logger.error("Aspect ratios can't be None")
        elif isinstance(aspect_ratios, (int, float)):
            aspect_ratios = ((aspect_ratios,),) * len(anchor_sizes)
        elif isinstance(aspect_ratios, List):
            aspect_ratios = (tuple(aspect_ratios),) * len(anchor_sizes)
        else:
            raise NotImplementedError

        # feature map size for the bbox head
        box_fm_size = getattr(opts, "model.detection.mask_rcnn.bbox_head_fm_size", 7)
        mask_fm_size = getattr(opts, "model.detection.mask_rcnn.mask_head_fm_size", 14)

        # set-up the backbone
        backbone_lr_multiplier = getattr(
            opts, "model.detection.mask_rcnn.backbone_lr_multiplier"
        )
        backbone = MaskRCNNEncoder(
            opts,
            encoder=encoder,
            output_strides=output_strides,
            projection_channels=projection_channels,
            encoder_lr_multiplier=backbone_lr_multiplier,
        )

        # create RPN anchor generator
        rpn_anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )

        # create RPN Head
        rpn_head = RPNHead(
            opts=opts,
            in_channels=projection_channels,
            num_anchors=rpn_anchor_generator.num_anchors_per_location()[0],
            conv_depth=2,
        )

        # box related parameters
        representation_size = getattr(
            opts, "model.detection.mask_rcnn.representation_size", 1024
        )
        output_strides_str = [str(os) for os in output_strides]
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=output_strides_str, output_size=box_fm_size, sampling_ratio=2
        )

        box_fm_size_conv_layer = getattr(
            opts, "model.detection.mask_rcnn.box_fm_size_conv_layer", [256] * 4
        )
        box_head = FastRCNNConvFCHead(
            opts=opts,
            input_size=(projection_channels, box_fm_size, box_fm_size),
            conv_layers=box_fm_size_conv_layer,
            fc_layers=[representation_size],
        )

        box_predictor = FastRCNNPredictor(
            in_channels=representation_size, num_classes=self.n_detection_classes
        )

        # mask related parameters
        mask_fm_size_conv_layer = getattr(
            opts, "model.detection.mask_rcnn.mask_fm_size_conv_layer", [256] * 4
        )
        mask_dilation = getattr(opts, "model.detection.mask_rcnn.mask_dilation", 1)
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=output_strides_str, output_size=mask_fm_size, sampling_ratio=2
        )

        mask_dilation = mask_dilation
        mask_head = MaskRCNNHeads(
            opts=opts,
            in_channels=projection_channels,
            layers=mask_fm_size_conv_layer,
            dilation=mask_dilation,
        )

        mask_predictor = MaskRCNNPredictor(
            opts=opts,
            in_channels=mask_fm_size_conv_layer[-1],
            dim_reduced=256,
            num_classes=self.n_detection_classes,
        )

        # RPN and box detection related hyper-parameters
        rpn_pre_nms_top_n_train = getattr(
            opts, "model.detection.mask_rcnn.rpn_pre_nms_top_n_train", 2000
        )
        rpn_pre_nms_top_n_test = getattr(
            opts, "model.detection.mask_rcnn.rpn_pre_nms_top_n_test", 1000
        )
        rpn_post_nms_top_n_train = getattr(
            opts, "model.detection.mask_rcnn.rpn_post_nms_top_n_train", 2000
        )
        rpn_post_nms_top_n_test = getattr(
            opts, "model.detection.mask_rcnn.rpn_post_nms_top_n_test", 1000
        )
        rpn_nms_thresh = getattr(opts, "model.detection.mask_rcnn.rpn_nms_thresh", 0.7)
        rpn_fg_iou_thresh = getattr(
            opts, "model.detection.mask_rcnn.rpn_fg_iou_thresh", 0.7
        )
        rpn_bg_iou_thresh = getattr(
            opts, "model.detection.mask_rcnn.rpn_bg_iou_thresh", 0.3
        )
        rpn_batch_size_per_image = getattr(
            opts, "model.detection.mask_rcnn.rpn_batch_size_per_image", 256
        )
        rpn_positive_fraction = getattr(
            opts, "model.detection.mask_rcnn.rpn_positive_fraction", 0.5
        )
        rpn_score_thresh = getattr(
            opts, "model.detection.mask_rcnn.rpn_score_thresh", 0.0
        )

        box_score_thresh = getattr(
            opts, "model.detection.mask_rcnn.box_score_thresh", 0.05
        )
        box_nms_thresh = getattr(opts, "model.detection.mask_rcnn.box_nms_thresh", 0.5)
        box_detections_per_img = getattr(
            opts, "model.detection.mask_rcnn.box_detections_per_img", 100
        )
        box_fg_iou_thresh = getattr(
            opts, "model.detection.mask_rcnn.box_fg_iou_thresh", 0.5
        )
        box_bg_iou_thresh = getattr(
            opts, "model.detection.mask_rcnn.box_bg_iou_thresh", 0.5
        )
        box_batch_size_per_image = getattr(
            opts, "model.detection.mask_rcnn.box_batch_size_per_image", 512
        )
        box_positive_fraction = getattr(
            opts, "model.detection.mask_rcnn.box_positive_fraction", 0.25
        )

        # kwargs = {"_skip_resize": True}
        self.model = MaskRCNN(
            backbone=backbone,
            # we don't use mean-std normalization
            image_mean=[0.0] * 3,
            image_std=[1.0] * 3,
            # RPN parameters
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
            rpn_batch_size_per_image=rpn_batch_size_per_image,
            rpn_positive_fraction=rpn_positive_fraction,
            rpn_score_thresh=rpn_score_thresh,
            # Box parameters
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_bg_iou_thresh,
            box_batch_size_per_image=box_batch_size_per_image,
            box_positive_fraction=box_positive_fraction,
            bbox_reg_weights=None,
            box_predictor=box_predictor,
            # Mask parameters
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
            # **kwargs
        )

        del self.encoder

        self.reset_norm_layer_opts(default_norm=default_norm)
        self.update_layer_norm_eps()

    def update_layer_norm_eps(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.eps = 1e-6

    def set_norm_layer_opts(self):
        mask_rcnn_norm_layer = getattr(
            self.opts, "model.detection.mask_rcnn.norm_layer", None
        )
        if mask_rcnn_norm_layer is None:
            logger.error("Please specify norm layer")

        default_norm = getattr(self.opts, "model.normalization.name", None)
        setattr(self.opts, "model.normalization.name", mask_rcnn_norm_layer)
        return default_norm

    def reset_norm_layer_opts(self, default_norm):
        setattr(self.opts, "model.normalization.name", default_norm)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--model.detection.mask-rcnn.backbone-projection-channels",
            type=int,
            default=256,
            help="Projection channels for the encoder in Mask-RCNN",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.backbone-lr-multiplier",
            type=float,
            default=1.0,
            help="LR multiplier for MASK RCNN head",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.output-strides",
            type=int,
            nargs="+",
            default=[4, 8, 16, 32, 64],
            help="Extract backbone feature maps from these output strides. "
            "If output stride is greater than 32, extra layers are added.",
        )
        group.add_argument(
            "--model.detection.mask-rcnn.anchor-sizes",
            type=int,
            nargs="+",
            action="append",
            default=[32, 64, 128, 256, 512],
            help="Anchor sizes at each output stride",
        )
        group.add_argument(
            "--model.detection.mask-rcnn.aspect-ratio",
            type=float,
            nargs="+",
            default=[0.5, 1.0, 2.0],
            help="Aspect ratios. These are the same for all feature maps",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.bbox-head-fm-size",
            type=int,
            default=7,
            help="Feature map size for the box head",
        )
        group.add_argument(
            "--model.detection.mask-rcnn.mask-head-fm-size",
            type=int,
            default=14,
            help="Feature map size for the max head",
        )
        group.add_argument(
            "--model.detection.mask-rcnn.representation-size",
            type=int,
            default=1024,
            help="Size of the intermediate representation in Mask RCNN",
        )
        # box_fm_size_conv_layer = getattr(opts, "", [256] * 4)
        group.add_argument(
            "--model.detection.mask-rcnn.box-fm-size-conv-layer",
            type=int,
            nargs="+",
            default=[256] * 4,
            help="Feature dim of each Convolution layer in the Faster RCNN head. Defaults to [256, 256, 256, 256]",
        )
        group.add_argument(
            "--model.detection.mask-rcnn.mask-fm-size-conv-layer",
            type=int,
            nargs="+",
            default=[256] * 4,
            help="Feature dim of each Convolution layer in the Mask RCNN head. Defaults to [256, 256, 256, 256]",
        )
        group.add_argument(
            "--model.detection.mask-rcnn.mask-dilation",
            type=int,
            default=1,
            help="Dilation rate in Mask RCNN head. Defaults to 1",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-pre-nms-top-n-train",
            type=int,
            default=2000,
            help="Number of proposals to keep before applying NMS during training",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-pre-nms-top-n-test",
            type=int,
            default=1000,
            help="Number of proposals to keep before applying NMS during test",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-post-nms-top-n-train",
            type=int,
            default=2000,
            help="Number of proposals to keep after applying NMS during training",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-post-nms-top-n-test",
            type=int,
            default=1000,
            help="Number of proposals to keep after applying NMS during test",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-nms-thresh",
            type=float,
            default=0.7,
            help="NMS threshold used for postprocessing the RPN proposals",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-fg-iou-thresh",
            type=float,
            default=0.7,
            help="minimum IoU between the anchor and the GT box so that they can be "
            "considered as positive during training of the RPN.",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-bg-iou-thresh",
            type=float,
            default=0.7,
            help="minimum IoU between the anchor and the GT box so that they can be "
            "considered as negative during training of the RPN.",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-batch-size-per-image",
            type=int,
            default=256,
            help="Number of anchors that are sampled during training of the RPN for computing the loss",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-positive-fraction",
            type=float,
            default=0.5,
            help="Proportion of positive anchors in a mini-batch during training of the RPN",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.rpn-score-thresh",
            type=float,
            default=0.0,
            help="During inference, only return proposals with a classification score greater than rpn_score_thresh",
        )

        #
        group.add_argument(
            "--model.detection.mask-rcnn.box-score-thresh",
            type=float,
            default=0.05,
            help="During inference, only return proposals with a classification score greater than box_score_thresh",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.box-nms-thresh",
            type=float,
            default=0.5,
            help="During inference, NMS threshold for the prediction head.",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.box-detections-per-img",
            type=int,
            default=100,
            help="Maximum number of detections per image, for all classes",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.box-fg-iou-thresh",
            type=float,
            default=0.5,
            help="Minimum IoU between the proposals and the GT box so that they can be considered as "
            "positive during training of the classification head",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.box-bg-iou-thresh",
            type=float,
            default=0.5,
            help="Minimum IoU between the proposals and the GT box so that they can be considered as "
            "negative during training of the classification head",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.box-batch-size-per-image",
            type=int,
            default=512,
            help="Number of proposals that are sampled during training of the classification head",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.box-positive-fraction",
            type=float,
            default=0.25,
            help="Proportion of positive proposals in a mini-batch during training of the classification head",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.norm-layer",
            type=str,
            default=None,
            help="Mask RCNN Norm layer",
        )

        group.add_argument(
            "--model.detection.mask-rcnn.disable-fpn",
            action="store_true",
            help="Do not use FPN",
        )
        return parser

    def reset_generalized_rcnn_transform(self, height, width):
        self.model.transform.fixed_size = (width, height)

    def get_trainable_parameters(
        self,
        weight_decay: float = 0.0,
        no_decay_bn_filter_bias: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[List, List]:

        all_params = []
        all_params_lr = []

        # backbone parameters
        if hasattr(self.model.backbone, "get_trainable_parameters"):
            (
                backbone_params,
                backbone_lrs,
            ) = self.model.backbone.get_trainable_parameters(
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name="model.backbone.",
            )
            all_params.extend(backbone_params)
            all_params_lr.extend(backbone_lrs)
        else:
            logger.error(
                "Backbone model must implement get_trainable_parameters function."
            )

        # rpn parameters
        rpn_param_list = parameter_list(
            named_parameters=self.model.rpn.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name="model.rpn.",
            *args,
            **kwargs,
        )

        all_params.extend(rpn_param_list)
        all_params_lr.extend([1.0] * len(rpn_param_list))

        # ROI head params
        roi_param_list = parameter_list(
            named_parameters=self.model.roi_heads.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name="model.roi_heads.",
            *args,
            **kwargs,
        )

        all_params.extend(roi_param_list)
        all_params_lr.extend([1.0] * len(roi_param_list))

        return all_params, all_params_lr

    def forward(
        self, x: Dict, *args, **kwargs
    ) -> Union[Tuple[Tensor, ...], Tuple[Any, ...], Dict]:

        if isinstance(x, Dict):
            input_tensor = x["image"]
            input_labels = x["label"]
        else:
            raise NotImplementedError(
                "Input to MaskRCNN should be a Dict of List of Tensors"
            )

        assert isinstance(input_tensor, List)
        assert isinstance(input_labels, List)

        in_channels, in_height, in_width = input_tensor[0].shape

        self.reset_generalized_rcnn_transform(height=in_height, width=in_width)

        # The mask rcnn model expects labels, since it computes the loss.
        outputs = self.model(input_tensor, targets=input_labels)

        if not self.training:
            detections = []
            for i, elem in enumerate(outputs):
                # We must normalize by image size, since this is what the downstream
                # evaluator expects.
                elem["boxes"][:, 0::2] /= input_tensor[i].shape[2]
                elem["boxes"][:, 1::2] /= input_tensor[i].shape[1]

                # predicted masks are in [N, 1, H, W] format
                # for evaluation, we need them in [N, H, W] format
                masks = elem["masks"]
                # [N, 1, H, W] --> [N, H, W]
                masks = masks.squeeze(1)

                elem_detections = DetectionPredTuple(
                    labels=elem["labels"],
                    scores=elem["scores"],
                    boxes=elem["boxes"],
                    masks=masks,
                )
                detections.append(elem_detections)
            return {"detections": detections}

        if hasattr(self.model.backbone, "get_augmented_tensor"):
            outputs["augmented_tensor"] = self.model.backbone.get_augmented_tensor()

        return outputs

    @torch.no_grad()
    def predict(self, x: Tensor, *args, **kwargs) -> DetectionPredTuple:
        """Predict the bounding boxes given an image tensor"""
        assert isinstance(x, Tensor) and x.ndim == 4, "Expected 4D tensor as an input"

        bsz, channels, in_height, in_width = x.shape
        if bsz != 1:
            logger.error(
                "Prediction is supported with a batch size of 1 in {}".format(
                    self.__class__.__name__
                )
            )

        self.reset_generalized_rcnn_transform(height=in_height, width=in_width)

        outputs = self.model(x)

        if isinstance(outputs, List) and len(outputs) == 1:
            outputs = outputs[0]

        if isinstance(outputs, Dict) and {"boxes", "labels", "scores"}.issubset(
            outputs.keys()
        ):
            # resize the boxes
            outputs["boxes"][:, 0::2] /= in_width
            outputs["boxes"][:, 1::2] /= in_height

            # predicted masks are in [N, 1, H, W] format
            # for evaluation, we need them in [N, H, W] format
            masks = outputs["masks"]
            # [N, 1, H, W] --> [N, H, W]
            masks = masks.squeeze(1)

            detections = DetectionPredTuple(
                labels=outputs["labels"],
                scores=outputs["scores"],
                boxes=outputs["boxes"],
                masks=masks,
            )
            return detections
        else:
            logger.error(
                "Output should be a dict with boxes, scores, and labels as keys. Got: {}".format(
                    type(outputs)
                )
            )

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes."""
        img_channels = 3
        height = 320
        width = 320
        n_classes = 80

        # GT boxes have the same shape as anchors. So, we use anchors as GT boxes
        n_boxes = 1

        gt_boxes = torch.tensor([2, 20, 3, 40]).reshape(-1, 4).float()
        gt_box_labels = torch.randint(
            low=0,
            high=n_classes,
            size=(n_boxes,),
            dtype=torch.long,
        )

        img_tensor = torch.randn(img_channels, height, width, dtype=torch.float)
        labels = {
            "box_labels": gt_box_labels,
            "box_coordinates": gt_boxes,
        }

        return {
            "samples": {
                "image": [img_tensor] * batch_size,
                "label": [
                    {
                        "labels": gt_box_labels,
                        "boxes": gt_boxes,
                        "masks": torch.zeros(1, height, width, dtype=torch.long),
                    }
                ]
                * batch_size,
            },
            "targets": labels,
        }

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the activation checkpointing module class in the encoder."""
        return self.model.backbone.get_activation_checkpoint_submodule_class()
