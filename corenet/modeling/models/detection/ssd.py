#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import argparse
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.ops import batched_nms

from corenet.modeling.anchor_generator import build_anchor_generator
from corenet.modeling.layers import AdaptiveAvgPool2d, ConvLayer2d, SeparableConv2d
from corenet.modeling.matcher_det import build_matcher
from corenet.modeling.misc.init_utils import initialize_conv_layer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.detection import DetectionPredTuple
from corenet.modeling.models.detection.base_detection import BaseDetection
from corenet.modeling.modules import SSDHead
from corenet.utils import logger
from corenet.utils.common_utils import is_coreml_conversion


@MODEL_REGISTRY.register(name="ssd", type="detection")
class SingleShotMaskDetector(BaseDetection):
    """
    This class implements a `Single Shot Object Detector <https://arxiv.org/abs/1512.02325>`_

    Args:
        opts: command-line arguments
        encoder (BaseImageEncoder): Encoder network (e.g., ResNet or MobileViT)
    """

    coordinates = 4  # 4 coordinates (x1, y1, x2, y2) or (x, y, w, h)

    def __init__(self, opts, encoder: BaseImageEncoder) -> None:

        anchor_gen_name = getattr(opts, "anchor_generator.name", None)
        if anchor_gen_name is None or anchor_gen_name != "ssd":
            logger.error("For SSD, we need --anchor-generator.name to be ssd")
        anchor_box_generator = build_anchor_generator(opts=opts)

        output_strides_aspect_ratio = anchor_box_generator.output_strides_aspect_ratio
        output_strides = list(output_strides_aspect_ratio.keys())
        anchors_aspect_ratio = list(output_strides_aspect_ratio.values())

        n_os = len(output_strides)

        if getattr(opts, "matcher.name") != "ssd":
            logger.error("For SSD, we need --matcher.name as ssd")

        super().__init__(opts=opts, encoder=encoder)

        # delete layers that are not required in detection network
        self.encoder.classifier = None
        self.encoder.conv_1x1_exp = None

        proj_channels = getattr(
            opts, "model.detection.ssd.proj_channels", [512, 256, 256, 128, 128, 64]
        )

        proj_channels = proj_channels + [128] * (n_os - len(proj_channels))

        if n_os != len(anchors_aspect_ratio) != len(proj_channels):
            logger.error(
                "SSD model requires anchors to be defined for feature maps from each output stride. Also"
                "len(anchors_aspect_ratio) == len(output_strides) == len(proj_channels). "
                "Got len(output_strides)={}, len(anchors_aspect_ratio)={}, len(proj_channels)={}."
                " Please specify correct arguments using following arguments: "
                "\n--model.detection.ssd.anchors-aspect-ratio "
                "\n--model.detection.ssd.output-strides"
                "\n--model.detection.ssd.proj-channels".format(
                    n_os, len(anchors_aspect_ratio), len(proj_channels)
                )
            )
        extra_layers = {}
        enc_channels_list = []
        in_channels = self.enc_l5_channels

        extra_proj_list = [256] * (len(output_strides) - len(proj_channels))
        proj_channels = proj_channels + extra_proj_list
        for idx, os in enumerate(output_strides):
            out_channels = proj_channels[idx]
            if os == 8:
                enc_channels_list.append(self.enc_l3_channels)
            elif os == 16:
                enc_channels_list.append(self.enc_l4_channels)
            elif os == 32:
                enc_channels_list.append(self.enc_l5_channels)
            elif os > 32 and os != -1:
                extra_layers["os_{}".format(os)] = SeparableConv2d(
                    opts=opts,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    use_act=True,
                    use_norm=True,
                    stride=2,
                )
                enc_channels_list.append(out_channels)
                in_channels = out_channels
            elif os == -1:
                extra_layers["os_{}".format(os)] = nn.Sequential(
                    AdaptiveAvgPool2d(output_size=1),
                    ConvLayer2d(
                        opts=opts,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        use_act=True,
                        use_norm=False,
                    ),
                )
                enc_channels_list.append(out_channels)
                in_channels = out_channels
            else:
                raise NotImplementedError
        self.extra_layers = None if not extra_layers else nn.ModuleDict(extra_layers)
        if self.extra_layers is not None:
            self.reset_layers(module=self.extra_layers)

        self.fpn = None
        if getattr(opts, "model.detection.ssd.use_fpn", False):
            from corenet.modeling.modules import FeaturePyramidNetwork

            fpn_channels = getattr(opts, "model.detection.ssd.fpn_out_channels", 256)
            self.fpn = FeaturePyramidNetwork(
                opts=opts,
                in_channels=enc_channels_list,
                output_strides=output_strides,
                out_channels=fpn_channels,
            )
            # update the enc_channels_list
            enc_channels_list = [fpn_channels] * len(output_strides)
            # for FPN, we do not need to do projections
            proj_channels = enc_channels_list

        # Anchor box related parameters
        self.conf_threshold = getattr(opts, "model.detection.ssd.conf_threshold", 0.01)
        self.nms_threshold = getattr(opts, "model.detection.ssd.nms_iou_threshold", 0.5)
        self.top_k = getattr(opts, "model.detection.ssd.top_k", 400)
        self.objects_per_image = getattr(
            opts, "model.detection.ssd.objects_per_image", 200
        )

        self.anchor_box_generator = anchor_box_generator

        anchors_aspect_ratio = self.anchor_box_generator.num_anchors_per_os()

        # Create SSD detection and classification heads
        anchor_steps = self.anchor_box_generator.step

        self.ssd_heads = nn.ModuleList()

        for os, in_dim, proj_dim, n_anchors, step in zip(
            output_strides,
            enc_channels_list,
            proj_channels,
            anchors_aspect_ratio,
            anchor_steps,
        ):
            self.ssd_heads += [
                SSDHead(
                    opts=opts,
                    in_channels=in_dim,
                    n_classes=self.n_detection_classes,
                    n_coordinates=self.coordinates,
                    n_anchors=n_anchors,
                    proj_channels=proj_dim,
                    kernel_size=3 if os != -1 else 1,
                    stride=step,
                )
            ]

        self.anchors_aspect_ratio = anchors_aspect_ratio
        self.output_strides = output_strides

        self.match_prior = build_matcher(opts=opts)
        self.step = self.anchor_box_generator.step

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.detection.ssd.anchors-aspect-ratio",
            type=int,
            nargs="+",
            action="append",
            default=[[2, 3]] * 4,
            help="Anchors aspect ratio in each feature map obtained at different output strides.",
        )
        group.add_argument(
            "--model.detection.ssd.output-strides",
            type=int,
            nargs="+",
            default=[16, 32, 64, 128],
            help="Extract feature maps from these output strides.",
        )
        group.add_argument(
            "--model.detection.ssd.proj-channels",
            type=int,
            nargs="+",
            default=[512] * 4,
            help="Projection channels for feature map obtained at each output stride",
        )

        # depreciated
        group.add_argument(
            "--model.detection.ssd.min-box-size",
            type=float,
            default=None,
            help="Min. box size. Value between 0 and 1. Good default value is 0.1",
        )
        group.add_argument(
            "--model.detection.ssd.max-box-size",
            type=float,
            default=None,
            help="Max. box size. Value between 0 and 1. Good default value is 1.05",
        )

        # Depreciated
        group.add_argument(
            "--model.detection.ssd.center-variance",
            type=float,
            default=None,
            help="Center variance.",
        )
        group.add_argument(
            "--model.detection.ssd.size-variance",
            type=float,
            default=None,
            help="Size variance.",
        )
        group.add_argument(
            "--model.detection.ssd.iou-threshold",
            type=float,
            default=None,
            help="IOU Threshold.",
        )

        # inference related arguments
        group.add_argument(
            "--model.detection.ssd.conf-threshold",
            type=float,
            default=0.01,
            help="Confidence threshold. For evaluation on COCO, set to 0.01, so that we can compute mAP",
        )
        group.add_argument(
            "--model.detection.ssd.top-k",
            type=int,
            default=400,
            help="Keep only top-k objects before NMS",
        )
        group.add_argument(
            "--model.detection.ssd.objects-per-image",
            type=int,
            default=200,
            help="Keep only these many objects after NMS",
        )
        group.add_argument(
            "--model.detection.ssd.nms-iou-threshold",
            type=float,
            default=0.5,
            help="NMS IoU threshold ",
        )

        # FPN
        group.add_argument(
            "--model.detection.ssd.fpn-out-channels",
            type=int,
            default=256,
            help="Number of output channels in FPN",
        )
        group.add_argument(
            "--model.detection.ssd.use-fpn",
            action="store_true",
            help="Use SSD with FPN",
        )

        return parser

    @staticmethod
    def reset_layers(module) -> None:
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method="xavier_uniform")

    @staticmethod
    def process_anchors_ar(anchor_ar: List) -> List:
        assert isinstance(anchor_ar, list)
        new_ar = []
        for ar in anchor_ar:
            if ar in new_ar:
                continue
            new_ar.append(ar)
        return new_ar

    def get_backbone_features(self, x: Tensor) -> Dict[str, Tensor]:
        # extract features from the backbone network
        enc_end_points: Dict = self.encoder.extract_end_points_all(x)

        end_points: Dict = dict()
        for idx, os in enumerate(self.output_strides):
            if os == 8:
                end_points["os_{}".format(os)] = enc_end_points.pop("out_l3")
            elif os == 16:
                end_points["os_{}".format(os)] = enc_end_points.pop("out_l4")
            elif os == 32:
                end_points["os_{}".format(os)] = enc_end_points.pop("out_l5")
            else:
                x = end_points["os_{}".format(self.output_strides[idx - 1])]
                end_points["os_{}".format(os)] = self.extra_layers["os_{}".format(os)](
                    x
                )

        if self.fpn is not None:
            # apply Feature Pyramid Network
            end_points = self.fpn(end_points)

        return end_points

    def ssd_forward(
        self,
        end_points: Dict[str, Tensor],
        device: Optional[torch.device] = torch.device("cpu"),
        *args,
        **kwargs
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]:

        locations = []
        confidences = []
        anchors = []

        for os, ssd_head in zip(self.output_strides, self.ssd_heads):
            x = end_points["os_{}".format(os)]
            fm_h, fm_w = x.shape[2:]
            loc, pred = ssd_head(x)

            locations.append(loc)
            confidences.append(pred)

            anchors_fm_ctr = self.anchor_box_generator(
                fm_height=fm_h, fm_width=fm_w, fm_output_stride=os, device=device
            )
            anchors.append(anchors_fm_ctr.to(device=device))

        locations = torch.cat(locations, dim=1)
        confidences = torch.cat(confidences, dim=1)

        anchors = torch.cat(anchors, dim=0)
        anchors = anchors.unsqueeze(dim=0)

        return confidences, locations, anchors

    def forward(
        self, x: Union[Tensor, Dict]
    ) -> Union[Tuple[Tensor, ...], Tuple[Any, ...], Dict]:
        if isinstance(x, Dict):
            input_tensor = x["image"]
        elif isinstance(x, Tensor):
            input_tensor = x
        else:
            raise NotImplementedError(
                "Input to SSD should be either a Tensor or a Dict of Tensors"
            )

        device = input_tensor.device
        backbone_end_points: Dict = self.get_backbone_features(input_tensor)

        if not is_coreml_conversion(self.opts):
            confidences, locations, anchors = self.ssd_forward(
                end_points=backbone_end_points, device=device
            )

            output_dict = {"scores": confidences, "boxes": locations}

            if not self.training:
                # compute the detection results during evaluation
                scores = nn.Softmax(dim=-1)(confidences)
                boxes = self.match_prior.convert_to_boxes(
                    pred_locations=locations, anchors=anchors
                )

                detections = self.postprocess_detections(boxes=boxes, scores=scores)
                output_dict["detections"] = detections

            return output_dict
        else:
            return self.ssd_forward(end_points=backbone_end_points, is_prediction=False)

    @torch.no_grad()
    def predict(self, x: Tensor, *args, **kwargs) -> DetectionPredTuple:
        """Predict the bounding boxes given an image tensor"""
        bsz, channels, width, height = x.shape
        if bsz != 1:
            logger.error(
                "Prediction is supported with a batch size of 1 in {}".format(
                    self.__class__.__name__
                )
            )

        device = x.device
        enc_end_points: Dict = self.get_backbone_features(x)
        confidences, locations, anchors = self.ssd_forward(
            end_points=enc_end_points, device=device
        )

        scores = nn.Softmax(dim=-1)(confidences)

        boxes = self.match_prior.convert_to_boxes(
            pred_locations=locations, anchors=anchors
        )
        detections = self.postprocess_detections(boxes=boxes, scores=scores)[0]
        return detections

    @torch.no_grad()
    def postprocess_detections(
        self, boxes: Tensor, scores: Tensor
    ) -> List[DetectionPredTuple]:
        """Post process detections, including NMS"""
        # boxes [B, N, 4]
        # scores [B, N]
        # labels [B, N]

        batch_size = boxes.shape[0]
        n_classes = scores.shape[-1]

        device = boxes.device
        box_dtype = boxes.dtype
        scores_dtype = scores.dtype

        results = []
        for b_id in range(batch_size):
            object_labels = []
            object_boxes = []
            object_scores = []

            for class_index in range(1, n_classes):
                probs = scores[b_id, :, class_index]
                mask = probs > self.conf_threshold
                probs = probs[mask]
                if probs.size(0) == 0:
                    continue
                masked_boxes = boxes[b_id, mask, :]

                # keep only top-k indices
                num_topk = min(self.top_k, probs.size(0))
                probs, idxs = probs.topk(num_topk)
                masked_boxes = masked_boxes[idxs, ...]

                object_boxes.append(masked_boxes)
                object_scores.append(probs)
                object_labels.append(
                    torch.full_like(
                        probs, fill_value=class_index, dtype=torch.int64, device=device
                    )
                )

            if len(object_scores) == 0:
                output = DetectionPredTuple(
                    labels=torch.empty(0, device=device, dtype=torch.long),
                    scores=torch.empty(0, device=device, dtype=scores_dtype),
                    boxes=torch.empty(0, 4, device=device, dtype=box_dtype),
                )
            else:
                # concatenate all results
                object_scores = torch.cat(object_scores, dim=0)
                object_boxes = torch.cat(object_boxes, dim=0)
                object_labels = torch.cat(object_labels, dim=0)

                # non-maximum suppression
                keep = batched_nms(
                    object_boxes, object_scores, object_labels, self.nms_threshold
                )
                keep = keep[: self.objects_per_image]

                output = DetectionPredTuple(
                    labels=object_labels[keep],
                    scores=object_scores[keep],
                    boxes=object_boxes[keep],
                )
            results.append(output)
        return results

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes."""
        img_channels = 3
        height = 320
        width = 320
        n_classes = 80

        def generate_anchors(height, width):
            """Generate anchors **on-the-fly** based on the input resolution."""
            anchors = []
            for output_stride in self.output_strides:
                if output_stride == -1:
                    fm_width = fm_height = 1
                else:
                    fm_width = int(math.ceil(width / output_stride))
                    fm_height = int(math.ceil(height / output_stride))
                fm_anchor = self.anchor_box_generator(
                    fm_height=fm_height,
                    fm_width=fm_width,
                    fm_output_stride=output_stride,
                )
                anchors.append(fm_anchor)
            anchors = torch.cat(anchors, dim=0)
            return anchors

        # GT boxes have the same shape as anchors. So, we use anchors as GT boxes
        gt_boxes = generate_anchors(height=height, width=width)
        gt_boxes = gt_boxes.unsqueeze(0).expand(batch_size, -1, -1)

        gt_box_labels = torch.randint(
            low=0,
            high=n_classes,
            size=(batch_size, gt_boxes.shape[1]),
            dtype=torch.long,
        )

        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )
        labels = {
            "box_labels": gt_box_labels,
            "box_coordinates": gt_boxes,
        }

        return {"samples": img_tensor, "targets": labels}
