#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import math
from typing import List, Mapping, Tuple, Union

import torch
from torch import Tensor

from corenet.data.collate_fns import COLLATE_FN_REGISTRY
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.detection.coco_base import COCODetection
from corenet.data.transforms import image_pil as T
from corenet.data.transforms.common import Compose
from corenet.modeling.anchor_generator import build_anchor_generator
from corenet.modeling.matcher_det import build_matcher
from corenet.utils import logger


@DATASET_REGISTRY.register(name="coco_ssd", type="detection")
class COCODetectionSSD(COCODetection):
    """Dataset class for the MS COCO Object Detection using Single Shot Object Detector (SSD).

    Args:
        opts: Command-line arguments
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        if getattr(opts, "matcher.name") != "ssd":
            logger.error("For SSD, we need --matcher.name as ssd")

        anchor_gen_name = getattr(opts, "anchor_generator.name")
        if anchor_gen_name is None or anchor_gen_name != "ssd":
            logger.error("For SSD, we need --anchor-generator.name to be ssd")

        super().__init__(opts=opts, *args, **kwargs)

        # we build the anchor generator and matching inside the dataset
        # so that we can use it with variable batch samplers.
        self.anchor_box_generator = build_anchor_generator(opts=opts, is_numpy=True)
        self.match_prior = build_matcher(opts=opts)

        # output strides for generating anchors
        self.output_strides = self.anchor_box_generator.output_strides

    def _training_transforms(self, size: Tuple[int, int], *args, **kwargs) -> Compose:
        """Data augmentation during training.

        Order of transformation is SSDCroping, PhotometricDistort, RandomHorizontalFlip, Resize,
            BoxPercentCoords, ToTensor

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.common.Compose.`
        """
        aug_list = [
            T.SSDCroping(opts=self.opts),
            T.PhotometricDistort(opts=self.opts),
            T.RandomHorizontalFlip(opts=self.opts),
            T.Resize(opts=self.opts, img_size=size),
            T.BoxPercentCoords(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: Tuple[int, int], *args, **kwargs) -> Compose:
        """Data augmentation during validation or evaluation.

        Default order of transformation is Resize, BoxPercentCoords, ToTensor.

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.common.Compose.`
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.BoxPercentCoords(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]
        return Compose(opts=self.opts, img_transforms=aug_list)

    def generate_anchors(self, height: int, width: int) -> Tensor:
        """Given the height and width of the input to the SSD model, generate anchors

        Args:
            height: Height of the input image to the SSD model
            width: Width of the input image to the SSD model

        Returns:
            Tensor containing anchor locations
        """
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

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int], *args, **kwargs
    ) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
        """Returns the sample corresponding to the input sample index. Returned sample is transformed
        into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Returns:
            A dictionary with `samples` and `targets` as keys corresponding to input and labels of
            a sample, respectively.

        Shapes:
            The shape of values in output dictionary, output_data, are as follows:

            output_data["samples"]["image"]: Shape is [Channels, Height, Width]
            output_data["targets"]["box_labels"]: Shape is [Num of matched boxes]
            output_data["targets"]["box_coordinates"]: Shape is [Num of matched boxes, 4]
            output_data["targets"]["image_id"]: Shape is [1]
            output_data["targets"]["image_width"]: Shape is [1]
            output_data["targets"]["image_height"]: Shape is [1]

        """
        crop_size_h, crop_size_w, img_index = sample_size_and_index

        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_fname = self.get_image(image_id=image_id)
        im_width, im_height = image.size
        boxes, labels, _ = self.get_boxes_and_labels(
            image_id=image_id, image_width=im_width, image_height=im_height
        )

        data = {"image": image, "box_labels": labels, "box_coordinates": boxes}

        data = transform_fn(data)

        # convert to priors
        anchors = self.generate_anchors(height=crop_size_h, width=crop_size_w)

        gt_coordinates, gt_labels = self.match_prior(
            gt_boxes=data["box_coordinates"],
            gt_labels=data["box_labels"],
            anchors=anchors,
        )

        output_data = {
            "samples": {"image": data.pop("image")},
            "targets": {
                "box_labels": gt_labels,
                "box_coordinates": gt_coordinates,
                "image_id": torch.tensor(image_id),
                "image_width": torch.tensor(im_width),
                "image_height": torch.tensor(im_height),
            },
        }

        return output_data

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\tmatcher={self.match_prior}"
            f"\n\tanchor_generator={self.anchor_box_generator}"
        )
        return extra_repr_str


@COLLATE_FN_REGISTRY.register(name="coco_ssd_collate_fn")
def coco_ssd_collate_fn(
    batch: List[Mapping[str, Union[Tensor, Mapping[str, Tensor]]]],
    opts: argparse.Namespace,
) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
    """Combines a list of dictionaries into a single dictionary by concatenating matching fields.

    For expected keys, see the keys in the output of `__getitem__` function of COCODetectionSSD class.

    Args:
        batch: A list of dictionaries
        opts: Command-line arguments

    Returns:
        A dictionary with `samples` and `targets` as keys.
    """
    new_batch = {
        "samples": {"image": []},
        "targets": {
            "box_labels": [],
            "box_coordinates": [],
            "image_id": [],
            "image_width": [],
            "image_height": [],
        },
    }

    for b_id, batch_ in enumerate(batch):
        # prepare inputs
        new_batch["samples"]["image"].append(batch_["samples"]["image"])

        # prepare outputs
        new_batch["targets"]["box_labels"].append(batch_["targets"]["box_labels"])
        new_batch["targets"]["box_coordinates"].append(
            batch_["targets"]["box_coordinates"]
        )
        new_batch["targets"]["image_id"].append(batch_["targets"]["image_id"])
        new_batch["targets"]["image_width"].append(batch_["targets"]["image_width"])
        new_batch["targets"]["image_height"].append(batch_["targets"]["image_height"])

    # stack inputs
    new_batch["samples"]["image"] = torch.stack(new_batch["samples"]["image"], dim=0)

    # stack outputs
    new_batch["targets"]["box_labels"] = torch.stack(
        new_batch["targets"]["box_labels"], dim=0
    )

    new_batch["targets"]["box_coordinates"] = torch.stack(
        new_batch["targets"]["box_coordinates"], dim=0
    )

    new_batch["targets"]["image_id"] = torch.stack(
        new_batch["targets"]["image_id"], dim=0
    )

    new_batch["targets"]["image_width"] = torch.stack(
        new_batch["targets"]["image_width"], dim=0
    )

    new_batch["targets"]["image_height"] = torch.stack(
        new_batch["targets"]["image_height"], dim=0
    )

    return new_batch
