#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List, Mapping, Tuple, Union

import torch
from torch import Tensor

from corenet.data.collate_fns import COLLATE_FN_REGISTRY
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.detection.coco_base import COCODetection
from corenet.data.transforms import image_pil as T
from corenet.data.transforms.common import Compose


@DATASET_REGISTRY.register(name="coco_mask_rcnn", type="detection")
class COCODetectionMaskRCNN(COCODetection):
    """Dataset class for the MS COCO Object Detection using Mask RCNN ."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != COCODetectionMaskRCNN:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.detection.coco-mask-rcnn.use-lsj-aug",
            action="store_true",
            help="Use large scale jitter augmentation for training Mask RCNN model",
        )

        return parser

    def _training_transforms(
        self, size: Tuple[int, int], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data augmentation during training.

        Default order of transformation is Resize, RandomHorizontalFlip, ToTensor.
        When large-scale jittering is enabled, Resize is replaced with ScaleJitter and FixedSizeCrop

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """

        if getattr(self.opts, "dataset.detection.coco_mask_rcnn.use_lsj_aug"):
            # Apply large scale jittering, following https://arxiv.org/abs/2012.07177
            aug_list = [
                T.ScaleJitter(opts=self.opts),
                T.FixedSizeCrop(opts=self.opts),
                T.RandomHorizontalFlip(opts=self.opts),
                T.ToTensor(opts=self.opts),
            ]
        else:
            # standard augmentation for Mask-RCNN
            aug_list = [
                T.Resize(opts=self.opts, img_size=size),
                T.RandomHorizontalFlip(opts=self.opts),
                T.ToTensor(opts=self.opts),
            ]

        return Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(
        self, size: Tuple[int, int], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data augmentation during validation or evaluation.

        Default order of transformation is Resize, ToTensor.

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]
        return Compose(opts=self.opts, img_transforms=aug_list)

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
            output_data["samples"]["label]["labels"]: Shape is [Num of boxes]
            output_data["samples"]["label"]["boxes"]: Shape is [Num of boxes, 4]
            output_data["samples"]["label"]["masks"]: Shape is [Num of boxes, Height, Width]
            output_data["targets"]["image_id"]: Shape is [1]
            output_data["targets"]["image_width"]: Shape is [1]
            output_data["targets"]["image_height"]: Shape is [1]
        """
        crop_size_h, crop_size_w, img_index = sample_size_and_index

        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_name = self.get_image(image_id=image_id)
        im_width, im_height = image.size

        boxes, labels, mask = self.get_boxes_and_labels(
            image_id=image_id,
            image_width=im_width,
            image_height=im_height,
            include_masks=True,
        )

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes,
            "mask": mask,
        }

        if transform_fn is not None:
            data = transform_fn(data)

        output_data = {
            "samples": {
                "image": data["image"],
                # PyTorch Mask RCNN implementation expect labels as an input. Because we do not want to change
                # the training infrastructure, we pass labels as part of image key and
                # handle it in the model.
                "label": {
                    "labels": data["box_labels"],
                    "boxes": data["box_coordinates"],
                    "masks": data["mask"],
                },
            },
            "targets": {
                "image_id": torch.tensor(image_id),
                "image_width": torch.tensor(im_width),
                "image_height": torch.tensor(im_height),
            },
        }

        return output_data


@COLLATE_FN_REGISTRY.register(name="coco_mask_rcnn_collate_fn")
def coco_mask_rcnn_collate_fn(
    batch: List[Mapping[str, Union[Tensor, Mapping[str, Tensor]]]],
    opts: argparse.Namespace,
    *args,
    **kwargs
) -> Mapping[str, Union[List[Tensor], Mapping[str, List[Tensor]]]]:
    """Combines a list of dictionaries into a single dictionary by concatenating matching fields.

    For expected keys, see the keys in the output of `__getitem__` function of COCODetectionMaskRCNN class.

    Args:
        batch: A list of dictionaries
        opts: Command-line arguments

    Returns:
        A dictionary with `samples` and `targets` as keys.
    """
    new_batch = {"samples": {"image": [], "label": []}, "targets": []}

    for b_id, batch_ in enumerate(batch):
        new_batch["samples"]["image"].append(batch_["samples"]["image"])
        new_batch["samples"]["label"].append(batch_["samples"]["label"])
        new_batch["targets"].append(batch_["targets"])

    return new_batch
