#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.datasets import BaseImageDataset


class BaseDetectionDataset(BaseImageDataset):
    """Base Dataset class for object detection datasets."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseDetectionDataset:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--evaluation.detection.save-overlay-boxes",
            action="store_true",
            help="enable this flag to visualize predicted masks on top of input image",
        )
        group.add_argument(
            "--evaluation.detection.mode",
            type=str,
            default="validation_set",
            required=False,
            choices=["single_image", "image_folder", "validation_set"],
            help="Contribution of mask when overlaying on top of RGB image.",
        )
        group.add_argument(
            "--evaluation.detection.path",
            type=str,
            default=None,
            help="Path of the image or image folder (only required for single_image and image_folder modes).",
        )
        group.add_argument(
            "--evaluation.detection.num-classes",
            type=int,
            default=None,
            help="Number of segmentation classes used during training.",
        )
        group.add_argument(
            "--evaluation.detection.resize-input-images",
            action="store_true",
            default=False,
            help="Resize input images to fixed size during detection evaluation.",
        )

        return parser
