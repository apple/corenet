#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from typing import List

import numpy as np

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.segmentation.base_segmentation import (
    BaseImageSegmentationDataset,
)


@DATASET_REGISTRY.register(name="pascal", type="segmentation")
class PascalVOCDataset(BaseImageSegmentationDataset):
    """Dataset class for the PASCAL VOC 2012 dataset

    The structure of PASCAL VOC dataset should be like this: ::

        pascal_voc/VOCdevkit/VOC2012/Annotations
        pascal_voc/VOCdevkit/VOC2012/JPEGImages
        pascal_voc/VOCdevkit/VOC2012/SegmentationClass
        pascal_voc/VOCdevkit/VOC2012/SegmentationClassAug_Visualization
        pascal_voc/VOCdevkit/VOC2012/ImageSets
        pascal_voc/VOCdevkit/VOC2012/list
        pascal_voc/VOCdevkit/VOC2012/SegmentationClassAug
        pascal_voc/VOCdevkit/VOC2012/SegmentationObject

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        use_coco_data = getattr(opts, "dataset.pascal.use_coco_data")
        coco_root_dir = getattr(opts, "dataset.pascal.coco_root_dir")
        root = self.root

        voc_root_dir = os.path.join(root, "VOC2012")
        voc_list_dir = os.path.join(voc_root_dir, "list")

        coco_data_file = None
        if self.is_training:
            # use the PASCAL VOC 2012 train data with augmented data
            data_file = os.path.join(voc_list_dir, "train_aug.txt")
            if use_coco_data and coco_root_dir is not None:
                coco_data_file = os.path.join(coco_root_dir, "train_2017.txt")
                assert os.path.isfile(
                    coco_data_file
                ), "COCO data file does not exist at: {}".format(coco_root_dir)
        else:
            data_file = os.path.join(voc_list_dir, "val.txt")

        self.images = []
        self.masks = []
        with open(data_file, "r") as lines:
            for line in lines:
                line_split = line.split(" ")
                rgb_img_loc = voc_root_dir + os.sep + line_split[0].strip()
                mask_img_loc = voc_root_dir + os.sep + line_split[1].strip()
                assert os.path.isfile(
                    rgb_img_loc
                ), "RGB file does not exist at: {}".format(rgb_img_loc)
                assert os.path.isfile(
                    mask_img_loc
                ), "Mask image does not exist at: {}".format(rgb_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(mask_img_loc)

        # if COCO data (mapped in PASCAL VOC format) needs to be used during training
        if self.is_training and coco_data_file is not None:
            with open(coco_data_file, "r") as lines:
                for line in lines:
                    line_split = line.split(" ")
                    rgb_img_loc = coco_root_dir + os.sep + line_split[0].rstrip()
                    mask_img_loc = coco_root_dir + os.sep + line_split[1].rstrip()
                    assert os.path.isfile(rgb_img_loc)
                    assert os.path.isfile(mask_img_loc)
                    self.images.append(rgb_img_loc)
                    self.masks.append(mask_img_loc)
        self.use_coco_data = use_coco_data
        self.ignore_label = 255
        self.background_idx = 0
        self.check_dataset()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != PascalVOCDataset:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.pascal.use-coco-data",
            action="store_true",
            default=False,
            help="Use MS-COCO data for training with PASCAL VOC dataset. Defaults to False.",
        )
        group.add_argument(
            "--dataset.pascal.coco-root-dir",
            type=str,
            default=None,
            help="Location of MS-COCO data. Defaults to None.",
        )
        return parser

    @staticmethod
    def color_palette() -> List[int]:
        """Class index to RGB color mapping. The list index corresponds to class id.
        Note that the color list is flattened."""
        color_codes = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]

        color_codes = np.asarray(color_codes).flatten()
        return list(color_codes)

    @staticmethod
    def class_names() -> List[str]:
        """Class index to class name mapping. Class index corresponds to list index"""
        return [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted_plant",
            "sheep",
            "sofa",
            "train",
            "tv_monitor",
        ]
