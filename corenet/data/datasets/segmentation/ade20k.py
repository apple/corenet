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


@DATASET_REGISTRY.register(name="ade20k", type="segmentation")
class ADE20KDataset(BaseImageSegmentationDataset):
    """Dataset class for the ADE20K dataset

    The structure of the dataset should be something like this: ::

    ADEChallengeData2016/annotations/training/*.png
    ADEChallengeData2016/annotations/validation/*.png

    ADEChallengeData2016/images/training/*.jpg
    ADEChallengeData2016/images/validation/*.jpg

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        root = self.root

        image_dir = os.path.join(
            root, "images", "training" if self.is_training else "validation"
        )
        annotation_dir = os.path.join(
            root, "annotations", "training" if self.is_training else "validation"
        )

        images = []
        masks = []
        for file_name in os.listdir(image_dir):
            if file_name.endswith(".jpg"):
                img_f_name = "{}/{}".format(image_dir, file_name)
                mask_f_name = "{}/{}".format(
                    annotation_dir, file_name.replace("jpg", "png")
                )

                if os.path.isfile(img_f_name) and os.path.isfile(mask_f_name):
                    images.append(img_f_name)
                    masks.append(mask_f_name)

        self.images = images
        self.masks = masks
        self.ignore_label = 255
        self.background_idx = 0
        self.check_dataset()

    @staticmethod
    def adjust_mask_value() -> int:
        """Adjust the mask value by this factor"""
        # because we do not include background index for ADE20k, we shift mask labels by 1
        return 1

    @staticmethod
    def color_palette() -> List[int]:
        """Class index to RGB color mapping. The list index corresponds to class id.
        Note that the color list is flattened."""
        color_codes = [
            [0, 0, 0],  # background
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ]
        color_codes = np.asarray(color_codes).flatten()
        return list(color_codes)

    @staticmethod
    def class_names() -> List[str]:
        """Class index (index of a list corresponds to class id) to class name"""
        return [
            "background",
            "wall",
            "building",
            "sky",
            "floor",
            "tree",
            "ceiling",
            "road",
            "bed ",
            "windowpane",
            "grass",
            "cabinet",
            "sidewalk",
            "person",
            "earth",
            "door",
            "table",
            "mountain",
            "plant",
            "curtain",
            "chair",
            "car",
            "water",
            "painting",
            "sofa",
            "shelf",
            "house",
            "sea",
            "mirror",
            "rug",
            "field",
            "armchair",
            "seat",
            "fence",
            "desk",
            "rock",
            "wardrobe",
            "lamp",
            "bathtub",
            "railing",
            "cushion",
            "base",
            "box",
            "column",
            "signboard",
            "chest of drawers",
            "counter",
            "sand",
            "sink",
            "skyscraper",
            "fireplace",
            "refrigerator",
            "grandstand",
            "path",
            "stairs",
            "runway",
            "case",
            "pool table",
            "pillow",
            "screen door",
            "stairway",
            "river",
            "bridge",
            "bookcase",
            "blind",
            "coffee table",
            "toilet",
            "flower",
            "book",
            "hill",
            "bench",
            "countertop",
            "stove",
            "palm",
            "kitchen island",
            "computer",
            "swivel chair",
            "boat",
            "bar",
            "arcade machine",
            "hovel",
            "bus",
            "towel",
            "light",
            "truck",
            "tower",
            "chandelier",
            "awning",
            "streetlight",
            "booth",
            "television receiver",
            "airplane",
            "dirt track",
            "apparel",
            "pole",
            "land",
            "bannister",
            "escalator",
            "ottoman",
            "bottle",
            "buffet",
            "poster",
            "stage",
            "van",
            "ship",
            "fountain",
            "conveyer belt",
            "canopy",
            "washer",
            "plaything",
            "swimming pool",
            "stool",
            "barrel",
            "basket",
            "waterfall",
            "tent",
            "bag",
            "minibike",
            "cradle",
            "oven",
            "ball",
            "food",
            "step",
            "tank",
            "trade name",
            "microwave",
            "pot",
            "animal",
            "bicycle",
            "lake",
            "dishwasher",
            "screen",
            "blanket",
            "sculpture",
            "hood",
            "sconce",
            "vase",
            "traffic light",
            "tray",
            "ashcan",
            "fan",
            "pier",
            "crt screen",
            "plate",
            "monitor",
            "bulletin board",
            "shower",
            "radiator",
            "glass",
            "clock",
            "flag",
        ]
