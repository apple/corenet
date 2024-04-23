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


@DATASET_REGISTRY.register(name="coco_stuff", type="segmentation")
class COCOStuffDataset(BaseImageSegmentationDataset):
    """`COCOStuff <https://arxiv.org/abs/1612.03716>`_ dataset.

    The segmenation mask IDs are from 0 to 171 with 0 representing unlabelled/background
    area. So, it comprises of a total of 172 classes. The dataset includes annotation for
    all 164k images in COCO 2017 dataset.

    With 'cocostuff' as the root directory, the expected structure of images and annotations is as follows:

        cocostuff/
        ├── images/
        │   ├── train2017/
        │   │   ├── *.jpg
        │   ├── val2017/
        │   │   ├── *.jpg
        ├── annotations/
        │   ├── train2017/
        │   │   ├── *.png
        │   ├── val2017/
        │   │   ├── *.png

    ...note:
        The dataset has total of 182 classes, but labels are provided only for 171 classes. Therefore, unnannotated
        classes needs to be remapped before training. This can be done by running the following script:

        >>> python tools/converter_coco_stuff.py --src-dir cocostuff/annotations

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        root = self.root

        year = 2017
        split = "train" if self.is_training else "val"
        img_dir = os.path.join(root, "images/{}{}".format(split, year))
        annotation_dir = os.path.join(root, "annotations/{}{}".format(split, year))

        self.masks = []
        self.images = []

        file_names = os.listdir(img_dir)
        for file_name in file_names:
            if not file_name.endswith(".jpg"):
                continue
            jpg_file = os.path.join(img_dir, file_name)
            png_file = os.path.join(annotation_dir, file_name.replace(".jpg", ".png"))
            self.images.append(jpg_file)
            self.masks.append(png_file)

        self.ignore_label = 255
        self.background_idx = 0
        self.check_dataset()

    @staticmethod
    def color_palette() -> List[int]:
        """Class index to RGB color mapping. The list index corresponds to class id.
        Note that the color list is flattened."""
        color_codes = [
            [0, 0, 0],
            [0, 192, 64],
            [0, 192, 64],
            [0, 64, 96],
            [128, 192, 192],
            [0, 64, 64],
            [0, 192, 224],
            [0, 192, 192],
            [128, 192, 64],
            [0, 192, 96],
            [128, 192, 64],
            [128, 32, 192],
            [0, 0, 224],
            [0, 0, 64],
            [0, 160, 192],
            [128, 0, 96],
            [128, 0, 192],
            [0, 32, 192],
            [128, 128, 224],
            [0, 0, 192],
            [128, 160, 192],
            [128, 128, 0],
            [128, 0, 32],
            [128, 32, 0],
            [128, 0, 128],
            [64, 128, 32],
            [0, 160, 0],
            [0, 0, 0],
            [192, 128, 160],
            [0, 32, 0],
            [0, 128, 128],
            [64, 128, 160],
            [128, 160, 0],
            [0, 128, 0],
            [192, 128, 32],
            [128, 96, 128],
            [0, 0, 128],
            [64, 0, 32],
            [0, 224, 128],
            [128, 0, 0],
            [192, 0, 160],
            [0, 96, 128],
            [128, 128, 128],
            [64, 0, 160],
            [128, 224, 128],
            [128, 128, 64],
            [192, 0, 32],
            [128, 96, 0],
            [128, 0, 192],
            [0, 128, 32],
            [64, 224, 0],
            [0, 0, 64],
            [128, 128, 160],
            [64, 96, 0],
            [0, 128, 192],
            [0, 128, 160],
            [192, 224, 0],
            [0, 128, 64],
            [128, 128, 32],
            [192, 32, 128],
            [0, 64, 192],
            [0, 0, 32],
            [64, 160, 128],
            [128, 64, 64],
            [128, 0, 160],
            [64, 32, 128],
            [128, 192, 192],
            [0, 0, 160],
            [192, 160, 128],
            [128, 192, 0],
            [128, 0, 96],
            [192, 32, 0],
            [128, 64, 128],
            [64, 128, 96],
            [64, 160, 0],
            [0, 64, 0],
            [192, 128, 224],
            [64, 32, 0],
            [0, 192, 128],
            [64, 128, 224],
            [192, 160, 0],
            [0, 192, 0],
            [192, 128, 96],
            [192, 96, 128],
            [0, 64, 128],
            [64, 0, 96],
            [64, 224, 128],
            [128, 64, 0],
            [192, 0, 224],
            [64, 96, 128],
            [128, 192, 128],
            [64, 0, 224],
            [192, 224, 128],
            [128, 192, 64],
            [192, 0, 96],
            [192, 96, 0],
            [128, 64, 192],
            [0, 128, 96],
            [0, 224, 0],
            [64, 64, 64],
            [128, 128, 224],
            [0, 96, 0],
            [64, 192, 192],
            [0, 128, 224],
            [128, 224, 0],
            [64, 192, 64],
            [128, 128, 96],
            [128, 32, 128],
            [64, 0, 192],
            [0, 64, 96],
            [0, 160, 128],
            [192, 0, 64],
            [128, 64, 224],
            [0, 32, 128],
            [192, 128, 192],
            [0, 64, 224],
            [128, 160, 128],
            [192, 128, 0],
            [128, 64, 32],
            [128, 32, 64],
            [192, 0, 128],
            [64, 192, 32],
            [0, 160, 64],
            [64, 0, 0],
            [192, 192, 160],
            [0, 32, 64],
            [64, 128, 128],
            [64, 192, 160],
            [128, 160, 64],
            [64, 128, 0],
            [192, 192, 32],
            [128, 96, 192],
            [64, 0, 128],
            [64, 64, 32],
            [0, 224, 192],
            [192, 0, 0],
            [192, 64, 160],
            [0, 96, 192],
            [192, 128, 128],
            [64, 64, 160],
            [128, 224, 192],
            [192, 128, 64],
            [192, 64, 32],
            [128, 96, 64],
            [192, 0, 192],
            [0, 192, 32],
            [64, 224, 64],
            [64, 0, 64],
            [128, 192, 160],
            [64, 96, 64],
            [64, 128, 192],
            [0, 192, 160],
            [192, 224, 64],
            [64, 128, 64],
            [128, 192, 32],
            [192, 32, 192],
            [64, 64, 192],
            [0, 64, 32],
            [64, 160, 192],
            [192, 64, 64],
            [128, 64, 160],
            [64, 32, 192],
            [192, 192, 192],
            [0, 64, 160],
            [192, 160, 192],
            [192, 192, 0],
            [128, 64, 96],
            [192, 32, 64],
            [192, 64, 128],
            [64, 192, 96],
            [64, 160, 64],
            [64, 64, 0],
        ]

        color_codes = np.asarray(color_codes).flatten()
        return list(color_codes)

    @staticmethod
    def class_names() -> List[str]:
        """Class index to class name mapping. Class index corresponds to list index"""

        return [
            "unlabeled",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
            "banner",
            "blanket",
            "branch",
            "bridge",
            "building-other",
            "bush",
            "cabinet",
            "cage",
            "cardboard",
            "carpet",
            "ceiling-other",
            "ceiling-tile",
            "cloth",
            "clothes",
            "clouds",
            "counter",
            "cupboard",
            "curtain",
            "desk-stuff",
            "dirt",
            "door-stuff",
            "fence",
            "floor-marble",
            "floor-other",
            "floor-stone",
            "floor-tile",
            "floor-wood",
            "flower",
            "fog",
            "food-other",
            "fruit",
            "furniture-other",
            "grass",
            "gravel",
            "ground-other",
            "hill",
            "house",
            "leaves",
            "light",
            "mat",
            "metal",
            "mirror-stuff",
            "moss",
            "mountain",
            "mud",
            "napkin",
            "net",
            "paper",
            "pavement",
            "pillow",
            "plant-other",
            "plastic",
            "platform",
            "playingfield",
            "railing",
            "railroad",
            "river",
            "road",
            "rock",
            "roof",
            "rug",
            "salad",
            "sand",
            "sea",
            "shelf",
            "sky-other",
            "skyscraper",
            "snow",
            "solid-other",
            "stairs",
            "stone",
            "straw",
            "structural-other",
            "table",
            "tent",
            "textile-other",
            "towel",
            "tree",
            "vegetable",
            "wall-brick",
            "wall-concrete",
            "wall-other",
            "wall-panel",
            "wall-stone",
            "wall-tile",
            "wall-wood",
            "water-other",
            "waterdrops",
            "window-blind",
            "window-other",
            "wood",
        ]
