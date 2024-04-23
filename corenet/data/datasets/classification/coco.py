#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from functools import cached_property
from typing import Any, Dict, List, Tuple, Union

import torch
from pycocotools.coco import COCO

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
    BaseImageDataset,
)
from corenet.data.transforms.image_pil import BaseTransformation


@DATASET_REGISTRY.register(name="coco", type="classification")
class COCOClassification(BaseImageDataset):
    """`COCO <https://cocodataset.org/#home>`_ dataset for multi-label object classification.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        year = 2017
        split = "train" if self.is_training else "val"
        ann_file = os.path.join(
            self.root, "annotations/instances_{}{}.json".format(split, year)
        )
        self.img_dir = os.path.join(self.root, "images/{}{}".format(split, year))
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.ids)

    def _training_transforms(
        self, size: Union[int, Tuple[int, int]]
    ) -> BaseTransformation:
        """Returns transformations applied to the input image in training mode.

        These transformations are the same as the 'BaseImageClassificationDataset'.

        Args:
            size: Size for resizing the input image. Expected to be an integer (width=height) or a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        return BaseImageClassificationDataset._training_transforms(self, size)

    def _validation_transforms(
        self, *unused_args, **unused_kwargs
    ) -> BaseTransformation:
        """Returns transformations applied to the input in validation mode.

        These transformations are the same as the 'BaseImageClassificationDataset'.

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        return BaseImageClassificationDataset._validation_transforms(self)

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Returns the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index).

        Returns:
            A dictionary with `samples`, `sample_id` and `targets` as keys corresponding to input, index, and label of
            a sample, respectively.

        Shapes:
            The output data dictionary contains three keys (samples, sample_id, and target). The values of these
            keys has the following shapes:
                data["samples"]: Shape is [image_channels, image_height, image_width]
                data["sample_id"]: Shape is 1
                data["targets"]: Shape is [num_classes]
        """

        crop_size_h, crop_size_w, img_index = sample_size_and_index

        coco = self.coco
        img_id = self.ids[img_index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        target = torch.zeros((3, self.n_classes), dtype=torch.long)
        # Steps to produce multi-label classification labels
        # Step 1: Group the target labels into three categories based on object area
        # Step 2: Produce the binary label for each class by computing maximum value
        #   along the first dimension (i.e., dim=0).

        # Step 1
        for obj in annotations:
            if obj["area"] < 32 * 32:  # small objects
                target[0][self.cat2cat[obj["category_id"]]] = 1
            elif obj["area"] < 96 * 96:  # medium objects
                target[1][self.cat2cat[obj["category_id"]]] = 1
            else:  # large objects
                target[2][self.cat2cat[obj["category_id"]]] = 1
        # Step 2
        target = target.amax(dim=0)

        img_path = os.path.join(self.img_dir, coco.loadImgs(img_id)[0]["file_name"])
        input_img = self.read_image_pil(img_path)

        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        data = transform_fn({"image": input_img})

        data["samples"] = data.pop("image")
        data["targets"] = target
        data["sample_id"] = img_index
        return data

    @cached_property
    def n_classes(self):
        return len(self.class_names)

    @cached_property
    def class_names(self) -> List[str]:
        """Returns the names of object classes in the COCO dataset."""
        return [
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
            "fire",
            "hydrant",
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
            "microwave oven",
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
        ]
