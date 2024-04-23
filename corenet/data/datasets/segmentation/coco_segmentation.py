#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from torch import Tensor

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.segmentation.base_segmentation import (
    BaseImageSegmentationDataset,
)


@DATASET_REGISTRY.register(name="coco", type="segmentation")
class COCOSegmentationDataset(BaseImageSegmentationDataset):
    """Dataset class for the COCO dataset that maps classes to PASCAL VOC classes

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        year = 2017
        split = "train" if self.is_training else "val"
        ann_file = os.path.join(
            self.root, "annotations/instances_{}{}.json".format(split, year)
        )
        self.img_dir = os.path.join(self.root, "images/{}{}".format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.ids = list(self.coco.imgs.keys())

        self.ignore_label = 255
        self.background_idx = 0

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
            output_data["targets"]["mask"]: Shape is [Height, Width]

        """
        crop_size_h, crop_size_w, img_index = sample_size_and_index

        _transform = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        coco = self.coco
        img_id = self.ids[img_index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]

        rgb_img = self.read_image_pil(os.path.join(self.img_dir, path))
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        mask = self._gen_seg_mask(
            cocotarget, img_metadata["height"], img_metadata["width"]
        )

        data = {"image": rgb_img, "mask": None if self.is_evaluation else mask}

        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = mask

        output_data = {"samples": data["image"], "targets": data["mask"]}

        if self.is_evaluation:
            im_width, im_height = rgb_img.size
            img_name = path.replace("jpg", "png")
            mask = output_data.pop("targets")
            output_data["targets"] = {
                "mask": mask,
                "file_name": img_name,
                "im_width": im_width,
                "im_height": im_height,
            }

        return output_data

    def _gen_seg_mask(self, target, h: int, w: int) -> np.ndarray:
        """Generates a mask in PASCAL VOC format"""
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        coco_to_pascal = self.coco_to_pascal_mapping()
        for instance in target:
            rle = coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = coco_mask.decode(rle)
            cat = instance["category_id"]
            if cat in coco_to_pascal:
                c = coco_to_pascal.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(
                    np.uint8
                )
        return mask

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def class_names() -> List[str]:
        """PASCAL VOC classes names"""
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

    @staticmethod
    def coco_to_pascal_mapping() -> List[int]:
        """COCO to PASCAL VOC class mapping"""
        return [
            0,
            5,
            2,
            16,
            9,
            44,
            6,
            3,
            17,
            62,
            21,
            67,
            18,
            19,
            4,
            1,
            64,
            20,
            63,
            7,
            72,
        ]
