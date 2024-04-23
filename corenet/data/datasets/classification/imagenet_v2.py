#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import tarfile
from pathlib import Path
from typing import Dict, Tuple

import torch

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.dataset_base import BaseImageDataset
from corenet.data.transforms import image_pil as T
from corenet.data.transforms.common import Compose
from corenet.utils import logger
from corenet.utils.download_utils import get_local_path

IMAGENETv2_SPLIT_LINK_MAP = {
    "matched_frequency": {
        "url": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz",
        "extracted_folder_name": "imagenetv2-matched-frequency-format-val",
    },
    "threshold_0.7": {
        "url": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz",
        "extracted_folder_name": "imagenetv2-threshold0.7-format-val",
    },
    "top_images": {
        "url": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz",
        "extracted_folder_name": "imagenetv2-top-images-format-val",
    },
}


@DATASET_REGISTRY.register(name="imagenet_v2", type="classification")
class Imagenetv2Dataset(BaseImageDataset):
    """
    `ImageNetv2 Dataset <https://arxiv.org/abs/1902.10811>`_ for studying the robustness of models trained on ImageNet dataset

    Args:
        opts: command-line arguments
    """

    def __init__(
        self,
        opts,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(opts=opts, *args, **kwargs)
        if self.is_training:
            logger.error(
                "{} can only be used for evaluation".format(self.__class__.__name__)
            )

        split = getattr(opts, "dataset.imagenet_v2.split", None)
        if split is None or split not in IMAGENETv2_SPLIT_LINK_MAP.keys():
            logger.error(
                "Please specify split for ImageNetv2. Supported ImageNetv2 splits are: {}".format(
                    IMAGENETv2_SPLIT_LINK_MAP.keys()
                )
            )

        split_path = get_local_path(opts, path=IMAGENETv2_SPLIT_LINK_MAP[split]["url"])
        with tarfile.open(split_path) as tf:
            tf.extractall(self.root)

        root = Path(
            "{}/{}".format(
                self.root,
                IMAGENETv2_SPLIT_LINK_MAP[split]["extracted_folder_name"],
            )
        )
        file_names = list(root.glob("**/*.jpeg"))
        self.file_names = file_names

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        if cls != Imagenetv2Dataset:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.imagenet-v2.split",
            type=str,
            default="matched-frequency",
            help="ImageNetv2 dataset. Possible choices are: {}".format(
                [
                    f"{i + 1}: {split_name}"
                    for i, split_name in enumerate(IMAGENETv2_SPLIT_LINK_MAP.keys())
                ]
            ),
            choices=IMAGENETv2_SPLIT_LINK_MAP.keys(),
        )
        return parser

    def _validation_transforms(self, *args, **kwargs):
        """Data transforms during validation

        Order of transform is Resize, CenterCrop, ToTensor

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, sample_size_and_index: Tuple) -> Dict:
        """Returns the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Shapes:
            The output data dictionary contains three keys (samples, sample_id, and target). The values of these
            keys has the following shapes:
                data["samples"]: Shape is [Channels, Height, Width]
                data["sample_id"]: Shape is 1
                data["targets"]: Shape is 1

        Returns:
            A dictionary with `samples`, `sample_id` and `targets` as keys corresponding to input, index and label of
            a sample, respectively.
        """
        crop_size_h, crop_size_w, img_index = sample_size_and_index

        # same for validation and evaluation
        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        # infer target label from the file name
        # file names are organized as SPLIT_NAME-format-val/class_idx/*.jpg
        # Example: All images in this folder (imagenetv2-matched-frequency-format-val/0/*.jpg) belong to class 0
        img_path = str(self.file_names[img_index])
        target = int(self.file_names[img_index].parent.name)

        input_img = self.read_image_pil(img_path)
        if input_img is None:
            # Sometimes images are corrupt
            # Skip such images
            logger.log("Img index {} is possibly corrupt.".format(img_index))
            input_tensor = torch.zeros(
                size=(3, crop_size_h, crop_size_w), dtype=torch.float
            )
            target = -1
            data = {"image": input_tensor}
        else:
            data = {"image": input_img}
            data = transform_fn(data)

        data["samples"] = data["image"]
        data["targets"] = target
        data["sample_id"] = img_index

        return data

    def __len__(self) -> int:
        return len(self.file_names)
