#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)


@DATASET_REGISTRY.register(name="imagenet", type="classification")
class ImageNetDataset(BaseImageClassificationDataset):
    """
    ImageNet dataset that follows the structure of ImageClassificationDataset.

    "ImageNet: A large-scale hierarchical image database"
    Jia Deng; Wei Dong; Richard Socher; Li-Jia Li; Kai Li; Li Fei-Fei
    2009 IEEE Conference on Computer Vision and Pattern Recognition
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        BaseImageClassificationDataset.__init__(
            self,
            opts=opts,
            *args,
            **kwargs,
        )
