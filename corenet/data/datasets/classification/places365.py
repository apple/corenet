#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)


@DATASET_REGISTRY.register(name="places365", type="classification")
class Places365Dataset(BaseImageClassificationDataset):
    """
    Places365 dataset that follows the structure of ImageClassificationDataset.

    "Places: A 10 million Image Database for Scene Recognition"
    B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017
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
