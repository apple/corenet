#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List

from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification import (
    ZERO_SHOT_IMAGE_CLASSIFICATION_DATASET_REGISTRY,
    BaseZeroShotImageClassificationDataset,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.imagenet_class_names import (
    IMAGENET_CLASS_NAMES,
)


@ZERO_SHOT_IMAGE_CLASSIFICATION_DATASET_REGISTRY.register(name="imagenet")
class ImageNetDatasetZeroShot(BaseZeroShotImageClassificationDataset):
    """ImageNet dataset for zero-shot evaluation of image-text models."""

    @property
    def class_names(self) -> List[str]:
        """Return the name of the classes present in the dataset."""
        return IMAGENET_CLASS_NAMES
