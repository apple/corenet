#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import random
from typing import List

from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.imagenet import (
    ImageNetDatasetZeroShot,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.imagenet_a import (
    ImageNetADatasetZeroShot,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.imagenet_r import (
    ImageNetRDatasetZeroShot,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.imagenet_sketch import (
    ImageNetSketchDatasetZeroShot,
)

TOTAL_SAMPLES = 100


class MockImageNetDatasetZeroShot(ImageNetDatasetZeroShot):
    """Mock the ImageNetDatasetZeroShot without initializing from image folders."""

    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = False,
        is_evaluation: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Mock the init logic for ImageNet dataset.

        Specifically, we replace the samples and targets with random data so that actual
        dataset is not required for testing purposes.
        """
        # super() is not called here intentionally.
        self.opts = opts
        self.root = None
        self.samples = [
            ["img_path", random.randint(1, 4)] for _ in range(TOTAL_SAMPLES)
        ]
        self.targets = [class_id for img_path, class_id in self.samples]
        self.imgs = [img_path for img_path, class_id in self.samples]
        self.is_training = is_training
        self.is_evaluation = is_evaluation

    @property
    def class_names(self) -> List[str]:
        return ["cat", "dog"]


class MockImageNetADatasetZeroShot(
    MockImageNetDatasetZeroShot, ImageNetADatasetZeroShot
):
    def __init__(self, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetA dataset."""
        MockImageNetDatasetZeroShot.__init__(self, *args, **kwargs)


class MockImageNetRDatasetZeroShot(
    MockImageNetDatasetZeroShot, ImageNetRDatasetZeroShot
):
    def __init__(self, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetR dataset."""
        MockImageNetDatasetZeroShot.__init__(self, *args, **kwargs)


class MockImageNetSketchDatasetZeroShot(
    MockImageNetDatasetZeroShot, ImageNetSketchDatasetZeroShot
):
    def __init__(self, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetSketch dataset."""
        MockImageNetDatasetZeroShot.__init__(self, *args, **kwargs)
