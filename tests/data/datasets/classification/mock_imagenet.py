#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import random
from typing import Optional

import numpy as np
from PIL import Image

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.imagenet import ImageNetDataset
from corenet.data.datasets.classification.imagenet_a import (
    IMAGENET_A_CLASS_SUBLIST,
    ImageNetADataset,
)
from corenet.data.datasets.classification.imagenet_r import (
    IMAGENET_R_CLASS_SUBLIST,
    ImageNetRDataset,
)
from corenet.data.datasets.classification.imagenet_sketch import ImageNetSketchDataset

TOTAL_SAMPLES = 10


@DATASET_REGISTRY.register(name="mock_imagenet", type="classification")
class MockImageNetDataset(ImageNetDataset):
    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = False,
        is_evaluation: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Mock the init logic for ImageNet dataset.

        Specifically, we replace the samples and targets with random data so that actual dataset is not
        required for testing purposes.
        """
        # super() is not called here intentionally.
        self.opts = opts
        self.root = None
        self.targets = [random.randint(0, self.n_classes) for _ in range(TOTAL_SAMPLES)]
        self.imgs = ["img_path" for _ in range(TOTAL_SAMPLES)]
        self.samples = [
            [img_path, target_label]
            for img_path, target_label in zip(self.imgs, self.targets)
        ]
        self.is_training = is_training
        self.is_evaluation = is_evaluation

    @property
    def n_classes(self):
        return 1000

    @staticmethod
    def read_image_pil(path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_image_pil function.

        Instead of reading a PIL image at location specified by `path`, a random PIL
        image is returned. The randomness in height and width dimensions may allow us to
        catch errors in transform functions.
        """
        width = random.randint(32, 64)
        height = random.randint(32, 64)
        im_arr = np.random.randint(
            low=0, high=255, size=(width, height, 3), dtype=np.uint8
        )
        return Image.fromarray(im_arr).convert("RGB")


@DATASET_REGISTRY.register(name="mock_imagenet_a", type="mock_classification")
class MockImageNetADataset(MockImageNetDataset, ImageNetADataset):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetA dataset."""
        MockImageNetDataset.__init__(self, opts, *args, **kwargs)
        self.post_init_checks()

    @property
    def n_classes(self):
        return len(IMAGENET_A_CLASS_SUBLIST)


@DATASET_REGISTRY.register(name="mock_imagenet_r", type="mock_classification")
class MockImageNetRDataset(MockImageNetDataset, ImageNetRDataset):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetR dataset."""
        MockImageNetDataset.__init__(self, opts, *args, **kwargs)
        self.post_init_checks()

    @property
    def n_classes(self):
        return len(IMAGENET_R_CLASS_SUBLIST)


@DATASET_REGISTRY.register(name="mock_imagenet_sketch", type="mock_classification")
class MockImageNetSketchDataset(MockImageNetDataset, ImageNetSketchDataset):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetSketch dataset."""
        MockImageNetDataset.__init__(self, opts, *args, **kwargs)
        self.post_init_checks()
