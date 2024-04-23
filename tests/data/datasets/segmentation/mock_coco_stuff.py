#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Optional

import numpy as np
from PIL import Image

from corenet.data.datasets.segmentation.coco_stuff import COCOStuffDataset

TOTAL_SAMPLES = 8


from corenet.data.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register(name="mock_coco_stuff", type="segmentation")
class MockCOCOStuffDataset(COCOStuffDataset):
    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = False,
        is_evaluation: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Mock the init logic for ImageNet dataset

        Specifically, we replace the samples and targets with random data so that actual dataset is not
        required for testing purposes.
        """
        # super() is not called here intentionally.
        self.opts = opts
        self.root = None
        self.images = ["dummy_img_path.jpg" for _ in range(TOTAL_SAMPLES)]
        self.masks = ["dummy_mask_path.png" for _ in range(TOTAL_SAMPLES)]
        self.ignore_label = 255
        self.background_idx = 0
        self.n_classes = 171
        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.check_dataset()

    @staticmethod
    def read_image_pil(path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_image_pil function

        Instead of reading a PIL RGB image at location specified by `path`, a random PIL
        is returned.
        """
        im_arr = np.random.randint(low=0, high=255, size=(40, 20), dtype=np.uint8)
        return Image.fromarray(im_arr).convert("RGB")

    def read_mask_pil(self, path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_mask_pil function

        Instead of reading a mask at location specified by `path`, a random PIL mask is returned.
        """
        im_arr = np.random.randint(
            low=0, high=self.n_classes, size=(40, 20), dtype=np.uint8
        )
        return Image.fromarray(im_arr).convert("L")
