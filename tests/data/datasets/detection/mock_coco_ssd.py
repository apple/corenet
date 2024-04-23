#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

import numpy as np
from PIL import Image

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.detection.coco_ssd import COCODetectionSSD


@DATASET_REGISTRY.register(name="mock_coco_ssd", type="detection")
class MockCOCODetectionSSD(COCODetectionSSD):
    @staticmethod
    def read_image_pil(path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_image_pil function.

        Instead of reading a PIL image at location specified by `path`, a random PIL
        image is returned.
        """
        im_arr = np.random.randint(low=0, high=255, size=(64, 64, 3), dtype=np.uint8)
        return Image.fromarray(im_arr).convert("RGB")
