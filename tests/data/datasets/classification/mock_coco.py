#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

import numpy as np
from PIL import Image

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.coco import COCOClassification


@DATASET_REGISTRY.register(name="mock_coco", type="classification")
class MockCOCOClassification(COCOClassification):
    @staticmethod
    def read_image_pil(path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_image_pil function.

        Instead of reading a PIL image at location specified by `path`, a random PIL
        image is returned.
        """
        im_arr = np.random.randint(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
        return Image.fromarray(im_arr).convert("RGB")
