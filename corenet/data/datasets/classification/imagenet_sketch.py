#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""ImageNetSketch dataset, a distribution shift of ImageNet."""
import argparse

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_imagenet_shift_dataset import (
    BaseImageNetShiftDataset,
)


@DATASET_REGISTRY.register(name="imagenet_sketch", type="classification")
class ImageNetSketchDataset(BaseImageNetShiftDataset):
    """ImageNetSketch dataset, a distribution shift of ImageNet.

    Data set is created from Google Image queries "sketch of __", where __ is the
    standard class name. Search is only within the "black and white" color scheme.

    @inproceedings{wang2019learning,
        title={Learning Robust Global Representations by Penalizing Local Predictive
        Power},
        author={Wang, Haohan and Ge, Songwei and Lipton, Zachary and Xing, Eric P},
        booktitle={Advances in Neural Information Processing Systems},
        pages={10506--10518},
        year={2019}
    }
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        """Initialize ImageNetSketchDataset."""
        BaseImageNetShiftDataset.__init__(self, opts=opts, *args, **kwargs)

    @staticmethod
    def class_id_to_imagenet_class_id(class_id: int) -> int:
        """Return `class_id` as the ImageNet Sketch classes are the same as ImageNet."""
        return class_id
