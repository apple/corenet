#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""ImageNetA dataset, a distribution shift of ImageNet."""
import argparse

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_imagenet_shift_dataset import (
    BaseImageNetShiftDataset,
)
from corenet.data.datasets.classification.imagenet_synsets import (
    IMAGENET_A_SYNSETS,
    IMAGENET_SYNSETS,
)

IMAGENET_A_CLASS_SUBLIST = [
    IMAGENET_SYNSETS.index(IMAGENET_A_SYNSETS[synset])
    for synset in range(len(IMAGENET_A_SYNSETS))
]


@DATASET_REGISTRY.register(name="imagenet_a", type="classification")
class ImageNetADataset(BaseImageNetShiftDataset):
    """ImageNetA dataset, a distribution shift of ImageNet.

    ImageNet-A contains real-world, unmodified natural images that cause model accuracy
    to substantially degrade.

    @article{hendrycks2021nae,
    title={Natural Adversarial Examples},
    author={Dan Hendrycks and Kevin Zhao and Steven Basart and Jacob Steinhardt and Dawn
    Song},
    journal={CVPR},
    year={2021}
    }
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        """Initialize ImageNetA."""
        BaseImageNetShiftDataset.__init__(self, opts=opts, *args, **kwargs)

    @staticmethod
    def class_id_to_imagenet_class_id(class_id: int) -> int:
        """Return the mapped class index using precomputed mapping."""
        return IMAGENET_A_CLASS_SUBLIST[class_id]
