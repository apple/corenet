#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""ImageNetR dataset, a distribution shift of ImageNet."""
import argparse

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_imagenet_shift_dataset import (
    BaseImageNetShiftDataset,
)
from corenet.data.datasets.classification.imagenet_synsets import (
    IMAGENET_R_SYNSETS,
    IMAGENET_SYNSETS,
)

IMAGENET_R_CLASS_SUBLIST = [
    IMAGENET_SYNSETS.index(IMAGENET_R_SYNSETS[synset])
    for synset in range(len(IMAGENET_R_SYNSETS))
]


@DATASET_REGISTRY.register(name="imagenet_r", type="classification")
class ImageNetRDataset(BaseImageNetShiftDataset):
    """ImageNetR dataset, a distribution shift of ImageNet.

    ImageNet-R(endition) contains art, cartoons, deviantart, graffiti, embroidery,
    graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures,
    sketches, tattoos, toys, and video game renditions of ImageNet classes.

    @article{hendrycks2021many,
    title={The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution
    Generalization},
    author={Dan Hendrycks and Steven Basart and Norman Mu and Saurav Kadavath and Frank
    Wang and Evan Dorundo and Rahul Desai and Tyler Zhu and Samyak Parajuli and Mike Guo
    and Dawn Song and Jacob Steinhardt and Justin Gilmer},
    journal={ICCV},
    year={2021}
    }

    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        """Initialize ImageNetR."""
        BaseImageNetShiftDataset.__init__(self, opts=opts, *args, **kwargs)

    @staticmethod
    def class_id_to_imagenet_class_id(class_id: int) -> int:
        """Return the mapped class index using precomputed mapping."""
        return IMAGENET_R_CLASS_SUBLIST[class_id]
