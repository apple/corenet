#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Base class for ImageNet distribution shift datasets."""
import argparse
from typing import Any, Dict, Tuple

from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)


class BaseImageNetShiftDataset(BaseImageClassificationDataset):
    """ImageNet Distribution Shift Dataset.

    This base class supports ImageNet out-of-distribution datasets. The class names for
    datasets are a subset of ImageNet. The `__getitem__` method projects the
    labels to the classes of ImageNet to allow zero-shot evaluation.

    Args:
        opts: An argparse.Namespace instance.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        """Initialize BaseImageNetShiftDataset."""
        BaseImageClassificationDataset.__init__(
            self,
            opts=opts,
            *args,
            **kwargs,
        )
        # The class ids are converted to their equivalent ImageNet class ids
        # We manually set the n_classes and overwrite the n_classes set by
        # ImageFolder
        self.n_classes = 1000

        self.post_init_checks()

    def post_init_checks(self) -> None:
        """Verify the dataset is correctly initialized. Also called in testing."""
        if self.is_training:
            raise Exception(
                "{} can only be used for evaluation".format(self.__class__.__name__)
            )
        model_classes = getattr(self.opts, "model.classification.n_classes")
        # Note: ImageNet distribution shift subsets can have classes less than 1000
        # In such a case, a proper mapping from ImageNet classes to ImageNet distribution shift dataset needs to be done.
        assert (
            self.n_classes <= model_classes
        ), f"The dataset expects {self.n_classes} unique labels, but the model is trained on {model_classes} unique labels. "

    @staticmethod
    def class_id_to_imagenet_class_id(class_id: int) -> int:
        """Return the corresponding class index from ImageNet given a class index."""
        raise NotImplementedError(
            "Subclasses should implement the mapping to imagenet class ids."
        )

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Return the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w,
                sample_index)

        Returns:
            A dictionary with `samples`, `sample_id` and `targets` as keys corresponding
            to input, index and label of a sample, respectively.

        Shapes:
            The output data dictionary contains three keys (samples, sample_id, and
            target). The values of these keys has the following shapes:
                data["samples"]: Shape is [Channels, Height, Width]
                data["sample_id"]: Shape is 1
                data["targets"]: Shape is 1
        """
        data = BaseImageClassificationDataset.__getitem__(self, sample_size_and_index)
        data["targets"] = self.class_id_to_imagenet_class_id(data["targets"])

        return data
