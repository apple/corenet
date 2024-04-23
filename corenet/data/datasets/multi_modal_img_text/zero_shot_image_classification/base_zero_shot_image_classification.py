#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List, Tuple

from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.templates import (
    generate_text_prompts_clip,
)


class BaseZeroShotImageClassificationDataset(BaseImageClassificationDataset):
    """Base dataset class for zero shot image classification tasks.

    ...note:
        The directory structure for zero-shot image classification datasets should be the same
        as the image classification datasets. See 'BaseImageClassificationDataset' for more details.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == BaseZeroShotImageClassificationDataset:
            group = parser.add_argument_group(cls.__name__)

            group.add_argument(
                "--dataset.multi-modal-img-text.zero-shot-img-cls-dataset-name",
                type=str,
                default=None,
                help="Name of the dataset for zero-shot image classification evaluation. Defaults to None.",
            )
        return parser

    def __getitem__(self, img_index: int) -> Tuple[str, List[List[str]], int]:
        """Returns image path and text templates for a given image index.

        Args:
            img_index: Index of the image.

        Returns:
            Tuple containing image path, list of captions, and image label
        """
        img_path, image_label = self.samples[img_index]
        return img_path, self.text_prompts, image_label

    @property
    def class_names(self) -> List[str]:
        """Returns the list containing the name of the classes in the dataset.

        The order of class names in the returned list determine the numerical class
        label.
        """
        raise NotImplementedError(
            "Sub-classes should define `class_names` that returns the list of class"
            " names in the order of class labels."
        )

    @property
    def text_prompts(self) -> List[List[str]]:
        """Generates text prompts.

        A nested list that represents prompts for multiple classes is returned. Each inner list contains
        a list of prompts for a specific class.
        """
        class_names = self.class_names
        text_prompts = []
        for class_name in class_names:
            text_prompts.append(self.generate_text_prompts(class_name.lower()))
        return text_prompts

    def generate_text_prompts(self, class_name: str) -> List[str]:
        """Return a list of prompts for the given class name."""
        return generate_text_prompts_clip(class_name)
