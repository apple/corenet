#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import collections
import os
import random
from typing import Any, List, Optional, Tuple, Union

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def file_has_valid_image_extension(filename: str) -> bool:
    return file_has_allowed_extension(filename, IMG_EXTENSIONS)


def file_has_allowed_extension(
    filename: str, extensions: Union[str, Tuple[str, ...]]
) -> bool:
    """Checks if a file has an allowed extension.

    Args:
        filename: Path to a file.
        extensions: A string or a tuple of strings specifying the file extensions.

    Returns:
        True if the filename ends with one of given extensions, else False
    """
    return filename.lower().endswith(extensions)


def get_image_paths(directory: str) -> List[str]:
    """Returns a list of paths to all image files in the input directory and its subdirectories."""
    image_paths = []
    for root, _, fnames in sorted(os.walk(directory, topdown=False)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if file_has_valid_image_extension(path):
                image_paths.append(path)

    return image_paths


def select_random_subset(
    random_seed: int,
    num_total_samples: int,
    num_samples_to_select: Optional[int] = None,
    percentage_of_samples_to_select: Optional[float] = None,
) -> List[int]:
    """
    Randomly selects a subset of samples.

    Only one of `num_samples_to_select` and `percentage_of_samples_to_select` should be provided.
    Selects all the samples if neither of them are provided.

    Args:
        random_seed: An integer seed to use for random selection.
        num_total_samples: Total number of samples in the set that is being subsampled.
        num_samples_to_select: An optional integer indicating the number of samples to select.
        percentage_of_samples_to_select: An optional float in the range (0,100] indicating the percentage of
            samples to select.

    Returns:
        A list of (integer) indices of the selected samples.

     Raises:
        ValueError if both `num_samples_to_select` and `percentage_of_samples_to_select` are provided.
    """
    if (
        num_samples_to_select is not None
        and percentage_of_samples_to_select is not None
    ):
        raise ValueError(
            "Only one of `num_samples_to_select` and `percentage_of_samples_to_select` should be provided."
        )

    if num_samples_to_select is not None and num_samples_to_select < 1:
        raise ValueError("`num_samples_to_select` should be greater than 0.")

    if percentage_of_samples_to_select is not None:
        if not 0 < percentage_of_samples_to_select <= 100:
            raise ValueError(
                "`percentage_of_samples_to_select` should be in the range (0, 100]."
            )

    sample_indices = list(range(num_total_samples))
    rng = random.Random(random_seed)
    rng.shuffle(sample_indices)

    if num_samples_to_select is None and percentage_of_samples_to_select is None:
        return sample_indices

    if num_samples_to_select is None:
        num_samples_to_select = int(
            percentage_of_samples_to_select * num_total_samples / 100
        )

    num_samples_to_select = min(num_samples_to_select, num_total_samples)
    return sample_indices[:num_samples_to_select]


def select_samples_by_category(
    sample_category_labels: List[Any],
    random_seed: int,
    num_samples_per_category: Optional[int] = None,
    percentage_of_samples_per_category: Optional[float] = None,
) -> List[int]:
    """
    Randomly selects a specified number/percentage of samples from each category.

    Only one of `num_samples_per_category` and `percentage_of_samples_per_category` should be provided.
    Selects all the samples if neither of them are provided.

    Args:
        sample_category_labels: A list of category labels.
        random_seed: An integer seed to use for random selection.
        num_samples_per_category: An optional integer indicating the number of samples to select from each category.
        percentage_of_samples_per_category: An optional float in the range (0, 100] indicating the percentage of
            samples to select from each category.

    Returns:
        A list of (integer) indices of the selected samples.

    Raises:
        ValueError if both `num_samples_per_category` and `percentage_of_samples_per_category` are provided.
    """
    if (
        num_samples_per_category is not None
        and percentage_of_samples_per_category is not None
    ):
        raise ValueError(
            "Only one of `num_samples_per_category` and `percentage_of_samples_per_category` should be provided."
        )

    if num_samples_per_category is None and percentage_of_samples_per_category is None:
        return list(range(len(sample_category_labels)))

    if num_samples_per_category is not None and num_samples_per_category < 1:
        raise ValueError("`num_samples_per_category` should be greater than 0.")

    if percentage_of_samples_per_category is not None:
        if not 0 < percentage_of_samples_per_category <= 100:
            raise ValueError(
                "`percentage_of_samples_per_category` should be in the range (0, 100]."
            )

    category_specific_samples = collections.defaultdict(list)
    for ind, label in enumerate(sample_category_labels):
        category_specific_samples[label].append(ind)

    rng = random.Random(random_seed)
    selected_sample_indices = []
    for label, sample_indices in category_specific_samples.items():
        rng.shuffle(sample_indices)
        if num_samples_per_category:
            num_samples = num_samples_per_category
        else:
            num_samples = int(
                percentage_of_samples_per_category * len(sample_indices) / 100
            )
        num_samples = min(num_samples, len(sample_indices))
        selected_sample_indices += sample_indices[:num_samples]

    return selected_sample_indices
