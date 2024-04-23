#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
import pytest

from corenet.data.datasets.utils import common


def _test_data():
    num_samples_per_class = {"a": 5, "b": 10, "c": 15, "d": 20}
    num_classes = len(num_samples_per_class)
    num_total_samples = int(np.sum(list(num_samples_per_class.values())))
    min_num_samples_per_class = np.min(list(num_samples_per_class.values()))

    labels = []
    for class_label, num_samples in num_samples_per_class.items():
        labels += [class_label] * num_samples

    return (
        labels,
        num_samples_per_class,
        min_num_samples_per_class,
        num_classes,
        num_total_samples,
    )


def test_select_specified_number_of_samples_per_category():
    (
        labels,
        num_samples_per_class,
        min_num_samples_per_class,
        num_classes,
        _,
    ) = _test_data()
    random_seed = 0
    for num_samples_to_select_per_class in range(min_num_samples_per_class + 1):
        if num_samples_to_select_per_class > 0:
            selected_sample_indices = common.select_samples_by_category(
                sample_category_labels=labels,
                random_seed=random_seed,
                num_samples_per_category=num_samples_to_select_per_class,
            )
            assert (
                len(selected_sample_indices)
                == num_samples_to_select_per_class * num_classes
            )

            for c in num_samples_per_class:
                num_selected_samples_in_class_c = len(
                    [ind for ind in selected_sample_indices if labels[ind] == c]
                )
                assert (
                    num_selected_samples_in_class_c == num_samples_to_select_per_class
                )
        else:
            with pytest.raises(ValueError):
                common.select_samples_by_category(
                    sample_category_labels=labels,
                    random_seed=random_seed,
                    num_samples_per_category=num_samples_to_select_per_class,
                )


def test_select_specified_percentage_of_samples_per_category():
    labels, num_samples_per_class, _, _, _ = _test_data()
    random_seed = 0
    percentages = [0, 20, 40, 60, 80, 100, 120]
    for percentage in percentages:
        if 0 < percentage <= 100:
            selected_sample_indices = common.select_samples_by_category(
                sample_category_labels=labels,
                random_seed=random_seed,
                percentage_of_samples_per_category=percentage,
            )
            num_expected_samples = np.sum(
                [int(val * percentage / 100) for val in num_samples_per_class.values()]
            )
            assert len(selected_sample_indices) == num_expected_samples

            for label, num_samples in num_samples_per_class.items():
                num_selected_samples_in_class = len(
                    [ind for ind in selected_sample_indices if labels[ind] == label]
                )
                assert num_selected_samples_in_class == int(
                    num_samples * percentage / 100
                )
        else:
            with pytest.raises(ValueError):
                common.select_samples_by_category(
                    sample_category_labels=labels,
                    random_seed=random_seed,
                    percentage_of_samples_per_category=percentage,
                )


def test_select_samples_by_category_with_default_arguments():
    labels, num_samples_per_class, _, _, num_total_samples = _test_data()

    random_seed = 0
    selected_sample_indices = common.select_samples_by_category(
        sample_category_labels=labels, random_seed=random_seed
    )
    assert len(selected_sample_indices) == num_total_samples

    for label, num_samples in num_samples_per_class.items():
        num_selected_samples_in_class = len(
            [ind for ind in selected_sample_indices if labels[ind] == label]
        )
        assert num_selected_samples_in_class == num_samples


def test_reproducibility_of_sample_selection_by_category():
    labels, _, min_num_samples_per_class, _, _ = _test_data()

    n_seeds = 5
    percentages = [20, 40, 60, 80, 100]

    for random_seed in range(n_seeds):
        for num_samples_to_select_per_class in range(1, min_num_samples_per_class + 1):
            selected_sample_indices1 = common.select_samples_by_category(
                sample_category_labels=labels,
                random_seed=random_seed,
                num_samples_per_category=num_samples_to_select_per_class,
            )
            selected_sample_indices2 = common.select_samples_by_category(
                sample_category_labels=labels,
                random_seed=random_seed,
                num_samples_per_category=num_samples_to_select_per_class,
            )
            assert selected_sample_indices1 == selected_sample_indices2

        for percentage in percentages:
            selected_sample_indices1 = common.select_samples_by_category(
                sample_category_labels=labels,
                random_seed=random_seed,
                percentage_of_samples_per_category=percentage,
            )

            selected_sample_indices2 = common.select_samples_by_category(
                sample_category_labels=labels,
                random_seed=random_seed,
                percentage_of_samples_per_category=percentage,
            )
            assert selected_sample_indices1 == selected_sample_indices2


def test_file_has_valid_image_extension():
    for ext in common.IMG_EXTENSIONS:
        assert common.file_has_valid_image_extension("dummy" + ext)


def test_select_random_subset():
    num_total_samples = 10
    for num_samples_to_select in range(num_total_samples):
        if num_samples_to_select > 0:
            indices1 = common.select_random_subset(
                random_seed=0,
                num_total_samples=num_total_samples,
                num_samples_to_select=num_samples_to_select,
            )
            assert len(indices1) == num_samples_to_select

            indices2 = common.select_random_subset(
                random_seed=1,
                num_total_samples=num_total_samples,
                num_samples_to_select=num_samples_to_select,
            )
            assert len(indices2) == num_samples_to_select

            assert indices1 != indices2
        else:
            with pytest.raises(ValueError):
                common.select_random_subset(
                    random_seed=0,
                    num_total_samples=num_total_samples,
                    num_samples_to_select=num_samples_to_select,
                )

    for percentage_of_samples in [0, 20, 40, 60, 80, 100, 120]:
        if 0 < percentage_of_samples <= 100:
            indices1 = common.select_random_subset(
                random_seed=0,
                num_total_samples=num_total_samples,
                percentage_of_samples_to_select=percentage_of_samples,
            )
            assert len(indices1) == percentage_of_samples * num_total_samples / 100

            indices2 = common.select_random_subset(
                random_seed=1,
                num_total_samples=num_total_samples,
                percentage_of_samples_to_select=percentage_of_samples,
            )
            assert len(indices2) == percentage_of_samples * num_total_samples / 100

            assert indices1 != indices2
        else:
            with pytest.raises(ValueError):
                common.select_random_subset(
                    random_seed=0,
                    num_total_samples=num_total_samples,
                    percentage_of_samples_to_select=percentage_of_samples,
                )


def test_get_image_paths():
    expected_img_paths = {
        "tests/data/datasets/classification/dummy_images/training/class1/dummy_image1.jpg",
        "tests/data/datasets/classification/dummy_images/training/class1/dummy_image2.jpg",
        "tests/data/datasets/classification/dummy_images/training/class2/dummy_image1.jpg",
        "tests/data/datasets/classification/dummy_images/training/class2/dummy_image2.jpg",
        "tests/data/datasets/classification/dummy_images/validation/class1/dummy_image1.jpg",
        "tests/data/datasets/classification/dummy_images/validation/class1/dummy_image2.jpg",
        "tests/data/datasets/classification/dummy_images/validation/class2/dummy_image1.jpg",
        "tests/data/datasets/classification/dummy_images/validation/class2/dummy_image2.jpg",
    }
    img_paths = common.get_image_paths(
        "tests/data/datasets/classification/dummy_images"
    )
    assert set(img_paths) == expected_img_paths
