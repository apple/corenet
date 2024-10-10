#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import copy
import random
from typing import List

import numpy as np
import pytest

from corenet.data.sampler import build_sampler
from tests.configs import get_config


@pytest.mark.parametrize(
    "tasks, batch_size, num_repeats, trunc_ra_sampler",
    [
        (["task_1"], 1, 4, True),
        (["task_1", "task_2"], 2, 8, True),
        (["task_1"], 1, 4, False),
        (["task_1", "task_2"], 2, 8, False),
    ],
)
def test_chain_sampler(
    tasks: List[str],
    config_file: str,
    batch_size: int,
    num_repeats: int,
    trunc_ra_sampler: bool,
) -> None:
    opts = get_config(config_file=config_file)

    sampler_list = []
    n_data_samples = {}
    samples = 1000
    total_expected_samples = 0
    for task_name in tasks:
        # we randomly decide number of repeats so that number of repeats for each task is different
        # Expected result is deterministic, despite this randomness.
        reps = random.choice([1, num_repeats])

        sampler_dict = {
            "task_name": task_name,
            "train_batch_size0": batch_size,
            "sampler_config": {
                "name": "variable_batch_sampler",
                "truncated_repeat_aug_sampler": trunc_ra_sampler,
                "num_repeats": reps,
                "vbs": {
                    "crop_size_width": 512,
                    "crop_size_height": 512,
                    "max_n_scales": 10,
                    "min_crop_size_width": 256,
                    "max_crop_size_width": 768,
                    "min_crop_size_height": 256,
                    "max_crop_size_height": 768,
                    "check_scale": 16,
                },
            },
        }
        total_expected_samples += samples * (1 if trunc_ra_sampler else reps)
        n_data_samples[task_name] = samples
        sampler_list.append(sampler_dict)

    setattr(opts, "sampler.chain_sampler", sampler_list)

    sampler = build_sampler(opts, n_data_samples=n_data_samples, is_training=True)

    np.testing.assert_equal(len(sampler), total_expected_samples)


@pytest.mark.parametrize(
    "tasks, batch_size, num_repeats, truncated_rep",
    [
        (["task_1"], 1, 4, True),
        (["task_1", "task_2"], 2, 2, True),
        (["task_1"], 1, 2, False),
        (["task_1", "task_2"], 2, 4, False),
    ],
)
def test_sampling_mode(
    tasks: List[str],
    config_file: str,
    batch_size: int,
    num_repeats: int,
    truncated_rep: bool,
) -> None:
    n_samples = 8

    sampler_list = []
    n_data_samples = {}
    total_expected_samples = 0
    for task_name in tasks:
        sampler_dict = {
            "task_name": task_name,
            "train_batch_size0": batch_size,
            "sampler_config": {
                "name": "batch_sampler",
                "bs": {"crop_size_width": 512, "crop_size_height": 512},
            },
        }
        total_expected_samples += n_samples
        n_data_samples[task_name] = n_samples
        sampler_list.append(sampler_dict)

    n_repeats_ = num_repeats
    if truncated_rep:
        # when truncated repetition is enabled, then total samples in the dataset
        # are the same as before repetition
        n_repeats_ = 1

    expected_seq_out = [
        t for t in tasks for i in range(n_samples) for j in range(n_repeats_)
    ]
    expected_interleave_order = [
        t
        for _ in range(n_samples // batch_size)
        for _ in range(n_repeats_)
        for t in tasks
        for _ in range(batch_size)
    ]

    for mode in ["sequential", "interleave"]:
        opts = get_config(config_file=config_file)

        setattr(opts, "sampler.chain_sampler", copy.deepcopy(sampler_list))

        setattr(opts, "sampler.chain_sampler_mode", mode)
        setattr(opts, "sampler.truncated_repeat_aug_sampler", truncated_rep)
        setattr(opts, "sampler.num_repeats", num_repeats)

        sampler = build_sampler(opts, n_data_samples=n_data_samples, is_training=True)

        sampler_order = [b[len(b) - 1] for batch in sampler for b in batch]
        if mode == "sequential":
            np.testing.assert_equal(expected_seq_out, sampler_order)
        else:
            np.testing.assert_equal(expected_interleave_order, sampler_order)


def pytest_generate_tests(metafunc):
    configs = ["tests/data/samplers/test_chain_sampler_config.yaml"]
    metafunc.parametrize("config_file", configs)
