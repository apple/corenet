#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Any, Callable, List, Mapping

import pytest
import torch

from corenet.data.collate_fns.collate_functions import (
    default_collate_fn,
    image_classification_data_collate_fn,
    unlabeled_image_data_collate_fn,
)


@pytest.mark.parametrize(
    "collate_fn, channels_last",
    [
        (image_classification_data_collate_fn, True),
        (image_classification_data_collate_fn, False),
        (unlabeled_image_data_collate_fn, True),
        (unlabeled_image_data_collate_fn, False),
        (default_collate_fn, True),
    ],
)
def test_image_data_collate_functions(
    collate_fn: Callable[
        [List[Mapping[str, Any]], argparse.Namespace], Mapping[str, Any]
    ],
    channels_last: bool,
) -> None:
    num_samples = 3
    samples = []
    for ind in range(num_samples):
        samples.append(
            {"samples": torch.ones((3, 5, 5)), "sample_id": ind, "targets": ind}
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common.channels-last", dest="common.channels_last", action="store_true"
    )
    parser.add_argument(
        "--no-common.channels-last", dest="common.channels_last", action="store_false"
    )
    if channels_last:
        opts = parser.parse_args(["--common.channels-last"])
    else:
        opts = parser.parse_args(["--no-common.channels-last"])

    output = collate_fn(samples, opts)

    assert output["samples"].shape == (num_samples, 3, 5, 5)
    assert output["sample_id"].shape == (num_samples,)
    assert output["targets"].shape == (num_samples,)
    assert output["sample_id"].tolist() == list(range(num_samples))

    if (
        collate_fn == image_classification_data_collate_fn
        or collate_fn == default_collate_fn
    ):
        assert output["targets"].tolist() == list(range(num_samples))
    elif collate_fn == unlabeled_image_data_collate_fn:
        assert output["targets"].tolist() == [0] * num_samples
    else:
        raise ValueError("Trying to test unknown collate function.")
