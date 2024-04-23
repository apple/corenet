#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Dict

import pytest
import torch

from corenet.data.datasets.utils import video


def _get_invalid_label(timestamp: float) -> Dict:
    return {
        "x0": -1,
        "x1": -1,
        "y0": -1,
        "y1": -1,
        "speaking_label": -1,
        "looking_label": -1,
        "is_visible": False,
        "timestamp": timestamp,
    }


def test_fetch_frame_label():
    # x0 goes from 0 to 1 over different periods of time, and speaking_label
    # toggles. Other labels are unchanged.
    rectangles_dict = {
        "identity0": [
            {
                "timestamp": 0.0,
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "speaking_label": False,
                "is_visible": True,
                "looking_label": False,
            },
            {
                "timestamp": 1.0,
                "x0": 1,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "speaking_label": True,
                "is_visible": True,
                "looking_label": False,
            },
        ],
        "identity1": [
            {
                "timestamp": 0.0,
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "speaking_label": True,
                "is_visible": True,
                "looking_label": False,
            },
            {
                "timestamp": 0.25,
                "x0": 1,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "speaking_label": False,
                "is_visible": True,
                "looking_label": False,
            },
        ],
    }

    def _fetch_frame_label_fn(timestamp):
        return video._fetch_frame_label(
            "speaking_label",
            timestamp,
            rectangles_dict,
            carry_over_keys=["speaking_label", "looking_label", "is_visible"],
            required_keys=["speaking_label", "looking_label"],
        )

    # Test the first label.
    ret = _fetch_frame_label_fn(0)
    assert ret["identity0"] == rectangles_dict["identity0"][0]
    assert ret["identity1"] == rectangles_dict["identity1"][0]

    # Test an interpolated label.
    ret = _fetch_frame_label_fn(0.25)
    expected_id0 = rectangles_dict["identity0"][0].copy()
    expected_id0["x0"] = 0.25
    expected_id0["timestamp"] = 0.25
    assert ret["identity0"] == expected_id0

    assert ret["identity1"] == rectangles_dict["identity1"][1]

    # We should get an interpolated label for identity0, and an "empty" label
    # for identity1 since there are no more labels.
    ret = _fetch_frame_label_fn(0.5)
    expected_id0 = rectangles_dict["identity0"][0].copy()
    expected_id0["x0"] = 0.50
    expected_id0["timestamp"] = 0.50
    expected_id0["speaking_label"] = False
    assert ret["identity0"] == expected_id0
    assert ret["identity1"] == _get_invalid_label(0.50)

    # We should get empty labels.
    ret = _fetch_frame_label_fn(-1)
    expected_id0 = _get_invalid_label(-1)
    assert ret["identity0"] == expected_id0

    expected_id1 = _get_invalid_label(-1)
    assert ret["identity1"] == expected_id1


RECTANGLES_DICT = {
    "identity0": [
        {
            "timestamp": 0.0,
            "x0": 0,
            "y0": 0,
            "x1": 1,
            "y1": 1,
            "action_label": 0,
            "is_visible": True,
        },
        {
            "timestamp": 0.1,
            "x0": 0,
            "y0": 0,
            "x1": 1,
            "y1": 1,
            "action_label": 1,
            "is_visible": True,
        },
        {
            "timestamp": 0.2,
            "x0": 0,
            "y0": 0,
            "x1": 1,
            "y1": 1,
            "action_label": 1,
            "is_visible": True,
        },
        {
            "timestamp": 0.3,
            "x0": 0,
            "y0": 0,
            "x1": 1,
            "y1": 1,
            "action_label": 0,
            "is_visible": True,
        },
        {
            "timestamp": 1.0,
            "x0": 0,
            "y0": 0,
            "x1": 1,
            "y1": 1,
            "action_label": 1,
            "is_visible": True,
        },
        {
            "timestamp": 1.1,
            "x0": 0,
            "y0": 0,
            "x1": 1,
            "y1": 1,
            "action_label": 1,
            "is_visible": True,
        },
    ],
}

# Below is a schematic of @rectangles_dict, where the asterisks denote labels
# (with the corresponding class int above). The letters below denote the
# regions.
#
#    0       1       1       0              1       1
#    *       *       *       *              *       *
# +++++++++++++++++++++++++++++++++ // ++++++++++++++++++
#        A   [       B       )   C      C   [   D   )   E

RECTANGLES_TEST_CASES = [
    (0.04, (0, -1.0), "A"),
    (0.06, (0, -1.0), "A"),
    (0.11, (1, 0.01 / 0.2), "B"),
    (0.15, (1, 0.05 / 0.2), "B"),
    (0.19, (1, 0.09 / 0.2), "B"),
    (0.23, (1, 0.13 / 0.2), "B"),
    (0.31, (-1, -1.0), "C"),
    (0.90, (-1, -1.0), "C"),
    (1.05, (1, 0.05 / 0.1), "D"),
    (1.11, (-1, -1.0), "E"),
]


@pytest.mark.parametrize("test_case", RECTANGLES_TEST_CASES)
def test_fetch_frame_with_progress(test_case):

    timestamp, (expected_action_label, expected_progress), _ = test_case

    ret0 = video._fetch_frame_label(
        "action_label",
        timestamp,
        RECTANGLES_DICT,
        0.2,
        set([1]),
        ["action_label"],
    )["identity0"]
    assert ret0["action_label"] == expected_action_label
    assert ret0["progress"] == pytest.approx(expected_progress)
