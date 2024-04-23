#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from corenet.utils.object_utils import apply_recursively, flatten_to_dict


def test_apply_on_values():
    d = {
        "top1": 1.112311,
        "prob_hist": {"max": [0.10003, 0.3, 0.5, 0.09997]},
        "accuracy_per_class": [0.8286, 0.9124],
    }

    new_d = apply_recursively(d, lambda x: round(x, 2))

    assert str(new_d["top1"]) == "1.11"
    assert str(new_d["prob_hist"]["max"][0]) == "0.1"


def test_flatten_to_dict():
    original = {
        "top1": 1.112311,
        "prob_hist": {"max": [0.10003, 0.3, 0.5, 0.09997]},
        "accuracy_per_class": [0.8286, 0.9124],
    }
    flattened = {
        "metric/top1": 1.112311,
        "metric/prob_hist/max_0": 0.10003,
        "metric/prob_hist/max_1": 0.3,
        "metric/prob_hist/max_2": 0.5,
        "metric/prob_hist/max_3": 0.09997,
        "metric/accuracy_per_class_0": 0.8286,
        "metric/accuracy_per_class_1": 0.9124,
    }
    assert flatten_to_dict(original, "metric") == flattened
