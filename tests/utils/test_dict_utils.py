#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from corenet.utils.dict_utils import filter_keys


def test_extract_keys():
    d = {"x": 2, "y": 3, "z": 4}

    assert filter_keys(d, ["x", "y"]) == {"x": 2, "y": 3}
    assert filter_keys(d, ["w"]) == {}
