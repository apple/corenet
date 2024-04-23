#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Any, Dict, List, Tuple, Union

import pytest

from corenet.options.parse_args import JsonValidator


@pytest.mark.parametrize(
    "expected_type,valid,valid_parsed,invalid",
    [
        (None, "null", None, "1"),
        (int, "1", 1, "1.0"),
        (float, "1.0", 1.0, '"1"'),
        (float, "1", 1.0, "s"),
        (bool, "true", True, "null"),
        (List[int], "[1, 2,3]", [1, 2, 3], "{1: 2}"),
        (List[int], "[]", [], '["s"]'),
        (Tuple[int, int], "[1, 2]", (1, 2), "[1, 2, 3]"),
        (Dict[str, Tuple[int, float]], '{"x": [1, 2]}', {"x": (1, 2.0)}, '{"x": "y"}'),
        (Union[Tuple[int, Any], int], "[1,null]", (1, None), "[null,1]"),
        (Union[Tuple[int, int], int], "1", 1, '"1"'),
    ],
)
def test_json_validator(
    expected_type: type, valid: str, valid_parsed: Any, invalid: str
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=JsonValidator(expected_type))

    class ArgparseFailure(Exception):
        pass

    def _exit(status, message):
        raise ArgparseFailure(f"Unexpected argparse failure: {message}")

    parser.exit = (
        _exit  # override exit to raise exception, rather than invoking sys.exit()
    )

    opts = parser.parse_args([f"--x={valid}"])
    assert opts.x == valid_parsed
    assert repr(opts.x) == repr(valid_parsed)  # check types

    with pytest.raises(ArgparseFailure):
        parser.parse_args([f"--x={invalid}"])
