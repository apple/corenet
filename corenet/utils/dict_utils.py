#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Collection, Dict, Optional


def filter_keys(
    d: Dict,
    whitelist: Optional[Collection[str]] = None,
) -> Dict:
    """Returns a copy of the input dict @d, with a subset of keys that are in
    @whitelist.

    Args:
        d: Input dictionary that will be copied with a subset of keys.
        whitelist: List of keys to keep in the output (if exist in input dict).
    """

    return {key: d[key] for key in whitelist if key in d}
