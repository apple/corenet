#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Any, List

import pytest

from corenet.options.opts import get_training_arguments
from corenet.utils.download_utils import download_assets_in_parallel


def dummy_download_fn(index: int, local_dst_dir: str, args, kwargs) -> None:
    """Dummy download function.

    Tests if kwargs passed from 'download_assets_in_parallel' can be accessed inside 'dummy_download_fn'.
    """
    dummy_kwarg_data = kwargs.get("dummy_kwarg")
    # Indexing should not raise an error.
    dummy_kwarg_data[index]


@pytest.mark.parametrize("asset_names", [["a", "b", "c", "d", "e"], [1, 2, 3], [1]])
def test_download_assets_in_parallel(asset_names: List[Any]) -> None:
    """Test for download_assets_in_parallel function.

    Args:
        asset_names: A list of assets that are handled by 'download_func' in 'download_assets_in_parallel'.
    """
    function_kwargs = {"dummy_kwarg": asset_names}
    opts = get_training_arguments(parse_args=True, args=[])

    record_indices = download_assets_in_parallel(
        opts=opts,
        local_dst_dir="trash/dummy_test",
        num_assets=len(asset_names),
        download_func=dummy_download_fn,
        **function_kwargs,
    )
    assert len(record_indices) == len(asset_names)
