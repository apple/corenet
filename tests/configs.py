#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any

from corenet.options.opts import get_training_arguments
from corenet.options.utils import load_config_file


def get_config(
    config_file: str = None, disable_ddp_distributed: bool = True
) -> argparse.Namespace:
    """Produces a resolved config (i.e. opts) object to be used in tests.

    Args:
        config_file: If provided, the contents of the @config_file path will override
          the default configs.
        disable_ddp_distributed: ``ddp.distributed`` config entry is not defined in
          the parser, but rather set by the entrypoints on the fly based on the
          availability of multiple gpus. In the tests, we usually don't want to use
          ``ddp.distributed``, even if multiple gpus are available.
    """
    parser = get_training_arguments(parse_args=False)
    opts = parser.parse_args([])
    setattr(opts, "common.config_file", config_file)
    opts = load_config_file(opts)

    if disable_ddp_distributed:
        setattr(opts, "ddp.use_distributed", False)

    return opts


# If slow, this can be turned into a "session"-scoped fixture
# @pytest.fixture(scope='session')
def default_training_opts() -> argparse.Namespace:
    opts = get_training_arguments(args=[])
    return opts


def modify_attr(opts: argparse.Namespace, key: str, value: Any) -> None:
    """Similar to the builtin setattr() function, but ensures the key already exists to
    avoid typos or missed renames during refactoring.
    """
    assert hasattr(opts, key), f"Invalid attribute {key}."
    setattr(opts, key, value)
