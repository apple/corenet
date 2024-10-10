#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

from corenet.options.opts import get_training_arguments
from corenet.options.utils import load_config_file


def get_config(
    config_file: Optional[str] = None,
    disable_ddp_distributed: bool = True,
    tmpdir: Optional[Path] = None,
    add_arguments: Optional[
        Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
    ] = None,
) -> argparse.Namespace:
    """Produces a resolved config (i.e. opts) object to be used in tests.

    Args:
        config_file: If provided, the contents of the @config_file path will override
            the default configs.
        disable_ddp_distributed: ``ddp.distributed`` config entry is not defined in
            the parser, but rather set by the entrypoints on the fly based on the
            availability of multiple gpus. In the tests, we usually don't want to use
            ``ddp.distributed``, even if multiple gpus are available.
        tmpdir: If provided, overrides `opts.artifacts_root` and `opts.logs_root` to
            "{tmpdir}/artifacts" and "{tmpdir}/logs". Defaults to None (no-op).
        add_arguments: If provided, wraps the argument parser to modify the parser or
            to add additional arguments dynamically. Defaults to None.
    """
    if config_file is not None:
        args = ["--common.config-file", config_file]
    else:
        args = []

    opts = get_training_arguments(args=args, add_arguments=add_arguments)

    if disable_ddp_distributed:
        setattr(opts, "ddp.use_distributed", False)

    if tmpdir is not None:
        setattr(opts, "common.results_loc", str(tmpdir / "results"))
        setattr(opts, "common.logs_loc", str(tmpdir / "logs"))
        Path(getattr(opts, "common.results_loc")).mkdir(exist_ok=True, parents=True)
        Path(getattr(opts, "common.logs_loc")).mkdir(exist_ok=True, parents=True)
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
