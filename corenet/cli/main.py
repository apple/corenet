#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import importlib
from itertools import chain
from typing import Dict, List, Optional, Tuple

from corenet.cli.entrypoints import entrypoints as oss_entrypoints

try:
    from corenet.internal.cli.entrypoints import entrypoints as internal_entrypoints
except ModuleNotFoundError:
    internal_entrypoints = {}


def main(args: Optional[List[str]] = None) -> None:
    """
    We are planning to deprecate `corenet-train`, `corenet-eval`, ... commands for
    `corenet train` (the dash is removed), `corenet eval`, ... because adding/renaming
    entrypoints will require `pip install -e .`. Most users don't reinstall corenet
    after pulling the git repo. Hence, relying on a single entrypoint `corenet` with
    subcommands is more future proof.
    """
    entrypoints = {
        k.replace("corenet-", ""): v
        for k, v in chain(oss_entrypoints.items(), internal_entrypoints.items())
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("entrypoint", choices=list(entrypoints.keys()))
    entrypoint_opts, args = parser.parse_known_args(args)
    module_name, func_name = entrypoints[entrypoint_opts.entrypoint]
    getattr(importlib.import_module(module_name), func_name)(args)
