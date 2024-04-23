#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict, Tuple

# Entrypoints is a mapping from shell executable name to (module, function) pair.
# Having too many entrypoints in setup.py limits us our ability to add features or
# refactor the code, because users who pull the latest changes will have to re-install
# corenet in order for `setup.py` changes to apply.
# A better practice is to stop introducing new entrypoints, and add subcommands to the
# main `corenet` entrypoint. Currently, `corenet train` is identical to `corenet-train`.
entrypoints: Dict[str, Tuple[str, str]] = {
    "corenet-train": ("corenet.cli.main_train", "main_worker"),
    "corenet-eval": ("corenet.cli.main_eval", "main_worker"),
    "corenet-eval-llmadapters": (
        "corenet.cli.main_eval_llmadapters",
        "main_eval_llmadapters",
    ),
    "corenet-eval-seg": ("corenet.cli.main_eval", "main_worker_segmentation"),
    "corenet-eval-det": ("corenet.cli.main_eval", "main_worker_detection"),
    "corenet-convert": ("corenet.cli.main_conversion", "main_worker_conversion"),
    "corenet": ("corenet.cli.main", "main"),
}

console_scripts = [
    f"{entrypoint} = {module}:{func}"
    for entrypoint, (module, func) in entrypoints.items()
]
