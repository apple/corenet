#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import importlib
import os
import re
import sys
from typing import Optional, Sequence

from corenet.constants import LIBRARY_ROOT
from corenet.utils import logger


def import_modules_from_folder(
    folder_name: str, extra_roots: Optional[Sequence[str]] = None
) -> None:
    """Automatically import all modules from public library root folder, in addition
    to the @extra_roots directories.

    The @folder_name directory must exist in LIBRARY_ROOT, but existence in @extra_roots
    is optional.

    Args:
        folder_name: Name of the folder to search for its internal and public modules.
        extra_roots: By default, this function only imports from
            `LIBRARY_ROOT/{folder_name}/**/*.py`. For any extra_root provided, it will
            also import `LIBRARY_ROOT/{extra_root}/{folder_name}/**/*.py` modules.
    """
    if not LIBRARY_ROOT.joinpath(folder_name).exists():
        logger.error(
            f"{folder_name} doesn't exist in the public library root directory."
        )

    base_dirs = ["."]
    if extra_roots is not None:
        base_dirs += sorted(extra_roots)
    for base_dir in base_dirs:
        if base_dir.startswith("corenet/") and folder_name.startswith("corenet/"):
            base_dir = os.path.join(base_dir, re.sub("^corenet/", "", folder_name))
        else:
            base_dir = os.path.join(base_dir, folder_name)
        for path in sorted(LIBRARY_ROOT.glob(os.path.join(base_dir, "**/*.py"))):
            filename = path.name
            if (
                filename[0] not in (".", "_")
                and not filename.endswith("_test.py")
                and not filename.startswith("test_")
            ):
                module_name = str(
                    path.relative_to(LIBRARY_ROOT).with_suffix("")
                ).replace(os.sep, ".")
                importlib.import_module(module_name)


# For some libraries, the name of the module to be imported is different with the name
# of the library (i.e. to install using pip).
MODULE_NAME_2_LIBRARY_NAME_MAPPING = {
    "ffmpeg": "ffmpeg-python",
}


def ensure_library_is_available(module_name: str) -> None:
    """Ensures @module_name is imported, when the corresponding library is an optional
    dependency.

    Args:
        module_name: Name of the module that should be imported before calling this
            function using the following snippet:
            ```
                try:
                    import <module_name>
                    # OR: import <module_name>.<member1>.<member2>
                    # OR: from <module_name>.<member1> import <member2>
                except ModuleNotFoundError:
                    pass
            ```

    Returns: None if optional dependency is installed. Otherwise, raises an error.
    """
    if module_name in sys.modules:
        # The above condition has confirmed that module_name is already imported.
        return

    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        library_name = MODULE_NAME_2_LIBRARY_NAME_MAPPING.get(module_name, module_name)
        logger.error(
            f"{library_name} (an optional dependency) is not installed. Please"
            f" run 'pip install {library_name}'."
        )

    # Module is available, but not imported yet. Otherwise, the `if` condition at the
    # top of this function would have returned before reaching this line.
    raise RuntimeError(
        f"Please import {module_name} before calling"
        f' ensure_library_is_available("{module_name}")'
    )


def import_core_modules():
    """
    This function imports the core functionality of CoreNet, consisting of all modules
    that contain registered classes (e.g. datasets, models, etc.) and entrypoints.

    For further details, please see the docstring of
    ``corenet.utils.import_utils.test_ensure_pythonpath_is_not_required``.
    """
    from corenet.options.opts import get_training_arguments

    get_training_arguments(args=[])  # Imports registered classes.

    from corenet.cli.entrypoints import entrypoints

    for module_name, _ in entrypoints.values():
        importlib.import_module(module_name)
