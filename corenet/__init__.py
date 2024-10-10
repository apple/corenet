#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import warnings

from corenet.constants import is_external_env, is_test_env
from corenet.utils.logger import match_warning_message

KNOWN_WARNINGS = [
    ##
    # Torchtext deprecation warning:
    ##
    match_warning_message(
        " /!\\ IMPORTANT WARNING ABOUT TORCHTEXT STATUS /!\\ Torchtext is deprecated"
        " and the last released version will be 0.18 (this one)."
    ),
    ##
    # The following warnings are raised by tests/metrics/test_coco_map.py:
    ##
    match_warning_message(
        "Creating a tensor from a list of numpy.ndarrays is extremely slow. Please"
        " consider converting the list to a single numpy.ndarray with numpy.array()"
        " before converting to a tensor."
    ),
    match_warning_message("unclosed file <"),
    ##
    # The following warnings are raised by tests/loss_fns/test_class_weighting.py:
    ##
    match_warning_message(
        "`torch.testing.assert_allclose()` is deprecated since 1.12 and will be removed"
        " in a future release. Please use `torch.testing.assert_close()` instead. You"
        " can find detailed upgrade instructions in"
        " https://github.com/pytorch/pytorch/issues/61844."
    ),
    ##
    # The following warnings are raised by "import torchvision.datasets":
    ##
    match_warning_message(
        "torch.utils._pytree._register_pytree_node is deprecated. Please use"
        " torch.utils._pytree.register_pytree_node instead."
    ),
    ##
    # The following warnings are raised by "import coremltools" (macOS only):
    ##
    match_warning_message(
        "Call to deprecated create function",
        "Note: Create unlinked descriptors is going to go away. Please use get/find"
        " descriptors from generated code or query the descriptor_pool.",
    ),
]


def suppress_known_warnings() -> None:
    """
    Suppresses warnings that are known to be safe for corenet, to avoid overwhelming
    the standard error outputs (especially with multiple subprocesses).

    Notes:
    - We should invoke this function as early as possible (i.e. in corenet/__init__.py)
        because some of the warnings are during the execution of the import statements.
        We should suppress the known warnings before importing other modules.
    - We SHOULD NOT invoke this function when corenet is imported from external
        libraries because it enters the `warnings.catch_warnings(record=True)` context
        manager without invoking its __exit__() method. This is only safe when we are
        not already inside a `with warnings.catch_warnings():` context, which is the
        case for corenet entrypoints.
    """
    try:
        # Importing corenet.internal inside the function to avoid cyclic dependency.
        from corenet.internal import KNOWN_WARNINGS_INTERNAL
    except ModuleNotFoundError:
        KNOWN_WARNINGS_INTERNAL = []

    known_warnings = KNOWN_WARNINGS + KNOWN_WARNINGS_INTERNAL
    if len(known_warnings) == 0:
        return

    if is_test_env():
        # In the test env, convert unsuppressed/unhandled warnings to errors.
        # Filters listed later take precedence over those listed before them. Invoking
        # filterwarnings("error") first so that it becomes the default when warning is
        # no handled by other filters.
        warnings.filterwarnings("error")

    warnings.catch_warnings(record=True).__enter__()
    for known_warning in known_warnings:
        warnings.filterwarnings("ignore", message=known_warning)


if not is_external_env():
    suppress_known_warnings()
