#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

from torch import nn

from corenet.constants import is_test_env
from corenet.modeling.models import get_model
from corenet.options.utils import extract_opts_with_prefix_replacement
from corenet.utils import logger


def build_cls_teacher_from_opts(opts: argparse.Namespace) -> nn.Module:
    """Helper function to build a classification teacher model from command-line arguments

    Args:
        opts: command-line arguments

    Returns:
        A teacher model
    """
    pretrained_model = getattr(opts, "teacher.model.classification.pretrained")

    pytest_env = is_test_env()
    if not pytest_env and pretrained_model is None:
        logger.error(
            "For distillation, please specify teacher weights using teacher.model.classification.pretrained"
        )
    teacher_opts = extract_opts_with_prefix_replacement(
        opts, "teacher.model.", "model."
    )

    # build teacher model
    return get_model(teacher_opts, category="classification")
