#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.text_tokenizer.base_tokenizer import BaseTextTokenizer
from corenet.utils import logger
from corenet.utils.registry import Registry

TOKENIZER_REGISTRY = Registry(
    "tokenizer",
    base_class=BaseTextTokenizer,
    lazy_load_dirs=[
        "corenet/data/text_tokenizer",
        "corenet/third_party/data/text_tokenizer",
    ],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_tokenizer(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # add arguments for text_tokenizer
    parser = BaseTextTokenizer.add_arguments(parser)

    # add class specific arguments
    parser = TOKENIZER_REGISTRY.all_arguments(parser)
    return parser


def build_tokenizer(opts, *args, **kwargs) -> BaseTextTokenizer:
    """Helper function to build the text tokenizer from command-line arguments.

    Args:
        opts: Command-line arguments

    Returns:
        Image projection head module.
    """
    tokenizer_name = getattr(opts, "text_tokenizer.name", None)

    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if tokenizer_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    tokenizer = TOKENIZER_REGISTRY[tokenizer_name](opts, *args, **kwargs)
    return tokenizer
