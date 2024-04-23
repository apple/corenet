#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import collections
import os
import re

import yaml

from corenet.options.errors import UnrecognizedYamlConfigEntry
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master
from corenet.utils.download_utils import get_local_path

try:
    # Workaround for DeprecationWarning when importing Collections
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

DEFAULT_CONFIG_DIR = "config"
META_PARAMS_REGEX = r"tasks|include_configs"
# To remove dummy entries corresponding to anchors, which are defined with a prefix `_anchor_`
META_PARAMS_REGEX += r"|_anchor_.*"
try:
    from corenet.internal.utils.opts import (
        META_PARAMS_REGEX as INTERNAL_META_PARAMS_REGEX,
    )
except ModuleNotFoundError:
    pass  # public version does not contain "internal"
else:
    META_PARAMS_REGEX += "|" + INTERNAL_META_PARAMS_REGEX


def flatten_yaml_as_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections_abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_file(opts):
    config_file_name = getattr(opts, "common.config_file", None)
    if config_file_name is None:
        return opts
    is_master_node = is_master(opts)

    if is_master_node:
        config_file_name = get_local_path(opts=opts, path=config_file_name)

    if not os.path.isfile(config_file_name):
        if len(config_file_name.split("/")) == 1:
            # loading files from default config folder
            new_config_file_name = "{}/{}".format(DEFAULT_CONFIG_DIR, config_file_name)
            if not os.path.isfile(new_config_file_name) and is_master_node:
                logger.error(
                    "Configuration file neither exists at {} nor at {}".format(
                        config_file_name, new_config_file_name
                    )
                )
            else:
                config_file_name = new_config_file_name
        else:
            # If absolute path of the file is passed
            if not os.path.isfile(config_file_name) and is_master_node:
                logger.error(
                    "Configuration file does not exists at {}".format(config_file_name)
                )

    setattr(opts, "common.config_file", config_file_name)
    with open(config_file_name, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)
                elif "local_" not in k and not re.match(META_PARAMS_REGEX, k):
                    logger.warning(UnrecognizedYamlConfigEntry(k))
        except yaml.YAMLError as exc:
            if is_master_node:
                logger.error(
                    "Error while loading config file: {}. Error message: {}".format(
                        config_file_name, str(exc)
                    )
                )

    # override arguments
    override_args = getattr(opts, "override_args", None)
    if override_args is not None:
        for override_k, override_v in override_args.items():
            if hasattr(opts, override_k):
                setattr(opts, override_k, override_v)
            elif "local_" not in k and not re.match(META_PARAMS_REGEX, k):
                logger.warning(UnrecognizedYamlConfigEntry(override_k))

    return opts


def extend_selected_args_with_prefix(
    parser: argparse.ArgumentParser, match_prefix: str, additional_prefix: str
) -> argparse.ArgumentParser:
    """
    Helper function to select arguments with certain prefix and duplicate them with a replaced prefix.
    An example use case is distillation, where we want to add --teacher.model.* as a prefix to all --model.* arguments.

    In that case, we provide the following arguments:
    * match_prefix="--model."
    * additional_prefix="--teacher.model."

    Args:
        match_prefix: Prefix to select arguments for duplication.
            The value should start with "--", contain no underscores, and with ".".
        additional_prefix: Prefix to replace the @match_prefix in duplicated arguments.
            The value should start with "--", contain no underscores, and with ".".
    """
    # all arguments are stored as actions
    options = parser._actions

    regexp = r"--[^_]+\."
    assert re.match(
        regexp, match_prefix
    ), f"match prefix '{match_prefix}' should match regexp '{regexp}'"
    assert re.match(
        regexp, additional_prefix
    ), f"additional prefix '{additional_prefix}' should match regexp '{regexp}'"

    for option in options:
        option_strings = option.option_strings
        # option strings are stored as a list
        for option_string in option_strings:
            if option_string.startswith(match_prefix):
                parser.add_argument(
                    option_string.replace(match_prefix, additional_prefix),
                    nargs=(
                        "?"
                        if isinstance(option, argparse._StoreTrueAction)
                        else option.nargs
                    ),
                    const=option.const,
                    default=option.default,
                    type=option.type,
                    choices=option.choices,
                    help=option.help,
                    metavar=option.metavar,
                )
    return parser


def extract_opts_with_prefix_replacement(
    opts: argparse.Namespace,
    match_prefix: str,
    replacement_prefix: str,
) -> argparse.Namespace:
    """
    Helper function to extract a copy options with certain prefix and return them with an alternative prefix.
    An example usage is distillation, when we have used @extend_selected_args_with_prefix to add --teacher.model.*
        arguments to argparser, and now we want to re-use the handlers of model.* opts by teacher.model.* opts

    Args:
        match_prefix: Prefix to select opts for extraction.
            The value should not contain dashes and should end with "."
        replacement_prefix: Prefix to replace the @match_prefix
            The value should not contain dashes and should end with "."
    """
    regexp = r"[^-]+\."
    assert re.match(
        regexp, match_prefix
    ), f"match prefix '{match_prefix}' should match regexp '{regexp}'"
    assert re.match(
        regexp, replacement_prefix
    ), f"replacement prefix '{replacement_prefix}' should match regexp '{regexp}'"

    opts_dict = vars(opts)
    result_dict = {
        # replace teacher with empty string in "teacher.model.*" to get model.*
        key.replace(match_prefix, replacement_prefix): value
        for key, value in opts_dict.items()
        # filter keys related to teacher
        if key.startswith(match_prefix)
    }

    return argparse.Namespace(**result_dict)
