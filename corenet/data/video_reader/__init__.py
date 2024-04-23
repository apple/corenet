#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.video_reader.base_av_reader import BaseAVReader
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master
from corenet.utils.registry import Registry

VIDEO_READER_REGISTRY = Registry(
    "video_reader",
    base_class=BaseAVReader,
    lazy_load_dirs=["corenet/data/video_reader"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_video_reader(parser: argparse.ArgumentParser):
    parser = BaseAVReader.add_arguments(parser=parser)

    # add video reader specific arguments
    parser = VIDEO_READER_REGISTRY.all_arguments(parser)
    return parser


def get_video_reader(
    opts: argparse.Namespace, log: bool = True, *args, **kwargs
) -> BaseAVReader:
    """Helper function to build the video reader from command-line arguments.

    Args:
        opts: Command-line arguments
        log: When True, the video reader details will be logged to stdout.
    """

    video_reader_name = getattr(opts, "video_reader.name")

    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if video_reader_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    video_reader = VIDEO_READER_REGISTRY[video_reader_name](opts, *args, **kwargs)

    is_master_node = is_master(opts)
    if log and is_master_node:
        logger.log("Video reader details: ")
        print("{}".format(video_reader))
    return video_reader
