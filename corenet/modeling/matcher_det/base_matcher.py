#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse


class BaseMatcher(object):
    """
    Base class for matching anchor boxes and labels for the task of object detection
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super(BaseMatcher, self).__init__()
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add class-specific arguments"""
        return parser

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
