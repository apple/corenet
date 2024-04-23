#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict


class BaseTransformation(object):
    """
    Base class for augmentation methods
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        self.opts = opts

    def __call__(self, data: Dict) -> Dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser
