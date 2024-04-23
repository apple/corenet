#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Iterable, Union

from torch import Tensor
from torch.optim import SGD

from corenet.optims import OPTIM_REGISTRY
from corenet.optims.base_optim import BaseOptim


@OPTIM_REGISTRY.register(name="sgd")
class SGDOptimizer(BaseOptim, SGD):
    """
    `SGD <http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_ optimizer

    Args:
        opts: Command-line arguments
        model_params: Model parameters
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        model_params: Iterable[Union[Tensor, Dict]],
        *args,
        **kwargs
    ) -> None:
        BaseOptim.__init__(self, opts=opts)
        nesterov = getattr(opts, "optim.sgd.nesterov")
        momentum = getattr(opts, "optim.sgd.momentum")

        SGD.__init__(
            self,
            params=model_params,
            lr=self.lr,
            momentum=momentum,
            weight_decay=self.weight_decay,
            nesterov=nesterov,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != SGDOptimizer:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--optim.sgd.momentum",
            default=0.9,
            type=float,
            help="The value of momemtum in SGD. Defaults to 0.9",
        )
        group.add_argument(
            "--optim.sgd.nesterov",
            action="store_true",
            default=False,
            help="Use nesterov momentum in SGD. Defaults to False.",
        )
        return parser
