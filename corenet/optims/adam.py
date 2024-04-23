#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Iterable, Union

from torch import Tensor
from torch.optim import Adam

from corenet.optims import OPTIM_REGISTRY
from corenet.optims.base_optim import BaseOptim


@OPTIM_REGISTRY.register(name="adam")
class AdamOptimizer(BaseOptim, Adam):
    """
    `Adam <https://arxiv.org/abs/1412.6980>`_ optimizer

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
        beta1 = getattr(opts, "optim.adam.beta1")
        beta2 = getattr(opts, "optim.adam.beta2")
        ams_grad = getattr(opts, "optim.adam.amsgrad")
        eps = getattr(opts, "optim.adam.eps", None)
        Adam.__init__(
            self,
            params=model_params,
            lr=self.lr,
            betas=(beta1, beta2),
            eps=self.eps if eps is None else eps,
            weight_decay=self.weight_decay,
            amsgrad=ams_grad,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add arguments for ADAM optimizer"""
        if cls != AdamOptimizer:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--optim.adam.beta1",
            type=float,
            default=0.9,
            help="Value of Beta1 in ADAM optimizer. Defaults to 0.9.",
        )
        group.add_argument(
            "--optim.adam.beta2",
            type=float,
            default=0.98,
            help="Value of Beta2 in ADAM optimizer. Defaults to 0.98.",
        )
        group.add_argument(
            "--optim.adam.amsgrad",
            action="store_true",
            default=False,
            help="Use AMSGrad in ADAM. Defaults to False.",
        )
        group.add_argument(
            "--optim.adam.eps",
            type=float,
            default=None,
            help="Value of epsilon in Adam optimizer. Defaults to None."
            "When this value is None, the default value in base optimizer is used.",
        )
        return parser
