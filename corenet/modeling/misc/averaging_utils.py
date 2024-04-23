#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from copy import deepcopy
from typing import Optional

import torch
from torch import nn


class EMA(object):
    """
    For a given model, this class computes the exponential moving average of weights

    Args:
        model (torch.nn.Module): Model
        ema_momentum (Optional[float]): Momentum value shows the contribution of weights at current iteration. Default: 0.0005
        device (Optional[str]): Device (CPU or GPU) on which model resides. Default: cpu
    """

    def __init__(
        self,
        model: nn.Module,
        ema_momentum: Optional[float] = 0.0005,
        device: Optional[str] = "cpu",
        *args,
        **kwargs
    ) -> None:
        # make a deep copy of the model for accumulating moving average of parameters and set to eval mode
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.momentum = ema_momentum
        self.device = device
        if device:
            self.ema_model.to(device=device)
        self.ema_has_module = hasattr(self.ema_model, "module")
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update_parameters(self, model):
        # correct a mismatch in state dict keys
        has_module = hasattr(model, "module") and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                if has_module:
                    # .module is added if we use DistributedDataParallel or DataParallel wrappers around model
                    k = "module." + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_((ema_v * (1.0 - self.momentum)) + (self.momentum * model_v))


def arguments_ema(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="EMA", description="Exponential moving average arguments"
    )
    group.add_argument(
        "--ema.enable", action="store_true", help="Use exponential moving average"
    )
    group.add_argument(
        "--ema.momentum",
        type=float,
        default=0.0001,
        help="EMA momentum. Defaults to 0.0001.",
    )
    return parser
