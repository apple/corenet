#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import argparse
import inspect
from typing import Any, Dict, Tuple

import torch
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig

from corenet.modeling.models import BaseAnyNNModel
from corenet.optims.base_optim import BaseOptim
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master

FSDP_SHARDING_STRATEGY_MAP = {
    # In full shard, parameters, gradients, and optimizer states are sharded (aka ZERO 3)
    "full_shard": ShardingStrategy.FULL_SHARD,
    # hybrid_shard is the same as full shard, except sharding is done within a node.
    # TODO: Revisit hybrid sharding in future because of the below issue.
    # https://github.com/pytorch/pytorch/issues/102904
    # "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    # In no-shard, parameters, gradients, and optimizer states are not sharded
    "no_shard": ShardingStrategy.NO_SHARD,
    # In grad_op_shard, gradients and optimizer states are sharded (aka as Zero)
    "grad_op_shard": ShardingStrategy.SHARD_GRAD_OP,
}

FSDP_DATATYPE_CONVERSION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

FSDP_BACKWARD_PREFETCH = {
    # pre enables prefetching next set of parameters before computing gradients for current set of parameters.
    "pre": BackwardPrefetch.BACKWARD_PRE,
    # post enables prefetching next set of parameters after computing gradients for current set of parameters.
    "post": BackwardPrefetch.BACKWARD_POST,
}


class FullyShardedDataParallelWrapper(FullyShardedDataParallel):
    def __init__(
        self,
        opts: argparse.Namespace,
        model: BaseAnyNNModel,
    ) -> None:

        param_dtype = getattr(opts, "fsdp.parameter_datatype")
        reduce_dtype = getattr(opts, "fsdp.gradient_reduction_datatype")
        buffer_dtype = getattr(opts, "fsdp.buffer_datatype")

        if param_dtype not in FSDP_DATATYPE_CONVERSION:
            logger.error(
                f"Supported data type for parameters in FSDP are {list(FSDP_DATATYPE_CONVERSION.keys())}. \
                    Got: {param_dtype}."
            )
        if reduce_dtype not in FSDP_DATATYPE_CONVERSION:
            logger.error(
                f"Supported data type for gradient reduction in FSDP are {list(FSDP_DATATYPE_CONVERSION.keys())}. \
                    Got: {reduce_dtype}."
            )
        if buffer_dtype not in FSDP_DATATYPE_CONVERSION:
            logger.error(
                f"Supported data type for buffer in FSDP are {list(FSDP_DATATYPE_CONVERSION.keys())}. \
                    Got: {buffer_dtype}."
            )

        prefetching_option = getattr(opts, "fsdp.backward_prefetching")
        if prefetching_option not in FSDP_BACKWARD_PREFETCH:
            logger.error(
                f"Supported backward pre-fetching options are {list(FSDP_BACKWARD_PREFETCH.keys())}. \
                Got: {prefetching_option}."
            )

        fsdp_precision_policy = MixedPrecision(
            param_dtype=FSDP_DATATYPE_CONVERSION[param_dtype],
            reduce_dtype=FSDP_DATATYPE_CONVERSION[reduce_dtype],
            buffer_dtype=FSDP_DATATYPE_CONVERSION[buffer_dtype],
        )

        fsdp_parameters = inspect.signature(FullyShardedDataParallel).parameters

        # Enabing `use_orig_params` tells FSDP not to flatten parameters, and enables us to specify different LR/weight decay values.
        # `use_orig_params` feature is available in PyTorch versions > 2.0
        extra_args_fsdp = (
            dict(use_orig_params=True)
            if "use_orig_params" in fsdp_parameters
            else dict()
        )
        if "limit_all_gathers" in fsdp_parameters and getattr(
            opts, "fsdp.limit_all_gathers"
        ):
            extra_args_fsdp["limit_all_gathers"] = True

        if "cpu_offload" in fsdp_parameters and getattr(opts, "fsdp.cpu_offload"):
            extra_args_fsdp["cpu_offload"] = CPUOffload(offload_params=True)

        sharding_strategy = getattr(opts, "fsdp.sharding_strategy")
        if sharding_strategy not in FSDP_SHARDING_STRATEGY_MAP:
            logger.error(
                f"Supported sharding strategies for FSDP are: {list(FSDP_SHARDING_STRATEGY_MAP.keys())}. Got: {sharding_strategy}."
            )

        # get fsdp wrapping policy
        fsdp_wrap_policy = model.get_fsdp_wrap_policy()

        super().__init__(
            model,
            sharding_strategy=FSDP_SHARDING_STRATEGY_MAP[sharding_strategy],
            auto_wrap_policy=fsdp_wrap_policy,
            mixed_precision=fsdp_precision_policy,
            backward_prefetch=FSDP_BACKWARD_PREFETCH[prefetching_option],
            **extra_args_fsdp,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add FSDP-specific arguments"""
        if cls == FullyShardedDataParallelWrapper:
            group = parser.add_argument_group(title=cls.__name__)
            group.add_argument(
                "--fsdp.sharding-strategy",
                type=str,
                default=None,
                choices=list(FSDP_SHARDING_STRATEGY_MAP.keys()),
                help="Sharding strategy for FSDP. Defaults to None.",
            )
            group.add_argument(
                "--fsdp.backward-prefetching",
                type=str,
                default="pre",
                choices=["pre", "post"],
                help="Backward prefetching. Supported modes are `pre` and `post`. \
                    `pre` and `post` prefetches the next set of parameters before and after \
                    the current set of parameter's gradient computation respectively. \
                    Defaults to `pre`.",
            )
            group.add_argument(
                "--fsdp.parameter-datatype",
                type=str,
                default="bfloat16",
                choices=list(FSDP_DATATYPE_CONVERSION.keys()),
                help="Specify the data type of model parameters.  See FSDP documentation for details. \
                    Defaults to `bfloat16`.",
            )
            group.add_argument(
                "--fsdp.gradient-reduction-datatype",
                type=str,
                default="bfloat16",
                choices=list(FSDP_DATATYPE_CONVERSION.keys()),
                help="Specify the data type for gradient reduction. See FSDP documentation for details. \
                    Defaults to `bfloat16`.",
            )
            group.add_argument(
                "--fsdp.buffer-datatype",
                type=str,
                default="bfloat16",
                choices=list(FSDP_DATATYPE_CONVERSION.keys()),
                help="Specify the data type for buffers. See FSDP documentation for details. \
                    Defaults to `bfloat16`.",
            )
            group.add_argument(
                "--fsdp.limit-all-gathers",
                action="store_true",
                help="Enabling this flag allows FSDP to explicitly synchronize the CPU threads and \
                    prevent too many in-flight all-gathers. Enabling this can \
                    help lower the number of CUDA malloc retries. Defaults to `False`. \
                    Note: In older PyTorch versions, this flag may not be available.",
            )
            group.add_argument(
                "--fsdp.cpu-offload",
                action="store_true",
                help="Enable CPU offloading. Defaults to `False`. \
                    Note: In older PyTorch versions, this flag may not be available.",
            )

        return parser


def get_fsdp_model_optimizer_state_dict_on_rank0(
    model: FullyShardedDataParallelWrapper, optimizer: BaseOptim
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Aggregates the model and optimizer states from all shards on rank0 and return it.

    Args:
        model: Model (partially) sharded by FSDP.
        optimizer: Optimizer.
    """
    with FullyShardedDataParallelWrapper.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        # config for model state aggregation
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        # config for optimizer state aggregation
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = model.state_dict()
        # Returns the state dict of optimzier for the ``model`` that is (partially) sharded by FSDP.
        optim_state = FullyShardedDataParallel.optim_state_dict(
            model=model, optim=optimizer
        )
        return model_state, optim_state
