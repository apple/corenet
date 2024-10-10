#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import datetime
import socket
from typing import Optional

import torch
import torch.distributed as dist

from corenet.utils import logger


def is_master(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    return node_rank == 0


def dist_barrier():
    dist.barrier()


def dist_monitored_barrier(
    timeout: Optional[float] = None,
    wait_all_ranks: Optional[bool] = False,
    group: Optional = None,
):
    dist.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)


def is_start_rank_node(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    def_rank = getattr(opts, "ddp.start_rank", 0)
    return node_rank == def_rank


def get_world_size():
    return dist.get_world_size()


def get_node_rank():
    return dist.get_rank()


def distributed_init(opts) -> int:
    ddp_url = getattr(opts, "ddp.dist_url", None)
    is_master_node = is_master(opts)
    if ddp_url is None:
        ddp_port = getattr(opts, "ddp.dist_port", 6006)
        hostname = socket.gethostname()
        ddp_url = "tcp://{}:{}".format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank")
    world_size = getattr(opts, "ddp.world_size")
    if world_size < 1:
        logger.error("World size should be > 0 for DDP.")
    if node_rank < 0:
        logger.error("Node rank should be >=0 for DDP.")

    if torch.distributed.is_initialized():
        logger.warning("DDP is already initialized and cannot be initialize twice!")
    else:
        logger.info("distributed init (rank {}): {}".format(node_rank, ddp_url))

        dist_backend = getattr(opts, "ddp.backend", "nccl")  # "gloo"

        if dist_backend is None and dist.is_nccl_available():
            dist_backend = "nccl"
            if is_master_node:
                logger.log(
                    "Using NCCL as distributed backend with version={}".format(
                        torch.cuda.nccl.version()
                    )
                )
        elif dist_backend is None:
            dist_backend = "gloo"

        dist.init_process_group(
            backend=dist_backend,
            init_method=ddp_url,
            timeout=datetime.timedelta(seconds=3600),
            world_size=world_size,
            rank=node_rank,
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank


def is_rank_0_worker_0(opts: argparse.Namespace) -> bool:
    """Check if the current process is worker 0 of rank 0.

    Args:
        opts: Command-line arguments.

    Returns:
        A boolean indicating whether the current process is worker 0 of rank 0
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and worker_info.id == 0 and is_master(opts):
        return True
    return False
