#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import math
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlsplit

import torch

from corenet.constants import TMP_CACHE_LOC
from corenet.utils import logger, resources
from corenet.utils.ddp_utils import dist_barrier, is_start_rank_node


def get_local_path(
    opts: argparse.Namespace,
    path: str,
    cache_loc: str = TMP_CACHE_LOC,
    force_delete: Optional[bool] = None,
    use_start_rank: bool = True,
    sync_ranks: bool = True,
    max_retries: int = 5,
    parallel_download: bool = False,
    max_download_workers: int = 1,
    *args,
    **kwargs,
) -> str:
    """
    Helper function to download data from remote servers (e.g., S3/HTTP/HTTPS).

    Args:
        opts: Command-line arguments.
        path: Path of the remote file that needs to be downloaded.
        cache_loc: Location where file needs to be stored after downloading.
        force_delete: Force the file to be deleted if it is present in the `cache_loc`.
        use_start_rank: Download the files on the start rank of each node.
        sync_ranks: Synchronize DDP ranks after downloading.
        max_retries: Maximum download retries. Defaults to 5.
        parallel_download: If enabled, files are downloaded in parallel.
        max_download_workers: Maximum number of workers for downloading. Should satisfy 1 <= 'max_download_workers' <= num_cpus.

    Returns:
        If file(s) is(are) remote, this function downloads it to `cache_loc` and
        returns the local path(s). Otherwise, its a noop.
    """
    # To avoid circular imports, get_transfer_client is imported inside get_local_path.
    from corenet.data.io.transfer_clients import get_transfer_client

    client_name = urlsplit(path).scheme.lower()

    client = get_transfer_client(
        opts,
        transfer_client_name=client_name,
        max_retries=max_retries,
        force_delete=force_delete,
        only_download_on_start_rank=use_start_rank,
        synchronize_distributed_ranks=sync_ranks,
        parallel_download=parallel_download,
        max_download_workers=max_download_workers,
        raise_error_if_transfer_client_not_present=False,
    )

    if client is None:
        # For 'get_local_path', it is a no-op if file is locally present or client is not present
        # However, 'get_transfer_client()' can raise an error (if required) by enabling 'raise_error_if_transfer_client_not_present'.
        return path
    else:
        return client.download(path, cache_loc)


class DownloadFunc(Protocol):
    def __call__(
        file_idx: int, local_dst_dir: str, args: List[Any], kwargs: Dict[Any, Any]
    ) -> Any:
        """Protocol for a callable that downloads an index of a dataset to `dst_location`.
        Args:
            file_idx: Index of the asset to be downloaded.
            local_dst_dir: Local directory path where asset should be stored after downloading locally.
            args: 'download_func' specific args.
            kwargs: 'download_func' specific kwargs.
        """
        ...


def download_assets_in_parallel(
    opts: argparse.Namespace,
    download_func: DownloadFunc,
    local_dst_dir: str,
    num_assets: int,
    shard_asset: bool = False,
    max_threads: Optional[int] = None,
    index_padding: bool = False,
    *args,
    **kwargs,
) -> List[Any]:
    """
    Helper function to download assets in parallel.

    Args:
        opts: Command-line arguments.
        download_func: Custom function to be used for downloading/processing assets.
        dst_dir: Local directory path where assets should be stored after downloading. If 'dst_dir' does not exist, it will be created.
        num_assets: Total number of assets to be downloaded.
        shard_asset: Shard assets among nodes.
        max_threads: Maximum number of threads. If not specified, then max_threads=num_cpus.
        index_padding: Pad indices to download the same number of assets on each node.
        args and kwargs: 'download_func' specific args and kwargs.

    Returns:
        A list of indices.

    ...note:
        Users need to specify the 'num_assets'. This can be pre-computed (e.g., number of files in the S3 bucket) and passed
        to 'download_assets_in_parallel' function.

    """
    Path(local_dst_dir).mkdir(exist_ok=True, parents=True)

    cuda_available = torch.cuda.is_available()

    # For CPU, world_size and n_gpus_per_node should be 1 to avoid division-by-zero errors
    world_size = max(getattr(opts, "ddp.world_size"), 1)
    n_gpus_per_node = torch.cuda.device_count() if cuda_available else 1
    current_device = torch.cuda.current_device() if cuda_available else 0

    curr_node_rank = getattr(opts, "ddp.rank")
    node_id = curr_node_rank // n_gpus_per_node

    total_assets = int(math.ceil(num_assets / world_size)) * world_size
    asset_indices = [idx for idx in range(num_assets)]
    if index_padding:
        asset_indices += asset_indices[: (total_assets - num_assets)]

    if shard_asset:
        # Each node downloads portion of the assets.
        # Note that total number of GPUs (a.k.a world size) = Total number of nodes * number of GPUs per Node
        n_nodes = max(1, world_size // n_gpus_per_node)
        assets_per_node = int(math.ceil(len(asset_indices) / n_nodes))
    else:
        # Each node downloads all assets.
        assets_per_node = len(asset_indices)

    asset_indices_node_i = asset_indices[
        node_id * assets_per_node : (node_id + 1) * assets_per_node
    ]

    # now divide the assets among each GPU in the node to download in parallel
    asset_indices_node_i_rank_j = asset_indices_node_i[
        current_device : len(asset_indices_node_i) : n_gpus_per_node
    ]

    start_time = time.time()

    # download assets in parallel on each rank
    n_cpus = resources.cpu_count()
    if max_threads is None:
        # use all CPUs
        max_threads = n_cpus

    n_process_per_gpu = min(max(1, n_cpus // n_gpus_per_node), max_threads)
    with Pool(processes=n_process_per_gpu) as pool:
        pool.starmap(
            download_func,
            [
                (file_idx, local_dst_dir, args, kwargs)
                for file_idx in asset_indices_node_i_rank_j
            ],
        )

    # We set the default value of ddp.use_distributed here as it is not an exposed
    # parameter using command-line arguments
    if getattr(opts, "ddp.use_distributed", False):
        # synchronize between all DDP jobs
        dist_barrier()

    download_time = round((time.time() - start_time) / 60.0, 3)

    if is_start_rank_node(opts):
        # display information on start ranks
        logger.log(
            f"Function {download_func.__name__} took {download_time:.2f} minutes to process/download {len(asset_indices_node_i_rank_j)} assets."
        )

    return asset_indices_node_i
