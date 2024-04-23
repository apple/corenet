#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import argparse
import os
import random
import time
from argparse import ArgumentParser
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import boto3
import requests
from boto3.s3.transfer import S3Transfer, TransferConfig
from joblib import Parallel, delayed

from corenet.constants import if_test_env
from corenet.utils import logger, resources
from corenet.utils.common_utils import construct_local_path_from_remote
from corenet.utils.ddp_utils import dist_barrier, is_master, is_start_rank_node
from corenet.utils.registry import Registry


class BaseClient(object):
    """Base class for transfer clients.

    Args:
        opts: Command-line arguments.
        max_retries: Maximum download retries. Defaults to 5.
        force_delete: Force the file to be deleted if it is present locally. The local path is a concatenation of 'dst_location' and basename of 'remote_file'
        only_download_on_start_rank: Download the files on the start rank of each node.
        synchronize_distributed_ranks: Synchronize DDP ranks after downloading.
        parallel_download: If enabled, files are downloaded in parallel in 'download_multiple_files' function.
        max_download_workers: Maximum number of workers for downloading. Should satisfy 1 <= 'max_download_workers' <= num_cpus.

    ...note:
        During regular training, setting 'force_delete' to 'None' is equivalent to specifying 'force_delete=True.' However, when running tests with pytest, 'None' is
        interpreted as 'False' to optimize test execution speed.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        max_retries: int = 5,
        force_delete: Optional[bool] = None,
        only_download_on_start_rank: bool = True,
        synchronize_distributed_ranks: bool = True,
        parallel_download: bool = False,
        max_download_workers: int = 1,
        *args,
        **kwargs,
    ) -> None:
        n_cpus = resources.cpu_count()
        if max_download_workers < 1 or max_download_workers > n_cpus:
            raise RuntimeError(
                f"Maximum number of download workers should be between 1 and number of available CPUs. Got: {max_download_workers}."
            )
        self.opts = opts
        self.max_retries = max_retries
        self.force_delete = force_delete
        self.download_on_start_rank = only_download_on_start_rank
        self.synchronize_distributed_ranks = synchronize_distributed_ranks
        self.parallel_download = parallel_download
        self.is_master_node = is_master(opts)
        self.n_download_workers = max_download_workers

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add client-specific arguments."""
        if cls != BaseClient:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        return parser

    @cached_property
    def start_rank_node(self) -> bool:
        """Check whether this process runs on the starting rank of its node."""
        opts = self.opts
        return is_start_rank_node(opts)

    @cached_property
    def sync_distributed_ranks(self) -> bool:
        """Synchronize the ranks during distributed training."""
        opts = self.opts
        sync_ranks = self.synchronize_distributed_ranks
        # 'ddp.use_distributed' is a private argument, therefore, default value is set.
        return getattr(opts, "ddp.use_distributed", False) and sync_ranks

    def _download_fn(self, remote_path: str, local_path: str) -> None:
        """Logic for downloading 'remote_path' to 'local_path'. Throws exception if it fails.

        Args:
            remote_path: Remote file path.
            local_path: Local path where file should be downloaded.
        """
        raise NotImplementedError(
            f"Child classes of {self.__class__.__name__} must implement downloading logic."
        )

    def _download_with_retries(self, remote_path: str, local_path: str) -> None:
        """Download 'remote_path' to 'local_path' with retries."""
        for num_tries in range(self.max_retries):
            try:
                return self._download_fn(remote_path, local_path)
            except Exception as e:
                wait_duration = 2**num_tries * random.uniform(0.5, 1.0)
                logger.warning(
                    f"Failed download attempt {num_tries}, Retrying in {wait_duration} seconds for {remote_path}."
                )
                time.sleep(wait_duration)
        try:
            return self._download_fn(remote_path, local_path)
        except Exception as e:
            raise RuntimeError(
                f"Unable to download {remote_path} after {self.max_retries} retries with error {str(e)} using {type(self)}."
            ) from e

    def _download_single_file(self, remote_path: str, dst_dir: str) -> str:
        """Download single remote file locally.

        Args:
            remote_path: Path of the remote file.
            dst_dir: Local storage directory.

        Returns:
            Local file path.
        """
        force_delete = self.force_delete
        if force_delete is None:
            # An alternative approach is to move this logic to the argument's default value in the function signature:
            #     def get_local_path(..., force_delete = if_test_env(False, otherwise=True), ...):
            # But that won't always work because pytest may set PYTEST_CURRENT_TEST env set loading this module.
            force_delete = if_test_env(False, otherwise=True)

        local_path = construct_local_path_from_remote(
            remote_path=remote_path, local_dir=dst_dir
        )

        if os.path.isfile(local_path) and force_delete:
            # If file exists, remove it and then download again
            if self.download_on_start_rank:
                # Remove the file from start rank and let other ranks in DDP keep waiting till file is deleted
                if is_start_rank_node(self.opts):
                    os.remove(local_path)
                else:
                    while not os.path.isfile(local_path):
                        time.sleep(if_test_env(0, otherwise=1))
                        continue
            else:
                # All ranks in DDP deletes the file
                os.remove(local_path)

        if not os.path.isfile(local_path):
            if self.download_on_start_rank:
                # Download the file using start rank and let other ranks in DDP keep waiting till file is downloaded
                if self.start_rank_node:
                    self._download_with_retries(remote_path, local_path)
                else:
                    while not os.path.isfile(local_path):
                        time.sleep(if_test_env(0, otherwise=1))
                        continue
            else:
                # All ranks in DDP downloads the file
                self._download_with_retries(remote_path, local_path)

        if self.sync_distributed_ranks:
            # syncronize between ranks
            dist_barrier()
        return local_path

    def _download_multiple_files(
        self, remote_file_paths: List[str], dst_dir: str
    ) -> List[str]:
        """Download multiple remote files locally either sequentially or simultaneously.

        Args:
            remote_file_paths: List of remote file paths.
            dst_dir: Local storage directory.

        Returns:
            List of local file paths.
        """
        total_files = len(remote_file_paths)
        if total_files < 1:
            raise RuntimeError("Need at least one file for downloading.")

        if self.parallel_download and total_files > 1:
            n_workers = min(self.n_download_workers, total_files)
            local_paths = Parallel(n_jobs=n_workers)(
                delayed(self._download_single_file)(remote_file_path, dst_dir)
                for remote_file_path in remote_file_paths
            )
        else:
            local_paths = [
                self._download_single_file(
                    remote_path=remote_file_path, dst_dir=dst_dir
                )
                for remote_file_path in remote_file_paths
            ]
        return local_paths

    def download(
        self, remote_file_paths: Union[str, List[str]], dst_dir: str
    ) -> Union[str, List[str]]:
        """Download remote files locally.

        Args:
            remote_file_paths: Single (or list) remote file path(s).
            dst_dir: Local storage directory.

        Returns:
            Single (or list) local file path(s).
        """
        Path(dst_dir).mkdir(exist_ok=True, parents=True)

        if isinstance(remote_file_paths, List):
            return self._download_multiple_files(remote_file_paths, dst_dir)
        elif isinstance(remote_file_paths, str):
            return self._download_single_file(remote_file_paths, dst_dir)
        else:
            raise NotImplementedError(
                f"Supported file paths for downloading are string or List[str]. Got: {type(remote_file_paths)}."
            )


TRANSFER_CLIENT_REGISTRY = Registry(
    registry_name="transfer_client_registry",
    base_class=BaseClient,
    lazy_load_dirs=["corenet/data/io"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


@TRANSFER_CLIENT_REGISTRY.register(name="s3")
class S3Client(BaseClient):
    """Client to download files from S3."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        endpoint_url = getattr(opts, "common.s3.endpoint_url")
        aws_access_key_id = getattr(opts, "common.s3.aws_access_key_id")
        aws_secret_access_key = getattr(opts, "common.s3.aws_secret_access_key")
        aws_session_token = getattr(opts, "common.s3.aws_session_token")
        super().__init__(opts, *args, **kwargs)
        self.boto_transfer_client = self.init_client(
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

    def transfer_config(self) -> TransferConfig:
        opts = self.opts
        multi_part_threshold = getattr(opts, "common.s3.multipart_threshold")
        transfer_config = TransferConfig(
            multipart_threshold=multi_part_threshold,
            num_download_attempts=self.max_retries,
        )
        return transfer_config

    def init_client(
        self,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ) -> S3Transfer:
        """Initialize S3Transfer client.

        Args:
            endpoint_url: Endpoint URL. If none, botocore will automatically construct the
                appropriate URL to use when communicating with a service.
            aws_access_key_id: The access key to use when creating the client. If None, the
                credentials configured for the session will automatically be used.
            aws_secret_access_key: The secret key to use when creating the client. Same semantics
                as 'aws_access_key_id'.
            aws_session_token: The session token to use when creating the client. Same semantics
                as 'aws_access_key_id'.
        Returns:
            An instance of 'S3Transfer'.
        """

        transfer_config = self.transfer_config()
        boto_client = boto3.client(
            service_name="s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        boto_transfer_client = S3Transfer(boto_client, transfer_config)
        return boto_transfer_client

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        """S3 client specific arguments.

        ...note:
            More details can be found in boto3, an AWS SDK for Python.
        """
        if cls != S3Client:
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--common.s3.endpoint-url",
            type=str,
            default=None,
            help="Endpoint URL for S3 client. Defaults to None.",
        )
        group.add_argument(
            "--common.s3.aws-access-key-id",
            type=str,
            default=None,
            help="AWS Access key id. Defaults to None.",
        )
        group.add_argument(
            "--common.s3.aws-secret-access-key",
            type=str,
            default=None,
            help="AWS secret access key. Defaults to None.",
        )
        group.add_argument(
            "--common.s3.aws-session-token",
            type=str,
            default=None,
            help="AWS session token. Defaults to None.",
        )
        group.add_argument(
            "--common.s3.multipart-threshold",
            type=str,
            default=32 * 1024 * 1024,
            help="The partition size of each part for a multipart transfer. Defaults to 32 MB.",
        )

        return parser

    def _download_fn(self, remote_path: str, local_path: str) -> None:
        parsed_url = urlparse(remote_path)
        self.boto_transfer_client.download_file(
            bucket=parsed_url.netloc,
            key=parsed_url.path.lstrip("/"),
            filename=local_path,
        )


@TRANSFER_CLIENT_REGISTRY.register(name="http")
class HTTPClient(BaseClient):
    """Client to download files from HTTP."""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

    @cached_property
    def proxies(self):
        return {
            "https": os.environ.get("HTTPS_PROXY", None),
            "http": os.environ.get("HTTP_PROXY", None),
        }

    def _download_fn(self, remote_path: str, local_path: str) -> None:
        response = requests.get(remote_path, stream=True)
        if response.status_code == 403:
            # Try with the HTTP/HTTPS proxy from ENV
            response = requests.get(remote_path, stream=True, proxies=self.proxies)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.raw.read())
        else:
            raise IOError(f"Download failed with status code {response.status_code}.")


@TRANSFER_CLIENT_REGISTRY.register(name="https")
class HTTPSClient(HTTPClient):
    """Client to download files from HTTPS."""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)


def get_transfer_client(
    opts: argparse.Namespace,
    transfer_client_name: str,
    max_retries: int = 5,
    force_delete: Optional[bool] = None,
    only_download_on_start_rank: bool = True,
    synchronize_distributed_ranks: bool = True,
    parallel_download: bool = False,
    max_download_workers: int = 1,
    raise_error_if_transfer_client_not_present: bool = True,
) -> Optional[BaseClient]:
    """
    Helper function to get transfer client.

    Args:
        opts: Command-line arguments.
        transfer_client_name: Name of the transfer client.
        max_retries: Maximum download retries. Defaults to 5.
        force_delete: Force the file to be deleted if it is present in the `cache_loc`.
        only_download_on_start_rank: Download the files on the start rank of each node.
        synchronize_distributed_ranks: Syncronize DDP ranks after downloading.
        parallel_download: If enabled, files are downloaded in parallel.
        max_download_workers: Maximum number of workers for downloading. Should satisfy 1 <= 'max_download_workers' <= num_cpus.
        raise_error_if_transfer_client_not_present: Raise an error if client is not present.

    Returns:
        The transfer client requested will be returned if it's available. Otherwise, an error will be raised if the 'raise_error_if_transfer_client_not_present' flag is set to True, or None will be returned.

    ...note:
        In certain places, we may want to control the error handling different (e.g., 'get_local_path' is a no-op when path is a local). To enable such case,
        use 'raise_error_if_transfer_client_not_present' flag.
    """

    if transfer_client_name in TRANSFER_CLIENT_REGISTRY.keys():
        client_cls = TRANSFER_CLIENT_REGISTRY[transfer_client_name]
        client = client_cls(
            opts,
            max_retries=max_retries,
            force_delete=force_delete,
            only_download_on_start_rank=only_download_on_start_rank,
            synchronize_distributed_ranks=synchronize_distributed_ranks,
            parallel_download=parallel_download,
            max_download_workers=max_download_workers,
        )
        return client
    elif raise_error_if_transfer_client_not_present:
        raise RuntimeError(
            f"Requested transfer client, i.e. {transfer_client_name}, is not available. \
                Availble clients are {list(TRANSFER_CLIENT_REGISTRY.keys())}"
        )
    else:
        return None


def transfer_client_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Transfer client arguments."""
    return TRANSFER_CLIENT_REGISTRY.all_arguments(parser)
