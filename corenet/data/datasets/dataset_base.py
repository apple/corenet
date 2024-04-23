#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import argparse
from abc import ABC
from functools import cached_property
from typing import Any, Dict, Optional, TypedDict

import torch
from PIL import Image
from torch.utils import data

from corenet.data.transforms import BaseTransformation
from corenet.data.video_reader import get_video_reader
from corenet.utils import logger
from corenet.utils.ddp_utils import (
    get_node_rank,
    get_world_size,
    is_master,
    is_start_rank_node,
)


class BaseDataset(data.Dataset, ABC):
    """Base class for creating datasets. Sub-classes must implement __getitem__,
    _training_transforms, and _validation_transforms functions.

    Args:
        opts: Command-line arguments
        is_training: Training mode or not. Defaults to True.
        is_evaluation: Evaluation mode or not. Defaults to False.

    ...note::
        `is_training` is used to indicate whether the dataset is used for training or
        validation. On the other hand, `is_evaluation` mode is used to indicate the
        dataset is used for testing.

        Theoretically, `is_training=False` and `is_evaluation=True` should be the same.
        However, for some datasets (especially segmentation), validation dataset
        transforms are different from test transforms because each image has different
        resolution, making it difficult to construct batches. Therefore, we treat these
        two modes different.

        For datasets, where validation and testing transforms are the same, we set
        evaluation transforms the same as the validation transforms.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = True,
        is_evaluation: bool = False,
        *args,
        **kwargs,
    ) -> None:
        # Do not remove the default value here.
        if getattr(opts, "dataset.trove.enable", False):
            opts = self.load_from_server(
                opts=opts, is_training=is_training, is_evaluation=is_evaluation
            )

        assert (
            not is_training or not is_evaluation
        ), "is_training and is_evaluation cannot be both True"
        if is_training:
            self.mode = "train"
        elif is_evaluation:
            self.mode = "test"
        else:
            self.mode = "val"

        self.root = getattr(opts, f"dataset.root_{self.mode}")
        if self.mode == "test" and not getattr(opts, f"dataset.root_test"):
            # Only use root_test when applicable. Most datasets only define root_val.
            self.root = getattr(opts, f"dataset.root_val")

        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.opts = opts

        self.device = getattr(self.opts, "dev.device", torch.device("cpu"))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments"""
        if cls != BaseDataset:
            # Don't re-register arguments in subclasses that don't override
            # `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--dataset.root-train",
            type=str,
            default="",
            help="Root location of train dataset",
        )
        group.add_argument(
            "--dataset.root-val",
            type=str,
            default="",
            help="Root location of valid dataset",
        )
        group.add_argument(
            "--dataset.root-test",
            type=str,
            default="",
            help="Root location of test dataset",
        )
        group.add_argument(
            "--dataset.disable-val",
            action="store_true",
            default=False,
            help="Disable validation during training",
        )

        group.add_argument(
            "--dataset.name",
            type=str,
            default=None,
            help="Dataset name (e.g., imagenet). Defaults to None.",
        )
        group.add_argument(
            "--dataset.category",
            type=str,
            default=None,
            help="Dataset category (e.g., segmentation, classification). Defaults to None.",
        )
        group.add_argument(
            "--dataset.percentage-of-samples",
            type=float,
            default=100.0,
            help="Percentage of samples to use from the dataset.",
        )
        group.add_argument(
            "--dataset.sample-selection-random-seed",
            type=int,
            default=0,
            help="Random seed for selecting a subset of samples to use from the dataset.",
        )
        group.add_argument(
            "--dataset.train-batch-size0",
            default=128,
            type=int,
            help="Training batch size on GPU-0. Defaults to 128. "
            "Note that we scale it depending on total GPUs available for training. For"
            " example, if 2 GPUs are available and value of `dataset.train_batch_size0`"
            " is 128, then effective batch size will be 256.",
        )
        group.add_argument(
            "--dataset.val-batch-size0",
            default=1,
            type=int,
            help="Batch size on GPU-0 for validation. Defaults to 1. "
            "Note that we scale it depending on total GPUs available for training. For"
            " example, if 2 GPUs are available and value of `dataset.val_batch_size0`"
            " is 128, then effective batch size will be 256.",
        )
        group.add_argument(
            "--dataset.eval-batch-size0",
            default=1,
            type=int,
            help="Batch size on GPU-0 for testing or evaluation. Defaults to 1."
            "Note that we scale it automatically depending on total number of GPUs"
            " available. We recommend to run evaluation on a single GPU machine.",
        )
        group.add_argument(
            "--dataset.workers",
            default=-1,
            type=int,
            help="Number of data workers. Defaults to -1."
            "When number of workers are specified as -1, then total number of workers"
            " is equal to the number of available CPUs.",
        )

        group.add_argument(
            "--dataset.persistent-workers",
            action="store_true",
            default=False,
            help="Enabling this argument allows us to use same workers for loading data"
            " throughout the training. Defaults to False.",
        )
        group.add_argument(
            "--dataset.pin-memory",
            action="store_true",
            default=False,
            help="Enabling this allows us to use pin memory option in data loader. "
            "Defaults to False.",
        )
        group.add_argument(
            "--dataset.prefetch-factor",
            type=int,
            default=2,
            help="Number of samples loaded in advance by each data worker. Defaults to 2.",
        )

        group.add_argument(
            "--dataset.padding-index",
            type=int,
            default=None,
            help="Padding index for text vocabulary. Defaults to None.",
        )

        group.add_argument(
            "--dataset.text-vocab-size",
            type=int,
            default=-1,
            help="Text vocabulary size. Defaults to -1.",
        )
        group.add_argument(
            "--dataset.text-context-length",
            type=int,
            default=None,
            help="Context length for text encoder. Defaults to None.",
        )

        return parser

    @staticmethod
    def load_from_server(
        opts: argparse.Namespace, is_training: bool, is_evaluation: bool
    ) -> Optional[argparse.Namespace]:
        """Helper function to load dataset from server."""
        try:
            from corenet.internal.utils.server_utils import load_from_data_server

            opts = load_from_data_server(
                opts=opts, is_training=is_training, is_evaluation=is_evaluation
            )
            return opts

        except ImportError as e:
            import traceback

            traceback.print_exc()
            logger.error(
                "Unable to load data. Please load data manually. Error: {}".format(e)
            )

    def _training_transforms(self, *args, **kwargs) -> BaseTransformation:
        """Data transforms for training"""
        raise NotImplementedError

    def _validation_transforms(self, *args, **kwargs) -> BaseTransformation:
        """Data transforms for validation"""
        raise NotImplementedError

    def _evaluation_transforms(self, *args, **kwargs) -> BaseTransformation:
        """Data transforms for evaluation/testing"""
        return self._validation_transforms(*args, **kwargs)

    def get_augmentation_transforms(self, *args, **kwargs) -> BaseTransformation:
        """Helper function to get data transforms depending on the
        mode (training, evaluation, or validation)"""
        if self.is_training:
            transform = self._training_transforms(*args, **kwargs)
        elif self.is_evaluation:
            transform = self._evaluation_transforms(*args, **kwargs)
        else:
            transform = self._validation_transforms(*args, **kwargs)
        return transform

    def share_dataset_arguments(self) -> Dict[str, Any]:
        """Function that can be used by sub-classes to share dataset-specific options.
        It returns a mapping. An example is {"model.classification.n_classes", 1000}

        By default, we return an empty dictionary
        """
        return {}

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, sample_size_and_index: Any) -> Any:
        """Returns the sample corresponding to the input sample index."""
        raise NotImplementedError

    def extra_repr(self) -> str:
        r"""Extra information to be represented in __repr__. Each line in the output
        string should be prefixed with ``\t``.
        """

        return f"\n\tnum_samples={len(self)}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"\n\troot={self.root} "
            f"\n\tis_training={self.is_training} "
            f"{self.extra_repr()}"
            f"\n)"
        )

    def get_item_metadata(self, item_idx: int) -> Dict:
        """Returns the metadata for given @item_idx. This method could be used by
        samplers for sampling dynamic batches based on the metadata of the items.

        Args:
            item_idx: The index of sample to provide metadata for. The indexing
                should be aligned with how ``self.__getitem__(item_idx)`` sequences the
                dataset items.

        Returns: A dict containing the metadata. Each sampler may require a specific
            schema to be returned by this function.
        """
        raise NotImplementedError()

    @property
    def worker_id(self) -> int:
        """Returns the current worker id when loading data with multiple workers in dataloader.

        ...note:
            When 'get_worker_info' is None, '0' is returned to indicate it's a single process.

        ...warning:
            This function should be called after dataset is wrapped inside dataloader.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return worker_info.id
        return 0

    @property
    def num_workers(self) -> int:
        """Returns the number of workers used to load data with multi-processing in dataloader.

        ...note:
            When 'get_worker_info' is None, '1' is returned to indicate it's a single process.

        ...warning:
            This function should be called after dataset is wrapped inside dataloader.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return worker_info.num_workers
        return 1

    @property
    def is_master_node(self) -> bool:
        """Check if the current node is the master node in the distributed system.

        Returns:
            True if the current node is the master node, False otherwise.

        ...warning:
            This function should be used with distributed training.
        """
        return is_master(self.opts)

    @property
    def is_start_rank_node(self) -> bool:
        """Check if the GPU in a node is the first GPU or not during distributed training.

        Returns:
            True if the current GPU is the first GPU on each node. False otherwise.

        ...warning:
            This function should be used with distributed training.
        """
        return is_start_rank_node(self.opts)

    @property
    def world_size(self) -> int:
        """Returns the number of processes in the current process group in distributed training.

        ...warning:
            This function should be used with distributed training.
        """
        return max(1, getattr(self.opts, "ddp.world_size"))

    @property
    def rank(self) -> int:
        """Returns the rank of the current process in distributed training.

        ...note:
            Rank is a unique identifier assigned to each process within a distributed
            process group, and are always consecutive integers ranging from 0 to 'world_size'.

        ...warning:
            This function should be used with distributed training.
        """
        return max(0, getattr(self.opts, "ddp.rank"))


class BaseImageDataset(BaseDataset, ABC):
    """Base Dataset class for Image datasets."""

    @staticmethod
    def read_image_pil(path: str) -> Optional[Image.Image]:
        """Reads a PIL image.

        Args:
            path: Path of image file.

        Returns:
            If there are no exceptions (e.g., because of corrupted images), PIL Image is
            returned. Otherwise, None.
        """
        try:
            return Image.open(path).convert("RGB")
        except:
            # for any runtime exception while reading an image (typically arises from
            # corrupted images), we return None.
            return None

    def extra_repr(self) -> str:
        r"""Extra information to be represented in __repr__. Each line in the output
        string should be prefixed with ``\t``.
        """
        from corenet.utils.tensor_utils import image_size_from_opts

        return (
            super().extra_repr()
            + f"\n\ttransforms={self.get_augmentation_transforms(size=image_size_from_opts(self.opts))}"
        )


class BaseVideoDataset(BaseDataset, ABC):
    """Base Dataset class for video datasets.

    Args:
        opts: Command-line arguments
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts, *args, **kwargs)

        self.video_reader = get_video_reader(opts=opts, is_training=self.is_training)

        self._deprecated_clips_per_video = getattr(opts, "dataset.clips_per_video")
        self._deprecated_n_frames_per_clip = getattr(opts, "dataset.n_frames_per_clip")

    @property
    def clips_per_video(self) -> int:
        logger.warning(
            DeprecationWarning(
                "The --dataset.clips-per-video argument is deprecated. Please use"
                " VideoClipBatchSampler and its corresponding arguments."
            )
        )
        return self._deprecated_clips_per_video

    @clips_per_video.setter
    def _deprecated_set_clips_per_video(self, value: int) -> None:
        self._deprecated_clips_per_video = value

    @property
    def n_frames_per_clip(self) -> int:
        logger.warning(
            DeprecationWarning(
                "The --dataset.n-frames-per-clip argument is deprecated. Please use"
                " VideoClipBatchSampler and its corresponding arguments."
            )
        )
        return self._deprecated_n_frames_per_clip

    @n_frames_per_clip.setter
    def _deprecated_set_n_frames_per_clip(self, value: int) -> None:
        self._deprecated_n_frames_per_clip = value

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseVideoDataset:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--dataset.clips-per-video",
            type=int,
            default=1,
            help="The number of clips that each video file gets split into. Default"
            " value is 1, i.e., we don't split videos into multiple clips.",
        )
        group.add_argument(
            "--dataset.n-frames-per-clip",
            type=int,
            default=64,
            help="The number of frames to read from the video file into each clip."
            " Defaults to 64.",
        )

        return parser

    def get_item_metadata(self, item_idx: int) -> VideoMetadataDict:
        """Returns the metadata for given @item_idx. This method is used by
        VideoClipSampler for sampling dynamic clips based on the duration of the items.

        Subclasses should override this method if they use VideoClipSampler.

        Args:
            item_idx: The index of video file to provide metadata. The indexing
                should be aligned with how ``self.__getitem__(item_idx)`` sequences the
                dataset items.

        Returns: A dict containing the metadata. Please see @VideoMetadataDict
        documentation.
        """
        raise NotImplementedError()

    def get_item_local_path(self, item_idx: int) -> str:
        """Returns the local video path for given @item_idx. Implementing this method is
        optional, but having a unified interface reduces the cost of adding features.

        Args:
            item_idx: The index of video file to provide metadata. The indexing
                should be aligned with how ``self.__getitem__(item_idx)`` sequences the
                dataset items.

        Returns:
            str: The local path (downloaded if required) to the video file.
        """
        raise NotImplementedError()


# The ``total=False`` annotation marks all dict entries as NotRequired.
class VideoMetadataDict(TypedDict, total=False):
    """
    This class is an alias of Dict, in addition to optional standard key names and
    value types. The fields will be required or optional depending on the Sampler.
    """

    video_fps: float  # (Required for VideoClipBatchSampler)
    total_video_frames: int  # (Required for VideoClipBatchSampler)
    video_duration: (
        float  # video_fps * total_video_frames  (Required for VideoClipBatchSampler)
    )
    audio_fps: float


class BaseIterableDataset(BaseDataset, data.IterableDataset):
    """Base class for Iterable datasets."""

    pass
