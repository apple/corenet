#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Iterator, Tuple

from corenet.data.sampler import SAMPLER_REGISTRY
from corenet.data.sampler.batch_sampler import BatchSampler, BatchSamplerDDP


@SAMPLER_REGISTRY.register(name="video_batch_sampler")
class VideoBatchSampler(BatchSampler):
    """Standard Batch Sampler for videos. This sampler yields batches of fixed (1) batch size,
    (2) spatial resolutions, (3) frames per clip, and (4) number of clips per video.

    Args:
        opts: command line argument
        n_data_samples: Number of samples in the dataset
        is_training: Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        self.default_frames = getattr(opts, "sampler.bs.num_frames_per_clip")

        self.clips_per_video = getattr(opts, "sampler.bs.clips_per_video")

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int]]:
        indices = self.get_indices()

        start_index = 0
        batch_size = self.batch_size_gpu0
        indices_len = len(indices)
        while start_index < indices_len:

            end_index = min(start_index + batch_size, indices_len)
            batch_ids = indices[start_index:end_index]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (
                        self.crop_size_h,
                        self.crop_size_w,
                        b_id,
                        self.default_frames,
                        self.clips_per_video,
                    )
                    for b_id in batch_ids
                ]
                yield batch

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.Namespace:
        if cls != VideoBatchSampler:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--sampler.bs.num-frames-per-clip",
            default=8,
            type=int,
            help="Number of frames per video clip. Defaults to 8.",
        )
        group.add_argument(
            "--sampler.bs.clips-per-video",
            default=1,
            type=int,
            help="Number of clips per video. Defaults to 1.",
        )
        return parser

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t n_clips={self.clips_per_video}"
            f"\n\t n_frames_per_clip={self.default_frames}"
        )
        return extra_repr_str


@SAMPLER_REGISTRY.register(name="video_batch_sampler_ddp")
class VideoBatchSamplerDDP(BatchSamplerDDP):
    """DDP version of VideoBatchSampler

    Args:
        opts: command line argument
        n_data_samples: Number of samples in the dataset
        is_training: Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        self.default_frames = getattr(opts, "sampler.bs.num_frames_per_clip")
        self.clips_per_video = getattr(opts, "sampler.bs.clips_per_video")

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int]]:
        indices_rank_i = self.get_indices_rank_i()

        start_index = 0
        batch_size = self.batch_size_gpu0
        indices_len = len(indices_rank_i)
        while start_index < indices_len:
            end_index = min(start_index + batch_size, indices_len)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (
                        self.crop_size_h,
                        self.crop_size_w,
                        b_id,
                        self.default_frames,
                        self.clips_per_video,
                    )
                    for b_id in batch_ids
                ]
                yield batch

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t n_clips={self.clips_per_video}"
            f"\n\t n_frames_per_clip={self.default_frames}"
        )
        return extra_repr_str
