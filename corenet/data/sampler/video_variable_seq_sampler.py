#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import random
from typing import Iterator, Optional, Tuple

from corenet.data.sampler import SAMPLER_REGISTRY
from corenet.data.sampler.utils import make_video_pairs
from corenet.data.sampler.variable_batch_sampler import (
    VariableBatchSampler,
    VariableBatchSamplerDDP,
)
from corenet.utils import logger


@SAMPLER_REGISTRY.register(name="video_variable_seq_sampler")
class VideoVariableSeqSampler(VariableBatchSampler):
    """Extends `Variably-size multi-scale batch sampler <https://arxiv.org/abs/2110.02178?context=cs.LG>` for videos.
    This sampler yields batches of variable (1) batch size, (2) spatial resolutions,
    (3) frames per clip, and (4) number of clips per video.

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
        self.default_frames = getattr(opts, "sampler.vbs.num_frames_per_clip")
        min_clips_per_video = getattr(opts, "sampler.vbs.min_clips_per_video")
        self.max_clips_per_video = getattr(opts, "sampler.vbs.max_clips_per_video")
        self.clips_per_video = getattr(opts, "sampler.vbs.clips_per_video")
        if min_clips_per_video is None:
            logger.error(
                "Please specify min. clips per video using --sampler.vbs.min-clips-per-video."
            )

        self.min_clips_per_video = min_clips_per_video
        self.random_video_clips = False
        if is_training:
            # override img_batch_tuples
            self.img_batch_tuples = make_video_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                default_frames=self.default_frames,
            )
            self.random_video_clips = getattr(opts, "sampler.vbs.random_video_clips")
        else:
            if self.clips_per_video is None:
                logger.error(
                    "For modes other than training, clips per video can't be None"
                )
            self.img_batch_tuples = [
                (self.crop_size_h, self.crop_size_w, self.default_frames)
            ]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != VideoVariableSeqSampler:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--sampler.vbs.num-frames-per-clip",
            default=8,
            type=int,
            help="Default frames per video. Defaults to 8",
        )

        group.add_argument(
            "--sampler.vbs.random-video-clips",
            action="store_true",
            default=False,
            help="Sample number of clips per video randomly during training between min and max values specified using "
            "--sampler.vbs.min-clips-per-video and --sampler.vbs.max-clips-per-video arguments respectively",
        )
        group.add_argument(
            "--sampler.vbs.min-clips-per-video",
            type=int,
            default=1,
            help="Minimum number of clips per video. Used only for training. Defaults to 1.",
        )
        group.add_argument(
            "--sampler.vbs.max-clips-per-video",
            type=int,
            default=5,
            help="Maximum number of clips per video. Used only for training. Defaults to 5.",
        )
        group.add_argument(
            "--sampler.vbs.clips-per-video",
            type=int,
            default=1,
            help="Number of clips per video. Defaults to 1.",
        )
        group.add_argument(
            "--sampler.vbs.min-frames-per-clip",
            type=int,
            default=8,
            help="Minimum number of frames per clip. Defaults to 8.",
        )

        return parser

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int]]:
        indices = self.get_indices()

        start_index = 0
        indices_len = len(indices)
        while start_index < indices_len:
            if self.random_video_clips:
                # randomly sample number of clips and adjust frames per clip
                n_clips = max(
                    1,
                    random.randint(self.min_clips_per_video, self.max_clips_per_video),
                )
                batch_size = max(
                    self.batch_size_gpu0,
                    self.batch_size_gpu0 * (self.clips_per_video // n_clips),
                )
            else:
                n_clips = self.clips_per_video
                batch_size = self.batch_size_gpu0

            crop_h, crop_w, n_frames = random.choice(self.img_batch_tuples)
            end_index = min(start_index + batch_size, indices_len)
            batch_ids = indices[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if len(batch_ids) != batch_size:
                batch_ids += indices[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:

                batch = [
                    (crop_h, crop_w, b_id, n_frames, n_clips) for b_id in batch_ids
                ]
                yield batch

    def update_scales(
        self, epoch: int, is_master_node: bool = False, *args, **kwargs
    ) -> None:
        if type(self).update_scales is not VideoVariableSeqSampler.update_scales:
            # Do nothing when a subclass overrides this method and calls super().update_scales
            return

        if is_master_node and self.scale_inc:
            logger.warning(
                f"Update scale function is not yet implemented for {self.__class__.__name__}"
            )

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t var_num_clips_training=(min={self.min_clips_per_video}, max={self.max_clips_per_video})"
            f"\n\t fixed_num_clips_val={self.clips_per_video}"
        )
        return extra_repr_str


@SAMPLER_REGISTRY.register(name="video_variable_seq_sampler_ddp")
class VideoVariableSeqSamplerDDP(VariableBatchSamplerDDP):
    """DDP variant of VideoVariableSeqSampler

    Args:
        opts: command line argument
        n_data_samples (int): Number of samples in the dataset
        is_training (Optional[bool]): Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        n_data_samples: int,
        is_training: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        self.default_frames = getattr(opts, "sampler.vbs.num_frames_per_clip")

        self.random_video_clips = False
        self.min_clips_per_video = getattr(opts, "sampler.vbs.min_clips_per_video")
        self.max_clips_per_video = getattr(opts, "sampler.vbs.max_clips_per_video")
        self.clips_per_video = getattr(opts, "sampler.vbs.clips_per_video")
        if self.min_clips_per_video is None:
            logger.error(
                "Please specify min. clips per video using --sampler.vbs.min-clips-per-video."
            )

        if is_training:
            # override img_batch_tuples
            self.img_batch_tuples = make_video_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                default_frames=self.default_frames,
            )
            self.random_video_clips = getattr(opts, "sampler.vbs.random_video_clips")
        else:
            if self.clips_per_video is None:
                logger.error(
                    "For modes other than training, clips per video can't be None"
                )

            self.img_batch_tuples = [
                (self.crop_size_h, self.crop_size_w, self.default_frames)
            ]

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int]]:
        indices_rank_i = self.get_indices_rank_i()

        start_index = 0
        n_samples_rank_i = len(indices_rank_i)
        while start_index < n_samples_rank_i:
            if self.random_video_clips:
                # randomly sample number of clips and adjust batch size
                n_clips = max(
                    1,
                    random.randint(self.min_clips_per_video, self.max_clips_per_video),
                )
                batch_size = max(
                    self.batch_size_gpu0,
                    self.batch_size_gpu0 * (self.clips_per_video // n_clips),
                )
            else:
                n_clips = self.clips_per_video
                batch_size = self.batch_size_gpu0

            crop_h, crop_w, n_frames = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, n_samples_rank_i)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (crop_h, crop_w, b_id, n_frames, n_clips) for b_id in batch_ids
                ]
                yield batch

    def update_scales(
        self, epoch: int, is_master_node: bool = False, *args, **kwargs
    ) -> None:
        if type(self).update_scales is not VideoVariableSeqSamplerDDP.update_scales:
            # Do nothing when a subclass overrides this method and calls super().update_scales
            return
        if is_master_node and self.scale_inc:
            logger.warning(
                f"Update scale function is not yet implemented for {self.__class__.__name__}"
            )

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t var_num_clips_training=(min={self.min_clips_per_video}, max={self.max_clips_per_video})"
            f"\n\t fixed_num_clips_val={self.clips_per_video}"
        )
        return extra_repr_str
