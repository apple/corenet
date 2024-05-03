#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from corenet.data.sampler import SAMPLER_REGISTRY
from corenet.data.sampler.variable_batch_sampler import (
    VariableBatchSampler,
    VariableBatchSamplerDDP,
)
from corenet.utils import logger


@SAMPLER_REGISTRY.register(name="video_clip_batch_sampler")
class VideoClipBatchSampler(VariableBatchSampler):
    """
    Batch sampler for videos. Different with VideoBatchSampler that samples each video
    the same number of clips despite the length of the video, VideoClipBatchSampler
    samples number of clips proportional to the length of the video. In addition, by
    setting scales of image (--sampler.vbs.min-crop-size-width,
    --sampler.vbs.mmax-crop-size-width, --sampler.vbs.min-crop-size-height,
    --sampler.vbs.mmax-crop-size-height, --sampler.vbs.max-n-scales) and frame rate
    (--sampler.vcbs.min-clip-fps-scale, --sampler.vcbs.max-clip-fps-scale and
    --sampler.vcbs.video-fps-num-scales), this sampler can also sample videos with
    variable image size and frame rate.

    When VideoClipBatchSampler is used, a tuple (crop_h, crop_w, video_id, num_frames,
    num_clips, video_fps, audio_fps, num_samples_per_clip) is returned. (`crop_h`,
    `crop_w`) is the image size to use, `video_id` is the index of the video the current
    clips come from, `num_clips` is the number of clip to sample from the current video,
    which is proportional to the length of the video; `video_fps` and `audio_fps` is the
    frame rate of video and audio to sample; `num_samples_per_clip` is the number of
    samples to generate for each clip at training time, this variable is only valid at
    training time.

    Note that the variable image size and frame rate are only applied during the
    training time. The batch size is adjusted accordingly with the image size and video
    length. By setting `--sampler.vcbs.max-num-clips-per-batch`, we have a upper bound
    of a batch in case a certain video is too long and cause OOM problem.

    Args:
        opts: Command line argument.
        n_data_samples: Number of samples in the dataset.
        is_training: Training or validation mode. Default: False.
        get_item_metadata: A callable that provides sample metadata, given sample index.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="Batch sampler for videos",
            description="Arguments related to variable batch sampler",
        )
        group.add_argument(
            "--sampler.vcbs.num-frames-per-clip",
            default=8,
            type=int,
            help="Number of frames per video clip. Default to 8.",
        )
        group.add_argument(
            "--sampler.vcbs.video-fps",
            type=int,
            default=8,
            help="The desired frame rate of the clip. Default to 8.",
        )
        group.add_argument(
            "--sampler.vcbs.audio-fps",
            type=int,
            default=16000,
            help="The frame rate of audio. Default to 16000.",
        )
        group.add_argument(
            "--sampler.vcbs.min-clip-fps-scale",
            type=float,
            default=1,
            help="The minimal scale to apply to the desired video/audio frame rate of"
            " the clip. Default to 1.",
        )
        group.add_argument(
            "--sampler.vcbs.max-clip-fps-scale",
            type=float,
            default=2.5,
            help="The maximal scale to apply to desired video/audio frame rate of the"
            " clip. Default to 2.5.",
        )
        group.add_argument(
            "--sampler.vcbs.video-fps-num-scales",
            type=float,
            default=5,
            help="The maximal scale to apply to desired frame rate of the clip. Default"
            " to 5.",
        )
        group.add_argument(
            "--sampler.vcbs.num-clips-per-second-train",
            type=int,
            default=1,
            help="The number of clips per second for training, default to 1. This is"
            " used to determine the frequency to sample.",
        )
        group.add_argument(
            "--sampler.vcbs.num-clips-per-second-val",
            type=int,
            default=4,
            help="The number of clips per second for validation, default to 4. This is"
            "used to determine the frequency to sample.",
        )
        group.add_argument(
            "--sampler.vcbs.max-num-clips-per-batch",
            type=int,
            default=50,
            help="The maximal number of clips per batch, default to 50. This is used to"
            " avoid memory leak if videos are too long.",
        )
        group.add_argument(
            "--sampler.vcbs.num-samples-per-clip",
            type=int,
            default=1,
            help="The number of samples to generate for each clip at training time."
            " Default to 1.",
        )
        return parser

    def __init__(
        self,
        opts,
        n_data_samples: int,
        is_training: Optional[bool] = False,
        get_item_metadata: Optional[Callable[[int], Dict]] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        self.default_frames = getattr(opts, "sampler.vcbs.num_frames_per_clip")
        self.video_fps = getattr(opts, "sampler.vcbs.video_fps")
        self.audio_fps = getattr(opts, "sampler.vcbs.audio_fps")
        self.min_fps_scale = getattr(opts, "sampler.vcbs.min_clip_fps_scale")
        self.max_fps_scale = getattr(opts, "sampler.vcbs.max_clip_fps_scale")
        self.num_fps_scale = getattr(opts, "sampler.vcbs.video_fps_num_scales")
        self.num_samples_per_clip = getattr(opts, "sampler.vcbs.num_samples_per_clip")
        self.num_clips_per_second_train = getattr(
            opts, "sampler.vcbs.num_clips_per_second_train"
        )
        self.num_clips_per_second_val = getattr(
            opts, "sampler.vcbs.num_clips_per_second_val"
        )
        self.max_num_clips_per_batch = getattr(
            opts, "sampler.vcbs.max_num_clips_per_batch"
        )
        self.is_training = is_training
        self.get_item_metadata = get_item_metadata

        if is_training:
            frame_rate_scales = np.linspace(
                self.min_fps_scale,
                self.max_fps_scale,
                self.num_fps_scale,
            )
            self.frame_rate_scales = list(set(frame_rate_scales) | {1})
        else:
            self.frame_rate_scales = [1.0]
            self.num_samples_per_clip = 1

    def __iter__(self) -> List:
        indices = self.get_indices()
        start_index = 0
        n_samples = len(indices)
        while start_index < n_samples:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)
            h_scale = crop_h / self.crop_size_h
            w_scale = crop_w / self.crop_size_w
            fps_scale = random.choice(self.frame_rate_scales)
            video_fps = int(self.video_fps * fps_scale)
            audio_fps = int(self.audio_fps * fps_scale)

            # Find the maximal batch size to contain no more than
            # `self.max_num_clips_per_batch` clips.
            batch_end_index = min(start_index + batch_size, n_samples)
            end_index = start_index
            sum_batch_clips = 0
            num_batch_clips = []
            while end_index < batch_end_index:
                # Computed the weighted number of clips in the batch, taking video
                # length, image size into account to make sure the batch fits into the
                # memory.
                num_clips_per_second = (
                    self.num_clips_per_second_train
                    if self.is_training
                    else self.num_clips_per_second_val
                )
                metadata = self.get_item_metadata(self.img_indices[end_index])
                if num_clips_per_second > 0:
                    num_clips = max(
                        1,
                        int(metadata["video_duration"] * num_clips_per_second),
                    )
                else:
                    clip_duration = self.default_frames / self.video_fps
                    num_clips = int(
                        metadata["total_video_frames"]
                        - clip_duration * metadata["video_fps"]
                    )
                num_batch_clips.append(num_clips)
                sum_batch_clips += num_clips * h_scale * w_scale
                end_index += 1
                if sum_batch_clips > self.max_num_clips_per_batch:
                    break

            video_ids = indices[start_index:end_index]
            start_index += len(video_ids)

            if len(video_ids) > 0:
                batch = [
                    (
                        crop_h,
                        crop_w,
                        video_id,
                        self.default_frames,
                        num_batch_clips[i],
                        video_fps,
                        audio_fps,
                        self.num_samples_per_clip,
                    )
                    for i, video_id in enumerate(video_ids)
                ]
                yield batch
            else:
                logger.warning("No data in the current batch.")

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n \t base_im_size=(h={}, w={})\n \t base_batch_size={}\n \t video_fps={}\n \taudio_fps={}\n \tn_frames={}".format(
            self.crop_size_h,
            self.crop_size_w,
            self.batch_size_gpu0,
            self.video_fps,
            self.audio_fps,
            self.default_frames,
        )
        repr_str += self.extra_repr()
        repr_str += "\n)"
        return repr_str


@SAMPLER_REGISTRY.register(name="video_clip_batch_sampler_ddp")
class VideoClipBatchSamplerDDP(VariableBatchSamplerDDP):
    """Batch sampler for videos.

    Args:
        opts: Command line argument.
        n_data_samples: Number of samples in the dataset.
        is_training: Training or validation mode. Default: False.
        get_item_metadata: A callable that provides sample metadata, given sample index.
    """

    def __init__(
        self,
        opts,
        n_data_samples: int,
        is_training: Optional[bool] = False,
        get_item_metadata: Optional[Callable[[int], Dict]] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        self.default_frames = getattr(opts, "sampler.vcbs.num_frames_per_clip")
        self.video_fps = getattr(opts, "sampler.vcbs.video_fps")
        self.audio_fps = getattr(opts, "sampler.vcbs.audio_fps")
        self.min_fps_scale = getattr(opts, "sampler.vcbs.min_clip_fps_scale")
        self.max_fps_scale = getattr(opts, "sampler.vcbs.max_clip_fps_scale")
        self.num_fps_scale = getattr(opts, "sampler.vcbs.video_fps_num_scales")
        self.num_samples_per_clip = getattr(opts, "sampler.vcbs.num_samples_per_clip")
        self.num_clips_per_second_train = getattr(
            opts, "sampler.vcbs.num_clips_per_second_train"
        )
        self.num_clips_per_second_val = getattr(
            opts, "sampler.vcbs.num_clips_per_second_val"
        )
        self.max_num_clips_per_batch = getattr(
            opts, "sampler.vcbs.max_num_clips_per_batch"
        )
        self.is_training = is_training
        self.get_item_metadata = get_item_metadata

        if is_training:
            frame_rate_scales = np.linspace(
                self.min_fps_scale,
                self.max_fps_scale,
                self.num_fps_scale,
            )
            self.frame_rate_scales = list(set(frame_rate_scales) | {1})
        else:
            self.frame_rate_scales = [1.0]

    def __iter__(self) -> List:
        indices = self.get_indices_rank_i()
        start_index = 0
        n_samples = len(indices)
        while start_index < n_samples:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)
            h_scale = crop_h / self.crop_size_h
            w_scale = crop_w / self.crop_size_w
            fps_scale = random.choice(self.frame_rate_scales)
            video_fps = int(self.video_fps * fps_scale)
            audio_fps = int(self.audio_fps * fps_scale)

            # Find the maximal batch size to contain no more than
            # `self.max_num_clips_per_batch` clips.
            batch_end_index = min(start_index + batch_size, n_samples)
            end_index = start_index
            sum_batch_clips = 0
            num_batch_clips = []
            while end_index < batch_end_index:
                # Computed the weighted number of clips in the batch, taking video
                # length, image size into account to make sure the batch fits into the
                # memory.
                num_clips_per_second = (
                    self.num_clips_per_second_train
                    if self.is_training
                    else self.num_clips_per_second_val
                )
                metadata = self.get_item_metadata(self.img_indices[end_index])
                if num_clips_per_second > 0:
                    num_clips = max(
                        1,
                        int(metadata["video_duration"] * num_clips_per_second),
                    )
                else:
                    clip_duration = self.default_frames / self.video_fps
                    num_clips = int(
                        metadata["total_video_frames"]
                        - clip_duration * metadata["video_fps"]
                    )
                num_batch_clips.append(num_clips)
                sum_batch_clips += num_clips * h_scale * w_scale
                end_index += 1
                if sum_batch_clips > self.max_num_clips_per_batch:
                    break

            video_ids = indices[start_index:end_index]
            start_index += len(video_ids)

            if len(video_ids) > 0:
                batch = [
                    (
                        crop_h,
                        crop_w,
                        video_id,
                        self.default_frames,
                        num_batch_clips[i],
                        video_fps,
                        audio_fps,
                        self.num_samples_per_clip,
                    )
                    for i, video_id in enumerate(video_ids)
                ]
                yield batch
            else:
                logger.warning("No data in the current batch.")

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n \t base_im_size=(h={}, w={})\n \t base_batch_size={}\n \t video_fps={}\n \taudio_fps={}\n \tn_frames={}".format(
            self.crop_size_h,
            self.crop_size_w,
            self.batch_size_gpu0,
            self.video_fps,
            self.audio_fps,
            self.default_frames,
        )
        repr_str += self.extra_repr()
        repr_str += "\n)"
        return repr_str
