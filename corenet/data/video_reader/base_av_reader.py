#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import random
from typing import Dict, Optional, Tuple, Union

import av
import numpy
import torch
import torch.nn.functional as F
from torch import Tensor

from corenet.data.transforms import BaseTransformation
from corenet.data.transforms import image_pil as T
from corenet.data.transforms.common import Compose
from corenet.options.utils import (
    extend_selected_args_with_prefix,
    extract_opts_with_prefix_replacement,
)
from corenet.utils import logger


class VideoDurationDoesNotMatchAudioDurationError(AssertionError):
    pass


audio_video_duration_warnings = set()


def check_audio_video_duration(
    vid_filename: str, audio_sec: float, video_sec: float
) -> None:
    """Check audio/video alignment, throw an exception or print a warning if it is off."""
    difference = numpy.fabs(audio_sec - video_sec)
    if difference > 0.1:
        message = f"Audio duration {audio_sec} mismatches video duration {video_sec} for {vid_filename}"
        if difference > 0.2:
            raise VideoDurationDoesNotMatchAudioDurationError(message)
        if vid_filename not in audio_video_duration_warnings:
            audio_video_duration_warnings.add(vid_filename)
            logger.warning(message)


class BaseAVReader(object):
    """
    Base AudioVideo Reader

    Args:
        opts: command line arguments
        is_training: Training or validation mode. Default: `False`.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if (
            cls != BaseAVReader
        ):  # Shouldn't run for subclasses that don't override add_arguments
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--video-reader.name",
            type=str,
            default="pyav",
            help="Name of video reader.",
        )
        group.add_argument(
            "--video-reader.fast-video-decoding",
            action="store_true",
            help="Multi-threaded fast video decoding using pyav.",
        )
        group.add_argument(
            "--video-reader.frame-stack-format",
            type=str,
            default="sequence_first",
            choices=["sequence_first", "channel_first"],
            help="Sequence first (NTCHW) or channel first (NCTHW) format for stacking"
            " video frames.",
        )
        # We want to be able to re-use the "image" augmentations on the video frames.
        # As the use of "--image-augmentation.*" argparse prefix for video datasets can
        # be confusing, we use "--frame-augmentation.*" prefix.
        parser = extend_selected_args_with_prefix(
            parser,
            match_prefix="--image-augmentation.",
            additional_prefix="--frame-augmentation.",
        )
        return parser

    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.opts = opts
        self.fast_decoding = getattr(opts, "video_reader.fast_video_decoding")
        frame_stack_format = getattr(
            opts, "video_reader.frame_stack_format", "sequence_first"
        )

        if frame_stack_format not in ["sequence_first", "channel_first"]:
            logger.error(
                "Frame stacking format should be either sequence_first or channel_first."
            )
        self.channel_first_format = frame_stack_format == "channel_first"

        self.frame_transforms = self.get_frame_transform(opts, is_training=is_training)

        self.num_frames_cache = dict()

    @staticmethod
    def get_frame_transform(
        opts: argparse.Namespace, is_training: bool, *args, **kwargs
    ) -> BaseTransformation:
        if is_training:
            auto_augment = getattr(
                opts,
                "frame_augmentation.auto_augment.enable",
            )
            rand_augment = getattr(
                opts,
                "frame_augmentation.rand_augment.enable",
            )
            random_erase = getattr(
                opts,
                "frame_augmentation.random_erase.enable",
            )
        # We want to be able to re-use the "image" augmentations on the video frames.
        # As the use of "--image-augmentation.*" argparse prefix for video datasets can
        # be confusing, we use "--frame-augmentation.*" prefix.
        frame_opts = extract_opts_with_prefix_replacement(
            opts,
            match_prefix="frame_augmentation.",
            replacement_prefix="image_augmentation.",
        )
        if is_training:
            aug_list = []
            if auto_augment and rand_augment:
                logger.error(
                    "AutoAugment and RandAugment are mutually exclusive. Use either of"
                    " them, but not both."
                )
            elif auto_augment:
                aug_list.append(T.AutoAugment(opts=frame_opts))
            elif rand_augment:
                aug_list.append(T.RandAugment(opts=frame_opts))

            aug_list.append(T.ToTensor(opts=frame_opts))

            if random_erase:
                aug_list.append(T.RandomErasing(opts=frame_opts))

            return Compose(opts=frame_opts, img_transforms=aug_list)
        else:
            return T.ToTensor(opts=frame_opts)

    def __repr__(self):
        return "{}(\n\tfast_decoding={}\n\tchannel_first_format={}\n)".format(
            self.__class__.__name__,
            self.fast_decoding,
            self.channel_first_format,
        )

    def check_video(self, filename: str) -> bool:
        try:
            # Adapted from basic demo: https://pyav.org/docs/stable/#basic-demo
            with av.open(filename) as container:
                # Decode the first video channel.
                for frame in container.decode(video=0):
                    frame_idx = frame.index
                    break
                return True
        except Exception as e:
            return False

    def read_video(
        self,
        filename: str,
        stream_idx: int = 0,
        audio_sample_rate: int = -1,
        custom_frame_transforms: Optional[BaseTransformation] = None,
        video_only: bool = False,
        *args,
        **kwargs,
    ) -> Dict:
        raise NotImplementedError

    def build_video_metadata(
        self, video_path: str
    ) -> Dict[str, Union[str, float, int]]:
        """Generate the metadata for a given video.

        Args:
            video_path: A video file path.

        Returns:
            The metadata of the corresponding video. The generated metadata format is:
            {
                "filename": <str>,
                "video_fps": <float>,
                "total_video_frames" <int>,
                "video_duration": <float>,
            }
        """
        raise NotImplementedError

    def num_frames(self, filename: str) -> int:
        if filename in self.num_frames_cache:
            return self.num_frames_cache[filename]
        else:
            with av.open(filename) as container:
                total_frames = container.streams.video[0].frames
            self.num_frames_cache[filename] = total_frames
            return total_frames

    @staticmethod
    def random_sampling(
        total_video_frames: int,
        video_frames_per_clip: int,
        clips_per_video: int,
        total_audio_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        For a given video, sample `clips_per_video` indices randomly along with aligned
        audio indices (optionally).

        Args:
            total_video_frames : number of video frames in the given video.
            video_frames_per_clip : number of frames required per clip.
            clips_per_video : number of clips needed from a given video.
            total_audio_frames : number of audio frames in the given video.

        Return:
            vclip_ids : indices corresponding to video frames [Tensor (clips_per_video x
                video_frames_per_clip)].
            aclip_ids : indices corresponding to audio frames [Tensor (clips_per_video x
                audio_frames_per_clip)].
        """
        clip_start_frame_ids = torch.randint(
            total_video_frames - video_frames_per_clip + 1, (clips_per_video,)
        )
        vclip_ids = clip_start_frame_ids[:, None] + torch.arange(video_frames_per_clip)

        aclip_ids = None
        if total_audio_frames is not None:
            a_start_frame = (
                clip_start_frame_ids * total_audio_frames / total_video_frames
            ).long()
            a_step_size = int(
                video_frames_per_clip * total_audio_frames / total_video_frames
            )
            aclip_ids = a_start_frame[:, None] + torch.arange(a_step_size)
        return vclip_ids, aclip_ids

    @staticmethod
    def uniform_sampling(
        total_video_frames: int,
        video_frames_per_clip: int,
        clips_per_video: int,
        total_audio_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        For a given video, sample `clips_per_video` indices uniformly along with aligned
        audio indices (optionally).

        Args:
            total_video_frames : number of video frames in the given video.
            video_frames_per_clip : number of frames required per clip.
            clips_per_video : number of clips needed from a given video.
            total_audio_frames : number of audio frames in the given video.

        Return:
            vclip_ids : indices corresponding to video frames [Tensor (clips_per_video x
                video_frames_per_clip)].
            aclip_ids : indices corresponding to audio frames [Tensor (clips_per_video x
                audio_frames_per_clip)].
        """
        clip_start_frame_ids = torch.linspace(
            0, total_video_frames - video_frames_per_clip, clips_per_video
        )
        vclip_ids = (
            clip_start_frame_ids[:clips_per_video, None]
            + torch.arange(video_frames_per_clip)
        ).long()

        aclip_ids = None
        if total_audio_frames is not None:
            a_start_frame = (
                clip_start_frame_ids * total_audio_frames / total_video_frames
            ).long()
            a_step_size = int(
                video_frames_per_clip * total_audio_frames / total_video_frames
            )
            aclip_ids = a_start_frame[:clips_per_video, None] + torch.arange(
                a_step_size
            )
        return vclip_ids, aclip_ids

    def read_video_file_into_clips(
        self,
        vid_filename: str,
        num_frames_per_clip: int,
        clips_per_video: int,
        is_training: bool,
        video_only: bool = False,
        output_video_fps: float = -1,
        output_audio_fps: int = -1,
        num_samples_per_clip: int = 1,
        custom_frame_transforms: Optional[BaseTransformation] = None,
        *args,
        **kwargs,
    ) -> Dict:
        """Read a video file into clips and sample the clips at the specified
        video/audio frame rate. First, we read all the video and audio frames into the
        memory, where audio is at `output_audio_fps` if specified; then we sample
        `clips_per_video` clips from the entire video/audio tensor. If the desired video
        frame rate is specified, we subsample the video at frame rate
        `output_video_fps`. Despite whether the video is subsampled or not, there are
        `num_frames_per_clip` video frames in each clip.

        Args:
            vid_filename: The path of the video to be read.
            num_frames_per_clip: Number of frames per clip to read.
            clips_per_video: Number of clips to read for each video.
            is training: A boolean of whether the model is in training.
            output_video_fps: The frame rate of the output video. Default is -1, which
                means no resampling is required.
            output_audio_fps: The frame rate of the output audio. Default is -1, which
                means no resampling is required.
            num_samples_per_clip: Number of random samples to generate per clip.
            custom_frame_transforms: If provided, the transformation will be used
                instead of the default @BaseAVReader.get_frame_transforms.
                Note: Be careful when customizing frame transforms, because there might
                    exist slight differences between the data type of frames read by
                    different AVReaders before ToTensor() gets applied.

        Tensor shape abbreviations:
            N: Number of clips.
            T, T_audio, T_video: Temporal lengths.
            C: Number of color channels.
            H, W: Height, Width.

        Returns: A dictionary of the following format {
            "audio": Tensor [N,T_audio,C],
            "video": Tensor [N,T_video,C,H,W],
            metadata: {
                "audio_fps": float,
                "video_fps": float,
                "filename": str,
                "video_frame_timstamps": Tensor[N,T_video]
            },
        }
        """

        av_data = self.read_video(
            vid_filename,
            audio_sample_rate=output_audio_fps,
            video_only=video_only,
            custom_frame_transforms=custom_frame_transforms,
            # At this stage, we read the video with its original frame rate, so that
            # we can introduce small perterbations in the frame selection process
            # (i.e. down sampling) during training.
            video_fps=-1,
        )
        torch_video = av_data["video"]
        torch_audio = av_data["audio"]
        metadata = av_data["metadata"]
        original_video_fps = metadata["video_fps"]
        audio_fps = metadata["audio_fps"]

        assert isinstance(
            torch_video, Tensor
        ), f"Video read from {vid_filename} expected to be a tensor."

        num_video_frames = torch_video.shape[0]
        # If we need to downsample the frames, read the video frames the same length in
        # second, then downsample output frames.
        if output_video_fps > 0:
            clip_duration = num_frames_per_clip / output_video_fps
            num_video_frames_to_sample = min(
                num_video_frames,
                round(clip_duration * original_video_fps),
            )
            metadata["video_fps"] = output_video_fps
        # If no frame resampling is required, read @num_frames_per_clip
        # consecutive frames.
        else:
            num_video_frames_to_sample = num_frames_per_clip
            clip_duration = num_frames_per_clip / original_video_fps

        if torch_audio is None:
            num_audio_frames = None
        else:
            num_audio_frames = torch_audio.shape[0]
            check_audio_video_duration(
                vid_filename,
                num_audio_frames / audio_fps,
                num_video_frames / original_video_fps,
            )

        if num_video_frames >= num_frames_per_clip:
            sampling_fn = self.random_sampling if is_training else self.uniform_sampling
            vclip_ids, aclip_ids = sampling_fn(
                total_video_frames=num_video_frames,
                video_frames_per_clip=num_video_frames_to_sample,
                clips_per_video=clips_per_video,
                total_audio_frames=num_audio_frames,
            )
        else:
            vclip_ids = aclip_ids = None

        num_samples_per_clip = num_samples_per_clip if is_training else 1
        video_clips, metadata = self._generate_video_clips(
            full_video_tensor=torch_video,
            vclip_ids=vclip_ids,
            metadata=metadata,
            clips_per_video=clips_per_video,
            num_frames_per_clip=num_frames_per_clip,
            original_video_fps=original_video_fps,
            output_video_fps=output_video_fps,
            is_training=is_training,
            num_samples_per_clip=num_samples_per_clip,
        )

        if torch_audio is None:
            audio_clips = None
        else:
            audio_clips = self._generate_audio_clips(
                full_audio_tensor=torch_audio,
                aclip_ids=aclip_ids,
                clip_duration=clip_duration,
                clips_per_video=clips_per_video,
                audio_fps=audio_fps,
                num_samples_per_clip=num_samples_per_clip,
            )

        if audio_clips is not None:
            assert (
                video_clips.shape[0] == audio_clips.shape[0]
            ), f"Video and audio doesn't have the same number of clips, got {video_clips.shape[0]} and {audio_clips.shape[0]}."

        return {
            "video": video_clips,
            "audio": audio_clips,
            "metadata": metadata,
        }

    def _generate_video_clips(
        self,
        full_video_tensor: torch.Tensor,
        vclip_ids: Optional[torch.Tensor],
        metadata: Dict,
        num_frames_per_clip: int,
        clips_per_video: int,
        original_video_fps: float,
        output_video_fps: float,
        is_training: bool,
        num_samples_per_clip: Optional[int],
    ) -> Tuple[torch.Tensor, Dict]:
        """Given entire video tensor of a video file and the indices of the sampled
        video frames, return the video clips. If there's not enough frames, the last
        frame will be padded. If `output_video_fps` is smaller than
        `original_video_fps`, video frames will be downsampled accordingly.

        Args:
            full_video_tensor: A [T x 3 x H x W] tensor of all frames of a video, where
                T is the total number of frames, H and W is the height and width of each
                frame image.
            vclip_ids: A [clips_per_video x N] tensor of the index of the sampled frames
                in `full_video_tensor`, where N is the number of sampled frames.
            metadata: A dictionary of the video's metadata information.
            num_frames_per_clip: Number of frames of the output clips.
            clips_per_video: Number of clips per video.
            original_video_fps: The frame rate of the video.
            output_video_fps: The frame rate of the output video clips.
            is_training: Whether it's in training mode. No randomness is applied if set
                False.
            num_samples_per_clip: Number of samples per clip to generate when the frames
                are downsampled.

        Returns:
            video_clips: A [clips_per_video x num_frames_per_clip x 3 x H x W] tensor.
        """
        (
            num_video_frames,
            frame_channels,
            frame_height,
            frame_width,
        ) = full_video_tensor.shape

        if num_video_frames < num_frames_per_clip:
            # Repeat the last frame.
            num_video_frames_to_pad = num_frames_per_clip - num_video_frames
            padded_video = torch.zeros(
                size=(
                    num_frames_per_clip,
                    frame_channels,
                    frame_height,
                    frame_width,
                ),
                dtype=full_video_tensor.dtype,
                device=full_video_tensor.device,
            )
            padded_video[:num_video_frames] = full_video_tensor
            padded_video[num_video_frames:, :, :, :] = full_video_tensor[-1].unsqueeze(
                0
            )
            num_repeats = (
                clips_per_video * num_samples_per_clip
                if is_training
                else clips_per_video
            )
            video_clips = padded_video.repeat((num_repeats, 1, 1, 1, 1))
            vclip_ids = torch.zeros((num_repeats, 1)) + torch.arange(num_video_frames)
            vclip_ids = F.pad(vclip_ids, (0, num_video_frames_to_pad), mode="replicate")
        else:
            if 0 < output_video_fps < original_video_fps:
                vclip_ids = _downsample_frame_indices(
                    frame_indices=vclip_ids,
                    input_fps=original_video_fps,
                    output_num_frames=num_frames_per_clip,
                    output_fps=output_video_fps,
                    num_samples=num_samples_per_clip,
                    random_frames=is_training,
                )
            else:
                if num_samples_per_clip > 1:
                    # Duplicate video if multiple samples are generated for each clip at
                    # training time.
                    vclip_ids = torch.repeat_interleave(
                        vclip_ids, num_samples_per_clip, dim=0
                    )
            video_clips = full_video_tensor[vclip_ids]

        metadata["video_frame_timestamps"] = vclip_ids / original_video_fps
        if self.channel_first_format:
            video_clips = video_clips.transpose(1, 2)
        return video_clips, metadata

    def _generate_audio_clips(
        self,
        full_audio_tensor: torch.Tensor,
        aclip_ids: Optional[torch.Tensor],
        clip_duration: float,
        clips_per_video: int,
        audio_fps: float,
        num_samples_per_clip: Optional[int] = 1,
    ) -> torch.Tensor:
        """Given entire audio tensor of a video file and the indices of the sampled
        audio frames, pad to the desire shape if needed and return the audio clips.

        Args:
            full_audio_tensor: A [T x C] tensor of all frames of a audio, where T is the
                total number of frames, C is the number of channels. The audio is mono
                when C == 1 and stero when C == 2.
            aclip_ids: A [clips_per_video x N] tensor of the index of the sampled frames
                in `full_audio_tensor`, where N is the number of sampled frames.
            clip_duration: The duration in second of each clip.
            clips_per_video: Number of clips per video to generate.
            audio_fps: The frame rate of the audio. The audio is not changed in this
                function.
            num_samples_per_clip: Number of samples to generate for each clip. This is
                to match the shape of the video if multiple samples are generated during
                training.

        Returns:
            A [clips_per_video x N x C] tensor as the audio clips, where N is the number
            of frames within `clip_duration` time at `audio_fps` frame rate.
        """
        num_audio_frames = full_audio_tensor.shape[0]
        # Compute the output clip length in second with the output fps and number of
        # frames.
        expected_num_audio_frames = int(clip_duration * audio_fps)
        if full_audio_tensor.shape[0] < expected_num_audio_frames:
            num_audio_frames_to_pad = (
                expected_num_audio_frames - full_audio_tensor.shape[0]
            )
            full_audio_tensor = F.pad(
                full_audio_tensor, (0, 0, 0, num_audio_frames_to_pad)
            )
            audio_clips = full_audio_tensor.repeat((clips_per_video, 1, 1))
        else:
            # [num_frames, channels] --> [num_clips, per_clip_audio, channels]
            audio_clips = full_audio_tensor[aclip_ids]
            num_audio_frames_to_pad = expected_num_audio_frames - audio_clips.shape[1]
            audio_clips = F.pad(audio_clips, (0, 0, 0, num_audio_frames_to_pad))

        # Duplicate audio if multiple samples are generated for each clip at training
        # time.
        audio_clips = torch.repeat_interleave(audio_clips, num_samples_per_clip, dim=0)

        return audio_clips

    def dummy_audio_video_clips(
        self,
        clips_per_video: int,
        num_frames_to_sample: int,
        height: int,
        width: int,
        audio_fps: int = 16000,
        video_fps: Union[float, int] = 30,
    ) -> Dict:
        # [K, C, T, H, W] or # [K, N, T, H, W]
        # K --> number of clips, C --> Image channels, N --> Number of frames per clip,
        # H --> Height, W --> Width.
        video_tensor_size = (
            (clips_per_video, 3, num_frames_to_sample, height, width)
            if self.channel_first_format
            else (clips_per_video, num_frames_to_sample, 3, height, width)
        )

        video_clips = torch.zeros(
            size=video_tensor_size,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        # Dummy clips for audio
        audio_tensor_size = (
            clips_per_video,
            int(num_frames_to_sample * audio_fps / video_fps),
            1,
        )
        audio_clips = torch.zeros(
            size=audio_tensor_size,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        metadata = {
            "video_fps": video_fps,
            "audio_fps": audio_fps,
            "filename": "dummy.mp4",
        }
        return {
            "video": video_clips,
            "audio": audio_clips,
            "metadata": metadata,
        }


def _downsample_frame_indices(
    frame_indices: torch.Tensor,
    input_fps: float,
    output_num_frames: int,
    output_fps: float,
    num_samples: int,
    random_frames: bool,
) -> torch.Tensor:
    """Downsample frames from a frame indices at a given @output_fps and with a given
    @output_num_frames.

    This is used as a helper function for batch samplers that construct shorter video
    clips from a longer video clip. For example, we might load a 70-frame clip at 30fps
    and use it to construct 4 different batch samples corresponding to 8-frame clips at
    8fps.

    Args:
        frame_indices: A [K, N] tensor as the index of each frame, where K is the batch
            dimension, N is the number of frames.
        input_fps: The frame rate to be downsampled from.
        output_num_frames: The number of frames to output in each sample.
        output_fps: The desired fps of the output frames.
        num_samples: The number of batch samples to construct from the data.
        random_frames: If False, the input indices will be used to construct a single
            example. If True, random offsets will be applied to the frame indices as an
            augmentation.

    Returns:
        output_frame_indices: A [K_out, N_out] tensor as the frame indices to sample,
        where:
            - N_out = @output_num_frames
            - K_out = K * @num_samples.
    """
    if not random_frames and num_samples != 1:
        raise ValueError(
            "Frames are deterministic, so set num_samples to 1. Got num_samples={num_samples}."
        )
    assert num_samples >= 1, f"num_samples has to be positive, got {num_samples}."

    assert (
        input_fps > output_fps
    ), f"Output fps {output_fps} has to be smaller than the input fps {input_fps}."

    frames_per_clip = frame_indices.shape[1]
    assert (
        frames_per_clip >= output_num_frames
    ), f"Need to load more frames sample, can't sample {output_num_frames} from {frames_per_clip} frames."

    output_frame_indices = []
    desired_length_seconds = output_num_frames / output_fps
    num_frames_to_sample = min(frames_per_clip, int(desired_length_seconds * input_fps))
    positions = (torch.linspace(0, num_frames_to_sample, output_num_frames + 1)).long()

    for _ in range(num_samples):
        if random_frames:
            # Choose a starting timestamp for the frames.
            last_valid_idx = frames_per_clip - num_frames_to_sample
            frame_start_idx = random.randint(0, last_valid_idx)
        else:
            # We don't apply any offsets or randomness to the video frames, so we start
            # the video frames at index 0.
            frame_start_idx = 0

        selected_frame_indices = frame_indices[
            :,
            frame_start_idx : frame_start_idx + num_frames_to_sample,
        ]

        if random_frames:
            # Choose the middle of the chunk as the label. Then, choose the frame at
            # random from the time slices.
            selected_frame_locations = []
            for i in range(output_num_frames):
                # Select a random frame in the chunk.
                # NOTE: random.randint() is inclusive on both ends, while
                # torch.randint() is exclusive for the higher range.
                selection = random.randint(positions[i], positions[i + 1] - 1)
                selected_frame_locations.append(selection)
        else:
            # We choose frames deterministically. Since we don't apply any offsets to
            # the video, the correct frames are at the beginning of each range.
            selected_frame_locations = positions[:-1]

        output_frame_indices.append(selected_frame_indices[:, selected_frame_locations])

    return torch.cat(output_frame_indices)
