#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import subprocess
import sys
import warnings
from math import isclose
from typing import Any, Dict, Optional, Tuple, Union

import torch

from corenet.data.transforms.base_transforms import BaseTransformation
from corenet.data.transforms.common import Compose
from corenet.data.video_reader import VIDEO_READER_REGISTRY, ffmpeg_utils
from corenet.data.video_reader.base_av_reader import BaseAVReader
from corenet.utils.import_utils import ensure_library_is_available

try:
    import ffmpeg
except ImportError:
    pass


@VIDEO_READER_REGISTRY.register(name="ffmpeg")
class FFMPEGReader(BaseAVReader):
    """
    This is an experimental AVReader that decodes videos using ffmpeg subprocess.
    This reader handles memory better than DecordReader with large datasets. Hence, we
    can enable --dataset.persistent_workers and --dataset.pin_memory, without OOM Error,
    to speedup the training. However, the improvement in accuracy isn't guaranteed yet.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        ensure_library_is_available("ffmpeg")
        super().__init__(opts, *args, **kwargs)

    def read_video(
        self,
        filename: str,
        stream_idx: int = 0,
        audio_sample_rate: int = -1,
        video_fps: float = -1,
        custom_frame_transforms: Optional[BaseTransformation] = None,
        video_only: bool = False,
        threads: int = 1,
        crop_w_h_x_y: Optional[Tuple[int, int, int, int]] = None,
        ffmpeg_loglevel: str = "error",
        *args,
        **kwargs,
    ) -> Dict:
        """Reads the video frames and audio samples of a video file into torch Tensors.

        Args:
            filename: Path of the video file.
            stream_idx: Video stream index, for files with multiple video streams. This
                subclass only supports videos with a single video stream. Defaults to 0.
            audio_sample_rate: Controls the audio sample rate when reading audio. If not
                specified (-1), the file's original sample rate gets used.
                Defaults to -1.
            video_fps: Controls the frame rate for reading video. If not specified (-1),
                the file's average frame rate gets used. If the input video is encoded
                with dynamic frame rate, this reader instructs ffmpeg to read the video
                with constant (average) frame rate.
            custom_frame_transforms: If provided, the given transformation gets used,
                rather then the default ``BaseAVReader.get_frame_transform()`` for
                transforming individual frames. Defaults to None.
            video_only: When True, the audio stream gets skipped. Defaults to False.
            threads: Number of cpu threads to use for decoding and transforming the
                video. Note that we don't have full control over ffmpeg, and some
                ffmpeg components may ignore this flag. Defaults to 1.
            crop_w_h_x_y: If provided, the frames will be cropped as early as possible
                within ffmpeg pipeline, before being sent to Python. Defaults to None.
                For example, given crop_w_h_x_y=(10, 20, 50, 100), the video tensor will
                be a tensor of shape [T, C, 40, 30], cropped at 50<=x<60 and 100<=y<120,
                where T is the temporal length and is the number of channels.
            ffmpeg_loglevel: Controls the log level of ffmpeg library. NOTE: Values
                other than "error" may cause too many lines of log, and may result in
                buffer overflows resulting in halted training. Defaults to "error".

        Tensor shape abbreviations:
            T, T_audio, T_video: Temporal lengths.
            C: Number of color channels.
            H, W: Height, Width.

        Returns: A dictionary of the following format {
            "audio": Tensor [T_audio,C],
            "video": Tensor [T_video,C,H,W],
            metadata: {
                "audio_fps": float,
                "video_fps": float,
                "filename": str,
            },
        }

        Note:
            * For random cropping, please use custom_frame_transforms argument. This
            argument (crop_w_h_x_y) translates to `crop=out_w:out_h:x:y` static ffmpeg
            cli argument that applies the same bounding box to all frames.
        """
        if stream_idx != 0:
            raise NotImplementedError(
                f"Reading videos with stream_idx={stream_idx} is not supported yet."
            )

        try:
            video_metadata, extras = ffmpeg_utils.get_video_metadata(
                filename, return_extras=True
            )
            if extras["rotation"] != 0:
                raise NotImplementedError(
                    "Reading videos with rotated frames"
                    f" (rotation={extras['rotation']}) is not implemented yet."
                )

            video = ffmpeg.input(
                filename,
                threads=str(threads),
                loglevel=ffmpeg_loglevel,
            ).video
            if crop_w_h_x_y is not None:
                width, height, x, y = crop_w_h_x_y
                video = video.crop(width=width, height=height, x=x, y=y)
            else:
                height = video_metadata["height"]
                width = video_metadata["width"]
            if video_fps != -1:
                video = video.filter("fps", fps=video_fps)
            video = video.output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                threads=str(threads),
                loglevel=ffmpeg_loglevel,
            )
            video = video.global_args(
                "-threads",
                str(threads),
                "-loglevel",
                ffmpeg_loglevel,
            )
            video = subprocess.run(
                video.compile(),
                capture_output=True,
                # See https://github.com/kkroening/ffmpeg-python/issues/782
                stdin=subprocess.DEVNULL,
            ).stdout

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                video = torch.frombuffer(video, dtype=torch.uint8)
            video = video.reshape(-1, height, width, 3)

            if video_fps == -1:
                video_fps = video_metadata["video_fps"]

            expected_frames = int(video_metadata["video_duration"] * video_fps)
            if not isclose(expected_frames, video.shape[0], rel_tol=0.05, abs_tol=1):
                raise ValueError(
                    "Expected"
                    f" {video_metadata['video_duration']}*{video_fps}={expected_frames} video"
                    f" frames, but got {video.shape[0]} frames."
                )

            video = video.permute(0, 3, 1, 2)  # [T,H,W,C] -> [T,C,H,W]

            if not video_only:
                audio, audio_metadata = self.read_audio(
                    filename, audio_sample_rate=audio_sample_rate, threads=threads
                )
        except ffmpeg.Error as e:
            raise RuntimeError(e.stderr) from e

        video = self._transform_video_frames(
            video,
            (
                self.frame_transforms
                if custom_frame_transforms is None
                else custom_frame_transforms
            ),
        )

        result = {
            "audio": audio if not video_only else None,
            "video": video,
            "metadata": {
                "audio_fps": audio_metadata["audio_fps"] if not video_only else None,
                "video_fps": video_fps,
                "filename": filename,
            },
        }
        return result

    def _transform_video_frames(
        self, video: torch.Tensor, transformation: BaseTransformation
    ) -> torch.Tensor:
        """Applies frame_transforms to the individual video frames.

        Args:
            video: Tensor[T,C,W,H], to be transformed.
            frame_transforms: Transformation that operates on {"image": Tensor[C,W,H]}.

        Returns:
            Transformed tensor of shape [T,C,W,H].

        Note:
            * If the transformation is a No-Op (ie. ``Compose([])``), returns the input
            as is. The No-Op transformation can be used by datasets that apply ToTensor
            after cropping, to save compute.
        """
        if isinstance(transformation, Compose) and transformation.img_transforms == []:
            # No-Op frame transform
            pass
        else:
            video = torch.stack(
                [transformation({"image": frame})["image"] for frame in video]
            )

        return video

    @classmethod
    def read_audio(
        cls, filename: str, audio_sample_rate: int = -1, threads: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reads the audio tensor and audio stream's metadata of a given video file.

        Args:
            filename: Path of the video file.
            audio_sample_rate: Controls the audio sample rate when reading audio. If not
                specified (-1), the file's original sample rate gets used.
                Defaults to -1.
            threads: Number of cpu threads to use for decoding and transforming the
                video. Note that we don't have full control over ffmpeg, and some
                ffmpeg components may ignore this flag. Defaults to 1.

        Returns:
            (audio_tensor, metadata) tuple, where audio_tensor has shape [T,C] and the
            metadata has the following schema: {
                "audio_fps": float,
                "audio_duration": float,
                "audio_channels": int,
            }.
        """
        audio_metadata = cls.build_audio_metadata(filename)
        if audio_sample_rate == -1:
            audio_sample_rate = audio_metadata["audio_fps"]
        else:
            audio_metadata["audio_fps"] = audio_sample_rate

        # F16LE is 16-bit little-endian signed PCM (raw) audio.
        # See: https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-audio-raw.html
        if sys.byteorder == "little":
            audio_format = "f32le"
        elif sys.byteorder == "big":
            audio_format = "f32be"
        else:
            raise NotImplementedError(f"Unknown byte order '{sys.byteorder}'.")

        audio, _ = (
            ffmpeg.input(filename, threads=str(threads))
            .audio.output(
                "pipe:",
                format=audio_format,
                acodec=f"pcm_{audio_format}",
                ar=str(audio_sample_rate),
                threads=str(threads),
            )
            .global_args("-vn", "-threads", str(threads))
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = torch.frombuffer(audio, dtype=torch.float32)
        audio = audio.reshape(-1, audio_metadata["audio_channels"])
        expected_frames = int(audio_metadata["audio_duration"] * audio_sample_rate)
        if not isclose(expected_frames, audio.shape[0], rel_tol=0.05, abs_tol=1):
            raise ValueError(
                "Expected"
                f" {audio_metadata['audio_duration']}*{audio_sample_rate}={expected_frames} audio"
                f" frames, but got {audio.shape[0]} frames."
            )
        return audio, audio_metadata

    @classmethod
    def build_video_metadata(cls, video_path: str) -> Dict[str, Union[str, float, int]]:
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
                "width": <int>,
                "height": <int>,
            }
        """
        return ffmpeg_utils.get_video_metadata(video_path)

    @classmethod
    def build_audio_metadata(cls, video_path: str) -> Dict[str, Union[str, float, int]]:
        """Generate the audio metadata for a given video.

        Args:
            video_path: A video file path.

        Returns:
            The audio metadata of the corresponding video. The metadata format is:
            {
                "audio_channels": int,
                "audio_fps": int,
                "total_audio_frames": int,
                "audio_duration": float,
                "audio_channels": int,
            }
        """
        return ffmpeg_utils.get_audio_metadata(video_path)
