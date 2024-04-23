#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys
from typing import Dict, Optional, Union

from corenet.data.transforms.base_transforms import BaseTransformation
from corenet.utils.import_utils import ensure_library_is_available

try:
    import decord
except ImportError:
    pass

import av
import torch

from corenet.data.video_reader import VIDEO_READER_REGISTRY
from corenet.data.video_reader.pyav_reader import BaseAVReader
from corenet.utils import logger


@VIDEO_READER_REGISTRY.register(name="decord")
class DecordAVReader(BaseAVReader):
    """
    Video Reader using Decord.
    """

    def __init__(self, *args, **kwargs):
        ensure_library_is_available("decord")
        super().__init__(*args, **kwargs)

    def read_video(
        self,
        av_file: str,
        stream_idx: int = 0,
        audio_sample_rate: int = -1,
        custom_frame_transforms: Optional[BaseTransformation] = None,
        video_only: bool = False,
        *args,
        **kwargs
    ) -> Dict:
        video_frames = audio_frames = None
        video_fps = audio_fps = None
        decord.bridge.set_bridge("torch")
        # We have to use av package to obtain audio fps, which is not available in
        # decord.
        with av.open(str(av_file)) as container:
            available_streams = []
            for stream in container.streams:
                if stream.type == "audio":
                    # Skip audio stream if audio not required.
                    if video_only:
                        continue
                    audio_fps = container.streams.audio[0].sample_rate
                available_streams.append(stream.type)
        for stream_type in available_streams:
            if stream_type == "video":
                with open(str(av_file), "rb") as f:
                    video_reader = decord.VideoReader(f, ctx=decord.cpu(0))
                    n_video_frames = video_reader._num_frame
                    video_frames = []
                    frame_transforms = (
                        self.frame_transforms
                        if custom_frame_transforms is None
                        else custom_frame_transforms
                    )
                    for _ in range(n_video_frames):
                        video_frame = video_reader.next()  # H, W, C
                        video_frame = video_frame.permute(2, 0, 1)  # C, H, W
                        video_frame = frame_transforms({"image": video_frame})["image"]
                        video_frames.append(video_frame)
                    video_frames = torch.stack(video_frames)
                    video_fps = video_reader.get_avg_fps()
            if stream_type == "audio":
                with open(str(av_file), "rb") as f:
                    audio_reader = decord.AudioReader(
                        f, ctx=decord.cpu(0), sample_rate=audio_sample_rate
                    )
                    audio_frames = torch.tensor(audio_reader._array).transpose(0, 1)
                    audio_fps = (
                        audio_sample_rate if audio_sample_rate > 0 else audio_fps
                    )

        return {
            "audio": audio_frames,  # expected format T x C
            "video": video_frames,  # expected format T x C x H x W
            "metadata": {
                "audio_fps": audio_fps,
                "video_fps": video_fps,
                "filename": av_file,
            },
        }

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
        vmetadata = {}
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        with av.open(video_path) as container:
            vmetadata["filename"] = video_path
            video_stream = container.streams.video[0]
            vmetadata["total_video_frames"] = len(vr)
            vmetadata["video_fps"] = float(vr.get_avg_fps())
            vmetadata["video_duration"] = (
                vmetadata["total_video_frames"] / vmetadata["video_fps"]
            )
        return vmetadata

    def build_audio_metadata(
        self, video_path: str
    ) -> Dict[str, Union[str, float, int]]:
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
            }
        """
        metadata = {}
        # Decord doesn't provide audio_fps. Thus, we use PyAV.
        with av.open(str(video_path)) as container:
            for stream in container.streams:
                container.seek(0)
                if stream.type == "audio":
                    if "audio_fps" in metadata:
                        raise ValueError(
                            "Multiple audio streams exist while at most 1 is expected."
                        )
                    audio_stream = container.streams.audio[0]
                    metadata["audio_channels"] = len(audio_stream.layout.channels)
                    metadata["audio_fps"] = audio_stream.sample_rate
                    # `audio_stream.frames` does not work for unknown reason.
                    metadata["total_audio_frames"] = self._get_total_audio_frames(
                        video_path, audio_stream.sample_rate
                    )
                    metadata["audio_duration"] = (
                        metadata["total_audio_frames"] / metadata["audio_fps"]
                    )
        return metadata

    @staticmethod
    def _get_total_audio_frames(video_path: str, sample_rate: Union[int, float]) -> int:
        """Returns the total number frames in the audio stream of @video_path.

        Args:
            video_path: Path to the local video file.
            sample_rate: Sample rate of the audio stream.
        """
        with open(str(video_path), "rb") as f:
            # FIXME: Type of the @sample_rate is Union[int,float], but decord expects an
            # integer. We should investigate what happens when floating point values are
            # passed to this function. This issue might cause some misalignment between
            # the audio and the video.
            audio_reader = decord.AudioReader(
                f, ctx=decord.cpu(0), sample_rate=sample_rate
            )
            result = audio_reader.shape[1]
        return result
