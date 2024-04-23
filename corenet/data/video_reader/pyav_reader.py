#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict, Optional, Union

import av
import numpy
import torch

from corenet.data.transforms.base_transforms import BaseTransformation
from corenet.data.video_reader import VIDEO_READER_REGISTRY, BaseAVReader
from corenet.utils import logger


@VIDEO_READER_REGISTRY.register(name="pyav")
class PyAVReader(BaseAVReader):
    """
    Video Reader using PyAV.
    """

    def read_video(
        self,
        av_file: str,
        stream_idx: int = 0,
        audio_sample_rate: int = -1,
        custom_frame_transforms: Optional[BaseTransformation] = None,
        video_only: bool = False,
        *args,
        **kwargs,
    ) -> Dict:
        with av.open(av_file) as container:
            audio_frames = video_frames = None
            audio_fps = video_fps = None
            for stream in container.streams:
                if self.fast_decoding:
                    # use multi-threading for decoding
                    stream.thread_type = "AUTO"

                container.seek(0)
                if stream.type == "audio":
                    # Skip audio stream if audio not required.
                    if video_only:
                        continue
                    # Compute audio frame stats.
                    assert (
                        audio_fps is None
                    ), f"Multiple audio streams exist in '{av_file}', while only one is expected. (stream_idx={stream_idx})"
                    assert audio_frames is None

                    audio_stream = container.streams.audio[stream_idx]
                    n_audio_channels = len(audio_stream.layout.channels)
                    audio_frames = []
                    if audio_sample_rate > 0:
                        resampler = av.AudioResampler(rate=audio_sample_rate)
                    for frame in container.decode(audio=stream_idx):
                        if audio_sample_rate > 0:
                            frame = resampler.resample(frame)[0]
                        audio_frames.append(
                            frame.to_ndarray().reshape(n_audio_channels, -1)
                        )
                    audio_frames = torch.from_numpy(
                        numpy.concatenate(audio_frames, axis=1)
                    ).transpose(1, 0)

                    audio_fps = (
                        audio_sample_rate
                        if audio_sample_rate > 0
                        else audio_stream.sample_rate
                    )

                elif stream.type == "video":
                    assert video_fps is None
                    assert video_frames is None

                    video_stream = container.streams.video[stream_idx]
                    n_frames = video_stream.frames
                    width = video_stream.width
                    height = video_stream.height
                    video_fps = float(video_stream.base_rate)

                    video_frames = torch.empty(
                        size=(n_frames, 3, height, width), dtype=torch.float
                    )
                    frame_transforms = (
                        self.frame_transforms
                        if custom_frame_transforms is None
                        else custom_frame_transforms
                    )
                    for i, video_frame in enumerate(container.decode(video=stream_idx)):
                        video_frame = video_frame.to_image()
                        video_frame = frame_transforms({"image": video_frame})["image"]
                        video_frames[i] = video_frame
            return {
                "audio": audio_frames,
                "video": video_frames,
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
        with av.open(video_path) as container:
            vmetadata["filename"] = video_path
            video_stream = container.streams.video[0]
            # NOTE: av package may return invalid 0 as the total number of frames.
            vmetadata["total_video_frames"] = video_stream.frames
            if vmetadata["total_video_frames"] == 0:
                logger.warning(f"Invalid frame number 0 for {video_path}.")
            vmetadata["video_fps"] = float(video_stream.base_rate)
            vmetadata["video_duration"] = (
                vmetadata["total_video_frames"] / vmetadata["video_fps"]
            )
        return vmetadata
