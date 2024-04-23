#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys

import pytest
import torch

from corenet.data.transforms.common import Compose
from corenet.data.video_reader.decord_reader import DecordAVReader
from corenet.data.video_reader.ffmpeg_reader import FFMPEGReader
from corenet.data.video_reader.ffmpeg_utils import IS_FFMPEG_INSTALLED
from corenet.data.video_reader.pyav_reader import PyAVReader
from tests.configs import get_config


@pytest.mark.parametrize("reader_class", [PyAVReader, DecordAVReader, FFMPEGReader])
@pytest.mark.parametrize(
    "num_frames_per_clip,clips_per_video,is_training,output_video_fps,output_audio_fps,num_samples_per_clip,frame_stack_format",
    [
        # Training cases.
        # num_frames_per_clip > num_frames_in_video.
        (15, 3, True, -1, -1, 2, "sequence_first"),
        # Perfect training case.
        (16, 3, True, 8, 16000, 2, "sequence_first"),
        # clips_per_video = 1,
        (60, 1, True, 16, -1, 1, "sequence_first"),
        (120, 3, True, 8, 16000, 2, "channel_first"),
        # Evaluation cases.
        # Perfect validation case.
        (16, 3, False, -1, -1, 2, "sequence_first"),
        # clips_per_video = 1,
        (30, 1, False, 8, 16000, 1, "sequence_first"),
        (120, 3, False, 16, -1, 2, "channel_first"),
    ],
)
def test_video_reader_clips(
    reader_class,
    num_frames_per_clip,
    clips_per_video,
    is_training,
    output_video_fps,
    output_audio_fps,
    num_samples_per_clip,
    frame_stack_format,
):
    if reader_class is DecordAVReader and "decord" not in sys.modules:
        pytest.skip("Decord (optional dependncy) is not installed.")
    if reader_class is FFMPEGReader and not IS_FFMPEG_INSTALLED:
        pytest.skip("FFmpeg (optional dependncy) is not installed.")
    # This test makes sure that the dimension of the outputs match the expectations
    if frame_stack_format == "channel_first":
        channel_idx = 1
        time_idx = 2
    else:
        channel_idx = 2
        time_idx = 1
    opts = get_config()
    setattr(opts, "dataset.collate_fn_name_train", "default_collate_fn")
    setattr(opts, "dataset.collate_fn_name_val", "default_collate_fn")
    setattr(opts, "video_reader.frame_stack_format", frame_stack_format)
    testfile = "tests/data/dummy_video.mov"
    silent_testfile = "tests/data/dummy_silent_video.mov"
    reader = reader_class(opts, is_training=is_training)
    silent_reader = reader_class(opts, is_training=is_training)
    video_channels = 3
    video, audio, metadata = reader.read_video_file_into_clips(
        testfile,
        num_frames_per_clip=num_frames_per_clip,
        clips_per_video=clips_per_video,
        is_training=is_training,
        output_video_fps=output_video_fps,
        output_audio_fps=output_audio_fps,
        num_samples_per_clip=num_samples_per_clip,
    ).values()
    (
        silent_video,
        silent_audio,
        silent_metadata,
    ) = reader.read_video_file_into_clips(
        silent_testfile,
        num_frames_per_clip=num_frames_per_clip,
        clips_per_video=clips_per_video,
        is_training=is_training,
        output_video_fps=output_video_fps,
        output_audio_fps=output_audio_fps,
        num_samples_per_clip=num_samples_per_clip,
        video_only=True,
    ).values()
    assert video.shape == silent_video.shape
    assert (
        metadata["video_frame_timestamps"].shape
        == silent_metadata["video_frame_timestamps"].shape
    )
    assert metadata["video_fps"] == silent_metadata["video_fps"]
    assert silent_audio is None
    assert silent_metadata["audio_fps"] is None
    assert video.shape[0] == audio.shape[0]
    assert metadata["video_frame_timestamps"].shape == (
        video.shape[0],
        video.shape[time_idx],
    )
    assert video.ndim == 5 and audio.ndim == 3
    if is_training:
        assert video.shape[0] == clips_per_video * num_samples_per_clip
    else:
        assert video.shape[0] == clips_per_video
    assert video.shape[time_idx] == num_frames_per_clip
    assert video.shape[channel_idx] == video_channels
    assert (
        video.shape[time_idx]
        - audio.shape[1] * metadata["video_fps"] / metadata["audio_fps"]
        < 1
    ), "Audio and video are of different lengths in seconds."

    if output_video_fps > 0:
        assert metadata["video_fps"] == output_video_fps
    if output_audio_fps > 0:
        assert metadata["audio_fps"] == output_audio_fps


@pytest.mark.parametrize("reader_class", [PyAVReader, DecordAVReader, FFMPEGReader])
@pytest.mark.parametrize(
    "num_frames_per_clip,clips_per_video,is_training,output_video_fps,output_audio_fps,num_samples_per_clip,frame_stack_format",
    [
        # Training cases.
        # Perfect training case.
        (8, 3, True, -1, -1, 1, "sequence_first"),
        (16, 3, True, 8, 16000, 2, "channel_first"),
        (120, 3, True, 8, 16000, 2, "channel_first"),
        # Evaluation cases.
        # Perfect case.
        (8, 3, False, -1, -1, 1, "sequence_first"),
        (16, 3, False, 8, 16000, 2, "channel_first"),
        (120, 3, False, 8, 16000, 2, "channel_first"),
    ],
)
def test_video_reader_values(
    reader_class,
    num_frames_per_clip,
    clips_per_video,
    is_training,
    output_video_fps,
    output_audio_fps,
    num_samples_per_clip,
    frame_stack_format,
):
    if reader_class is DecordAVReader and "decord" not in sys.modules:
        pytest.skip("Decord (optional dependncy) is not installed.")
    if reader_class is FFMPEGReader and not IS_FFMPEG_INSTALLED:
        pytest.skip("FFmpeg (optional dependncy) is not installed.")
    reader_name = "pyav" if reader_class is PyAVReader else "decord"
    # This test makes sure that the values in the clips for perfect videos are different
    opts = get_config()
    setattr(opts, "dataset.collate_fn_name_train", "default_collate_fn")
    setattr(opts, "dataset.collate_fn_name_val", "default_collate_fn")
    setattr(opts, "video_reader.name", reader_name)
    setattr(opts, "video_reader.frame_stack_format", frame_stack_format)
    testfile = "tests/data/dummy_video.mov"
    reader = reader_class(opts, is_training=is_training)
    video, audio, metadata = reader.read_video_file_into_clips(
        testfile,
        num_frames_per_clip=num_frames_per_clip,
        clips_per_video=clips_per_video,
        is_training=is_training,
        output_video_fps=output_video_fps,
        output_audio_fps=output_audio_fps,
        num_samples_per_clip=num_samples_per_clip,
    ).values()

    # Dummy video has 114 frames.
    total_frames = 114
    if num_frames_per_clip > total_frames:
        assert torch.allclose(
            video[:-1], video[1:]
        ), "Not all video clips are the same."
        assert torch.allclose(
            audio[:-1], audio[1:]
        ), "Not all audio clips are the same."
    else:
        assert not torch.allclose(
            video[:-1], video[1:]
        ), "All video clips are the same."
        assert not torch.allclose(
            audio[:-1], audio[1:]
        ), "All audio clips are the same."


@pytest.mark.parametrize("reader_class", [PyAVReader, DecordAVReader, FFMPEGReader])
def test_build_video_metadata(reader_class):
    if reader_class is DecordAVReader and "decord" not in sys.modules:
        pytest.skip("Decord (optional dependncy) is not installed.")
    if reader_class is FFMPEGReader and not IS_FFMPEG_INSTALLED:
        pytest.skip("FFmpeg (optional dependncy) is not installed.")

    # This test makes sure that the values in the clips for perfect videos are different
    opts = get_config()

    reader = reader_class(opts)

    testfile = "tests/data/dummy_video.mov"
    expected_metadata = {
        "filename": testfile,
        "video_fps": 30,
        "total_video_frames": 114,
        "video_duration": 3.8,
    }
    metadata = reader.build_video_metadata(testfile)
    assert metadata["filename"] == expected_metadata["filename"]
    assert round(metadata["video_fps"], 0) == expected_metadata["video_fps"]
    assert metadata["total_video_frames"] == expected_metadata["total_video_frames"]
    assert round(metadata["video_duration"], 1) == expected_metadata["video_duration"]
