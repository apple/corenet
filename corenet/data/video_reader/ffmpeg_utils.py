#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import functools
import logging
import math
import subprocess
import time
import warnings
from io import IOBase
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import ffmpeg

    subprocess.check_call(
        ["ffmpeg", "-h"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    IS_FFMPEG_INSTALLED = True
except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
    IS_FFMPEG_INSTALLED = False

from corenet.data.transforms.base_transforms import BaseTransformation
from corenet.data.transforms.video import ToPixelArray, ToTensor
from corenet.utils import logger
from corenet.utils.context_managers import context_tensor_threads
from corenet.utils.import_utils import ensure_library_is_available
from corenet.utils.io_utils import make_temp_file


class FFMPEGError(Exception):
    def __init__(self, message: str, stderr: Union[IOBase, bytes]):
        if isinstance(stderr, IOBase):
            stderr = stderr.read()
        super().__init__(f"{message}:\n{stderr.decode(errors='replace')}")


def _parse_stream_duration(stream: Dict[str, Any]) -> float:
    """Extract duration (seconds) from audio/video stream headers.

    Args:
        stream: FFProbe headers (i.e. metadata) for an audio/video stream.

    Returns:
        The parsed duration of the input stream metadata, in seconds.
    """
    if "duration" in stream:
        duration = float(stream["duration"])
    else:
        duration = stream["tags"]["DURATION"]
        hh, mm, ss = map(float, duration.split(":"))
        duration = hh * 3600 + mm * 60 + ss

    return duration


@functools.lru_cache(maxsize=1)
def _ffprobe_cached(video_path: str) -> Dict[str, Any]:
    ensure_library_is_available(module_name="ffmpeg")
    try:
        return ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        raise FFMPEGError(
            f"Could not extract ffmpeg metadata for {video_path}", e.stderr
        )


def get_video_metadata(
    video_path: str, return_extras: bool = False
) -> Union[Dict[str, Union[str, float, int]], Tuple[Dict, Dict]]:
    """Generate the metadata for a given video.

    Args:
        video_path: A video file path.
        return_extras: When True, returns additional information as described in
            "Returns" section. Defaults to False.

    Returns:
        When @return_extras is False (default), returns the following metadata:
        metadata = {
            "filename": <str>,
            "video_fps": <float>,
            "total_video_frames" <int>,
            "video_duration": <float>,
            "width": <int>,
            "height": <int>,
            "start_time": <float>,  # When greater than 0, the frame that ffmpeg reads
                                    # should be presented at timestamp <start_time>.
        }

        When @return_extras is True, returns `(metadata, extras)` tuple, where:
        extras = {
            "raw": <dict>,  # The raw json output of ffprobe.
            "rotation": <int>,  # Rotation of raw frames, in {0, 90, 180, 270}.
        }
    """
    ensure_library_is_available(module_name="ffmpeg")

    probe = _ffprobe_cached(video_path)
    metadata = None
    for stream in probe["streams"]:
        if stream.get("codec_type") != "video":
            continue
        if stream.get("disposition", {}).get("attached_pic", 0) != 0:
            # In some videos, the video thumbnail image is stored as a stream with
            # codec_type=video and disposition.attached_pic tag. We shall skip
            # thumbnail image streams.
            continue
        if metadata is not None:
            raise ValueError(f"Found more than 1 video stream in {video_path}.")
        num, denom = map(int, stream["avg_frame_rate"].split("/"))
        frame_rate = num / denom
        duration = _parse_stream_duration(stream)

        # In older ffmpeg/ffprobe versions, rotation metadata was returned as
        # stream["tags"]["rotate"]. In the newer versions, rotation is returned as
        # stream["side_data_list"][<index>]["rotation"].
        # See: https://superuser.com/a/1724964
        rotation_metadata_values = []
        if "tags" in stream and "rotate" in stream["tags"]:
            rotation_metadata_values.append(int(stream["tags"]["rotate"]))
        for side_data in stream.get("side_data_list", []):
            if "rotation" in side_data:
                rotation_metadata_values.append(int(side_data["rotation"]))

        if rotation_metadata_values:
            assert (
                len(set(rotation_metadata_values)) == 1
            ), f"Got inconsistent rotation metadata {rotation_metadata_values}."
            rotation = rotation_metadata_values[0]
            assert (
                rotation % 90 == 0
            ), f"Got unexpected rotation value {rotation} in {video_path}."
            rotation = (rotation + 360) % 360  # Ensure positive value.
        else:
            rotation = 0

        metadata = {
            "filename": video_path,
            "video_fps": frame_rate,
            "total_video_frames": int(math.ceil(duration * frame_rate)),
            "video_duration": duration,
            "width": stream["width"] if rotation % 180 == 0 else stream["height"],
            "height": stream["height"] if rotation % 180 == 0 else stream["width"],
            "start_time": float(stream.get("start_time", "0.0")),
        }
        if return_extras:
            extras = {
                "raw": stream,
                "rotation": rotation,
            }
    if metadata is None:
        raise ValueError(f"Could not find a video stream in {video_path}.")

    if return_extras:
        return metadata, extras
    else:
        return metadata


def get_audio_metadata(video_path: str) -> Optional[Dict[str, Union[str, float, int]]]:
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
    probe = _ffprobe_cached(video_path)
    result = None
    for stream in probe["streams"]:
        if stream.get("codec_type") != "audio":
            continue
        if result is not None:
            raise ValueError(f"Found more than 1 audio stream in {video_path}.")
        sample_rate = int(stream["sample_rate"])
        duration = _parse_stream_duration(stream)

        result = {
            "filename": video_path,
            "audio_fps": sample_rate,
            "total_audio_frames": int(duration * sample_rate),
            "audio_duration": duration,
            "audio_channels": int(stream["channels"]),
        }
    return result


def get_flags_to_replicate_audio_codec(input_filename: str) -> Dict:
    """Extracts ffmpeg flags to replicate input audio codec properties in the output.

    Args:
        input_filename: Path of the input audio/video file.

    Returns:
        A dictionary containing ffmpeg flags to be consumed by ffmpeg-python package.
    Each {key: value} entry of this dictionary will be converted to "-key value"
    command-line flags for ffmpeg command.

    Note:
        The goal is to replicate a close behavior to "-c:a copy" while re-encoding the
    audio ("-c:a copy" does not re-encode the audio). Re-encoding the audio is 
    necessary when we trim the video during processing, because "-c:a copy" produces
    wrong durations. 
    
    For example, you can use the following command to extract a 1.5 seconds audio clip,
    but you will get a 2.04 second result. But replacing "-c:a copy" with "-c:a aac"
    corrects the duration.

    ffmpeg -ss 0.5 -i ./tests/data/dummy_video.mov -c:a copy -to 1.5 /tmp/a.mov -y \
        2>/dev/null && ffprobe /tmp/a.mov 2>&1 | grep Duration

    As explained in https://trac.ffmpeg.org/ticket/977, one alternative is to move -ss
    flag to after -i, that solves the duration problem but produces out of sync audio:
    https://superuser.com/questions/1001299/ffmpeg-video-cutting-error-out-of-sync-audio.
    """
    probe = _ffprobe_cached(input_filename)
    audio_stream = None
    for stream in probe["streams"]:
        if stream.get("codec_type") != "audio":
            continue
        if audio_stream is not None:
            raise ValueError(f"Found more than 1 audio stream in {input_filename}.")
        audio_stream = stream
    if audio_stream is None:
        raise ValueError(f"Could not find audio stream in {input_filename}.")

    codec_name = audio_stream["codec_name"]
    result = {
        "acodec": codec_name,
        "ac": audio_stream["channels"],
    }
    if codec_name == "aac":
        # AAC is a common mp4 audio encoding: https://trac.ffmpeg.org/wiki/Encode/AAC
        result["b:a"] = audio_stream["bit_rate"]
    elif codec_name.startswith("pcm_"):
        # pcm_s16le is the codec name for raw audio (stereo,16bit,little-endian)
        result["ar"] = audio_stream["sample_rate"]
    else:
        logging.error(
            "The current implementation does not know how to replicate audio codec"
            f" '{codec_name}'."
        )
    return result


def transform_video_file(
    input_filename: str,
    output_filename: str,
    transform: BaseTransformation,
    output_dimensions: Tuple[int, int] = None,
    threads: int = 1,
    max_pixels_per_batch: int = 1,
    from_timestamp: Optional[float] = None,
    to_timestamp: Optional[float] = None,
    ffmpeg_loglevel: str = "error",
    logging_interval: int = 5 * 60,
    encoder_acodec: str = "copy",
    encoder_vcodec: str = "mjpeg",
    encoder_flags: Optional[Dict] = None,
) -> None:
    """
    Reads batches of video frames using ffmpeg, transforms them, and writes results
    back to the output file. This utility is specially useful for transforming long
    videos that cannot be loaded into RAM as a whole.

    Args:
        input_filename: Path to the input video file.
        output_filename: Path to the output video file.
        transform (BaseTransformation): A transformation that is compatible with the
            video transformations in ``corenet.data.transforms.video`` module. The
            transformation should expect a dict with the following schema: {
                "samples": {
                    "video": Tensor[NxTxCxHxW],
                    "metadata": {
                        "video_fps": float,
                        "video_frame_timestamps": Tensor[NxT],
                    },
                },
                "targets": {},
            }
        output_dimensions: Expected frame size (height, weight) of the outputs of the
            transformation.
        threads: Number of threads to use. Note that some ffmpeg components may ignore
            the threads option.
        max_pixels_per_batch: Determines the batch size for reading and processing
            frames. If you want to process 10 frames at a time, pass 10*C*H*W. Passing
            number of pixels, rather than number of frames, allows us to dynamically
            adapt the batch size given input video frame dimensions.
        from_timestamp: If provided (in seconds), only the frames after this timestamp
            will be written to the output file. Defaults to 0, that skips frames with
            negative timestamp.
        to_timestamp: If provided (in seconds), only the frames before this timestamp
            will be written to the output file. Defaults to None, when the video
            duration (from metadata) is used as to_timestamp.
        ffmpeg_loglevel: Controls the log level of ffmpeg library. NOTE: Values other
            than "error" may cause too much logs. In our experience, too much logs can
            lead to buffer overflow that cause hanging processes. Choices:
            "quiet", "panic", "fatal"=, "error", "warning", "info", "verbose", "debug",
            "trace". Defaults to "error". See: https://ffmpeg.org/ffmpeg.html
        logging_interval: Controls the interval of logging the progress. The unit is
            seconds. Defaults to 5 minutes.
        encoder_acodec: FFMpeg audio codec for encoding the output file. Defaults to
            "copy", that uses the same encoding as the input video. When @from_timestamp
            or @to_timestamp arguments are provided, the audio-codec should be specified
            (other than "copy") because we must re-encode audio to avoid lagged frames.
        encoder_vcodec: FFMpeg video codec for encoding the input file. Defaults to
            "mjpeg", that simply encodes individual frames as jpeg. As video frames are
            processed in python and passed to the ffmpeg as input, passing
            encoder_vcodec="copy" will not copy the encoding from the input file.
        encoder_flags: Extra ffmpeg flags to be used for output audio/video encoder.
            For example, pass {"b:a": "192k"} to set mjpeg audio bitrate to 192k. This
            argument overrides any other encoder_* argument Defaults to {}. Each
            {key:value} gets translated to `-key value` ffmpeg CLI flags.
    """
    ensure_library_is_available(module_name="ffmpeg")

    if encoder_flags is None:
        encoder_flags = {}

    metadata = get_video_metadata(input_filename)
    height_in = metadata["height"]
    width_in = metadata["width"]
    if output_dimensions is None:
        height_out, width_out = height_in, width_in
    else:
        height_out, width_out = output_dimensions
    video_fps = metadata["video_fps"]

    is_trimmed = from_timestamp is not None or to_timestamp is not None
    if is_trimmed:
        assert encoder_flags.get("acodec", encoder_acodec) != "copy", (
            "Audio codec 'copy' is not valid when from_timestamp/to_timestamp"
            " arguments are provided because audio must be re-encoded."
        )
        assert encoder_flags.get("vcodec", encoder_vcodec) != "copy", (
            "Video codec 'copy' is not valid when from_timestamp/to_timestamp"
            " arguments are provided because audio must be re-encoded."
        )
        if from_timestamp is None:
            from_timestamp = 0.0
        else:
            assert from_timestamp >= 0.0
        if to_timestamp is None:
            to_timestamp = metadata["video_duration"]

        trim_flags = {
            "ss": from_timestamp,
            "to": to_timestamp,
        }
    else:
        trim_flags = {}

    tmp_audio_filename = None
    video_decode_process = encoder_process = None
    try:
        video_decode_process_command: List[str] = (
            ffmpeg.input(input_filename)
            .video.output("pipe:", format="rawvideo", pix_fmt="rgb24", **trim_flags)
            .global_args(
                "-threads", str(max(1, threads // 2)), "-loglevel", ffmpeg_loglevel
            )
            .compile()
        )
        video_decode_process = subprocess.Popen(
            video_decode_process_command,
            # See https://github.com/kkroening/ffmpeg-python/issues/782
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        processed_video = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width_out, height_out),
            r=video_fps,
            loglevel=ffmpeg_loglevel,
        ).video

        if is_trimmed:
            # Copy trimmed audio with re-encoding to tmp_audio_filename
            tmp_audio_filename = make_temp_file(suffix=Path(input_filename).suffix)
            extract_audio_command = (
                ffmpeg.output(
                    ffmpeg.input(input_filename)
                    .audio.filter_("atrim", start=from_timestamp, end=to_timestamp)
                    .filter_("asetpts", "PTS-STARTPTS"),
                    filename=tmp_audio_filename,
                    acodec="pcm_s16le",  # Store raw audio in the intermediate file.
                    **{"q:a": 0},  # Use highest possible quality.
                )
                .overwrite_output()
                .global_args("-threads", str(threads), "-loglevel", ffmpeg_loglevel)
                .compile()
            )
            extract_audio_process = subprocess.Popen(
                extract_audio_command,
                # See https://github.com/kkroening/ffmpeg-python/issues/782
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = extract_audio_process.communicate()
            if extract_audio_process.poll():
                raise FFMPEGError("Failed to re-encode the audio.", stderr)

            processed_audio = ffmpeg.input(tmp_audio_filename).audio
        else:
            processed_audio = ffmpeg.input(input_filename).audio

        encoder_flags.setdefault("acodec", encoder_acodec)
        encoder_flags.setdefault("vcodec", encoder_vcodec)

        encoder_process_command = (
            ffmpeg.output(
                processed_video,
                processed_audio,
                output_filename,
                **encoder_flags,
            )
            .global_args(
                "-threads", str(max(1, threads // 2)), "-loglevel", ffmpeg_loglevel
            )
            .overwrite_output()
            .compile()
        )
        encoder_process = subprocess.Popen(
            encoder_process_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        latest_log = start_time = time.time()
        processed_frames = 0
        current_timestamp = 0.0

        @functools.cache
        def get_total_frames():
            if is_trimmed:
                return int((to_timestamp - from_timestamp) * video_fps)
            else:
                return get_video_metadata(input_filename)["total_video_frames"]

        opts = argparse.Namespace()
        to_tensor = ToTensor(opts)
        to_pixel_array = ToPixelArray(opts)
        with context_tensor_threads(threads):
            while True:
                max_frames = int(
                    max(1, max_pixels_per_batch / width_in / height_in / 3)
                )

                if time.time() > latest_log + logging_interval:
                    logger.log(
                        "Processing frames"
                        f" {processed_frames}:{processed_frames+max_frames} /"
                        f" {get_total_frames()} [{input_filename}]"
                        f" [Velocity={processed_frames/(time.time()-start_time)*60:.1f}"
                        " frames/minute]"
                    )
                    latest_log = time.time()

                frames = video_decode_process.stdout.read(
                    max_frames * width_in * height_in * 3
                )

                if not frames:
                    break
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    frames = torch.frombuffer(frames, dtype=torch.uint8)
                frames = frames.reshape(1, -1, height_in, width_in, 3)
                num_frames = frames.shape[1]

                video_frame_timestamps = (
                    torch.arange(num_frames) / video_fps + current_timestamp
                )
                video_frame_timestamps = video_frame_timestamps[None, :]

                frames = frames.permute(0, 1, 4, 2, 3)  # NTHWC -> NTCHW
                data = {
                    "samples": {
                        "video": frames,
                        "metadata": {
                            "video_fps": video_fps,
                            "video_frame_timestamps": video_frame_timestamps,
                            "filename": input_filename,
                        },
                    },
                    "targets": {},
                }
                data = to_tensor(data)
                data = transform(data)
                data = to_pixel_array(data)

                frames = data["samples"]["video"]
                frames = frames.transpose(0, 1, 3, 4, 2)  # NTCHW -> NTHWC

                expected_shape = (1, num_frames, height_out, width_out, 3)
                assert (
                    frames.shape == expected_shape
                ), f"Got shape {frames.shape}, expected {expected_shape}."
                assert (
                    frames.dtype == np.uint8
                ), f"Unexpected dtype {frames.dtype}, expected uint8."
                processed_frames += num_frames
                current_timestamp += num_frames / video_fps

                encoder_process.stdin.write(frames.data.tobytes())
                encoder_process.stdin.flush()

        if output_filename:
            encoder_process.stdin.close()
        video_decode_process.wait()
        if output_filename:
            encoder_process.wait()
    except BrokenPipeError:
        raise FFMPEGError(
            f"Encoder process failed with command='{' '.join(encoder_process.args)}'",
            encoder_process.stderr,
        )
    finally:
        if tmp_audio_filename is not None:
            Path(tmp_audio_filename).unlink(missing_ok=True)
        if video_decode_process is not None:
            video_decode_process.kill()
        if encoder_process is not None:
            encoder_process.kill()
