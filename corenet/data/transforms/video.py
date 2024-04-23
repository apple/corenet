#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import datetime
import hashlib
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import psutil
import torch
import torchaudio
from torch.nn import functional as F
from torchvision.io import write_video
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as FV

from corenet.data.transforms import TRANSFORMATIONS_REGISTRY, BaseTransformation
from corenet.data.transforms.utils import *
from corenet.options.parse_args import JsonValidator
from corenet.utils import logger

SUPPORTED_PYTORCH_INTERPOLATIONS = ["nearest", "bilinear", "bicubic"]


def _check_interpolation(interpolation):
    if interpolation not in SUPPORTED_PYTORCH_INTERPOLATIONS:
        inter_str = "Supported interpolation modes are:"
        for i, j in enumerate(SUPPORTED_PYTORCH_INTERPOLATIONS):
            inter_str += "\n\t{}: {}".format(i, j)
        logger.error(inter_str)
    return interpolation


def _crop_fn(data: Dict, i: int, j: int, h: int, w: int) -> Dict:
    """Crop the video in `data`.

    Args:
        data: A dictionary of data. The format is:
            {
                "samples":{
                    "video": A video tensor of shape [...x H x W], where H and W are the
                        height and width.
                    "audio": An audio tensor.
                }
            }
        i: The height coordinate of the top left corner of the cropped rectangle.
        j: The width coordinate of the top left corner of the cropped rectangle.
        h: The height of the cropped rectangle.
        w: The width of the cropped rectangle.

    Returns:
        A dictionary of the same format as `data` where `data["samples"]["videos"]` is
        the cropped video.
    """
    img = data["samples"]["video"]
    check_rgb_video_tensor(img)

    crop_image = img[..., i : i + h, j : j + w]
    data["samples"]["video"] = crop_image

    mask = data.get("mask", None)
    if mask is not None:
        crop_mask = mask[..., i : i + h, j : j + w]
        data["samples"]["mask"] = crop_mask
    return data


def _resize_fn(
    data: Dict,
    size: Union[Sequence, int],
    interpolation: Optional[str] = "bilinear",
) -> Dict:
    """Resize the video in `data`.

    Args:
        data: A dictionary of data. The format is:
            {
                "samples":{
                    "video": A video tensor of shape [... x H x W], where H and W are the
                        height and width.
                    "mask": An optional entry of the mask tensor of shape [... x H x W],
                        where H and W are the height and width.
                    "audio": An audio tensor.
                }
            }
        size: The size of video to resize to.
        interpolation: The method of interpolation to use. Choices are: "bilinear",
        "nearest", "linear", "bicubic", "trilinear", "area", "nearest-exact", default to
        "bilinear".

    Returns:
        A dictionary of the same format as `data` where `data["samples"]["videos"]` is
        the cropped video.
    """

    video = data["samples"]["video"]

    if isinstance(size, Sequence) and len(size) == 2:
        size_h, size_w = size[0], size[1]
    elif isinstance(size, int):
        h, w = video.shape[-2:]
        if (w <= h and w == size) or (h <= w and h == size):
            return data

        if w < h:
            size_h = int(size * h / w)

            size_w = size
        else:
            size_w = int(size * w / h)
            size_h = size
    else:
        raise TypeError(
            "Supported size args are int or tuple of length 2. Got inappropriate size"
            " arg: {}".format(size)
        )
    if isinstance(interpolation, str):
        interpolation = _check_interpolation(interpolation)
    n, tc1, tc2, h, w = video.shape
    # Since video could be either NTCHW or NCTHW format, we reshape the 5D tensor into
    # 4D and transpose back to 5D.
    video = F.interpolate(
        input=video.reshape(n, tc1 * tc2, h, w),
        size=(size_h, size_w),
        mode=interpolation,
        align_corners=True if interpolation != "nearest" else None,
    )
    data["samples"]["video"] = video.reshape(n, tc1, tc2, size_h, size_w)

    mask = data["samples"].get("mask", None)
    if mask is not None:
        mask = F.interpolate(input=mask, size=(size_h, size_w), mode="nearest")
        data["samples"]["mask"] = mask

    return data


def check_rgb_video_tensor(clip: torch.Tensor) -> None:
    """Check if the video tensor is the right type and shape.

    Args:
        clip: A video clip tensor of shape [N x C x T x H x W] or
        [N x C x T x H x W], where N is the number of clips, T is the number
        of frames of the clip, C is the number of image channels,
        H and W are the height and width of the frame image.
    """
    if not isinstance(clip, torch.FloatTensor):
        logger.error("Video clip is not an instance of FloatTensor.")
    if clip.dim() != 5:
        logger.error("Video clip is not a 5-d tensor (NTCHW or NCTHW).")


@TRANSFORMATIONS_REGISTRY.register(name="to_tensor", type="video")
class ToTensor(BaseTransformation):
    """
    This method converts an image into a tensor.

    Tensor shape abbreviations:
        N: Number of clips.
        T, T_audio, T_video: Temporal lengths.
        C: Number of color channels.
        H, W: Height, Width.

    .. note::
        We do not perform any mean-std normalization. If mean-std normalization is
        desired, please modify this class.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts)

    def __call__(self, data: Dict) -> Dict:
        # [N, C, T, H, W] or [N, T, C, H, W].
        clip = data["samples"]["video"]
        if not isinstance(clip, torch.Tensor):
            clip = torch.from_numpy(clip)

        if not isinstance(clip, torch.FloatTensor):
            # Convert to float, and normalize between 0 and 1.
            clip = clip / 255.0

        check_rgb_video_tensor(clip)

        data["samples"]["video"] = clip
        return data


@TRANSFORMATIONS_REGISTRY.register(name="to_array", type="video")
class ToPixelArray(BaseTransformation):
    """
    This method is an inverse of ToTensor, converting a float tensor in range [0,1] back
    to a numpy uint8 array in range [0,255].

    Tensor shape abbreviations:
        N: Number of clips.
        T: Temporal length.
        C: Number of color channels.
        H, W: Height, Width.
    """

    def __call__(self, data: Dict) -> Dict:
        # [N, C, T, H, W] or [N, T, C, H, W].
        video = data["samples"]["video"]
        video = (video * 255.0).round().numpy().astype(np.uint8)
        data["samples"]["video"] = video
        return data


@TRANSFORMATIONS_REGISTRY.register(name="save-inputs", type="video")
class SaveInputs(BaseTransformation):
    def __init__(
        self,
        opts: argparse.Namespace,
        get_frame_captions: Optional[Callable[[Dict], List[str]]] = None,
        *args,
        **kwargs,
    ) -> None:
        """Saves the clips that are returned by VideoDataset.__getitem__() to disk
        for debugging use cases. This transformation operates on multiple clips that
        are extracted out of a single raw video. The video and audio of the clips are
        concatenated and saved into 1 video file.

        1 raw input video ==> VideoDataset.__getitem__() ==>
            multiple clips in data["samples"]["video"] ==> SaveInputs() ==>
            1 output debugging video.

        This is useful for visualizing training and/or validation videos to make
        sure preprocessing logic is behaving as expected.

        Args:
            opts: Command line options.
            get_frame_captions: If provided, this function returns a list of strings
                (one string per video frame). The frame captions will be added to the
                video as subtitles.
        """
        self.get_frame_captions = get_frame_captions
        self.enable = getattr(opts, "video_augmentation.save_inputs.enable")
        save_dir = getattr(opts, "video_augmentation.save_inputs.save_dir")
        if self.enable and save_dir is None:
            logger.error(
                "Please provide value for --video_augmentation.save-inputs.save-dir"
            )
        process_start_time = datetime.datetime.fromtimestamp(
            psutil.Process(os.getpid()).create_time()
        ).strftime("%Y-%m-%d %H:%M")
        self.save_dir = Path(save_dir, process_start_time).expanduser()
        self.symlink_to_original = getattr(
            opts, "video_augmentation.save_inputs.symlink_to_original"
        )

    def __call__(self, data: Dict) -> Dict:
        if not self.enable:
            return data
        original_path = data["samples"]["metadata"]["filename"]
        original_basename = os.path.basename(original_path)
        original_path_hash = hashlib.md5(str(original_path).encode()).hexdigest()[:5]
        output_video_path = Path(
            self.save_dir,
            f"{datetime.datetime.now().isoformat()[:5]}_{original_path_hash}_{original_basename}",
        )

        self.save_video_with_annotations(
            data=data,
            output_video_path=output_video_path,
        )
        if self.symlink_to_original:
            os.symlink(
                original_path,
                output_video_path.with_suffix(f".original.{output_video_path.suffix}"),
            )
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--video-augmentation.save-inputs.save-dir",
            type=str,
            default=None,
            help=(
                "Path to the folder for saving output debugging videos. Defaults to"
                " None."
            ),
        )
        group.add_argument(
            "--video-augmentation.save-inputs.add-labels",
            action="store_true",
            default=False,
            help=(
                "If set, write the class label on each frame of the video. Defaults to"
                " False."
            ),
        )
        group.add_argument(
            "--video-augmentation.save-inputs.enable",
            action="store_true",
            default=False,
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Defaults to False.".format(cls.__name__)
            ),
        )
        group.add_argument(
            "--video-augmentation.save-inputs.symlink-to-original",
            action="store_true",
            default=False,
            help=(
                "If True, a symlink to original video sample will be created besides"
                "the saved inputs for easier debugging. Defaults to False."
            ),
        )
        return parser

    @staticmethod
    def _srt_format_timestamp(t: float) -> str:
        t = int(t * 1000)
        t, millis = divmod(t, 1000)
        t, ss = divmod(t, 60)
        t, mm = divmod(t, 60)
        hh = t
        return f"{0 if hh<10 else ''}{hh}:{0 if mm<10 else ''}{mm}:{ss},{millis:0>3}"

    def save_video_with_annotations(
        self,
        data: Dict,
        output_video_path: Path,
    ) -> None:
        """Save a video with audio and captions.

        Args:
            data: Dataset output dict. Schema: {
                "samples": {
                    "video": Tensor[N x T X C x H x W],
                    "audio": Tensor[N x T_audio x C],  # Optional
                    "audio_raw": Tensor[N x T_audio x C],  # Optional - if provided,
                                                           # "audio" will be ignored.
                    "metadata": {
                        "video_fps": Union[float,int],
                        "audio_fps": Union[float,int],
                    }
                }
            }
            output_video_path: Path for saving the video.
            get_frame_captions: A callback that receives @data as input and returns a
               list of captions (one string per video frame). If provided, the captions
               will be added to the output video as subtitles.
        """
        video = data["samples"]["video"]  # N x T x C x H x W
        video = video.reshape(-1, *video.shape[2:])  # (N*T) x C x H x W
        video_fps = data["samples"]["metadata"]["video_fps"]

        if "audio_raw" in data:
            audio = data["samples"]["audio_raw"]  # N x T_audio x C
        else:
            audio = data["samples"].get("audio")  # N x T_audio x C

        if audio is not None:
            audio = audio.reshape(-1, *audio.shape[2:])  # N*T_audio x C
            audio_fps = int(round(data["samples"]["metadata"]["audio_fps"]))

        video = (video * 255).round().to(dtype=torch.uint8).cpu()
        video = video.permute([0, 2, 3, 1])  # N x H x W x C

        suffix = output_video_path.suffix
        assert suffix in (
            ".mp4",
            ".mov",
            ".mkv",
        ), f"{suffix} format is not supported by SaveInputs yet."
        output_video_path.parent.mkdir(exist_ok=True, parents=True)
        if audio is not None or self.get_frame_captions is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_video = Path(tmp_dir, "video" + suffix)
                write_video(str(tmp_video), video_array=video, fps=video_fps)
                command = ["ffmpeg", "-i", tmp_video]

                if audio is not None:
                    tmp_audio = str(Path(tmp_dir, "audio.wav"))
                    torchaudio.save(tmp_audio, audio.transpose(0, 1), audio_fps)
                    command.extend(["-i", tmp_audio])

                command.extend(["-c:v", "libx264"])
                if audio is not None:
                    command.extend(["-c:a", "aac"])

                if self.get_frame_captions:
                    captions = self.get_frame_captions(data)
                    tmp_srt = str(Path(tmp_dir, "subtitle.srt"))
                    with open(tmp_srt, "wt") as srt:
                        for i, caption in enumerate(captions):
                            srt.write(
                                f"{i+1}\n"
                                f"{self._srt_format_timestamp(i / video_fps)} --> "
                                f"{self._srt_format_timestamp((i+1) / video_fps)}\n"
                                f"{caption}\n\n"
                            )
                    command.extend(
                        [
                            "-vf",
                            f"subtitles={tmp_srt}:force_style='Alignment=6,Fontsize=48,Outline=8'",
                        ]
                    )

                subprocess.check_output(
                    [*command, f"file:{output_video_path}"],
                    stderr=subprocess.PIPE,
                )
        else:
            write_video(str(output_video_path), video_array=video, fps=video_fps)

    def __repr__(self) -> str:
        return (
            "{}(save_dir={}, add_labels={}, symlink_to_original={}, enable={})".format(
                self.__class__.__name__,
                self.save_dir,
                self.add_labels,
                self.symlink_to_original,
                self.enable,
            )
        )


@TRANSFORMATIONS_REGISTRY.register(name="random_resized_crop", type="video")
class RandomResizedCrop(BaseTransformation):
    """
    This class crops a random portion of an image and resize it to a given size.
    """

    def __init__(self, opts, size: Union[Tuple, int], *args, **kwargs) -> None:
        interpolation = getattr(
            opts,
            "video_augmentation.random_resized_crop.interpolation",
        )
        scale = getattr(opts, "video_augmentation.random_resized_crop.scale")
        ratio = getattr(
            opts,
            "video_augmentation.random_resized_crop.aspect_ratio",
        )

        if not isinstance(scale, Sequence) or (
            isinstance(scale, Sequence)
            and len(scale) != 2
            and 0.0 <= scale[0] < scale[1]
        ):
            logger.error(
                "--video-augmentation.random-resized-crop.scale should be a tuple of"
                f" length 2 such that 0.0 <= scale[0] < scale[1]. Got: {scale}."
            )

        if not isinstance(ratio, Sequence) or (
            isinstance(ratio, Sequence)
            and len(ratio) != 2
            and 0.0 < ratio[0] < ratio[1]
        ):
            logger.error(
                "--video-augmentation.random-resized-crop.aspect-ratio should be a"
                f" tuple of length 2 such that 0.0 < ratio[0] < ratio[1]. Got: {ratio}."
            )

        ratio = (round(ratio[0], 3), round(ratio[1], 3))

        super().__init__(opts=opts)

        self.scale = scale
        self.size = setup_size(size=size)

        self.interpolation = _check_interpolation(interpolation)
        self.ratio = ratio
        self.enable = getattr(opts, "video_augmentation.random_resized_crop.enable")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--video-augmentation.random-resized-crop.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Defaults to False.".format(cls.__name__)
            ),
        )
        group.add_argument(
            "--video-augmentation.random-resized-crop.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Desired interpolation method. Defaults to bilinear",
        )
        group.add_argument(
            "--video-augmentation.random-resized-crop.scale",
            type=JsonValidator(Tuple[float, float]),
            default=(0.08, 1.0),
            help=(
                "Specifies the lower and upper bounds for the random area of the crop,"
                " before resizing. The scale is defined with respect to the area of the"
                " original image. Defaults to (0.08, 1.0)."
            ),
        )
        group.add_argument(
            "--video-augmentation.random-resized-crop.aspect-ratio",
            type=JsonValidator(Union[float, tuple]),
            default=(3.0 / 4.0, 4.0 / 3.0),
            help=(
                "lower and upper bounds for the random aspect ratio of the crop,"
                " before resizing. Defaults to (3./4., 4./3.)."
            ),
        )
        return parser

    def get_params(self, height: int, width: int) -> (int, int, int, int):
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop.
        in_ratio = (1.0 * width) / height
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, data: Dict) -> Dict:
        clip = data["samples"]["video"]
        check_rgb_video_tensor(clip=clip)

        height, width = clip.shape[-2:]

        i, j, h, w = self.get_params(height=height, width=width)
        data = _crop_fn(data=data, i=i, j=j, h=h, w=w)
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return "{}(scale={}, ratio={}, interpolation={}, enable={})".format(
            self.__class__.__name__,
            self.scale,
            self.ratio,
            self.interpolation,
            self.enable,
        )


@TRANSFORMATIONS_REGISTRY.register(name="random_short_side_resize_crop", type="video")
class RandomShortSizeResizeCrop(BaseTransformation):
    """
    This class first randomly resizes the input video such that shortest side is between
    specified minimum and maximum values, adn then crops a desired size video.

    .. note::
        This class assumes that the video size after resizing is greater than or equal
        to the desired size.
    """

    def __init__(self, opts, size: Union[Tuple, int], *args, **kwargs) -> None:
        interpolation = getattr(
            opts,
            "video_augmentation.random_short_side_resize_crop.interpolation",
        )
        short_size_min = getattr(
            opts,
            "video_augmentation.random_short_side_resize_crop.short_side_min",
        )
        short_size_max = getattr(
            opts,
            "video_augmentation.random_short_side_resize_crop.short_side_max",
        )

        if short_size_min is None:
            logger.error(
                "Short side minimum value can't be None in {}".format(
                    self.__class__.__name__
                )
            )
        if short_size_max is None:
            logger.error(
                "Short side maximum value can't be None in {}".format(
                    self.__class__.__name__
                )
            )

        if short_size_max <= short_size_min:
            logger.error(
                "Short side maximum value should be >= short side minimum value in {}."
                " Got: {} and {}".format(
                    self.__class__.__name__, short_size_max, short_size_min
                )
            )

        super().__init__(opts=opts)
        self.short_side_min = short_size_min
        self.size = size
        self.short_side_max = short_size_max
        self.interpolation = _check_interpolation(interpolation)
        self.enable = getattr(
            opts, "video_augmentation.random_short_side_resize_crop.enable"
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Defaults to False.".format(cls.__name__)
            ),
        )
        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Desired interpolation method. Defaults to bilinear",
        )
        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.short-side-min",
            type=int,
            default=None,
            help="Minimum value for video's shortest side. Defaults to None.",
        )
        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.short-side-max",
            type=int,
            default=None,
            help="Maximum value for video's shortest side. Defaults to None.",
        )
        return parser

    def get_params(self, height, width) -> Tuple[int, int, int, int]:
        th, tw = self.size

        if width == tw and height == th:
            return 0, 0, height, width

        i = random.randint(0, height - th)
        j = random.randint(0, width - tw)
        return i, j, th, tw

    def __call__(self, data: Dict) -> Dict:
        short_dim = random.randint(self.short_side_max, self.short_side_max)
        # resize the video so that shorter side is short_dim
        data = _resize_fn(data, size=short_dim, interpolation=self.interpolation)

        clip = data["samples"]["video"]
        check_rgb_video_tensor(clip=clip)
        height, width = clip.shape[-2:]
        i, j, h, w = self.get_params(height=height, width=width)
        # Crop the video.
        return _crop_fn(data=data, i=i, j=j, h=h, w=w)

    def __repr__(self) -> str:
        return "{}(size={}, short_size_range=({}, {}), interpolation={}, enable={})".format(
            self.__class__.__name__,
            self.size,
            self.short_side_min,
            self.short_side_max,
            self.interpolation,
            self.enable,
        )


@TRANSFORMATIONS_REGISTRY.register(name="random_crop", type="video")
class RandomCrop(BaseTransformation):
    """
    This method randomly crops a video area.

    .. note::
        This class assumes that the input video size is greater than or equal to the
        desired size.
    """

    def __init__(self, opts, size: Union[Tuple, int], *args, **kwargs) -> None:
        size = setup_size(size=size)
        super().__init__(opts=opts)
        self.size = size
        self.enable = getattr(opts, "video_augmentation.random_crop.enable")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--video-augmentation.random-crop.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Defaults to False.".format(cls.__name__)
            ),
        )
        return parser

    def get_params(self, height: int, width: int) -> Tuple[int, int, int, int]:
        th, tw = self.size

        if width == tw and height == th:
            return 0, 0, height, width

        i = random.randint(0, height - th)
        j = random.randint(0, width - tw)
        return i, j, th, tw

    def __call__(self, data: Dict) -> Dict:
        clip = data["samples"]["video"]
        check_rgb_video_tensor(clip=clip)
        height, width = clip.shape[-2:]
        i, j, h, w = self.get_params(height=height, width=width)
        return _crop_fn(data=data, i=i, j=j, h=h, w=w)

    def __repr__(self) -> str:
        return "{}(crop_size={}, enable={})".format(
            self.__class__.__name__, self.size, self.enable
        )


@TRANSFORMATIONS_REGISTRY.register(name="random_horizontal_flip", type="video")
class RandomHorizontalFlip(BaseTransformation):
    """
    This class implements random horizontal flipping method
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        p = getattr(opts, "video_augmentation.random_horizontal_flip.p", 0.5)
        super().__init__(opts=opts)
        self.p = p
        self.enable = getattr(opts, "video_augmentation.random_horizontal_flip.enable")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--video-augmentation.random-horizontal-flip.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Defaults to False.".format(cls.__name__)
            ),
        )
        group.add_argument(
            "--video-augmentation.random-horizontal-flip.p",
            type=float,
            default=0.5,
            help="Probability for random horizontal flip. Defaults to 0.5.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if random.random() <= self.p:
            clip = data["samples"]["video"]
            check_rgb_video_tensor(clip=clip)
            clip = torch.flip(clip, dims=[-1])
            data["samples"]["video"] = clip

            mask = data.get("mask", None)
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])
                data["mask"] = mask

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(flip probability={self.p},"
            f" enable={self.enable})"
        )


@TRANSFORMATIONS_REGISTRY.register(name="center_crop", type="video")
class CenterCrop(BaseTransformation):
    """
    This class implements center cropping method.

    .. note::
        This class assumes that the input size is greater than or equal to the desired
        size.
    """

    def __init__(self, opts, size: Sequence or int, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        if isinstance(size, Sequence) and len(size) == 2:
            self.height, self.width = size[0], size[1]
        elif isinstance(size, Sequence) and len(size) == 1:
            self.height = self.width = size[0]
        elif isinstance(size, int):
            self.height = self.width = size
        else:
            logger.error("Scale should be either an int or tuple of ints.")

        self.enable = getattr(opts, "video_augmentation.center_crop.enable")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--video-augmentation.center-crop.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Defaults to False.".format(cls.__name__)
            ),
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        height, width = data["samples"]["video"].shape[-2:]
        i = (height - self.height) // 2
        j = (width - self.width) // 2
        return _crop_fn(data=data, i=i, j=j, h=self.height, w=self.width)

    def __repr__(self) -> str:
        return "{}(size=(h={}, w={}), enable={})".format(
            self.__class__.__name__, self.height, self.width, self.enable
        )


@TRANSFORMATIONS_REGISTRY.register(name="resize", type="video")
class Resize(BaseTransformation):
    """
    This class implements resizing operation.

    .. note::
    Two possible modes for resizing.
    1. Resize while maintaining aspect ratio. To enable this option, pass int as a size.
    2. Resize to a fixed size. To enable this option, pass a tuple of height and width
        as a size.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        size = getattr(opts, "video_augmentation.resize.size", None)
        if size is None:
            logger.error("Size can not be None in {}".format(self.__class__.__name__))

        # Possible modes.
        # 1. Resize while maintaining aspect ratio. To enable this option, pass int as a
        # size.
        # 2. Resize to a fixed size. To enable this option, pass a tuple of height and
        # width as a size.

        if isinstance(size, Sequence) and len(size) > 2:
            logger.error(
                "The length of size should be either 1 or 2 in {}".format(
                    self.__class__.__name__
                )
            )

        interpolation = getattr(
            opts, "video_augmentation.resize.interpolation", "bilinear"
        )
        super().__init__(opts=opts)

        self.size = size
        self.interpolation = _check_interpolation(interpolation)
        self.enable = getattr(opts, "video_augmentation.resize.enable")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--video-augmentation.resize.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Defaults to False.".format(cls.__name__)
            ),
        )
        group.add_argument(
            "--video-augmentation.resize.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Interpolation for resizing. Defaults to bilinear",
        )
        group.add_argument(
            "--video-augmentation.resize.size",
            type=int,
            nargs="+",
            default=None,
            help=(
                "Resize video to the specified size. If int is passed, then shorter"
                " side is resized to the specified size and longest side is resized"
                " while maintaining aspect ratio. Defaults to None."
            ),
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return "{}(size={}, interpolation={}, enable={})".format(
            self.__class__.__name__, self.size, self.interpolation, self.enable
        )


@TRANSFORMATIONS_REGISTRY.register(name="crop_by_bounding_box", type="video")
class CropByBoundingBox(BaseTransformation):
    """Crops video frames based on bounding boxes and adjusts the @targets
    "box_coordinates" annotations.
    Before cropping, the bounding boxes are expanded with @multiplier, while the
    "box_coordinates" cover the original areas of the image.
    Note that the cropped images may be padded with 0 values in the boundaries of the
    cropped image when the bounding boxes are near the edges.

    Frames with invalid bounding boxes (with x0=y0=x1=y1=-1, or with area <5) will be
    blacked out in the output. Alternatively, we could have dropped them, which is not
    implemented yet.
    """

    BBOX_MIN_AREA = 5  # Minimum valid bounding box area (in pixels).

    def __init__(
        self,
        opts: argparse.Namespace,
        image_size: Optional[Tuple[int, int]] = None,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.is_training = is_training
        self.multiplier = getattr(
            opts, "video_augmentation.crop_by_bounding_box.multiplier"
        )
        self.multiplier_range = getattr(
            opts, "video_augmentation.crop_by_bounding_box.multiplier_range"
        )
        if image_size is None:
            self.image_size = getattr(
                opts, "video_augmentation.crop_by_bounding_box.image_size"
            )
        else:
            self.image_size = image_size
            assert image_size is not None, (
                "Please provide --video-augmentation.crop-by-bounding-box.image_size"
                " argument."
            )
        self.channel_first = getattr(
            opts, "video_augmentation.crop_by_bounding_box.channel_first"
        )
        self.interpolation = getattr(
            opts, "video_augmentation.crop_by_bounding_box.interpolation"
        )

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        """
        Tensor shape abbreviations:
            N: Number of clips.
            T, T_audio, T_video: Temporal lengths.
            C: Number of color channels.
            H, W: Height, Width.

        Args:
            data: mapping of: {
                "samples": {
                    "video": Tensor of shape: [N, C, T, H, W] if self.channel_first else [N, T, C, H, W]
                },
                "targets": {
                    "traces": {
                        "<object_trace_uuid>": {
                            "box_coordinates": FloatTensor[N, T, 4],  # x0, y0, x1, y1
                        }
                    },
                    "labels": IntTensor[N, T],
                }
            }

        Note:
            This transformation does not modify the "labels". If frames that are
            blacked out due to having invalid bounding boxes need a different label,
            datasets should alter the labels according to the following logic:
            ```
                data = CropByBoundingBox(opts)(data)
                trace, = data["targets"]["traces"].values()
                is_blacked_out = torch.all(trace["box_coordinates"] == -1, dim=2)
                data["targets"]["labels"][is_blacked_out] = <custom_label>
            ```
        """

        traces = data["targets"]["traces"]
        trace_identity = random.choice(list(traces.keys()))
        trace = traces[trace_identity]
        video = data["samples"]["video"]
        if self.channel_first:
            video = video.movedim(2, 1)

        N, T, C, H, W = video.shape
        expected_box_coordinates_shape = (N, T, 4)

        box_coordinates = trace["box_coordinates"]
        assert box_coordinates.shape == expected_box_coordinates_shape, (
            f"Unexpected shape {trace['box_coordinates'].shape} !="
            f" {expected_box_coordinates_shape}"
        )
        if self.is_training and self.multiplier_range is not None:
            multiplier = random.uniform(*self.multiplier_range)
        else:
            multiplier = self.multiplier

        expanded_corners, box_coordinates = self.expand_boxes(
            trace["box_coordinates"], multiplier, height=H, width=W
        )  # (NxTx4, NxTx4)

        expanded_corners = (
            (expanded_corners * torch.tensor([W, H, W, H]).float()).round().int()
        )  # NxTx4

        result = torch.empty(
            [N * T, C, *self.image_size],
            dtype=video.dtype,
            device=video.device,
        )
        for images, crop_corners, result_placeholder in zip(
            video.reshape(-1, C, H, W), expanded_corners.reshape(-1, 4).tolist(), result
        ):
            # TODO: add video_augmentation.crop_by_bounding_box.antialias argument to
            # experiment on antialias parameter of torchvision's resize function.
            width = crop_corners[2] - crop_corners[0]
            height = crop_corners[3] - crop_corners[1]
            if (
                width * height < CropByBoundingBox.BBOX_MIN_AREA
                or width < 0
                or height < 0
            ):
                # If the bounding box is invalid or too small, avoid cropping.
                result_placeholder[...] = 0.0  # Create black frames
            else:
                result_placeholder[...] = FV.resized_crop(
                    images,
                    left=crop_corners[0],
                    top=crop_corners[1],
                    width=width,
                    height=height,
                    size=self.image_size,
                    interpolation=InterpolationMode[self.interpolation.upper()],
                    antialias=True,
                )
        data["samples"]["video"] = result.reshape(N, T, C, *self.image_size)
        data["targets"]["traces"] = {
            trace_identity: {**trace, "box_coordinates": box_coordinates}
        }
        return data

    def expand_boxes(
        self, box_coordinates: torch.Tensor, multiplier: float, width: int, height: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            box_coordinates: Tensor of shape [..., 4] with (x0, y0, x1, y1) in [0,1].
            multiplier: The multiplier to expand the bounding box coordinates.

        Outputs (tuple items):
            expanded_corners: Tensor of shape [..., 4] with (x0, y0, x1, y1), containing
                the coordinates for cropping. Because of the expansion, coordinates
                could be negative or >1.
            box_coordinates: Tensor of shape [..., 4] with (x0, y0, x1, y1) in [0,1] to
                be used as bounding boxes after cropping.
            height: Height of the frame (in pixels).
            width: Width of the frame (in pixels).
        """
        x0 = box_coordinates[..., 0]  # Shape: NxT
        y0 = box_coordinates[..., 1]
        x1 = box_coordinates[..., 2]
        y1 = box_coordinates[..., 3]
        area = (x1 - x0) * width * (y1 - y0) * height
        invisible_mask = area < CropByBoundingBox.BBOX_MIN_AREA

        dw = (x1 - x0) * (multiplier - 1) / 2
        dh = (y1 - y0) * (multiplier - 1) / 2
        expanded_corners = torch.stack(
            [
                x0 - dw,
                y0 - dh,
                x1 + dw,
                y1 + dh,
            ],
            dim=-1,
        )

        # If multiplier is 1, new box_coordinates should cover the whole image (i.e.
        # [0., 0., 1., 1.]), as image was cropped based on the box_coordinates. For
        # multiplier > 1, new box_coordinates should have a small margin within the
        # boundaries (i.e. [new_x0, new_y0, 1-new_x0, 1-new_y0]).
        box_coordinates = torch.empty_like(box_coordinates)
        box_coordinates[..., :2] = self.get_new_x0(multiplier)
        box_coordinates[..., 2:] = 1 - box_coordinates[..., :2]
        expanded_corners[invisible_mask] = -1
        box_coordinates[invisible_mask] = -1
        return expanded_corners, box_coordinates

    @classmethod
    def get_new_x0(cls, multiplier: float) -> float:
        # new_width = old_width * multiplier
        # new_x0 = [(new_width - old_width) / 2] / new_width
        # => new_x0 = (1 - 1/multiplier) / 2
        return (1 - 1 / multiplier) / 2

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Default to False.".format(cls.__name__)
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.image-size",
            type=JsonValidator(Tuple[int, int]),
            default=None,
            help=(
                "Sizes [height, width] of the video frames after cropping. Defaults to"
                " None"
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.channel-first",
            action="store_true",
            default=False,
            help=(
                "If true, the video shape is [N, C, T, H, W]. Otherwise:"
                " [N, T, C, H, W]. Defaults to False."
            ),
        ),
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.multiplier-range",
            type=float,
            nargs=2,
            default=None,
            help=(
                "The bounding boxes get randomly expanded within the range before"
                " cropping. Useful for zooming in/out. Default None means no expansion"
                " of the bounding box."
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.multiplier",
            type=float,
            default=1,
            help=(
                "The bounding boxes get expanded by this multiplier before cropping."
                " Useful for zooming in/out. Defaults to 1."
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Desired interpolation method. Defaults to bilinear.",
        )

        return parser

    def __repr__(self) -> str:
        return "{}(image size={}, channel_first={}, multiplier={})".format(
            self.__class__.__name__,
            self.image_size,
            self.channel_first,
            self.multiplier,
        )


@TRANSFORMATIONS_REGISTRY.register(name="shuffle-audios", type="video")
class ShuffleAudios(BaseTransformation):
    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool,
        is_evaluation: bool,
        item_index: int,
        *args,
        **kwargs,
    ) -> None:
        """Transforms a batch of audio-visual clips. Generates binary labels, useful for
        self-supervised audio-visual training.

        At each invocation, a subset of clips within video (batch) get their audios
        shuffled. The ratio of clips that participate in the shuffling is configurable
        by argparse options.

        When training, the shuffle order is random. When evaluating, the shuffle order
        is deterministic.

        Args:
            is_training: When False, decide to shuffle the audios or not
                deterministically.
            is_evaluation: Combined with @is_training, determines which shuffle ratio
                argument to use (train/val/eval).
            item_index: Used for deterministic shuffling based on the item_index.
        """
        super().__init__(opts, *args, **kwargs)
        self.item_index = item_index
        self.is_training = is_training
        if is_training:
            self.shuffle_ratio = getattr(
                opts, "video_augmentation.shuffle_audios.shuffle_ratio_train"
            )
        elif is_evaluation:
            self.shuffle_ratio = getattr(
                opts, "video_augmentation.shuffle_audios.shuffle_ratio_test"
            )
        else:
            self.shuffle_ratio = getattr(
                opts, "video_augmentation.shuffle_audios.shuffle_ratio_val"
            )
        self.generate_frame_level_targets = getattr(
            opts,
            "video_augmentation.shuffle_audios.generate_frame_level_targets",
        )
        self.target_key = getattr(opts, "video_augmentation.shuffle_audios.target_key")
        self.debug_mode = getattr(opts, "video_augmentation.shuffle_audios.debug_mode")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--video-augmentation.shuffle-audios.shuffle-ratio-train",
            type=float,
            default=0.5,
            help=(
                "Ratio of training videos with shuffled audio samples. Defaults to 0.5."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.shuffle-ratio-val",
            type=float,
            default=0.5,
            help=(
                "Ratio of validation videos with shuffled audio samples. Defaults to "
                " 0.5."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.shuffle-ratio-test",
            type=float,
            default=0.5,
            help="Ratio of test videos with shuffled audio samples. Defaults to 0.5.",
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.generate-frame-level-targets",
            default=False,
            action="store_true",
            help=(
                "If true, the generated targets will be 2-dimensional (n_clips x "
                "n_frames). Otherwise, targets will be 1 dimensional (n_clips)."
                " Defaults to False."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.target-key",
            default="is_shuffled",
            type=str,
            help=(
                "Defaults to 'is_shuffled'. Name of the sub-key in data['targets'] "
                " to store the labels tensor. For each clip index `i`, we will have"
                " data['targets']['is_shuffled'][i] == 0 iff audio of the clip matches"
                " the video, otherwise 1."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.debug-mode",
            default=False,
            action="store_true",
            help=(
                "If enabled, the permutation used for shuffling the clip audios will be"
                " added to data['samples']['metadata']['shuffled_audio_permutation']"
                " for debugging purposes. Defaults to False."
            ),
        )

        return parser

    @staticmethod
    def _single_cycle_permutation(
        numel: int, is_training: bool, device: torch.device
    ) -> torch.LongTensor:
        """
        Returns a permutation of values 0 to @numel-1 that has the following property:
        For each index 0 <= i < numel: result[i] != i.

        Args:
            numel: Number of elements in the output permutation (must be >1).
            is_training: If true, the output permutation will be deterministic.
            device: Torch device (e.g. cuda, cpu) to use for output tensor.
        """
        assert numel > 1, "Cannot create a single-cycle permutation with <= 1 elements."

        deterministic_single_cycle_perm = torch.roll(
            torch.arange(numel, device=device), numel // 2
        )
        if not is_training:
            return deterministic_single_cycle_perm

        random_perm = torch.randperm(numel, device=device)

        random_perm_inv = torch.empty_like(random_perm)
        random_perm_inv[random_perm] = torch.arange(numel, device=device)

        # Proof that this implementation satisfies output[i] != i criteria:
        # 1. We know deterministic_single_cycle_perm[i] != i, because of the way it is
        #    constructed ([n//2, n//2+1, ..., n, 1, 2, ..., n//2-1]).
        # 2. ``rand_perm`` is a non-deterministic random permutation, and
        #    ``rand_perm_inv`` is the inverse of `rand_perm`. That means for each
        #    0 <= i < numel, we have: rand_perm_inv[rand_perm[i]] == i.
        # 3. Proof by contradiction: Let's assume, for 0 <= i < numel, i == output[i]:
        #    Thus: random_perm[deterministic_single_cycle_perm[random_perm_inv]][i] == i
        # 4. For any two torch tensors a, b that expression `a[b]`` is valid, we have
        #    a[b][i] == a[b[i]]. Thus, we can rewrite the assumption of step 3 as:
        #    i == random_perm[deterministic_single_cycle_perm[random_perm_inv[i]]]
        # 5. Now, apply rand_perm_inv[] on both sides of the equality:
        #    rand_perm_inv[i] == deterministic_single_cycle_perm[random_perm_inv[i]]
        #    Then, alias rand_perm_inv[i] as x. Then we will have:
        #    x == deterministic_single_cycle_perm[x]
        # 6. Assumption of step (3) leads to (5) which contradicts (1). Thus, assumption
        #    of step (3) is false. Thus, output[i] != i
        return random_perm[deterministic_single_cycle_perm[random_perm_inv]]

    def _random_outcome(self, n: int) -> torch.Tensor:
        """Returns a pseudo random tensor of size n in range [0, 1]. For evaluation,
        the outcome is a deterministic function of n and `self.item_index`

        Args:
            n: Length of the output tensor.

        Returns: A tensor of length n, of random floats uniformly distributed between
            0-1. The output is deterministic iff self.is_training is False.
        """
        if self.is_training:
            return torch.rand(n)
        else:
            return (
                (((self.item_index + 1) % torch.pi) * (torch.arange(n) + 1)) % torch.pi
            ) / torch.pi

    def _random_participants_mask(self, n: int) -> torch.BoolTensor:
        """Returns a pseudo random boolean tensor of size n, where exactly ``int(
        self.shuffle_ratio * n)`` indices are True, and the rest are False.
        """
        x = self._random_outcome(n)
        x = x.argsort() < self.shuffle_ratio * n - 1e-8
        return x

    def __call__(self, data: Dict) -> Dict:
        audio = data["samples"]["audio"]
        N = len(audio)
        if N == 1:
            shuffled_permutation = torch.tensor([0], device=audio.device)
            is_shuffling_participant_mask = torch.tensor([False], device=audio.device)
        elif N > 1:
            shuffled_permutation = self._single_cycle_permutation(
                N, device=audio.device, is_training=self.is_training
            )
            is_shuffling_participant_mask = self._random_participants_mask(N)
            shuffled_permutation = torch.where(
                is_shuffling_participant_mask,
                shuffled_permutation,
                torch.arange(N),
            )
        else:
            raise ValueError("Insufficient clips (N={N}) in batch.")

        data["samples"]["audio"] = audio[shuffled_permutation]
        if self.debug_mode:
            data["samples"]["metadata"][
                "shuffled_audio_permutation"
            ] = shuffled_permutation

        target_dims = 2 if self.generate_frame_level_targets else 1
        labels = torch.zeros(
            data["samples"]["video"].shape[:target_dims],
            device=audio.device,
            dtype=torch.long,
        )
        labels[is_shuffling_participant_mask] = 1.0  # 1 means shuffled
        data["targets"][self.target_key] = labels
        return data
