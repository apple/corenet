#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import io
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image

from corenet.data.transforms import TRANSFORMATIONS_REGISTRY, BaseTransformation


def _image_to_bytes(x: torch.Tensor, **kwargs) -> io.BytesIO:
    """
    Take an image in [0, 1] and save it as file bytes using PIL.

    Args:
        x: an image tensor in [C, H, W] order, where C is the number of channels,
            and H, W are the height and width.
        kwargs: any keyword arguments that can be passed to PIL's Image.save().

    Returns:
        The file bytes.

    """
    assert x.min() >= 0
    assert x.max() <= 1
    x = (x * 255).byte().permute(1, 2, 0).cpu().numpy()  # Bytes in H, W, C order

    img = Image.fromarray(x)
    byte_array = io.BytesIO()

    img.save(byte_array, **kwargs)
    return byte_array


def _bytes_to_int32(byte_array: io.BytesIO) -> torch.Tensor:
    """
    Convert a byte array to int32 values.

    Args:
        byte_array: The input byte array.
    Returns:
        The int32 tensor.
    """
    buf = np.frombuffer(byte_array.getvalue(), dtype=np.uint8)
    # The copy operation is required to avoid a warning about non-writable
    # tensors.
    buf = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
    return buf


@TRANSFORMATIONS_REGISTRY.register(name="pil_save", type="image_torch")
class PILSave(BaseTransformation):
    """
    Encode an image with a supported file encoding.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.file_encoding = getattr(opts, "image_augmentation.pil_save.file_encoding")
        self.quality = getattr(opts, "image_augmentation.pil_save.quality")
        self.opts = opts

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Serialize an image as file bytes.

        Args:
            data: A dictionary containing a key called "samples", which contains
                an image tensor of shape [C, H, W].

        Returns:
            The transformed data.
        """
        x = data["samples"]

        if self.file_encoding == "fCHW":
            x = (x * 255).byte().to(dtype=torch.int32).reshape(-1)
        elif self.file_encoding == "fHWC":
            x = (x * 255).byte().to(dtype=torch.int32).permute(1, 2, 0).reshape(-1)
        elif self.file_encoding == "TIFF":
            x = _bytes_to_int32(_image_to_bytes(x, format="tiff"))
        elif self.file_encoding == "PNG":
            x = _bytes_to_int32(_image_to_bytes(x, format="png", compress_level=0))
        elif self.file_encoding == "JPEG":
            quality = getattr(self.opts, "image_augmentation.pil_save.quality")
            x = _bytes_to_int32(_image_to_bytes(x, format="jpeg", quality=quality))
        else:
            raise NotImplementedError(
                f"Invalid file encoding {self.file_encoding}. Expected one of 'fCHW, fHWC, TIFF, PNG, JPEG'."
            )
        data["samples"] = x
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(file_encoding={self.file_encoding}, quality={self.quality})"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.pil-save.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.pil-save.file-encoding",
            choices=("fCHW", "fHWC", "TIFF", "PNG", "JPEG"),
            help="The type of file encoding to use. Defaults to TIFF.",
            default="TIFF",
        )
        group.add_argument(
            "--image-augmentation.pil-save.quality",
            help="JPEG quality if using JPEG encoding. Defaults to 100.",
            type=int,
            default=100,
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="shuffle_bytes", type="image_torch")
class ShuffleBytes(BaseTransformation):
    """
    Reorder the bytes in a 1-dimensional buffer.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.mode = getattr(opts, "image_augmentation.shuffle_bytes.mode")
        self.stride = getattr(opts, "image_augmentation.shuffle_bytes.stride")
        window_size = getattr(opts, "image_augmentation.shuffle_bytes.window_size")
        self.window_shuffle = torch.randperm(window_size)

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Reorder the bytes of a 1-dimensional buffer.

        Args:
            data: A dictionary containing a key called "samples", which contains
                a tensor of shape [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]
        if not x.dim() == 1:
            raise ValueError(f"Expected 1d input, got {x.shape}")

        if self.mode == "reverse":
            x = torch.fliplr(x.view(1, -1))[0]
        elif self.mode == "random_shuffle":
            x = x[torch.randperm(x.shape[0])]
        elif self.mode == "cyclic_half_length":
            x = torch.roll(x, x.shape[0] // 2)
        elif self.mode == "stride":
            l = len(x)
            values = []
            for i in range(self.stride):
                values.append(x[i :: self.stride])
            x = torch.cat(values, dim=0)
            assert len(x) == l
        elif self.mode == "window_shuffle":
            l = len(x)
            window_size = self.window_shuffle.shape[0]
            num_windows = l // window_size
            values = []
            for i in range(num_windows):
                chunk = x[i * window_size : (i + 1) * window_size]
                values.append(chunk[self.window_shuffle])

            # Add the last bits that fall outside the shuffling window.
            values.append(x[num_windows * window_size :])
            x = torch.cat(values, dim=0)
            assert len(x) == l
        else:
            raise NotImplementedError(
                f"mode={self.mode} not implemented. Expected one of 'reverse, random_shuffle, cyclic_half_length, stride, window_shuffle'."
            )
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.shuffle-bytes.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.shuffle-bytes.mode",
            default="reverse",
            help="The mode to use when shuffling bytes. Defaults to 'reverse'.",
            choices=(
                "reverse",
                "random_shuffle",
                "cyclic_half_length",
                "stride",
                "window_shuffle",
            ),
        )
        group.add_argument(
            "--image-augmentation.shuffle-bytes.stride",
            type=int,
            default=1024,
            help="The stride of the window used in shuffling operations that are windowed. Defaults to 1024.",
        )
        group.add_argument(
            "--image-augmentation.shuffle-bytes.window-size",
            type=int,
            default=1024,
            help="The size of the window used in shuffling operations that are windowed. Defaults to 1024.",
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="mask_positions", type="image_torch")
class MaskPositions(BaseTransformation):
    """
    Mask out values in a 1-dimensional buffer using a fixed masking pattern.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.keep_frac = getattr(opts, "image_augmentation.mask_positions.keep_frac")
        self._cached_masks = None

    def _generate_masks(self, N: int) -> torch.Tensor:
        if self._cached_masks is None:
            g = torch.Generator()
            # We want to fix the mask across all inputs, so we fix the seed.
            # Choose a seed with a good balance of 0 and 1 bits. See:
            # https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.manual_seed
            g.manual_seed(2147483647)
            random_mask = torch.zeros([N], requires_grad=False, dtype=torch.bool)
            random_mask[torch.randperm(N, generator=g)[: int(self.keep_frac * N)]] = 1
            self._cached_masks = random_mask
        return self._cached_masks

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Mask values in a 1-dimensional buffer with a fixed masking pattern.

        Args:
            data: A dictionary containing a key called "samples", which contains
                a tensor of shape [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]
        mask = self._generate_masks(x.shape[0])
        x = x[mask]
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.mask-positions.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.mask-positions.keep-frac",
            type=float,
            default=0.5,
            help="The fraction of bytes to keep. Defaults to 0.5.",
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="byte_permutation", type="image_torch")
class BytePermutation(BaseTransformation):
    """
    Remap byte values in [0, 255] to new values in [0, 255] using a permutation.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)

        g = torch.Generator()
        g.manual_seed(2147483647)
        self.mask = torch.randperm(256, generator=g)

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Remap byte values in [0, 255] to new values in [0, 255] using a permutation.

        Args:
            data: A dictionary containing a key called "samples", which contains
                a tensor of shape [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]

        if x.dim() != 1:
            raise ValueError(f"Expected 1d tensor. Got {x.shape}.")
        x = torch.index_select(self.mask, dim=0, index=x)
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.byte-permutation.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="random_uniform", type="image_torch")
class RandomUniformNoise(BaseTransformation):
    """
    Add random uniform noise to integer values.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.opts = opts

        self.width_range = getattr(
            opts, "image_augmentation.random_uniform.width_range"
        )

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Add random uniform noise to byte values.

        Args:
            data: A dict containing a tensor in its "samples" key. The tensor
                contains integers representing byte values. Integers are used
                because negative padding values may be added later. The shape
                of the tenor is [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]
        noise = torch.randint_like(x, self.width_range[0], self.width_range[1] + 1)
        dtype = x.dtype
        x = x.int()
        x = x + noise
        x = x % 256
        x = x.to(dtype)
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.random-uniform.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-uniform.width-range",
            type=int,
            nargs=2,
            default=[-5, 5],
            help="The range of values from which to add noise. It is specified"
            " as [low, high] (inclusive). Defaults to [-5, 5].",
        )
        return parser
