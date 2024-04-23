#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Iterator, Tuple

from corenet.constants import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
from corenet.data.sampler import SAMPLER_REGISTRY
from corenet.data.sampler.base_sampler import BaseSampler, BaseSamplerDDP


@SAMPLER_REGISTRY.register(name="batch_sampler")
class BatchSampler(BaseSampler):
    """Standard Batch Sampler for data parallel. This sampler yields batches of fixed batch size
    and spatial resolutions.

    Args:
        opts: command line argument
        n_data_samples: Number of samples in the dataset
        is_training: Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        # spatial dimensions
        crop_size_w: int = getattr(opts, "sampler.bs.crop_size_width")
        crop_size_h: int = getattr(opts, "sampler.bs.crop_size_height")

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        img_indices = self.get_indices()

        start_index = 0
        batch_size = self.batch_size_gpu0
        n_samples = len(img_indices)
        while start_index < n_samples:

            end_index = min(start_index + batch_size, n_samples)
            batch_ids = img_indices[start_index:end_index]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids
                ]
                yield batch

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\tbase_im_size=(h={self.crop_size_h}, w={self.crop_size_w})"
            f"\n\tbase_batch_size={self.batch_size_gpu0}"
        )
        return extra_repr_str

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BatchSampler:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--sampler.bs.crop-size-width",
            default=DEFAULT_IMAGE_WIDTH,
            type=int,
            help="Base crop size (along width) during training",
        )
        group.add_argument(
            "--sampler.bs.crop-size-height",
            default=DEFAULT_IMAGE_HEIGHT,
            type=int,
            help="Base crop size (along height) during training",
        )
        return parser


@SAMPLER_REGISTRY.register(name="batch_sampler_ddp")
class BatchSamplerDDP(BaseSamplerDDP):
    """DDP variant of BatchSampler

    Args:
        opts: command line argument
        n_data_samples: Number of samples in the dataset
        is_training: Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        crop_size_w: int = getattr(
            opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH
        )
        crop_size_h: int = getattr(
            opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT
        )

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        indices_rank_i = self.get_indices_rank_i()
        start_index = 0
        batch_size = self.batch_size_gpu0

        n_samples_rank_i = len(indices_rank_i)
        while start_index < n_samples_rank_i:
            end_index = min(start_index + batch_size, n_samples_rank_i)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids
                ]
                yield batch

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\tbase_im_size=(h={self.crop_size_h}, w={self.crop_size_w})"
            f"\n\tbase_batch_size={self.batch_size_gpu0}"
        )
        return extra_repr_str
