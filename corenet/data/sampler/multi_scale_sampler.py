#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import random
from typing import Iterator, Tuple

from corenet.constants import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
from corenet.data.sampler import SAMPLER_REGISTRY
from corenet.data.sampler.base_sampler import BaseSampler, BaseSamplerDDP
from corenet.data.sampler.utils import image_batch_pairs
from corenet.utils import logger


@SAMPLER_REGISTRY.register(name="multi_scale_sampler")
class MultiScaleSampler(BaseSampler):
    """Multi-scale batch sampler for data parallel. This sampler yields batches of fixed batch size, but each batch
    has different spatial resolution.

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

        crop_size_w = getattr(opts, "sampler.msc.crop_size_width")
        crop_size_h = getattr(opts, "sampler.msc.crop_size_height")

        min_crop_size_w = getattr(opts, "sampler.msc.min_crop_size_width")
        max_crop_size_w = getattr(opts, "sampler.msc.max_crop_size_width")

        min_crop_size_h = getattr(opts, "sampler.msc.min_crop_size_height")
        max_crop_size_h = getattr(opts, "sampler.msc.max_crop_size_height")

        check_scale_div_factor = getattr(opts, "sampler.msc.check_scale")
        max_img_scales = getattr(opts, "sampler.msc.max_n_scales")

        scale_inc = getattr(opts, "sampler.msc.scale_inc")

        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
            )
            # over-ride the batch-size
            self.img_batch_tuples = [
                (h, w, self.batch_size_gpu0) for h, w, b in self.img_batch_tuples
            ]
        else:
            self.img_batch_tuples = [(crop_size_h, crop_size_w, self.batch_size_gpu0)]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != MultiScaleSampler:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--sampler.msc.crop-size-width",
            default=DEFAULT_IMAGE_WIDTH,
            type=int,
            help=f"Base crop size (along width) during training. Defaults to {DEFAULT_IMAGE_WIDTH}.",
        )
        group.add_argument(
            "--sampler.msc.crop-size-height",
            default=DEFAULT_IMAGE_HEIGHT,
            type=int,
            help=f"Base crop size (along height) during training. Defaults to {DEFAULT_IMAGE_HEIGHT}.",
        )

        group.add_argument(
            "--sampler.msc.min-crop-size-width",
            default=160,
            type=int,
            help="Min. crop size along width during training. Defaults to 160.",
        )
        group.add_argument(
            "--sampler.msc.max-crop-size-width",
            default=320,
            type=int,
            help="Max. crop size along width during training. Defaults to 320.",
        )

        group.add_argument(
            "--sampler.msc.min-crop-size-height",
            default=160,
            type=int,
            help="Min. crop size along height during training. Defaults to 160.",
        )
        group.add_argument(
            "--sampler.msc.max-crop-size-height",
            default=320,
            type=int,
            help="Max. crop size along height during training. Defaults to 320.",
        )
        group.add_argument(
            "--sampler.msc.max-n-scales",
            default=5,
            type=int,
            help="Max. scales in variable batch sampler. Defaults to 5.",
        )
        group.add_argument(
            "--sampler.msc.check-scale",
            default=32,
            type=int,
            help="Image scales should be divisible by this factor. Defaults to 32.",
        )
        group.add_argument(
            "--sampler.msc.scale-inc",
            action="store_true",
            default=False,
            help="Increase image scales during training. Defaults to False.",
        )

        return parser

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        img_indices = self.get_indices()
        start_index = 0
        n_samples = len(img_indices)
        while start_index < n_samples:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, n_samples)
            batch_ids = img_indices[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if len(batch_ids) != batch_size:
                batch_ids += img_indices[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def update_scales(
        self, epoch: int, is_master_node: bool = False, *args, **kwargs
    ) -> None:
        if type(self).update_scales is not MultiScaleSampler.update_scales:
            # Do nothing when a subclass overrides this method and calls super().update_scales
            return

        if is_master_node and self.scale_inc:
            logger.warning(
                f"Update scale function is not yet implemented for {self.__class__.__name__}."
            )

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t base_im_size=(h={self.crop_size_h}, w={self.crop_size_w})"
            f"\n\t base_batch_size={self.batch_size_gpu0}"
            f"\n\t scales={self.img_batch_tuples}"
        )
        return extra_repr_str


@SAMPLER_REGISTRY.register(name="multi_scale_sampler_ddp")
class MultiScaleSamplerDDP(BaseSamplerDDP):
    """DDP version of MultiScaleSampler

    Args:
        opts: command line argument
        n_data_samples: Number of samples in the dataset
        is_training: Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        crop_size_w = getattr(opts, "sampler.msc.crop_size_width")
        crop_size_h = getattr(opts, "sampler.msc.crop_size_height")

        min_crop_size_w = getattr(opts, "sampler.msc.min_crop_size_width")
        max_crop_size_w = getattr(opts, "sampler.msc.max_crop_size_width")

        min_crop_size_h = getattr(opts, "sampler.msc.min_crop_size_height")
        max_crop_size_h = getattr(opts, "sampler.msc.max_crop_size_height")

        check_scale_div_factor = getattr(opts, "sampler.msc.check_scale")

        max_img_scales = getattr(opts, "sampler.msc.max_n_scales")
        scale_inc = getattr(opts, "sampler.msc.scale_inc")

        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h
        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w

        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.num_replicas,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
            )
            self.img_batch_tuples = [
                (h, w, self.batch_size_gpu0) for h, w, b in self.img_batch_tuples
            ]
        else:
            self.img_batch_tuples = [
                (self.crop_size_h, self.crop_size_w, self.batch_size_gpu0)
            ]

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        indices_rank_i = self.get_indices_rank_i()

        start_index = 0
        n_samples_rank_i = len(indices_rank_i)
        while start_index < n_samples_rank_i:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, n_samples_rank_i)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t base_im_size=(h={self.crop_size_h}, w={self.crop_size_w})"
            f"\n\t base_batch_size={self.batch_size_gpu0}"
            f"\n\t scales={self.img_batch_tuples}"
        )
        return extra_repr_str

    def update_scales(
        self, epoch: int, is_master_node: bool = False, *args, **kwargs
    ) -> None:
        if type(self).update_scales is not MultiScaleSamplerDDP.update_scales:
            # Do nothing when a subclass overrides this method and calls super().update_scales
            return

        if is_master_node and self.scale_inc:
            logger.warning(
                f"Update scale function is not yet implemented for {self.__class__.__name__}"
            )
