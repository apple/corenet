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


@SAMPLER_REGISTRY.register(name="variable_batch_sampler")
class VariableBatchSampler(BaseSampler):
    """Variably-size multi-scale batch sampler <https://arxiv.org/abs/2110.02178?context=cs.LG>` for data parallel.
    This sampler yields batches with variable spatial resolution and batch size.

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

        crop_size_w = getattr(opts, "sampler.vbs.crop_size_width")
        crop_size_h = getattr(opts, "sampler.vbs.crop_size_height")

        min_crop_size_w = getattr(opts, "sampler.vbs.min_crop_size_width")
        max_crop_size_w = getattr(opts, "sampler.vbs.max_crop_size_width")

        min_crop_size_h = getattr(opts, "sampler.vbs.min_crop_size_height")
        max_crop_size_h = getattr(opts, "sampler.vbs.max_crop_size_height")

        check_scale_div_factor = getattr(opts, "sampler.vbs.check_scale")
        max_img_scales = getattr(opts, "sampler.vbs.max_n_scales")

        scale_inc = getattr(opts, "sampler.vbs.scale_inc")
        min_scale_inc_factor = getattr(opts, "sampler.vbs.min_scale_inc_factor")
        max_scale_inc_factor = getattr(opts, "sampler.vbs.max_scale_inc_factor")
        scale_ep_intervals = getattr(opts, "sampler.vbs.ep_intervals")
        if isinstance(scale_ep_intervals, int):
            scale_ep_intervals = [scale_ep_intervals]

        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

        self.min_scale_inc_factor = min_scale_inc_factor
        self.max_scale_inc_factor = max_scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals

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
        else:
            self.img_batch_tuples = [(crop_size_h, crop_size_w, self.batch_size_gpu0)]

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
        """Update the scales in variable batch sampler at specified epoch intervals during training."""
        if epoch in self.scale_ep_intervals and self.scale_inc:
            self.min_crop_size_w += int(
                self.min_crop_size_w * self.min_scale_inc_factor
            )
            self.max_crop_size_w += int(
                self.max_crop_size_w * self.max_scale_inc_factor
            )

            self.min_crop_size_h += int(
                self.min_crop_size_h * self.min_scale_inc_factor
            )
            self.max_crop_size_h += int(
                self.max_crop_size_h * self.max_scale_inc_factor
            )

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
            if is_master_node:
                logger.log("Scales updated in {}".format(self.__class__.__name__))
                logger.log("New scales: {}".format(self.img_batch_tuples))

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t base_im_size=(h={self.crop_size_h}, w={self.crop_size_w})"
            f"\n\t base_batch_size={self.batch_size_gpu0}"
            f"\n\t scales={self.img_batch_tuples}"
            f"\n\t scale_inc={self.scale_inc}"
            f"\n\t min_scale_inc_factor={self.min_scale_inc_factor}"
            f"\n\t max_scale_inc_factor={self.max_scale_inc_factor}"
            f"\n\t ep_intervals={self.scale_ep_intervals}"
        )

        return extra_repr_str

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != VariableBatchSampler:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--sampler.vbs.crop-size-width",
            default=DEFAULT_IMAGE_WIDTH,
            type=int,
            help=f"Base crop size (along width) during training. Defaults to {DEFAULT_IMAGE_WIDTH}.",
        )
        group.add_argument(
            "--sampler.vbs.crop-size-height",
            default=DEFAULT_IMAGE_HEIGHT,
            type=int,
            help=f"Base crop size (along height) during training. Defaults to {DEFAULT_IMAGE_HEIGHT}.",
        )

        group.add_argument(
            "--sampler.vbs.min-crop-size-width",
            default=160,
            type=int,
            help="Min. crop size along width during training. Defaults to 160.",
        )
        group.add_argument(
            "--sampler.vbs.max-crop-size-width",
            default=320,
            type=int,
            help="Max. crop size along width during training. Defaults to 320.",
        )

        group.add_argument(
            "--sampler.vbs.min-crop-size-height",
            default=160,
            type=int,
            help="Min. crop size along height during training. Defaults to 160.",
        )
        group.add_argument(
            "--sampler.vbs.max-crop-size-height",
            default=320,
            type=int,
            help="Max. crop size along height during training. Defaults to 320.",
        )
        group.add_argument(
            "--sampler.vbs.max-n-scales",
            default=5,
            type=int,
            help="Max. scales in variable batch sampler. Defaults to 5.",
        )
        group.add_argument(
            "--sampler.vbs.check-scale",
            default=32,
            type=int,
            help="Image scales should be divisible by this factor. Defaults to 32.",
        )
        group.add_argument(
            "--sampler.vbs.ep-intervals",
            default=[40],
            type=int,
            help="Epoch intervals at which scales should be adjusted. Defaults to 40.",
        )
        group.add_argument(
            "--sampler.vbs.min-scale-inc-factor",
            default=1.0,
            type=float,
            help="Factor by which we should increase the minimum scale. Defaults to 1.0",
        )
        group.add_argument(
            "--sampler.vbs.max-scale-inc-factor",
            default=1.0,
            type=float,
            help="Factor by which we should increase the maximum scale. Defaults to 1.0",
        )
        group.add_argument(
            "--sampler.vbs.scale-inc",
            action="store_true",
            default=False,
            help="Increase image scales during training. Defaults to False.",
        )

        return parser


@SAMPLER_REGISTRY.register(name="variable_batch_sampler_ddp")
class VariableBatchSamplerDDP(BaseSamplerDDP):
    """DDP version of  VariableBatchSampler

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

        crop_size_w = getattr(opts, "sampler.vbs.crop_size_width")
        crop_size_h = getattr(opts, "sampler.vbs.crop_size_height")

        min_crop_size_w = getattr(opts, "sampler.vbs.min_crop_size_width")
        max_crop_size_w = getattr(opts, "sampler.vbs.max_crop_size_width")

        min_crop_size_h = getattr(opts, "sampler.vbs.min_crop_size_height")
        max_crop_size_h = getattr(opts, "sampler.vbs.max_crop_size_height")

        check_scale_div_factor = getattr(opts, "sampler.vbs.check_scale")
        max_img_scales = getattr(opts, "sampler.vbs.max_n_scales")

        scale_inc = getattr(opts, "sampler.vbs.scale_inc")
        min_scale_inc_factor = getattr(opts, "sampler.vbs.min_scale_inc_factor")
        max_scale_inc_factor = getattr(opts, "sampler.vbs.max_scale_inc_factor")
        scale_ep_intervals = getattr(opts, "sampler.vbs.ep_intervals")
        if isinstance(scale_ep_intervals, int):
            scale_ep_intervals = [scale_ep_intervals]

        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h
        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w

        self.min_scale_inc_factor = min_scale_inc_factor
        self.max_scale_inc_factor = max_scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals
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

    def update_scales(self, epoch: int, is_master_node=False, *args, **kwargs) -> None:
        """Update the scales in variable batch sampler at specified epoch intervals during training."""
        if (epoch in self.scale_ep_intervals) and self.scale_inc:
            self.min_crop_size_w += int(
                self.min_crop_size_w * self.min_scale_inc_factor
            )
            self.max_crop_size_w += int(
                self.max_crop_size_w * self.max_scale_inc_factor
            )

            self.min_crop_size_h += int(
                self.min_crop_size_h * self.min_scale_inc_factor
            )
            self.max_crop_size_h += int(
                self.max_crop_size_h * self.max_scale_inc_factor
            )

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
            if is_master_node:
                logger.log("Scales updated in {}".format(self.__class__.__name__))
                logger.log("New scales: {}".format(self.img_batch_tuples))

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\t base_im_size=(h={self.crop_size_h}, w={self.crop_size_w})"
            f"\n\t base_batch_size={self.batch_size_gpu0}"
            f"\n\t scales={self.img_batch_tuples}"
            f"\n\t scale_inc={self.scale_inc}"
            f"\n\t min_scale_inc_factor={self.min_scale_inc_factor}"
            f"\n\t max_scale_inc_factor={self.max_scale_inc_factor}"
            f"\n\t ep_intervals={self.scale_ep_intervals}"
        )

        return extra_repr_str
