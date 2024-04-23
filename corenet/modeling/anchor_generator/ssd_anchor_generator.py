#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from itertools import product
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from corenet.modeling.anchor_generator import ANCHOR_GEN_REGISTRY, BaseAnchorGenerator
from corenet.utils import logger


@ANCHOR_GEN_REGISTRY.register(name="ssd")
class SSDAnchorGenerator(BaseAnchorGenerator):
    """
    This class generates anchors (or priors) ``on-the-fly`` for the
    `single shot object detector (SSD) <https://arxiv.org/abs/1512.02325>`_.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        output_strides = getattr(
            opts, "anchor_generator.ssd.output_strides", [32, 64, 128, 256, -1]
        )
        aspect_ratios = getattr(
            opts, "anchor_generator.ssd.aspect_ratios", [[2, 3]] * len(output_strides)
        )

        min_ratio = getattr(opts, "anchor_generator.ssd.min_scale_ratio", 0.1)
        max_ratio = getattr(opts, "anchor_generator.ssd.max_scale_ratio", 1.05)
        no_clipping = getattr(opts, "anchor_generator.ssd.no_clipping", False)

        step = getattr(opts, "anchor_generator.ssd.step", [1])
        if isinstance(step, int):
            step = [step] * len(output_strides)
        elif isinstance(step, List) and len(step) <= len(output_strides):
            step = step + [1] * (len(output_strides) - len(step))
        else:
            logger.error(
                "--anchor-generator.ssd.step should be either a list of ints with the same length as "
                "the output strides OR an integer"
            )

        super().__init__()
        aspect_ratios = [list(set(ar)) for ar in aspect_ratios]
        output_strides_aspect_ratio = dict()
        for k, v in zip(output_strides, aspect_ratios):
            output_strides_aspect_ratio[k] = v
        self.output_strides_aspect_ratio = output_strides_aspect_ratio
        self.output_strides = output_strides
        self.anchors_dict = dict()

        self.num_output_strides = len(output_strides)
        self.num_aspect_ratios = len(aspect_ratios)

        scales = np.linspace(min_ratio, max_ratio, len(output_strides) + 1)
        self.sizes = dict()
        for i, s in enumerate(output_strides):
            self.sizes[s] = {
                "min": scales[i],
                "max": (scales[i] * scales[i + 1]) ** 0.5,
                "step": step[i],
            }

        self.clip = not no_clipping
        self.min_scale_ratio = min_ratio
        self.max_scale_ratio = max_ratio
        self.step = step

    def __repr__(self):
        return "{}(min_scale_ratio={}, max_scale_ratio={}, n_output_strides={}, n_aspect_ratios={}, clipping={})".format(
            self.__class__.__name__,
            self.min_scale_ratio,
            self.max_scale_ratio,
            self.num_output_strides,
            self.num_aspect_ratios,
            self.clip,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Adds SSD anchor generator-specific arguments to the parser
        """
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--anchor-generator.ssd.output-strides",
            nargs="+",
            type=int,
            help="Output strides of the feature maps for which we want to generate anchors",
        )
        group.add_argument(
            "--anchor-generator.ssd.aspect-ratios",
            nargs="+",
            type=float,
            action="append",
            help="Aspect ratios at each output stride",
        )

        # prior box arguments
        # SSD sample priors between min and max box sizes.
        # for example, if we use feature maps from three spatial levels (or output strides), then we
        # sample width and height for anchor boxes as:
        # scales = np.linspace(min_box_size, max_box_size, len(output_strides) + 1)
        # min_box dimensions for the first feature map is scales[0] * feature_map_dimensions
        # while the max_box dimensions will be sqrt(scales[0] * scales[1]) * feature_map dimensions. And so on
        group.add_argument(
            "--anchor-generator.ssd.min-scale-ratio",
            type=float,
            help="Min. scale ratio",
        )
        group.add_argument(
            "--anchor-generator.ssd.max-scale-ratio",
            type=float,
            help="Max. scale ratio",
        )
        group.add_argument(
            "--anchor-generator.ssd.no-clipping",
            action="store_true",
            help="Don't clip the anchors",
        )
        group.add_argument(
            "--anchor-generator.ssd.step",
            type=int,
            default=[1],
            nargs="+",
            help="Step between pixels",
        )
        return parser

    def num_anchors_per_os(self) -> List:
        """
        Returns anchors per output stride for SSD
        """
        return [2 + 2 * len(ar) for os, ar in self.output_strides_aspect_ratio.items()]

    @torch.no_grad()
    def _generate_anchors(
        self,
        height: int,
        width: int,
        output_stride: int,
        device: Optional[str] = "cpu",
        *args,
        **kwargs
    ) -> Tensor:
        min_size_h = self.sizes[output_stride]["min"]
        min_size_w = self.sizes[output_stride]["min"]

        max_size_h = self.sizes[output_stride]["max"]
        max_size_w = self.sizes[output_stride]["max"]
        aspect_ratio = self.output_strides_aspect_ratio[output_stride]

        step = max(1, self.sizes[output_stride]["step"])

        default_anchors_ctr = []

        start_step = max(0, step // 2)

        # Note that feature maps are in NCHW format
        for y, x in product(
            range(start_step, height, step), range(start_step, width, step)
        ):

            # [x, y, w, h] format
            cx = (x + 0.5) / width
            cy = (y + 0.5) / height

            # small box size
            default_anchors_ctr.append([cx, cy, min_size_w, min_size_h])

            # big box size
            default_anchors_ctr.append([cx, cy, max_size_w, max_size_h])

            # change h/w ratio of the small sized box based on aspect ratios
            for ratio in aspect_ratio:
                ratio = ratio**0.5
                default_anchors_ctr.extend(
                    [
                        [cx, cy, min_size_w * ratio, min_size_h / ratio],
                        [cx, cy, min_size_w / ratio, min_size_h * ratio],
                    ]
                )

        default_anchors_ctr = torch.tensor(
            default_anchors_ctr, dtype=torch.float, device=device
        )
        if self.clip:
            default_anchors_ctr = torch.clamp(default_anchors_ctr, min=0.0, max=1.0)

        return default_anchors_ctr
