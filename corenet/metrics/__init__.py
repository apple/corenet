#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.utils.registry import Registry

METRICS_REGISTRY = Registry(
    "metrics",
    lazy_load_dirs=["corenet/metrics"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_stats(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Statistics", description="Statistics")
    group.add_argument(
        "--stats.val", type=str, default=["loss"], nargs="+", help="Name of statistics"
    )
    group.add_argument(
        "--stats.train",
        type=str,
        default=["loss"],
        nargs="+",
        help="Name of statistics",
    )
    group.add_argument(
        "--stats.checkpoint-metric",
        type=str,
        default="loss",
        help="Metric to use for saving checkpoints",
    )
    group.add_argument(
        "--stats.checkpoint-metric-max",
        action="store_true",
        default=False,
        help="Maximize checkpoint metric",
    )
    group.add_argument(
        "--stats.coco-map.iou-types",
        type=str,
        default=["bbox"],
        nargs="+",
        choices=("bbox", "segm"),
        help="Types of IOU to compute for MSCoco.",
    )

    return parser
