#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from pathlib import Path

import pytest
import torch

from corenet.cli import main_train
from corenet.train_eval_pipelines.default_train_eval import DefaultTrainEvalPipeline
from tests.configs import get_config
from tests.test_utils import unset_pretrained_models_from_opts


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/engine/dummy_configs/imagenet_classification/mobilevit.yaml",
        "tests/engine/dummy_configs/imagenet_classification/mobilevit_v2.yaml",
        "tests/engine/dummy_configs/ade20k_segmentation/deeplabv3_mobilenetv2.yaml",
        "tests/engine/dummy_configs/coco_detection/resnet_ssd.yaml",
        "tests/engine/dummy_configs/coco_detection/resnet_mask_rcnn.yaml",
        "tests/engine/dummy_configs/image_text_clip/clip_vit.yaml",
        # add a configuration to test range augment
        "tests/engine/dummy_configs/imagenet_classification/efficientnet_b0.yaml",
        "tests/engine/dummy_configs/language_modeling_gpt/gpt.yaml",
    ],
)
def test_training_engine(config_file: str, tmp_path: Path) -> None:
    opts = get_config(config_file=config_file)

    # Parallel tests causes issues when save_dir is accessed by multiple workers.
    # Therefore, we use a unique random path here and use that as a save location.
    save_dir = str(tmp_path)
    setattr(opts, "common.results_loc", save_dir)

    # Set device-related args that are not exposed to users
    n_gpus = torch.cuda.device_count()
    device = "cuda" if n_gpus > 0 else "cpu"

    setattr(opts, "dev.num_gpus", n_gpus)
    setattr(opts, "dev.device_id", None)
    setattr(opts, "dev.device", torch.device(device))

    if n_gpus == 0:
        # Need to disable mixed_precision for testing on CPU only.
        setattr(opts, "common.mixed_precision", False)

    norm_name = getattr(opts, "model.normalization.name")

    if norm_name is not None and norm_name in ["sync_batch_norm", "sbn"]:
        # on CPUs, Sync BN won't work.
        setattr(opts, "model.normalization.name", "batch_norm")

    assert (
        getattr(opts, "train_eval_pipeline.name") == "default"
    ), "This unit-test has does not support configs with custom TrainEvalPipelines yet."

    # removing pretrained models (if any) for now to reduce test time as well as access issues
    unset_pretrained_models_from_opts(opts)

    train_eval_pipeline = DefaultTrainEvalPipeline(opts)
    train_eval_pipeline.launcher(main_train.callback)

    assert Path(save_dir, "train/checkpoint_last.pt").exists()
