#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from pathlib import Path
from typing import Dict

import pytest
from torch import Tensor

from corenet.loss_fn import build_loss_fn
from corenet.modeling import get_model
from tests.configs import get_config
from tests.test_utils import unset_pretrained_models_from_opts


# We use a batch size of 1 to catch error that may arise due to reshaping operations inside the model
@pytest.mark.parametrize("batch_size", [1, 2])
def test_model(config_file: str, batch_size: int):
    opts = get_config(config_file=config_file)
    setattr(opts, "common.debug_mode", True)

    # removing pretrained models (if any) for now to reduce test time as well as access issues
    unset_pretrained_models_from_opts(opts)

    model = get_model(opts)

    criteria = build_loss_fn(opts)

    inputs = None
    targets = None
    if hasattr(model, "dummy_input_and_label"):
        inputs_and_targets = model.dummy_input_and_label(batch_size)
        inputs = inputs_and_targets["samples"]
        targets = inputs_and_targets["targets"]

    assert inputs is not None, (
        "Input tensor can't be None. This is likely because "
        "{} does not implement dummy_input_and_label function".format(
            model.__class__.__name__
        )
    )
    assert targets is not None, (
        "Label tensor can't be None. This is likely because "
        "{} does not implement dummy_input_and_label function".format(
            model.__class__.__name__
        )
    )

    try:
        outputs = model(inputs)

        loss = criteria(
            input_sample=inputs,
            prediction=outputs,
            target=targets,
            epoch=0,
            iterations=0,
        )

        print(f"Loss: {loss}")

        if isinstance(loss, Tensor):
            loss.backward()
        elif isinstance(loss, Dict):
            loss["total_loss"].backward()
        else:
            raise RuntimeError("The output of criteria should be either Dict or Tensor")

        # If there are unused parameters in gradient computation, print them
        # This may be useful for debugging purposes
        unused_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                unused_params.append(name)
        if len(unused_params) > 0:
            print("Unused parameters: {}".format(unused_params))

    except Exception as e:
        if (
            isinstance(e, ValueError)
            and str(e).find("Expected more than 1 value per channel when training") > -1
            and batch_size == 1
        ):
            # For segmentation models (e.g., PSPNet), we pool the tensor so that they have a spatial size of 1.
            # In such a case, batch norm needs a batch size > 1. Otherwise, we can't compute the statistics, raising
            # ValueError("Expected more than 1 value per channel when training"). If we encounter this error
            # for a batch size of 1, we skip it.
            pytest.skip(str(e))
        else:
            raise e


def exclude_yaml_from_test(yaml_file_path: Path) -> bool:
    """Check if a yaml file should be excluded from test based on first line marker.

    Args:
        yaml_file_path: path to the yaml file to check

    Returns:
        True if yaml should be excluded, and False otherwise.

    """
    part0 = yaml_file_path.parts[0]
    if part0 == ".":
        part0 = yaml_file_path.parts[1]

    if part0 in ("pipeline.yaml", "results", "venv", ".tox"):
        return True

    with open(yaml_file_path, "r") as f:
        first_line = f.readline().rstrip()
        return first_line.startswith("#") and first_line.lower().replace(
            " ", ""
        ).startswith("#pytest:disable")


def pytest_generate_tests(metafunc):
    configs = [
        str(x) for x in Path(".").rglob("**/*.yaml") if not exclude_yaml_from_test(x)
    ]

    metafunc.parametrize("config_file", configs)
