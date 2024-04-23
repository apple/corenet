#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

import pytest
import torch

from corenet.loss_fn.neural_augmentation import NeuralAugmentation


@pytest.mark.parametrize("batch_size", [1, 2])
def test_neural_aug_loss_in_out(batch_size: int) -> None:
    # These tests check the input and output formats are correct or not.
    # build configuration
    parser = argparse.ArgumentParser()
    parser = NeuralAugmentation.add_arguments(parser)

    opts = parser.parse_args([])
    setattr(opts, "scheduler.max_epochs", 20)

    # build loss function
    neural_aug_loss_fn = NeuralAugmentation(opts)
    pred_tensor = {
        "augmented_tensor": torch.zeros(
            size=(batch_size, 3, 224, 224), dtype=torch.float
        )
    }

    # Three input cases:
    # Case 1: Input image is a tensor
    # Case 2: Input is a dictionary, with image as a mandatory key and value as a batch of input image tensor
    # Case 3: Input is a dictionary, with image as a mandatory key and value as a list of input image tensor
    input_case_1 = torch.randint(low=0, high=1, size=(batch_size, 3, 224, 224))
    input_case_2 = {
        "image": torch.randint(low=0, high=1, size=(batch_size, 3, 224, 224))
    }
    input_case_3 = {
        "image": [torch.randint(low=0, high=1, size=(1, 3, 224, 224))] * batch_size
    }

    for inp in [input_case_1, input_case_2, input_case_3]:
        loss = neural_aug_loss_fn(inp, pred_tensor)
        assert isinstance(
            loss, torch.Tensor
        ), "Loss should be an instance of torch.Tensor"
        assert loss.dim() == 0, "Loss value should be a scalar"
