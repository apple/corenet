#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import random
import sys
from typing import List, Optional, Union

import numpy as np
import pytest

sys.path.append("..")

from corenet.optims.scheduler import build_scheduler
from tests.configs import get_config

LR_TOLERANCE = 1e-3

MAX_LRS = np.linspace(0.001, 0.1, 3)
WARMUP_ITERATIONS = [None, 100, 1000, 10000]

BATCH_SIZE = 100
DATASET_SIZE = 20000


def run_test(
    scheduler, num_epochs: int, num_batches: int, return_all_lrs: Optional[bool] = False
) -> Union[List, float]:
    end_lr = [] if return_all_lrs else 0.0
    curr_iter = 0
    for ep in range(num_epochs):
        for _ in range(num_batches):
            lr = scheduler.get_lr(ep, curr_iter=curr_iter)
            curr_iter += 1

        # keep only epoch-wise LRs
        if return_all_lrs:
            end_lr.append(lr)
        else:
            end_lr = lr

    return end_lr


@pytest.mark.parametrize("start_lr", MAX_LRS)
@pytest.mark.parametrize("warmup_iteration", WARMUP_ITERATIONS)
def test_polynomial_scheduler(start_lr, warmup_iteration, *args, **kwargs):
    opts = get_config()
    setattr(opts, "scheduler.max_iterations", 100000)
    setattr(opts, "scheduler.name", "polynomial")
    num_iterations = getattr(opts, "scheduler.max_iterations", 100000)
    num_batches = DATASET_SIZE // BATCH_SIZE
    num_epochs = num_iterations // num_batches

    # Test for iteration-based samplers
    setattr(opts, "scheduler.is_iteration_based", True)
    setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
    end_lr = round(start_lr / random.randint(2, 10), 5)
    setattr(opts, "scheduler.polynomial.start_lr", start_lr)
    setattr(opts, "scheduler.polynomial.end_lr", end_lr)
    scheduler = build_scheduler(opts)
    lr = run_test(scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches)
    np.testing.assert_allclose(end_lr, lr, atol=LR_TOLERANCE)

    # Test for epoch-based samplers
    setattr(opts, "scheduler.is_iteration_based", False)
    setattr(opts, "scheduler.max_epochs", num_epochs)
    setattr(opts, "scheduler.adjust_period_for_epochs", True)
    setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
    end_lr = round(start_lr / random.randint(2, 10), 5)
    setattr(opts, "scheduler.polynomial.start_lr", start_lr)
    setattr(opts, "scheduler.polynomial.end_lr", end_lr)
    scheduler = build_scheduler(opts)
    lr = run_test(scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches)

    np.testing.assert_allclose(end_lr, lr, atol=LR_TOLERANCE)


@pytest.mark.parametrize("start_lr", MAX_LRS)
@pytest.mark.parametrize("warmup_iteration", WARMUP_ITERATIONS)
def test_cosine_scheduler(start_lr, warmup_iteration, *args, **kwargs):
    opts = get_config()
    setattr(opts, "scheduler.max_iterations", 100000)
    setattr(opts, "scheduler.name", "cosine")

    num_iterations = getattr(opts, "scheduler.max_iterations", 100000)
    num_batches = DATASET_SIZE // BATCH_SIZE
    num_epochs = num_iterations // num_batches

    # first test for iteration-based samplers
    setattr(opts, "scheduler.is_iteration_based", True)
    setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
    end_lr = round(start_lr / random.randint(2, 10), 5)
    setattr(opts, "scheduler.cosine.max_lr", start_lr)
    setattr(opts, "scheduler.cosine.min_lr", end_lr)
    scheduler = build_scheduler(opts)
    lr = run_test(scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches)
    np.testing.assert_allclose(end_lr, lr, atol=LR_TOLERANCE)

    # Test for epoch-based samplers
    setattr(opts, "scheduler.is_iteration_based", False)
    setattr(opts, "scheduler.max_epochs", num_epochs)
    setattr(opts, "scheduler.adjust_period_for_epochs", True)
    setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
    end_lr = round(start_lr / random.randint(2, 10), 5)
    setattr(opts, "scheduler.cosine.max_lr", start_lr)
    setattr(opts, "scheduler.cosine.min_lr", end_lr)
    scheduler = build_scheduler(opts)
    lr = run_test(scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches)
    np.testing.assert_allclose(end_lr, lr, atol=LR_TOLERANCE)


@pytest.mark.parametrize("start_lr", MAX_LRS)
@pytest.mark.parametrize("warmup_iteration", WARMUP_ITERATIONS)
def test_fixed_scheduler(start_lr, warmup_iteration, *args, **kwargs):
    opts = get_config()
    setattr(opts, "scheduler.max_iterations", 100000)
    setattr(opts, "scheduler.name", "fixed")

    num_iterations = getattr(opts, "scheduler.max_iterations", 100000)
    num_batches = DATASET_SIZE // BATCH_SIZE
    num_epochs = num_iterations // num_batches

    # Test for iteration-based samplers
    setattr(opts, "scheduler.is_iteration_based", True)
    setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
    setattr(opts, "scheduler.fixed.lr", start_lr)
    scheduler = build_scheduler(opts)
    lr = run_test(scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches)
    np.testing.assert_allclose(start_lr, lr, atol=LR_TOLERANCE)

    # Test for epoch-based samplers
    setattr(opts, "scheduler.is_iteration_based", False)
    setattr(opts, "scheduler.max_epochs", num_epochs)
    setattr(opts, "scheduler.adjust_period_for_epochs", True)
    setattr(opts, "scheduler.warmup_iterations", warmup_iteration)
    setattr(opts, "scheduler.fixed.lr", start_lr)
    scheduler = build_scheduler(opts)
    lr = run_test(scheduler=scheduler, num_epochs=num_epochs, num_batches=num_batches)
    np.testing.assert_allclose(start_lr, lr, atol=LR_TOLERANCE)
