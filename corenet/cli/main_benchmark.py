#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import time
from typing import Optional

import torch
from torch.cuda.amp import autocast

from corenet.engine.utils import autocast_fn
from corenet.modeling import get_model
from corenet.options.opts import get_benchmarking_arguments
from corenet.utils import logger
from corenet.utils.common_utils import device_setup
from corenet.utils.pytorch_to_coreml import convert_pytorch_to_coreml
from corenet.utils.tensor_utils import create_rand_tensor


def cpu_timestamp(*args, **kwargs):
    # perf_counter returns time in seconds
    return time.perf_counter()


def cuda_timestamp(cuda_sync=False, device=None, *args, **kwargs):
    if cuda_sync:
        torch.cuda.synchronize(device=device)
    # perf_counter returns time in seconds
    return time.perf_counter()


def step(
    time_fn,
    model,
    example_inputs,
    autocast_enable: False,
    amp_precision: Optional[str] = "float16",
):
    start_time = time_fn()
    with autocast_fn(enabled=autocast_enable, amp_precision=amp_precision):
        model(example_inputs)
    end_time = time_fn(cuda_sync=True)
    return end_time - start_time


def main_benchmark():
    # set-up
    opts = get_benchmarking_arguments()
    # device set-up
    opts = device_setup(opts)

    norm_layer = getattr(opts, "model.normalization.name", "batch_norm")
    if norm_layer.find("sync") > -1:
        norm_layer = norm_layer.replace("sync_", "")
        setattr(opts, "model.normalization.name", norm_layer)
    device = getattr(opts, "dev.device", torch.device("cpu"))
    if torch.cuda.device_count() == 0:
        device = torch.device("cpu")
    time_fn = cpu_timestamp if device == torch.device("cpu") else cuda_timestamp
    warmup_iterations = getattr(opts, "benchmark.warmup_iter", 10)
    iterations = getattr(opts, "benchmark.n_iter", 50)
    batch_size = getattr(opts, "benchmark.batch_size", 1)
    mixed_precision = (
        False
        if device == torch.device("cpu")
        else getattr(opts, "common.mixed_precision", False)
    )
    mixed_precision_dtype = getattr(opts, "common.mixed_precision_dtype", "float16")

    # load the model
    model = get_model(opts)
    model.eval()
    # print model information
    model.info()

    example_inp = create_rand_tensor(opts=opts, device="cpu", batch_size=batch_size)

    # cool down for 5 seconds
    time.sleep(5)

    if getattr(opts, "benchmark.use_jit_model", False):
        converted_models_dict = convert_pytorch_to_coreml(
            opts=None,
            pytorch_model=model,
            input_tensor=example_inp,
            jit_model_only=True,
        )
        model = converted_models_dict["jit"]
    model = model.to(device=device)
    example_inp = example_inp.to(device=device)
    model.eval()

    with torch.no_grad():
        # warm-up
        for i in range(warmup_iterations):
            step(
                time_fn=time_fn,
                model=model,
                example_inputs=example_inp,
                autocast_enable=mixed_precision,
                amp_precision=mixed_precision_dtype,
            )

        n_steps = n_samples = 0.0

        # run benchmark
        for i in range(iterations):
            step_time = step(
                time_fn=time_fn,
                model=model,
                example_inputs=example_inp,
                autocast_enable=mixed_precision,
                amp_precision=mixed_precision_dtype,
            )
            n_steps += step_time
            n_samples += batch_size

        logger.info(
            "Number of samples processed per second: {:.3f}".format(n_samples / n_steps)
        )


if __name__ == "__main__":
    main_benchmark()
