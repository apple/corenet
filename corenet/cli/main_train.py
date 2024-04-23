#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional

from torch.distributed.elastic.multiprocessing import errors

from corenet.options.opts import get_training_arguments
from corenet.train_eval_pipelines import (
    TRAIN_EVAL_PIPELINE_REGISTRY,
    BaseTrainEvalPipeline,
)


@errors.record
def callback(train_eval_pipeline: BaseTrainEvalPipeline) -> None:
    """
    This function will be invoked on each gpu worker process.

    Args:
        train_eval_pipeline: Provides major pipeline components. The class to be used is
            configurable by "--train-eval-pipeline.name" opt. By default, an instance of
            ``train_eval_pipelines.TrainEvalPipeline`` will be passed to this function.
    """
    train_sampler = train_eval_pipeline.train_sampler
    train_eval_pipeline.training_engine.run(train_sampler=train_sampler)


def main_worker(args: Optional[List[str]] = None):
    opts = get_training_arguments(args=args)
    pipeline_name = getattr(opts, "train_eval_pipeline.name")
    train_eval_pipeline = TRAIN_EVAL_PIPELINE_REGISTRY[pipeline_name](opts=opts)
    launcher = train_eval_pipeline.launcher
    launcher(callback)


if __name__ == "__main__":
    main_worker()
