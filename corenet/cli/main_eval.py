#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import List, Optional

from corenet.options.opts import get_training_arguments
from corenet.train_eval_pipelines import (
    TRAIN_EVAL_PIPELINE_REGISTRY,
    BaseTrainEvalPipeline,
)


def main(train_eval_pipeline: BaseTrainEvalPipeline):
    """
    This function will be invoked on each gpu worker process.

    Args:
        train_eval_pipeline: Provides major pipeline components. The class to be used is
            configurable by "--train-eval-pipeline.name" opt. By default, an instance of
            ``train_eval_pipelines.TrainEvalPipeline`` will be passed to this function.
    """
    evaluation_engine = train_eval_pipeline.evaluation_engine
    evaluation_engine.run()


def main_worker(args: Optional[List[str]] = None):
    opts = get_training_arguments(args=args)
    pipeline_name = getattr(opts, "train_eval_pipeline.name")
    train_eval_pipeline = TRAIN_EVAL_PIPELINE_REGISTRY[pipeline_name](opts=opts)
    launcher = train_eval_pipeline.launcher
    launcher(main)


# for segmentation and detection, we follow a different evaluation pipeline that allows to save the results too
def main_worker_segmentation(args: Optional[List[str]] = None, **kwargs):
    from corenet.engine.eval_segmentation import main_segmentation_evaluation

    main_segmentation_evaluation(args=args, **kwargs)


def main_worker_detection(args: Optional[List[str]] = None, **kwargs):
    from corenet.engine.eval_detection import main_detection_evaluation

    main_detection_evaluation(args=args, **kwargs)


if __name__ == "__main__":
    main_worker()
