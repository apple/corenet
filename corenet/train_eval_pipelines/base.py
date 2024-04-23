#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Callable

from corenet.engine.default_trainer import DefaultTrainer
from corenet.engine.evaluation_engine import Evaluator
from corenet.utils.registry import Registry


def Callback(Protocol):
    def __call__(self, train_eval_pipeline: BaseTrainEvalPipeline) -> Any: ...


class BaseTrainEvalPipeline:
    def __init__(
        self,
        opts: argparse.Namespace,
    ) -> None:
        """TrainEvalPipeline class is responsible for instantiating the components of
        training, evaluation, and/or pipelines that use those common components.

        The consumers of this class should be able to get an instance of any component
        by accessing the corresponding property. Example usage:

        >>> cfg = get_training_arguments()
        >>> pipeline = TrainEvalPipeline(cfg)
        >>> dataset, model = pipeline.dataset, pipeline.model

        Args:
            opts: Commandline options.
        """
        self.opts = opts

    def __init_subclass__(cls) -> None:
        for attr in dir(cls):
            if attr.startswith("_"):
                continue
            if isinstance(getattr(cls, attr), Callable):
                raise TypeError(
                    f"TrainEvalPipelines should only have [cached] properties."
                    f"'{cls}.{attr}' should not be a Callable. However, you can have a"
                    f"property/cached_property that returns a Callable. Also, you can"
                    f"have private methods that are named with '_' prefix."
                )

    def __getstate__(self) -> argparse.Namespace:
        return self.opts

    def __setstate__(self, opts: argparse.Namespace) -> None:
        self.opts = opts

    @property
    def evaluation_engine(self) -> Evaluator:
        """Creates the Evaluator instance that is used by main_eval.py"""
        raise NotImplementedError()

    @property
    def training_engine(self) -> DefaultTrainer:
        """Creates the Trainer instance that is used by corenet/cli/main_train.py"""
        raise NotImplementedError()

    @property
    def launcher(self) -> Callable[[Callback], None]:
        """Creates the entrypoints that spawn training and evaluation subprocesses."""
        raise NotImplementedError()


TRAIN_EVAL_PIPELINE_REGISTRY = Registry(
    registry_name="train_eval_pipeline",
    base_class=BaseTrainEvalPipeline,
    lazy_load_dirs=["corenet/train_eval_pipelines"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)
