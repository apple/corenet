#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Callable, List, Optional, Union

import coremltools as ct

from corenet.constants import DEFAULT_LOGS_DIR, DEFAULT_RESULTS_DIR
from corenet.data.collate_fns import arguments_collate_fn
from corenet.data.datasets import arguments_dataset
from corenet.data.io.transfer_clients import transfer_client_arguments
from corenet.data.sampler import add_sampler_arguments
from corenet.data.text_tokenizer import arguments_tokenizer
from corenet.data.transforms import arguments_augmentation
from corenet.data.video_reader import arguments_video_reader
from corenet.loss_fn import add_loss_fn_arguments
from corenet.metrics import METRICS_REGISTRY, arguments_stats
from corenet.modeling import modeling_arguments
from corenet.optims import arguments_optimizer
from corenet.optims.scheduler import arguments_scheduler
from corenet.options.parse_args import JsonValidator
from corenet.options.utils import load_config_file
from corenet.utils import logger


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # convert values into dict
        override_dict = {}
        for val in values:
            if val.find("=") < 0:
                logger.error(
                    "For override arguments, a key-value pair of the form key=value is expected. Got: {}".format(
                        val
                    )
                )
            val_list = val.split("=")
            if len(val_list) != 2:
                logger.error(
                    "For override arguments, a key-value pair of the form key=value is expected with only one value per key. Got: {}".format(
                        val
                    )
                )
            override_dict[val_list[0]] = val_list[1]

        # determine the type of each value from parser actions and set accordingly
        options = parser._actions
        for option in options:
            option_dest = option.dest
            if option_dest in override_dict:
                val = override_dict[option_dest]
                if type(option.default) == bool and option.nargs == 0:
                    # Boolean argument
                    # value could be false, False, true, True
                    override_dict[option_dest] = (
                        True if val.lower().find("true") > -1 else False
                    )
                elif option.nargs is None:
                    # when nargs is not defined, it is usually a string, int, and float.
                    override_dict[option_dest] = option.type(val)
                elif option.nargs in ["+", "*"]:
                    # for list, we expect value to be comma separated
                    val_list = val.split(",")
                    override_dict[option_dest] = [option.type(v) for v in val_list]
                else:
                    logger.error(
                        "Following option is not yet supported for overriding. Please specify in config file. Got: {}".format(
                            option
                        )
                    )
        setattr(namespace, "override_args", override_dict)


def arguments_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="Common arguments", description="Common arguments"
    )

    group.add_argument(
        "--taskname",
        type=str,
        default="",
        help="Name of the task (can have arbitrary values)",
    )
    group.add_argument("--common.seed", type=int, default=0, help="Random seed")
    group.add_argument(
        "--common.config-file", type=str, default=None, help="Configuration file"
    )
    group.add_argument(
        "--common.results-loc",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory where results will be stored. Defaults to {DEFAULT_RESULTS_DIR}.",
    )
    group.add_argument(
        "--common.logs-loc",
        type=str,
        default=DEFAULT_LOGS_DIR,
        help=f"Directory where logs will be stored. Defaults to {DEFAULT_LOGS_DIR}.",
    )
    group.add_argument(
        "--common.run-label",
        type=str,
        default="run_1",
        help="Label id for the current run",
    )
    group.add_argument(
        "--common.eval-stage-name",
        type=str,
        default="evaluation",
        help="Name to be used while logging in evaluation stage.",
    )

    group.add_argument(
        "--common.resume", type=str, default=None, help="Resume location"
    )
    group.add_argument(
        "--common.finetune",
        type=str,
        default=None,
        help="Checkpoint location to be used for finetuning",
    )
    group.add_argument(
        "--common.finetune-ema",
        type=str,
        default=None,
        help="EMA Checkpoint location to be used for finetuning",
    )

    group.add_argument(
        "--common.mixed-precision",
        action="store_true",
        help="Enable mixed precision training. Defaults to False. \
            Note that this argument is not applicable for FSDP training. For mixed precision training with FSDP, \
            please see respective arguments in FSDP.",
    )
    group.add_argument(
        "--common.mixed-precision-dtype",
        type=str,
        default="float16",
        help="Mixed precision training data type",
    )
    group.add_argument(
        "--common.accum-freq",
        type=int,
        default=1,
        help="Accumulate gradients for this number of iterations",
    )
    group.add_argument(
        "--common.accum-after-epoch",
        type=int,
        default=0,
        help="Start accumulation after this many epochs",
    )
    group.add_argument(
        "--common.log-freq",
        type=int,
        default=100,
        help="Display after these many iterations",
    )
    group.add_argument(
        "--common.auto-resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    group.add_argument(
        "--common.grad-clip", type=float, default=None, help="Gradient clipping value"
    )
    group.add_argument(
        "--common.k-best-checkpoints",
        type=int,
        default=5,
        help="Keep k-best checkpoints",
    )
    group.add_argument(
        "--common.save-all-checkpoints",
        action="store_true",
        default=False,
        help="If True, will save checkpoints from all epochs",
    )

    group.add_argument(
        "--common.channels-last",
        action="store_true",
        default=False,
        help="Use channel last format during training. "
        "Note that some models may not support it, so we recommend to use it with caution.",
    )

    group.add_argument(
        "--common.tensorboard-logging",
        action="store_true",
        help="Enable tensorboard logging",
    )
    group.add_argument(
        "--common.file-logging", action="store_true", help="Enable file logging."
    )
    group.add_argument(
        "--common.override-kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Override arguments. Example. To override the value of --sampler.vbs.crop-size-width, "
        "we can pass override argument as "
        "--common.override-kwargs sampler.vbs.crop_size_width=512 \n "
        "Note that keys in override arguments do not contain -- or -",
    )

    group.add_argument(
        "--common.enable-coreml-compatible-module",
        action="store_true",
        help="Use coreml compatible modules (if applicable) during inference",
    )

    group.add_argument(
        "--common.debug-mode",
        action="store_true",
        help="You can use this flag for debugging purposes.",
    )

    # intermediate checkpoint related args
    group.add_argument(
        "--common.save-interval-freq",
        type=int,
        default=0,
        help="Save checkpoints every N updates. Defaults to 0",
    )

    group.add_argument(
        "--common.eval-every-k-iterations",
        type=int,
        default=0,
        help="Evaluate model every k iterations. Defaults to 0.",
    )
    group.add_argument(
        "--common.set-grad-to-none",
        action="store_true",
        help="Set gradients to none instead of zero after optimization step. This can help in reducing \
            GPU memory usage and can moderately improve training speed. Defaults to False. \
            Please be cautious when computing grad_norm for debugging purposes.",
    )

    try:
        from corenet.internal.utils.opts import arguments_internal

        parser = arguments_internal(parser=parser)
    except ModuleNotFoundError:
        logger.debug("Cannot load internal arguments, skipping.")

    return parser


def arguments_ddp(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="DDP arguments")
    group.add_argument(
        "--ddp.rank",
        type=int,
        default=0,
        help="Node rank for distributed training. Defaults to 0.",
    )
    group.add_argument(
        "--ddp.world-size",
        type=int,
        default=-1,
        help="World size for DDP. Defaults to -1, meaning use all GPUs.",
    )
    group.add_argument(
        "--ddp.dist-url", type=str, default=None, help="DDP URL. Defaults to None."
    )
    group.add_argument(
        "--ddp.dist-port",
        type=int,
        default=30786,
        help="DDP Port. Only used when --ddp.dist-url is not specified. Defaults to 30768.",
    )
    group.add_argument(
        "--ddp.device-id", type=int, default=None, help="Device ID. Defaults to None."
    )
    group.add_argument(
        "--ddp.backend", type=str, default="nccl", help="DDP backend. Default is nccl"
    )
    group.add_argument(
        "--ddp.find-unused-params",
        action="store_true",
        default=False,
        help="Find unused params in model. useful for debugging with DDP. Defaults to False.",
    )

    group.add_argument(
        "--ddp.use-deprecated-data-parallel",
        action="store_true",
        default=False,
        help="Use Data parallel for training. This flag is not recommended for training and should be used only for debugging. \
            The support for this flag will be deprecating in future.",
    )

    return parser


def arguments_train_eval_pipeline(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="TrainEvalPipeline")
    group.add_argument(
        "--train-eval-pipeline.name",
        type=str,
        default="default",
        help=(
            "Name of the TrainEvalPipeline to use for constructing pipeline components."
            " Defaults to 'default' pipeline (see corenet/train_eval_pipelines/train_eval.py)"
        ),
    )
    return parser


def arguments_lm_eval(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add arguments related to 3rd party LM evaluation.

    Args:
        parser: The parser to add arguments to.
    Returns:
        The updated parser.
    """
    group = parser.add_argument_group("LM Evaluation arguments")
    group.add_argument(
        "--lm-eval-wrapper.add-sot-token",
        action="store_true",
        default=False,
        help="Certain tokenizers (e.g., Llamav1 Tokenizer) adds start of text token. However, by default, \
             some libraries set it to false, leading to inconsistent results. This flags allows to control if start of \
            text token to be added to input sentence or not. ",
    )
    group.add_argument(
        "--llmadapters-evaluation.datasets",
        type=str,
        default=(
            "boolq",
            "piqa",
            "social_i_qa",
            "hellaswag",
            "winogrande",
            "ARC-Easy",
            "ARC-Challenge",
            "openbookqa",
        ),
        help="The commonsense datasets on which to evaluate. Options include boolq, piqa, social_i_qa, hellaswag, winogrande, ARC-Easy, ARC-Challenge, and openbookqa.",
        nargs="+",
        choices=(
            "boolq",
            "piqa",
            "social_i_qa",
            "hellaswag",
            "winogrande",
            "ARC-Easy",
            "ARC-Challenge",
            "openbookqa",
        ),
    )
    group.add_argument(
        "--llmadapters-evaluation.multiple-choice",
        action="store_true",
        help="If set, do multiple-choice evaluation. Otherwise, use generation-style evaluation.",
    )
    group.add_argument(
        "--llmadapters-evaluation.limit",
        type=int,
        help="If set, run a limited number of evaluation samples.",
        default=None,
    )
    group.add_argument(
        "--llmadapters-evaluation.dataset-dir",
        type=str,
        help="Directory that contains the datasets on which to evaluate.",
        default="LLM-Adapters/dataset/",
    )
    group.add_argument(
        "--lm-harness-evaluation.tasks-info",
        type=JsonValidator,
        help="Task dictionary for LM Harness evaluation.",
        default=None,
    )

    return parser


def parser_to_opts(parser: argparse.ArgumentParser, args: Optional[List[str]] = None):
    # parse args
    opts = parser.parse_args(args)
    opts = load_config_file(opts)
    return opts


def get_training_arguments(
    args: Optional[List[str]] = None,
    parse_args: bool = True,
    add_arguments: Optional[
        Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
    ] = None,
) -> Union[argparse.ArgumentParser, argparse.Namespace]:
    """Adds the CoreNet training arguments to the argument parser.

    Args:
        args: If provided, argparser ignores the CLI arguments and parses the given
            list. Defaults to None, that parses CLI arguments from `sys.argv`.
        parse_args: If true, parses the arguments. Otherwise, just creates the parser.
            Defaults to True.
        add_arguments: If provided, wraps the argument parser to modify the parser or
            to add additional arguments dynamically. Useful for entrypoint-specific
            arguments. Defaults to None.

    Returns:
        By default, returns the `opts` config object. If parse_args=True is passed,
        returns the argument parser.
    """
    parser = argparse.ArgumentParser(description="Training arguments", add_help=True)

    # transfer client related arguments
    parser = transfer_client_arguments(parser)

    # dataset related arguments
    parser = arguments_dataset(parser=parser)

    # cvnet arguments, including models
    parser = modeling_arguments(parser=parser)

    # sampler related arguments
    parser = add_sampler_arguments(parser=parser)

    # collate fn  related arguments
    parser = arguments_collate_fn(parser=parser)

    # transform related arguments
    parser = arguments_augmentation(parser=parser)

    # Video reader related arguments
    # Should appear after arguments_augmentations(parser=parser) because "--frame-augmentation.*" depends on "--image-augmentation.*"
    parser = arguments_video_reader(parser=parser)

    # loss function arguments
    parser = add_loss_fn_arguments(parser=parser)

    # optimizer arguments
    parser = arguments_optimizer(parser=parser)
    parser = arguments_scheduler(parser=parser)

    # DDP arguments
    parser = arguments_ddp(parser=parser)

    # stats arguments
    parser = arguments_stats(parser=parser)

    # common
    parser = arguments_common(parser=parser)

    # text tokenizer arguments
    parser = arguments_tokenizer(parser=parser)

    # metric arguments
    parser = METRICS_REGISTRY.all_arguments(parser=parser)

    parser = arguments_train_eval_pipeline(parser=parser)

    if add_arguments is not None:
        parser = add_arguments(parser)

    if parse_args:
        return parser_to_opts(parser, args)
    else:
        return parser


def get_lm_eval_arguments(
    parse_args: Optional[bool] = True, args: Optional[List[str]] = None
) -> argparse.ArgumentParser:
    parser = get_training_arguments(
        parse_args=False,
        args=args,
    )
    parser = arguments_lm_eval(parser)

    if parse_args:
        return parser_to_opts(parser, args)
    else:
        return parser


def get_conversion_arguments(args: Optional[List[str]] = None):
    parser = get_training_arguments(parse_args=False)

    # Arguments related to coreml conversion
    group = parser.add_argument_group("Conversion arguments")
    group.add_argument(
        "--conversion.coreml-extn",
        type=str,
        default="mlmodel",
        help="Extension for converted model. Default is mlmodel",
    )
    group.add_argument(
        "--conversion.input-image-path",
        type=str,
        default=None,
        help="Path of the image to be used for conversion",
    )

    group.add_argument(
        "--conversion.minimum-deployment-target",
        type=str,
        default=None,
        choices=list([target.name for target in ct.target]),
        help="A member of the coremltools.target enum. Defaults to None",
    )
    group.add_argument(
        "--conversion.compute-precision",
        type=str,
        default=None,
        choices=list([precision.name for precision in ct.precision]),
        help="A member of the coremltools.precision enum. Defaults to None",
    )

    # Arguments related to server.
    group.add_argument(
        "--conversion.bucket-name", type=str, help="Model job's bucket name"
    )
    group.add_argument("--conversion.task-id", type=str, help="Model job's id")
    group.add_argument(
        "--conversion.viewers",
        type=str,
        nargs="+",
        default=None,
        help="Users who can view your models on server",
    )

    # parse args
    return parser_to_opts(parser, args=args)


def get_benchmarking_arguments(args: Optional[List[str]] = None):
    parser = get_training_arguments(parse_args=False)

    #
    group = parser.add_argument_group("Benchmarking arguments")
    group.add_argument(
        "--benchmark.batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    group.add_argument(
        "--benchmark.warmup-iter", type=int, default=10, help="Warm-up iterations"
    )
    group.add_argument(
        "--benchmark.n-iter",
        type=int,
        default=100,
        help="Number of iterations for benchmarking",
    )
    group.add_argument(
        "--benchmark.use-jit-model",
        action="store_true",
        help="Convert the model to JIT and then benchmark it",
    )

    # parse args
    return parser_to_opts(parser, args=args)
