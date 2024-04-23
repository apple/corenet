#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.modeling.anchor_generator import arguments_anchor_gen
from corenet.modeling.image_projection_layers import arguments_image_projection_head
from corenet.modeling.layers import arguments_nn_layers
from corenet.modeling.matcher_det import arguments_box_matcher
from corenet.modeling.misc.averaging_utils import EMA, arguments_ema
from corenet.modeling.misc.common import parameter_list
from corenet.modeling.models import arguments_model, get_model
from corenet.modeling.models.detection import DetectionPredTuple
from corenet.modeling.neural_augmentor import arguments_neural_augmentor
from corenet.modeling.text_encoders import arguments_text_encoder
from corenet.options.utils import extend_selected_args_with_prefix


def modeling_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # text encoder arguments (usually for multi-modal tasks)
    parser = arguments_text_encoder(parser)
    # image projection head arguments (usually for multi-modal tasks)
    parser = arguments_image_projection_head(parser)
    # model arguments
    parser = arguments_model(parser)
    # neural network layer argumetns
    parser = arguments_nn_layers(parser)
    # EMA arguments
    parser = arguments_ema(parser)
    # anchor generator arguments (for object detection)
    parser = arguments_anchor_gen(parser)
    # box matcher arguments (for object detection)
    parser = arguments_box_matcher(parser)
    # neural aug arguments
    parser = arguments_neural_augmentor(parser)

    # Add teacher as a prefix to enable distillation tasks
    # keep it as the last entry
    parser = extend_selected_args_with_prefix(
        parser, match_prefix="--model.", additional_prefix="--teacher.model."
    )

    return parser
