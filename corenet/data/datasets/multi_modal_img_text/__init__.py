#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.datasets.multi_modal_img_text.base_multi_modal_img_text import (
    BaseMultiModalImgText,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification import (
    arguments_zero_shot_image_classification_dataset,
)


def arguments_multi_modal_img_text(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:

    parser = arguments_zero_shot_image_classification_dataset(parser)
    parser = BaseMultiModalImgText.add_arguments(parser)
    return parser
