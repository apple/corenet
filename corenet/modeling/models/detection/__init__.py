#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from collections import namedtuple

DetectionPredTuple = namedtuple(
    typename="DetectionPredTuple",
    field_names=("labels", "scores", "boxes", "masks"),
    defaults=(None, None, None, None),
)
