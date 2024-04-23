#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.modeling.models.audio_classification import base_audio_classification


def test_base_audio_classification_adds_arguments() -> None:
    opts = argparse.Namespace()
    model = base_audio_classification.BaseAudioClassification(opts)

    parser = argparse.ArgumentParser()
    model.add_arguments(parser)
    assert hasattr(parser.parse_args([]), "model.audio_classification.name")
