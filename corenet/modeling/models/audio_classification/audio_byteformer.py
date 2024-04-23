#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Union

from torch import Tensor

from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.audio_classification.base_audio_classification import (
    BaseAudioClassification,
)
from corenet.modeling.models.classification.byteformer import ByteFormer


@MODEL_REGISTRY.register(name="byteformer", type="audio_classification")
class AudioByteFormer(ByteFormer, BaseAudioClassification):
    """Identical to byteformer.ByteFormer, but registered as an audio classification
    model."""

    def forward(self, x: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        """
        Perform a forward pass on input bytes. The input is a dictionary
        containing the input tensor. The tensor is stored as an integer tensor
        of shape [batch_size, sequence_length]. Integer tensors are used because
        the tensor usually contains mask tokens.

        Args:
            x: A dictionary containing {"audio": audio_bytes}.

        Returns:
            The output logits.
        """
        return super().forward(x["audio"], *args, **kwargs)

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """
        Get a dummy input and label that could be passed to the model.

        Args:
            batch_size: The batch size to use for the generated inputs.

        Returns:
            A dict with
                {
                    "samples": {"audio": tensor of shape [batch_size, sequence_length]},
                    "targets": tensor of shape [batch_size],
                }
        """
        input_and_label = super().dummy_input_and_label(batch_size)

        ret = {
            "samples": {"audio": input_and_label["samples"]},
            "targets": input_and_label["targets"],
        }
        return ret
