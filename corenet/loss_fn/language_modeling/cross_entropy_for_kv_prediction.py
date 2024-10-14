#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria
from corenet.utils import logger


@LOSS_REGISTRY.register(
    name="cross_entropy_for_kv_prediction", type="language_modeling"
)
class CrossEntropyForKVPrediction(BaseCriteria):
    """Cross entropy loss function for language modeling tasks.

    Used in KV Prediction experiments (https://arxiv.org/abs/2410.08391).

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace) -> None:
        super().__init__(opts)

        self.ignore_idx = getattr(
            opts, "loss.language_modeling.cross_entropy_for_kv_prediction.ignore_index"
        )
        if self.training:
            self.label_smoothing = getattr(
                opts,
                "loss.language_modeling.cross_entropy_for_kv_prediction.label_smoothing",
            )
        else:
            # for validation/test sets, we compute standard CE loss
            self.label_smoothing = 0.0

        self.use_z_loss = getattr(
            opts, "loss.language_modeling.cross_entropy_for_kv_prediction.use_z_loss"
        )
        self.z_loss_eps = getattr(
            opts, "loss.language_modeling.cross_entropy_for_kv_prediction.z_loss_eps"
        )
        self.auxiliary_loss = getattr(
            opts,
            "loss.language_modeling.cross_entropy_for_kv_prediction.auxiliary_loss",
        )
        self.base_loss = getattr(
            opts, "loss.language_modeling.cross_entropy_for_kv_prediction.base_loss"
        )
        self.kv_loss = getattr(
            opts, "loss.language_modeling.cross_entropy_for_kv_prediction.kv_loss"
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add cross-entropy criteria-specific arguments to the parser."""
        if cls == CrossEntropyForKVPrediction:
            group = parser.add_argument_group(title=cls.__name__)
            group.add_argument(
                "--loss.language-modeling.cross-entropy-for-kv-prediction.ignore-index",
                type=int,
                default=-1,
                help=f"Target value that is ignored and does not contribute to "
                f"the input gradient in {cls.__name__}. Defaults to -1.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy-for-kv-prediction.label-smoothing",
                type=float,
                default=0.0,
                help=f"Specifies the amount of smoothing when computing the loss in {cls.__name__}, "
                f"where 0.0 means no smoothing. Defaults to 0.0.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy-for-kv-prediction.use-z-loss",
                action="store_true",
                default=False,
                help="Use z-loss with cross-entropy loss. Defaults to False.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy-for-kv-prediction.z-loss-eps",
                default=1.0e-4,
                type=float,
                help="Epsilon value for z-loss. Defaults to 0.0001.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy-for-kv-prediction.auxiliary-loss",
                default=0.0,
                type=float,
                help="Multiplicative factor for auxiliary loss.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy-for-kv-prediction.base-loss",
                default=1.0,
                type=float,
                help="Multiplicative factor for base loss.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy-for-kv-prediction.kv-loss",
                default=0.0,
                type=float,
                help="Multiplicative factor for KV loss.",
            )

        return parser

    def forward(
        self,
        input_sample: Any,
        prediction: Union[Dict[str, Tensor], Tensor],
        target: Tensor,
        epoch: Optional[int] = None,
        iterations: Optional[int] = None,
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Computes the loss.

        Args:
            input_sample: Input samples to model.
            prediction: Output of model. It can be a tensor or mapping of (string: Tensor). In case of mapping,
            `logits` is a required key.
            target: Target label tensor containing values in the range `[0, vocabulary size)`.
            epoch: Training epoch.
            iterations: Training iteration.

        Shapes:
            input_sample: This loss function does not care about this argument.
            prediction:
                * When prediction is a tensor, then shape is [batch size, sequence length, vocabulary size]
                * When prediction is a dictionary, then the shape of prediction["logits"] is [batch size, sequence length, vocabulary size]
            target: The shape of target tensor is [batch size, sequence length]

        Returns:
            Either of the following is returned as an output:
            1. 0-dimensional tensor containing the scalar loss value.
            2. Mapping of the form (string: 0-dimensional tensor) is returned with 'total_loss' as
                a mandatory key.

        ...note:
            While epoch and iteration values are currently not utilized in language modeling loss functions, they may be
            incorporated in future developments or research.
        """
        if isinstance(prediction, Tensor):
            raise ValueError(f"This loss requires a dictionary.")
        elif isinstance(prediction, Dict):
            expected_keys = {
                "auxiliary_logits",
                "past_keys",
                "past_values",
                "base_past_keys",
                "base_past_values",
            }
            if not expected_keys.issubset(prediction):
                logger.error(
                    f"Expected keys {expected_keys=}." f"Got keys {prediction.keys()=}"
                )

            loss = self._compute_loss(
                prediction=prediction["logits"],
                target=target,
                auxiliary=prediction["auxiliary_logits"],
                past_keys=prediction["past_keys"],
                past_values=prediction["past_values"],
                base_past_keys=prediction["base_past_keys"],
                base_past_values=prediction["base_past_values"],
            )

            return loss
        else:
            logger.error(
                f"Prediction should be either a Tensor or Dictionary[str, Tensor]. Got: {type(prediction)}"
            )

    def _compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        auxiliary: Tensor,
        past_keys: List[Tensor],
        past_values: List[Tensor],
        base_past_keys: List[Tensor],
        base_past_values: List[Tensor],
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Computes cross-entropy loss between prediction and target tensors.

        Args:
            prediction: Predicted tensor of shape [batch size, sequence length, vocabulary size].
            target: Target label tensor containing values in the range `[0, vocabulary size)`. The shape of tensor is
                [batch size, sequence length].
            auxiliary: The Auxiliary model's logits. A tensor of shape [batch size, sequence length, vocabulary size].
            past_keys: The keys from the KV cache generated by the KV predicter.
            past_values: The values from the KV cache generated by the KV predicter.
            base_past_keys: The keys from the KV cache generated by the Base model.
            base_past_values: The values from the KV cache generated by the Base model.

        Returns:
            A dictionary with total_loss, base_loss, auxiliary_loss, and the key/value cache
            prediction losses at each layer, 'k_loss/i' and 'v_loss/i' (if the key/value
            cache loss is activated).
        """
        batch_size, seq_length, vocab_size = prediction.shape
        prediction = prediction.reshape(batch_size * seq_length, vocab_size)
        target = target.reshape(batch_size * seq_length)
        ce_loss = F.cross_entropy(
            input=prediction,
            target=target,
            ignore_index=self.ignore_idx,
            label_smoothing=self.label_smoothing,
        )
        ret = {"base_loss": ce_loss}

        if auxiliary is not None:
            batch_size, seq_length, vocab_size = auxiliary.shape
            auxiliary = auxiliary.reshape(batch_size * seq_length, vocab_size)
            auxiliary_loss = F.cross_entropy(
                input=auxiliary,
                target=target,
                ignore_index=self.ignore_idx,
                label_smoothing=self.label_smoothing,
            )
            ret["auxiliary_loss"] = auxiliary_loss

        if self.use_z_loss:

            def get_zloss(x):
                # Adaption of Eq. (5) for z-loss computation in https://arxiv.org/pdf/2202.08906.pdf (non-router use-case).
                valid_tokens = (target != self.ignore_idx).type_as(x)
                # do not compute z_loss for ignored indices
                z_loss = (x * valid_tokens[:, None]).logsumexp(-1).pow(2).sum()
                z_loss *= self.z_loss_eps / valid_tokens.sum()
                return z_loss

            ret["z_loss_base"] = get_zloss(prediction)
            if auxiliary is not None:
                ret["z_loss_auxiliary"] = get_zloss(auxiliary)

        if self.kv_loss > 0:
            ignore_positions = target.view(batch_size, seq_length) == self.ignore_idx
            k_losses = get_cache_losses(
                base_past_keys,
                past_keys,
                ignore_positions=ignore_positions,
            )
            total_k_loss = 0
            for i, k_loss in enumerate(k_losses):
                ret[f"k_loss/{i}"] = k_loss
                total_k_loss += k_loss
            ret["k_loss/total"] = total_k_loss
            ret["k_loss/average"] = total_k_loss / len(k_losses)

            v_losses = get_cache_losses(
                base_past_values,
                past_values,
                ignore_positions=ignore_positions,
            )
            total_v_loss = 0
            for i, v_loss in enumerate(v_losses):
                ret[f"v_loss/{i}"] = v_loss
                total_v_loss += v_loss
            ret["v_loss/total"] = total_v_loss
            ret["v_loss/average"] = total_v_loss / len(v_losses)

        ret["total_loss"] = (
            (ret["base_loss"] + ret.get("z_loss_base", 0)) * self.base_loss
            + (ret.get("auxiliary_loss", 0) + ret.get("z_loss_auxiliary", 0))
            * self.auxiliary_loss
            + (ret.get("k_loss/total", 0) + ret.get("v_loss/total", 0)) * self.kv_loss
        )
        return ret

    def extra_repr(self) -> str:
        loss_info_str = (
            f"\n\t ignore_idx={self.ignore_idx}"
            f"\n\t label_smoothing={self.label_smoothing}"
        )
        if self.use_z_loss:
            loss_info_str += (
                f"\n\t use_z_loss={self.use_z_loss}"
                f"\n\t z_loss_eps={self.z_loss_eps}"
            )
        if self.kv_loss:
            loss_info_str += f"\n\t kv_loss={self.kv_loss}"

        return loss_info_str


def get_cache_losses(
    base_cache: List[Tensor],
    predicted_cache: List[Tensor],
    ignore_positions=None,
) -> List[Tensor]:
    """
    Compute the loss between two caches.

    Args:
        base_cache: The cache from the base model, of shape [[batch_size, num_heads, seq_len, head_dim] * num_layers]
        predicted_cache: The predicted cache, of shape [[batch_size, num_heads, seq_len, head_dim] * num_layers]
        ignore_positions: A [batch_size, seq_len] tensor with positions to ignore.
    Returns:
        `list(torch.Tensor)` a list of losses.
    """
    cache_losses = []
    for layer in range(len(base_cache)):
        target = base_cache[layer]
        prediction = predicted_cache[layer]

        assert target.dim() == 4
        assert prediction.dim() == 4
        batch_size, num_heads, seq_len, dim = target.shape
        assert prediction.shape == target.shape

        if ignore_positions is not None:
            keep_positions = (
                ignore_positions.logical_not()
                .view(batch_size, 1, seq_len, 1)
                .expand(-1, prediction.shape[1], -1, prediction.shape[3])
            )
            target = target[keep_positions]
            prediction = prediction[keep_positions]
        loss = F.l1_loss(prediction, target).abs()
        cache_losses.append(loss)
    return cache_losses
