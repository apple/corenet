#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from corenet.modeling.models import get_model

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ModuleNotFoundError:
    PretrainedConfig = object
    PreTrainedModel = object
    CausalLMOutputWithPast = None


class CorenetToHFPretrainedConfig(PretrainedConfig):
    """
    An adapter to build a CoreNet config that inherits from
    PreTrainedConfig.

    Mainly used for adaptation to 3rd party code.

    Args:
        kwargs: Arguments to pass to PretrainedConfig.
    """

    model_type = "causal_lm"

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)


class CorenetToHFPretrainedModel(PreTrainedModel):
    """
    An adapter to build a CoreNet model that inherits from
    PreTrainedModel.

    Mainly used for adaptation to 3rd party code.

    Args:
        config: The _CorenetToHFPretrainedConfig
            that defines the model. This essentially
            contains the standard CoreNet model arguments.
        vocab_size: The vocabulary size.
    """

    config_class = CorenetToHFPretrainedConfig

    def __init__(self, config: CorenetToHFPretrainedConfig, vocab_size: int) -> None:
        super().__init__(config)
        opts = argparse.Namespace(**vars(config))

        model = get_model(opts)
        model.eval()

        self.lm_head = None
        self.model = model
        self.vocab_size = vocab_size
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[
            torch.FloatTensor
        ] = None,  # No need, internally apply causal masking
        token_type_ids: Optional[
            torch.LongTensor
        ] = None,  # No need, we do not differentiate between tokens
        position_ids: Optional[
            torch.LongTensor
        ] = None,  # Needed for llama mdoels, not needed for openelm
        head_mask: Optional[
            torch.FloatTensor
        ] = None,  # No need, we are not masking heads
        inputs_embeds: Optional[
            torch.FloatTensor
        ] = None,  # No need, they will be computed internally
        encoder_hidden_states: Optional[
            torch.Tensor
        ] = None,  # No need, we don't have encoder
        encoder_attention_mask: Optional[
            torch.FloatTensor
        ] = None,  # No need, we don't have encoder
        use_cache: Optional[
            bool
        ] = None,  # We can use it, map to `use_kv_cache` args internally
        output_attentions: Optional[bool] = None,  # No need, we are not using them
        output_hidden_states: Optional[bool] = None,  # No need, we are not using them
        return_dict: Optional[
            bool
        ] = None,  # We can return as a dict, we internally deal with it
        cache_position: Optional[torch.LongTensor] = None,  # No need, not used.
    ) -> CausalLMOutputWithPast:
        """
        The forward function to compute model outputs.

        Note, many arguments are not supported, and are only
        present due to inheritance. Additionally, the model is
        assumed to use causal attention.

        Args:
            input_ids: The input token ids.
            past_key_values: The key-value cache.
            attention_mask: Unused.
            token_type_ids: Unused.
            position_ids: Unused.
            head_mask: Unused.
            inputs_embeds: Unused.
            encoder_hidden_states: Unused.
            encoder_attention_mask: Unused.
            use_cache: Whether we are going to use the cache or not.
            output_attentions: Unused.
            output_hidden_states: Unused.
            return_dict: Unused.
            cache_position: Unused
        """
        assert token_type_ids is None
        # assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None
        assert encoder_hidden_states is None
        assert encoder_attention_mask is None
        assert not output_attentions
        assert not output_hidden_states
        assert return_dict is None or return_dict

        is_causal = True
        if use_cache:
            # For generative tasks, we have two scenarios: (1) processing prefix and (2) generation
            # For the first case, @n_input_tokens > 1 and we do not have past_key_values. Therefore,
            # we need to run model with causal mask
            # For the second case, @n_input_tokens should be 1. In this case, we have already processed
            # prefix and will use it's cached keys and values for the current token.

            # input_ids are of the shape [Batch, Seq_length]
            n_input_tokens = input_ids.shape[1]

            if n_input_tokens == 1 and past_key_values is not None:
                is_causal = False
            elif n_input_tokens > 1 and past_key_values is None:
                is_causal = True
            else:
                raise NotImplementedError("Not yet supported")
            if past_key_values is None:
                past_keys = None
                past_values = None
            else:
                past_keys, past_values = past_key_values

            model_inputs = {
                "input_ids": input_ids,
                "past_keys": past_keys,
                "past_values": past_values,
                "use_kv_cache": use_cache,
                "is_causal": is_causal,
            }
        else:
            model_inputs = input_ids

        model_outputs = self.model(model_inputs)
        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
            past_key_values = model_outputs["past_keys"], model_outputs["past_values"]
        elif isinstance(model_outputs, torch.Tensor):
            logits = model_outputs
            past_key_values = None

        # We may have added extra tokens in the LM classifier to make training faster. Remove such tokens
        logits = logits[..., : self.vocab_size]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Prepare inputs to be passed to the model by building a dictionary of
        arguments from the inputs.

        Args:
            input_ids: The input ids.
            past_key_values: The key-value cache.
            use_cache: If set, use the key-value cache.
            attention_mask: The attention mask to use in attention layers.

            Other keyword arguments are ignored.
        """
        if past_key_values is not None:
            # All tokens except the last token were processed in the previous time step.
            # so, we do not need to process them again.
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
        return model_inputs
