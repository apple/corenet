#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""
LoRA and DoRA models for parameter-efficient finetuning.

Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
"""
from __future__ import annotations

import math
import re
import warnings
from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from corenet.modeling.layers import embedding, linear_layer


def check_adapters_to_merge(
    module: BaseTunerLayer, adapter_names: Optional[List[str]] = None
) -> List[str]:
    """
    Helper function to check which adapters should be merged.

    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.

    Args:
        module: The module in which to check the adapters.
        adapter_names: The names of the adapters to check.

    Returns:
        A list of adapters that are not merged.
    """
    if adapter_names is None:
        adapter_names = module.active_adapters

    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        if adapter_names:
            warnings.warn(
                f"Already following adapters were merged {','.join(module.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names


class BaseTunerLayer(ABC):
    """
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_pluggable:
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters:
            The name of the active adapter.
    """

    active_adapter = None

    # All names of layers that may contain adapter (trainable) weights.
    adapter_layer_names: Tuple[str, ...] = ()
    # All names of other parameters that may contain adapter-related parameters.
    other_param_names: Tuple[str, ...] = ()

    # Indicates whether all adapters should be disabled.
    _disable_adapters: bool = False

    # The currently active adapter(s).
    _active_adapter: Union[List[str], str] = "default"

    # All merged adapter(s).
    merged_adapters: List[str] = []

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer that the adapter wraps.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        Returns:
            The base layer.
        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    @property
    def weight(self) -> torch.Tensor:
        """
        Get the weight of the base layer that this adapter.

        Returns:
            The base layer's weight.
        """
        return self.get_base_layer().weight

    @property
    def bias(self) -> torch.Tensor:
        """
        Get the bias of the base layer that this adapter.

        Returns:
            The base layer's bias.
        """
        return self.get_base_layer().bias

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights.

        Args:
            safe_merge: If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names: The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        raise NotImplementedError

    def unmerge(self) -> None:
        """
        Unmerge the adapters.
        """
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        """
        Check whether the adapters are merged.

        Returns:
            True if the adapters are merged.
        """
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        """
        Check if all adapters are disabled.

        Returns:
            True if the adapters are all disabled.
        """
        # Use a property to ensure that disable_adapters is not set directly.
        # See also @enable_adapters.
        return self._disable_adapters

    @property
    def active_adapter(self) -> Union[List[str], str]:
        """
        Get the active adapter.

        Returns:
            The name of the active adapter.
        """
        # Use a property to ensure that active_adapter is not set directly.
        # See also @set_adapter.
        return self._active_adapter

    @property
    def active_adapters(self) -> List[str]:
        """
        Get all active adapters.

        Returns:
            A list of active adapters.
        """
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled: If True, adapters will be enabled. If False, they will be disabled.
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: Union[str, List[str]]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_names: Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter.
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> List[str]:
        """
        Return a sorted list of all available adapter names.

        Returns:
            A sorted list of adapter names.
        """
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # We check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name: The name of the adapter to delete.
        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # Choose a new active adapter.
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # No active adapters left, set a new default adapter. To do so,
                # get the list of all adapters existing adapter names and
                # choose the first one.
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])


class LoraLayer(BaseTunerLayer):
    """A LoRA (https://arxiv.org/abs/2106.09685) layer.

    See also DoRA, an extension of LoRA: https://arxiv.org/abs/2402.09353.

    As per the HuggingFace API, this layer can contain multiple different adapters.

    Args:
        base_layer: The layer to wrap with LoRA.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layers.
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged.
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
        self._caches: dict[str, Any] = {}

        base_layer = self.get_base_layer()
        if isinstance(base_layer, linear_layer.LinearLayer):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, embedding.Embedding):
            in_features, out_features = (
                base_layer.num_embeddings,
                base_layer.embedding_dim,
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: Union[str, bool],
        use_rslora: bool,
        use_dora: bool = False,
    ) -> None:
        """
        Create an adapter with the given name.

        Args:
            adapter_name: The name of the new adapter.
            r: The LoRA rank.
            lora_alpha: The LoRA alpha parameter.
            lora_dropout: The dropout to use with the LoRA module.
            init_lora_weights: If a string, it specifies the style of initialization. If a bool value, it specifies
                whether to initialize the weights with Kaiming uniform, or whether to skip initialization.
        """
        raise NotImplementedError(f"Implement @update_layer in the base class.")

    def reset_lora_parameters(
        self, adapter_name: str, init_lora_weights: Union[str, bool]
    ) -> None:
        """
        Reset the LoRA parameters.

        Args:
            adapter_name: The adapter on which to reset the parameters.
            init_lora_weights: If a string, it specifies the style of initialization. If a bool value, it specifies
                whether to initialize the weights with Kaiming uniform, or whether to skip initialization.
        """
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(
                    self.lora_A[adapter_name].weight, a=math.sqrt(5)
                )
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(
                    self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name]
                )
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def _get_weight_norm(
        self, weight: torch.Tensor, lora_weight: torch.Tensor, scaling: float
    ) -> torch.Tensor:
        """
        Calculate L2 norm of the combination of @weight and
        @lora_weight * @scaling. The norm is taken across the
        dimension with index 1.

        No specific shape is required, but normally, the dimension with
        index 1 will be the "input_dim" dimension of the weight tensor.

        Args:
            weight: The matrix whose norm to calculate.
            lora_weight: The LoRA weight to add to the weight matrix.
            scaling: The scaling to apply to the LoRA weight.

        Returns:
            The norm of the dimension with index 1 (typically the input
            feature dimension).
        """
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def dora_init(self, adapter_name: str) -> None:
        """
        Initialize the adapter with DoRA (https://arxiv.org/abs/2402.09353).

        Args:
            adapter_name: The name of the adapter to initialize.
        """
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        scaling = self.scaling[adapter_name]
        weight = self.weight
        if weight.data.ndim == 4:  # For handling LoRAs applied to Conv2Ds.
            # Leave this check to make it clear that this codepath needs to
            # be edited if Conv2d support is added.
            raise NotImplementedError(f"Conv2d is not currently supported.")
        else:
            lora_weight = lora_B.weight @ lora_A.weight
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)

        self.lora_magnitude_vector = nn.ParameterDict()
        self.lora_magnitude_vector[adapter_name] = nn.Parameter(
            weight_norm, requires_grad=True
        )
        self.adapter_layer_names = self.adapter_layer_names[:] + (
            "lora_magnitude_vector",
        )

    def _cache_store(self, key: str, value: Any) -> None:
        """
        Store @value in the cache using @key.

        Args:
            key: The key to use.
            value: The value to store.
        """
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        """
        Pop a value from the cache with index @key.

        Args:
            key: The key to use.

        Returns:
            The value corresponding to the key.
        """
        value = self._caches.pop(key)
        return value

    def _apply_dora(
        self,
        x: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        active_adapter: str,
    ) -> torch.Tensor:
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.

        Args:
            x: The inputs. The shape depends on the type of the wrapped
                layer, but the inputs will be a suitable shape for the
                wrapped layer.
            lora_A: LoRA's A matrix. The shape depends on the type of the
                wrapped layer.
            lora_B: LoRA's B matrix. The shape depends on the type of the
                wrapped layer.
            scaling: The LoRA scale parameter.
            active_adapter: The adapter to use.

        Returns:
            The weight combined with DoRA.
        """
        lora_weight = lora_B.weight @ lora_A.weight
        magnitude = self.lora_magnitude_vector[active_adapter]
        weight = self.get_base_layer().weight
        quant_state = getattr(self.get_base_layer(), "state", None)
        weight = weight.to(x.dtype)
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        # See section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V + \delta V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + \delta V ||_c dynamically
        # reflects the updates of \delta V , it won’t receive any gradient
        # during backpropagation".
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, weight)
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        return result_dora

    def set_scale(self, adapter: str, scale: float) -> None:
        """
        Set the LoRA scale.

        Args:
            adapter: The adapter to use.
            scale: The scale to set.
        """

        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        """
        Apply @scale to the active adapter.

        Args:
            scale: The scale to apply.
        """
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_A.keys():
                self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale: float = None) -> None:
        """
        If @scale is not None, apply the inverse of @scale to the active adapter.

        If scale is None, set the active adapter's scale to (alpha / r).

        Args:
            scale: The scale to apply, or None.
        """
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_A.keys():
                if scale is None:
                    self.scaling[active_adapter] = (
                        self.lora_alpha[active_adapter] / self.r[active_adapter]
                    )
                else:
                    self.scaling[active_adapter] /= scale

    def _check_forward_args(
        self, x: torch.Tensor, *args: List[Any], **kwargs: Dict[str, Any]
    ):
        """
        Check if the arguments are compatible with the configs and state of the model.

        Args:
            x: The input.
            args: The forward pass args.
            kwargs: The forward pass kwargs.
        """
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self,
        x: torch.Tensor,
        *args: List[Any],
        adapter_names: List[str],
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        extra argument that allows mixing different adapters in the same batch at inference time.

        Args:
            x: The input. Its shape is suitable for input to the wrapped layer.
            args: The forward pass args.
            adapter_names: The adapters to use for each batch element. Its length should equal the batch size.
            kwargs: The forward pass kwargs.
        """
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append(
                [index for index, item in enumerate(adapter_names) if item == adapter]
            )

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # Getting the sub-batch, passing it to LoRA layers and updating the
            # corresponding indices of the linear layer output.
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


class Linear(nn.Module, LoraLayer):
    """
    Create a LoRA linear layer.

    Args:
        base_layer: The layer to wrap.
        adapter_name: The name of the adapter. Needed since BaseFineTune
            supports using multiple adapters at once.
        r: The LoRA rank.
        lora_alpha: The LoRA alpha parameter.
        lora_dropout: The LoRA dropout parameter.
        init_lora_weights: Whether to initialize LoRA weights.
        use_rslora: Whether to use RS Lora.
        use_dora: Whether to use DoRA.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: str,
        use_rslora: bool,
        use_dora: bool = False,
    ) -> None:
        """
        Create an adapter with the given name.

        Args:
            adapter_name: The name of the new adapter.
            r: The LoRA rank.
            lora_alpha: The LoRA alpha parameter.
            lora_dropout: The dropout to use with the LoRA module.
            init_lora_weights: If a string, it specifies the style of initialization. If a bool value, it specifies
                whether to initialize the weights with Kaiming uniform, or whether to skip initialization.
        """

        if r <= 0:
            raise ValueError(f"Unexpected lora rank {r=}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Create the LoRA parameters.
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        if self.weight.dtype.is_floating_point or self.weight.dtype.is_complex:
            self.to(self.weight.device, dtype=self.weight.dtype)
        else:
            self.to(self.weight.device)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights.

        Args:
            safe_merge: If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names: The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # Handle dora.
                        # Since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(
                            orig_weights, delta_weight, scaling=1
                        ).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value.
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = (
                            self.lora_magnitude_vector[active_adapter] / weight_norm
                        )
                        orig_weights = dora_factor.view(-1, 1) * (
                            orig_weights + delta_weight
                        )

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # Handle dora.
                        # Since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(
                            base_layer.weight, delta_weight, scaling=1
                        ).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = (
                            self.lora_magnitude_vector[active_adapter] / weight_norm
                        )
                        new_weight = dora_factor.view(-1, 1) * (
                            base_layer.weight.data + delta_weight
                        )
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge the adapters.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = (
                        self.lora_magnitude_vector[active_adapter] / weight_norm
                    )
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter:
                The name of the adapter for which the delta weight should be computed.

        Returns:
            The delta weight. It will be a 2-dimensional tensor of the same
            shape as the wrapped layer's weight.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(
        self, x: torch.Tensor, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Run a forward pass with the active adapters.

        Args:
            x: The input of shape [batch_size, ..., feature_dim].
            args: Forward pass args.
            kwargs: Forward pass kwargs.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self._apply_dora(
                        x, lora_A, lora_B, scaling, active_adapter
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        """
        Get a string representation of the layer.

        Returns:
            A string representation of the layer.
        """
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoraLayer):
    """
    Create a LoRA embedding layer.

    Args:
        base_layer: The layer to wrap.
        adapter_name: The name of the adapter. Needed since BaseFineTune
            supports using multiple adapters at once.
        r: The LoRA rank.
        lora_alpha: The LoRA alpha parameter.
        lora_dropout: The LoRA dropout parameter.
        init_lora_weights: Whether to initialize LoRA weights.
        use_rslora: Whether to use RS Lora.
        use_dora: Whether to use DoRA.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer)

        # DoRA is not supported yet.
        if use_dora:
            raise ValueError("DoRA is not yet supported for Embedding layers.")

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: Union[str, bool],
        use_rslora: bool,
        use_dora: bool = False,
    ) -> None:
        """
        Create an adapter with the given name.

        Args:
            adapter_name: The name of the new adapter.
            r: The LoRA rank.
            lora_alpha: The LoRA alpha parameter.
            lora_dropout: The dropout to use with the LoRA module.
            init_lora_weights: If a string, it specifies the style of initialization. If a bool value, it specifies
                whether to initialize the weights with Kaiming uniform, or whether to skip initialization.
        """
        if use_dora:
            raise ValueError(f"DoRA is not yet supported.")
        if r <= 0:
            raise ValueError(f"Invalid LoRA rank {r=}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # The layer is already completely initialized, this is an update.
            self.to(base_layer.weight.device, dtype=weight.dtype)

        self.set_adapter(self.active_adapters)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights.

        Args:
            safe_merge: If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names: The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = (
                        base_layer.weight.data + self.get_delta_weight(active_adapter)
                    )
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge the adapters.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(
                    active_adapter
                )

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter:
                The name of the adapter for which the delta weight should be
                computed.
        Returns:
            The delta weight. It will have the same shape as the wrapped
            layer's weight.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self,
        x: torch.Tensor,
        *args: List[Any],
        adapter_names: List[str],
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        extra argument that allows mixing different adapters in the same batch at inference time.

        Args:
            x: The input.
            args: The forward pass args.
            adapter_names: The adapters to use for each batch element. Its length should equal the batch size.
            kwargs: The forward pass kwargs.
        Returns:
            The result of the forward pass.
        """
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append(
                [index for index, item in enumerate(adapter_names) if item == adapter]
            )

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Embed the inputs using self.base_layer and the given LoRA weight.

        Args:
            input: The inputs to embed, of shape [batch_size, seq_len].
            weight: The LoRA weight to use.
        Returns:
            The output embedding, of shape [batch_size, seq_len, feature_dim].
        """
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(
        self, x: torch.Tensor, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Run a forward pass with the active adapters.

        Args:
            x: The input of shape [batch_size, seq_len].
            args: Forward pass args.
            kwargs: Forward pass kwargs.
        Returns:
            The output of shape [batch_size, seq_len, feature_dim].
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        """
        Get a string representation of the layer.

        Returns:
            A string representation of the layer.
        """
        rep = super().__repr__()
        return "lora." + rep


@dataclass
class LoRAParams:
    """
    The LoRA parameters used to construct a layer.

    Args:
        adapter_name: The name of the adapter. Needed since BaseFineTune
            supports using multiple adapters at once.
        r: The LoRA rank.
        lora_alpha: The LoRA alpha parameter.
        lora_dropout: The LoRA dropout parameter.
        init_lora_weights: Whether to initialize LoRA weights.
        use_rslora: Whether to use RS Lora.
        use_dora: Whether to use DoRA.
    """

    adapter_name: str
    r: int
    lora_alpha: int
    lora_dropout: float
    init_lora_weights: Union[bool, str]
    use_rslora: bool
    use_dora: bool

    @classmethod
    def from_json(cls, config: Dict[str, Any]) -> LayerConfig:
        """
        Build a LoRAParams from a dictionary.

        Args:
            config: The dictionary.

        Returns:
            The LayerConfig.
        """
        return LoRAParams(**config)

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the LoRAParams to a dictionary.

        Returns:
            The dictionary.
        """
        return asdict(self)


@dataclass
class LayerConfig:
    """
    A configuration that gives which layers to apply a LoRAParams config to.

    Args:
        regex: A regex specifying which modules the LoRAParams apply to. Every matching
            layer will be wrapped with LoRA.
        module_type: The type of LoRA module.
        params: The LoRAParams specifying the LoRA configuration.
    """

    regex: str
    module_type: str
    params: LoRAParams

    @classmethod
    def from_json(cls, config: Dict[str, Any]) -> LayerConfig:
        """
        Build a LayerConfig from a dictionary.

        Args:
            config: The dictionary.

        Returns:
            The LayerConfig.
        """
        return LayerConfig(
            config["regex"],
            config["module_type"],
            LoRAParams.from_json(config["params"]),
        )


@dataclass
class LoRAConfig:
    """
    A configuration of LoRA parameters for a model.

    Args:
        layer_configs: A list of LayerConfigs that specify where to apply LoRA parameters for a model.
    """

    layer_configs: List[LayerConfig]

    @classmethod
    def from_json(cls, config: Dict[str, Any]) -> LoRAConfig:
        """
        Build a LayerConfig from a dictionary.

        Args:
            config: The dictionary.

        Returns:
            The LayerConfig.
        """

        layer_configs = []
        for elem in config:
            layer_configs.append(LayerConfig.from_json(elem))
        return LoRAConfig(layer_configs)


def make_lora_layer(module: nn.Module, layer_config: LoRAConfig) -> LoraLayer:
    """
    Create a LoRA layer that wraps @module using the given @layer_config.

    Args:
        module: The module to wrap.
        layer_config: The LoRA layer config.

    Returns:
        The LoRA layer.
    """
    return {"embedding": Embedding, "linear": Linear}[layer_config.module_type](
        module, **layer_config.params.to_json()
    )


def get_module_to_parents(
    model: nn.Module,
) -> Dict[nn.Module, List[Tuple[str, nn.Module]]]:
    """
    Get a dictionary mapping a module to all of its parents.

    Note that a module can have multiple parents, e.g. if a layer is shared.

    Args:
        model: The model for which to build the mapping.

    Returns:
        A dictionary containing:
            {
                child_module: [("name_of_parent_1", parent1), ...]
            }
    """
    ret = {module: [] for module in model.modules()}
    for parent in model.modules():
        for child_name, child in parent.named_children():
            ret[child].append([parent, child_name])
    return ret


def add_lora_layers(model: nn.Module, lora_config: Dict[str, Any]) -> None:
    """
    Add LoRA layers to the given @model using the given @lora_config.

    Args:
        model: The model to add LoRA parameters to.
        lora_config: The configuration specifying where to add LoRA parameters.
    """
    lora_config = LoRAConfig.from_json(lora_config)

    # Create a graph pointing from each module to its parents.
    # Each module can have multiple parents (e.g. in the case of
    # a shared layer).
    module_to_parents = get_module_to_parents(model)

    replaced_layers = set()
    module_remapping = {}

    for name, module in model.named_modules(remove_duplicate=False):
        for layer_config in lora_config.layer_configs:
            if re.match(layer_config.regex, name):
                if name in replaced_layers:
                    # We do not allow @name to match multiple different regexes. It leads to ambiguity
                    # in which LoRAConfig should be used.
                    raise ValueError(
                        f"Layer {name} matched multiple regexes. Invalid lora_config:\n{lora_config}."
                    )
                replaced_layers.add(name)

                if module in module_remapping:
                    # This can happen even if the check above (making sure @name wasn't already in @replaced_layers)
                    # succeeds. This is because a module can have multiple names (e.g. if it is shared between two layers),
                    # which means it can match multiple different regexes without the @name having already been encountered.
                    raise ValueError(
                        f"Module with name {name} has already been converted to LoRA, and would be again."
                    )

                module_remapping[module] = make_lora_layer(module, layer_config)

    for module, new_module in module_remapping.items():
        for parent, child_name in module_to_parents[module]:
            setattr(parent, child_name, new_module)
