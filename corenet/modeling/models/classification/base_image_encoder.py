#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn

from corenet.constants import is_test_env
from corenet.modeling.layers import LinearLayer
from corenet.modeling.misc.init_utils import initialize_fc_layer
from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel
from corenet.modeling.neural_augmentor import build_neural_augmentor
from corenet.utils import logger


@MODEL_REGISTRY.register(name="__base__", type="classification")
class BaseImageEncoder(BaseAnyNNModel):
    """Base class for different image classification models"""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the backbone
        # We allow that using output_stride and replace_stride_with_dilation arguments
        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True
        self.output_stride = output_stride

        self.model_conf_dict = dict()
        self.neural_augmentor = build_neural_augmentor(opts=opts, *args, **kwargs)
        if getattr(opts, "model.classification.gradient_checkpointing"):
            logger.error(
                "The argument, --model.classification.gradient-checkpointing, is deprecated and should not be used. \
                Please implement 'get_activation_checkpoint_submodule_class' and use --model.activation-checkpointing instead."
            )

        self.enable_layer_wise_lr_decay = getattr(
            opts, "model.classification.enable_layer_wise_lr_decay"
        )
        self.layer_wise_lr_decay_rate = getattr(
            opts, "model.classification.layer_wise_lr_decay_rate"
        )

    @property
    def n_classes(self) -> int:
        """Number of classes that model is or will be trained to classify."""
        n_classes = getattr(self.opts, "model.classification.n_classes")
        if n_classes is None:
            logger.error(
                f"Number of classes in {self.__class__.__name__} cannot be None. Please specify using 'model.classification.n_classes' argument in configuration file."
            )
        return n_classes

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add image classification model-specific arguments"""
        if cls != BaseImageEncoder:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--model.classification.classifier-dropout",
            type=float,
            default=0.0,
            help="Dropout rate in classifier",
        )

        group.add_argument(
            "--model.classification.name", type=str, default=None, help="Model name"
        )
        group.add_argument(
            "--model.classification.n-classes",
            type=int,
            default=1000,
            help="Number of classes in the dataset",
        )
        group.add_argument(
            "--model.classification.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained backbone",
        )
        group.add_argument(
            "--model.classification.freeze-batch-norm",
            action="store_true",
            help="Freeze batch norm layers",
        )
        group.add_argument(
            "--model.classification.activation.name",
            default=None,
            type=str,
            help="Non-linear function name (e.g., relu)",
        )
        group.add_argument(
            "--model.classification.activation.inplace",
            action="store_true",
            help="Inplace non-linear functions",
        )
        group.add_argument(
            "--model.classification.activation.neg-slope",
            default=0.1,
            type=float,
            help="Negative slope in leaky relu",
        )

        group.add_argument(
            "--model.classification.finetune-pretrained-model",
            action="store_true",
            help="Finetune a pretrained model",
        )
        group.add_argument(
            "--model.classification.n-pretrained-classes",
            type=int,
            default=None,
            help="Number of pre-trained classes",
        )

        group.add_argument(
            "--model.classification.gradient-checkpointing",
            action="store_true",
            help="Checkpoint output of each spatial level in the classification backbone. Note that"
            "we only take care of checkpointing in {}. If custom forward functions are used, please"
            "implement checkpointing accordingly. "
            "This option is deprecated in favor or --model.activation-checkpointing.",
        )

        group.add_argument(
            "--model.classification.enable-layer-wise-lr-decay",
            action="store_true",
            default=False,
            help="Enable layer-wise LR.",
        )
        group.add_argument(
            "--model.classification.layer-wise-lr-decay-rate",
            type=float,
            default=1.0,
            help="Layer-wise LR decay range. Each model needs to define how layer-wise LR should be decayed."
            "For ViT, we decay layer_wise_lr_decay_rate ** (n_layers - i), where i is the layer index.",
        )

        return parser

    def check_model(self) -> None:
        """Check to see if model is adhering to the image encoder structure. Sub-classes
        are not required to adhere to this structure. This is only required for easy
        integration with downstream tasks."""
        assert (
            self.model_conf_dict
        ), "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, "Please implement self.conv_1"
        assert self.layer_1 is not None, "Please implement self.layer_1"
        assert self.layer_2 is not None, "Please implement self.layer_2"
        assert self.layer_3 is not None, "Please implement self.layer_3"
        assert self.layer_4 is not None, "Please implement self.layer_4"
        assert self.layer_5 is not None, "Please implement self.layer_5"
        assert self.conv_1x1_exp is not None, "Please implement self.conv_1x1_exp"
        assert self.classifier is not None, "Please implement self.classifier"

    def update_classifier(self, opts: argparse.Namespace, n_classes: int) -> None:
        """This function updates the classification layer in a model. Useful for fine-tuning purposes."""
        logger.warning(
            "We encourage to use model scopes (`--model.resume-exclude-scopes`, `--model.ignore-missing-scopes`, "
            "and `--model.rename-scopes-map`) for updating classifier for fine-tuning tasks. We will be "
            "deprecating this function in future."
        )

        linear_init_type = getattr(opts, "model.layer.linear_init", "normal")
        if isinstance(self.classifier, nn.Sequential):
            in_features = self.classifier[-1].in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)
            self.classifier[-1] = layer
        else:
            in_features = self.classifier.in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)

            # re-init head
            head_init_scale = 0.001
            layer.weight.data.mul_(head_init_scale)
            layer.bias.data.mul_(head_init_scale)

            self.classifier = layer

    def _forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        """Run a layer of the model, optionally with checkpointing"""
        # Larger models with large input image size may not be able to fit into memory.
        # We can use gradient checkpointing to enable training with large models and large inputs
        return layer(x)

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Extract feature maps from different spatial levels of the model.

        Args:
            x: Input image tensor
            use_l5: Include features from `layer_5` in the output dictionary. Defaults to True.
            use_l5_exp: Include features from `conv_1x1_exp` in the output dictionary. Defaults to False.

        Returns:
            A mapping containing the name and output at each spatial-level of the model.

        ...note:
            This is useful for down-stream tasks.
        """
        out_dict = {}  # Use dictionary over NamedTuple so that JIT is happy

        if self.training and self.neural_augmentor is not None:
            x = self.neural_augmentor(x)
            out_dict["augmented_tensor"] = x

        x = self._forward_layer(self.conv_1, x)  # 112 x112
        x = self._forward_layer(self.layer_1, x)  # 112 x112
        out_dict["out_l1"] = x

        x = self._forward_layer(self.layer_2, x)  # 56 x 56
        out_dict["out_l2"] = x

        x = self._forward_layer(self.layer_3, x)  # 28 x 28
        out_dict["out_l3"] = x

        x = self._forward_layer(self.layer_4, x)  # 14 x 14
        out_dict["out_l4"] = x

        if use_l5:
            x = self._forward_layer(self.layer_5, x)  # 7 x 7
            out_dict["out_l5"] = x

            if use_l5_exp:
                x = self._forward_layer(self.conv_1x1_exp, x)
                out_dict["out_l5_exp"] = x
        return out_dict

    def extract_end_points_l4(self, x: Tensor, *args, **kwargs) -> Dict[str, Tensor]:
        """This function is similar to `extract_end_points_all`, with an exception that
        it only returns output in a dictionary form till `layer_4` of the model.
        """
        return self.extract_end_points_all(x, use_l5=False)

    def extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        """This function is similar to `extract_end_points_all`. However, it returns a single tensor as the
        output of the last layer instead of a dictionary, and is typically used during classification tasks where
        intermediate feature maps are not required.
        """
        x = self._forward_layer(self.conv_1, x)
        x = self._forward_layer(self.layer_1, x)
        x = self._forward_layer(self.layer_2, x)
        x = self._forward_layer(self.layer_3, x)

        x = self._forward_layer(self.layer_4, x)
        x = self._forward_layer(self.layer_5, x)
        x = self._forward_layer(self.conv_1x1_exp, x)
        return x

    def forward_classifier(self, x: Tensor, *args, **kwargs) -> Tensor:
        """A helper function to extract features and running a classifier."""
        # We add another classifier function so that the classifiers
        # that do not adhere to the structure of BaseEncoder can still
        # use neural augmentor
        x = self.extract_features(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model, optionally training the model with
        neural augmentation."""
        if self.neural_augmentor is not None:
            if self.training:
                x_aug = self.neural_augmentor(x)
                prediction = self.forward_classifier(x_aug)  # .detach()
                out_dict = {"augmented_tensor": x_aug, "logits": prediction}
            else:
                out_dict = {
                    "augmented_tensor": None,
                    "logits": self.forward_classifier(x),
                }
            return out_dict
        else:
            x = self.forward_classifier(x, *args, **kwargs)
            return x

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[List[Mapping], List[float]]:
        """Get parameters for training along with the learning rate.

        Args:
            weight_decay: weight decay
            no_decay_bn_filter_bias: Do not decay BN and biases. Defaults to False.

        Returns:
             Returns a tuple of length 2. The first entry is a list of dictionary with three keys
             (params, weight_decay, param_names). The second entry is a list of floats containing
             learning rate for each parameter.
        """
        if self.enable_layer_wise_lr_decay:
            return self._apply_layer_wise_lr(
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                *args,
                **kwargs,
            )
        return super().get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs,
        )

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        img_channels = 3
        if is_test_env():
            # We use smaller spatial resolution, for faster testing.
            # We use 32 because most ImageNet models do a down-sampling by a factor of 32 before
            # global average pooling and classification layer.
            height = 32
            width = 32
        else:
            # this is the typical resolution used in ImageNet datasets
            height = 224
            width = 224
        n_labels = 10
        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )
        label_tensor = torch.randint(low=0, high=n_labels, size=(batch_size,)).long()
        return {"samples": img_tensor, "targets": label_tensor}

    def get_exportable_model(self) -> nn.Module:
        """
        This function can be used to prepare the architecture for inference. For example,
        re-parameterizing branches when possible. The functionality of this method may vary
        from model to model, so child model classes have to implement this method, if such a
        transformation exists.
        """
        return self

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        """Helper function to build a model.

        Args:
            opts: Command-line arguments

        Returns:
            An instance of `corenet.modeling.models.BaseAnyNNModel`.
        """
        default_opt_info = set_model_specific_opts_before_model_building(opts)
        model = cls(opts, *args, **kwargs)

        unset_model_specific_opts_after_model_building(opts, default_opt_info)

        if getattr(opts, "model.classification.freeze_batch_norm"):
            cls.freeze_norm_layers(opts=opts, model=model)
        return model


# TODO: Find models and configurations that uses `set_model_specific_opts_before_model_building` and
#  `unset_model_specific_opts_after_model_building` functions. Find a more explicit way of satisfying this requirement,
#  such as namespacing config entries in a more composable way so that we no longer have conflicting config entries.


def set_model_specific_opts_before_model_building(
    opts: argparse.Namespace,
) -> Dict[str, Any]:
    """Override library-level defaults with model-specific default values.

    Args:
        opts: Command-line arguments

    Returns:
        A dictionary containing the name of arguments that are updated along with their original values.
        This dictionary is used in `unset_model_specific_opts_after_model_building` function to unset the
        model-specific to library-specific defaults.
    """

    cls_act_fn = getattr(opts, "model.classification.activation.name")
    default_opts_info = {}
    if cls_act_fn is not None:
        # Override the default activation arguments with classification network specific arguments
        default_act_fn = getattr(opts, "model.activation.name", "relu")
        default_act_inplace = getattr(opts, "model.activation.inplace", False)
        default_act_neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)

        setattr(opts, "model.activation.name", cls_act_fn)
        setattr(
            opts,
            "model.activation.inplace",
            getattr(opts, "model.classification.activation.inplace", False),
        )
        setattr(
            opts,
            "model.activation.neg_slope",
            getattr(opts, "model.classification.activation.neg_slope", 0.1),
        )

        default_opts_info["model.activation.name"] = default_act_fn
        default_opts_info["model.activation.inplace"] = default_act_inplace
        default_opts_info["model.activation.neg_slope"] = default_act_neg_slope
    return default_opts_info


def unset_model_specific_opts_after_model_building(
    opts: argparse.Namespace, default_opts_info: Dict[str, Any], *ars, **kwargs
) -> None:
    """Given command-line arguments and a mapping of opts that needs to be unset, this function
    unsets the library-level defaults that were over-ridden previously
    in `set_model_specific_opts_before_model_building`.
    """
    assert isinstance(default_opts_info, dict), (
        f"Please ensure set_model_specific_opts_before_model_building() "
        f"returns a dict."
    )

    if default_opts_info:
        for k, v in default_opts_info.items():
            setattr(opts, k, v)
