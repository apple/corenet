#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from corenet.modeling.image_projection_layers import build_image_projection_head
from corenet.modeling.image_projection_layers.base_image_projection import (
    get_in_feature_dimension,
)
from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel, get_model
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.multi_modal_img_text.base_multi_modal_img_text import (
    BaseMultiModalImageText,
)
from corenet.modeling.text_encoders import BaseTextEncoder, build_text_encoder
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master


@MODEL_REGISTRY.register(name="clip", type="multi_modal_image_text")
class CLIP(BaseMultiModalImageText):
    """Model for contrastive language image pre-training.

    See `CLIP <https://arxiv.org/abs/2103.00020>` paper for details.

    Args:
        opts: Command-line arguments.
        image_encoder: Image encoder.
        text_encoder: Text encoder.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        image_encoder: BaseImageEncoder,
        text_encoder: BaseTextEncoder,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.image_encoder: BaseImageEncoder = image_encoder
        self.text_encoder: BaseTextEncoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.use_distributed = getattr(opts, "ddp.use_distributed", False)
        self.cached_text_features = None
        self.reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add CLIP specific arguments"""
        if cls == CLIP:
            group = parser.add_argument_group(title=cls.__name__)
            group.add_argument(
                "--model.multi-modal-image-text.clip.projection-dim",
                type=int,
                default=256,
                help="Project image and text features to this dimensionality",
            )
        return parser

    def reset_parameters(self) -> None:
        """Helper function to reset model weights.

        Currently, we only reset the @logit_scale.
        """
        torch.nn.init.constant_(self.logit_scale, math.log(1.0 / 0.07))

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[List[Dict], List[float]]:
        """Get parameters for training along with the learning rate.

        Args:
            weight_decay: weight decay.
            no_decay_bn_filter_bias: Do not decay BN and biases. Defaults to False.

        Returns:
             Returns a tuple of length 2. The first entry is a list of dictionary with three keys
             (params, weight_decay, param_names). The second entry is a list of floats containing
             learning rate for each parameter.

        Note:
            Kwargs may contain module_name. To avoid multiple arguments with the same name,
            we pop it and concatenate with image and text encoders.
        """
        prev_module_name = kwargs.pop("module_name", "")
        image_param_list, image_lr_mult = self.image_encoder.get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=prev_module_name + "image_encoder.",
            *args,
            **kwargs,
        )
        # The learning rate list from image encoder returns 1.0 as a LR multiplier.
        # Update the learning rate to the specified value.
        image_lr_mult = [self.lr_multiplier_img_encoder] * len(image_lr_mult)

        text_param_list, text_lr_mult = self.text_encoder.get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=prev_module_name + "text_encoder.",
            *args,
            **kwargs,
        )
        # The learning rate list from text encoder returns 1.0 as a LR multiplier.
        # Update the learning rate to the specified value.
        text_lr_mult = [self.lr_multiplier_text_encoder] * len(text_lr_mult)

        # We need to add the logit scale
        logit_scale_param_list = [
            {
                "params": [self.logit_scale],
                "weight_decay": 0.0,
                "param_names": ["logit_scale"],
            }
        ]
        logit_scale_lr_mult = [1.0] * len(logit_scale_param_list)

        return (
            image_param_list + text_param_list + logit_scale_param_list,
            image_lr_mult + text_lr_mult + logit_scale_lr_mult,
        )

    def dummy_input_and_label(
        self, batch_size: int
    ) -> Dict[str, Union[Dict[str, Tensor], Tensor]]:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        img_channels = 3
        height = 224
        width = 224
        vocab_size = 10
        seq_length = 5
        num_obj_classes = 2
        num_captions_per_class = 2
        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )

        if self.training:
            text_tensor = torch.randint(
                low=0, high=vocab_size, size=(batch_size, seq_length)
            ).long()
        else:
            text_tensor = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, num_obj_classes, num_captions_per_class, seq_length),
            ).long()

        return {
            "samples": {"image": img_tensor, "text": text_tensor},
            "targets": text_tensor,
        }

    def _exponentiate_and_clip_logits(self, max_scale: float = 100.0) -> Tensor:
        """Exponentiate and clip the logit scale.

        Args:
            max_scale: Maximum value of logit scale.

        Returns:
            A tensor of shape [1].
        """
        scale = self.logit_scale.exp()
        scale = torch.clamp(scale, 0, max_scale)
        return scale

    def _reset_cached_text_features(self, mode_str: str) -> None:
        """Reset cached text features.

        Args:
            mode: String specifying the mode of the model (e.g., train or eval).
        """
        if self.cached_text_features is not None:
            if is_master(self.opts):
                logger.log(
                    f"Resetting {self.__class__.__name__}'s cache in {mode_str} mode."
                )
            self.cached_text_features = None

    def train(self, mode: bool = True):
        """Sets the module in training mode when @mode is enabled.

        ...note:
            We override this function to reset cached text features before starting evaluation or training.
            This resetting is required so that features cached from one model may not be used by other model.
            An example of such a use case is when training CLIP model with and without exponential
            moving average.
        """
        mode_str = "train" if mode else "eval"
        self._reset_cached_text_features(mode_str=mode_str)
        return super().train(mode)

    def forward(self, input: Dict[str, Optional[Tensor]]) -> Dict[str, Tensor]:
        """Forward pass of CLIP model.

        Args:
            input: A dictionary containing tensors under keys 'image', 'text', and 'padding_mask'.
                The shape of inputs is:
                input["image"]: The shape of tensor is [batch size, image channels, image width, image height]
                input["text"]:
                    For pre-training, the shape of tensor is [batch size, sequence length]
                    For zero-shot image classification, the shape of tensor is
                        [batch size, number of classes, number of captions, sequence length].
                input["padding_mask"]: The shape of tensor is [batch size, sequence length]

        Returns:
            A dictionary containing tensors under keys 'image', 'text', 'logit_scale', 'zero_shot_image_logits',
            and 'augmented_tensor'.

            output["image"]: The shape of image embeddings is [batch size, hidden dimension]
            output["text"]: The shape of text embeddings during pre-training is [hidden dimension, batch size]. For
             zero-shot image classification, the shape is [hidden dimension, number of classes.]
            output["logit_scale"]: The shape of tensor is [1].
            output["zero_shot_image_logits"]: The shape of tensor is [batch size, number of classes]. This is returned
                only during evaluation and is set to 'None' during training.
            output["augmented_tensor"]: The shape of tensor is [batch size, image channels, image width, image height].
                This is only returned during training if RangeAugment (https://arxiv.org/abs/2212.10553) is enabled.
                Otherwise, it is set to None.
        """

        images = input.get("image")
        text_tokens = input.get("text")
        padding_mask = input.get("padding_mask", None)

        # [batch_size, image_channels, image_height, image_width] -> [batch_size, hidden_dim]
        image_encoder_out = self.image_encoder(images)
        augmented_tensor = None
        if isinstance(image_encoder_out, Dict):
            if not {"augmented_tensor", "logits"}.issubset(image_encoder_out.keys()):
                logger.error(
                    "Output of image classifier must contain logits and augmented_tensor"
                    " as keys. Got keys: {}".format(image_encoder_out.keys())
                )
            image_embeddings = image_encoder_out["logits"]
            augmented_tensor = image_encoder_out["augmented_tensor"]
        elif isinstance(image_encoder_out, Tensor):
            image_embeddings = image_encoder_out
        else:
            logger.error("The output of image encoder should be either Dict or Tensor")

        if not self.training:
            # During zero-shot image classification, the embedding vector is returned for each class
            # Because the captions and classes are the same for all images in a batch, embeddings are returned only
            # for the first image.
            # [batch_size, num_classes, num_captions, sequence_length] --> [hidden_dim, num_classes]
            if self.cached_text_features is None:
                text_embeddings = self.text_encoder(
                    text_tokens=text_tokens, key_padding_mask=padding_mask
                )
                self.cached_text_features = text_embeddings
            else:
                text_embeddings = self.cached_text_features
        else:
            # During pre-training, the embeddings are only returned for end-of-text token.

            # Note that text embeddings are transposed (i.e., batch is not the first dimension).
            # [batch_size, sequence_length] --> [hidden_dim, batch_size]
            text_embeddings = self.text_encoder(
                text_tokens=text_tokens, key_padding_mask=padding_mask
            )

        if not self.training:
            assert (
                text_embeddings.shape[0] == image_embeddings.shape[1]
            ), "The hidden dimension of image and text towers is different. Please check."
            # This means that we are running a zero-shot set-up.
            # [batch_size, hidden_dim] x [hidden_dim, num_classes] --> [batch_size, num_classes]
            zero_shot_image_logits = 100.0 * image_embeddings @ text_embeddings
            return {
                "image": None,
                "text": None,
                "logit_scale": self._exponentiate_and_clip_logits(),
                "zero_shot_image_logits": zero_shot_image_logits,
                "augmented_tensor": None,
            }
        else:
            return {
                "image": image_embeddings,
                "text": text_embeddings,
                "logit_scale": self._exponentiate_and_clip_logits(),
                "zero_shot_image_logits": None,
                "augmented_tensor": augmented_tensor,
            }

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        """Build the CLIP model.

        Args:
            opts: Command-line arguments.

        Returns:
            An instance of CLIP model.
        """
        projection_dim = getattr(
            opts, "model.multi_modal_image_text.clip.projection_dim"
        )
        if projection_dim < 1:
            logger.error("Projection dimension should be > 1. Got: {}.")

        image_encoder: BaseImageEncoder = get_model(
            opts=opts, category="classification", *args, **kwargs
        )
        text_encoder: BaseTextEncoder = build_text_encoder(
            opts=opts, projection_dim=projection_dim, *args, **kwargs
        )

        # replace the classifier in image encoder with the task specific classifier
        image_encoder.classifier = update_image_classifier(
            opts,
            image_classifier=image_encoder.classifier,
            projection_dim=projection_dim,
        )

        model = cls(
            opts,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            *args,
            **kwargs,
        )

        if getattr(opts, "model.multi_modal_image_text.freeze_batch_norm"):
            cls.freeze_norm_layers(opts, model)
        return model

    def get_activation_checkpoint_submodule_class(
        self,
    ) -> Union[Callable, List[Callable]]:
        """Activation checkpoint sub-module classes.

        Returns:
            For CLIP, image and text encoders activation checkpoint sub-module classes are returned. If both of them are the same,
            then only image encoder's activation checkpoint sub-module class is returned because activation checkpointing is applied
            recursively.
        """
        img_encoder_ckpt_module = (
            self.image_encoder.get_activation_checkpoint_submodule_class()
        )
        text_encoder_ckpt_module = (
            self.text_encoder.get_activation_checkpoint_submodule_class()
        )
        if img_encoder_ckpt_module == text_encoder_ckpt_module:
            return img_encoder_ckpt_module
        return [img_encoder_ckpt_module, text_encoder_ckpt_module]


def update_image_classifier(
    opts, image_classifier: nn.Module, projection_dim: int, *args, **kwargs
) -> nn.Module:
    """Update the classifier."""
    in_features = get_in_feature_dimension(image_classifier)
    new_img_classifier = build_image_projection_head(
        opts, in_dim=in_features, out_dim=projection_dim
    )
    return new_img_classifier
