#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlsplit

import torch
from PIL import Image
from torch import Tensor

from corenet.constants import DATA_CACHE_DIR
from corenet.data.collate_fns import COLLATE_FN_REGISTRY
from corenet.data.datasets.dataset_base import BaseImageDataset
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification import (
    build_zero_shot_image_classification_dataset,
)
from corenet.data.datasets.utils.text import caption_preprocessing
from corenet.data.io.transfer_clients import BaseClient, get_transfer_client
from corenet.data.text_tokenizer import build_tokenizer
from corenet.data.transforms import image_pil as T
from corenet.data.transforms.common import Compose
from corenet.utils import logger


class BaseMultiModalImgText(BaseImageDataset):
    """
    Base class for Image-Text multi-modal learning.

    Args:
        opts: Command-line arguments.

    ...note:
        As a standard practice, we use web image-text datasets for pre-training and measure the zero-shot performance on
        standard image classification datasets for validation/evaluation purposes.
    """

    def __init__(
        self,
        opts,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(
            opts=opts,
            *args,
            **kwargs,
        )
        self.text_tokenizer = build_tokenizer(opts=opts)
        self.cache_loc = DATA_CACHE_DIR
        self._transfer_client = None

        if not self.is_training:
            self.cached_zero_shot_captions = None
            self.cached_max_seq_length = None
            self.dataset = build_zero_shot_image_classification_dataset(
                self.opts, *args, **kwargs
            )
        else:
            self.dataset = self.get_image_text_dataset()

    def get_image_text_dataset(self) -> Optional[Any]:
        """Helper function to download or process image-text dataset.

        Recommended way for downloading image-text datasets is to download on-the-fly. In such a case, we do not need
        to implement this function and return None. See 'corenet/data/datasets/multi_modal_img_text/img_text_tar_dataset.py' as an example.

        However, for small datasets (e.g., Flickr), we can download the dataset before training is started. In such a case, we
        need to implement this function. See 'corenet/data/datasets/multi_modal_img_text/flickr.py' as an example.
        """
        raise NotImplementedError("Child classes must implement this function.")

    def _get_transfer_client(self, file_path: str) -> BaseClient:
        """Get transfer client for a given file path.

        Args:
            opts: Command-line arguments.
            file_path: File path.

        Returns:
            An instance of BaseTransferClient.

        ...note:
            1. Some of the clients are not pickle-able (e.g., S3). Therefore, this function should not be
            called inside the '__init__' function.
            2. This function is added as a class function so that we can re-use it in future sub-classes.
        """

        if self._transfer_client is None:
            opts = self.opts
            client_name = urlsplit(file_path).scheme.lower()

            self._transfer_client = get_transfer_client(
                opts,
                transfer_client_name=client_name,
                force_delete=False,
                only_download_on_start_rank=False,
                synchronize_distributed_ranks=False,
                parallel_download=False,
            )
        return self._transfer_client

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.text_tokenizer.vocab_size

    @property
    def padding_index(self) -> Optional[int]:
        """Padding index.

        ...note:
            If padding index is specified and the length of tokenized caption is less than desired context length,
            then tokenized caption is padded with the value of padding index.
        """
        return getattr(self.opts, "dataset.multi_modal_img_text.padding_index")

    @property
    def context_length(self) -> int:
        """Context length for text encoder."""
        opts = self.opts
        context_length = getattr(opts, "dataset.multi_modal_img_text.context_length")
        assert context_length is not None
        return context_length

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""

        if cls == BaseMultiModalImgText:
            group = parser.add_argument_group(cls.__name__)

            group.add_argument(
                "--dataset.multi-modal-img-text.context-length",
                type=int,
                default=77,
                help="Context length for the text model. Defaults to 77, the same as in the CLIP paper.",
            )
            group.add_argument(
                "--dataset.multi-modal-img-text.padding-index",
                type=int,
                default=None,
                help="Padding index. Defaults to None.",
            )

            group.add_argument(
                "--dataset.multi-modal-img-text.trunc-seq-len",
                action="store_true",
                default=False,
                help="Many sequences in a batch do not have lengths equal to specified context length. Enabling this flag "
                "allows us to truncate the sequences such that the sequence length of a batch is equal to sequence "
                "with max. non-padded tokens. Defaults to False.",
            )

        return parser

    def _transform_text(self, text_tensor: Tensor) -> Tuple[Tensor, int]:
        """Helper function to transform the text tensor. If the text tensor is smaller
        than the context length, it pads it with zeros and replaces the last token with EOT token.

        Args:
            text_tensor: Text tensor with N tokens. Shape is (N,).

        Returns:
            A Tuple of text tensor (whole length is equal to context length) and length of the tensor.

        ...note:
            1. If length of tokenized text is greater than context length, then it will be truncated.
            2. If length of tokenized text is smaller than context length, then it is padded.
        """
        captions_tensor = torch.zeros(size=(self.context_length,), dtype=torch.long)

        text_len = text_tensor.shape[0]
        if text_len > self.context_length:
            text_tensor = text_tensor[: self.context_length]
            text_tensor[-1] = self.text_tokenizer.eot_token_id
            text_len = self.context_length
        captions_tensor[:text_len] = text_tensor[:text_len]
        return captions_tensor, text_len

    def _training_transforms(
        self, size: Tuple[int, int], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data augmentation during training.

        The default order is RandomResizedCrop, Optional[RandAugment or AutoAugment], ToTensor, Optional[RandomErase]

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`

        .. note::
            1. AutoAugment and RandAugment are mutually exclusive.
            2. Mixup and CutMix are applied on batches are implemented in trainer.
        """
        aug_list = [
            T.RandomResizedCrop(opts=self.opts, size=size),
        ]
        auto_augment = getattr(
            self.opts, "image_augmentation.auto_augment.enable", False
        )
        rand_augment = getattr(
            self.opts, "image_augmentation.rand_augment.enable", False
        )
        if auto_augment and rand_augment:
            logger.error(
                "AutoAugment and RandAugment are mutually exclusive. Use either of them, but not both"
            )
        elif auto_augment:
            aug_list.append(T.AutoAugment(opts=self.opts))
        elif rand_augment:
            aug_list.append(T.RandAugment(opts=self.opts))

        aug_list.append(T.ToTensor(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_erase.enable", False):
            aug_list.append(T.RandomErasing(opts=self.opts))

        return Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(
        self, size: Union[Tuple, int], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data transforms during validation or evaluation
         The order is Resize, CenterCrop, ToTensor

         Args:
            size: Size for resizing the input image. Expected to be an integer (width=height) or a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return Compose(opts=self.opts, img_transforms=aug_list)

    def _process_img_caption(
        self,
        input_img: Image.Image,
        captions_str: Union[str, List[str], List[List[str]]],
        img_transform_fn: T.BaseTransformation,
    ) -> Tuple[Tensor, Tensor, int]:
        """Apply data augmentation to images and pre-processing to text captions

        Args:
            input_img: Input PIL Image
            captions_str: Text captions
            img_transform_fn: Image transform functions

        Returns:
            A tuple of image tensor, caption tensor, and max. sequence length of a sequence in caption tensor.

        ...note:
            Zero-shot captions are the same for all images, so we cache them after tokenization during evaluation.
            If tokenized cached captions are available, they are returned.
        """

        data = {"image": input_img}
        img_tensor = img_transform_fn(data)["image"]

        if (
            hasattr(self, "cached_zero_shot_captions")
            and self.cached_zero_shot_captions is not None
        ):
            # return the tokenized cached captions
            return (
                img_tensor,
                self.cached_zero_shot_captions,
                self.cached_max_seq_length,
            )

        max_seq_len = 0
        # process caption
        if isinstance(captions_str, str):
            captions_tensor, max_seq_len = self._transform_text(
                self.text_tokenizer(caption_preprocessing(captions_str))
            )
        elif isinstance(captions_str, List):
            captions_tensor = []
            for captions_str_i in captions_str:
                if isinstance(captions_str_i, List):
                    # captions_str is [ [Num_templates_per_class] * Num_classes]
                    captions_tensor_i = []
                    for captions_str_i_j in captions_str_i:
                        seq, seq_len = self._transform_text(
                            self.text_tokenizer(caption_preprocessing(captions_str_i_j))
                        )
                        captions_tensor_i.append(seq)
                        max_seq_len = max(max_seq_len, seq_len)
                    captions_tensor_i = torch.stack(captions_tensor_i, dim=0)
                    captions_tensor.append(captions_tensor_i)
                elif isinstance(captions_str_i, str):
                    # captions_str is [Num_templates_per_image]
                    seq, seq_len = self._transform_text(
                        self.text_tokenizer(caption_preprocessing(captions_str_i))
                    )
                    captions_tensor.append(seq)
                    max_seq_len = max(max_seq_len, seq_len)
                else:
                    logger.error(
                        "Got captions_str of type {}: {} from {}".format(
                            type(captions_str_i), captions_str_i, captions_str
                        )
                    )
            # the shape of tensor is [Num_classes, captions_per_class, caption_length]
            # or [Captions_per_image, caption_length]
            captions_tensor = torch.stack(captions_tensor, dim=0)
        else:
            captions_tensor = None
            logger.error(
                "Captions should be either string, List[String] or List[List[str]]"
            )

        if (
            hasattr(self, "cached_zero_shot_captions")
            and self.cached_zero_shot_captions is None
        ):
            # Cache the tokenized captions during evaluation because they are the same for all images.
            self.cached_zero_shot_captions = captions_tensor
            self.cached_max_seq_length = max_seq_len

        return img_tensor, captions_tensor, max_seq_len

    def get_zero_shot_image_text_pair(
        self, sample_index: int
    ) -> Tuple[Image.Image, Union[str, List[str], List[List[str]]], int]:
        """Get image-text pair for zero-shot dataset along with classification label.

        Args:
            sample_index: Sample index

        Returns:
            A tuple of PIL image, captions, and class label
        """
        assert (
            self.dataset is not None
        ), f"For zero-shot image-classification datasets, {self.dataset} should not be None."
        img_path, captions_str, class_label = self.dataset[sample_index]
        input_img = self.read_image_pil(img_path)
        return input_img, captions_str, class_label

    def get_image_text_dataset_pair(self, sample_index: int) -> Tuple[Image.Image, str]:
        """Get image and text caption pair from the noisy image-text dataset. Sub-classes may implement
        this method to use '__getitem__' of BaseMultiModalImgText class."""
        raise NotImplementedError(
            f"Child classes may implement this function to use '__getitem__' of {self.__class__.__name__}."
        )

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int]
    ) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
        """Returns the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Returns:
            A dictionary with `samples` and `targets` as keys corresponding to input and label of
            a sample, respectively.

        Shapes:
            The shape of values in output dictionary, output_data, are as follows:

            output_data["samples"]["image"]: Shape is [Channels, Height, Width]
            output_data["samples"]["text"]: Shape is
                * [Context_Length] (single caption, as in CLIP datasets)
                * [Num_classes, Num_Captions, Context_length] (multiple captions per class, as in 0-shot Imagenet dataset)
            output_data["samples"]["padding_mask"]: Same as output_data["samples"]["text"]
            output_data["samples"]["max_seq_len"]: Shape is [1]
            output_data["targets"]: Shape is [1]
        """
        crop_size_h, crop_size_w, sample_index = sample_size_and_index
        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        if self.is_training:
            # read captions and image path from image-text dataset.
            input_img, captions_str = self.get_image_text_dataset_pair(sample_index)
            # for image-text datasets, we do not have class labels. So, we set to -1.
            class_label = -1
        else:
            # read captions and image path from zero-shot image classification dataset
            input_img, captions_str, class_label = self.get_zero_shot_image_text_pair(
                sample_index
            )

        (
            img_tensor,
            captions_tensor,
            max_seq_len,
        ) = self._process_img_caption(
            input_img=input_img,
            captions_str=captions_str,
            img_transform_fn=transform_fn,
        )

        padding_mask = None
        if self.padding_index is not None:
            padding_mask = captions_tensor == self.padding_index

        data = {
            "samples": {
                "image": img_tensor,
                "text": captions_tensor,
                "padding_mask": padding_mask,
                "max_seq_len": max_seq_len,
            },
            "targets": class_label,
            "zero_shot": not self.is_training,
        }

        return data

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        if self.is_training:
            return super().__repr__()
        return self.dataset.__repr__()


@COLLATE_FN_REGISTRY.register(name="multi_modal_img_text_collate_fn")
def multi_modal_img_text_collate_fn(
    batch: List[Mapping[str, Union[Tensor, Mapping[str, Tensor]]]],
    opts: argparse.Namespace,
) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
    """Combines a list of dictionaries into a single dictionary by concatenating matching fields."""
    images = []
    text_tokens = []
    padding_mask = []
    labels = []

    truncate_seq_len = getattr(opts, "dataset.multi_modal_img_text.trunc_seq_len")

    zero_shot = batch[0].pop("zero_shot")

    max_seq_len_in_batch = 1  # at least one token is required in the sequence
    for i, batch_i in enumerate(batch):
        inputs_i = batch_i.pop("samples")
        img_tensor = inputs_i.pop("image")
        if img_tensor is None:
            continue
        images.append(img_tensor)
        labels.append(batch_i.pop("targets"))

        text_data = inputs_i.pop("text")
        pad_mask = inputs_i.pop("padding_mask")
        max_seq_len_in_batch = max(max_seq_len_in_batch, inputs_i.pop("max_seq_len", 0))
        if not zero_shot or (zero_shot and i == 0):
            # For zero-shot, all text captions are the same
            # so, we only aggregate for one batch element
            text_tokens.append(text_data)
            if pad_mask is not None:
                padding_mask.append(pad_mask)

    images = torch.stack(images, dim=0)
    text_tokens = torch.stack(text_tokens, dim=0)

    # truncate tokens based on the max. seq length
    if not truncate_seq_len:
        max_seq_len_in_batch = text_tokens.shape[-1]
    text_tokens = text_tokens[..., :max_seq_len_in_batch]

    if len(padding_mask) != 0:
        padding_mask = torch.stack(padding_mask, dim=0)
        padding_mask = padding_mask[..., :max_seq_len_in_batch]
    else:
        padding_mask = None

    labels = torch.tensor(labels, dtype=torch.long)

    channels_last = getattr(opts, "common.channels_last")
    if channels_last:
        images = images.to(memory_format=torch.channels_last)

    return {
        "samples": {
            "image": images,
            "text": text_tokens,
            "padding_mask": padding_mask,
        },
        "targets": labels,
    }
