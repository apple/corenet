#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from numbers import Number
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import BaseMetric
from corenet.metrics.retrieval_cmc import DISTANCE_REGISTRY
from corenet.utils import logger
from corenet.utils.tensor_utils import all_gather_list


@METRICS_REGISTRY.register("image_text_retrieval")
class ImageTextRetrievalMetric(BaseMetric):
    """
    Computes the image-text retrieval metrics for a list of images and their captions
    using the distance between their embeddings.

    Expects predictions to contain two keys:
        image (Tensor): [batch, hidden_dim]
        text (Tensor): [batch * num_captions, hidden_dim]

    Computes the following metrics:
        image2text
            recall@1, recall@5, recall@10, mean_rank, median_rank
        text2image
            recall@1, recall@5, recall@10, mean_rank, median_rank

    NOTE: each image MUST have the same number of captions.
    """

    def __init__(
        self,
        image: str = "image",
        text: str = "text",
        opts: Dict[str, Any] = None,
        is_distributed: bool = False,
    ) -> None:
        # Ignoring pred_key and target_key as we won't be using them
        # The issue is, both text and image are in the prediction, so pred_key and
        # target_key don't make sense here. We can still use pred_key to support nested
        # dicts, but it didn't seem required.
        super().__init__(opts, is_distributed)
        self._image_key = image
        self._text_key = text

        distance_metric = getattr(
            opts, "stats.metrics.img_text_retrieval.distance_metric"
        )
        self.measure = DISTANCE_REGISTRY[distance_metric]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add metric specific arguments"""
        if cls == ImageTextRetrievalMetric:
            parser.add_argument(
                "--stats.metrics.img-text-retrieval.distance-metric",
                type=str,
                default="cosine",
                choices=list(DISTANCE_REGISTRY.keys()),
                help="Distance to use for nearest-neighbor calculation.",
            )
        return parser

    def reset(self) -> None:
        self._images = []
        self._texts = []

    def update(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
        batch_size: int = 1,
    ) -> None:
        images = prediction[self._image_key]
        texts = prediction[self._text_key]

        if not isinstance(images, Tensor) or not isinstance(texts, Tensor):
            logger.error(
                "ImageTextRetrievalMetric only works on Tensor, got {} and {}.".format(
                    type(images), type(texts)
                )
            )
            return

        if self.is_distributed:
            images = all_gather_list(images)
            texts = all_gather_list(texts)
        else:
            images = [images.detach()]
            texts = [texts.detach()]

        self._images.extend(images)
        self._texts.extend(texts)

    def get_aggregates(self) -> Tuple[Tensor, Tensor]:
        self._images = [torch.cat(self._images, dim=0)]
        self._texts = [torch.cat(self._texts, dim=0)]

        return self._images[0], self._texts[0]

    def _text2image(
        self, images: Tensor, texts: Tensor, num_captions: int
    ) -> torch.LongTensor:
        """
        Compute the distance between embeddings for text captions and their respective images.

        Args:
            image: A tensor of image embeddings. Shape: [batch, hidden_dim]
            text: A tensor of text embeddings. Shape: [batch * num_captions, hidden_dim]
            num_captions: The number of captions paired with a single image.

        Returns:
            A tensor containing ranks of the corresponding image among all images.
        """
        ranks = torch.zeros(images.shape[0], dtype=torch.long)
        for i, image in enumerate(images):
            # [1, hidden_dim] dist [batch * num_captions, hidden_dim] --> [batch * num_captions]
            # i.e. dists of size: [num_texts]
            dists = self.measure(image.unsqueeze(0), texts).squeeze(0)

            # find the rank of the best scoring caption among num_captions
            inds = torch.argsort(dists) // num_captions
            ranks[i] = (inds == i).nonzero()[0, 0]
        return ranks

    def _image2text(
        self, images: Tensor, texts: Tensor, num_captions: int
    ) -> torch.LongTensor:
        """
        Compute the distance between embeddings for images and their respective captions.

        Args:
            image: A tensor of image embeddings. Shape: [batch, hidden_dim]
            text: A tensor of text embeddings. Shape: [batch * num_captions, hidden_dim]
            num_captions: The number of captions paired with a single image.

        Returns:
            A tensor containing ranks of the closest caption to each image among all
            captions.
        """
        ranks = torch.zeros(texts.shape[0], dtype=torch.long)
        for i, text in enumerate(texts):
            # [1, hidden_dim] cos [batch, hidden_dim] --> [batch]
            # i.e. dists of size: [num_images]
            dists = self.measure(text.unsqueeze(0), images).squeeze(0)

            # find the rank of the corresponding image
            inds = torch.argsort(dists)
            ranks[i] = (inds == (i // num_captions)).nonzero()[0, 0]
        return ranks

    def compute(self) -> Union[Number, Dict[str, Number]]:
        # image: [batch, hidden_dim]
        # text:  [batch, num_captions, hidden_dim] or [batch * num_captions, hidden_dim]
        images, texts = self.get_aggregates()

        # make sure text shape is: [batch * num_captions, hidden_dim]
        if texts.dim() == 3:
            # [batch, num_captions, hidden_dim] --> [batch * num_captions, hidden_dim]
            texts = texts.reshape(-1, texts.shape[-1])

        num_images = images.shape[0]
        num_texts = texts.shape[0]
        assert num_texts % num_images == 0, "Number of captions is not consistent"
        num_captions = num_texts // num_images

        with torch.no_grad():
            i2t_ranks = self._image2text(images, texts, num_captions)
            t2i_ranks = self._text2image(images, texts, num_captions)

        return {
            "text2image": self._rank_metrics(t2i_ranks),
            "image2text": self._rank_metrics(i2t_ranks),
        }

    def _rank_metrics(self, ranks: torch.LongTensor) -> Dict[str, Number]:
        return {
            "recall@1": 100 * (ranks < 1).float().mean().item(),
            "recall@5": 100 * (ranks < 5).float().mean().item(),
            "recall@10": 100 * (ranks < 10).float().mean().item(),
            "mean_rank": 1 + ranks.float().mean().item(),
            "median_rank": 1 + ranks.median().item(),
        }
