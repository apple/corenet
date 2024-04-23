#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Any, Dict, Tuple, Union

import torch
from torchvision.datasets import ImageFolder

from corenet.data.datasets.dataset_base import BaseImageDataset
from corenet.data.datasets.utils.common import select_samples_by_category
from corenet.data.transforms import image_pil
from corenet.data.transforms.common import Compose
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master


class BaseImageClassificationDataset(BaseImageDataset, ImageFolder):
    """Image Classification Dataset.


    This base class can be used to represent any image classification dataset which is stored in a way that meets the
    expectations of `torchvision.datasets.ImageFolder`. New image classification datasets can be derived from this
    similar to ImageNetDataset (imagenet.py) or Places365Dataset (places365.py) and overwrite the data transformations
    as needed. This dataset also supports sampling a random subset of the training set to be used for training. The
    subset size is determined by the arguments `dataset.num_samples_per_category` and `dataset.percentage_of_samples`
    in the input `opts`. Only one of these two should be specified. When `dataset.percentage_of_samples` is specified,
    data is sampled from all classes according to this percentage such that the distribution of classes does not change.
     The randomness in sampling is controlled by the `dataset.sample_selection_random_seed` in the input `opts`.

    Args:
        opts: An argparse.Namespace instance.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        BaseImageDataset.__init__(
            self,
            opts=opts,
            *args,
            **kwargs,
        )
        root = self.root
        ImageFolder.__init__(
            self,
            root=root,
            transform=None,
            target_transform=None,
            is_valid_file=None,
        )

        self.n_classes = len(list(self.class_to_idx.keys()))

        master = is_master(self.opts)
        if master:
            logger.log("Number of categories: {}".format(self.n_classes))
            logger.log("Total number of samples: {}".format(len(self.samples)))

        num_samples_per_category = getattr(
            self.opts, "dataset.num_samples_per_category"
        )
        percentage_of_samples = getattr(self.opts, "dataset.percentage_of_samples")

        if self.is_training and (
            num_samples_per_category > 0 or (0 < percentage_of_samples < 100)
        ):
            if num_samples_per_category > 0 and (0 < percentage_of_samples < 100):
                raise ValueError(
                    "Both `dataset.num_samples_per_category` and `dataset.percentage_of_samples` are specified. "
                    "Please specify only one."
                )

            random_seed = getattr(self.opts, "dataset.sample_selection_random_seed")
            if num_samples_per_category > 0:
                selected_sample_indices = select_samples_by_category(
                    sample_category_labels=self.targets,
                    random_seed=random_seed,
                    num_samples_per_category=num_samples_per_category,
                )
                if master:
                    logger.log(
                        "Using {} samples per category.".format(
                            num_samples_per_category
                        )
                    )
            else:
                selected_sample_indices = select_samples_by_category(
                    sample_category_labels=self.targets,
                    random_seed=random_seed,
                    percentage_of_samples_per_category=percentage_of_samples,
                )
                if master:
                    logger.log(
                        "Using {} percentage of samples per category.".format(
                            percentage_of_samples
                        )
                    )

            self.samples = [self.samples[ind] for ind in selected_sample_indices]
            self.imgs = [self.imgs[ind] for ind in selected_sample_indices]
            self.targets = [self.targets[ind] for ind in selected_sample_indices]
        elif master:
            logger.log("Using all samples in the dataset.")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Adds dataset related arguments to the parser.

        Args:
            parser: An argparse.Namespace instance

        Returns:
            Input argparse.Namespace instance with additional arguments.
        """
        if cls != BaseImageClassificationDataset:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--dataset.num-samples-per-category",
            type=int,
            default=-1,
            help="Number of samples to use per category. If set to -1, all samples will be used.",
        )
        return parser

    def _training_transforms(
        self, size: Union[Tuple[int, int], int], *args, **kwargs
    ) -> image_pil.BaseTransformation:
        """
        Returns transformations applied to the input in training mode.

        Order of transformations: RandomResizedCrop, RandomHorizontalFlip, One of AutoAugment or RandAugment or
        TrivialAugmentWide, RandomErasing

        Batch-based augmentations such as Mixup and CutMix are implemented in trainer.

        Args:
            size: Size for resizing the input image. Expected to be an integer (width=height) or a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        if not getattr(self.opts, "image_augmentation.random_resized_crop.enable"):
            raise ValueError(
                "`image_augmentation.random_resized_crop.enable` must be set to True in input options."
            )

        aug_list = [image_pil.RandomResizedCrop(opts=self.opts, size=size)]

        if getattr(self.opts, "image_augmentation.random_horizontal_flip.enable"):
            aug_list.append(image_pil.RandomHorizontalFlip(opts=self.opts))

        auto_augment = getattr(self.opts, "image_augmentation.auto_augment.enable")
        rand_augment = getattr(self.opts, "image_augmentation.rand_augment.enable")
        trivial_augment_wide = getattr(
            self.opts, "image_augmentation.trivial_augment_wide.enable"
        )
        if bool(auto_augment) + bool(rand_augment) + bool(trivial_augment_wide) > 1:
            logger.error(
                "Only one of AutoAugment, RandAugment and TrivialAugmentWide should be used."
            )
        elif auto_augment:
            aug_list.append(image_pil.AutoAugment(opts=self.opts))
        elif rand_augment:
            if getattr(self.opts, "image_augmentation.rand_augment.use_timm_library"):
                aug_list.append(image_pil.RandAugmentTimm(opts=self.opts))
            else:
                aug_list.append(image_pil.RandAugment(opts=self.opts))
        elif trivial_augment_wide:
            aug_list.append(image_pil.TrivialAugmentWide(opts=self.opts))

        aug_list.append(image_pil.ToTensor(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_erase.enable"):
            aug_list.append(image_pil.RandomErasing(opts=self.opts))

        return Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, *args, **kwargs) -> image_pil.BaseTransformation:
        """
        Returns transformations applied to the input in validation mode.

        Oder of augmentations: Resize followed by CenterCrop
        """

        if not getattr(self.opts, "image_augmentation.resize.enable"):
            raise ValueError(
                "`image_augmentation.resize.enable` must be set to True in input options."
            )
        aug_list = [image_pil.Resize(opts=self.opts)]

        if not getattr(self.opts, "image_augmentation.center_crop.enable"):
            raise ValueError(
                "`image_augmentation.center_crop.enable` must be set to True in input options."
            )

        aug_list.append(image_pil.CenterCrop(opts=self.opts))
        aug_list.append(image_pil.ToTensor(opts=self.opts))

        return Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Returns the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Returns:
            A dictionary with `samples`, `sample_id` and `targets` as keys corresponding to input, index and label of
            a sample, respectively.

        Shapes:
            The output data dictionary contains three keys (samples, sample_id, and target). The values of these
            keys has the following shapes:
                data["samples"]: Shape is [Channels, Height, Width]
                data["sample_id"]: Shape is 1
                data["targets"]: Shape is 1
        """
        crop_size_h, crop_size_w, sample_index = sample_size_and_index
        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        img_path, target = self.samples[sample_index]
        input_img = self.read_image_pil(img_path)
        if input_img is None:
            # Sometimes images are corrupt
            # Skip such images
            logger.log("Img index {} is possibly corrupt.".format(sample_index))
            input_tensor = torch.zeros(
                size=(3, crop_size_h, crop_size_w), dtype=torch.float
            )
            target = -1
            data = {"image": input_tensor}
        else:
            data = {"image": input_img}
            data = transform_fn(data)

        data["samples"] = data.pop("image")
        data["targets"] = target
        data["sample_id"] = sample_index

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        return extra_repr_str + f"\n\t num_classes={self.n_classes}"
