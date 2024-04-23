#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from corenet.data.datasets.dataset_base import BaseImageDataset
from corenet.data.transforms import image_pil as T
from corenet.data.transforms.common import Compose
from corenet.utils import logger
from corenet.utils.color_map import Colormap


class BaseImageSegmentationDataset(BaseImageDataset):
    """Base Dataset class for Image Segmentation datasets. Sub-classes must define `ignore_label`
    and `background_idx` variable.

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.masks = None
        self.images = None

        # ignore label and background indices are dataset specific. So, child classes
        # need to implement these
        self.ignore_label = None
        self.background_idx = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseImageSegmentationDataset:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)

        # segmentation evaluation related arguments
        group.add_argument(
            "--evaluation.segmentation.apply-color-map",
            action="store_true",
            default=False,
            help="Apply color map to different classes in segmentation masks. Useful in visualization "
            "+ some competitions (e.g, PASCAL VOC) accept submissions with colored segmentation masks."
            "Defaults to False.",
        )
        group.add_argument(
            "--evaluation.segmentation.save-overlay-rgb-pred",
            action="store_true",
            default=False,
            help="Enable this flag to visualize predicted masks on top of input image. "
            "Defaults to False.",
        )
        group.add_argument(
            "--evaluation.segmentation.save-masks",
            action="store_true",
            default=False,
            help="Save predicted masks without colormaps. Useful for submitting to "
            "competitions like Cityscapes. Defaults to False.",
        )
        group.add_argument(
            "--evaluation.segmentation.overlay-mask-weight",
            default=0.5,
            type=float,
            help="Contribution of mask when overlaying on top of RGB image. Defaults to 0.5.",
        )
        group.add_argument(
            "--evaluation.segmentation.mode",
            type=str,
            default="validation_set",
            choices=["single_image", "image_folder", "validation_set"],
            help="Contribution of mask when overlaying on top of RGB image. Defaults to validation_set.",
        )
        group.add_argument(
            "--evaluation.segmentation.path",
            type=str,
            default=None,
            help="Path of the image or image folder (only required for single_image and image_folder modes). "
            "Defaults to None.",
        )
        group.add_argument(
            "--evaluation.segmentation.resize-input-images",
            action="store_true",
            default=False,
            help="Enable resizing input images while maintaining aspect ratio during segmentation evaluation."
            "Defaults to False.",
        )

        group.add_argument(
            "--evaluation.segmentation.resize-input-images-fixed-size",
            action="store_true",
            default=False,
            help="Enable resizing input images to fixed size during segmentation evaluation. "
            "Defaults to False.",
        )
        return parser

    def check_dataset(self) -> None:
        # TODO: Remove this check in future
        assert self.masks is not None, "Please specify masks for segmentation data"
        assert self.images is not None, "Please specify images for segmentation data"
        assert (
            self.ignore_label is not None
        ), "Please specify ignore label for segmentation dataset"
        assert (
            self.background_idx is not None
        ), "Please specify background index for segmentation dataset"

    def _training_transforms(self, size: Tuple[int, int]) -> T.BaseTransformation:
        """Data augmentation during training.

        Order of transformation is RandomShortSizeResize, RandomHorizontalFlip, RandomCrop,
            Optional[RandomGaussianBlur], Optional[PhotometricDistort], Optional[RandomRotate].

        If random order is enabled, then the order of transforms is shuffled, with an exception to RandomShortSizeResize.
        These transforms are followed by ToTensor.

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        first_aug = T.RandomShortSizeResize(opts=self.opts)
        aug_list = [
            T.RandomHorizontalFlip(opts=self.opts),
            T.RandomCrop(opts=self.opts, size=size, ignore_idx=self.ignore_label),
        ]

        if getattr(self.opts, "image_augmentation.random_gaussian_noise.enable"):
            aug_list.append(T.RandomGaussianBlur(opts=self.opts))

        if getattr(self.opts, "image_augmentation.photo_metric_distort.enable"):
            aug_list.append(T.PhotometricDistort(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_rotate.enable"):
            aug_list.append(T.RandomRotate(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_order.enable"):
            new_aug_list = [
                first_aug,
                T.RandomOrder(opts=self.opts, img_transforms=aug_list),
                T.ToTensor(opts=self.opts),
            ]
            return Compose(opts=self.opts, img_transforms=new_aug_list)
        else:
            aug_list.insert(0, first_aug)
            aug_list.append(T.ToTensor(opts=self.opts))
            return Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(
        self, size: Tuple[int, int], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data augmentation during validation.

        Order of transformation is Resize, ToTensor

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        aug_list = [T.Resize(opts=self.opts), T.ToTensor(opts=self.opts)]
        return Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(
        self, size: Union[int, Tuple[int, int]], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data augmentation during testing/evaluation.

        Order of transformation is Optional[Resize], ToTensor

        Args:
            size: Size for resizing the input image. Expected to be an int or a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`

        ...note::
            When `evaluation.segmentation.resize_input_images` is enabled, then images are resized
                while maintaining the aspect ratio. If size is a tuple of integers, then min(size)
                is used as a size.

            When `evaluation.segmentation.resize_input_images_fixed_size` is enabled, then images
                are resized to the given size.

        """
        aug_list = []

        resize_maintain_ar = getattr(
            self.opts, "evaluation.segmentation.resize_input_images"
        )
        resize_fixed_size = getattr(
            self.opts, "evaluation.segmentation.resize_input_images_fixed_size"
        )

        if resize_maintain_ar:
            assert resize_fixed_size is False
            # A standard practice for tasks of segmentation is to resize images while maintaining
            # aspect ratio. To do so during evaluation, we pass min(img_size) as size as an
            # argument to resize function. The resize function then resizes image while
            # maintaining aspect ratio.
            aug_list.append(T.Resize(opts=self.opts, img_size=min(size)))
        elif resize_fixed_size:
            assert resize_maintain_ar is False
            # we want to resize while maintaining aspect ratio. So, we pass size as an
            # argument to resize function
            aug_list.append(T.Resize(opts=self.opts, img_size=size))
        # default is no resizing
        aug_list.append(T.ToTensor(opts=self.opts))
        return Compose(opts=self.opts, img_transforms=aug_list)

    @staticmethod
    def adjust_mask_value() -> int:
        """Adjust the mask value by this factor"""
        # Some datasets (e.g., ADE20k) requires us to adjust the mask value.
        # By default, we set to 0. But child classes can adjust it
        return 0

    def __len__(self) -> int:
        """Number of samples in segmentation dataset"""
        return len(self.images)

    @staticmethod
    def color_palette() -> List[int]:
        """Class index to RGB color mapping. The list index corresponds to class id.
        Note that the color list is flattened."""
        # Child classes may override this method (optionally)
        return Colormap().get_color_map_list()

    @staticmethod
    def class_names() -> List[str]:
        """Class index to name. The list index should correspond to class id"""
        # Child classes may implement these methods (optionally)
        raise NotImplementedError

    @staticmethod
    def read_mask_pil(path: str) -> Optional[Image.Image]:
        """Reads mask image and returns as a PIL image"""
        try:
            mask = Image.open(path)
            if mask.mode != "L":
                logger.error("Mask mode should be L. Got: {}".format(mask.mode))
            return mask
        except:
            return None

    @staticmethod
    def convert_mask_to_tensor(mask: Image.Image) -> Tensor:
        """Convert PIL mask to Tensor"""
        # convert to tensor
        mask = np.array(mask)
        if len(mask.shape) > 2 and mask.shape[-1] > 1:
            mask = np.ascontiguousarray(mask.transpose(2, 0, 1))
        return torch.as_tensor(mask, dtype=torch.long)

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int], *args, **kwargs
    ) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
        """Returns the sample corresponding to the input sample index. Returned sample is transformed
        into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Returns:
            A dictionary with `samples` and `targets` as keys corresponding to input and labels of
            a sample, respectively.

        Shapes:
            The shape of values in output dictionary, output_data, are as follows:

            output_data["samples"]["image"]: Shape is [Channels, Height, Width]
            output_data["targets"]["mask"]: Shape is [Height, Width]
        """

        crop_size_h, crop_size_w, img_index = sample_size_and_index
        transform = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        mask = self.read_mask_pil(self.masks[img_index])
        img = self.read_image_pil(self.images[img_index])

        if (img.size[0] != mask.size[0]) or (img.size[1] != mask.size[1]):
            logger.error(
                "Input image and mask sizes are different. Input size: {} and Mask size: {}".format(
                    img.size, mask.size
                )
            )

        data = {"image": img}
        if not self.is_evaluation:
            data["mask"] = mask

        data = transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = self.convert_mask_to_tensor(mask)

        output_data = {
            "samples": data["image"],
            # ignore dataset specific indices in mask
            "targets": data["mask"] - self.adjust_mask_value(),
        }

        if self.is_evaluation:
            im_width, im_height = img.size
            img_name = self.images[img_index].split(os.sep)[-1].replace("jpg", "png")
            mask = output_data.pop("targets")
            output_data["targets"] = {
                "mask": mask,
                "file_name": img_name,
                "im_width": im_width,
                "im_height": im_height,
            }

        return output_data
