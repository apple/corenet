#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch import Tensor

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.detection.base_detection import BaseDetectionDataset
from corenet.data.transforms import image_pil as T
from corenet.data.transforms.common import Compose
from corenet.utils import logger


@DATASET_REGISTRY.register(name="coco", type="detection")
class COCODetection(BaseDetectionDataset):
    """Base class for the MS COCO Object Detection Dataset. Sub-classes should implement
    training and validation transform functions.

    Args:
        opts: command-line arguments

    .. note::
        This class implements basic functions (e.g., reading image and annotations), and does not implement
        training/validation transforms. Detector specific sub-classes should extend this class and implement those
        methods. See `coco_ssd.py` as an example for SSD.
    """

    def __init__(
        self,
        opts,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)

        split = "train" if self.is_training else "val"
        year = 2017
        ann_file = os.path.join(
            self.root, "annotations/instances_{}{}.json".format(split, year)
        )

        # disable printing, so that pycocotools print statements are not printed
        logger.disable_printing()

        self.coco = COCO(ann_file)
        self.img_dir = os.path.join(self.root, "images/{}{}".format(split, year))
        self.ids = (
            list(self.coco.imgToAnns.keys())
            if self.is_training
            else list(self.coco.imgs.keys())
        )

        coco_categories = sorted(self.coco.getCatIds())
        background_idx = 0 if getattr(opts, "dataset.detection.no_background_id") else 1
        self.coco_id_to_contiguous_id = {
            coco_id: i + background_idx for i, coco_id in enumerate(coco_categories)
        }
        self.contiguous_id_to_coco_id = {
            v: k for k, v in self.coco_id_to_contiguous_id.items()
        }
        self.num_classes = len(self.contiguous_id_to_coco_id.keys()) + background_idx

        # enable printing
        logger.enable_printing()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != COCODetection:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.detection.no-background-id",
            action="store_true",
            default=False,
            help="Do not include background id in detection class labels. Defaults to False.",
        )
        return parser

    def _evaluation_transforms(
        self, size: tuple, *args, **kwargs
    ) -> T.BaseTransformation:
        """Evaluation or Inference transforms (Resize (Optional) --> Tensor).

        .. note::
            Resizing the input to the same resolution as the detector's input is not enabled by default.
            It can be enabled by passing **--evaluation.detection.resize-input-images** flag.

        """
        aug_list = []
        if getattr(self.opts, "evaluation.detection.resize_input_images"):
            aug_list.append(T.Resize(opts=self.opts, img_size=size))

        aug_list.append(T.ToTensor(opts=self.opts))
        return Compose(opts=self.opts, img_transforms=aug_list)

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
            output_data["targets"]["box_labels"]: Shape is [Num of boxes]
            output_data["targets"]["box_coordinates"]: Shape is [Num of boxes, 4]
            output_data["targets"]["image_id"]: Shape is [1]
            output_data["targets"]["image_width"]: Shape is [1]
            output_data["targets"]["image_height"]: Shape is [1]
        """

        crop_size_h, crop_size_w, img_index = sample_size_and_index

        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_name = self.get_image(image_id=image_id)
        im_width, im_height = image.size

        boxes, labels, mask = self.get_boxes_and_labels(
            image_id=image_id,
            image_width=im_width,
            image_height=im_height,
            include_masks=True,
        )

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes,
            "mask": mask,
        }

        if transform_fn is not None:
            data = transform_fn(data)

        output_data = {
            "samples": {
                "image": data["image"],
            },
            "targets": {
                "box_labels": data["box_labels"],
                "box_coordinates": data["box_coordinates"],
                "mask": data["mask"],
                "image_id": torch.tensor(image_id),
                "image_width": torch.tensor(im_width),
                "image_height": torch.tensor(im_height),
            },
        }

        return output_data

    def __len__(self):
        return len(self.ids)

    def get_boxes_and_labels(
        self,
        image_id: int,
        image_width: int,
        image_height: int,
        *args,
        include_masks=False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get the boxes and label information for a given image_id

        Args:
            image_id: Image ID
            image_width: Width of the image
            image_height: Height of the image
            include_masks: Return instance masks or not

        Returns:
            A tuple of length 3:
                * Numpy array containing bounding box information in xyxy format.
                    The shape of array is [Num_of_boxes, 4].
                * Numpy array containing labels for each of the box. The shape of array is [Num_of_boxes]
                * When include_masks is enabled, a numpy array of instance masks is returned. The shape of the
                    array is [Num_of_boxes, image_height, image_width]

        """
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)

        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array(
            [self._xywh2xyxy(obj["bbox"], image_width, image_height) for obj in ann],
            np.float32,
        ).reshape((-1, 4))
        labels = np.array(
            [self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann],
            np.int64,
        ).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]

        masks = None
        if include_masks:
            masks = []
            for obj in ann:
                rle = coco_mask.frPyObjects(
                    obj["segmentation"], image_height, image_width
                )
                m = coco_mask.decode(rle)
                if len(m.shape) < 3:
                    mask = m.astype(np.uint8)
                else:
                    mask = (np.sum(m, axis=2) > 0).astype(np.uint8)
                masks.append(mask)

            if len(masks) > 0:
                masks = np.stack(masks, axis=0)
            else:
                masks = np.zeros(shape=(0, image_height, image_width), dtype=np.uint8)
            masks = masks.astype(np.uint8)
            masks = torch.from_numpy(masks)
            masks = masks[keep]
            assert len(boxes) == len(labels) == len(masks)
            return boxes, labels, masks
        else:
            return boxes, labels, None

    def _xywh2xyxy(
        self, box: List[int], image_width: int, image_height: int
    ) -> List[int]:
        """Convert boxes from xywh format to xyxy format"""
        x1, y1, w, h = box
        return [
            max(0, x1),
            max(0, y1),
            min(x1 + w, image_width),
            min(y1 + h, image_height),
        ]

    def get_image(self, image_id: int) -> Tuple:
        """Return the PIL image for a given image id"""
        file_name = self.coco.loadImgs(image_id)[0]["file_name"]
        image_file = os.path.join(self.img_dir, file_name)
        image = self.read_image_pil(image_file)
        return image, file_name

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\n\t num_classes={self.num_classes}"

    @staticmethod
    def class_names() -> List[str]:
        """Name of the classes in the COCO dataset"""
        return [
            "background",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
