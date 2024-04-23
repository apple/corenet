#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycocotools import mask as maskUtils

from corenet.metrics import coco_map
from corenet.metrics.stats import Statistics
from corenet.modeling.models.detection import DetectionPredTuple


def test_coco_map():
    predictions = {
        "detections": [
            DetectionPredTuple(
                labels=torch.empty(0, dtype=torch.long),
                scores=torch.empty(0, dtype=torch.float32),
                boxes=torch.empty(0, 4, dtype=torch.float32),
            )
        ]
        * 3
    }
    targets = [
        {"image_id": 0, "image_width": 128, "image_height": 128},
        {"image_id": 1, "image_width": 256, "image_height": 256},
        {"image_id": 2, "image_width": 160, "image_height": 192},
    ]

    coco_annotations_path = os.path.join(os.path.dirname(__file__), "../data", "coco")
    opts = argparse.Namespace(**{"dataset.root_val": coco_annotations_path})
    stats = Statistics(opts=opts, metric_names=["coco_map"])
    stats.update(predictions, targets)
    bbox_map = stats.avg_statistics("coco_map", "bbox")

    np.testing.assert_equal(0.0, bbox_map)


def get_prediction_from_annotation(
    annotations: Dict, annotation_idx: int, category_mapping: Dict[int, int]
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    annotation = annotations["annotations"][annotation_idx]
    image_id = annotation["image_id"]
    image = [im for im in annotations["images"] if im["id"] == image_id]
    assert len(image) == 1, f"Too many images with a single id."
    image = image[0]
    h, w = image["height"], image["width"]
    boxes = annotation["bbox"]

    # Boxes are in (x1, y1, w, h) format. Convert to (x1, y1, x2, y2) format.
    boxes[0] /= w
    boxes[1] /= h
    boxes[2] /= w
    boxes[3] /= h

    boxes[2] += boxes[0]
    boxes[3] += boxes[1]

    masks = annotation["segmentation"]
    rles = maskUtils.frPyObjects(masks, h, w)
    rle = maskUtils.merge(rles)
    binary_mask = maskUtils.decode(rle)

    label = category_mapping[annotation["category_id"]]

    return label, boxes, binary_mask


def get_detection_pred_tuple_from_annotations(
    annotations: Dict, annotation_ids: List[int], category_mapping: Dict[int, int]
) -> DetectionPredTuple:
    labels = []
    scores = []
    boxes = []
    masks = []

    for annotation_id in annotation_ids:
        label, box, binary_mask = get_prediction_from_annotation(
            annotations, annotation_id, category_mapping
        )
        labels.append(label)
        scores.append(1.0)
        boxes.append(box)
        masks.append(binary_mask)

    labels = torch.tensor(labels, dtype=torch.int64)
    scores = torch.tensor(scores)
    boxes = torch.tensor(boxes)
    masks = torch.tensor(masks).float()

    return DetectionPredTuple(labels, scores, boxes, masks)


def test_map() -> None:
    opts = argparse.Namespace()
    setattr(opts, "stats.coco_map.iou_types", ["bbox", "segm"])
    setattr(opts, "dataset.root_val", "tests/data/coco")
    evaluator = coco_map.COCOEvaluator(opts)

    # Read annotations from the annotations file.
    annotations_file = "tests/data/coco/annotations/instances_val2017.json"
    with open(annotations_file) as f:
        annotations = json.load(f)

    predictions = {
        "detections": [
            get_detection_pred_tuple_from_annotations(
                annotations, [0], evaluator.coco_id_to_contiguous_id
            ),
            get_detection_pred_tuple_from_annotations(
                annotations, [1], evaluator.coco_id_to_contiguous_id
            ),
        ]
    }

    # These targets correspond to the first two annotations in our .json file.
    targets = [
        {
            "image_id": torch.tensor(37777),
            "image_width": torch.tensor(352),
            "image_height": torch.tensor(230),
        },
        {
            "image_id": torch.tensor(397133),
            "image_width": torch.tensor(640),
            "image_height": torch.tensor(427),
        },
    ]

    evaluator.update(predictions, targets)

    results = evaluator.summarize_coco_results()
    assert results == {"bbox": 99.99999999999997, "segm": 99.99999999999997}
