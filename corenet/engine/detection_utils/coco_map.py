#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import io
import os
from contextlib import redirect_stdout
from typing import List, Optional

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from corenet.utils import logger


def coco_evaluation(
    opts,
    predictions: List[np.ndarray],
    split: Optional[str] = "val",
    year: Optional[int] = 2017,
    *args,
    **kwargs
) -> None:
    root = getattr(opts, "dataset.root_val", None)
    ann_file = os.path.join(root, "annotations/instances_{}{}.json".format(split, year))
    bkrnd_id = 0 if getattr(opts, "dataset.detection.no_background_id", False) else 1
    coco = COCO(ann_file)
    coco_categories = sorted(coco.getCatIds())

    coco_id_to_contiguous_id = {
        coco_id: i + bkrnd_id for i, coco_id in enumerate(coco_categories)
    }
    contiguous_id_to_coco_id = {v: k for k, v in coco_id_to_contiguous_id.items()}

    coco_results = {"bbox": []}

    for i, (image_id, boxes, labels, scores, masks) in enumerate(predictions):
        if labels.shape[0] == 0:
            continue

        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()

        coco_results["bbox"].extend(
            [
                {
                    "image_id": image_id,
                    "category_id": contiguous_id_to_coco_id[labels[k]],
                    "bbox": [
                        box[0],
                        box[1],
                        box[2] - box[0],
                        box[3] - box[1],
                    ],  # to xywh format
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

        if masks is not None:
            if "segm" not in coco_results:
                coco_results["segm"] = []

            # Masks are in [N, H, W] format
            rles = [
                maskUtils.encode(
                    np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]

            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            coco_results["segm"].extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": contiguous_id_to_coco_id[labels[seg_id]],
                        "segmentation": rle,
                        "score": scores[seg_id],
                    }
                    for seg_id, rle in enumerate(rles)
                ]
            )

    if len(coco_results) == 0:
        logger.error("Cannot compute COCO stats. Please check the predictions")

    for iou_type, coco_result in coco_results.items():
        with redirect_stdout(io.StringIO()):
            coco_dt = COCO.loadRes(coco, coco_result)

        # Run COCO evaluation
        coco_eval = COCOeval(coco, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def compute_quant_scores(opts, predictions: List, *args, **kwargs) -> None:
    dataset_name = getattr(opts, "dataset.name", None)
    if dataset_name.find("coco") > -1:
        coco_evaluation(opts=opts, predictions=predictions)
    else:
        raise NotImplementedError
