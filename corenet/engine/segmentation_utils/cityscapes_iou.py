#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import glob
import os

import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_semseg_eval

from corenet.utils import logger


def eval_cityscapes(pred_dir: str, gt_dir: str) -> None:
    """Utility to evaluate on cityscapes dataset"""
    cityscapes_semseg_eval.args.predictionPath = pred_dir
    cityscapes_semseg_eval.args.predictionWalk = None
    cityscapes_semseg_eval.args.JSONOutput = False
    cityscapes_semseg_eval.args.colorized = False

    gt_img_list = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
    if len(gt_img_list) == 0:
        logger.error("Cannot find ground truth images at: {}".format(gt_dir))

    pred_img_list = []
    for gt in gt_img_list:
        pred_img_list.append(
            cityscapes_semseg_eval.getPrediction(cityscapes_semseg_eval.args, gt)
        )

    results = cityscapes_semseg_eval.evaluateImgLists(
        pred_img_list, gt_img_list, cityscapes_semseg_eval.args
    )

    logger.info("Evaluation results summary")
    eval_res_str = "\n\t IoU_cls: {:.2f} \n\t iIOU_cls: {:.2f} \n\t IoU_cat: {:.2f} \n\t iIOU_cat: {:.2f}".format(
        100.0 * results["averageScoreClasses"],
        100.0 * results["averageScoreInstClasses"],
        100.0 * results["averageScoreCategories"],
        100.0 * results["averageScoreInstCategories"],
    )
    print(eval_res_str)
