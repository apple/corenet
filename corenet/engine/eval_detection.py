#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import glob
import os.path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
from tqdm import tqdm

from corenet.constants import SUPPORTED_IMAGE_EXTNS
from corenet.data import create_test_loader
from corenet.data.datasets.detection.coco_base import COCODetection
from corenet.engine.detection_utils.coco_map import compute_quant_scores
from corenet.engine.utils import autocast_fn
from corenet.modeling import get_model
from corenet.modeling.models.detection import DetectionPredTuple
from corenet.modeling.models.detection.base_detection import BaseDetection
from corenet.options.opts import get_training_arguments
from corenet.utils import logger, resources
from corenet.utils.common_utils import create_directories, device_setup
from corenet.utils.ddp_utils import is_master
from corenet.utils.download_utils import get_local_path
from corenet.utils.tensor_utils import image_size_from_opts, to_numpy
from corenet.utils.visualization_utils import draw_bounding_boxes

# Evaluation on MSCOCO detection task
object_names = COCODetection.class_names()


def predict_and_save(
    opts,
    input_tensor: Tensor,
    model: BaseDetection,
    input_np: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
    is_coco_evaluation: Optional[bool] = False,
    file_name: Optional[str] = None,
    output_stride: int = 32,
    orig_h: Optional[int] = None,
    orig_w: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Predict the detection outputs (e.g., bounding box) and optionally save them.

    Args:
        opts: Command-line arguments.
        input_tensor: Input image tensor of shape [batch size, image channels, height, width].
        model: Detection model.
        input_np: Input image numpy array of shape [original height, original width, image channels]. The 'input_np' does not
            go under any pre-processing and is useful for visualization purposes. When it is `None' and 'is_coco_evaluation' is
            not enabled, then 'input_np' is obtained from 'input_tensor'.
        device: Inference device.
        is_coco_evaluation: Evaluating on MS-COCO object detection or not.
        file_name: Name of the output image file. If it is not specified, then output will not be saved.
        output_stride: Output stride. Checks if input dimensions are multiple of output stride or not. This argument
            is used to avoid dimension mismatch errors.
        orig_h: Original image height. This may be different from @input_tensor's height because of image transforms. Useful
            for visualizing detection results.
        orig_w: Original image width. This may be different from @input_tensor's width because of image transforms. Useful
            for visualizing detection results.

    Returns:
        When 'is_coco_evaluation' is enabled, the function returns a tuple that includes bounding boxes for each object,
        predicted labels for each box, predicted scores for each box, and instance masks for each box.
        Otherwise, nothing is returned.
    """
    mixed_precision_training = getattr(opts, "common.mixed_precision")
    mixed_precision_dtype = getattr(opts, "common.mixed_precision_dtype")

    if input_np is None and not is_coco_evaluation:
        input_np = to_numpy(input_tensor).squeeze(  # convert to numpy
            0
        )  # remove batch dimension

    curr_height, curr_width = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_height // output_stride) * output_stride
    new_w = (curr_width // output_stride) * output_stride

    if new_h != curr_height or new_w != curr_width:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(
            input=input_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    # move data to device
    input_tensor = input_tensor.to(device)

    with autocast_fn(
        enabled=mixed_precision_training, amp_precision=mixed_precision_dtype
    ):
        # prediction
        # We dot scale inside the prediction function because we resize the input tensor such
        # that the dimensions are divisible by output stride.
        prediction: DetectionPredTuple = model.predict(input_tensor, is_scaling=False)

    if orig_w is None:
        assert orig_h is None
        orig_h, orig_w = input_np.shape[:2]
    elif orig_h is None:
        assert orig_w is None
        orig_h, orig_w = input_np.shape[:2]
    assert orig_h is not None and orig_w is not None

    # convert tensors to numpy
    boxes = prediction.boxes.cpu().numpy()
    labels = prediction.labels.cpu().numpy()
    scores = prediction.scores.cpu().numpy()

    masks = prediction.masks

    # Ensure that there is at least one mask
    if masks is not None and masks.shape[0] > 0:
        # masks are in [N, H, W] format
        # for interpolation, add a dummy batch dimension
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        # convert to binary masks
        masks = masks > 0.5
        masks = masks.cpu().numpy()

    boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2] * orig_w)
    boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2] * orig_h)

    if is_coco_evaluation:
        return boxes, labels, scores, masks

    detection_res_file_name = None
    if file_name is not None:
        file_name = file_name.split(os.sep)[-1].split(".")[0] + ".jpg"
        res_dir = "{}/detection_results".format(getattr(opts, "common.exp_loc", None))
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        detection_res_file_name = "{}/{}".format(res_dir, file_name)

    draw_bounding_boxes(
        image=input_np,
        boxes=boxes,
        labels=labels,
        scores=scores,
        masks=masks,
        # some models may not use background class which is present in class names.
        # adjust the class names
        object_names=(
            object_names[-model.n_detection_classes :]
            if hasattr(model, "n_detection_classes")
            else object_names
        ),
        is_bgr_format=True,
        save_path=detection_res_file_name,
    )


def read_and_process_image(
    opts: argparse.Namespace, file_path: str
) -> Tuple[Tensor, np.ndarray, int, int]:
    """Read and converts the input image into a tensor.

    Args:
        opts: Command-line arguments.
        file_path: Path of the image file.

    Returns:
        A tuple containing:
            1. Image tensor with shape [1, C, H, W]. The tensor values are normalized between 0 and 1.
            2. Original RGB image as numpy array.
            3. Height of the original image.
            4. Width of the original image.
    """
    input_img = Image.open(file_path).convert("RGB")
    input_np = np.array(input_img)
    orig_w, orig_h = input_img.size

    # Resize the image to the resolution that detector supports
    res_h, res_w = image_size_from_opts(opts)
    input_img = F_vision.resize(
        input_img,
        size=[res_h, res_w],
        interpolation=F_vision.InterpolationMode.BILINEAR,
    )
    input_tensor = F_vision.pil_to_tensor(input_img)
    input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
    return input_tensor, input_np, orig_h, orig_w


@torch.no_grad()
def predict_labeled_dataset(opts: argparse.Namespace) -> None:
    """Generate predictions for the labeled dataset.

    This function predicts detection outputs for RGB input images and calculates detection accuracy, measured
    using COCO API, through comparison between predicted detection outputs and ground truth labels.

    Args:
        opts: Command-line arguments.
    """
    device = getattr(opts, "dev.device")

    # set-up data loaders
    test_loader = create_test_loader(opts)

    # set-up the model
    model = get_model(opts)
    model.eval()
    model.info()
    model = model.to(device=device)

    predictions = []
    for batch in tqdm(test_loader):
        samples, targets = batch["samples"], batch["targets"]

        if isinstance(samples, Dict):
            assert (
                "image" in samples
            ), "Samples does not contain 'image' key. Please check."
            input_tensor = samples["image"]
        elif isinstance(samples, torch.Tensor):
            input_tensor = samples
        else:
            raise NotImplementedError(
                "Only dictionary and tensor types are supported for the input sample."
            )

        orig_w = targets["image_width"].item()
        orig_h = targets["image_height"].item()
        image_id = targets["image_id"].item()

        boxes, labels, scores, masks = predict_and_save(
            opts=opts,
            input_tensor=input_tensor,
            model=model,
            device=device,
            is_coco_evaluation=True,
            orig_w=orig_w,
            orig_h=orig_h,
        )

        predictions.append([image_id, boxes, labels, scores, masks])

    compute_quant_scores(opts=opts, predictions=predictions)


@torch.no_grad()
def predict_image(opts: argparse.Namespace, file_path: str) -> None:
    """Generate predictions for an RGB image.

    Note that the predictions are saved in 'common.results_loc' directory.

    Args:
        opts: Command-line arguments.
        file_path: Path of the image file.
    """
    local_file_path = get_local_path(opts, file_path)
    if not os.path.isfile(local_file_path):
        logger.error("Image file does not exist at: {}".format(local_file_path))

    input_tensor, input_imp_copy, orig_h, orig_w = read_and_process_image(
        opts, local_file_path
    )

    file_name = os.path.basename(local_file_path)

    device = getattr(opts, "dev.device")
    # set-up the model
    model = get_model(opts)
    model.eval()
    model.info()
    model = model.to(device=device)

    predict_and_save(
        opts=opts,
        input_tensor=input_tensor,
        input_np=input_imp_copy,
        file_name=file_name,
        model=model,
        device=device,
        orig_h=orig_h,
        orig_w=orig_w,
    )


@torch.no_grad()
def predict_images_in_folder(opts: argparse.Namespace) -> None:
    """Generate predictions for all RGB images stored in a folder.

    The prediction outputs are saved in 'common.results_loc' directory.

    Args:
        opts: Command-line arguments.

    ...note:
        The path of the folder containong RGB images is supplied using 'evaluation.detection.path' argument.
    """
    img_folder_path = getattr(opts, "evaluation.detection.path")
    if img_folder_path is None:
        logger.error(
            "Image folder is not passed. Please use 'evaluation.detection.path' argument to pass the location of image folder."
        )
    elif not os.path.isdir(img_folder_path):
        logger.error(f"Image folder does not exist at {img_folder_path}. Please check.")

    img_files = []
    for e in SUPPORTED_IMAGE_EXTNS:
        img_files_with_extn = glob.glob("{}/*{}".format(img_folder_path, e))
        if len(img_files_with_extn) > 0 and isinstance(img_files_with_extn, list):
            img_files.extend(img_files_with_extn)

    if len(img_files) == 0:
        logger.error(f"Image files are not found at {img_folder_path}. Please check.")

    logger.log(
        f"Number of image files found at {img_folder_path} are {len(img_files)}."
    )

    device = getattr(opts, "dev.device")
    # set-up the model
    model = get_model(opts)
    model.eval()
    model.info()
    model = model.to(device=device)

    for image_fname in tqdm(img_files):
        input_tensor, input_np, orig_h, orig_w = read_and_process_image(
            opts, image_fname
        )

        predict_and_save(
            opts=opts,
            input_tensor=input_tensor,
            input_np=input_np,
            file_name=os.path.basename(image_fname),
            model=model,
            device=device,
            orig_h=orig_h,
            orig_w=orig_w,
        )


def main_detection_evaluation(args: Optional[List[str]] = None) -> None:
    """Entrypoint for detection evaluation."""
    opts = get_training_arguments(args=args)

    dataset_name = getattr(opts, "dataset.name")
    if dataset_name.find("coco") > -1:
        # replace model specific datasets (e.g., coco_ssd) with base COCO dataset
        setattr(opts, "dataset.name", "coco")
        logger.log(
            "For evaluation, we use a base coco dataset class instead of model-specific COCO dataset class."
        )

    # device set-up
    opts = device_setup(opts)
    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc")
    run_label = getattr(opts, "common.run_label")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    logger.log("Results (if any) will be stored here: {}".format(exp_dir))

    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus")
    assert (
        num_gpus <= 1
    ), f"Testing of detection models is supported either on CPUs or a single GPU. Got: {num_gpus}."

    # DDP is required neither for a single GPU task nor for a CPU-only task. Therefore, it is disabled.
    # If it is required, please ensure that the DDP environment is properly initialized.
    setattr(opts, "ddp.use_distributed", False)

    dataset_workers = getattr(opts, "dataset.workers")

    if dataset_workers == -1:
        # No of data workers = no of CPUs (if not specified or -1)
        setattr(opts, "dataset.workers", resources.cpu_count())
        logger.log("Setting number of dataset workers as number of available CPUs.")

    eval_mode = getattr(opts, "evaluation.detection.mode")

    if eval_mode == "single_image":
        # test a single image
        img_f_name = getattr(opts, "evaluation.detection.path")
        predict_image(opts, img_f_name)
    elif eval_mode == "image_folder":
        # test all images in a folder
        predict_images_in_folder(opts=opts)
    elif eval_mode == "validation_set":
        assert (
            getattr(opts, "dataset.eval_batch_size0") == 1
        ), "For evaluation on validation set, we need a batch size of 1."
        # evaluate and compute stats for labeled image dataset
        # This is useful for generating results for validation set and compute quantitative results
        predict_labeled_dataset(opts=opts)
    else:
        logger.error(
            f"Supported modes are 'single_image', 'image_folder', and 'validation_set'. Got: {eval_mode}."
        )
