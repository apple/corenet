#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import copy
import glob
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
from tqdm import tqdm

from corenet.constants import SUPPORTED_IMAGE_EXTNS
from corenet.data import create_test_loader
from corenet.engine.utils import autocast_fn
from corenet.metrics.confusion_mat import ConfusionMatrix
from corenet.modeling import get_model
from corenet.modeling.models import BaseAnyNNModel
from corenet.options.opts import get_training_arguments
from corenet.utils import logger, resources
from corenet.utils.color_map import Colormap
from corenet.utils.common_utils import create_directories, device_setup
from corenet.utils.ddp_utils import is_master
from corenet.utils.download_utils import get_local_path
from corenet.utils.tensor_utils import image_size_from_opts
from corenet.utils.visualization_utils import convert_to_cityscape_format

"""
Notes:

1) We have separate scripts for evaluating segmentation models because the size of input images varies and
we do not want to apply any resizing operations to input because that distorts the quality and hurts the performance.

2) [Optional] We want to save the outputs in the same size as that of the input image.
"""


def predict_and_save(
    opts: argparse.Namespace,
    input_tensor: Tensor,
    file_name: str,
    orig_h: int,
    orig_w: int,
    model: BaseAnyNNModel,
    target_mask: Optional[Tensor] = None,
    device: torch.device = torch.device("cpu"),
    conf_mat: Optional[ConfusionMatrix] = None,
    color_map: List = None,
    orig_image: Optional[Image.Image] = None,
    adjust_label: Optional[int] = 0,
    is_cityscape: bool = False,
) -> None:
    """Predict the segmentation mask and optionally save them.

    Args:
        opts: Command-line arguments.
        input_tensor: Input image tensor of shape [batch size, image channels, height, width].
        file_name: Name of the image file.
        orig_h: Original image height. This may be different from @input_tensor's height because of image transforms.
        orig_w: Original image width. This may be different from @input_tensor's width because of image transforms.
        model: Segmentation model.
        target_mask: Optional target segmentation mask. If supplied, it will be used for computing confusion matrix with model's mask prediction.
        device: Inference device.
        conf_mat: Instance of ConfusionMatrix.
        color_map: Color mapping for converting predicted binary masks into colored representations.
        orig_image: The original RGB image, if provided, is used for overlaying segmentation predictions.
        adjust_label: Increase the predicted and target mask value by this factor.
        is_cityscape: Generating predictions on Cityscape dataset.

    ...note:
        1. If 'evaluation.segmentation.apply_color_map' is enabled using command-line arguments or configuration file,
            then masks are also converted to colored representations.
        2. If 'evaluation.segmentation.save_masks' is enabled using command-line arguments or configuration file,
            then binary masks are also saved.

    """

    mixed_precision_training = getattr(opts, "common.mixed_precision")
    mixed_precision_dtype = getattr(opts, "common.mixed_precision_dtype")

    output_stride = getattr(opts, "model.segmentation.output_stride")
    if output_stride == 1:
        # we set it to 32 because most of the ImageNet models have 5 downsampling stages (2^5 = 32)
        output_stride = 32

    if orig_image is None:
        orig_image = F_vision.to_pil_image(input_tensor[0])

    curr_h, curr_w = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_h // output_stride) * output_stride
    new_w = (curr_w // output_stride) * output_stride

    if new_h != curr_h or new_w != curr_w:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(
            input=input_tensor, size=(new_h, new_w), mode="bilinear", align_corners=True
        )

    file_name = file_name.split(os.sep)[-1].split(".")[0] + ".png"

    # move data to device
    input_tensor = input_tensor.to(device)
    if target_mask is not None:
        target_mask = target_mask.to(device)

    with autocast_fn(
        enabled=mixed_precision_training, amp_precision=mixed_precision_dtype
    ):
        # prediction
        pred = model(input_tensor, orig_size=(orig_h, orig_w))

    if isinstance(pred, Tuple) and len(pred) == 2:
        # when segmentation mask from decoder and auxiliary decoder are returned
        pred = pred[0]
    elif isinstance(pred, Tensor):
        pred = pred
    else:
        raise NotImplementedError(
            "Predicted must should be either an instance of Tensor or Tuple[Tensor, Tensor]"
        )

    num_classes = pred.shape[1]
    pred_mask = pred.argmax(1).squeeze(0)

    if target_mask is not None and conf_mat is not None:
        conf_mat.update(
            target=target_mask,
            prediction=pred,
        )

    save_dir = getattr(opts, "common.exp_loc")
    # For some dataset, we need to adjust the labels. For example, we need adjust by 1 for ADE20k
    pred_mask = pred_mask + adjust_label
    if target_mask is not None:
        target_mask = target_mask + adjust_label

    # Visualize results
    if getattr(opts, "evaluation.segmentation.apply_color_map"):

        apply_color_map_to_segmentation_masks(
            opts=opts,
            orig_image=orig_image,
            pred_mask=pred_mask,
            target_mask=target_mask,
            save_dir=save_dir,
            color_map=color_map,
            file_name=file_name,
        )

    if getattr(opts, "evaluation.segmentation.save_masks"):
        save_segmentation_masks(
            pred_mask=pred_mask,
            file_name=file_name,
            is_cityscape=is_cityscape,
            save_dir=save_dir,
        )


def save_segmentation_masks(
    pred_mask: Tensor,
    file_name: str,
    save_dir: str,
    is_cityscape: Optional[bool] = False,
) -> None:
    """Save predicted segmentation masks.

    The value of pixels in the segmentation mask ranges between 0 and number of classes minus one. If the
    segmentation dataset is Cityscapes, then segmentation masks are converted into Cityscapes format.

    Args:
        pred_mask: Predicted segmentation mask.
        file_name: File name.
        save_dir: Path of the directory where segmentation masks will be saved.
        is_cityscape: Predictions are on the Cityscape dataset or not.
    """
    no_color_mask_dir = "{}/predictions_no_cmap".format(save_dir)
    if not os.path.isdir(no_color_mask_dir):
        os.makedirs(no_color_mask_dir, exist_ok=True)
    no_color_mask_f_name = "{}/{}".format(no_color_mask_dir, file_name)

    if is_cityscape:
        # convert mask values to cityscapes format
        pred_mask = convert_to_cityscape_format(img=pred_mask)
    pred_mask_pil = F_vision.to_pil_image(pred_mask.byte())
    pred_mask_pil.save(no_color_mask_f_name)


def apply_color_map_to_segmentation_masks(
    opts: argparse.Namespace,
    orig_image: Image.Image,
    pred_mask: Tensor,
    target_mask: Tensor,
    file_name: str,
    save_dir: str,
    color_map: Optional[List] = None,
) -> None:
    """Apply color map to segmentation masks.

    Args:
        orig_image: Original RGB image.
        pred_mask: Predicted segmentation tensor.
        target_mask: Ground truth segmentation tensor.
        file_name: File name.
        save_dir: Path of the directory where colored segmentation masks will be saved.
        color_map: Color mapping for converting predicted binary masks into colored representations.
    """

    alpha = getattr(opts, "evaluation.segmentation.overlay_mask_weight")
    save_overlay_rgb_pred = getattr(
        opts, "evaluation.segmentation.save_overlay_rgb_pred"
    )

    if color_map is None:
        color_map = Colormap().get_color_map_list()

    # convert predicted tensor to PIL images, apply color map and save
    pred_mask_pil = F_vision.to_pil_image(pred_mask.byte())
    pred_mask_pil.putpalette(color_map)
    pred_mask_pil = pred_mask_pil.convert("RGB")
    pred_color_mask_dir = "{}/predictions_cmap".format(save_dir)
    if not os.path.isdir(pred_color_mask_dir):
        os.makedirs(pred_color_mask_dir, exist_ok=True)
    color_mask_f_name = "{}/{}".format(pred_color_mask_dir, file_name)
    pred_mask_pil.save(color_mask_f_name)
    logger.log("Predicted mask is saved at: {}".format(color_mask_f_name))

    if target_mask is not None:
        # convert target tensor to PIL images, apply colormap, and save
        target_mask_pil = F_vision.to_pil_image(target_mask.byte())
        target_mask_pil.putpalette(color_map)
        target_mask_pil = target_mask_pil.convert("RGB")
        target_color_mask_dir = "{}/gt_cmap".format(save_dir)
        if not os.path.isdir(target_color_mask_dir):
            os.makedirs(target_color_mask_dir, exist_ok=True)
        gt_color_mask_f_name = "{}/{}".format(target_color_mask_dir, file_name)
        target_mask_pil.save(gt_color_mask_f_name)
        logger.log("Target mask is saved at: {}".format(color_mask_f_name))

    if save_overlay_rgb_pred and orig_image is not None:
        # overlay predicted mask on top of original image and save

        if pred_mask_pil.size != orig_image.size:
            # resize if input image size is not the same as predicted mask.
            # this is likely in case of labeled datasets where we use transforms on the input image
            orig_image = F_vision.resize(
                orig_image,
                size=pred_mask_pil.size[::-1],
                interpolation=F_vision.InterpolationMode.BILINEAR,
            )

        overlay_img = Image.blend(pred_mask_pil, orig_image, alpha=alpha)
        overlay_mask_dir = "{}/predictions_overlay".format(save_dir)
        if not os.path.isdir(overlay_mask_dir):
            os.makedirs(overlay_mask_dir, exist_ok=True)
        overlay_mask_f_name = "{}/{}".format(overlay_mask_dir, file_name)
        overlay_img.save(overlay_mask_f_name)
        logger.log(
            "RGB image blended with mask is saved at: {}".format(overlay_mask_f_name)
        )

        # save original image
        rgb_image_dir = "{}/rgb_images".format(save_dir)
        if not os.path.isdir(rgb_image_dir):
            os.makedirs(rgb_image_dir, exist_ok=True)
        rgb_image_f_name = "{}/{}".format(rgb_image_dir, file_name)
        orig_image.save(rgb_image_f_name)
        logger.log("Original RGB image is saved at: {}".format(overlay_mask_f_name))


def read_and_process_image(
    opts: argparse.Namespace, file_path: str
) -> Tuple[Tensor, Image.Image, int, int]:
    """Read and converts the input image into a tensor.

    Args:
        opts: Command-line arguments.
        file_path: Path of the image file.

    Returns:
        A tuple containing:
            1. Image tensor with shape [1, C, H, W]. The tensor values are normalized between 0 and 1.
            2. Original RGB image.
            3. Height of the original image.
            4. Width of the original image.
    """

    input_img = Image.open(file_path).convert("RGB")
    input_pil = copy.deepcopy(input_img)
    orig_w, orig_h = input_img.size

    # Resize the image while maitaining the aspect ratio
    res_h, res_w = image_size_from_opts(opts)

    input_img = F_vision.resize(
        input_img,
        size=min(res_h, res_w),
        interpolation=F_vision.InterpolationMode.BILINEAR,
    )
    input_tensor = F_vision.pil_to_tensor(input_img)
    input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
    return input_tensor, input_pil, orig_h, orig_w


@torch.no_grad()
def predict_labeled_dataset(opts: argparse.Namespace) -> None:
    """Generate predictions for the labeled dataset.

    This function predicts segmentation labels for RGB input images and calculates segmentation accuracy, measured
    by mean intersection over union, through comparison between predicted segmentation masks and ground truth segmentation masks.

    Args:
        opts: Command-line arguments.
    """
    device = getattr(opts, "dev.device", torch.device("cpu"))
    dataset_name = getattr(opts, "dataset.name", "")

    # set-up data loaders
    test_loader = create_test_loader(opts)

    # set-up the model
    model = get_model(opts)
    model.eval()
    model.info()
    model = model.to(device=device)

    if model.training:
        logger.log("Model is in training mode. Switching to evaluation mode")
        model.eval()

    color_map = Colormap().get_color_map_list()
    adjust_label = 0
    is_cityscape = False
    conf_mat = ConfusionMatrix()
    if hasattr(test_loader.dataset, "color_palette"):
        color_map = test_loader.dataset.color_palette()

    if hasattr(test_loader.dataset, "adjust_mask_value"):
        adjust_label = test_loader.dataset.adjust_mask_value()

    if dataset_name is not None and dataset_name.lower() == "cityscapes":
        is_cityscape = True

    for batch in tqdm(test_loader):
        samples, targets = batch["samples"], batch["targets"]
        batch_size = samples.shape[0]
        assert (
            batch_size == 1
        ), "We recommend to run segmentation evaluation with a batch size of 1"

        predict_and_save(
            opts=opts,
            input_tensor=samples,
            file_name=targets["file_name"][0],
            orig_w=targets["im_width"][0].item(),
            orig_h=targets["im_height"][0].item(),
            model=model,
            target_mask=targets["mask"],
            device=device,
            conf_mat=conf_mat,
            color_map=color_map,
            adjust_label=adjust_label,
            is_cityscape=is_cityscape,
        )

    metrics = conf_mat.compute()
    logger.info("Quantitative results")
    print(
        "global correct: {:.2f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.2f}".format(
            metrics["accuracy_global"] * 100,
            ["{:.2f}".format(i * 100) for i in metrics["class_accuracy"]],
            ["{:.2f}".format(i * 100) for i in metrics["iou"]],
            metrics["mean_iou"] * 100,
        )
    )

    is_city_dataset = getattr(opts, "dataset.name", "") == "cityscapes"
    if is_city_dataset:
        from corenet.engine.segmentation_utils.cityscapes_iou import eval_cityscapes

        pred_dir = "{}/predictions_no_cmap/".format(
            getattr(opts, "common.exp_loc", None)
        )
        gt_dir = os.path.join(getattr(opts, "dataset.root_val", None), "gtFine/val/")
        eval_cityscapes(pred_dir=pred_dir, gt_dir=gt_dir)


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

    input_tensor, input_pil, orig_h, orig_w = read_and_process_image(
        opts, local_file_path
    )

    file_name = os.path.basename(file_path)

    device = getattr(opts, "dev.device")
    # set-up the model
    model = get_model(opts)
    model.eval()
    model.info()
    model = model.to(device=device)

    if model.training:
        logger.log("Model is in training mode. Switching to evaluation mode")
        model.eval()

    predict_and_save(
        opts=opts,
        input_tensor=input_tensor,
        file_name=file_name,
        orig_h=orig_h,
        orig_w=orig_w,
        model=model,
        target_mask=None,
        device=device,
        orig_image=input_pil,
    )


@torch.no_grad()
def predict_images_in_folder(opts: argparse.Namespace) -> None:
    """Generate predictions for all RGB images stored in a folder.

    Args:
        opts: Command-line arguments.

    ...note:
        The path of the folder containong RGB images is supplied using 'evaluation.segmentation.path' argument.
    """
    img_folder_path = getattr(opts, "evaluation.segmentation.path")
    if img_folder_path is None:
        logger.error(
            "Location of the folder containing images is not passed. Please use --evaluation.segmentation.path "
            "as an argument to pass the location of the folder".format(img_folder_path)
        )
    elif not os.path.isdir(img_folder_path):
        logger.error(
            "Folder containing images does not exist at: {}. Please check".format(
                img_folder_path
            )
        )

    img_files = []
    for e in SUPPORTED_IMAGE_EXTNS:
        img_files_with_extn = glob.glob("{}/*{}".format(img_folder_path, e))
        if len(img_files_with_extn) > 0 and isinstance(img_files_with_extn, list):
            img_files.extend(img_files_with_extn)

    if len(img_files) == 0:
        logger.error(
            "Number of image files found at {}: {}".format(
                img_folder_path, len(img_files)
            )
        )

    logger.log(
        "Number of image files found at {}: {}".format(img_folder_path, len(img_files))
    )

    device = getattr(opts, "dev.device", torch.device("cpu"))
    # set-up the model
    model = get_model(opts)
    model.eval()
    model.info()
    model = model.to(device=device)

    if model.training:
        logger.log("Model is in training mode. Switching to evaluation mode")
        model.eval()

    for image_fname in tqdm(img_files):
        input_tensor, input_pil, orig_h, orig_w = read_and_process_image(
            opts, file_path=image_fname
        )

        image_fname = image_fname.split(os.sep)[-1]

        predict_and_save(
            opts=opts,
            input_tensor=input_tensor,
            file_name=image_fname,
            orig_h=orig_h,
            orig_w=orig_w,
            model=model,
            target_mask=None,
            device=device,
            orig_image=input_pil,
        )


def main_segmentation_evaluation(args: Optional[List[str]] = None) -> None:
    """Entrypoint for segmentation evaluation."""
    opts = get_training_arguments(args=args)

    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank")
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc")
    run_label = getattr(opts, "common.run_label")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    logger.log("Results (if any) will be stored here: {}".format(exp_dir))

    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    # we disable the DDP setting for evaluating segmentation tasks
    setattr(opts, "ddp.use_distributed", False)

    dataset_workers = getattr(opts, "dataset.workers")

    if dataset_workers == -1:
        n_cpus = resources.cpu_count()
        # Set number of data workers to the same as number of CPUs if 'dataset.workers' is -1
        setattr(opts, "dataset.workers", n_cpus)

    # For segmentation during inference, images are not resized to a fixed size.
    # Therefore, we set batch size to 1.
    logger.log("Setting batch size to 1 for segmentation evaluation.")
    setattr(opts, "dataset.train_batch_size0", 1)
    setattr(opts, "dataset.val_batch_size0", 1)
    setattr(opts, "dev.device_id", None)

    eval_mode = getattr(opts, "evaluation.segmentation.mode")

    if eval_mode == "single_image":
        # Predict segmentation mask for a single RGB image
        num_seg_classes = getattr(opts, "model.segmentation.n_classes")
        assert num_seg_classes is not None
        img_f_name = getattr(opts, "evaluation.segmentation.path")
        predict_image(opts, img_f_name)
    elif eval_mode == "image_folder":
        # Predict segmentation masks for all RGB images in a folder
        # This is useful for generating results for test set
        num_seg_classes = getattr(opts, "model.segmentation.n_classes")
        assert num_seg_classes is not None
        predict_images_in_folder(opts=opts)
    elif eval_mode == "validation_set":
        # evaluate and compute stats for labeled image dataset
        # This is useful for generating results for validation set and compute quantitative results
        predict_labeled_dataset(opts=opts)
    else:
        logger.error(
            "Supported modes are 'single_image', 'image_folder', and 'validation_set'. Got: {}".format(
                eval_mode
            )
        )
