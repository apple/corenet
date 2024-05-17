#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import copy
import glob
import os
from glob import glob
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from corenet.utils import logger

# Generates remaps for COCO-Stuff labels to 171 classes.
# Skips some of the labels that are not annotated.
def generate_cocostuff_remap():
    cocostuff_skips = {11, 25, 28, 29, 44, 65, 67, 70, 82, 90, 179, 181}  # IDs that are skipped
    remap_info = {}
    new_id = 0
    for old_id in range(182):
        if old_id in cocostuff_skips:
            continue
        remap_info[old_id] = new_id
        new_id += 1
    remap_info[255] = 255  # Special case for the label 255
    return remap_info

# State-of-the-art models use 171 classes for COCO-Stuff. This is because some of the labels defined in
# COCO-Stuff are not annotated. So, 182 cocostuff labels are remapped to 171 labels.
_cocostuff_remap_info = generate_cocostuff_remap()

def remove_unannotated_mask_labels(src_path: str, dst_path: str) -> None:
    """Remap cocostuff labels.

    Args:
        src_path: File path of the mask image.
        dst_path: File path of the remapped mask image.
    """
    mask_np = np.array(Image.open(src_path))
    mask_copy = copy.deepcopy(mask_np)
    for old_id, new_id in _cocostuff_remap_info.items():
        mask_copy[mask_np == old_id] = new_id
    mask_pil = Image.fromarray(mask_copy)
    mask_pil.save(dst_path, "png")


def main(opts: argparse.Namespace) -> None:
    """
    Main function to remap COCOStuff labels.

    Args:
        opts: Command-line arguments.
    """

    src_dir = getattr(opts, "src_dir")

    src_sub_directories = [x[0] for x in os.walk(src_dir)]

    if not set(
        [os.path.join(f"{src_dir}", "train2017"), os.path.join(f"{src_dir}", "val2017")]
    ).issubset(src_sub_directories):
        logger.error(
            f"Src directory must contain {src_dir}/train2017 and {src_dir}/val2017 subdirectories with annotations. Got: {src_sub_directories}"
        )

    dst_dir = src_dir + "_remap"
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    n_jobs = getattr(opts, "num_jobs")

    for sub_dir in ["train2017", "val2017"]:
        Path(os.path.join(dst_dir, sub_dir)).mkdir(parents=True, exist_ok=True)
        src_png_files = glob.glob(f"{os.path.join(src_dir, sub_dir)}/*.png")
        dst_ping_files = [
            os.path.join(dst_dir, sub_dir, os.path.basename(src_png_file))
            for src_png_file in src_png_files
        ]

        Parallel(n_jobs=n_jobs)(
            delayed(remove_unannotated_mask_labels)(src_mask_path, dst_mask_path)
            for src_mask_path, dst_mask_path in tqdm(zip(src_png_files, dst_ping_files))
        )
        logger.info(f"Done processing {sub_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Remap COCO-Stuff labels")
    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Source directory that contains the train2017 and val2017 annotation folders.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=-1,
        help="Number of jobs for processing data in parallel with joblib. Defaults to -1 (i.e., use all CPUs).",
    )

    opts = parser.parse_args()
    main(opts)
