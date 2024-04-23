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

# State-of-the-art models use 171 classes for COCO-Stuff. This is because some of the labels defined in
# COCO-Stuff are not annotated. So, 182 cocostuff labels are remapped to 171 labels.
_cocostuff_remap_info = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    26: 24,
    27: 25,
    30: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    45: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    66: 60,
    69: 61,
    71: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    83: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    91: 80,
    92: 81,
    93: 82,
    94: 83,
    95: 84,
    96: 85,
    97: 86,
    98: 87,
    99: 88,
    100: 89,
    101: 90,
    102: 91,
    103: 92,
    104: 93,
    105: 94,
    106: 95,
    107: 96,
    108: 97,
    109: 98,
    110: 99,
    111: 100,
    112: 101,
    113: 102,
    114: 103,
    115: 104,
    116: 105,
    117: 106,
    118: 107,
    119: 108,
    120: 109,
    121: 110,
    122: 111,
    123: 112,
    124: 113,
    125: 114,
    126: 115,
    127: 116,
    128: 117,
    129: 118,
    130: 119,
    131: 120,
    132: 121,
    133: 122,
    134: 123,
    135: 124,
    136: 125,
    137: 126,
    138: 127,
    139: 128,
    140: 129,
    141: 130,
    142: 131,
    143: 132,
    144: 133,
    145: 134,
    146: 135,
    147: 136,
    148: 137,
    149: 138,
    150: 139,
    151: 140,
    152: 141,
    153: 142,
    154: 143,
    155: 144,
    156: 145,
    157: 146,
    158: 147,
    159: 148,
    160: 149,
    161: 150,
    162: 151,
    163: 152,
    164: 153,
    165: 154,
    166: 155,
    167: 156,
    168: 157,
    169: 158,
    170: 159,
    171: 160,
    172: 161,
    173: 162,
    174: 163,
    175: 164,
    176: 165,
    177: 166,
    178: 167,
    179: 168,
    180: 169,
    181: 170,
    # 255 is not a label and is ignored during training.
    255: 255,
}


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
