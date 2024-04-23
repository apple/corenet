#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional

import numpy as np


class Colormap(object):
    """
    Generate colormap for visualizing segmentation masks or bounding boxes.

    This is based on the MATLab code in the PASCAL VOC repository:
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def __init__(self, n: Optional[int] = 256, normalized: Optional[bool] = False):
        super(Colormap, self).__init__()
        self.n = n
        self.normalized = normalized

    @staticmethod
    def get_bit_at_idx(val, idx):
        return (val & (1 << idx)) != 0

    def get_color_map(self) -> np.ndarray:

        dtype = "float32" if self.normalized else "uint8"
        color_map = np.zeros((self.n, 3), dtype=dtype)
        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.get_bit_at_idx(c, 0) << 7 - j)
                g = g | (self.get_bit_at_idx(c, 1) << 7 - j)
                b = b | (self.get_bit_at_idx(c, 2) << 7 - j)
                c = c >> 3

            color_map[i] = np.array([r, g, b])
        color_map = color_map / 255 if self.normalized else color_map
        return color_map

    def get_box_color_codes(self) -> List:
        box_codes = []

        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.get_bit_at_idx(c, 0) << 7 - j)
                g = g | (self.get_bit_at_idx(c, 1) << 7 - j)
                b = b | (self.get_bit_at_idx(c, 2) << 7 - j)
                c = c >> 3
            box_codes.append((int(r), int(g), int(b)))
        return box_codes

    def get_color_map_list(self) -> List:
        cmap = self.get_color_map()
        cmap = np.asarray(cmap).flatten()
        return list(cmap)
