#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Dict, List

from corenet.data.transforms import TRANSFORMATIONS_REGISTRY, BaseTransformation


@TRANSFORMATIONS_REGISTRY.register(name="compose", type="common")
class Compose(BaseTransformation):
    """
    This method applies a list of transforms in a sequential fashion.
    """

    def __init__(self, opts, img_transforms: List, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        self.img_transforms = img_transforms

    def __call__(self, data: Dict) -> Dict:
        for t in self.img_transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        transform_str = ", ".join("\n\t\t\t" + str(t) for t in self.img_transforms)
        repr_str = "{}({}\n\t\t)".format(self.__class__.__name__, transform_str)
        return repr_str


@TRANSFORMATIONS_REGISTRY.register(name="identity", type="common")
class Identity(BaseTransformation):
    """
    This is a no-op transformation that returns its inputs unchanged.
    """

    def __call__(self, data: Dict) -> Dict:
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
