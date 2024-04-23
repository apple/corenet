#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import re
import sys
from pathlib import Path

sys.path.append("..")

from tests.configs import get_config
from tests.modeling.test_model import exclude_yaml_from_test


def test_neural_aug_backward_compatibility(config_file: str):
    opts = get_config(config_file=config_file)

    opts_dict = vars(opts)
    for k, v in opts_dict.items():
        if isinstance(v, str) and re.search(".*_with_na$", v):
            raise DeprecationWarning(
                "We deprecated the usage of _with_na loss functions. "
                "Please see projects/range_augment for examples."
            )


def pytest_generate_tests(metafunc):
    configs = [
        str(x)
        for x in Path("config").rglob("**/*.yaml")
        if not exclude_yaml_from_test(x)
    ]
    configs += [
        str(x)
        for x in Path("projects").rglob("**/*.yaml")
        if not exclude_yaml_from_test(x)
    ]
    metafunc.parametrize("config_file", configs)
