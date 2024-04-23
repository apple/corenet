#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from pathlib import Path

import pytest

from tests.configs import get_config


def test_load_config_file_produces_no_false_warnings() -> None:
    get_config()


def test_load_config_file_produces_true_warning(
    tmp_path: Path,
) -> None:
    config_path = tmp_path.joinpath("config.yaml")
    config_path.write_text("an_invalid_key: 2")
    with pytest.raises(ValueError, match="an_invalid_key"):
        get_config(str(config_path))
