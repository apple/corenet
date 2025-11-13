#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from corenet.constants import is_test_env


class UnrecognizedYamlConfigEntry(Exception):
    # Throws exception when a key in yaml config is not recognized by argparser.
    def __init__(self, key: str) -> None:
        message = (
            f"Yaml config key '{key}' was not recognized by argparser. If you think that you have already added "
            f"argument in corenet/options/opts.py file, then check for typos. If not, then please add it to corenet/options/opts.py."
        )
        super().__init__(message)

        if is_test_env():
            # Currently, we only raise an exception in test environment.
            raise ValueError(message)
