#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import tempfile


def make_temp_file(suffix: str = None) -> str:
    """Create a temporary file and return its path."""
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )
    tmp_file.close()
    return tmp_file.name
