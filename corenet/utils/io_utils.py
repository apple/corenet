#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import tempfile
from typing import Optional


def make_temp_file(
    suffix: str = None,
    prefix: Optional[str] = "corenet-tmp-",
    dir: Optional[str] = None,
) -> str:
    """Create a temporary file and return its path."""
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=prefix,
        dir=dir,
    )
    tmp_file.close()
    return tmp_file.name
