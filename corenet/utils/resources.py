#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

try:
    from corenet.internal.utils.resources import cpu_count
except ImportError:
    from multiprocessing import cpu_count

__all__ = ["cpu_count"]
