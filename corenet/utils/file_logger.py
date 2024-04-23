#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Union

import torch


class FileLogger:
    """
    A logger that writes scalar values to a file.
    """

    def __init__(self, fname: str) -> None:
        """
        Initialize a FileLogger.

        Since logged values may include tensors, the file is saved with
        torch.save.

        Args:
            fname: The name of the pytorch file to which the logger will save
                logs.
        """
        self.fname = fname

        if os.path.exists(self.fname):
            self.stats = torch.load(self.fname)
        else:
            # We hold stats in a dictionary keyed by epoch number. We use a
            # dictionary rather than a list to avoid the assumption that
            # len(self.stats) gives the current epoch. Such an assumption is
            # inconvenient if, for instance, we resume training mid-way through,
            # and suddenly enabling this FileLogger when it wasn't enabled
            # before.
            self.stats: Dict[int, Dict[str, Any]] = {"epochs": {}}

    def add_scalar(
        self,
        metric_name: str,
        metric_value: Union[float, int],
        epoch: int,
    ) -> None:
        """
        Add a scalar to the FileLogger.

        Args:
            metric_name: The name of the metric.
            metric_value: The value of the metric.
            epoch: The epoch number.
        """
        if epoch not in self.stats["epochs"]:
            self.stats["epochs"][epoch] = {"metrics": {}}

        self.stats["epochs"][epoch]["metrics"][metric_name] = metric_value

    def close(self) -> None:
        # Write to a temporary file, then use shutil.move (to make the write
        # atomic). This avoids creating a malformed file if the job crashes
        # during writing.
        temporary_file = tempfile.NamedTemporaryFile(delete=False)
        torch.save(self.stats, temporary_file.name)
        temporary_file.close()

        # NOTE: Do not use os.rename to avoid 'OSError: Invalid cross-device link'.
        # See: https://stackoverflow.com/questions/42392600/oserror-errno-18-invalid-cross-device-link
        shutil.move(temporary_file.name, self.fname)
