#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import fcntl
import pdb
import sys


class _ForkedPdb(pdb.Pdb):
    """A Pdb subclass for forked subprocesses. The first subprocess (or main process)
    that invokes this class launches the pdb, and the rest of the processes will wait.
    Uses a simple file-based lock mechanism for synchronization that works on Unix, but
    has not been tested on other operation systems.

    Usage:

    ```
    from corenet.utils import fpdb; fpdb.set_trace()
    ```

    Inspired by https://github.com/Lightning-AI/forked-pdb/blob/master/fpdb.py
    """

    def __init__(self, lockfile: str) -> None:
        self.lockfile = lockfile
        super().__init__()

    def interaction(self, *args, **kwargs) -> None:
        """Acquires the lockfile, then attaches the input stream of the subprocess to
        /dev/stdin, so that subprocess debugger can receive keyboard inputs.
        """
        with open(self.lockfile, "a") as lock:
            try:
                fcntl.lockf(lock, fcntl.LOCK_EX)

                _stdin = sys.stdin
                try:
                    sys.stdin = open("/dev/stdin")
                    pdb.Pdb.interaction(self, *args, **kwargs)
                finally:
                    sys.stdin = _stdin

            finally:
                fcntl.lockf(lock, fcntl.LOCK_UN)


def set_trace(lockfile: str = "/tmp/_corenet_fpdb.lockfile") -> None:
    """
    Launches a pdb debugger in a single-node multi-process job.  The first subprocess
    that invokes this function launches the pdb and makes the rest of subprocesses,
    that invoke this function, wait for the first subprocess's debugger.

    Args:
        lockfile: The path to the lockfile for synchronization between subprocesses.
        Defaults to a constant path in ``/tmp``. As long as there is a single concurrent
        training job running, the user won't need to override this argument.
    """
    _ForkedPdb(lockfile).set_trace(frame=sys._getframe().f_back)
