#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import subprocess
import sys
from pathlib import Path

from pytest_mock import MockerFixture

from corenet.utils import import_utils
from corenet.utils.import_utils import import_modules_from_folder


def test_import_utils(tmp_path: Path, mocker: MockerFixture) -> None:
    tmp_path_str = str(tmp_path)
    sys.path.append(tmp_path_str)
    mocker.patch.object(import_utils, "LIBRARY_ROOT", tmp_path)
    try:
        files = [
            "my_test_parent/child/module.py",
            "my_test_parent/child/nested/module.py",
            "my_test_parent/sibling.py",
            "my_internal/my_test_parent/child/module.py",
            "my_internal/my_test_parent/sibling.py",
            "my_internal/projects/A/my_test_parent/child/module.py",
            "my_internal/projects/B/my_test_parent/child/module.py",
        ]
        for path in files:
            path = tmp_path / path
            for package in path.parents:
                if package == tmp_path:
                    break
                package.mkdir(exist_ok=True, parents=True)
                if not package.joinpath("__init__.py").exists():
                    package.joinpath("__init__.py").write_bytes(b"")
            path.write_bytes(b"")

        import_modules_from_folder(
            "my_test_parent/child",
            extra_roots=["my_internal", "my_internal/projects/*"],
        )
        assert "my_test_parent.child.module" in sys.modules
        assert "my_test_parent.child.nested.module" in sys.modules
        assert "my_test_parent.sibling" not in sys.modules
        assert "my_internal.my_test_parent.child.module" in sys.modules
        assert "my_internal.my_test_parent.sibling" not in sys.modules
        assert "my_internal.projects.A.my_test_parent.child.module" in sys.modules
        assert "my_internal.projects.B.my_test_parent.child.module" in sys.modules
    finally:
        sys.path.remove(tmp_path_str)


def test_ensure_pythonpath_is_not_required():
    """
    In addition to the "corenet" folder, "tests" and "experimental" folders also contain
    python modules that are occasionally imported by certain entrypoints. In contrast
    with ``corenet.*``, that can be imported without ``PYTHONPATH=.`` environment
    variable, ``tests.*`` and ``experimental.*`` require setting ``PYTHONPATH=.``
    because ``pip install [-e] .`` only install ``corenet/**/*.py``.

    For project-specific code, users can set `PYTHONPATH=.` and import stuff from
    `tests.*` or `experimental.*`. However, we should ensure that the core functionality
    of CoreNet does not import from `tests.*` and `experimental.*`. Otherwise, CoreNet
    will fail in environments that `PYTHONPATH=.` is not set.

    In this test, invoke `import_core_modules()` in an isolated python process
    and assert that core functionality of CoreNet doesn't import anything from `tests.*`
    or `experimental.*`.
    """

    subprocess.check_call(
        [
            "python",
            "-c",
            """\
import sys                
from corenet.utils.import_utils import import_core_modules

import_core_modules()

for module_name in sys.modules:
    assert not module_name.startswith("tests"), "CoreNet modules shall not import from tests."
    assert not module_name.startswith("experimental"), "CoreNet modules shall not import from experimental."
""",
        ],
        # It is fine to import stuff from tests/ only in test environment.
        # We simulate running corenet outside of the test environment, so that we can
        # allow modules to import stuff from tests/ only during tests.
        env={k: v for k, v in os.environ.items() if k != "CORENET_ENTRYPOINT"},
    )
