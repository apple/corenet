#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List

from setuptools import find_packages, setup


def is_apple_silicon_macos() -> bool:
    return platform.machine() == "arm64" and platform.system() == "Darwin"


def parse_requirements(path: str) -> List[str]:
    """Parse a requirements file."""
    requirements = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            elif line:
                requirements.append(line)
    return requirements


def main() -> None:
    if sys.version_info < (3, 6):
        sys.exit("Sorry, Python >= 3.6 is required for CoreNet.")

    if sys.platform == "darwin":
        extra_compile_args = ["-stdlib=libc++", "-std=c++17", "-O3"]
    else:
        extra_compile_args = ["-std=c++17", "-O3"]

    (version,) = re.findall(
        r"\d+\.\d+\.\d+", Path("corenet/__version__.py").read_text()
    )

    requirements = parse_requirements("requirements.txt")
    optional_requirements = parse_requirements("requirements-optional.txt")

    is_internal = os.path.exists("internal")
    if is_internal:
        requirements += parse_requirements("internal/requirements.txt")
        optional_requirements += parse_requirements(
            "internal/requirements-optional.txt"
        )

    # When installed as a library in other projects, we don't need dev requirements.
    dev_requirement_regex = r"^(black|isort|pytest)"
    dev_requirements = []
    for req in requirements[:]:
        if re.match(dev_requirement_regex, req):
            dev_requirements.append(req)
            requirements.remove(req)

    # Dependencies w.r.t MLX
    if is_apple_silicon_macos():
        # MLX is only available on Apple Silicon macOS.
        # https://ml-explore.github.io/mlx/build/html/install.html#troubleshooting
        mlx_requirements = [
            "mlx>=0.9.0",
            "huggingface_hub",
        ]
    else:
        mlx_requirements = []  # Do not install anything

    sentence_piece_requirements = "sentencepiece>=0.2.0"

    sys.path.insert(0, "corenet/cli")
    from entrypoints import console_scripts

    if is_internal:
        sys.path.insert(0, "corenet/internal/cli")
        del sys.modules["entrypoints"]
        import entrypoints as internal_entrypoints

        console_scripts += internal_entrypoints.console_scripts

    setup(
        name="corenet",
        version=version,
        description="CoreNet: A library for training computer vision networks",
        url="https://github.com/apple/corenet.git",
        python_requires=">=3.9",
        setup_requires=[
            "setuptools>=18.0",
        ],
        install_requires=requirements,
        extras_require={
            "dev": dev_requirements,
            "optional": optional_requirements,
            "mlx": mlx_requirements,
            "sentencepiece": sentence_piece_requirements,
            "nltk": "nltk>=3.8.1",
        },
        packages=find_packages(include=["corenet*"]),
        data_files=[
            ("corenet-requirements", ["requirements.txt"]),
            ("corenet-requirements", ["requirements-optional.txt"]),
            ("corenet-config", get_files("config")),
            ("corenet-projects", get_files("projects")),
        ]
        + ([("corenet-internal", get_files("internal"))] if is_internal else []),
        test_suite="tests",
        entry_points={
            "console_scripts": console_scripts,
        },
        include_package_data=True,
    )


def get_files(path, relative_to=".") -> List[str]:
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    main()
