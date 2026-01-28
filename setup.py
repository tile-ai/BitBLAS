# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
from setuptools import setup, find_packages
from typing import List
import re
import os
import sys
import platform

# Environment variables False/True
PYPI_BUILD = os.environ.get("PYPI_BUILD", "False").lower() == "true"
PACKAGE_NAME = "bitblas"
ROOT_DIR = os.path.dirname(__file__)

# BitBLAS only supports Linux platform
assert sys.platform.startswith("linux"), "BitBLAS only supports Linux platform (including WSL)."


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


def find_version(version_file_path: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    # Read and store the version information from the VERSION file
    # Use 'strip()' to remove any leading/trailing whitespace or newline characters
    if not os.path.exists(version_file_path):
        raise FileNotFoundError(f"Version file not found at {version_file_path}")
    with open(version_file_path, "r") as version_file:
        version = version_file.read().strip()
    return version


def get_system_info():
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
            version_id_match = re.search(r'VERSION_ID="(\d+\.\d+)"', os_release)
            if version_id_match:
                version_id = version_id_match.group(1)
                distro = "ubuntu"
                return f"{distro}-{version_id}"
        except FileNotFoundError:
            pass
    return system


def get_bitblas_version(with_cuda=True, with_system_info=True) -> str:
    version = find_version(get_path(".", "VERSION"))
    local_version_parts = []
    if with_system_info:
        local_version_parts.append(get_system_info().replace("-", "."))
    if local_version_parts:
        version += f"+{'.'.join(local_version_parts)}"
    return version


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


package_data = {
    "bitblas": ["py.typed"],
}

setup(
    name=PACKAGE_NAME,
    version=get_bitblas_version(with_cuda=False, with_system_info=False)
    if PYPI_BUILD else get_bitblas_version(),
    packages=find_packages(where="."),
    package_dir={"": "."},
    author="Microsoft Research",
    description="A light weight framework to generate high performance CUDA/HIP code for BLAS operators.",
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    platforms=["Environment :: GPU :: NVIDIA CUDA", "Operating System :: POSIX :: Linux"],
    license="MIT",
    keywords="BLAS, CUDA, HIP, Code Generation, TVM",
    url="https://github.com/microsoft/BitBLAS",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    tests_require=[
        "yapf>=0.32.0",
        "toml>=0.10.2",
        "tomli>=2.0.1",
        "ruff>=0.1.5",
        "codespell>=2.2.6",
    ],
    package_data=package_data,
    include_package_data=True,
    data_files=[
        "requirements.txt",
    ],
)
