# setup.py
import platform
import sys

from setuptools import find_packages, setup


def verify_os():
    current_os = platform.system()
    if current_os != "Linux":
        print(f"Error: This package requires Linux OS. Current OS: {current_os}")
        sys.exit(1)


def get_default_dependencies():
    return [
        "torch>=2.6.0, <2.8.0",
        "transformers>=4.48.0, <5.0.0",
        "datasets>=4.0.0, <5.0.0",
        "dataclasses_json",
    ]


def get_optional_dependencies():
    """Get optional dependency groups."""
    extras = {
        "inference": [
            "llmcompressor==0.5.1",
        ],
        "train": [
            "lightning>=2.3.2, <3.0.0",
            "liger-kernel>=0.5.0, <1.0.0",
        ],
        "dev": [
            "matplotlib>=3.7.2",
            "flake8>=4.0.1.1",
            "black>=24.4.2",
            "isort>=5.13.2",
            "pytest>=7.1.2",
            "pre-commit",
            "pytest-xdist",
            "pytest-rerunfailures",
            "seaborn",
            "mkdocs",
            "mkdocs-material",
        ],
    }

    extras["all"] = extras["inference"] + extras["train"]
    extras["dev"] = extras["dev"] + extras["all"]

    return extras


verify_os()

setup(
    name="FMCHISEL",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["fmchisel*"]),
    install_requires=get_default_dependencies(),
    extras_require=get_optional_dependencies(),
)
