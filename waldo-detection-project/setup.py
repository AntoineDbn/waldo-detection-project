"""
Waldo Detection Project Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="waldo-detection-project",
    version="1.0.0",
    author="Antoine DEBIN",
    author_email="antoine.debin@gmail.com",
    description="AI-powered Where's Waldo detector using YOLOv8 and CLIP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntoineDbn/waldo-detection-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "waldo-train=training.train_yolo:main",
            "waldo-detect=inference.detect_with_clip:main",
            "waldo-extract=preprocessing.extract_waldo:main",
            "waldo-generate=data_generation.create_synthetic_data:main",
        ],
    },
)
