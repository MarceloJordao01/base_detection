from setuptools import setup, find_packages
from glob import glob
import os

package_name = "base_detection"

data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/config", ["config/base_detection_params.yaml"]),
    (f"share/{package_name}/launch", glob("launch/*.launch.py")),  # <-- instala os launch
]

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test", "tests"]),
    data_files=data_files,
    include_package_data=True,
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",
        "ultralytics",
        "torch",
    ],
    zip_safe=True,
    author="Você",
    author_email="voce@example.com",
    maintainer="Você",
    maintainer_email="voce@example.com",
    description="ROS2 node for HSV + YOLO base detection.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "base_detection_node = base_detection.base_detection_node:main",
        ],
    },
)
