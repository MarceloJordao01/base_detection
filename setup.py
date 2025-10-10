from setuptools import setup, find_packages
from glob import glob
import os
import sys

package_name = "base_detection"

def eprint(msg: str):
    sys.stderr.write(msg.rstrip() + "\n")

# --- Coleta de modelos em 'models/' ou 'model/' (ambos aceitos) ---
src_models_dirs = []
if os.path.isdir("models"):
    src_models_dirs.append("models")
if os.path.isdir("model"):   # compat: caso tenha criado no singular
    src_models_dirs.append("model")

models_src = []
models_target_dir = os.path.join("share", package_name, "models")

found_files = []
for d in src_models_dirs:
    files = [f for f in glob(os.path.join(d, "*")) if os.path.isfile(f)]
    if files:
        models_src.append((models_target_dir, files))
        found_files.extend(files)

# --- Coleta de arquivos de launch em 'launch/' ---
launch_src = []
launch_dir = "launch"
launch_target_dir = os.path.join("share", package_name, "launch")

# aceita .launch.py (padrão ROS2), .py e .xml (compatibilidade)
launch_files = []
if os.path.isdir(launch_dir):
    patterns = ["*.launch.py", "*.py", "*.xml"]
    for pat in patterns:
        launch_files.extend(glob(os.path.join(launch_dir, pat)))
    # garante que só arquivos reais vão
    launch_files = [f for f in launch_files if os.path.isfile(f)]
    if launch_files:
        launch_src.append((launch_target_dir, launch_files))

# --- data_files padrão + modelos + launch ---
data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/config", ["config/base_detection_params.yaml"]),
]
data_files.extend(models_src)
data_files.extend(launch_src)

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
    description="ROS2 node for HSV + YOLO base detection (instala modelos via data_files, logs em stderr).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "base_detection_node = base_detection.base_detection_node:main",
            # utilitário de verificação pós-instalação:
            # "base_detection_check_install = base_detection.check_install:main",
        ],
    },
)
