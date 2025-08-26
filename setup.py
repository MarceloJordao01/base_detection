from setuptools import setup, find_packages
from glob import glob
import os
import sys

# IMPORTANTÍSSIMO: NUNCA imprima em stdout no setup.py quando o colcon faz --dry-run.
# Use sempre sys.stderr.write(...) para logs, senão o colcon quebra ao fazer ast.literal_eval.
# Referências: impressão em setup.py pode interferir no colcon/--dry-run.  # noqa

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

# Logs (sempre stderr!):
if src_models_dirs:
    eprint(f"[setup] procurando modelos em: {', '.join(src_models_dirs)}")
else:
    eprint("[setup] diretórios 'models/' ou 'model/' não encontrados.")

if found_files:
    eprint(f"[setup] {len(found_files)} arquivo(s) de modelo detectado(s):")
    for f in found_files:
        eprint(f"        - {f}")
else:
    eprint("[setup] AVISO: nenhum arquivo de modelo foi encontrado; nada será instalado em share/.../models/")

# --- data_files padrão + modelos ---
data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/config", ["config/base_detection_params.yaml"]),
]
data_files.extend(models_src)

# Log do que será instalado (somente stderr)
eprint("[setup] data_files ->")
for target, files in data_files:
    eprint(f"    - {target}")
    for f in files:
        eprint(f"        * {f}")

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
            "base_detection_check_install = base_detection.check_install:main",
        ],
    },
)