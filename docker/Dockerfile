FROM osrf/ros:humble-desktop-full

# Usar bash nos RUNs (necessário para "source")
SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WS=/base_detection

# Dependências de build, ROS e visão computacional
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-cv-bridge \
    ros-humble-camera-calibration-parsers \
    libasio-dev \
    mesa-utils \
    libgl1-mesa-dri libglx-mesa0 libegl1-mesa \
    mesa-vulkan-drivers vulkan-tools \
 && rm -rf /var/lib/apt/lists/*

# Pacotes Python para detecção/visão
# - OpenCV, Ultralytics/YOLO, NumPy < 2 p/ compatibilidade com várias libs
RUN python3 -m pip install --no-cache-dir \
    transformations \
    ultralytics \
    'numpy<2,>=1.23'==1.26.4

# Criar workspace e copiar TODO o repositório para dentro de /base_detection
# Estrutura vira um workspace colcon: /base_detection/src/base_detection/<seus-arquivos>
RUN mkdir -p ${WS}/src/base_detection
WORKDIR ${WS}
COPY . ${WS}/src/base_detection/

# Resolver dependências via rosdep e compilar se houver pacotes ROS2 no repo
# (rosdep ignora o que não é pacote ROS; então é seguro mesmo com scripts puros Python)
RUN set -e; \
    source /opt/ros/humble/setup.bash; \
    rosdep init 2>/dev/null || true; \
    rosdep update; \
    apt-get update; \
    rosdep install --from-paths src --ignore-src -r -y || true; \
    rm -rf /var/lib/apt/lists/*; \
    colcon build --symlink-install || true

# Facilitar ambiente interativo
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source ${WS}/install/setup.bash" >> /root/.bashrc || true

# Diretório de trabalho padrão no container
WORKDIR ${WS}

# Abrir um bash interativo
CMD ["bash"]
