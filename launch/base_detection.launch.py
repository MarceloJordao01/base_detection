from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Caminho para o arquivo YAML de parâmetros
    params_file = os.path.join(
        get_package_share_directory('base_detection'),
        'config',
        'base_detection_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='base_detection',
            executable='base_detection_node',   # nome do entry point instalado
            name='base_detection',              # precisa bater com o YAML
            output='screen',
            parameters=[params_file]
        )
    ])
