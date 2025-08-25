from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('base_detection')
    default_yaml = os.path.join(pkg_share, 'config', 'base_detection.yaml')

    param_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_yaml,
        description='YAML de parâmetros para o nó base_detection'
    )

    node = Node(
        package='base_detection',
        executable='base_detection',
        name='base_detection',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    return LaunchDescription([param_file_arg, node])
