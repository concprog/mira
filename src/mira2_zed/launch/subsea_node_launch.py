from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="mira2_zed",
                executable="subsea_node",
                name="subsea_node",
                output="screen",
                parameters=[
                    # Add parameters here if needed
                ],
            ),
        ]
    )
