from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Launch Gazebo + turtlebot3 empty world
    gz = ExecuteProcess(
        cmd=['ros2', 'launch', 'turtlebot3_gazebo', 'turtlebot3_world.launch.py'],
        shell=False
    )

    gesture_node = Node(
        package='tb3_gesture_control',
        executable='gesture_node',
        name='gesture_node',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        gz,
        gesture_node
    ])
