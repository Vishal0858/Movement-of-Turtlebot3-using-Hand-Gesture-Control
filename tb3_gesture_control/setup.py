import os
from setuptools import setup, find_packages

package_name = 'tb3_gesture_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vishal Kumar',
    maintainer_email='vk041098@gmail.com',
    description='Gesture control for TurtleBot3 using MediaPipe and ROS2 Humble',
    license='MIT',
    entry_points={
        'console_scripts': [
            'gesture_node = tb3_gesture_control.gesture_node:main',
        ],
    },
)
