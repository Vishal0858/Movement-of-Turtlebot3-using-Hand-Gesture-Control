from setuptools import setup
import os
from glob import glob

package_name = 'ros2_gesture_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')), # Example if you add a launch file
    ],
    install_requires=[
        'setuptools',
        'opencv-python',  # Add dependency for OpenCV
        'mediapipe'       # Add dependency for MediaPipe
    ],
    zip_safe=True,
    maintainer='vishal',
    maintainer_email='m24irm008@iitj.ac.in',
    description='ROS 2 package for hand gesture detection using MediaPipe.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_detector = ros2_gesture_control.gesture_detector_node:main',
        ],
    },
)
