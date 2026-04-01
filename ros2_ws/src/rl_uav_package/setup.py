import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rl_uav_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Install ALL launch assets: .launch.py, .sdf, .png
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*'))),
            
        # Camera config YAML
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),

        # Register the Tello model directory so it is visible to Gazebo
        (os.path.join('share', package_name, 'models', 'tello'),
            glob(os.path.join('models', 'tello', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='RL-based autonomous UAV landing for DJI Tello',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_ppo = rl_uav_package.train_ppo:main'
        ],
    },
)