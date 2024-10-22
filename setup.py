import os
from glob import glob
from setuptools import setup

package_name = 'traj_gen'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'rviz'), glob('rviz/*rviz')),
        (os.path.join('share', package_name,'plotjuggler_layouts'), glob('plotjuggler_layouts/*xml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tareq Alqutami',
    maintainer_email='tareqaziz2010@gmail.com',
    description='trajectory generator in ros2 for uav applications',
    license='MIT',   
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'min_snap_traj_generator = traj_gen.min_snap_traj_generator_node:main',
            'ana_traj_generator = traj_gen.ana_traj_generator_node:main',
        ],
    },
)
