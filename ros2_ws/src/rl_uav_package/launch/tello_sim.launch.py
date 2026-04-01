import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('rl_uav_package')
    launch_dir = os.path.join(pkg_dir, 'launch')
    models_dir = os.path.join(pkg_dir, 'models')
    world_file = os.path.join(launch_dir, 'tello_world.sdf')

    gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    new_resource_path = f"{models_dir}:{launch_dir}"
    if gz_resource_path:
        new_resource_path = f"{new_resource_path}:{gz_resource_path}"

    # 1. Start Gazebo headless with our world
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', world_file],
        output='log',
        additional_env={'GZ_SIM_RESOURCE_PATH': new_resource_path},
    )

    # 2. The ROS 2 <-> Gazebo Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # ROS → GZ
            '/model/tello/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/tello/enable@std_msgs/msg/Bool]gz.msgs.Boolean',
            # GZ → ROS
            '/model/tello/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        remappings=[
            ('/model/tello/cmd_vel', '/cmd_vel'),
            ('/model/tello/odometry', '/odom'),
            ('/model/tello/enable', '/enable'),
        ],
        output='log'
    )

    return LaunchDescription([gazebo, bridge])