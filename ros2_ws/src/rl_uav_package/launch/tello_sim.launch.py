from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Start Gazebo with our new custom world
    world_file = '/workspace/ros2_ws/src/rl_uav_package/launch/tello_world.sdf'
    
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen'
    )

    # 2. The ROS 2 <-> Gazebo Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/tello/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/tello/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        ],
        remappings=[
            ('/model/tello/cmd_vel', '/cmd_vel'),
            ('/model/tello/odometry', '/odom')
        ],
        output='screen'
    )

    return LaunchDescription([gazebo, bridge])