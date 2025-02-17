from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    pkg_share = get_package_share_directory('kios_bt_planning')
    
    # 加载MoveIt配置
    moveit_config = os.path.join(
        pkg_share,
        'config',
        'moveit_config.yaml'
    )
    
    return LaunchDescription([
        # 启动Gazebo仿真器
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        
        # 启动机器人状态发布器
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': 'robot_description'}]
        ),
        
        # 启动MoveIt
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            name='move_group',
            output='screen',
            parameters=[moveit_config]
        ),
        
        # 启动RViz用于可视化
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_share, 'config', 'moveit.rviz')],
            output='screen'
        ),
        
        # 启动测试节点
        Node(
            package='kios_bt_planning',
            executable='test_simulation.py',
            name='simulation_test',
            output='screen'
        )
    ]) 