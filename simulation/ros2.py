#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import MoveGroupAction, Grasp
from control_msgs.msg import GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander import MoveGroupCommander, RobotCommander
from tf2_ros import TransformListener, Buffer
import moveit_commander
import sys

class KiosRobotActions(Node):
    def __init__(self):
        super().__init__('kios_robot_actions')
        
        # 初始化moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)
        
        # 创建机器人指挥官和规划组
        self.robot = RobotCommander()
        self.arm_group = MoveGroupCommander("arm")  # 根据你的机器人配置修改组名
        self.gripper_group = MoveGroupCommander("gripper")  # 根据你的机器人配置修改组名
        
        # 设置规划参数
        self.arm_group.set_planning_time(5.0)
        self.arm_group.set_num_planning_attempts(10)
        self.arm_group.set_max_velocity_scaling_factor(0.5)
        self.arm_group.set_max_acceleration_scaling_factor(0.5)
        
        # 创建TF监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    async def pick_up(self, hand: str, tool: str, part: str):
        """
        使用指定的工具抓取零件
        """
        try:
            # 获取物体位姿
            part_pose = self._get_object_pose(part)
            
            # 生成抓取姿势
            grasp = self._generate_grasp(part_pose, tool)
            
            # 移动到预抓取位置
            pre_grasp_pose = self._get_pre_grasp_pose(part_pose)
            self.arm_group.set_pose_target(pre_grasp_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to move to pre-grasp position for {part}")
                return False
                
            # 打开夹爪
            self._control_gripper(tool, "open")
            
            # 移动到抓取位置
            self.arm_group.set_pose_target(part_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to move to grasp position for {part}")
                return False
            
            # 关闭夹爪
            self._control_gripper(tool, "close")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Pick up failed: {str(e)}")
            return False

    async def put_down(self, hand: str, tool: str, part: str):
        """
        放下当前抓取的零件
        """
        try:
            # 获取放置位置
            place_pose = self._get_place_pose(part)
            
            # 移动到预放置位置
            pre_place_pose = self._get_pre_place_pose(place_pose)
            self.arm_group.set_pose_target(pre_place_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to move to pre-place position for {part}")
                return False
            
            # 移动到放置位置
            self.arm_group.set_pose_target(place_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to move to place position for {part}")
                return False
            
            # 打开夹爪
            self._control_gripper(tool, "open")
            
            # 移动到后撤位置
            self.arm_group.set_pose_target(pre_place_pose)
            self.arm_group.go(wait=True)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Put down failed: {str(e)}")
            return False

    async def insert(self, hand: str, tool: str, part1: str, part2: str):
        """
        将part1插入part2
        """
        try:
            # 获取目标插入位置
            insert_pose = self._get_insert_pose(part1, part2)
            
            # 移动到预插入位置
            pre_insert_pose = self._get_pre_insert_pose(insert_pose)
            self.arm_group.set_pose_target(pre_insert_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to move to pre-insert position")
                return False
            
            # 执行插入动作
            self.arm_group.set_pose_target(insert_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to perform insert operation")
                return False
            
            # 松开夹爪
            self._control_gripper(tool, "open")
            
            # 后撤
            self.arm_group.set_pose_target(pre_insert_pose)
            self.arm_group.go(wait=True)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Insert operation failed: {str(e)}")
            return False

    async def change_tool(self, hand: str, tool1: str, tool2: str):
        """
        更换工具
        """
        try:
            # 移动到工具架位置
            tool_rack_pose = self._get_tool_rack_pose(tool1)
            self.arm_group.set_pose_target(tool_rack_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to move to tool rack")
                return False
            
            # 放下当前工具
            self._release_tool(tool1)
            
            # 移动到新工具位置
            new_tool_pose = self._get_tool_rack_pose(tool2)
            self.arm_group.set_pose_target(new_tool_pose)
            success = self.arm_group.go(wait=True)
            
            if not success:
                self.get_logger().error(f"Failed to move to new tool position")
                return False
            
            # 抓取新工具
            self._grab_tool(tool2)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Tool change failed: {str(e)}")
            return False

    def _control_gripper(self, tool: str, command: str):
        """
        控制夹爪开关
        """
        if command == "open":
            width = 0.08  # 根据实际夹爪调整
        else:
            width = 0.0
            
        gripper_cmd = GripperCommand()
        gripper_cmd.position = width
        gripper_cmd.max_effort = 50.0  # 根据需要调整力度
        
        self.gripper_group.set_joint_value_target([width])
        self.gripper_group.go(wait=True)

    def _get_object_pose(self, object_name: str) -> PoseStamped:
        """
        获取物体的位姿
        需要实现具体的TF查询或视觉系统集成
        """
        try:
            # 从TF树中获取物体位姿
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # 目标坐标系
                object_name,  # 源坐标系
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            return pose
            
        except Exception as e:
            self.get_logger().error(f"Failed to get object pose: {str(e)}")
            return None

    def _get_pre_grasp_pose(self, target_pose: PoseStamped) -> PoseStamped:
        """
        计算预抓取位姿
        """
        pre_grasp = PoseStamped()
        pre_grasp.header = target_pose.header
        pre_grasp.pose = target_pose.pose
        pre_grasp.pose.position.z += 0.1  # 在目标位置上方10cm
        return pre_grasp

    def _get_pre_place_pose(self, target_pose: PoseStamped) -> PoseStamped:
        """
        计算预放置位姿
        """
        pre_place = PoseStamped()
        pre_place.header = target_pose.header
        pre_place.pose = target_pose.pose
        pre_place.pose.position.z += 0.1  # 在目标位置上方10cm
        return pre_place

    def _get_tool_rack_pose(self, tool: str) -> PoseStamped:
        """
        获取工具架上特定工具的位姿
        需要根据实际工具架布局实现
        """
        # 这里需要实现具体的工具位置查找逻辑
        tool_poses = {
            "parallelgripper": [0.5, 0.3, 0.1],
            "clampgripper": [0.5, 0.4, 0.1],
            "outwardgripper": [0.5, 0.5, 0.1],
            "inwardgripper": [0.5, 0.6, 0.1],
            "defaultgripper": [0.5, 0.7, 0.1],
        }
        
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = tool_poses[tool][0]
        pose.pose.position.y = tool_poses[tool][1]
        pose.pose.position.z = tool_poses[tool][2]
        # 设置工具架上工具的朝向
        pose.pose.orientation.w = 1.0
        
        return pose

def main(args=None):
    rclpy.init(args=args)
    robot_actions = KiosRobotActions()
    rclpy.spin(robot_actions)
    robot_actions.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
