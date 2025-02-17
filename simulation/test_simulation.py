#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from kios_plan.ros2 import KiosRobotActions
import moveit_commander
import sys

class SimulationTest(Node):
    def __init__(self):
        super().__init__('simulation_test')
        self.robot_actions = KiosRobotActions()
        
    async def run_test_sequence(self):
        """
        运行测试序列
        """
        try:
            # 测试更换工具
            self.get_logger().info("Testing tool change...")
            success = await self.robot_actions.change_tool(
                "left_hand", "defaultgripper", "parallelgripper"
            )
            self.get_logger().info(f"Tool change result: {success}")
            
            # 测试抓取
            self.get_logger().info("Testing pick up...")
            success = await self.robot_actions.pick_up(
                "left_hand", "parallelgripper", "gear1"
            )
            self.get_logger().info(f"Pick up result: {success}")
            
            # 测试插入
            self.get_logger().info("Testing insert...")
            success = await self.robot_actions.insert(
                "left_hand", "parallelgripper", "gear1", "shaft1"
            )
            self.get_logger().info(f"Insert result: {success}")

def main(args=None):
    rclpy.init(args=args)
    
    # 初始化测试节点
    test_node = SimulationTest()
    
    try:
        # 运行测试序列
        rclpy.spin_until_future_complete(
            test_node, 
            test_node.run_test_sequence()
        )
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 