import json
import os
from pprint import pprint
from typing import List, Tuple, Annotated, TypedDict
import operator
from dotenv import load_dotenv
import datetime

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from kios_bt.bt_stewardship import BehaviorTreeStewardship
from kios_scene.scene_factory import SceneFactory
from kios_bt.bt_factory import BehaviorTreeFactory
from kios_robot.robot_interface import RobotInterface
from kios_world.world_interface import WorldInterface

from kios_agent.kios_graph import (
    plan_updater,
    planner,
    seq_planner_chain,
    human_instruction_chain_v2,
)
from kios_agent.kios_routers import KiosRouterFactory, load_router_from_json

from langgraph.graph import StateGraph, END
from langsmith import traceable

load_dotenv()

from kios_utils.pybt_test import generate_bt_stewardship, render_dot_tree
from kios_utils.bblab_utils import setup_logger

import colorlog
import logging

# Setup logging
kios_handler = colorlog.StreamHandler()
kios_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s(AI)---[%(name)s]:%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "light_blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

lg_logger = setup_logger("Langgraph", level=logging.INFO)
kios_logger = setup_logger(
    name="KIOS", level=logging.INFO, special_handler=kios_handler
)
time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def render_bt(bt_json: json):
    """Render behavior tree"""
    test_class = BehaviorTreeFactory()
    bt = test_class.from_json_to_simple_bt(bt_json)
    bt_stewardship = generate_bt_stewardship(bt)
    render_dot_tree(bt_stewardship)


# Add validation functions from validate.py
def load_world_state_for_validation(world_state_json):
    """Load world state for validation"""
    # Extract available actions
    actions = {}
    if "actions" in world_state_json:
        for action in world_state_json["actions"]:
            actions[action["name"]] = {
                "parameters": action["parameters"],
                "preconditions": action["preconditions"],
                "effects": action["effects"]
            }
    
    # Extract current relations
    relations = {}
    if "relations" in world_state_json:
        for relation in world_state_json["relations"]:
            source = relation["source"]
            name = relation["name"]
            target = relation["target"]
            key = f"{name}({source}, {target})"
            relations[key] = True
    
    return {
        "world_state": world_state_json,
        "actions": actions,
        "relations": relations
    }

def validate_behavior_tree(bt_json, world_data):
    """Validate if behavior tree can be executed"""
    if not bt_json or not world_data:
        return False, "Invalid behavior tree or world state data"
    
    actions = world_data["actions"]
    relations = world_data["relations"]
    
    # Recursively validate nodes
    def validate_node(node):
        node_name = node.get("name", "")
        
        # Parse node type and action
        node_parts = node_name.split(":", 1)
        if len(node_parts) < 2:
            return False, f"Cannot parse node name: {node_name}"
        
        node_type = node_parts[0].strip()
        
        # Validate sequence node
        if node_type == "sequence":
            children = node.get("children", [])
            for child in children:
                success, message = validate_node(child)
                if not success:
                    return False, f"Child node of sequence failed validation: {message}"
            return True, "Sequence node validated successfully"
        
        # Validate selector node
        elif node_type == "selector":
            children = node.get("children", [])
            for child in children:
                success, message = validate_node(child)
                if success:
                    return True, "Selector node has at least one valid child"
            return False, "All children of selector node are invalid"
        
        # Validate target node
        elif node_type == "target":
            # Target node typically checks if a relation exists
            action_content = node_parts[1].strip()
            # Example: "at(robot_arm, point_A)"
            if "(" in action_content and ")" in action_content:
                relation_name = action_content.split("(")[0].strip()
                params_str = action_content.split("(")[1].split(")")[0].strip()
                params = [p.strip() for p in params_str.split(",")]
                
                # Check if this relation already exists
                relation_key = f"{relation_name}({params[0]}, {params[1]})"
                if relation_key in relations:
                    return True, f"Target relation '{relation_key}' already exists"
            
            # Target nodes always return success as they are just checks
            return True, "Target node validated successfully"
        
        # Validate precondition node
        elif node_type == "precondition":
            action_content = node_parts[1].strip()
            # Example: "at(robot_arm, home)"
            if "(" in action_content and ")" in action_content:
                relation_name = action_content.split("(")[0].strip()
                params_str = action_content.split("(")[1].split(")")[0].strip()
                params = [p.strip() for p in params_str.split(",")]
                
                # Check if this relation exists
                relation_key = f"{relation_name}({params[0]}, {params[1]})"
                if relation_key in relations:
                    return True, f"Precondition '{relation_key}' is satisfied"
                else:
                    return False, f"Precondition '{relation_key}' is not satisfied"
            
            return False, f"Cannot parse precondition: {action_content}"
        
        # Validate action node
        elif node_type == "action":
            action_content = node_parts[1].strip()
            # Example: "move(robot_arm, home, point_A)"
            if "(" in action_content and ")" in action_content:
                action_name = action_content.split("(")[0].strip()
                params_str = action_content.split("(")[1].split(")")[0].strip()
                params = [p.strip() for p in params_str.split(",")]
                
                # Check if action exists
                if action_name not in actions:
                    return False, f"Action '{action_name}' does not exist in world definition"
                
                # Check if parameter count matches
                expected_params = actions[action_name]["parameters"]
                if len(params) != len(expected_params):
                    return False, f"Parameter count mismatch for action '{action_name}': expected {len(expected_params)}, got {len(params)}"
                
                # Check preconditions
                for precond in actions[action_name]["preconditions"]:
                    if precond["type"] == "relation":
                        rel = precond["relation"]
                        rel_source = rel["source"]
                        rel_name = rel["name"]
                        rel_target = rel["target"]
                        
                        # Replace parameters
                        if rel_source == "source" and len(params) > 1:
                            rel_source = params[0]
                        if rel_target == "source" and len(params) > 1:
                            rel_target = params[0]
                        if rel_target == "target" and len(params) > 1:
                            rel_target = params[-1]
                        
                        rel_key = f"{rel_name}({rel_source}, {rel_target})"
                        if rel_key not in relations:
                            return False, f"Precondition '{rel_key}' for action '{action_name}' is not satisfied"
                
                return True, f"Action '{action_name}' validated successfully"
            
            return False, f"Cannot parse action: {action_content}"
        
        else:
            return False, f"Unknown node type: {node_type}"
    
    # Validate root node
    return validate_node(bt_json)


####################### Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
scene_path = os.path.join(current_dir, "simple_scene.json")
world_state_path = os.path.join(current_dir, "simple_world_state.json")

####################### Load scene
with open(scene_path, "r") as file:
    scene_json_object = json.load(file)

scene = SceneFactory().create_scene_from_json(scene_json_object)

####################### Load world state
world_interface = WorldInterface()
with open(world_state_path, "r") as file:
    world_state_json = json.load(file)
    world_interface.load_world_from_json(world_state_json)

####################### Robot interface
robot_interface = RobotInterface(
    robot_address="127.0.0.1",
    robot_port=12000,
)
robot_interface.setup_scene(scene)

####################### Behavior tree management
behavior_tree_stewardship = BehaviorTreeStewardship(
    world_interface=world_interface,
    robot_interface=robot_interface,
)

# Data directory
data_dir = os.environ.get("KIOS_DATA_DIR").format(username=os.getlogin())
prompt_dir = os.path.join(data_dir, "prompts")


# Graph state definition
class RobotMoveState(TypedDict):
    user_input: str  # User input for movement instruction
    plan: List[str]  # Movement plan
    action_sequence: List[str]  # Action sequence for current step
    world_state: Annotated[List[dict], operator.add]  # World state
    past_steps: Annotated[List[Tuple], operator.add]  # Past steps
    last_behavior_tree: dict  # Last behavior tree
    last_failed_node: dict  # Last failed node
    runtime_world_state: dict  # Runtime world state after execution
    BTExecutionHasSucceeded: bool  # Whether behavior tree execution succeeded


# User feedback
user_feedback: str = None


##################################################### Graph node functions
@traceable(name="user_input_node_step")
def user_input_step(state: RobotMoveState):
    """Get user input"""
    lg_logger.info(f"-----user_input_step-----")

    kios_logger.info(f"Please enter robot arm movement instruction (e.g., 'Move the robot arm from point A to point B then to point C'):")
    user_input = input()

    return {
        "user_input": user_input,
    }


@traceable(name="sequence_generate_step")
def sequence_generate_step(state: RobotMoveState):
    """Generate action sequence for current step"""
    lg_logger.info(f"-----sequence_generate_step-----")

    global user_feedback

    # Extract the current step from the plan
    this_step = state["plan"][0]
    kios_logger.info(f"Current step: {this_step}")

    # Use the latest world state
    latest_world_state = state["world_state"][-1]

    # Create input for sequence planner
    action_sequence = seq_planner_chain.invoke(
        {
            "user_feedback": user_feedback,
            "plan_goal": this_step,
            "world_state": latest_world_state,
            "start_world_state": latest_world_state,
            "task_instruction": this_step,
        }
    )

    # Check the type of action_sequence and handle accordingly
    if isinstance(action_sequence, str):
        kios_logger.info(f"Generated action sequence: {action_sequence}")
        # Convert string to list if needed
        action_steps = [action_sequence]
    elif hasattr(action_sequence, 'steps'):
        kios_logger.info(f"Generated action sequence: {action_sequence.steps}")
        action_steps = action_sequence.steps
    else:
        kios_logger.info(f"Unexpected action sequence type: {type(action_sequence)}")
        kios_logger.info(f"Action sequence content: {action_sequence}")
        # Fallback to using the raw value
        action_steps = [str(action_sequence)]

    return {
        "action_sequence": action_steps,
        "BTExecutionHasSucceeded": False,
    }


@traceable(name="behavior_tree_generate_step")
def behavior_tree_generate_step(state: RobotMoveState):
    """Generate behavior tree"""
    lg_logger.info(f"-----behavior_tree_generate_step-----")

    global user_feedback

    # Print available actions
    kios_logger.info("Available actions: move_to")
    
    # Use simplified behavior tree generation
    bt_skeleton = human_instruction_chain_v2.invoke(
        {
            "user_feedback": user_feedback,
            "last_behavior_tree": state["last_behavior_tree"],
            "action_sequence": state["action_sequence"],
        }
    )

    # Save behavior tree to log file
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"behavior_tree_{time_stamp}.json")
    with open(log_file, "w") as f:
        json.dump(bt_skeleton, f, indent=2)
    kios_logger.info(f"Behavior tree saved to {log_file}")

    render_bt(bt_skeleton)  # Render tree
    
    # Validate behavior tree before execution
    world_data = load_world_state_for_validation(state["world_state"][-1])
    is_valid, validation_message = validate_behavior_tree(bt_skeleton, world_data)
    
    if is_valid:
        kios_logger.info(f"✅ Behavior tree validation successful: {validation_message}")
    else:
        kios_logger.info(f"❌ Behavior tree validation failed: {validation_message}")
        kios_logger.info("Please provide feedback to fix the behavior tree:")
        user_feedback = input()
        return {
            "last_behavior_tree": bt_skeleton,
            "BTExecutionHasSucceeded": False,
        }

    kios_logger.info(f"How should I improve the behavior tree? Please provide feedback (use 'move_to' action):")
    user_feedback = input()
    
    return {
        "last_behavior_tree": bt_skeleton,
        "BTExecutionHasSucceeded": False,
    }


@traceable(name="behavior_tree_execute_step")
def behavior_tree_execute_step(state: RobotMoveState):
    """Execute behavior tree"""
    lg_logger.info(f"-----behavior_tree_execute_step-----")
    
    this_step = state["plan"][0]
    behavior_tree_skeleton = state["last_behavior_tree"]
    latest_world_state = state["world_state"][-1]

    # Save pre-execution information
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    bt_log_file = os.path.join(log_dir, f"bt_before_execute_{time_stamp}.json")
    with open(bt_log_file, "w") as f:
        json.dump(behavior_tree_skeleton, f, indent=2)
    kios_logger.info(f"Pre-execution behavior tree saved to {bt_log_file}")
    
    step_log_file = os.path.join(log_dir, f"current_step_{time_stamp}.txt")
    with open(step_log_file, "w") as f:
        f.write(f"Current step: {this_step}\n")
        f.write(f"Complete plan: {state['plan']}")
    kios_logger.info(f"Current step information saved to {step_log_file}")

    global behavior_tree_stewardship

    # Execute behavior tree
    tree_result = behavior_tree_stewardship.execute_behavior_tree_skeleton(
        world_state=latest_world_state,
        bt_skeleton=behavior_tree_skeleton,
        scene_json_object=scene_json_object,
        is_simulation=True,  # Use simulation mode
    )
    
    # Save execution results
    result_log_file = os.path.join(log_dir, f"bt_execution_result_{time_stamp}.json")
    with open(result_log_file, "w") as f:
        json.dump({
            "result": tree_result.result,
            "world_state": tree_result.world_state,
            "summary": tree_result.summary if hasattr(tree_result, "summary") else None
        }, f, indent=2)
    kios_logger.info(f"Execution results saved to {result_log_file}")

    # Check results
    if tree_result.result == "success":
        kios_logger.info(f"Behavior tree execution succeeded.")
        return {
            "BTExecutionHasSucceeded": True,
            "past_steps": (this_step, tree_result.result),
            "world_state": [tree_result.world_state],
            "runtime_world_state": tree_result.world_state,
        }
    else:
        kios_logger.info(f"Behavior tree execution failed!")
        return {
            "BTExecutionHasSucceeded": False,
            "world_state": [tree_result.world_state],
            "runtime_world_state": tree_result.world_state,
        }


@traceable(name="planner_step")
def planner_step(state: RobotMoveState):
    """Plan steps based on user input"""
    lg_logger.info(f"-----plan_step-----")

    # Simplified planning
    plan = planner.invoke(
        {
            "user_input": state["user_input"],
            "world_state": state["world_state"],
        }
    )
    kios_logger.info(f"Your instruction plan: {plan.steps}")

    return {"plan": plan.steps}


@traceable(name="plan_updater_step")
def plan_updater_step(state: RobotMoveState):
    """Update plan"""
    lg_logger.info(f"-----plan_updater_step-----")

    output = plan_updater.invoke(
        {
            "user_input": state["user_input"],
            "plan": state["plan"],
            "world_state": state["world_state"],
            "past_steps": state["past_steps"],
        }
    )
    return {
        "plan": output.steps,
        "last_behavior_tree": None,
    }


##################################################### Build graph

workflow = StateGraph(RobotMoveState)

# Add nodes
workflow.add_node("planner", planner_step)
workflow.add_node("sequence_generator", sequence_generate_step)
workflow.add_node("behavior_tree_generator", behavior_tree_generate_step)
workflow.add_node("behavior_tree_executor", behavior_tree_execute_step)
workflow.add_node("plan_updater", plan_updater_step)
workflow.add_node("user_input_node", user_input_step)
workflow.set_entry_point("user_input_node")

# Add edges
workflow.add_edge("planner", "sequence_generator")
workflow.add_edge("sequence_generator", "behavior_tree_generator")

router_factory = KiosRouterFactory()

# Load routers
user_feedback_router = load_router_from_json("user_feedback_router")


def user_feedback_should_end(state: RobotMoveState):
    """User feedback router"""
    lg_logger.debug(f"-----user_feedback_should_end-----")
    global user_feedback

    if user_feedback == "" or not user_feedback:
        return True

    while True:
        route = user_feedback_router(user_feedback)
        if route.name == None:
            user_feedback = input(
                "I don't understand your intent. Do you want to execute the plan or improve the behavior tree?"
            )
        else:
            break
    if route.name == "approve":
        user_feedback = None  # Clear user feedback
        return True  # Go to executor
    elif route.name == "rectify":
        # Keep feedback
        return False  # Return to generator
    else:
        raise ValueError(f"Unsupported route {route.name}!")


workflow.add_conditional_edges(
    "behavior_tree_generator",
    user_feedback_should_end,
    {
        True: "behavior_tree_executor",
        False: "sequence_generator",
    },
)

# Load executor routers
executor_success_router = load_router_from_json("executor_success_router")
executor_failure_router = load_router_from_json("executor_failure_router")


def executor_should_end(state: RobotMoveState):
    """Executor end router"""
    lg_logger.debug(f"-----executor_should_end-----")
    global user_feedback

    if state["BTExecutionHasSucceeded"] == True:
        kios_logger.info(f"Behavior tree execution succeeded.")
        kios_logger.info(f"Has the goal for this step been achieved? Are there any issues?")
        user_feedback = input()

        if user_feedback == "" or not user_feedback:
            user_feedback = None
            return True

        while True:
            route = executor_success_router(user_feedback)
            if route.name == None:
                kios_logger.info("I don't understand your intent. Has the goal been achieved, or are there any issues?")
                user_feedback = input()
            else:
                break

        if route.name in ["finish", "approve"]:
            user_feedback = None
            return True
        elif route.name in ["rectify", "disapprove"]:
            return False
        else:
            raise ValueError(f"Unsupported route {route.name}!")
    else:
        kios_logger.info(f"Behavior tree execution failed.")
        kios_logger.info(f"Please give me feedback to improve the behavior tree, or tell me what to do next.")
        user_feedback = input()
        if user_feedback == "" or not user_feedback:
            kios_logger.info("If you leave the input empty, I can't understand your instructions.")
            kios_logger.info("At least you should tell me what to do next.")
            user_feedback = input()

        while True:
            route = executor_failure_router(user_feedback)
            if route.name == None:
                kios_logger.info("I don't understand your intent. Has the goal been achieved, or are there any issues?")
                user_feedback = input()
            else:
                break

        if route.name in ["approve"]:
            user_feedback = None
            return True
        elif route.name in ["rectify"]:
            return False
        elif route.name in ["retry"]:
            return None
        else:
            raise ValueError(f"Unsupported route {route.name}!")


workflow.add_conditional_edges(
    "behavior_tree_executor",
    executor_should_end,
    {
        True: "plan_updater",
        False: "sequence_generator",
        None: "behavior_tree_executor",
    },
)


def plan_updater_should_end(state: RobotMoveState):
    """Plan updater end router"""
    lg_logger.debug(f"-----plan_updater_should_end-----")

    if state["plan"] == [] or len(state["plan"]) == 0:
        kios_logger.info("Movement plan completed.")
        return True
    else:
        kios_logger.info("Movement plan not yet completed.")
        kios_logger.info(f'Remaining steps: {state["plan"]}')
        return False


workflow.add_conditional_edges(
    "plan_updater",
    plan_updater_should_end,
    {
        True: "user_input_node",
        False: "sequence_generator",
    },
)

# Load user input router
user_input_router = load_router_from_json("user_input_router")


def user_input_should_end(state: RobotMoveState):
    """User input end router"""
    lg_logger.debug(f"-----user_input_should_end-----")

    if not state["user_input"] or state["user_input"] == "":
        return True

    route = user_input_router(state["user_input"])
    if route.name == None:
        kios_logger.info("I don't understand your instruction. Can you provide a new instruction?")
        return None
    elif route.name == "finish":
        return True
    elif route.name == "instruction":
        return False
    else:
        raise ValueError(f"Unsupported route {route.name}!")


workflow.add_conditional_edges(
    "user_input_node",
    user_input_should_end,
    {
        True: END,
        False: "planner",
        None: "user_input_node",
    },
)

app = workflow.compile()

config = {"recursion_limit": 500}

inputs = {
    "world_state": [world_state_json],
}


def core_run():
    """Run main program"""
    for event in app.stream(
        inputs,
        config=config,
    ):
        for k, v in event.items():
            if k != "__end__":
                pass


if __name__ == "__main__":
    core_run()