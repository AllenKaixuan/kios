#include "behavior_tree/condition_node/condition_node.hpp"

namespace Insertion
{
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    //* HAS OBJECT

    HasObject::HasObject(const std::string &name, const BT::NodeConfig &config, std::shared_ptr<kios::TreeState> tree_state_ptr, std::shared_ptr<kios::TaskState> task_state_ptr)
        : HyperMetaNode<BT::ConditionNode>(name, config, tree_state_ptr, task_state_ptr)
    {
    }
    BT::NodeStatus HasObject::tick()
    {
        if (is_success())
        {
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            return BT::NodeStatus::FAILURE;
        }
    }

    bool HasObject::is_success()
    {
        // ! TODO
        return true;
    }

    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    //* AT POSITION
    AtPosition::AtPosition(const std::string &name, const BT::NodeConfig &config, std::shared_ptr<kios::TreeState> tree_state_ptr, std::shared_ptr<kios::TaskState> task_state_ptr)
        : HyperMetaNode<BT::ConditionNode>(name, config, tree_state_ptr, task_state_ptr)
    {
    }

    BT::NodeStatus AtPosition::tick()
    {
        if (is_success())
        {
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            return BT::NodeStatus::FAILURE;
        }
    }

    bool AtPosition::is_success()
    {
        // ! TODO
        return false;
    }
} // namespace Insertion