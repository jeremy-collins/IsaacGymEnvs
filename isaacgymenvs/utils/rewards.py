from isaacgym.torch_utils import quat_conjugate, quat_mul
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import cvxpy as cp
import numpy as np
from isaacgymenvs.utils.manipulability import *

action_penalty = lambda act: torch.linalg.norm(act, dim=-1)
l2_dist = lambda x, y: torch.linalg.norm(x - y, dim=-1)
l1_dist = lambda x, y: torch.abs(x - y).sum(dim=-1)


@torch.jit.script
def l2_dist_exp(x, y, eps: float = 1e-1):
    return torch.exp(-torch.linalg.norm(x - y, dim=-1) / eps)


@torch.jit.script
def l2_dist_exp_normalized(x, target):
    return torch.exp(-torch.linalg.norm(x - target, dim=-1, keepdim=True) / target.abs()).sum(dim=-1)


@torch.jit.script
def l1_dist_exp(x, y, eps: float = 1e-1):
    return torch.exp(-torch.abs(x - y).sum(dim=-1) / eps)


@torch.jit.script
def l1_dist_inv(x, y, eps: float = 1e-1):
    return 1.0 / (torch.abs(x - y).sum(dim=-1) + eps)


@torch.jit.script
def l1_dist_exp_normalized(x, target):
    return torch.exp(-torch.abs(x - target).sum(dim=-1, keepdim=True) / target).sum(dim=-1)


@torch.jit.script
def rot_dist(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
    return 2.0 * rot_dist


@torch.jit.script
def rot_reward(object_rot, target_rot, rot_eps: float = 0.1):
    return 1.0 / torch.abs(rot_dist(object_rot, target_rot) + rot_eps)


@torch.jit.script
def rot_dist_delta(object_rot, target_rot, prev_rot_dist):
    return prev_rot_dist - rot_dist(object_rot, target_rot)


@torch.jit.script
def hand_dist(object_pos, hand_palm_pos, fingertip_pos):
    return torch.linalg.norm(object_pos - hand_palm_pos, dim=-1) + torch.linalg.norm(
        object_pos.unsqueeze(1) - fingertip_pos, dim=-1
    ).sum(dim=-1)


@torch.jit.script
def hand_pose_action_penalty(
    joint_pos, target_joint_pos, joint_deltas, joint_init, alpha: float = 0.5, beta: float = 1.0
):
    # combines 3 terms: control error (joint_pos - target_joint_pos), joint velocity (joint_deltas), and drift from default pos
    return (
        alpha * torch.linalg.norm(joint_pos - target_joint_pos, dim=-1)
        + (1 - alpha) * torch.linalg.norm(joint_deltas, dim=-1)
        + beta * torch.linalg.norm(joint_pos - joint_init, dim=-1)
    )


@torch.jit.script
def reach_bonus(object_dof_pos, goal_dof_pos, threshold: float = 0.1):
    task_dist = torch.abs(object_dof_pos - goal_dof_pos)
    return torch.where(task_dist < threshold, torch.ones_like(task_dist), torch.zeros_like(task_dist))


@torch.jit.script
def drop_penalty(object_pos, goal_pos, fall_dist: float = 0.24):
    object_pose_err = torch.linalg.norm(object_pos - goal_pos, dim=-1).view(-1, 1)
    return torch.where(object_pose_err > fall_dist, torch.ones_like(object_pose_err), torch.zeros_like(object_pose_err))


@torch.jit.script
def manipulability_reward(object_pos):  # (object_state, robot_state):
    return torch.ones_like(object_pos)


@torch.jit.script
def manipulability_frobenius_norm(manipulability):
    # returns the frobenius norm of the manipulability matrix. The first dimension is the batch dimension
    return torch.linalg.norm(manipulability, dim=(-2, -1), ord="fro")


@torch.jit.script
def manipulability_nuclear_norm(manipulability):
    # returns the spectral norm of the manipulability matrix. The first dimension is the batch dimension
    return torch.linalg.norm(manipulability, dim=(-2, -1), ord="nuc")


@torch.jit.script
def manipulability_spectral_norm(manipulability):
    # returns the spectral norm of the manipulability matrix. The first dimension is the batch dimension
    return torch.linalg.norm(manipulability, dim=(-2, -1), ord=2)


@torch.jit.script
def manipulability_goal_cond(manipulability, object_pose, goal_pose):
    # returns frobenius norm of manipulability matrix weighted by distance to goal
    err = object_pose - goal_pose  # (batch, 7)

    # (batch, 7), (batch, 7, 22) -> (batch, 22)
    weighted_manip = torch.einsum("ij,ijk->ik", err.float(), manipulability.float())
    return torch.linalg.norm(weighted_manip, ord=2, dim=-1)


@torch.jit.script
def manipulability_frobenius_norm_vectorized(manipulability, manip_obs, manip_goal, actions):
# def manipulability_frobenius_norm_vectorized(manipulability, obs, goal_obs, obs_keys, actions):
    # returns the frobenius norm of the manipulability matrix. The first dimension is the batch dimension
    # manipulability has shape (bs, output_dim) = (num_manips*input_dim, output_dim)
    # TODO: replace object pose with the observations we're using for manipulability
    input_dim = actions.shape[-1]
    output_dim = manip_obs.shape[-1]

    # (num_manips*input_dim, output_dim) -> (num_manips, output_dim, input_dim)
    manipulability_grouped = manipulability.reshape(-1, input_dim, output_dim).transpose(1, 2)
    # print('manipulability_grouped', manipulability_grouped)

    norms = torch.linalg.norm(manipulability_grouped, dim=(-2, -1), ord="fro")  # (num_manips)

    # each env corresponding to the same manipulability matrix gets the same reward
    rewards = norms.repeat_interleave(input_dim * 2)  # bs = num_manips*input_dim*2
    return rewards


@torch.jit.script
def manipulability_l2_norm_rand_vec(manipulability):
    # returns the frobenius norm of the manipulability matrix. The first dimension is the batch dimension
    # manipulability has shape (bs, output_dim). Note that the number of rows is no longer related to the input dim

    # taking the l2 norm of each row of the manipulability matrix
    return torch.linalg.norm(manipulability, ord=2, dim=-1)


def manipulability_neg_cost(action, manipulability, obj_curr, obj_des, max_action=0.1):
    # minimizing the cost function using cvxpy
    initial_action = torch.zeros_like(action)

    # Set up CVX problem
    x = cp.Variable(initial_action.shape[0])
    M = manipulability.T
    obj_next = obj_curr + M @ x
    cost = cp.norm(obj_next - obj_des)
    objective = cp.Minimize(cost)

    # we also want the first 6 dofs to be 0 and the action to be small (infinity norm <= max_action)
    # constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]
    constraints = [cp.norm(x, "inf") <= max_action]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    cost = torch.tensor(prob.value).float()
    # cost_normalized = prob.value / (torch.norm(obj_des - obj_curr) + 1e-6)
    # action = torch.tensor(x.value).float()

    return -cost


def manipulability_neg_cost_vectorized(actions, manipulability, obj_curr, obj_des, max_action=0.1):
    # minimizing the cost function using cvxpy
    initial_actions = torch.zeros_like(actions[0])
    input_dim = actions.shape[-1]
    output_dim = manipulability.shape[-1]

    # (num_manips*input_dim, output_dim) -> (num_manips, output_dim, input_dim)
    manipulability_grouped = manipulability.reshape(-1, input_dim, output_dim).transpose(1, 2)

    # taking every (input_dim*2)th row of obj_curr and obj_des since they are grouped by manipulability and should be the same
    obj_curr_grouped = obj_curr[::input_dim * 2]
    obj_des_grouped = obj_des[::input_dim * 2]

    costs = []
    for i in range(manipulability_grouped.shape[0]):
        # Set up CVX problem
        x = cp.Variable(initial_actions.shape[0])
        M = manipulability_grouped[i].detach().cpu().numpy()
        obj_curr = obj_curr_grouped[i].detach().cpu().numpy()
        obj_des = obj_des_grouped[i].detach().cpu().numpy()
        obj_next = obj_curr + M @ x
        cost = cp.norm(obj_next - obj_des)
        objective = cp.Minimize(cost)

        # we also want the first 6 dofs to be 0 and the action to be small (infinity norm <= max_action)
        constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]

        prob = cp.Problem(objective, constraints)
        prob.solve()
        cost = torch.tensor(prob.value).float()
        costs.append(cost)

    costs = torch.stack(costs, dim=0)
    rewards = -costs.repeat_interleave(input_dim * 2).to(actions.device)
    # costs_normalized = costs / (np.linalg.norm(obj_des - obj_curr, axis=-1) + 1e-6)
    # rewards = -costs_normalized.repeat_interleave(input_dim * 2).to(actions.device)

    return rewards

def manipulability_neg_cost_two_step_vectorized(actions, manip_initial, manip_final, manip_obs, manip_goal,
                                                future_manip_obs, future_manip_goal, max_action=0.1):
    # minimizing the cost function using cvxpy
    input_dim = actions.shape[-1]
    output_dim = manip_initial.shape[-1]

    # (num_manips*input_dim, output_dim) -> (num_manips, output_dim, input_dim)
    manip_grouped_initial = manip_initial.reshape(-1, input_dim, output_dim).transpose(1, 2)

    # taking every (input_dim*2)th row of obj_curr, obj_des, and action_initial since they are grouped by manipulability and should be the same
    obj_curr_grouped_initial = manip_obs[::input_dim * 2]
    obj_des_grouped_initial = manip_goal[::input_dim * 2]

    # action_initial = actions[::input_dim * 2].repeat_interleave(input_dim * 2, dim=0) # repeat the first action for each manipulability
    action_initial = actions

    costs_initial = []
    for i in range(manip_grouped_initial.shape[0]):
        # Set up CVX problem
        x = cp.Variable(action_initial.shape[1])
        M = manip_grouped_initial[i].detach().cpu().numpy()
        obj_curr_initial = obj_curr_grouped_initial[i].detach().cpu().numpy()
        obj_des_initial = obj_des_grouped_initial[i].detach().cpu().numpy()
        obj_next_inital = obj_curr_initial + M @ x
        cost_initial = cp.norm(obj_next_inital - obj_des_initial)
        objective = cp.Minimize(cost_initial)

        # we also want the first 6 dofs to be 0 and the action to be small (infinity norm <= max_action)
        # constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]
        constraints = [cp.norm(x, "inf") <= max_action]

        prob = cp.Problem(objective, constraints)
        prob.solve()
        cost_initial = torch.tensor(prob.value).float()
        costs_initial.append(cost_initial)

    manip_grouped_final = manip_final.reshape(-1, input_dim, output_dim).transpose(1, 2)
    obj_curr_grouped_final = future_manip_obs[::input_dim * 2]
    obj_des_grouped_final = future_manip_goal[::input_dim * 2]

    # printing all the shapes
    print('manip_grouped_final', manip_grouped_final.shape)
    print('obj_curr_grouped_final', obj_curr_grouped_final.shape)
    print('obj_des_grouped_final', obj_des_grouped_final.shape)

    costs_final = []
    for i in range(manip_grouped_final.shape[0]):
        # Set up CVX problem
        x = cp.Variable(action_initial.shape[1])
        M = manip_grouped_final[i].detach().cpu().numpy()
        obj_curr_final = obj_curr_grouped_final[i].detach().cpu().numpy()
        obj_des_final = obj_des_grouped_final[i].detach().cpu().numpy()
        obj_next_final = obj_curr_final + M @ x
        cost_final = cp.norm(obj_next_final - obj_des_final)
        objective = cp.Minimize(cost_final)

        # printing all the shapes
        print('x', x.shape)
        print('M', M.shape)
        print('obj_curr_final', obj_curr_final.shape)
        print('obj_des_final', obj_des_final.shape)
        print('obj_next_final', obj_next_final.shape)

        # we also want the first 6 dofs to be 0 and the action to be small (infinity norm <= max_action)
        # constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]
        constraints = [cp.norm(x, "inf") <= max_action]

        prob = cp.Problem(objective, constraints)
        prob.solve()
        cost_final = torch.tensor(prob.value).float()
        costs_final.append(cost_final)

    costs_initial = torch.stack(costs_initial, dim=0)
    costs_final = torch.stack(costs_final, dim=0)

    print('costs_initial', costs_initial.shape)
    print('costs_final', costs_final.shape)
    costs_total = costs_initial + costs_final

    rewards = -costs_total.repeat_interleave(input_dim * 2).to(actions.device)
    print('rewards', rewards.shape)
    # costs_normalized = costs / (np.linalg.norm(obj_des - obj_curr, axis=-1) + 1e-6)
    # rewards = -costs_normalized.repeat_interleave(input_dim * 2).to(actions.device)
    # manip_reset(env.gym, env.sim, **env.prev_bufs_manip) # setting to initial state

    return rewards

def tipped_penalty(object_rot, goal_rot, fall_dist: float = 0.24):
    object_pose_err = rot_dist(object_rot, goal_rot).view(-1, 1)
    return torch.where(object_pose_err > fall_dist, torch.ones_like(object_pose_err), torch.zeros_like(object_pose_err))


@torch.jit.script
def joint_limit_penalty(hand_dof_pos, dof_limit_low, dof_limit_high, dof_weights):
    """reward to penalize hand pose nearing min/max of joint limits"""
    # hand_dof_pos is num_envs x num_joints
    dof_weights = dof_weights.view(1, -1)
    return (
        torch.linalg.norm(dof_weights * hand_dof_pos - dof_weights * dof_limit_low, dim=-1, keepdim=True).sum(dim=1)
        + torch.linalg.norm(dof_weights * hand_dof_pos - dof_weights * dof_limit_high, dim=-1, keepdim=True).sum(dim=1)
    ).view(-1, 1)


def parse_reward_params(reward_params_dict):
    rew_params = {}
    for key, value in reward_params_dict.items():
        if isinstance(value, (DictConfig, dict)):
            function = value["reward_fn_partial"]
            function = instantiate(function)
            arguments = value["args"]
            if isinstance(arguments, str):
                arguments = [arguments]
            coefficient = value["scale"]
        else:
            function, arguments, coefficient = value
        rew_params[key] = (function, arguments, coefficient)
    print("Reward parameters:", rew_params)
    return rew_params
