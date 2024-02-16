from isaacgym.torch_utils import quat_conjugate, quat_mul
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

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
