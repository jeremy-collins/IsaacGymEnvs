import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from isaacgym.torch_utils import quat_conjugate, quat_mul

action_penalty = lambda act: torch.linalg.norm(act, dim=-1)
l2_dist = lambda x, y: torch.linalg.norm(x - y, dim=-1)
l1_dist = lambda x, y: torch.abs(x - y).sum(dim=-1)


@torch.jit.script
def l2_dist_exp(x, y, eps: float = 1e-1):
    return torch.exp(-torch.linalg.norm(x - y, dim=-1) / eps)

@torch.jit.script
def l2_dist_exp_normalized(x, target):
    return torch.exp(-torch.linalg.norm(x - target, dim=-1, keepdim=True) / target).sum(dim=-1)

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
    return torch.linalg.norm(object_pos - hand_palm_pos, dim=-1) + torch.linalg.norm(object_pos.unsqueeze(1) - fingertip_pos, dim=-1).sum(dim=-1)

@torch.jit.script
def reach_bonus(object_dof_pos, goal_dof_pos, threshold: float = 0.1):
    task_dist = torch.abs(object_dof_pos - goal_dof_pos)
    return torch.where(task_dist < threshold, torch.ones_like(task_dist), torch.zeros_like(task_dist))


@torch.jit.script
def drop_penalty(object_pos, goal_pos, fall_dist: float = 0.24):
    object_pose_err = torch.linalg.norm(object_pos - goal_pos, dim=-1).view(-1, 1) 
    return torch.where(object_pose_err > fall_dist, torch.ones_like(object_pose_err), torch.zeros_like(object_pose_err))

@torch.jit.script
def manipulability_reward(object_pos): # (object_state, robot_state):
    return torch.ones_like(object_pos)

@torch.jit.script
def manipulability_frobenius_norm(manipulability):
    # returns the spectral norm of the manipulability matrix. The first dimension is the batch dimension
    return torch.linalg.norm(manipulability, dim=(-2, -1), ord='fro')

@torch.jit.script
def manipulability_nuclear_norm(manipulability):
    # returns the spectral norm of the manipulability matrix. The first dimension is the batch dimension
    return torch.linalg.norm(manipulability, dim=(-2, -1), ord='nuc')

@torch.jit.script
def manipulability_spectral_norm(manipulability):
    # returns the spectral norm of the manipulability matrix. The first dimension is the batch dimension
    return torch.linalg.norm(manipulability, dim=(-2, -1), ord=2)

@torch.jit.script
def manipulability_goal_cond(manipulability, object_pose, goal_pose):
    # returns frobenius norm of manipulability matrix weighted by distance to goal
    err = object_pose - goal_pose # (batch, 7)

    # (batch, 7), (batch, 7, 22) -> (batch, 22)
    weighted_manip = torch.einsum('ij,ijk->ik', err.float(), manipulability.float())
    return torch.linalg.vector_norm(weighted_manip, ord=2, dim=-1)

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
