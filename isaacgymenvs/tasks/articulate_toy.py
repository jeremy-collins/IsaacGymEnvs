from articulate import *
import isaacgymenvs
from omegaconf import OmegaConf
from hydra import compose, initialize
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import cvxpy as cp

def optimize_action_gd(action, manipulability, obj_curr, obj_des, max_iters=1000, alpha=1., action_penalty=.01, clip_grad=None):
    hist_dict = {"costs": [], "action_norms": [], "grad_norms": [], "obj_dists": []}
    action = action.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([action], lr=alpha)
    for i in range(max_iters):
        
        # weighting the trigger more than the pose
        # obj_curr[-1] *= 2
        # obj_des[-1] *= 2

        M = manipulability.T # transposing because of artifact in parallel manipulability computation
        obj_next = obj_curr + M @ action
        cost = torch.norm(obj_next - obj_des) # + action_penalty * torch.norm(action)

        dC = 2 * (M.T @ M) @ action - 2 * M.T @ (obj_curr - obj_des) # gradient of cost wrt action

        # gradient clipping
        if clip_grad is not None and torch.norm(dC) > clip_grad:
            # print("clipping grad to", clip_grad)
            dC = dC / torch.norm(dC) * clip_grad

        # action = action - alpha * dC
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # projected gradient descent
        with torch.no_grad():
            action.clamp_(min=-0.1, max=0.1)

        cost_normalized = cost / torch.norm(obj_des - obj_curr)
        hist_dict["costs"].append(cost.item())
        hist_dict["action_norms"].append(torch.norm(action).item())
        hist_dict["grad_norms"].append(torch.norm(dC).item())
        hist_dict["obj_dists"].append(torch.norm(obj_next - obj_des).item())

    # # normalizing the action if greater than 0.01
    # action_mag = 0.01
    # print("action norm: ", torch.norm(action))
    # if torch.norm(action) > action_mag:
    #     print("normalizing action")
    #     action = action / torch.norm(action) * action_mag

    # analytical_action = torch.linalg.pinv(M) @ (obj_des - obj_curr)
    # print("analytical action", analytical_action)

    print("cost", cost.item())
    print("cost_normalized", cost_normalized.item())
    print("action norm", torch.norm(action).item())
    print("action", action)
    print("M.T @ (obj_curr - obj_des)", M.T @ (obj_curr - obj_des))
    # print("obj dist", torch.norm(obj_curr - obj_des).item())
    # print("M @ action", torch.norm(M @ action).item())
    # print("M mag", torch.norm(M).item())

    # visualizing the optimization
    # plot_info(axs, lines, hist_dict)

    return cost_normalized.item(), action


def optimize_action_cvxpy(action, manipulability, obj_curr, obj_des, ax, max_action=0.1):
    # minimizing the cost function using cvxpy

    # Set up CVX problem
    x = cp.Variable(action.shape[0])
    M = manipulability.T
    obj_next = obj_curr + M @ x
    cost = cp.norm(obj_next - obj_des)
    objective = cp.Minimize(cost)
    # we also want the first 6 dofs to be 0 and the action to be small (infinity norm <= 0.1)
    constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]
    # constraints = [cp.norm(x, "inf") <= max_action]

    prob = cp.Problem(objective, constraints)
    
    # Solve the problem
    prob.solve()
    cost = torch.tensor(prob.value).float()
    cost_normalized = prob.value / torch.norm(obj_des - obj_curr)
    action = torch.tensor(x.value).float()

    print("cost", cost.item())
    print("cost_normalized", cost_normalized.item())
    # print("M.T @ (obj_curr - obj_des)", M.T @ (obj_curr - obj_des))
    # print("action norm", torch.norm(action).item())
    # print("action", action)

    # return prob.value, x.value
    # return torch.tensor(prob.value).float(), torch.tensor(x.value).float()
    return cost_normalized, action

def plot_info(axs, lines, hist_dict):
    # Update plot data instead of creating new plots
    lines[0].set_data(range(len(hist_dict["costs"])), hist_dict["costs"])
    lines[1].set_data(range(len(hist_dict["action_norms"])), hist_dict["action_norms"])
    lines[2].set_data(range(len(hist_dict["grad_norms"])), hist_dict["grad_norms"])
    lines[3].set_data(range(len(hist_dict["obj_dists"])), hist_dict["obj_dists"])

    # titles
    axs[0].set_title("Normalized Cost")
    axs[1].set_title("Action Norm")
    axs[2].set_title("Grad Norm")
    axs[3].set_title("Obj Dist")

    # limits
    axs[0].set_xlim(0, len(hist_dict["costs"]))
    axs[1].set_xlim(0, len(hist_dict["action_norms"]))
    axs[2].set_xlim(0, len(hist_dict["grad_norms"]))
    axs[3].set_xlim(0, len(hist_dict["obj_dists"]))
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[3].set_ylim(0, 1)

    plt.show()
    plt.pause(0.01)

def plot_live_cost_bar(cost, ax):
    ax.clear()
    ax.bar(0, cost, color="r")
    ax.set_ylim(0, 1)
    ax.set_title(r"$1 - \bar{C}$")
    plt.show()
    plt.pause(0.01)


def init_toy_env():
    plt.ion()

    cost_fig, ax = plt.subplots(1, figsize=(2, 5))

    config_path = "../cfg"

    with initialize(config_path=config_path, job_name="test_env"):
        cfg = compose(config_name="config", overrides=[
                                                   "task=ManipulabilityVectorizedArticulateTaskSpray1_toy",
                                                #    "task.env.objectType=scissors",
                                                    "task.env.objectType=spray_bottle",
                                                        "train=ArticulateTaskPPONew",
                                                    "task.env.observationType=full_state",
                                                    "sim_device=cpu", 
                                                    "test=false",
                                                    "task.env.useRelativeControl=true",
                                                    "+env.set_start_pos=false",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos_dexgraspnet_batch_13.npy",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos_dexgraspnet_batch_13_xyz.npy",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos_dexgraspnet_batch_13_zyx.npy",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos_dexgraspnet_batch_13_no_transform.npy",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos.npy",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos_zeros.npy",
                                                    "+task.env.hand_init_path=allegro_hand_dof_default_pos_spray_6.npy",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos_dexgraspnet_batch_5_no_transform.npy",
                                                #    "+task.env.hand_init_path=dgn_grasp_world_obj_batch_5.npy",
                                                #    "+task.env.hand_init_path=allegro_hand_dof_default_pos_scissors_closer.npy",
                                                    "task.env.resetDofPosRandomInterval=0.",
                                                #    "task.env.load_default_pos=true",
                                                #    "checkpoint=./runs/full_state_spray/nn/full_state_spray.pth",
                                                    "num_envs=44",
                                                    "+task.env.dexGraspNet=false",
                                                    # "+task.env.manip_obs_keys=['object_pose', 'object_dof_pos']",
                                                    # "+task.env.manip_goal_keys=['goal_pose', 'goal_dof_pos']",
                                                    # "+task.env.manip_obs_keys=['object_pos']",
                                                    # "+task.env.manip_goal_keys=['goal_pos']",
                                                    "+task.env.manip_obs_keys=['object_dof_pos']",
                                                    "+task.env.manip_goal_keys=['goal_dof_pos']",
                                                    # "-m"
                                                    ])

    env = isaacgymenvs.make(cfg.seed, cfg.task, cfg.num_envs, cfg.sim_device,
                cfg.rl_device, cfg.graphics_device_id, cfg.headless,
                cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg)

    transform = env.gym.get_viewer_camera_transform(env.viewer, env.envs[0])
    pos = transform.p
    rot = transform.r
    cam_pos = gymapi.Vec3()
    cam_pos.x = 0.4
    cam_pos.y = -0.75
    cam_pos.z = 1.5
    cam_target = gymapi.Vec3()
    cam_target.x = 0.4
    cam_target.y = 0.4
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

    print("initial hand dofs pos: ", env.shadow_hand_dof_pos[0])
    print("env.num_envs", env.num_envs)
    # actions = torch.zeros(100000, env.num_envs, 22)
    # actions = torch.sin(torch.arange(0, 1000) / 10).unsqueeze(1).repeat(1, 22).unsqueeze(1).repeat(1, env.num_envs, 1)
    initial_action = torch.zeros(env.num_envs, 22)

    # TODO: implement function to change the pose to match dexgraspnet by applying actions

    obs, r, done, info = env.step(initial_action)
    print("initial hand dofs pos after action: ", env.shadow_hand_dof_pos[0])

    return env, ax

def single_goal_greedy(env, goal, goal_eps=0.01):
    total_cost = 0
    while True:
        initial_action = torch.zeros(22)
        cost, action = optimize_action_cvxpy(action=initial_action,
                                            manipulability=env.current_obs_dict["manipulability"],
                                            obj_curr=env.current_obs_dict["manip_obs"][0],
                                            obj_des=goal,
                                            ax=ax)
        total_cost += cost
        
        # adding noise to the optimal action
        # action += torch.randn_like(action) * 0.01

        plot_live_cost_bar((1 - cost), ax)
        
        action = action.unsqueeze(0).repeat(env.num_envs, 1) # This only works if the action is the same for all envs
        
        obs, r, done, info = env.step(action)

        # print("self.net_cf", env.net_cf.shape)
        # print("self.fingertip_indices", env.fingertip_indices)
        # print("all contacts", env.net_cf.view(env.num_envs, -1).shape)
        # print("all contacts", env.net_cf.view(env.num_envs, -1)[0])
        # print("fingertip contacts", env.net_cf.view(env.num_envs, -1)[:, env.fingertip_indices])
        print("manip norm", torch.norm(env.current_obs_dict["manipulability"]))
        # print("hand dofs pos: ", env.shadow_hand_dof_pos[0])
        # print("error from goal: ", env.current_obs_dict["manip_obs"][0] - env.current_obs_dict["manip_goal"][0])
        print("manip obs", env.current_obs_dict["manip_obs"][0])
        print("manip goal", goal)
        print("total cost", total_cost)

def traj_goal_greedy(goal_traj, env, goal_eps=0.01):
    initial_action = torch.zeros(22)
    traj_idx = 0
    total_cost = 0
    while traj_idx < len(goal_traj):
        goal = goal_traj[traj_idx]
        cost, action = optimize_action_cvxpy(action=initial_action,
                                            manipulability=env.current_obs_dict["manipulability"],
                                            obj_curr=env.current_obs_dict["manip_obs"][0],
                                            obj_des=goal,
                                            ax=ax)
        
        plot_live_cost_bar((1 - cost), ax)
        total_cost += cost

        # adding noise to the optimal action
        # action += torch.randn_like(action) * 0.01

        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        
        obs, r, done, info = env.step(action)

        print("goal error", torch.norm(env.current_obs_dict["manip_obs"][0] - goal))
        print("traj_idx", traj_idx)
        print("manip obs", env.current_obs_dict["manip_obs"][0])
        print("manip goal", goal)

        if torch.norm(env.current_obs_dict["manip_obs"][0] - goal) < goal_eps:
            traj_idx += 1
        
        # we set the goal to the nearest point in the trajectory to the current position
        else:
            traj_idx = torch.argmin(torch.norm(goal_traj - env.current_obs_dict["manip_obs"][0], dim=1))
            goal = goal_traj[traj_idx]

    print("reached goal")
    print("total cost", total_cost)

def get_hand_contacts(env):
    torch.set_printoptions(threshold=10_000)
    initial_action = torch.ones(env.num_envs, 22)

    while True:
        # cost, action = optimize_action_cvxpy(action=initial_action,
        #                                     manipulability=env.current_obs_dict["manipulability"],
        #                                     obj_curr=env.current_obs_dict["manip_obs"][0],
        #                                     obj_des=env.current_obs_dict["manip_goal"][0],
        #                                     ax=ax)
        
        # action = action.unsqueeze(0).repeat(env.num_envs, 1)
        obs, r, done, info = env.step(initial_action)
        
        contacts = env.net_cf.view(env.num_envs, -1)
        print("contacts", contacts[0])
        fingertip_contacts = contacts[:, env.fingertip_indices]


if __name__ == "__main__":
    env, ax = init_toy_env()

    # goal = env.current_obs_dict["manip_goal"][0]
    # single_goal_greedy(env, goal)

    # initial_obj_pose = torch.tensor([-0.12, -0.12, 0.2]) # , 0., 0., 0., 1.])
    # goal_obj_pose = torch.tensor([-0.0, -0.12, 0.2]) # , 0., 0., 0., 1.])

    # initial_obj_pose = env.current_obs_dict["manip_obs"][0]
    # goal_obj_pose = torch.tensor([0.075])
    # goal_traj = torch.stack([torch.linspace(initial_obj_pose[i], goal_obj_pose[i], 10) for i in range(initial_obj_pose.shape[0])], dim=1)
    # traj_goal_greedy(goal_traj, env)

    get_hand_contacts(env)