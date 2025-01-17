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
import time
from isaacgym import gymapi
from isaacgymenvs.utils.manipulability import *
from collections import deque
from mcts import mcts
import copy
import wandb
import argparse
import datetime


def optimize_action_gd(
    action,
    manipulability,
    obj_curr,
    obj_des,
    max_iters=1000,
    alpha=1.0,
    action_penalty=0.01,
    clip_grad=None,
):
    hist_dict = {"costs": [], "action_norms": [], "grad_norms": [], "obj_dists": []}
    action = action.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([action], lr=alpha)
    for i in range(max_iters):

        # weighting the trigger more than the pose
        # obj_curr[-1] *= 2
        # obj_des[-1] *= 2

        M = (
            manipulability.T
        )  # transposing because of artifact in parallel manipulability computation
        obj_next = obj_curr + M @ action
        cost = torch.norm(obj_next - obj_des)  # + action_penalty * torch.norm(action)

        dC = 2 * (M.T @ M) @ action - 2 * M.T @ (
            obj_curr - obj_des
        )  # gradient of cost wrt action

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


def optimize_action_cvxpy(env, goal, ax=None, max_action=0.1):
    # minimizing the cost function using cvxpy

    # Set up CVX problem
    x = cp.Variable(22)
    M = env.current_obs_dict["manipulability"].T.cpu()
    # M = get_manipulability_fd_toy(env, manip_obs_dict=env.current_obs_dict, obs_keys_manip=["object_dof_pos"]).T.cpu()
    # obj_curr = env.current_obs_dict["manip_obs"].cpu()
    obj_curr = env.current_obs_dict["object_dof_pos"].cpu()
    obj_des = goal.cpu()
    obj_next = obj_curr + M @ x
    cost = cp.norm(obj_next - obj_des)
    objective = cp.Minimize(cost)
    # we also want the first 6 dofs to be 0 and the action to be small (infinity norm <= 0.1)
    # constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]
    constraints = [cp.norm(x, "inf") <= max_action]

    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()
    cost = torch.tensor(prob.value).float()
    cost_normalized = prob.value / (torch.norm(obj_des - obj_curr) + 1e-9)
    action = torch.tensor(x.value).float()

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
        cfg = compose(
            config_name="config",
            overrides=[
                "task=ManipulabilityVectorizedArticulateTaskSpray1_toy",
                #    "task.env.objectType=scissors",
                "task.env.objectType=spray_bottle",
                "train=ArticulateTaskPPONew",
                "task.env.observationType=full_state",
                # "sim_device=cpu",
                # "pipeline=cpu",
                "test=false",
                "task.env.useRelativeControl=true",
                "+env.set_start_pos=false",
                "+task.env.hand_init_path=allegro_hand_dof_default_pos_spray_6.npy",
                # "+task.env.hand_init_path=allegro_hand_dof_default_pos_spray_5.npy",
                # "+task.env.hand_init_path=allegro_hand_dof_default_pos_spray_4.npy",
                "task.env.resetDofPosRandomInterval=0.",
                #    "task.env.load_default_pos=true",
                "num_envs=44",
                "+task.env.dexGraspNet=false",
                # "+task.env.manip_obs_keys=['object_pose', 'object_dof_pos']",
                # "+task.env.manip_goal_keys=['goal_pose', 'goal_dof_pos']",
                # "+task.env.manip_obs_keys=['object_pos']",
                # "+task.env.manip_goal_keys=['goal_pos']",
                "+task.env.manip_obs_keys=['object_dof_pos']",
                "+task.env.manip_goal_keys=['goal_dof_pos']",
                "headless=true",
                # "-m"
            ],
        )

    env = isaacgymenvs.make(
        cfg.seed,
        cfg.task,
        cfg.num_envs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )

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

    obs, r, done, info = env.step(
        initial_action
    )  # TODO: see if this is modifying the state too much
    print("initial hand dofs pos after action: ", env.shadow_hand_dof_pos[0])

    return env, ax


def get_goal_dist(env, goal):
    goal = goal.to(env.device)
    # return torch.norm(env.current_obs_dict["manip_obs"][0] - goal)
    return torch.norm(env.current_obs_dict["manip_obs"][0] - goal, p=1)


def single_goal_greedy(env, goal, goal_eps=0.01, max_iters=100):
    total_cost = 0
    # while True:
    for i in range(max_iters):
        initial_action = torch.zeros(22)
        cost, action = optimize_action_cvxpy(env=env, goal=goal)

        total_cost += cost

        # adding noise to the optimal action
        # action += torch.randn_like(action) * 0.01

        plot_live_cost_bar((1 - cost), ax)

        # action = torch.zeros(22)
        action = action.unsqueeze(0).repeat(
            env.num_envs, 1
        )  # This only works if the action is the same for all envs
        obs, r, done, info = env.step(action)

        # print("goal error", torch.norm(env.current_obs_dict["manip_obs"][0] - goal))
        goal_dist = get_goal_dist(env, goal)
        if goal_dist < goal_eps:
            break

    return total_cost, goal_dist


def traj_goal_greedy(goal_traj, env, goal_eps=0.01, max_iters=100):
    initial_action = torch.zeros(22)
    traj_idx = 0
    total_cost = 0
    # while traj_idx < len(goal_traj):
    for i in range(max_iters):
        goal = goal_traj[traj_idx]
        cost, action = optimize_action_cvxpy(env=env, goal=goal)

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
            traj_idx = torch.argmin(
                torch.norm(goal_traj - env.current_obs_dict["manip_obs"][0], dim=1)
            )
            goal = goal_traj[traj_idx]

        goal_dist = get_goal_dist(env, goal_traj[-1])

        if traj_idx == len(goal_traj):
            break

    print("reached goal")
    print("total cost", total_cost)

    return total_cost, goal_dist


def get_fingertip_contacts(env):
    contacts = env.net_cf.view(env.num_envs, -1, 3)
    fingertip_contacts = contacts[:, env.fingertip_indices]
    return fingertip_contacts


def get_binary_contacts(env, thresh=0.01):
    fingertip_contacts = get_fingertip_contacts(env)
    binary_contact = torch.norm(fingertip_contacts, dim=2) > thresh
    return binary_contact


def imagine_action(env, action, render=False):
    initial_state = get_state(env)

    if not render:
        env.force_render = False
    obs, r, done, info = env.step(action)  # set, simulate, refresh
    env.force_render = True

    next_obs_dict = env.current_obs_dict
    next_state = get_state(env)
    next_contacts = get_binary_contacts(env)

    # setting back to initial state
    set_state(env, target_state=initial_state)

    return next_obs_dict, next_state, next_contacts


def sync_states(env, state, input_dim, num_manips):
    # copying states in groups of input_dim*2 so we can compute manipulability in parallel
    state_synced = {}

    # Process actor root state tensor
    actor_root_state_tensor_rows = state["actor_root_state_tensor"].view(env.num_envs, 2, 13)[0::(input_dim * 2)]
    actor_root_state_tensor_copied_rows = actor_root_state_tensor_rows.repeat_interleave((input_dim * 2), dim=0)
    state_synced["actor_root_state_tensor"] = actor_root_state_tensor_copied_rows.view(-1, 13)

    # Process dof state tensor
    dof_state_tensor_rows = state["dof_state_tensor"].view(env.num_envs, 23, 2)[0::(input_dim * 2)]
    dof_state_tensor_copied_rows = dof_state_tensor_rows.repeat_interleave((input_dim * 2), dim=0)
    state_synced["dof_state_tensor"] = dof_state_tensor_copied_rows.view(-1, 2)

    # Process rigid body states
    rigid_body_tensor_rows = state["rigid_body_states"].view(env.num_envs, -1, 13)[0::(input_dim * 2)]
    rigid_body_tensor_copied_rows = rigid_body_tensor_rows.repeat_interleave((input_dim * 2), dim=0)
    state_synced["rigid_body_states"] = rigid_body_tensor_copied_rows.view(num_manips * input_dim * 2, -1, 13)

    # Process dof targets
    prev_target_tensor_rows = state["dof_targets"].view(env.num_envs, 23)[0::(input_dim * 2)]
    prev_target_tensor_copied_rows = prev_target_tensor_rows.repeat_interleave((input_dim * 2), dim=0)
    state_synced["dof_targets"] = prev_target_tensor_copied_rows.view(-1, 23)

    return state_synced


def get_manipulability_fd_toy(env, manip_obs_dict, obs_keys_manip, eps=1e-2):
    """
    Calculates finite difference manipulability by perturbing each action dimension separately across parallel environments.
    """
    obs = obs_dict_to_tensor(manip_obs_dict, obs_keys_manip, env.num_envs, env.device)
    num_envs = env.actions.shape[0]
    input_dim = env.actions.shape[1]
    output_dim = obs.shape[1]

    initial_state = get_state(env)
    assert (
        env.num_envs % (input_dim * 2) == 0
    ), "the number of environments must be divisible by 2 * input dim for vectorized finite difference manipulability calculation"

    num_manips = num_envs // (input_dim * 2)
    initial_state_synced = sync_states(env, initial_state, input_dim, num_manips)

    # identity matrix with adjacent rows having opposite signs ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], ...)
    eps_parallel = (
        torch.eye(input_dim, device=env.device)
        .repeat(num_manips, 1)
        .repeat_interleave(2, dim=0)
        * eps
    )  # (num_manips*input_dim*2, input_dim)
    eps_parallel[1::2] *= -1  # alternate rows have opposite signs

    if env.use_relative_control:
        env.actions = eps_parallel
    else:
        env.actions += eps_parallel

    manip_args = env.get_manip_args()
    obs_dict_1, _, _, _ = manip_step(manip_args)

    states = obs_dict_to_tensor(
        obs_dict_1, env.obs_keys_manip, env.num_envs, env.device
    )  # (num_manips*input_dim*2, output_dim)

    next_state_1 = states[0::2]  # (num_manips*input_dim, output_dim) # even rows
    next_state_2 = states[1::2]  # (num_manips*input_dim, output_dim) # odd rows

    print("next_state_1", next_state_1)
    print("next_state_2", next_state_2)

    # doutput/dinput
    manipulability_fd = (next_state_1 - next_state_2) / (
        2 * eps
    )  # (num_manips*input_dim, output_dim)

    # setting to initial states copied every input_sim * 2 envs
    set_state(env, target_state=initial_state_synced)

    return manipulability_fd


def branch_contact_modes(env, goal, render=True):
    # for each fingertip, we want to search through the joints that move the fingertip and choose the one that minimizes the cost
    # we can do this by imagining the action and checking the cost
    # get contacts for each fingertip
    fingertip_contacts = get_fingertip_contacts(env)
    contact_norms = torch.norm(fingertip_contacts, dim=2)
    initial_binary_contact = contact_norms > 0.01

    min_cost = np.ones(4) * np.inf
    min_action = torch.zeros(4, env.num_envs, 22)
    for i in range(4):  # for each fingertip
        for j in range(
            4 * i + 6, 4 * i + 10
        ):  # for each joint that moves the fingertip
            for k in np.linspace(-1, 1, 10):  # for each action
                # for k in np.concatenate((np.ones(20) * -0.1, np.ones(20) * 0.1)): # for each action
                print(i, j, k)
                action = torch.zeros(22)
                # action[0:6] = env.current_obs_dict["hand_joint_pos_init"][0][0:6]  - env.current_obs_dict["hand_dof_pos"][0][0:6] # set the first 6 dofs to the current position
                action[j] = k  # set joint j to value k for finger i
                action = action.unsqueeze(0).repeat(env.num_envs, 1)
                imagined_obs_dict, imagined_state, imagined_contacts = imagine_action(
                    env, action, render=render
                )

                # calculate cost
                # if cost is less than the current minimum cost, update the minimum cost and the action
                cost, action = optimize_action_cvxpy(env=env, goal=goal)

                print("initial contact", initial_binary_contact[0])
                print("imagined contact", imagined_contacts[0])
                print("imagined hand pos", imagined_obs_dict["hand_dof_pos"][0])
                if cost < min_cost[i] and not (
                    initial_binary_contact[0][i] and not imagined_contacts[0][i]
                ):
                    min_cost[i] = cost
                    min_action[i] = action

        print("min costs", min_cost)
        print("min actions", torch.round(min_action, decimals=2))


def branch_random(env, goal, branching_factor=10, noise_mag=1, num_perturb=1, num_opt=1, render=True):
    # creates num_branches random perturbations and applies the optimal action to them
    initial_state = get_state(env)
    initial_contacts = get_binary_contacts(env)[0]
    
    children = []
    rand_actions = []
    for perturb_action in range(num_perturb):
        # get random action
        rand_action = torch.rand((22)) * 2 * noise_mag - noise_mag # each element is between -noise_mag and noise_mag
        rand_action = rand_action.unsqueeze(0).repeat(env.num_envs, 1)

        if not render:
            env.force_render = False
        obs, r, done, info = env.step(
            rand_action
        )  # set, simulate, refresh

        rand_actions.append(rand_action[0])

    optimal_actions = []
    for opt_action in range(num_opt):
        # calculate optimal action
        cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)
        optimal_action = optimal_action.unsqueeze(0).repeat(env.num_envs, 1)
        obs, r, done, info = env.step(optimal_action)  # set, simulate, refresh
        env.force_render = True
        optimal_actions.append(optimal_action[0])
        print("optimal_actions len after append", len(optimal_actions))

    child_state = get_state(env)
    contacts = get_binary_contacts(env)

    children.append(
        {
            "state": child_state,
            "contacts": contacts[0],
            "actions": [rand_actions],
            "q*": [optimal_actions],
            "cost": cost,
        }
    )

    # setting state back to parent node
    set_state(env, target_state=initial_state)

    return children

def branch(env, goal, prune_func, branching_factor=10, noise_mag=1, num_perturb=1, num_opt=1, render=True, **prune_args):
    # creates num_branches random perturbations and prunes them based on the prune_func
    initial_state = get_state(env)
    initial_contacts = get_binary_contacts(env)[0]
    children = deque()  # Use deque for efficient pop and append operations
    
    while len(children) < branching_factor:
        rand_actions = []
        for _ in range(num_perturb):
            # get random action
            rand_action = torch.rand((22)) * 2 * noise_mag - noise_mag # each element is between -noise_mag and noise_mag
            rand_action = rand_action.unsqueeze(0).repeat(env.num_envs, 1)

            if not render:
                env.force_render = False
            obs, r, done, info = env.step(rand_action)  # set, simulate, refresh

            rand_actions.append(rand_action[0])

        optimal_actions = []
        for _ in range(num_opt):
            # calculate optimal action
            cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)
            optimal_action = optimal_action.unsqueeze(0).repeat(env.num_envs, 1)
            obs, r, done, info = env.step(optimal_action)  # set, simulate, refresh
            env.force_render = True
            optimal_actions.append(optimal_action[0])

        child_state = get_state(env)
        contacts = get_binary_contacts(env)

        print("len(rand_actions)", len(rand_actions))
        print("len(optimal_actions)", len(optimal_actions))

        child = {
                "state": child_state,
                "contacts": contacts[0],
                "actions": [rand_actions],
                "q*": [optimal_actions],
                "cost": cost,
            }
        
        prune_args["initial_contacts"] = initial_contacts

        if prune_func is not None:
            if prune_func(child, **prune_args):
                children.append(child)
        else:
            children.append(child)

        if len(children) == branching_factor:
            break

    # setting state back to parent node
    set_state(env, target_state=initial_state)

    return children

def prune_novel_contact(child, initial_contacts):
    return not torch.all(child["contacts"] == initial_contacts)

def prune_contact_flip(child, initial_contacts, num_flips=1):
    return torch.sum(initial_contacts != child["contacts"]) == num_flips

def prune_desired_contact(child, desired_contacts):
    return torch.all(child["contacts"] == desired_contacts)

def prune_no_contact(child, initial_contacts):
    return torch.any(child["contacts"])

def branch_novel_contact_random(child, initial_contacts, change_contact_prob=0.5):
    change_contact = torch.rand(1) < change_contact_prob

    if change_contact:
        # if the contact state is different from the parent node, return True
        return not torch.all(child["contacts"] == initial_contacts)
    else:
        # if the contact state is the same as the parent node, return True
        return torch.all(child["contacts"] == initial_contacts)

def test_imagine_action(env):
    """
    Applying imagined actions to stress test imagine_action.
    The state should remain unchanged if the function is working correctly.
    """
    for i in range(100):
        rand_action = torch.randn((22))
        rand_action = rand_action.unsqueeze(0).repeat(env.num_envs, 1)
        imagined_obs_dict, imagined_state, imagined_contacts = imagine_action(
            env, rand_action, render=True
        )
        time.sleep(0.1)


def set_state(env, target_state):
    """
    set the state of the env to target_state
    """
    # env.rb_forces *= 0 # TODO: see if we should save these instead
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(target_state["actor_root_state_tensor"])
    )

    env.gym.set_dof_state_tensor(
        env.sim, gymtorch.unwrap_tensor(target_state["dof_state_tensor"])
    )

    env.gym.apply_rigid_body_force_tensors(
        env.sim,
        gymtorch.unwrap_tensor(env.rb_forces),
        None,
        gymapi.LOCAL_SPACE,
    )

    env.gym.set_dof_position_target_tensor(
        env.sim,
        gymtorch.unwrap_tensor(target_state["dof_targets"]),
    )

    env.gym.simulate(env.sim)

    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # update the viewer
    env.gym.step_graphics(env.sim)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    env.gym.sync_frame_time(env.sim)


def get_state(env):
    # get gym GPU state tensors
    return {
        "actor_root_state_tensor": env.root_state_tensor.clone(),
        "dof_state_tensor": env.dof_state.clone(),
        "rigid_body_states": env.rigid_body_states.clone(),
        "dof_targets": env.cur_targets.clone(),
    }


def execute_action_sequence(env, traj, render=True):
    actions_list = traj["actions"]
    best_actions_list = traj["q*"]

    if not render:
        env.force_render = False

    for actions, best_actions in zip(actions_list, best_actions_list):
        for action in actions:
            action = action.unsqueeze(0).repeat(env.num_envs, 1)

            print("executing perturbed action")
            obs, r, done, info = env.step(action)  # set, simulate, refresh

        for best_action in best_actions:
            best_action = best_action.unsqueeze(0).repeat(env.num_envs, 1)

            print("executing optimal action")
            obs, r, done, info = env.step(best_action)  # set, simulate, refresh

        time.sleep(1)


def beam_search(
    env,
    goal,
    branch_func,
    prune_func,
    branching_factor=10,
    beam_width=3,
    max_iters=100,
    cost_thresh=0.1,
    num_opt=1,
    num_perturb=1,
    render=True,
):
    # Applies beam search in contact space, using branch_func() to expand nodes
    root_state = get_state(env)
    root_state = sync_states(env, root_state, input_dim=22, num_manips=1)
    contacts = get_binary_contacts(env)
    # cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)
    queue = deque(
        [
            {
                "state": root_state,
                "contacts": contacts,
                "actions": [],
                "q*": [],
                "cost": float("inf"),
            }
        ]
    )

    for _ in range(max_iters):
        new_candidates = []
        # Expand each state in the queue
        while queue:
            current_candidate = queue.popleft()

            # setting the state to the popped state
            set_state(env, target_state=current_candidate["state"])

            # print("current_candidate actions", current_candidate['actions'])
            print("depth", len(current_candidate["actions"]))
            print("current_candidate cost", current_candidate["cost"])
            cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)

            # Use the branching function to generate possible next actions
            children = branch_func(
                env,
                goal,
                prune_func,
                branching_factor,
                noise_mag=1,
                num_opt=num_opt,
                num_perturb=num_perturb,
                render=render,
            )

            # For each action, imagine the next state and add it as a new candidate
            for child in children:
                new_candidate = {
                    "state": child["state"],
                    "contacts": child["contacts"],
                    "actions": current_candidate["actions"] + child["actions"],
                    "q*": current_candidate["q*"] + child["q*"],
                    "cost": child["cost"],
                }

                new_candidates.append(new_candidate)

                if child["cost"] < cost_thresh:
                    # setting the state back to the root before returning
                    set_state(env, target_state=root_state)

                    # Early stopping if goal is reached
                    # return child # TODO: check here for bug. The branching function only returns one action
                    return new_candidate

        # Keep only the top N candidates with the lowest cost
        new_candidates.sort(key=lambda x: x["cost"])
        queue.extend(new_candidates[:beam_width])  # first = lowest

        if not queue:  # Stop if no candidates are left
            break

    # Return best sequence found, even if it doesn't meet the threshold
    best_sequence = (
        sorted(queue, key=lambda x: x["cost"], reverse=True)[0] if queue else []
    )

    # setting the state back to the root before returning
    set_state(env, target_state=root_state)

    return best_sequence


class ArticulateState:
    def __init__(
        self,
        env,
        branch_func,
        goal,
        branching_factor=10,
        cost_thresh=0.1,
        render=True,
    ):
        self.env = env
        root_state = get_state(env)
        root_state = sync_states(env, root_state, input_dim=22, num_manips=1)
        set_state(env, target_state=root_state)
        # contacts = get_binary_contacts(env)
        # self.state = {'state': root_state, 'contacts': contacts, 'actions': [], 'q*': [], 'cost': float('inf')}
        self.env_state = root_state
        self.branch_func = branch_func
        self.goal = goal
        self.branching_factor = branching_factor
        self.cost_thresh = cost_thresh
        self.render = render

    def getCurrentPlayer(self):
        return 0  # we are always the minimizing player

    def getPossibleActions(self):
        possible_states = self.branch_func(
            self.env, self.goal, self.branching_factor, self.render
        )
        possible_actions = [
            tuple(state["actions"][0].tolist()) for state in possible_states
        ]
        return possible_actions

    def takeAction(self, action):
        new_state = ArticulateState(
            self.env,
            self.branch_func,
            self.goal,
            self.branching_factor,
            self.cost_thresh,
            self.render,
        )
        action = torch.tensor(list(action)).unsqueeze(0).repeat(env.num_envs, 1)
        next_obs_dict, next_state, next_contacts = imagine_action(
            self.env, action, render=self.render
        )
        set_state(self.env, target_state=next_state)
        new_state.env_state = next_state
        return new_state

    def isTerminal(self):
        cost, optimal_action = optimize_action_cvxpy(env=self.env, goal=self.goal)
        return cost < self.cost_thresh

    def getReward(self):
        cost, optimal_action = optimize_action_cvxpy(env=self.env, goal=self.goal)
        return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="", help="Name of the run")
    parser.add_argument("--method", type=str, default="", help="Type of method/action to take")
    parser.add_argument("--branch_func", type=str, default="branch", help="Branching function to use")
    parser.add_argument("--prune_func", type=str, default=None, help="Branching function to use")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for beam search")
    parser.add_argument("--branching_factor", type=int, default=10, help="Branching factor for beam search")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of iterations for beam search")
    parser.add_argument("--cost_thresh", type=float, default=0.1, help="Cost threshold for beam search")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--num_opt", type=int, default=1, help="Number of greedy optimization steps to take at each node")
    parser.add_argument("--num_perturb", type=int, default=1, help="Number of perturbation steps to take at each node")

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="articulate-search")
    else:
        wandb.init(project="articulate-search", mode="disabled")

    # wandb.config.update(config)
    wandb.config.update(args)
    name_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # automatically set the run name using all the arguments that are not None
    wandb_name = "_".join([f"{k}={v}" for k, v in vars(args).items() if v is not None])
    wandb.run.name = f"{wandb_name}_{name_datetime}"

    env, ax = init_toy_env()

    # goal = env.current_obs_dict["manip_goal"][0]
    goal = torch.tensor([0.075])
    obs_keys_manip = ["object_dof_pos"]

    if args.method == "test_imagine_action":
        test_imagine_action(env)

    elif args.method == "single_goal_greedy":
        total_cost, goal_dist = single_goal_greedy(
            env, goal, goal_eps=0.01, max_iters=args.max_iters
        )
        wandb.log({"total_cost": total_cost, "goal_dist": goal_dist})

    elif args.method == "traj_goal_greedy":
        # trajectory goal
        initial_obj_pose = env.current_obs_dict["manip_obs"][0]
        goal_obj_pose = torch.tensor([0.075])
        goal_traj = torch.stack(
            [
                torch.linspace(initial_obj_pose[i], goal_obj_pose[i], 5)
                for i in range(initial_obj_pose.shape[0])
            ],
            dim=1,
        )
        total_cost, goal_dist = traj_goal_greedy(goal_traj, env)
        wandb.log({"total_cost": total_cost, "goal_dist": goal_dist})

    elif args.method == "beam_search":
        # search contact modes
        # goal = env.current_obs_dict["manip_goal"][0]

        # getting function with name args.branch_func
        branch_func = globals()[args.branch_func]  # get function by name
        prune_func = globals()[args.prune_func] if args.prune_func is not None else None
        best_traj = beam_search(
            env,
            goal=goal,
            branch_func=branch_func,
            prune_func=prune_func,
            branching_factor=args.branching_factor,
            beam_width=args.beam_width,
            max_iters=args.max_iters,
            cost_thresh=args.cost_thresh,
            num_opt=args.num_opt,
            num_perturb=args.num_perturb,
            render=True,
        )

        # best_traj = beam_search(env, goal=goal, branch_func=branch_random, branching_factor=100, beam_width=5, max_iters=100, cost_thresh=0.01, render=True)
        # best_traj = beam_search(env, goal=goal, branch_func=branch_novel_contact, branching_factor=10, beam_width=5, max_iters=100, cost_thresh=0.01, render=True)
        # best_traj = beam_search(env, goal=goal, branch_func=branch_novel_contact_random, branching_factor=10, beam_width=5, max_iters=100, cost_thresh=0.01, render=True)
        # best_traj = beam_search(env, goal=goal, branch_func=branch_sample_desired_contact, branching_factor=10, beam_width=5, max_iters=100, cost_thresh=0.01, render=True)
        # best_traj = beam_search(env, goal=goal, branch_func=branch_bit_flip_contact, branching_factor=10, beam_width=5, max_iters=100, cost_thresh=0.01, render=True)
        # TODO: implement a branching function that changes the contact state with a certain probability
        # TODO: implement a branching function that keeps the contact state history and maximizes diversity by choosing the least common contact state
        # TODO: try scissors task
        print(
            "best traj: \n", best_traj
        )  # TODO: make sure this returns the entire dict
        print("best action sequence:", best_traj["actions"])
        print("best cost: ", best_traj["cost"])
        # print("best action sequence torch:", torch.tensor(best_traj['actions']))

        time.sleep(1)
        print("executing best action sequence...")
        execute_action_sequence(env, best_traj, render=True)
        final_cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)
        print("cost after executing action sequence:", final_cost)
        print("goal dof pos:", goal)
        print("actual final dof pos:", env.current_obs_dict["object_dof_pos"][0])

        goal_dist = get_goal_dist(env, goal)
        wandb.log({"goal_dist": goal_dist})
        wandb.log({"best_action_sequence": best_traj["actions"]})

    elif args.method == "mcts":
        # monte carlo tree search
        best_traj = []
        root_state = get_state(env)
        for i in range(1):
            # current_state = get_state(env)
            # set_state(env, target_state=current_state)
            initial_state = ArticulateState(
                env,
                branch_random,
                goal,
                branching_factor=3,
                cost_thresh=0.1,
                render=True,
            )
            # print("initial state: ", initial_state)
            # mcts = mcts(timeLimit=1000)
            mcts = mcts(iterationLimit=10)
            action = mcts.search(initial_state)
            action = (
                torch.tensor(list(action), device=env.device)
                .unsqueeze(0)
                .repeat(env.num_envs, 1)
            )
            best_traj.append(action)
            print("executing next mcts action: ", action)
            obs, r, done, info = env.step(action)  # set, simulate, refresh
            print(f"goal dof pos after step {i}", goal)
            print(
                f"actual final dof pos after step {i}:",
                env.current_obs_dict["object_dof_pos"][0],
            )

        print("executing best action sequence...")
        set_state(env, target_state=root_state)
        for action in best_traj:
            # action = torch.tensor(list(action), device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
            obs, r, done, info = env.step(action)
            time.sleep(1)

        print("goal dof pos:", goal)
        print("actual final dof pos:", env.current_obs_dict["object_dof_pos"][0])

        goal_dist = get_goal_dist(env, goal)
        print("goal_dist", goal_dist)
        wandb.log({"goal_dist": goal_dist})
