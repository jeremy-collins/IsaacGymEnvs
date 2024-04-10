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
    constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]
    # constraints = [cp.norm(x, "inf") <= max_action]

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
        cfg = compose(config_name="config", overrides=[
                                                   "task=ManipulabilityVectorizedArticulateTaskSpray1_toy",
                                                #    "task.env.objectType=scissors",
                                                    "task.env.objectType=spray_bottle",
                                                        "train=ArticulateTaskPPONew",
                                                    "task.env.observationType=full_state",
                                                    "sim_device=cpu", 
                                                    "pipeline=cpu", 
                                                    "test=false",
                                                    "task.env.useRelativeControl=true",
                                                    "+env.set_start_pos=false",
                                                    "+task.env.hand_init_path=allegro_hand_dof_default_pos_spray_6.npy",
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

    obs, r, done, info = env.step(initial_action) # TODO: see if this is modifying the state too much
    print("initial hand dofs pos after action: ", env.shadow_hand_dof_pos[0])

    return env, ax

def single_goal_greedy(env, goal, goal_eps=0.01):
    total_cost = 0
    while True:
        initial_action = torch.zeros(22)
        cost, action = optimize_action_cvxpy(env=env, goal=goal)

        total_cost += cost
        
        # adding noise to the optimal action
        # action += torch.randn_like(action) * 0.01

        plot_live_cost_bar((1 - cost), ax)
        
        action = torch.zeros(22)
        action = action.unsqueeze(0).repeat(env.num_envs, 1) # This only works if the action is the same for all envs
        obs, r, done, info = env.step(action)

def traj_goal_greedy(goal_traj, env, goal_eps=0.01):
    initial_action = torch.zeros(22)
    traj_idx = 0
    total_cost = 0
    while traj_idx < len(goal_traj):
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
            traj_idx = torch.argmin(torch.norm(goal_traj - env.current_obs_dict["manip_obs"][0], dim=1))
            goal = goal_traj[traj_idx]

    print("reached goal")
    print("total cost", total_cost)

def get_fingertip_contacts(env):
    contacts = env.net_cf.view(env.num_envs, -1, 3)
    fingertip_contacts = contacts[:, env.fingertip_indices]
    return fingertip_contacts

def get_binary_contacts(env, thresh=0.01):
    fingertip_contacts = get_fingertip_contacts(env)
    binary_contact = (torch.norm(fingertip_contacts, dim=2) > thresh)
    return binary_contact

def imagine_action(env, action, render=False):
    initial_state = np.copy(env.gym.get_sim_rigid_body_states(env.sim, gymapi.STATE_ALL))
    # env_handles = env.envs
    # hand_handles = env.shadow_hands
    # actor_handles = env.object_actor_handles
    # initial_obj_dof_pos_targets = np.copy(env.gym.get_vec_actor_dof_states(env_handles, actor_handles, gymapi.STATE_ALL))
    # initial_obj_dof_vel_targets = np.copy(env.gym.get_vec_actor_dof_states(env_handles, actor_handles, gymapi.STATE_ALL))
    # initial_hand_dof_pos_targets = np.copy(env.gym.get_vec_actor_dof_states(env_handles, hand_handles, gymapi.STATE_ALL))
    # initial_hand_dof_vel_targets = np.copy(env.gym.get_vec_actor_dof_states(env_handles, hand_handles, gymapi.STATE_ALL))
    
    if not render:
        env.force_render = False
    obs, r, done, info = env.step(action) # set, simulate, refresh
    # env.gym.fetch_results(env.sim, True)
    env.force_render = True

    next_obs_dict = env.current_obs_dict
    next_state = np.copy(env.gym.get_sim_rigid_body_states(env.sim, gymapi.STATE_ALL))
    next_contacts = get_binary_contacts(env)

    # setting back to initial state
    set_state(env, target_state=initial_state)

    # for i in range(env.num_envs):
    #     env.gym.set_actor_dof_position_targets(env_handles[i], actor_handles[i], np.array([initial_obj_dof_pos_targets[i][0][0], initial_obj_dof_pos_targets[i][0][1]], dtype=np.float32))
    #     env.gym.set_actor_dof_velocity_targets(env_handles[i], actor_handles[i], np.array([initial_obj_dof_vel_targets[i][0][0], initial_obj_dof_vel_targets[i][0][1]], dtype=np.float32))

    #     initial_hand_dof_pos_targets_numpy = np.concatenate(([initial_hand_dof_pos_targets[i][j][0] for j in range(22)], [initial_hand_dof_pos_targets[i][j][1] for j in range(22)]), axis=0) # , dtype=np.float32)
    #     initial_hand_dof_vel_targets_numpy = np.concatenate(([initial_hand_dof_vel_targets[i][j][0] for j in range(22)], [initial_hand_dof_vel_targets[i][j][1] for j in range(22)]), axis=0) # , dtype=np.float32)
        
    #     env.gym.set_actor_dof_position_targets(env_handles[i], hand_handles[i], initial_hand_dof_pos_targets_numpy)
    #     env.gym.set_actor_dof_velocity_targets(env_handles[i], hand_handles[i], initial_hand_dof_vel_targets_numpy)
    
    # env.gym.simulate(env.sim)
    # env.gym.fetch_results(env.sim, True)
    # env.gym.refresh_rigid_body_state_tensor(env.sim)

    return next_obs_dict, next_state, next_contacts

def get_manipulability_fd_toy(env, manip_obs_dict, obs_keys_manip, eps=1e-2):
    '''
    M = get_manipulability_fd_toy(env, manip_obs_dict=env.current_obs_dict, obs_keys_manip=["object_dof_pos"]).T.cpu()
    Calculates finite difference manipulability by perturbing each action dimension separately across parallel environments.
    '''

    obs = obs_dict_to_tensor(manip_obs_dict, obs_keys_manip, env.num_envs, env.device)
    num_envs = env.actions.shape[0]
    input_dim = env.actions.shape[1]
    output_dim = obs.shape[1]
    # manipulability_fd = torch.empty((bs, output_dim, input_dim), dtype=torch.float64, device=self.device) # TODO: allocate outside of this function
    
    initial_state = np.copy(env.gym.get_sim_rigid_body_states(env.sim, gymapi.STATE_ALL)) # (num_envs*(23+7))

    assert num_envs % (input_dim * 2) == 0, "the number of environments must be divisible by 2 * input dim for vectorized finite difference manipulability calculation"
    
    num_manips = num_envs // (input_dim * 2)

    initial_state_synced = initial_state
    # # TODO: Move state sync into its own function
    # initial_state_rows = initial_state.reshape((num_envs, -1))[0::(input_dim * 2)] # (num_manips, 30) # select every input_dim-th row
    # repeated_rows = np.repeat(initial_state_rows, repeats=(input_dim * 2), axis=0)
    # initial_state_synced = repeated_rows.reshape((num_manips*input_dim*2, -1)) # (num_manips*input_dim*2, num_rigid_bodies) reshape to original shape

    # identity matrix with adjacent rows having opposite signs ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], ...)
    eps_parallel = torch.eye(input_dim, device=env.device).repeat(num_manips, 1).repeat_interleave(2, dim=0) * eps  # (num_manips*input_dim*2, input_dim)
    eps_parallel[1::2] *= -1 # alternate rows have opposite signs

    # env.actions += eps_parallel
    env.actions = eps_parallel

    manip_args = env.get_manip_args()
    obs_dict_1, _, _, _ = manip_step(manip_args)

    states = obs_dict_to_tensor(obs_dict_1, env.obs_keys_manip, env.num_envs, env.device) # (num_manips*input_dim*2, output_dim)

    next_state_1 = states[0::2] # (num_manips*input_dim, output_dim) # even rows
    next_state_2 = states[1::2] # (num_manips*input_dim, output_dim) # odd rows

    # doutput/dinput
    manipulability_fd = (next_state_1 - next_state_2) / (2 * eps) # (num_manips*input_dim, output_dim)

    # setting to initial states copied every input_sim * 2 envs
    set_state(env, target_state=initial_state_synced)

    return manipulability_fd

def branch_contact_modes(env, goal, render=False):
    # for each fingertip, we want to search through the joints that move the fingertip and choose the one that minimizes the cost
    # we can do this by imagining the action and checking the cost
    # get contacts for each fingertip
    fingertip_contacts = get_fingertip_contacts(env)
    contact_norms = torch.norm(fingertip_contacts, dim=2)
    initial_binary_contact = (contact_norms > 0.01)

    min_cost = np.ones(4) * np.inf
    min_action = torch.zeros(4, env.num_envs, 22)
    for i in range(4): # for each fingertip
        for j in range(4*i + 6, 4*i + 10): # for each joint that moves the fingertip
            for k in np.linspace(-1, 1, 10): # for each action
            # for k in np.concatenate((np.ones(20) * -0.1, np.ones(20) * 0.1)): # for each action
                print(i, j, k)
                action = torch.zeros(22)
                # action[0:6] = env.current_obs_dict["hand_joint_pos_init"][0][0:6]  - env.current_obs_dict["hand_dof_pos"][0][0:6] # set the first 6 dofs to the current position
                # print("initial hand pos", env.current_obs_dict["hand_joint_pos_init"][0])
                # print(env.current_obs_dict["hand_dof_pos"][0])
                action[j] = k # set joint j to value k for finger i
                action = action.unsqueeze(0).repeat(env.num_envs, 1)
                imagined_obs_dict, imagined_state, imagined_contacts = imagine_action(env, action, render=render)

                # calculate cost
                # if cost is less than the current minimum cost, update the minimum cost and the action
                cost, action = optimize_action_cvxpy(env=env, goal=goal)

                print("initial contact", initial_binary_contact[0])
                print("imagined contact", imagined_contacts[0])
                print("imagined hand pos", imagined_obs_dict["hand_dof_pos"][0])
                if cost < min_cost[i] and not (initial_binary_contact[0][i] and not imagined_contacts[0][i]):
                    min_cost[i] = cost
                    min_action[i] = action

        print("min costs", min_cost)
        print("min actions", torch.round(min_action, decimals=2))

def branch_random(env, goal, num_branches=3, noise_mag = .1):
    # creates num_branches random perturbations and applies the optimal action to them
    initial_state = np.copy(env.gym.get_sim_rigid_body_states(env.sim, gymapi.STATE_ALL))

    children = []

    for i in range(num_branches):
        rand_action = torch.randn((22))
        rand_action_copied = rand_action.unsqueeze(0).repeat(env.num_envs, 1)
        rand_action_normalized = rand_action_copied / torch.norm(rand_action_copied, dim=1, keepdim=True) * noise_mag
        # imagined_obs_dict, imagined_state, imagined_contacts = imagine_action(env, rand_action_normalized, render=True)

        obs, r, done, info = env.step(rand_action_normalized) # set, simulate, refresh
        # env.gym.fetch_results(env.sim, True)

        # calculate cost
        cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)

        # Execute optimal action to create state
        optimal_action = optimal_action.unsqueeze(0).repeat(env.num_envs, 1)
        obs, r, done, info = env.step(optimal_action) # set, simulate, refresh
        # env.gym.fetch_results(env.sim, True)

        child_state = np.copy(env.gym.get_sim_rigid_body_states(env.sim, gymapi.STATE_ALL)) # NOTE: we can also save the state after applying delta q*
        contacts = get_binary_contacts(env)

        children.append({'state': child_state, 'contacts': contacts, 'actions': [rand_action], 'q*': optimal_action, 'cost': cost})

        # setting state back to parent node
        set_state(env, target_state=initial_state)

    return children

def test_imagine_action(env):
    '''
    Applying imagined actions to stress test imagine_action.
    The state should remain unchanged if the function is working correctly.
    '''
    for i in range(100):
        rand_action = torch.randn((22))
        rand_action = rand_action.unsqueeze(0).repeat(env.num_envs, 1)
        imagined_obs_dict, imagined_state, imagined_contacts = imagine_action(env, rand_action, render=True)
        time.sleep(.1)


def set_state(env, target_state):
    env.gym.set_sim_rigid_body_states(env.sim, target_state, gymapi.STATE_ALL)
    # update the viewer
    env.gym.step_graphics(env.sim)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    env.gym.sync_frame_time(env.sim)

    # env.gym.refresh_rigid_body_state_tensor(env.sim)

def execute_action_sequence(env, traj, render=True, take_best_action=True):
    actions = traj['actions']
    best_actions = traj['q*']

    if not render:
        env.force_render = False

    for action, best_action in zip(actions, best_actions):
    # for action in actions:
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        best_action = best_action.unsqueeze(0).repeat(env.num_envs, 1)

        obs, r, done, info = env.step(action) # set, simulate, refresh
        # env.gym.fetch_results(env.sim, True)

        if take_best_action:
            obs, r, done, info = env.step(best_action) # set, simulate, refresh
            # env.gym.fetch_results(env.sim, True)

        # next_obs_dict = env.current_obs_dict
        # next_state = np.copy(env.gym.get_sim_rigid_body_states(env.sim, gymapi.STATE_ALL))
        # next_contacts = get_binary_contacts(env)

        time.sleep(1)

def beam_search(env, goal, branching_func, beam_width=3, max_iters=100, cost_thresh=0.1):
    # Applies beam search in contact space, using branching_func() to expand nodes
    root_state = np.copy(env.gym.get_sim_rigid_body_states(env.sim, gymapi.STATE_ALL))
    contacts = get_binary_contacts(env)
    cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)
    queue = deque([{'state': root_state, 'contacts': contacts, 'actions': [], 'q*': optimal_action, 'cost': float('inf')}])
    
    for _ in range(max_iters):
        new_candidates = []
        # Expand each state in the queue
        while queue:
            current_candidate = queue.popleft()
            # setting the state to the popped state
            set_state(env, target_state=current_candidate['state'])

            print("current_candidate", current_candidate)
            cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)
            
            # Use the branching function to generate possible next actions
            children = branching_func(env, goal)

            # For each action, imagine the next state and add it as a new candidate
            for child in children:
                if child['cost'] < cost_thresh:
                    # setting the state back to the root before returning
                    set_state(env, target_state=root_state)

                    # Early stopping if goal is reached
                    return child
                
                new_candidates.append({'state': child['state'],
                                       'contacts': child['contacts'],
                                       'actions': current_candidate['actions'] + child['actions'],
                                       'q*': child['q*'],
                                       'cost': child['cost']})
        
        # Keep only the top N candidates with the lowest cost
        new_candidates.sort(key=lambda x: x['cost'])
        queue.extend(new_candidates[:beam_width]) # first = lowest
        
        if not queue:  # Stop if no candidates are left
            break
    
    # Return best sequence found, even if it doesn't meet the threshold
    best_sequence = sorted(queue, key=lambda x: x['cost'], reverse=True)[0] if queue else []
   
    # setting the state back to the root before returning
    set_state(env, target_state=root_state)

    return best_sequence
                
if __name__ == "__main__":
    env, ax = init_toy_env()

    # test_imagine_action(env)
   
    # goal = env.current_obs_dict["manip_goal"][0]
    goal = torch.tensor([0.075])
    obs_keys_manip = ["object_dof_pos"]

    # single goal
    # single_goal_greedy(env, goal)

    # trajectory goal
    # initial_obj_pose = env.current_obs_dict["manip_obs"][0]
    # goal_obj_pose = torch.tensor([0.075])
    # goal_traj = torch.stack([torch.linspace(initial_obj_pose[i], goal_obj_pose[i], 5) for i in range(initial_obj_pose.shape[0])], dim=1)
    # traj_goal_greedy(goal_traj, env)

    # branch contact modes
    # goal = env.current_obs_dict["manip_goal"][0]
    # branch_contact_modes(env, goal, render=True)
    # branch_random(env, goal)

    best_traj = beam_search(env, goal=goal, branching_func=branch_random, beam_width=3, max_iters=100, cost_thresh=0.01)
    print("best traj: \n", best_traj) # TODO: make sure this returns the entire dict
    print("best action sequence:", best_traj['actions'])
    print("best cost: ", best_traj['cost'])
    # print("best action sequence torch:", torch.tensor(best_traj['actions']))

    time.sleep(1)
    print("executing best action sequence...")
    execute_action_sequence(env, best_traj, render=True, take_best_action=True)
    final_cost, optimal_action = optimize_action_cvxpy(env=env, goal=goal)
    print("cost after executing action sequence:", final_cost)
    print("goal dof pos:", goal)
    print("actual final dof pos:", env.current_obs_dict["object_dof_pos"][0])