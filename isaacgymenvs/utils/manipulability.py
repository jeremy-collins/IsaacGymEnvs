from typing import Union, Dict, Any, Tuple
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from utils.utils import obs_dict_to_tensor

def manip_step(
        kwargs : Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Unrolled step function for debugging manipulability function.

        Args:
            # actions: actions to apply
            kwargs: see manip_args in articulate.py for keys
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations
        """

        # # randomize actions
        # if self.dr_randomizations.get("actions", None):
        #     actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        # action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        kwargs["actions"] = torch.clamp(kwargs["actions"], -kwargs["clip_actions"], kwargs["clip_actions"])
        # apply actions
        # self.pre_physics_step(action_tensor)
        ######### UNROLLING PRE-PHYSICS STEP

        # modify self.cur_targets and self.prev_targets
        # actions = action_tensor.clone().to(kwargs["device"])
        # self.assign_act(actions)

        ######### UNROLLING ASSIGN_ACT

        if kwargs["use_relative_control"]:
            targets = (
                kwargs["prev_targets"][:, kwargs["actuated_dof_indices"]]
                + kwargs["shadow_hand_dof_speed_scale"] * kwargs["dt"] * kwargs["actions"]
            )
            kwargs["cur_targets"][:, kwargs["actuated_dof_indices"]] = tensor_clamp(
                targets,
                kwargs["shadow_hand_dof_lower_limits"][kwargs["actuated_dof_indices"]],
                kwargs["shadow_hand_dof_upper_limits"][kwargs["actuated_dof_indices"]],
            )
        else:
            kwargs["cur_targets"][:, kwargs["actuated_dof_indices"]] = scale(
                kwargs["actions"],
                kwargs["shadow_hand_dof_lower_limits"][kwargs["actuated_dof_indices"]],
                kwargs["shadow_hand_dof_upper_limits"][kwargs["actuated_dof_indices"]],
            )
            kwargs["cur_targets"][:, kwargs["actuated_dof_indices"]] = (
                kwargs["act_moving_average"] * kwargs["cur_targets"][:, kwargs["actuated_dof_indices"]]
                + (1.0 - kwargs["act_moving_average"])
                * kwargs["prev_targets"][:, kwargs["actuated_dof_indices"]]
            )
            kwargs["cur_targets"][:, kwargs["actuated_dof_indices"]] = tensor_clamp(
                kwargs["cur_targets"][:, kwargs["actuated_dof_indices"]],
                kwargs["shadow_hand_dof_lower_limits"][kwargs["actuated_dof_indices"]],
                kwargs["shadow_hand_dof_upper_limits"][kwargs["actuated_dof_indices"]],
            )

        kwargs["prev_targets"][:, kwargs["actuated_dof_indices"]] = kwargs["cur_targets"][
            :, kwargs["actuated_dof_indices"]
        ]
        kwargs["gym"].set_dof_position_target_tensor(
            kwargs["sim"], gymtorch.unwrap_tensor(kwargs["cur_targets"])
        )

        #########

        manip_substeps = 1
        for i in range(manip_substeps):
            # if self.force_render:
            #     self.render()
            # print("simulating (manip_step)")
            kwargs["gym"].simulate(kwargs["sim"])

        # to fix!
        # if self.device == "cpu":
        #     self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        # self.post_physics_step(simulate=simulate)
            
        ######### UNROLLING POST-PHYSICS STEP
            
        # self.progress_buf += 1

        # self.compute_observations(skip_manipulability=True)
        # self.compute_observations_manip()

        # UNROLLING COMPUTE OBSERVATIONS

        ############################
        # print("refreshing (manip_step)")
        kwargs["gym"].refresh_dof_state_tensor(kwargs["sim"])
        kwargs["gym"].refresh_actor_root_state_tensor(kwargs["sim"])
        kwargs["gym"].refresh_rigid_body_state_tensor(kwargs["sim"])

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        #     self.gym.refresh_force_sensor_tensor(self.sim)
        #     self.gym.refresh_dof_force_tensor(self.sim)

        if kwargs["num_objects"] > 1:
            palm_index = [kwargs["palm_index"] + b for b in kwargs["env_num_bodies"]]
        else:
            palm_index = kwargs["palm_index"]

        obs_dict = {}
        # obs_dict["hand_joint_pos"] = unscale(
        #     self.shadow_hand_dof_pos,
        #     self.shadow_hand_dof_lower_limits,
        #     self.shadow_hand_dof_upper_limits,
        # )

        obs_dict["hand_joint_pos"] = unscale(
            kwargs["shadow_hand_dof_pos"],
            kwargs["shadow_hand_dof_lower_limits"],
            kwargs["shadow_hand_dof_upper_limits"],
        )

        obs_dict["hand_joint_vel"] = kwargs["vel_obs_scale"] * kwargs["shadow_hand_dof_vel"]
        obs_dict["object_pose"] = kwargs["root_state_tensor"][kwargs["object_indices"], 0:7]
        obs_dict["object_pos"] = kwargs["root_state_tensor"][kwargs["object_indices"], 0:3]
        obs_dict["object_quat"] = kwargs["root_state_tensor"][kwargs["object_indices"], 3:7]
        obs_dict["goal_pose"] = kwargs["goal_states"][:, 0:7]
        obs_dict["goal_pos"] = kwargs["goal_states"][:, 0:3]
        obs_dict["goal_quat"] = kwargs["goal_states"][:, 3:7]
        obs_dict["object_lin_vel"] = kwargs["root_state_tensor"][kwargs["object_indices"], 7:10]
        obs_dict["object_ang_vel"] = (
            kwargs["vel_obs_scale"] * kwargs["root_state_tensor"][kwargs["object_indices"], 10:13]
        )

        obs_dict["object_dof_pos"] = kwargs["object_dof_pos"].view(kwargs["num_envs"], -1)
        if kwargs["scale_dof_pos"]:
            obs_dict["object_dof_pos"] = unscale(
                obs_dict["object_dof_pos"].view(
                    kwargs["num_envs"] // kwargs["num_objects"], kwargs["num_objects"], -1
                ),
                kwargs["object_dof_lower_limits"],
                kwargs["object_dof_upper_limits"],
            ).view(kwargs["num_envs"], -1)

        if isinstance(kwargs["object_target_dof_pos"], torch.Tensor):
            object_target_dof = (
                kwargs["object_target_dof_pos"].unsqueeze(0)
                .repeat(kwargs["num_envs"] // kwargs["num_objects"], 1)
                .unsqueeze(-1)
            )
        else:
            object_target_dof = kwargs["object_target_dof_pos"] * torch.ones_like(
                obs_dict["object_dof_pos"]
            )

        obs_dict["goal_dof_pos"] = object_target_dof.view(kwargs["num_envs"], -1)
        if kwargs["scale_dof_pos"]:
            obs_dict["goal_dof_pos"] = unscale(
                obs_dict["goal_dof_pos"].view(
                    kwargs["num_envs"] // kwargs["num_objects"], kwargs["num_objects"], -1
                ),
                kwargs["object_dof_lower_limits"],
                kwargs["object_dof_upper_limits"],
            ).view(kwargs["num_envs"], -1)

        obs_dict["hand_init_pos"] = kwargs["hand_init_pos"]
        obs_dict["hand_init_quat"] = kwargs["hand_init_quat"]
        obs_dict["hand_pos"] = kwargs["root_state_tensor"][kwargs["hand_indices"] + 1, 0:3]
        obs_dict["hand_quat"] = kwargs["root_state_tensor"][kwargs["hand_indices"] + 1, 3:7]
        # open and append hand_pos and hand_quat to a npz file for debugging
        # if os.path.exists("hand_pos_quat.npz"):
        #     d = np.load("hand_pos_quat.npz")
        #     hand_pos = np.concatenate([d['hand_pos'], obs_dict["hand_pos"].cpu().numpy()])
        #     hand_quat = np.concatenate([d['hand_quat'], obs_dict["hand_quat"].cpu().numpy()])
        # else:
        #     hand_pos = obs_dict["hand_pos"].cpu().numpy()[:1]
        #     hand_quat = obs_dict["hand_quat"].cpu().numpy()[:1]
        # np.savez("hand_pos_quat.npz", hand_pos=hand_pos, hand_quat=hand_quat)

        obs_dict["hand_palm_pos"] = kwargs["rigid_body_states"][:, palm_index, 0:3].view(
            kwargs["num_envs"], -1
        )
        obs_dict["hand_palm_quat"] = kwargs["rigid_body_states"][:, palm_index, 3:7].view(
            kwargs["num_envs"], -1
        )
        obs_dict["hand_palm_vel"] = kwargs["vel_obs_scale"] * kwargs["rigid_body_states"][
            :, palm_index, 7:10
        ].view(kwargs["num_envs"], -1)
        obs_dict["fingertip_pose_vel"] = kwargs["rigid_body_states"][
            :, kwargs["fingertip_indices"]
        ][:, :, 0:10].view(
            kwargs["num_envs"], -1, 10
        )  # n_envs x 4 x 10
        obs_dict["fingertip_pos"] = obs_dict["fingertip_pose_vel"][:, :, 0:3]
        obs_dict["fingertip_rot"] = obs_dict["fingertip_pose_vel"][:, :, 3:7]
        obs_dict["fingertip_vel"] = obs_dict["fingertip_pose_vel"][:, :, 7:10]
        obs_dict["actions"] = kwargs["actions"]
        obs_dict["hand_joint_pos_err"] = (
            kwargs["prev_targets"][:, kwargs["actuated_dof_indices"]] - obs_dict["hand_joint_pos"]
        )

        obs_dict["object_type"] = (
            to_torch(
                np.concatenate(
                    [
                        [
                            kwargs["SUPPORTED_PARTNET_OBJECTS"].index(otype)
                            for _ in kwargs["object_instance"].get(
                                otype, kwargs["asset_files_dict"][otype]
                            )
                        ]
                        for otype in kwargs["object_type"]
                    ]
                ),
                device=kwargs["device"],
            )
            .repeat(kwargs["num_envs"] // kwargs["num_objects"])
            .unsqueeze(-1)
        )

        obs_dict["object_instance"] = (
            to_torch(kwargs["env_instance_order"], device=kwargs["device"])
            .repeat(kwargs["num_envs"] // kwargs["num_objects"])
            .unsqueeze(-1)
        )

        object_instance_one_hot = torch.nn.functional.one_hot(
            obs_dict["object_instance"].to(torch.int64),
            num_classes=kwargs["max_obj_instances"],
        ).squeeze(-2)
        object_type_one_hot = torch.nn.functional.one_hot(
            obs_dict["object_type"].to(torch.int64),
            num_classes=len(kwargs["SUPPORTED_PARTNET_OBJECTS"]),
        ).squeeze(-2)
        obs_dict["object_instance_one_hot"] = object_instance_one_hot.to(kwargs["device"])
        obs_dict["object_type_one_hot"] = object_type_one_hot.to(kwargs["device"])


        kwargs["current_obs_dict"] = obs_dict
        # for key in self.obs_keys:
        #     if key in obs_dict:
        #         self.obs_dict[key][:] = obs_dict[key]

        obs_tensor = obs_dict_to_tensor(obs_dict, kwargs["obs_keys"], kwargs["num_envs"], kwargs["device"])
        # check obs shape
        assert (
            obs_tensor.shape[-1] == kwargs["num_obs_dict"][kwargs["obs_type"]]
        ), f"Obs shape {obs_tensor.shape} not correct!"
        kwargs["obs_buf"][:] = obs_tensor
        # if self.use_image_obs:
        #     IsaacGymCameraBase.compute_observations(self)

        ############################


        return (
            kwargs["current_obs_dict"],
            None, # self.rew_buf.to(self.rl_device),
            None, # self.reset_buf.to(self.rl_device),
            None # self.extras,
        )

def manip_reset(gym, sim, prev_actor_root_state_tensor, prev_dof_state_tensor, prev_rigid_body_tensor, prev_targets):
    result = gym.set_actor_root_state_tensor(
        sim,
        gymtorch.unwrap_tensor(prev_actor_root_state_tensor)
    )
    # print("set_actor_root_state_tensor in manip_reset with result", result)

    result = gym.set_dof_state_tensor(
        sim,
        gymtorch.unwrap_tensor(prev_dof_state_tensor)
    )
    # print("set_dof_state_tensor in manip_reset with result", result)

    # print("set_rigid_body_state_tensor in manip_reset")
    # gym.set_rigid_body_state_tensor( # doesn't work (flex backend only)
    #     sim,
    #     gymtorch.unwrap_tensor(prev_rigid_body_tensor)
    # )

    result = gym.set_dof_position_target_tensor(
        sim,
        gymtorch.unwrap_tensor(prev_targets),
    )
    # print("set_dof_position_target_tensor in manip_reset with result", result)

def get_manipulability_fd(kwargs):
        ''' Compute the finite difference jacobian to compute manipulability

        Args:
            f (function): dynamics function
            obs_keys: list of keys to obs_dict
            action: action
            eps: finite difference step
        
        '''

        prev_bufs_manip = {
        "prev_actor_root_state_tensor": kwargs["root_state_tensor"].clone(),
        "prev_dof_state_tensor": kwargs["dof_state_tensor"].clone(),
        "prev_rigid_body_tensor": kwargs["rigid_body_states"].clone(),
        }

        obs = obs_dict_to_tensor(kwargs["obs_dict"], kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"])
        bs = kwargs["actions"].shape[0]
        input_dim = kwargs["actions"].shape[1]
        output_dim = obs.shape[1]
        manipulability_fd = torch.empty((bs, output_dim, input_dim), dtype=torch.float64, device=kwargs["device"]) # TODO: allocate outside of this function

        for i in range(input_dim):
            kwargs["actions"][:, i] += kwargs["eps"] # perturbing the input (x + eps)

            obs_dict_1, _, _, _ = kwargs["f"](kwargs)
            kwargs["actions"][:, i] -= 2 * kwargs["eps"] # perturbing the input the other way (x - eps)

            obs_dict_2, _, _, _ = kwargs["f"](kwargs)
            kwargs["actions"][:, i] += kwargs["eps"] # set the input back to initial value

            next_state_1 = obs_dict_to_tensor(obs_dict_1, kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"])
            next_state_2 = obs_dict_to_tensor(obs_dict_2, kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"])

            # doutput/dinput
            manipulability_fd[:, :, i] = (next_state_1 - next_state_2) / (2 * kwargs["eps"]) # each col = doutput[:]/dinput[i]

        return manipulability_fd, prev_bufs_manip

def get_manipulability_fd_parallel_actions(kwargs):
    '''
    Calculates finite difference manipulability by perturbing each action dimension separately across parallel environments.

    Args:
        f (function): dynamics function
        obs_keys: list of keys to obs_dict
        action: action
        eps: finite difference step
    '''

    # prev_bufs_manip = {
    #     "prev_actor_root_state_tensor": kwargs["root_state_tensor"].clone(),
    #     "prev_dof_state_tensor": kwargs["dof_state_tensor"].clone(),
    #     "prev_rigid_body_tensor": kwargs["rigid_body_states"].clone(),
    # }

    obs = obs_dict_to_tensor(kwargs["obs_dict"], kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"])
    bs = kwargs["actions"].shape[0]
    input_dim = kwargs["actions"].shape[1]
    output_dim = obs.shape[1]
    # manipulability_fd = torch.empty((bs, output_dim, input_dim), dtype=torch.float64, device=self.device) # TODO: allocate outside of this function

    assert bs % (input_dim * 2) == 0, "the number of environments must be divisible by 2 * input dim for vectorized finite difference manipulability calculation"
    
    num_manips = bs // (input_dim * 2)

    # copying states so we can compute manipulability in parallel
    actor_root_state_tensor_rows = kwargs["root_state_tensor"].view(bs, 2, 13)[0::(input_dim * 2)] # (num_manips, 2, 13) select every (input_dim*2)-th row
    initial_actor_root_state_tensor_copied_rows = actor_root_state_tensor_rows.repeat_interleave((input_dim * 2), dim=0) # (num_manips*input_dim*2, 2, 13) copy each row input_dim times
    initial_actor_root_state_tensor_copied = initial_actor_root_state_tensor_copied_rows.view(-1, 13) # (num_manips*input_dim*2, 13) reshape to original shape

    dof_state_tensor_rows = kwargs["dof_state_tensor"].view(bs, 23, 2)[0::(input_dim * 2)] # (num_manips, 23, 2) # select every (input_dim*2)-th row
    initial_dof_state_tensor_copied_rows = dof_state_tensor_rows.repeat_interleave((input_dim * 2), dim=0) # (num_manips*input_dim, 24, 2) copy each row input_dim times
    initial_dof_state_tensor_copied = initial_dof_state_tensor_copied_rows.view(-1, 2) # (num_manips*input_dim*24, 2) reshape to original shape

    rigid_body_tensor_rows = kwargs["rigid_body_states"].view(bs, 30, 13)[0::(input_dim * 2)] # (num_manips, 30, 13) # select every input_dim-th row
    initial_rigid_body_tensor_copied_rows = rigid_body_tensor_rows.repeat_interleave((input_dim * 2), dim=0) # (num_manips*input_dim, 30, 13) copy each row input_dim times
    initial_rigid_body_tensor_copied = initial_rigid_body_tensor_copied_rows.view(-1, 30, 13) # (num_manips*input_dim, 30, 13) reshape to original shape

    prev_target_tensor_rows = kwargs["prev_targets"].view(bs, 23)[0::(input_dim * 2)] # (num_manips, 23) # select every input_dim-th row
    initial_prev_target_tensor_copied_rows = prev_target_tensor_rows.repeat_interleave((input_dim * 2), dim=0) # (num_manips*input_dim, 24) copy each row input_dim times
    initial_prev_target_tensor_copied = initial_prev_target_tensor_copied_rows.view(-1, 23) # (num_manips*input_dim, 23) reshape to original shape

    # eye with adjacent rows having opposite signs ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], ...)
    eps_parallel = torch.eye(input_dim, device=kwargs["device"]).repeat(num_manips, 1).repeat_interleave(2, dim=0) * kwargs["eps"]  # (num_manips*input_dim*2, input_dim)
    eps_parallel[1::2] *= -1 # alternate rows have opposite signs

    kwargs["actions"] += eps_parallel

    obs_dict_1, _, _, _ = kwargs["f"](kwargs)

    states = obs_dict_to_tensor(obs_dict_1, kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"]) # (num_manips*input_dim*2, output_dim)

    next_state_1 = states[0::2] # (num_manips*input_dim, output_dim) # even rows
    next_state_2 = states[1::2] # (num_manips*input_dim, output_dim) # odd rows

    # doutput/dinput
    manipulability_fd = (next_state_1 - next_state_2) / (2 * kwargs["eps"]) # (num_manips*input_dim, output_dim)

    prev_bufs_manip = {
        "prev_actor_root_state_tensor": initial_actor_root_state_tensor_copied.clone(),
        "prev_dof_state_tensor": initial_dof_state_tensor_copied.clone(),
        "prev_rigid_body_tensor": initial_rigid_body_tensor_copied.clone(),
        "prev_targets": initial_prev_target_tensor_copied.clone()
    }

    return manipulability_fd, prev_bufs_manip

# def get_manipulability_fd_rand_vec(f, obs_dict, obs_keys, action, eps=1e-2):
def get_manipulability_fd_rand_vec(kwargs):
    '''
    Calculates finite difference manipulability by randomly perturbing actions separately across parallel environments.

    Args:
        f (function): dynamics function
        obs_keys: list of keys to obs_dict
        action: action
        eps: finite difference step
    '''

    # These tensors are refreshed in compute_observations(), so they don't need updating beforehand
    prev_bufs_manip = {
        "prev_actor_root_state_tensor": kwargs["root_state_tensor"].clone(),
        "prev_dof_state_tensor": kwargs["dof_state_tensor"].clone(),
        "prev_rigid_body_tensor": kwargs["rigid_body_states"].clone(),
    }

    obs = obs_dict_to_tensor(kwargs["obs_dict"], kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"])
    bs = kwargs["actions"].shape[0]
    input_dim = kwargs["actions"].shape[1]
    output_dim = obs.shape[1]

    # making eps parallel a random unit vector for each row
    rand = torch.randn((bs, input_dim), device=kwargs["device"])
    eps_parallel = rand / torch.norm(rand, dim=1, keepdim=True) * kwargs["eps"]

    # for i in range(input_dim):
    kwargs["actions"] += eps_parallel # perturbing the input (x + eps for each dimension)

    obs_dict_1, _, _, _ = kwargs["f"](kwargs)
    kwargs["actions"] -= 2 * eps_parallel # perturbing the input the other way (x - eps for each dimension)

    obs_dict_2, _, _, _ = kwargs["f"](kwargs)
    kwargs["actions"] += eps_parallel # set the input back to initial value

    next_state_1 = obs_dict_to_tensor(obs_dict_1, kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"]) # (num_manips*input_dim, output_dim)
    next_state_2 = obs_dict_to_tensor(obs_dict_2, kwargs["obs_keys_manip"], kwargs["num_envs"], kwargs["device"]) # (num_manips*input_dim, output_dim)

    # doutput/dinput
    manipulability_fd = (next_state_1 - next_state_2) / (2 * kwargs["eps"]) # (num_manips*input_dim, output_dim)

    return manipulability_fd, prev_bufs_manip