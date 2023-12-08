# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from .base.vec_task import VecTask

SUPPORTED_PARTNET_OBJECTS = ["dispenser", "spray_bottle", "pill_bottle", "bottle", "spray_bottle2", "spray_bottle3"]


class AllegroHandGrasp(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.task_reward_scale = self.cfg["env"]["taskRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.object_target_dof_name = self.cfg["env"]["objectDofName"]
        self.object_target_dof_pos = self.cfg["env"]["objectDofTargetPos"]
        self.object_stiffness = self.cfg["env"].get("objectStiffness")
        self.object_damping = self.cfg["env"].get("objectDamping")

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.translation_scale = self.cfg["env"].get("translation_scale", 0.1)
        self.orientation_scale = self.cfg["env"].get("orientation_scale", 0.1)
        self.load_default_pos = self.cfg["env"].get("load_default_pos", False)

        self.object_type = self.cfg["env"]["objectType"]
        self.fix_object_base = self.cfg["env"].get("fixObjectBase", True)
        self.object_mass = self.cfg["env"].get("objectMass", 10)
        assert (
            self.object_type
            in [
                "block",
                "egg",
                "pen",
            ]
            + SUPPORTED_PARTNET_OBJECTS
        )

        self.ignore_z = self.object_type == "pen"

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
        }
        for obj in SUPPORTED_PARTNET_OBJECTS:
            self.asset_files_dict[obj] = f"urdf/objects/{obj}.urdf"

        assert "asset" in self.cfg["env"], "no asset section in config file"

        # update asset file paths
        self.asset_files_dict["block"] = self.cfg["env"]["asset"].get(
            "assetFileNameBlock", self.asset_files_dict["block"]
        )
        self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
        self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])
        self.asset_files_dict["bottle"] = self.cfg["env"]["asset"].get(
            "assetFileNameBottle", self.asset_files_dict["bottle"]
        )
        self.asset_files_dict["spray_bottle"] = self.cfg["env"]["asset"].get(
            "assetFileNameSprayBottle", self.asset_files_dict["spray_bottle"]
        )
        self.asset_files_dict["dispenser"] = self.cfg["env"]["asset"].get(
            "assetFileNameDispenser", self.asset_files_dict["dispenser"]
        )
        self.asset_files_dict["pill_bottle"] = self.cfg["env"]["asset"].get(
            "assetFileNamePillBottle", self.asset_files_dict["pill_bottle"]
        )

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]"
            )

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {"full_no_vel": 64, "full": 87, "full_state": 89}

        self.up_axis = "z"

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 109  # TODO: distinguish returned state vs observation

        assert self.cfg["env"]["numActions"] in [17, 22]  # 22  # 16 hand dof + 1 ee position
        if self.cfg["env"]["numActions"] == 22:  # 22  # 16 hand dof + 1 ee position
            self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type] = 109
        else:
            self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # setup viewer
        if self.viewer != None:
            # cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            # cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self._cam_pos = gymapi.Vec3(0.75, 0.75, 1.5)
            self._cam_target = gymapi.Vec3(0.75, -0.4, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, self._cam_pos, self._cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_dofs_with_object + self.num_object_dofs
            )[:, : self.num_shadow_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        if self.num_object_dofs > 0:
            self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
                :, self.num_shadow_hand_dofs : self.num_dofs_with_object
            ]
            self.goal_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
                :, self.num_dofs_with_object : self.num_dofs_with_object + self.num_object_dofs
            ]
            self.object_dof_pos = self.object_dof_state[..., 0]
            self.object_dof_vel = self.object_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.rb_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1])
        )

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        allegro_hand_asset_file = "urdf/kuka_allegro_description/allegro_grasp.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            allegro_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", allegro_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        print("Num hand dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs
        # self.gym.get_asset_actuator_count(shadow_hand_asset)

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        not_loaded = True
        if os.path.exists("allegro_hand_dof_default_pos.npy") and self.load_default_pos:
            self.shadow_hand_dof_default_pos = np.load("allegro_hand_dof_default_pos.npy")
            not_loaded = False
        self.shadow_hand_dof_default_vel = []
        # self.sensors = []
        # sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props["lower"][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props["upper"][i])
            if not_loaded:
                self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            # print("Max effort: ", shadow_hand_dof_props["effort"][i])
            shadow_hand_dof_props["effort"][i] = 0.5
            shadow_hand_dof_props["stiffness"][i] = 3
            shadow_hand_dof_props["damping"][i] = 0.1
            shadow_hand_dof_props["friction"][i] = 0.01
            shadow_hand_dof_props["armature"][i] = 0.001

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = self.fix_object_base
        self.object_asset = object_asset = self.gym.load_asset(
            self.sim, asset_root, object_asset_file, object_asset_options
        )
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        # DO NOT UNCOMMENT: for debugging purposes only
        # object_dof_props["driveMode"] = 1

        if self.object_type in SUPPORTED_PARTNET_OBJECTS:
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            self.object_target_dof_idx = self.gym.get_asset_dof_dict(object_asset)[self.object_target_dof_name]
            # if self.object_type == "spray_bottle":
            #     self.object_target_dof_idx = -1
            # else:
            #     self.object_target_dof_idx = 0
        else:
            self.num_object_dofs = 0

        if self.object_stiffness:
            object_dof_props["stiffness"][self.object_target_dof_idx] = self.object_stiffness
        if self.object_damping:
            object_dof_props["damping"][self.object_target_dof_idx] = self.object_damping

        self.num_dofs_with_object = self.num_shadow_hand_dofs + self.num_object_dofs

        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.25, self.up_axis_idx))
        shadow_hand_start_pose.r = (
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.5 * np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.97 * np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.25 * np.pi)
        )

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        if "start_pose.npz" in os.listdir(os.path.join(asset_root, os.path.dirname(object_asset_file))):
            start_pose = np.load(os.path.join(asset_root, os.path.dirname(object_asset_file), "start_pose.npz"))
            object_start_pose.p = gymapi.Vec3(*start_pose["pos"])
        else:
            object_start_pose.p.x = -0.12
            object_start_pose.p.y = -0.08
            object_start_pose.p.z = 0.124
        object_start_pose.r.w = 1.0
        object_start_pose.r.x = 0.0
        object_start_pose.r.y = 0.0
        object_start_pose.r.z = 0.0

        # object_start_pose.p.x = shadow_hand_start_pose.p.x
        # pose_dy, pose_dz = -0.2, 0.06

        # object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        # object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.15)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2
        max_agg_shapes = self.num_shadow_hand_shapes + 2

        self.shadow_hands = []
        self.object_actor_handles = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        self.fingertips = list(filter(lambda x: x.endswith('link_3'), self.gym.get_asset_rigid_body_dict(allegro_hand_asset).keys()))
        self.fingertip_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self.fingertips]
        self.object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.shadow_hand_rb_handles = list(range(self.num_shadow_hand_bodies))
        self.object_rb_handles = list(
            range(
                self.num_shadow_hand_bodies,
                self.num_shadow_hand_bodies + self.object_rb_count,
            )
        )

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(
                env_ptr, allegro_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0
            )
            self.hand_start_states.append(
                [
                    shadow_hand_start_pose.p.x,
                    shadow_hand_start_pose.p.y,
                    shadow_hand_start_pose.p.z,
                    shadow_hand_start_pose.r.x,
                    shadow_hand_start_pose.r.y,
                    shadow_hand_start_pose.r.z,
                    shadow_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.palm_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "palm_link")

            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.asymmetric_obs:
            #     for ft_handle in self.fingertip_handles:
            #         env_sensors = []
            #         env_sensors.append(self.gym.create_force_sensor(env_ptr, ft_handle, sensor_pose))
            #         self.sensors.append(env_sensors)

            #     self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 1)
            self.object_actor_handles.append(object_handle)
            rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            rb_props[0].mass = self.object_mass
            # if self.object_type == "spray_bottle":
            #     self.gym.set_actor_scale(env_ptr, object_handle, 0.5)
            assert self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, rb_props, True)
            self.object_init_state.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(
                env_ptr,
                goal_asset,
                goal_start_pose,
                "goal_object",
                i + self.num_envs,
                0,
                1,
            )
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr,
                    object_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )
                self.gym.set_rigid_body_color(
                    env_ptr,
                    goal_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] += 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.shadow_hand_rb_handles = to_torch(self.shadow_hand_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.fingertip_indices = to_torch(self.fingertip_indices, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_hand_reward(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_dof_pos[:, self.object_target_dof_idx],
            self.goal_pos,
            self.goal_dof_pos,
            self.hand_pos,
            self.fingertip_pos,
            self.dist_reward_scale,
            self.task_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
            (self.object_type == "pen"),
        )

        self.extras["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["goal_dist"] = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        self.extras["hand_dist"] = torch.norm(self.hand_pos - self.object_pos, p=2, dim=-1)
        self.extras["fingertip_dist"] = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), p=2, dim=-1).sum(-1)
        self.extras["full_hand_dist"] = self.extras['hand_dist'] + self.extras['fingertip_dist']

        self.extras["task_dist"] = (
            (self.goal_dof_pos - self.object_dof_pos[:, self.object_target_dof_idx]).abs().flatten()
        )
        self.extras["success"] = self.successes
        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets)
                )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.hand_pose_vel = self.rigid_body_states[:, self.palm_index, 0:10]
        self.hand_pos = self.hand_pose_vel[:, 0:3]
        self.hand_rot = self.hand_pose_vel[:, 3:7]
        self.hand_linvel = self.hand_pose_vel[:, 7:10]
        self.fingertip_pose_vel = self.rigid_body_states[:, self.fingertip_indices][:, :, 0:10]  # n_envs x 4 x 10
        self.fingertip_pos = self.fingertip_pose_vel[:, :, 0:3]
        self.fingertip_rot = self.fingertip_pose_vel[:, :, 3:7]
        self.fingertip_vel = self.fingertip_pose_vel[:, :, 7:10]

        # self.object_dof_pos = self.object_dof_state[..., 0]
        # self.object_dof_vel = self.object_dof_state[..., 1]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # for env, handle in zip(self.envs, self.object_actor_handles):
        #     joint_handle = self.gym.get_joint_handle(env, "object", self.object_target_dof_name)
        #     self.gym.set_joint_target_position(env, joint_handle, self.object_target_dof_pos)
        # self.prev_targets[:, self.num_shadow_hand_dofs:self.num_dofs_with_object] = self.object_target_dof_pos
        # assert self.gym.set_dof_position_target_tensor_indexed(
        #         self.sim,
        #         gymtorch.unwrap_tensor(self.prev_targets),
        #         gymtorch.unwrap_tensor(self.object_indices.to(torch.int32)),
        #         len(self.object_indices))
        # self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs:self.num_dofs_with_object, 0] = self.object_target_dof_pos
        # assert self.gym.set_dof_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.dof_state),
        #     gymtorch.unwrap_tensor(self.object_indices.to(torch.int32)),
        #     len(self.object_indices),
        #     )
        # for env, handle in zip(self.envs, self.object_actor_handles):
        #     joint_handle = self.gym.get_joint_handle(env, "object", self.object_target_dof_name)
        #     self.gym.set_joint_target_position(env, joint_handle, self.object_target_dof_pos)

        # if current time in seconds is divisible by 5, print object dof state
        # import time
        # current_time = int(time.time())
        # if current_time % 10 == 0:
        # print("object dof pos: ", self.gym.get_vec_actor_dof_states(self.envs, self.object_actor_handles[0], gymapi.STATE_POS))

        self.goal_pose = self.object_init_state[:, :7]
        self.goal_pos = self.goal_pose[:, 0:3]
        self.goal_dof_pos = (
            torch.ones_like(self.object_dof_state[:, self.object_target_dof_idx, 0]) * self.object_target_dof_pos
        )
        self.goal_rot = self.goal_states[:, 3:7]

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        else:
            print("Unkown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            # hand joint obs: 17
            self.obs_buf[:, 0 : self.num_shadow_hand_dofs] = unscale(
                self.shadow_hand_dof_pos,
                self.shadow_hand_dof_lower_limits,
                self.shadow_hand_dof_upper_limits,
            )

            # hand joint obs + obj cur and goal pos: 17 + 6 = 23
            start_idx = self.num_shadow_hand_dofs
            self.obs_buf[:, start_idx : start_idx + 3] = self.object_pos
            self.obs_buf[:, start_idx + 3 : start_idx + 6] = self.goal_pos
            start_idx += 6

            # obj cur and goal dof pos: 23 + 1 = 24
            self.obs_buf[:, start_idx : start_idx + 1] = self.object_dof_pos[:, self.object_target_dof_idx].unsqueeze(
                -1
            )
            self.obs_buf[:, start_idx + 1 : start_idx + 2] = self.goal_dof_pos.unsqueeze(-1)
            start_idx += 2

            # hand palm pos: 24 + 7 = 31
            self.obs_buf[:, start_idx : start_idx + 3] = self.hand_pos
            self.obs_buf[:, start_idx + 3 : start_idx + 7] = self.hand_rot
            start_idx += 7

            # hand joint actions: 31 + 17 = 48
            self.obs_buf[:, start_idx : start_idx + self.num_actions] = self.actions
            start_idx += self.num_actions
        else:
            self.obs_buf[:, 0 : self.num_shadow_hand_dofs] = unscale(
                self.shadow_hand_dof_pos,
                self.shadow_hand_dof_lower_limits,
                self.shadow_hand_dof_upper_limits,
            )
            self.obs_buf[:, self.num_shadow_hand_dofs : 2 * self.num_shadow_hand_dofs] = (
                self.vel_obs_scale * self.shadow_hand_dof_vel
            )
            start_idx = 2 * self.num_shadow_hand_dofs

            # 2 * 17 = 34
            self.obs_buf[:, start_idx : start_idx + 7] = self.object_pose
            start_idx += 7
            self.obs_buf[:, start_idx : start_idx + 3] = self.object_linvel
            start_idx += 3
            self.obs_buf[:, start_idx : start_idx + 3] = self.vel_obs_scale * self.object_angvel
            start_idx += 3

            # 34 + 13 = 47, add goal pos
            start_idx += 3
            self.obs_buf[:, start_idx : start_idx + 1] = self.object_dof_pos
            self.obs_buf[:, start_idx + 1 : start_idx + 2] = self.goal_dof_pos

            # 47 + 2 = 49, add palm pos/vel
            start_idx += 2
            self.obs_buf[:, start_idx : start_idx + 3] = self.hand_pos
            self.obs_buf[:, start_idx + 3 : start_idx + 6] = self.hand_linvel
            start_idx += 6

            # 49 + 6 = 54
            self.obs_buf[:, start_idx : start_idx + self.num_actions] = self.actions
            start_idx += self.num_actions
            # 54 + 17 = 71
        expected_dim = self.num_obs_dict[self.obs_type]
        if start_idx != expected_dim:
            raise AssertionError(
                "error in stacking observation dims, expected {} dims but got {}".format(expected_dim, start_idx)
            )

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0 : self.num_shadow_hand_dofs] = unscale(
                self.shadow_hand_dof_pos,
                self.shadow_hand_dof_lower_limits,
                self.shadow_hand_dof_upper_limits,
            )
            self.states_buf[:, self.num_shadow_hand_dofs : 2 * self.num_shadow_hand_dofs] = (
                self.vel_obs_scale * self.shadow_hand_dof_vel
            )
            self.states_buf[:, 2 * self.num_shadow_hand_dofs : 3 * self.num_shadow_hand_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )
            self.states_buf[:, 3 * self.num_shadow_hand_dofs : 3 * self.num_shadow_hand_dofs + 3] = self.hand_pos
            self.states_buf[:, 3 * self.num_shadow_hand_dofs + 3 : 3 * self.num_shadow_hand_dofs + 6] = self.hand_linvel

            # 3 * 17 + 6 = 57
            obj_obs_start = 3 * self.num_shadow_hand_dofs + 6  # 3 * num_actions + 6
            self.states_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 3 * num_actions + 19
            # 57 + 13 = 70, add goal pos
            self.states_buf[:, goal_obs_start : goal_obs_start + 1] = self.object_dof_pos[
                :, self.object_target_dof_idx
            ].unsqueeze(-1)
            self.states_buf[:, goal_obs_start + 1 : goal_obs_start + 2] = self.goal_dof_pos.unsqueeze(-1)

            # fingertip observations, state(pose and vel) + force-torque sensors
            # todo - add later
            # num_ft_states = 13 * self.num_fingertips  # 65
            # num_ft_force_torques = 6 * self.num_fingertips  # 30

            # fingertip_obs_start = goal_obs_start + 11  # 78
            # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
            #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

            # obs_end = 70 + 2 = 72
            # obs_total = obs_end + num_actions = 72 + 17 = 89
            obs_end = goal_obs_start + 2  # + num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end : obs_end + self.num_actions] = self.actions
            obs_total = obs_end + self.num_actions
        else:
            self.obs_buf[:, 0 : self.num_shadow_hand_dofs] = unscale(
                self.shadow_hand_dof_pos,
                self.shadow_hand_dof_lower_limits,
                self.shadow_hand_dof_upper_limits,
            )
            self.obs_buf[:, self.num_shadow_hand_dofs : 2 * self.num_shadow_hand_dofs] = (
                self.vel_obs_scale * self.shadow_hand_dof_vel
            )
            self.obs_buf[:, 2 * self.num_shadow_hand_dofs : 3 * self.num_shadow_hand_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )

            self.obs_buf[:, 3 * self.num_shadow_hand_dofs : 3 * self.num_shadow_hand_dofs + 3] = self.hand_pos
            self.obs_buf[:, 3 * self.num_shadow_hand_dofs + 3 : 3 * self.num_shadow_hand_dofs + 6] = self.hand_linvel

            obj_obs_start = 3 * self.num_shadow_hand_dofs + 6  # 17 * 3 + 6 = 57

            self.obs_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 57 + 13 = 70
            self.obs_buf[:, goal_obs_start : goal_obs_start + 1] = self.object_dof_pos[
                :, self.object_target_dof_idx
            ].unsqueeze(-1)
            self.obs_buf[:, goal_obs_start + 1 : goal_obs_start + 2] = self.goal_dof_pos.unsqueeze(-1)

            # TODO: fingertip observations, state(pose and vel) + force-torque sensors
            # num_ft_states = 13 * self.num_fingertips  # 65
            # num_ft_force_torques = 6 * self.num_fingertips  # 30
            # fingertip_obs_start = goal_obs_start + 11  # 78
            # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
            #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

            # obs_end = 70 + 2 = 72
            # obs_total = obs_end + num_actions = 72 + 17 = 89
            obs_end = goal_obs_start + 2
            self.obs_buf[:, obs_end : obs_end + self.num_actions] = self.actions
            obs_total = obs_end + self.num_actions

        expected_dim = self.num_obs_dict[self.obs_type]
        if obs_total != expected_dim:
            raise AssertionError(
                "error in stacking observation dims, expected {} dims but got {}".format(expected_dim, obs_total)
            )

    def compute_full_observations_new(self, no_vel=False):
        obs_dict = {}

        # hand joint positions
        obs_dict["hand_joint_pos"] = unscale(
            self.shadow_hand_dof_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )

        if no_vel:
            # object and goal positions
            obs_dict["object_pos"] = self.object_pos
            obs_dict["goal_pos"] = self.goal_pos

            # object and goal dof positions
            obs_dict["object_dof_pos"] = self.object_dof_pos[:, self.object_target_dof_idx].unsqueeze(-1)
            obs_dict["goal_dof_pos"] = self.goal_dof_pos.unsqueeze(-1)

            # hand palm position and rotation
            obs_dict["hand_palm_pos"] = self.hand_pos
            obs_dict["hand_palm_rot"] = self.hand_rot

            # hand joint actions
            obs_dict["hand_actions"] = self.actions

            # create observation key list for no velocity case
            obs_keys_no_vel = [
                "hand_joint_pos",
                "object_pos",
                "goal_pos",
                "object_dof_pos",
                "goal_dof_pos",
                "hand_palm_pos",
                "hand_palm_rot",
                "hand_actions",
            ]
        else:
            # hand joint velocities
            obs_dict["hand_joint_vel"] = self.vel_obs_scale * self.shadow_hand_dof_vel

            # object pose, linear and angular velocities
            obs_dict["object_pose"] = self.object_pose
            obs_dict["object_linvel"] = self.object_linvel
            obs_dict["object_angvel"] = self.vel_obs_scale * self.object_angvel

            # object and goal dof positions
            obs_dict["object_dof_pos"] = self.object_dof_pos[:, self.object_target_dof_idx].unsqueeze(-1)
            obs_dict["goal_dof_pos"] = self.goal_dof_pos.unsqueeze(-1)

            # hand palm position and linear velocity
            obs_dict["hand_palm_pos"] = self.hand_pos
            obs_dict["hand_palm_linvel"] = self.hand_linvel

            # hand joint actions
            obs_dict["hand_actions"] = self.actions

            # create observation key list for velocity case
            obs_keys_vel = [
                "hand_joint_pos",
                "hand_joint_vel",
                "object_pose",
                "object_linvel",
                "object_angvel",
                "object_dof_pos",
                "goal_dof_pos",
                "hand_palm_pos",
                "hand_palm_linvel",
                "hand_actions",
            ]

        # stack observations in buffer
        start_idx = 0
        for key in obs_keys_no_vel if no_vel else obs_keys_vel:
            end_idx = start_idx + obs_dict[key].shape[1]
            self.obs_buf[:, start_idx:end_idx] = obs_dict[key]
            start_idx = end_idx

        expected_dim = self.num_obs_dict[self.obs_type]
        if start_idx != expected_dim:
            raise AssertionError(
                "error in stacking observation dims, expected {} dims but got {}".format(expected_dim, start_idx)
            )

    def compute_full_state_new(self, asymm_obs=False):
        state_dict = {}
        state_keys_asymm_obs = []
        state_keys_no_asymm_obs = []

        # Hand DOF positions and velocities
        state_dict["hand_dof_pos"] = unscale(
            self.shadow_hand_dof_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        state_dict["hand_dof_vel"] = self.vel_obs_scale * self.shadow_hand_dof_vel
        state_dict["hand_force_torque"] = self.force_torque_obs_scale * self.dof_force_tensor

        # Hand palm position and linear velocity
        state_dict["hand_palm_pos"] = self.hand_pos
        state_dict["hand_palm_linvel"] = self.hand_linvel

        # Object pose and velocities
        state_dict["object_pose"] = self.object_pose
        state_dict["object_linvel"] = self.object_linvel
        state_dict["object_angvel"] = self.vel_obs_scale * self.object_angvel

        # Object and goal dof positions
        state_dict["object_dof_pos"] = self.object_dof_pos[:, self.object_target_dof_idx].unsqueeze(-1)
        state_dict["goal_dof_pos"] = self.goal_dof_pos.unsqueeze(-1)

        # Actions
        if hasattr(self, "actions"):
            state_dict["actions"] = self.actions
        else:
            state_dict["actions"] = torch.zeros(self.num_actions, device=self.actions.device)

        # Assemble keys for state dict
        state_keys_asymm_obs.extend(
            [
                "hand_dof_pos",
                "hand_dof_vel",
                "hand_force_torque",
                "hand_palm_pos",
                "hand_palm_linvel",
                "object_pose",
                "object_linvel",
                "object_angvel",
                "object_dof_pos",
                "goal_dof_pos",
                "actions",
            ]
        )

        state_keys_no_asymm_obs.extend(
            [
                "hand_dof_pos",
                "hand_dof_vel",
                "hand_force_torque",
                "hand_palm_pos",
                "hand_palm_linvel",
                "object_pose",
                "object_linvel",
                "object_angvel",
                "object_dof_pos",
                "goal_dof_pos",
                "actions",
            ]
        )

        # Stack states in buffer
        start_idx = 0
        state_keys = state_keys_asymm_obs if asymm_obs else state_keys_no_asymm_obs
        for key in state_keys:
            end_idx = start_idx + state_dict[key].shape[1]
            if asymm_obs:
                self.states_buf[:, start_idx:end_idx] = state_dict[key]
            else:
                self.obs_buf[:, start_idx:end_idx] = state_dict[key]
            start_idx = end_idx

        expected_dim = self.num_obs_dict[self.obs_type]
        if start_idx != expected_dim:
            raise AssertionError(
                "error in stacking observation dims, expected {} dims but got {}".format(expected_dim, start_idx)
            )

    def _reset_done(self):
        # overrides VecTask.reset_done to include goal_env_ids when calling reset_idx
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids, goal_env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_full_state()

        return self.obs_dict, env_ids

    def reset_target_pose(self, env_ids, apply_reset=False):
        self.goal_states[env_ids, 0:7] = self.goal_init_state[env_ids, 0:7]

        # sets goal object position
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = (
            self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        )

        # sets goal object rotation
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        # reset goal task pose
        if isinstance(self.object_target_dof_pos, torch.Tensor) and len(self.object_target_dof_pos) > 1:
            self.goal_dof_state[env_ids, :, 0] = self.object_target_dof_pos.repeat(
                len(env_ids) // len(self.object_target_dof_pos)
            ).unsqueeze(-1)
        else:
            self.goal_dof_state[env_ids, :, 0] = self.object_target_dof_pos  # TODO: resample goal dof state
        goal_indices = self.goal_object_indices[env_ids].to(torch.int32)
        self.prev_targets[
            :, self.num_dofs_with_object : self.num_dofs_with_object + self.num_object_dofs
        ] = self.object_target_dof_pos
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(goal_indices),
            len(env_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(goal_indices),
            len(env_ids),
        )

        # zeroes velocities
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
        )

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices),
                len(env_ids),
            )
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids=None):
        # generate random values
        rand_floats = (
            torch_rand_float(
                -1.0,
                1.0,
                (len(env_ids), self.num_shadow_hand_dofs * 2),
                device=self.device,
            )
            * 0
        )

        # reset start object target poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object position/orientation
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()

        # reset object velocity
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        if goal_env_ids is not None:
            object_indices = [
                self.object_indices[env_ids],
                self.goal_object_indices[env_ids],
                self.goal_object_indices[goal_env_ids],
            ]
        else:
            object_indices = [
                self.object_indices[env_ids],
                self.goal_object_indices[env_ids],
            ]
        object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices),
        )

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, : self.num_shadow_hand_dofs]

        pos = self.shadow_hand_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = (
            self.shadow_hand_dof_default_vel
            + self.reset_dof_vel_noise * rand_floats[:, self.num_shadow_hand_dofs : self.num_shadow_hand_dofs * 2]
        )
        self.prev_targets[env_ids, : self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_indices),
            len(env_ids),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices),
            len(env_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def move_hand_pos(self, actions=None):
        actions = actions if actions is not None else self.actions
        rb_forces, rb_torque = self.rb_forces.clone(), self.rb_torques.clone()
        rb_forces[:, 1, :] = actions[:, 0:3] * self.dt * self.translation_scale * 100000
        rb_forces[:, 1 + 26, :] = actions[:, 26:29] * self.dt * self.translation_scale * 100000
        rb_torque[:, 1, :] = actions[:, 3:6] * self.dt * self.orientation_scale * 1000
        rb_torque[:, 1 + 26, :] = actions[:, 29:32] * self.dt * self.orientation_scale * 1000
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(rb_forces),
            gymtorch.unwrap_tensor(rb_torque),
            gymapi.ENV_SPACE,
        )

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        if self.use_relative_control:
            targets = (
                self.prev_targets[:, self.actuated_dof_indices]
                + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                targets,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
        else:
            # self.move_hand_pos()
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(
                    self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                    device=self.device,
                )
                * self.object_rb_masses
                * self.force_scale
            )

            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                None,
                gymapi.LOCAL_SPACE,
            )

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (
                    (
                        self.hand_pos[i]
                        + quat_apply(
                            self.hand_rot[i],
                            to_torch([1, 0, 0], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )
                targety = (
                    (
                        self.hand_pos[i]
                        + quat_apply(
                            self.hand_rot[i],
                            to_torch([0, 1, 0], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )
                targetz = (
                    (
                        self.hand_pos[i]
                        + quat_apply(
                            self.hand_rot[i],
                            to_torch([0, 0, 1], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )

                p0 = self.hand_pos[i].cpu().numpy() # + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]],
                    [0.85, 0.1, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]],
                    [0.1, 0.85, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]],
                    [0.1, 0.1, 0.85],
                )

            # for i in range(self.num_envs):
            #     targetx = (
            #         (
            #             self.goal_pos[i]
            #             + quat_apply(
            #                 self.goal_rot[i],
            #                 to_torch([1, 0, 0], device=self.device) * 0.2,
            #             )
            #         )
            #         .cpu()
            #         .numpy()
            #     )
            #     targety = (
            #         (
            #             self.goal_pos[i]
            #             + quat_apply(
            #                 self.goal_rot[i],
            #                 to_torch([0, 1, 0], device=self.device) * 0.2,
            #             )
            #         )
            #         .cpu()
            #         .numpy()
            #     )
            #     targetz = (
            #         (
            #             self.goal_pos[i]
            #             + quat_apply(
            #                 self.goal_rot[i],
            #                 to_torch([0, 0, 1], device=self.device) * 0.2,
            #             )
            #         )
            #         .cpu()
            #         .numpy()
            #     )

            #     p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]],
            #         [0.85, 0.1, 0.1],
            #     )
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]],
            #         [0.1, 0.85, 0.1],
            #     )
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]],
            #         [0.1, 0.1, 0.85],
            #     )

            #     objectx = (
            #         (
            #             self.object_pos[i]
            #             + quat_apply(
            #                 self.object_rot[i],
            #                 to_torch([1, 0, 0], device=self.device) * 0.2,
            #             )
            #         )
            #         .cpu()
            #         .numpy()
            #     )
            #     objecty = (
            #         (
            #             self.object_pos[i]
            #             + quat_apply(
            #                 self.object_rot[i],
            #                 to_torch([0, 1, 0], device=self.device) * 0.2,
            #             )
            #         )
            #         .cpu()
            #         .numpy()
            #     )
            #     objectz = (
            #         (
            #             self.object_pos[i]
            #             + quat_apply(
            #                 self.object_rot[i],
            #                 to_torch([0, 0, 1], device=self.device) * 0.2,
            #             )
            #         )
            #         .cpu()
            #         .numpy()
            #     )

            #     p0 = self.object_pos[i].cpu().numpy()
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]],
            #         [0.85, 0.1, 0.1],
            #     )
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]],
            #         [0.1, 0.85, 0.1],
            #     )
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]],
            #         [0.1, 0.1, 0.85],
            #     )

    def _check_success(self):
        task_dist = self.extras["task_dist"]
        return torch.where(
            torch.abs(task_dist) <= self.success_tolerance,
            torch.ones_like(self.reset_goal_buf),
            torch.zeros_like(self.reset_goal_buf),
        )


class AllegroHandGraspMultiTask(AllegroHandGrasp):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.task_reward_scale = self.cfg["env"]["taskRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.object_target_dof_name = self.cfg["env"]["objectDofName"]
        self.object_target_dof_pos = list(self.cfg["env"]["objectDofTargetPos"])
        self.object_target_dof_idx = 0
        if isinstance(self.object_target_dof_pos, list):
            self.object_target_dof_pos = to_torch(self.object_target_dof_pos, device=sim_device, dtype=torch.float)
        else:
            raise AssertionError("objectTargetDofPos must be a list")
        self.object_stiffness = self.cfg["env"].get("objectStiffness")
        self.object_damping = self.cfg["env"].get("objectDamping")

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.translation_scale = self.cfg["env"].get("translation_scale", 0.1)
        self.orientation_scale = self.cfg["env"].get("orientation_scale", 0.1)

        self.object_types = self.cfg["env"]["objectTypes"]
        assert all(
            [
                otype
                in [
                    "block",
                    "egg",
                    "pen",
                    "bottle",
                    "dispenser",
                    "spray_bottle",
                    "pill_bottle",
                ]
                for otype in self.object_types
            ]
        )

        self.fix_object_base = self.cfg["env"].get("fixObjectBase", True)
        self.object_mass = self.cfg["env"].get("objectMass", 10)
        self.ignore_z = [otype == "pen" for otype in self.object_types]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
        }
        for obj in SUPPORTED_PARTNET_OBJECTS:
            self.asset_files_dict[obj] = "urdf/objects/{}.urdf".format(obj)

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get(
                "assetFileNameBlock", self.asset_files_dict["block"]
            )
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get(
                "assetFileNameEgg", self.asset_files_dict["egg"]
            )
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get(
                "assetFileNamePen", self.asset_files_dict["pen"]
            )
            self.asset_files_dict["bottle"] = self.cfg["env"]["asset"].get(
                "assetFileNameBottle", self.asset_files_dict["bottle"]
            )
            self.asset_files_dict["spray_bottle"] = self.cfg["env"]["asset"].get(
                "assetFileNameSprayBottle", self.asset_files_dict["spray_bottle"]
            )
            self.asset_files_dict["dispenser"] = self.cfg["env"]["asset"].get(
                "assetFileNameDispenser", self.asset_files_dict["dispenser"]
            )
            self.asset_files_dict["pill_bottle"] = self.cfg["env"]["asset"].get(
                "assetFileNamePillBottle", self.asset_files_dict["pill_bottle"]
            )

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]"
            )

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {"full_no_vel": 64, "full": 87, "full_state": 89}

        self.up_axis = "z"
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 109  # TODO: distinguish returned state vs observation

        assert self.cfg["env"]["numActions"] in [17, 22]  # 22  # 16 hand dof + 1 ee position
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        super(AllegroHandGrasp, self).__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # setup viewer
        if self.viewer != None:
            # cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            # cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self._cam_pos = gymapi.Vec3(0.75, 0.75, 1.5)
            self._cam_target = gymapi.Vec3(0.75, -0.4, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, self._cam_pos, self._cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_dofs_with_object + self.num_object_dofs
            )[:, : self.num_shadow_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        if self.num_object_dofs > 0:
            self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
                :, self.num_shadow_hand_dofs : self.num_dofs_with_object
            ]
            self.goal_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
                :, self.num_dofs_with_object : self.num_dofs_with_object + self.num_object_dofs
            ]
            self.object_dof_pos = self.object_dof_state[..., 0]
            self.object_dof_vel = self.object_dof_state[..., 1]

        num_objects = len(self.object_types)
        if num_objects > 1:
            assert self.num_envs % num_objects == 0, "Number of objects should ebvenly divide number of envs!"
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs // num_objects, -1, 13)
            self.num_bodies = self.rigid_body_states.shape[1]
        else:
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
            self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.rb_torque = torch.zeros(
            (self.num_envs // len(self.object_types), self.num_bodies, 3), device=self.device, dtype=torch.float
        )
        self.rb_forces = torch.zeros(
            (self.num_envs // num_objects, self.num_bodies, 3), dtype=torch.float, device=self.device
        )

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1])
        )

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        allegro_hand_asset_file = "urdf/kuka_allegro_description/allegro_grasp.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            allegro_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", allegro_hand_asset_file)

        object_asset_files = [self.asset_files_dict[object_type] for object_type in self.object_types]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        print("Num hand dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs
        # self.gym.get_asset_actuator_count(shadow_hand_asset)

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        # self.shadow_hand_dof_default_pos = np.load("allegro_hand_dof_default_pos.npy")
        self.shadow_hand_dof_default_vel = []
        # self.sensors = []
        # sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props["lower"][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props["upper"][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            # print("Max effort: ", shadow_hand_dof_props["effort"][i])
            shadow_hand_dof_props["effort"][i] = 0.5
            shadow_hand_dof_props["stiffness"][i] = 3
            shadow_hand_dof_props["damping"][i] = 0.1
            shadow_hand_dof_props["friction"][i] = 0.01
            shadow_hand_dof_props["armature"][i] = 0.001

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = self.fix_object_base
        # DO NOT UNCOMMENT, for debugging purposes
        # object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        goal_object_asset_options = gymapi.AssetOptions()
        goal_object_asset_options.fix_base_link = True
        object_assets = []
        goal_assets = []
        object_dof_indices = []
        num_object_dofs = []

        for object_asset_file in object_asset_files:
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            object_assets.append(object_asset)
            goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, goal_object_asset_options)
            goal_assets.append(goal_asset)

        for i, (object_type, object_asset_file) in enumerate(zip(self.object_types, object_asset_files)):
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)

            # TODO: expand this list to include full set of operable object types
            if object_type in SUPPORTED_PARTNET_OBJECTS:
                num_object_dofs.append(self.gym.get_asset_dof_count(object_asset))
                object_dof_indices.append(self.gym.get_asset_dof_dict(object_asset)[self.object_target_dof_name[i]])
            else:
                num_object_dofs.append(0)
            object_assets.append(object_asset)

        # TODO: relax this assumption, but treat non-target joints as fixed
        assert all([num_object_dof == 1 for num_object_dof in num_object_dofs]), "Only one DOF objects supported!"
        self.num_object_dofs = 1

        if self.object_stiffness:
            object_dof_props["stiffness"][self.object_target_dof_idx] = self.object_stiffness
        if self.object_damping:
            object_dof_props["damping"][self.object_target_dof_idx] = self.object_damping

        self.num_dofs_with_object = self.num_shadow_hand_dofs + self.num_object_dofs

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.25, self.up_axis_idx))
        shadow_hand_start_pose.r = (
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.5 * np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.97 * np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.25 * np.pi)
        )

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = -0.12
        object_start_pose.p.y = -0.08
        object_start_pose.p.z = 0.124
        object_start_pose.r.w = 1.0
        object_start_pose.r.x = 0.0
        object_start_pose.r.y = 0.0
        object_start_pose.r.z = 0.0

        # object_start_pose.p.x = shadow_hand_start_pose.p.x
        # pose_dy, pose_dz = -0.2, 0.06

        # object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        # object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if "pen" in self.object_types:  #  == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.15)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2
        max_agg_shapes = self.num_shadow_hand_shapes + 2

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        # self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.palm_rb_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "palm_link")

        object_rb_counts = [self.gym.get_asset_rigid_body_count(object_asset) for object_asset in object_assets]
        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.shadow_hand_rb_handles = list(range(self.num_shadow_hand_bodies))
        self.object_rb_handles = [
            list(range(shadow_hand_rb_count, shadow_hand_rb_count + object_rb_count))
            for object_rb_count in object_rb_counts
        ]

        object_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(
                env_ptr, allegro_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0
            )
            self.hand_start_states.append(
                [
                    shadow_hand_start_pose.p.x,
                    shadow_hand_start_pose.p.y,
                    shadow_hand_start_pose.p.z,
                    shadow_hand_start_pose.r.x,
                    shadow_hand_start_pose.r.y,
                    shadow_hand_start_pose.r.z,
                    shadow_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.palm_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "palm_link")

            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.asymmetric_obs:
            #     for ft_handle in self.fingertip_handles:
            #         env_sensors = []
            #         env_sensors.append(self.gym.create_force_sensor(env_ptr, ft_handle, sensor_pose))
            #         self.sensors.append(env_sensors)

            #     self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # add object
            object_asset = object_assets[i % len(object_assets)]
            goal_asset = goal_assets[i % len(goal_assets)]
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 1)
            object_handles.append(object_handle)
            # if self.object_type == "spray_bottle":
            #     self.gym.set_actor_scale(env_ptr, object_handle, 0.5)
            if i <= len(object_assets) - 1:
                object_rb_props = [
                    self.gym.get_actor_rigid_body_properties(env_ptr, object_handle) for object_handle in object_handles
                ]
                for rb_props in object_rb_props:
                    rb_props[0].mass = self.object_mass
                self.object_rb_masses = [[prop.mass for prop in object_rb_props] for object_rb_props in object_rb_props]

                for object_rb_handle, rb_props in zip(object_handles, object_rb_props):
                    assert self.gym.set_actor_rigid_body_properties(env_ptr, object_rb_handle, rb_props, True)

            self.object_init_state.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(
                env_ptr,
                goal_asset,
                goal_start_pose,
                "goal_object",
                i + self.num_envs,
                0,
                1,  # disable self-collision
            )
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if "block" not in self.object_types:
                self.gym.set_rigid_body_color(
                    env_ptr,
                    object_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )
                self.gym.set_rigid_body_color(
                    env_ptr,
                    goal_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_actor_handles = object_handles

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] += 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        # self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.shadow_hand_rb_handles = to_torch(self.shadow_hand_rb_handles, dtype=torch.long, device=self.device)
        object_rb_handles = [to_torch(rb, dtype=torch.long, device=self.device) for rb in self.object_rb_handles]
        object_rb_masses = [
            to_torch(rb_masses, dtype=torch.float, device=self.device) for rb_masses in self.object_rb_masses
        ]
        # number of bodies vary per env depending on object type, this gives us what to add to each rb_handle to get global index
        if self.num_envs > 1:
            env_rb_counts = [
                self.gym.get_env_rigid_body_count(env_ptr) for env_ptr in self.envs[: len(self.object_types)]
            ]
            self.env_num_bodies = to_torch(
                [0] + np.cumsum(env_rb_counts)[:-1].tolist(), dtype=torch.long, device=self.device
            )
            self.object_rb_handles = torch.cat(
                [
                    object_rb_handle + num_bodies
                    for object_rb_handle, num_bodies in zip(object_rb_handles, self.env_num_bodies)
                ]
            )
            self.object_rb_handles_per_env = self.object_rb_handles
            self.object_rb_masses = torch.cat(object_rb_masses)
        else:
            self.object_rb_handles = self.object_rb_handles[0]
            self.object_rb_masses = self.object_rb_masses[0]
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids, goal_env_ids=None):
        # generate random values
        rand_floats = (
            torch_rand_float(
                -1.0,
                1.0,
                (len(env_ids), self.num_shadow_hand_dofs * 2),
                device=self.device,
            )
            * 0
        )

        # reset start object target poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids // len(self.object_types), :, :] = 0.0

        # reset object position/orientation
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()

        # reset object velocity
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        if goal_env_ids is not None:
            object_indices = [
                self.object_indices[env_ids],
                self.goal_object_indices[env_ids],
                self.goal_object_indices[goal_env_ids],
            ]
        else:
            object_indices = [
                self.object_indices[env_ids],
                self.goal_object_indices[env_ids],
            ]
        object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices),
        )

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, : self.num_shadow_hand_dofs]

        pos = self.shadow_hand_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = (
            self.shadow_hand_dof_default_vel
            + self.reset_dof_vel_noise * rand_floats[:, self.num_shadow_hand_dofs : self.num_shadow_hand_dofs * 2]
        )
        self.prev_targets[env_ids, : self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_indices),
            len(env_ids),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices),
            len(env_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        palm_index = [self.palm_index + b for b in self.env_num_bodies]

        self.hand_pose_vel = self.rigid_body_states[:, palm_index, 0:10].view(self.num_envs, -1)
        self.hand_pos = self.hand_pose_vel[:, 0:3]
        self.hand_rot = self.hand_pose_vel[:, 3:7]
        self.hand_linvel = self.hand_pose_vel[:, 7:10]
        self.fingertip_pose_vel = self.rigid_body_states[:, self.fingertip_handles, 0:10]
        self.fingertip_pos = self.fingertip_pose_vel[:, :, 0:3]
        self.fingertip_rot = self.fingertip_pose_vel[:, :, 3:7]
        self.fingertip_linvel = self.fingertip_pose_vel[:, :, 7:10]

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # if current time in seconds is divisible by 5, print object dof state
        # import time
        # current_time = int(time.time())
        # if current_time % 5 == 0:
        #     print("object dof pos: ", self.object_dof_pos)

        self.goal_pose = self.object_init_state[:, :7]
        self.goal_pos = self.goal_pose[:, 0:3]
        self.goal_dof_pos = (
            torch.ones_like(self.object_dof_state[:, self.object_target_dof_idx, 0])
            * self.object_target_dof_pos.repeat(self.num_envs // len(self.object_types), 1).flatten()
        )
        self.goal_rot = self.goal_states[:, 3:7]

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        else:
            print("Unkown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_hand_reward(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_dof_pos[:, self.object_target_dof_idx],
            self.goal_pos,
            self.goal_dof_pos,
            self.hand_pos,
            self.fingertip_pos,
            self.dist_reward_scale,
            self.task_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
            ("pen" in self.object_types),
        )

        self.extras["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["goal_dist"] = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        self.extras["task_dist"] = (
            (self.goal_dof_pos - self.object_dof_pos[:, self.object_target_dof_idx]).abs().flatten()
        )

        self.extras["hand_dist"] = torch.norm(self.hand_pos - self.object_pos, p=2, dim=-1)
        self.extras["fingertip_dist"] = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), p=2, dim=-1).sum(-1)
        self.extras["full_hand_dist"] = self.extras['hand_dist'] + self.extras['fingertip_dist']

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets)
                )


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    task_pos,
    target_pos,
    task_target_pos,
    hand_pos,
    fingertip_pos,
    dist_reward_scale: float,
    task_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
    ignore_z_rot: bool,
):
    # Distance from the goal object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    # dist_rew = torch.exp(-goal_dist / fall_dist) * dist_reward_scale / 2

    # Distance from the hand to the object
    hand_dist = torch.norm(hand_pos - object_pos, p=2, dim=-1)
    fingertip_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), p=2, dim=-1).sum(dim=-1)
    hand_dist += fingertip_dist
    # dist_rew += torch.exp(-(hand_dist - 0.05) / fall_dist) * dist_reward_scale / 2
    dist_rew = hand_dist * dist_reward_scale

    # Distance to task completion
    task_dist = torch.abs(task_target_pos - task_pos)
    # Scales task reward from [0, task_reward_scale]
    # task_rew = (1 - (task_dist / task_target_pos)) * task_reward_scale
    task_rew = torch.exp(-task_dist / task_target_pos) * task_reward_scale

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + task reward + action regularization + success bonus + fall penalty
    reward = dist_rew + task_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(
        torch.abs(task_dist) <= success_tolerance,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(
            torch.abs(task_dist) <= success_tolerance,
            torch.zeros_like(progress_buf),
            progress_buf,
        )
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(
        quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
        quat_from_angle_axis(rand0 * np.pi, z_unit_tensor),
    )
    return rot
