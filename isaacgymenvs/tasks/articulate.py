import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from dmanip.utils import rewards

from .base.vec_task import VecTask

SUPPORTED_PARTNET_OBJECTS = ["dispenser", "spray_bottle", "pill_bottle", "bottle", "spray_bottle2", "spray_bottle3"]


class ArticulateTask(VecTask):
    obs_keys = [
        "hand_joint_pos",  # in [16, 22]
        "hand_joint_vel",  # in [16, 22]
        "object_pos",  # 3
        "object_quat",  # 4
        "goal_pos",  #  3
        "goal_quat",  # 4
        "object_lin_vel",  # 3
        "object ang vel",  # 3
        "object_dof_pos",  # in [1, 2]
        "goal_dof_pos",  # in [1, 2]
        "hand_palm_pos",  # 3
        "hand_palm_quat",  # 4
        "hand_palm_vel",  # 3
        "actions",  # in [16, 22]
        # TOTAL: 98 (22 + 22 + 7 + 7 + 3 + 3 + 2 + 2 + 3 + 3 + 22)
        # TOTAL (+no vel): 89
        # TOTAL (+no object_quat, goal_quat): 90
        # TOTAL (1 DoF wrist): 79 (17 + 17 + 7 + 7 + 3 + 3 + 2 + 2 + 3 + 3 + 17)
        # TOTAL (+no vel): 70 (17 + 17 + 7 + 7 + 3 + 3 + 2 + 2 + 17)
        # "hand_joint_vel",  # in [16,20]
        # "object_joint_vel",  # in [1,2,3]
        # "object_body_vel",  # 3
        # "object_body_torque",  # 3
        # "object_body_f",  # 3
        # "object_joint_pos_err",  # in [1,2,3]
        # "object_pos_err",  # 3
    ]

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

        self.reward_params = rewards.parse_reward_params(cfg["env"]["rewardParams"])
        self.success_tolerance = cfg["env"].get(
            "success_tolerance", 0.005
        )  # TODO: rename to either reach-threshold or success-tolerance
        self.reward_extras = {"success_tolerance": self.success_tolerance}
        self.reach_bonus = self.reward_params.get(
            "reach_bonus", (rewards.reach_bonus, ("object_joint_pos_err", "success_tolerance"), 100.0)
        )
        self.hand_start_position = cfg["env"]["handStartPosition"]
        self.hand_start_orientation = cfg["env"]["handStartOrientation"]

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

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.translation_scale = self.cfg["env"].get("translation_scale", 0.1)
        self.orientation_scale = self.cfg["env"].get("orientation_scale", 0.1)
        self.load_default_pos = self.cfg["env"].get("load_default_pos", False)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in SUPPORTED_PARTNET_OBJECTS, f"object type {self.object_type} not supported"

        for obj in SUPPORTED_PARTNET_OBJECTS:
            if f"assetFileName{obj.capitalize()}" in self.cfg["env"]["asset"]:
                self.asset_files_dict[obj] = self.cfg["env"]["asset"][f"assetFileName{obj.capitalize()}"]
            else:
                self.asset_files_dict[obj] = f"urdf/objects/{obj}.urdf"

        self.obs_type = self.cfg["env"]["obsType"]
        self.num_obs_dict = {
            "full_state": 94 if "wrist" not in self.cfg["env"]["handDofType"] else 79,
            "full_state_no_vel": 85 if "wrist" not in self.cfg["env"]["handDofType"] else 70,
        }
        assert self.obs_type in self.num_obs_dict, f"obs type {self.obs_type} not supported"

        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = self.num_obs_dict["full_state"]
        self.cfg["env"]["numStates"] = num_states

        if self.obs_type == "full_state_no_vel":
            self.obs_keys = filter(lambda x: "vel" not in x, self.obs_keys)

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        self.init_sim()

    def init_sim(self):
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

        if self.object_type in ["dispenser", "spray_bottle", "pill_bottle", "bottle"]:
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

        object_asset_options.disable_gravity = False
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

        self.fingertips = list(
            filter(lambda x: x.endswith("link_3"), self.gym.get_asset_rigid_body_dict(allegro_hand_asset).keys())
        )
        self.fingertip_indices = [
            self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self.fingertips
        ]
        self.object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.shadow_hand_rb_handles = list(range(self.num_shadow_hand_bodies))
        self.object_rb_handles = list(
            range(
                self.num_shadow_hand_bodies,
                self.num_shadow_hand_bodies + self.object_rb_count,
            )
        )

        for i in range(num_envs):
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

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        palm_index = [self.palm_index + b for b in self.env_num_bodies]


from gym import Wrapper
from omegaconf import OmegaConf


class IsaacGymCameraWrapper(Wrapper):
    def __init__(self, env, camera_spec):
        cam_pos = camera_spec["camera_pose"]["position"]
        cam_quat = camera_spec["camera_pose"]["rotation"]
        self.znear = camera_spec["near_plane"]
        self.fov_x = camera_spec["horizontal_fov"]
        camera_config = {
            "name": "hand_camera",
            "is_body_camera": True,
            "actor_name": "hand",
            "image_size": [self.height, self.width],
            "image_type": "rgb",
            # 'image_type': 'depth',
            "horizontal_fov": float(self.fov_x),
            "camera_pose": [cam_pos, cam_quat.tolist()],
            "near_plane": self.znear,
            "use_collision_geometry": True,
        }
        camera_config = OmegaConf.create(camera_config)
        camera_spec_dict_tactile = {camera_config["name"]: camera_config}
        self.camera_spec_dict.update(camera_spec_dict_tactile)
        self.env = env
        super().__init__(env)

    def create_camera_actors(self):
        for i in range(self.num_envs):
            env_ptr = self.env_ptrs[i]
            env_camera_handles = self.setup_env_cameras(env_ptr, self.camera_spec_dict)
            self.camera_handles_list.append(env_camera_handles)

            env_camera_tensors = self.create_tensors_for_env_cameras(env_ptr, env_camera_handles, self.camera_spec_dict)
            self.camera_tensors_list.append(env_camera_tensors)

    def setup_env_cameras(self, env_ptr, camera_spec_dict):
        camera_handles = {}
        for name, camera_spec in camera_spec_dict.items():
            camera_properties = gymapi.CameraProperties()
            camera_properties.height = camera_spec.image_size[0]
            camera_properties.width = camera_spec.image_size[1]
            camera_properties.enable_tensors = True
            camera_properties.horizontal_fov = camera_spec.horizontal_fov
            if "near_plane" in camera_spec:
                camera_properties.near_plane = camera_spec.near_plane

            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
            camera_handles[name] = camera_handle

            if camera_spec.is_body_camera:
                actor_handle = self.gym.find_actor_handle(env_ptr, camera_spec.actor_name)
                robot_body_handle = self.gym.find_actor_rigid_body_handle(
                    env_ptr, actor_handle, camera_spec.attach_link_name
                )

                self.gym.attach_camera_to_body(
                    camera_handle,
                    env_ptr,
                    robot_body_handle,
                    gymapi.Transform(
                        gymapi.Vec3(*camera_spec.camera_pose[0]), gymapi.Quat(*camera_spec.camera_pose[1])
                    ),
                    gymapi.FOLLOW_TRANSFORM,
                )
            else:
                transform = gymapi.Transform(
                    gymapi.Vec3(*camera_spec.camera_pose[0]), gymapi.Quat(*camera_spec.camera_pose[1])
                )
                self.gym.set_camera_transform(camera_handle, env_ptr, transform)
        return camera_handles

    def create_tensors_for_env_cameras(self, env_ptr, env_camera_handles, camera_spec_dict):
        env_camera_tensors = {}
        for name in env_camera_handles:
            camera_handle = env_camera_handles[name]
            if camera_spec_dict[name].image_type == "rgb":
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
                )
            elif camera_spec_dict[name].image_type == "depth":
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH
                )
            else:
                raise NotImplementedError(f"Camera type {camera_spec_dict[name].image_type} not supported")

            # wrap camera tensor in a pytorch tensor
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)

            # store references to the tensor that gets updated when render_all_camera_sensors
            env_camera_tensors[name] = torch_camera_tensor
        return env_camera_tensors

    def compute_observations(self):
        cameras = self.get_camera_image_tensors_dict()
        observations = self.env.compute_observations()
        self.obs_dict["rgb"] = cameras["hand_camera"]
        return observations

    def get_camera_image_tensors_dict(self):
        if self.cfg_task.env.use_camera or self.cfg_task.env.use_isaac_gym_tactile:
            # transforms and information must be communicated from the physics simulation into the graphics system
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            camera_image_tensors_dict = dict()

            for name in self.camera_spec_dict:
                camera_spec = self.camera_spec_dict[name]
                if camera_spec["image_type"] == "rgb":
                    num_channels = 3
                    camera_images = torch.zeros(
                        (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1], num_channels),
                        device=self.device,
                        dtype=torch.uint8,
                    )
                    for id in np.arange(self.num_envs):
                        camera_images[id] = self.camera_tensors_list[id][name][:, :, :num_channels].clone()
                elif camera_spec["image_type"] == "depth":
                    num_channels = 1
                    camera_images = torch.zeros(
                        (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1]),
                        device=self.device,
                        dtype=torch.float,
                    )
                    for id in np.arange(self.num_envs):
                        # Note that isaac gym returns negative depth
                        # See: https://carbon-gym.gitlab-master-pages.nvidia.com/carbgym/graphics.html?highlight=image_depth#camera-image-types
                        camera_images[id] = self.camera_tensors_list[id][name][:, :].clone() * -1.0
                        camera_images[id][camera_images[id] == np.inf] = 0.0
                else:
                    print(f'Image type {camera_spec["image_type"]} not supported!')
                camera_image_tensors_dict[name] = camera_images

            if self.nominal_tactile:
                for k in self.nominal_tactile:
                    # import ipdb; ipdb.set_trace()
                    depth_image = self.nominal_tactile[k] - camera_image_tensors_dict[k]  # depth_image_delta
                    # camera_image_tensors_dict[k] = depth_image  # visualize diff
                    # taxim_render_all = self.taxim_gelsight.render_batch(depth_image)
                    taxim_render_all = self.taxim_gelsight.render_tensorized(depth_image)
                    # taxim_render = self.taxim_gelsight.render(depth_image.cpu().numpy()[0])
                    camera_image_tensors_dict[f"{k}_taxim"] = taxim_render_all

                    # Optionally subsample tactile image
                    ssr = self.cfg_task.env.tactile_subsample_ratio
                    camera_image_tensors_dict[k] = camera_image_tensors_dict[k][:, ::ssr, ::ssr]
                    camera_image_tensors_dict[f"{k}_taxim"] = camera_image_tensors_dict[f"{k}_taxim"][:, ::ssr, ::ssr]
            else:
                pass
            return camera_image_tensors_dict
