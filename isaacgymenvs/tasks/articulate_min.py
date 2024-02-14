from typing import Union, Dict, Any, Tuple
from isaacgym import gymtorch

import numpy as np
import os

from gym import spaces
from isaacgym import gymapi
from isaacgym.torch_utils import *
from omegaconf import DictConfig, ListConfig
from isaacgymenvs.utils import rewards
from isaacgymenvs.tasks.utils import IsaacGymCameraBase
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from omegaconf import OmegaConf
from .base.vec_task import VecTask

import time
import wandb

from isaacgymenvs.utils.manipulability import *
import copy
import torch

SUPPORTED_PARTNET_OBJECTS = [
    "dispenser",
    "spray_bottle",
    "pill_bottle",
    "bottle",
    "stapler",
    "scissors",
    "pliers",
]

NUM_OBJECT_TYPES = 7
NUM_OBJECT_INSTANCES = 5  # per type


class ArticulateMinTask(VecTask, IsaacGymCameraBase):
    dict_obs_cls: bool = False
    obs_keys = [
        "hand_joint_pos",  # in [16, 22]
        "hand_joint_vel",  # in [16, 22]
        # "hand_joint_pos_err",  # in [16, 22]
        "object_pos",  # 3
        "object_quat",  # 4
        "goal_pos",  #  3
        "goal_quat",  # 4
        "object_lin_vel",  # 3
        "object_ang_vel",  # 3
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
        # "hand_pos",  # 3
        # "hand_quat",  # 3
        # "hand_init_pos",  # 3
        # "hand_init_quat",  # 3
    ]
    state_keys = obs_keys.copy()

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
        torch.autograd.set_detect_anomaly(True)  # TODO: remove this
        self.cfg = cfg
        self.dict_obs_cls = self.cfg["env"].get("useDictObs", False)
        if self.dict_obs_cls:
            assert "obsDims" in self.cfg["env"], "obsDims must be specified for dict obs"
        self.reward_params = rewards.parse_reward_params(cfg["env"]["rewardParams"])
        assert "fall_penalty" in self.reward_params
        assert "reach_bonus" in self.reward_params
        self.reach_bonus = self.reward_params["reach_bonus"]
        self.success_tolerance = cfg["env"].get(
            "success_tolerance", 0.005
        )  # TODO: rename to either reach-threshold or success-tolerance
        self.reward_extras = {
            "success_tolerance": self.success_tolerance,
        }
        # self.hand_start_position = cfg["env"]["handStartPosition"]
        # self.hand_start_orientation = cfg["env"]["handStartOrientation"]

        self.object_target_dof_name = self.cfg["env"]["objectDofName"]
        self.object_target_dof_pos = self.cfg["env"]["objectDofTargetPos"]
        self.scale_dof_pos = self.cfg["env"].get("scaleDofPos", False)
        if not isinstance(self.object_target_dof_pos, (ListConfig, list)):
            self.object_target_dof_pos = [self.object_target_dof_pos]
        self.object_target_dof_pos = to_torch(self.object_target_dof_pos, device=sim_device, dtype=torch.float)
        # else:
        #     raise AssertionError("objectDofTargetPos must be a list")

        # object joint settings
        self.object_stiffness = self.cfg["env"].get("objectStiffness")
        self.object_damping = self.cfg["env"].get("objectDamping")
        self.object_friction = self.cfg["env"].get("objectFriction")
        self.object_armature = self.cfg["env"].get("objectArmature")
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations

        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise: Union[float, dict] = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise: Union[float, dict] = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise: Union[float, dict] = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise: Union[float, dict] = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale: Union[float, dict] = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.debug_rewards = self.cfg["env"].get("debugRewards")

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.reward_extras["max_consecutive_successes"] = self.max_consecutive_successes = self.cfg["env"][
            "maxConsecutiveSuccesses"
        ]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.translation_scale = self.cfg["env"].get("translation_scale", 0.1)
        self.orientation_scale = self.cfg["env"].get("orientation_scale", 0.1)
        self.load_default_pos = self.cfg["env"].get("load_default_pos", True)  # default dof pos for hand
        self.set_start_pos = self.cfg["env"].get("set_start_pos", False)
        self.debug_zero_actions = self.cfg["env"].get("debug_zero_actions", False)
        self.hand_init_path = self.cfg["env"].get("hand_init_path", "allegro_hand_dof_default_pos.npy")
        self.log_reward_info = self.cfg["env"].get("log_reward_info", False)
        self.load_goal_asset = self.cfg["env"].get("loadGoalAsset", False)
        self.limit_hand_rotation = self.cfg["env"].get("limitHandRotation", False)

        self.object_type = self.cfg["env"]["objectType"]
        self.object_instance = self.cfg["env"].get("objectInstance", {})
        self.fix_object_base = self.cfg["env"].get("fixObjectBase", False)
        self.object_mass = self.cfg["env"].get("objectMass", 1)  # in KG
        self.object_mass_base_only = self.cfg["env"].get("objectMassBaseOnly", False)

        if not isinstance(self.object_type, ListConfig):
            self.object_type = [self.object_type]
        assert all(
            [object_type in SUPPORTED_PARTNET_OBJECTS for object_type in self.object_type]
        ), f"object type {self.object_type} not supported"

        self.use_image_obs = self.cfg["env"].get("enableCameraSensors", False)

        # locate object asset file paths
        self.asset_files_dict = {}
        use_object_instances = False
        for obj in SUPPORTED_PARTNET_OBJECTS:
            obj_name = "".join([x.capitalize() for x in obj.split("_")])
            if f"assetFileName{obj_name}" in self.cfg["env"]["asset"]:
                obj_assets = self.cfg["env"]["asset"][f"assetFileName{obj_name}"]
                if obj in self.object_instance:
                    use_object_instances = True
                if isinstance(obj_assets, str):
                    obj_assets = [obj_assets]
                self.asset_files_dict[obj] = obj_assets
            else:
                self.asset_files_dict[obj] = [f"urdf/objects/{obj}/mobility.urdf"]
                asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
                for idx in range(2, 5):
                    if os.path.exists(f"{asset_root}/urdf/objects/{obj}{idx}/mobility.urdf"):
                        self.asset_files_dict[obj].append(f"urdf/objects/{obj}{idx}/mobility.urdf")
        num_objects = 0
        obj_types = 0
        max_obj_instances = 1

        # TODO: add one-hot encoding for object type and instance (combined)
        for object_type in self.object_type:
            obj_types += 1
            if object_type in self.object_instance:
                num_objects += len(self.object_instance[object_type])
            else:
                num_objects += len(self.asset_files_dict[object_type])

        max_obj_instances = max([len(self.asset_files_dict[object_type]) for object_type in self.asset_files_dict])

        use_object_types = obj_types > 1
        use_object_instances = use_object_instances or max_obj_instances > 1
        self.max_obj_instances = max_obj_instances
        self.num_objects = num_objects
        assert len(self.object_target_dof_pos) == 1 or len(self.object_target_dof_pos) == self.num_objects, (
            f"objectDofTargetPos must be a list of length 1 or {self.num_objects}"
            if self.num_objects > 1
            else "objectDofTargetPos must be a list of length 1"
        )
        self.obs_type = self.cfg["env"]["observationType"]
        assert self.cfg["env"]["numActions"] in [17, 22], "Only 17 and 22 DoF hands are supported"
        full_state_dim = 98 if self.cfg["env"]["numActions"] == 22 else 85
        no_vel_dim = 89 if self.cfg["env"]["numActions"] == 22 else 76
        self.use_one_hot = self.cfg["env"].get("useOneHot", False)

        # TODO: handle object instances
        if use_object_types:
            added_dims = 1
            if self.use_one_hot:
                added_dims = NUM_OBJECT_TYPES  # len(SUPPORTED_PARTNET_OBJECTS)
            full_state_dim, no_vel_dim = (
                full_state_dim + added_dims,
                no_vel_dim + added_dims,
            )
            type_key = "object_type_one_hot" if self.use_one_hot else "object_type"
            self.obs_keys += [type_key]
        if use_object_instances:
            added_dims = 1
            if self.use_one_hot:
                added_dims = self.cfg["env"].get("numObjectInstances", NUM_OBJECT_INSTANCES)
            full_state_dim, no_vel_dim = (
                full_state_dim + added_dims,
                no_vel_dim + added_dims,
            )
            instance_key = "object_instance_one_hot" if self.use_one_hot else "object_instance"
            self.obs_keys += [instance_key]

        self.num_obs_dict = {
            "full_state": full_state_dim,
            "full_state_no_vel": no_vel_dim,
        }
        assert self.obs_type in self.num_obs_dict, f"obs type {self.obs_type} not supported"

        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = self.num_obs_dict["full_state"]

        # obs_buf is empty if using dict obs
        num_obs = self.num_obs_dict[self.obs_type] if not self.dict_obs_cls else 0

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = num_states

        if not self.dict_obs_cls:
            if self.obs_type == "full_state_no_vel":
                self.obs_keys = filter(lambda x: "vel" not in x, self.obs_keys)
            if self.obs_type == "full":
                self.obs_keys.remove("hand_palm_quat")
        else:
            self.obs_keys = self.cfg["env"]["obsKeys"]

        # use VecTask init function
        VecTask.__init__(
            self,
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        if self.use_relative_control:
            self.actions = torch.zeros(self.num_envs, self.cfg["env"]["numActions"], device=self.device)
        else:
            self.actions = scale(
                self.shadow_hand_dof_default_pos,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
        self.current_obs_dict = {}

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

        if self.cfg["env"].get("saveRigidBodyState", False):
            assert self.device == "cpu", "saveRigidBodyState only works with CPU tensors!"
            initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))
            np.save("initial_state.npy", initial_state)

        if self.cfg["env"].get("loadRigidBodyState", False):
            assert self.device == "cpu", "saveRigidBodyState only works with CPU tensors!"
            initial_state = np.load("initial_state.npy")
            self.gym.set_sim_rigid_body_states(self.sim, initial_state, gymapi.STATE_ALL)
            # print("set_sim_rigid_body_states in init_sim with result: ", result)
            self.gym.simulate(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_dofs_with_object + self.num_object_dofs * self.load_goal_asset
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
            self.goal_dof_state = None
            if self.load_goal_asset:
                self.goal_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
                    :,
                    self.num_dofs_with_object : self.num_dofs_with_object + self.num_object_dofs,
                ]
            self.object_dof_pos = self.object_dof_state[..., 0]
            self.object_dof_vel = self.object_dof_state[..., 1]

        if self.num_objects > 1:
            assert self.num_envs % self.num_objects == 0, "Number of objects should ebvenly divide number of envs!"
            # need to do this since number of object bodies varies per object instance
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
                self.num_envs // self.num_objects, -1, 13
            )
            self.num_bodies = self.rigid_body_states.shape[1]
        else:
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
            self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.hand_init_pos = self.root_state_tensor[self.hand_indices + 1][:, :3].clone()
        self.hand_init_quat = self.root_state_tensor[self.hand_indices + 1][:, 3:7].clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # if self.set_start_pos:
        self.prev_targets[:, self.actuated_dof_indices] = self.shadow_hand_dof_default_pos
        self.cur_targets[:, self.actuated_dof_indices] = self.shadow_hand_dof_default_pos

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.reward_extras["successes"] = self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.reward_extras["consecutive_successes"] = self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.rb_torque = torch.zeros(
            (self.num_envs // self.num_objects, self.num_bodies, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.rb_forces = torch.zeros(
            (self.num_envs // self.num_objects, self.num_bodies, 3),
            dtype=torch.float,
            device=self.device,
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

        self.prev_bufs = None

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

        # get asset files for hand and objects
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        allegro_hand_asset_file = "urdf/kuka_allegro_description/allegro_grasp.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            allegro_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", allegro_hand_asset_file)

        # get object assets according to object type and instance
        object_asset_files = []
        self.env_instance_order = []
        self.env_task_order = []
        for object_id, object_type in enumerate(self.object_type):
            object_id = SUPPORTED_PARTNET_OBJECTS.index(object_type)
            object_asset_file = self.asset_files_dict[object_type]
            object_instance_ids = range(len(object_asset_file))
            if isinstance(object_asset_file, (ListConfig, list)):
                if self.object_instance.get(object_type, []):
                    object_asset_file = [object_asset_file[i] for i in self.object_instance[object_type]]
                    object_instance_ids = self.object_instance[object_type]
                object_asset_files += list(object_asset_file)
                self.env_instance_order += list(object_instance_ids)
                self.env_task_order += [object_id] * len(object_instance_ids)
            else:
                object_asset_files.append(object_asset_file)
                self.env_task_order.append(object_id)

        # load shadow hand asset
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
        allegro_hand_default_pos = np.zeros(self.num_shadow_hand_dofs)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        print("self.shadow_hand_dof_default_pos initial: ", self.shadow_hand_dof_default_pos)
        self.shadow_hand_dof_default_vel = []

        # TODO: add option to load different hand initializations per object
        if os.path.exists(self.hand_init_path) and self.load_default_pos:
            allegro_hand_dof_default_pos = np.load(self.hand_init_path)
            # allegro_hand_default_pos[:6] = 0.
            # allegro_hand_default_pos[2] = -0.3
            assert len(allegro_hand_dof_default_pos) == self.num_shadow_hand_dofs
        else:
            allegro_hand_dof_default_pos = shadow_hand_dof_props["lower"] * 0.65 + shadow_hand_dof_props["upper"] * 0.35
            allegro_hand_dof_default_pos[:6] = 0.0
            # allegro_hand_default_pos[2] = -0.3

        # self.sensors = []
        # sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            # self.shadow_hand_dof_default_pos.append(allegro_hand_default_pos[i])
            if self.limit_hand_rotation and 2 < i < 6:
                self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props["lower"][i] * 0.01)
                self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props["upper"][i] * 0.01)
            else:
                self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props["lower"][i])
                self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props["upper"][i])
            self.shadow_hand_dof_default_vel.append(0.0)

            # print("Max effort: ", shadow_hand_dof_props["effort"][i])
            shadow_hand_dof_props["effort"][i] = 0.5
            if i < 6 and self.num_actions == 22:
                shadow_hand_dof_props["stiffness"][i] = 30
                shadow_hand_dof_props["damping"][i] = 10
                shadow_hand_dof_props["friction"][i] = 0.0
                shadow_hand_dof_props["armature"][i] = 0.0
            else:
                shadow_hand_dof_props["stiffness"][i] = 3
                shadow_hand_dof_props["damping"][i] = 0.1
            shadow_hand_dof_props["friction"][i] = 0.01
            shadow_hand_dof_props["armature"][i] = 0.001

        # assert len(self.shadow_hand_dof_default_pos) == self.num_shadow_hand_dofs

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        # self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        # add dof limits to rewards to compute joint penalty
        self.reward_extras["dof_limit_low"] = self.shadow_hand_dof_lower_limits.view(1, -1)
        self.reward_extras["dof_limit_high"] = self.shadow_hand_dof_upper_limits.view(1, -1)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load articulated object and goal assets
        self.object_assets = []
        self.object_dof_props = []
        self.goal_assets = []
        self.start_poses = []

        def load_object_goal_asset(object_asset_file, object_id):
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.fix_base_link = self.fix_object_base
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            object_target_dof_idx = self.gym.get_asset_dof_dict(object_asset)[self.object_target_dof_name]

            start_pose = None
            if "start_pose.npz" in os.listdir(os.path.join(asset_root, os.path.dirname(object_asset_file))):
                start_pose = np.load(os.path.join(asset_root, os.path.dirname(object_asset_file), "start_pose.npz"))

            # if object_id < SUPPORTED_PARTNET_OBJECTS.index(
            #     "scissors"
            # ):  # spring-loaded objects, use PD controller to restore position
            object_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)

            if self.object_stiffness:
                object_dof_props["stiffness"][object_target_dof_idx] = self.object_stiffness
            if self.object_damping:
                object_dof_props["damping"][object_target_dof_idx] = self.object_damping
            if self.object_armature:
                object_dof_props["armature"][object_target_dof_idx] = self.object_armature
            if self.object_friction:
                object_dof_props["friction"][object_target_dof_idx] = self.object_friction

            object_asset_options.disable_gravity = True
            object_asset_options.fix_base_link = True
            if self.load_goal_asset:
                goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            else:
                goal_asset = None
            return object_asset, goal_asset, object_dof_props, start_pose

        # Load object assets from from files, as well as object-specific pose data
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []
        self.object_dof_props = []
        for object_file, object_id in zip(object_asset_files, self.env_task_order):
            object_asset, goal_asset, object_dof_props, object_start_pose = load_object_goal_asset(
                object_file, object_id
            )
            self.object_dof_lower_limits.append(object_dof_props["lower"])
            self.object_dof_upper_limits.append(object_dof_props["upper"])
            self.object_assets.append(object_asset)
            self.object_dof_props.append(object_dof_props)
            self.goal_assets.append(goal_asset)
            self.start_poses.append(object_start_pose)

        if any([otype in SUPPORTED_PARTNET_OBJECTS for otype in self.object_type]):
            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
        self.num_object_bodies = list(map(lambda x: self.gym.get_asset_rigid_body_count(x), self.object_assets))
        self.num_object_shapes = [
            self.gym.get_asset_rigid_shape_count(object_asset) for object_asset in self.object_assets
        ]
        if self.load_goal_asset:
            self.num_goal_bodies = list(map(lambda x: self.gym.get_asset_rigid_body_count(x), self.goal_assets))
            self.num_goal_shapes = list(map(lambda x: self.gym.get_asset_rigid_shape_count(x), self.goal_assets))
        else:
            self.num_goal_bodies = [0] * len(self.goal_assets)
            self.num_goal_shapes = [0] * len(self.goal_assets)

        self.object_target_dof_idx = []
        for task_id, object_asset in zip(self.env_task_order, self.object_assets):
            object_type = SUPPORTED_PARTNET_OBJECTS[task_id]
            # TODO: remove redundant step here
            if object_type in SUPPORTED_PARTNET_OBJECTS:
                num_object_dofs = self.gym.get_asset_dof_count(object_asset)
                assert num_object_dofs == 1, "Currently only support 1 DoF objects"
                self.num_object_dofs = 1
                self.object_target_dof_idx = [0]
                # object_target_dof_idx = self.gym.get_asset_dof_dict(object_asset)[self.object_target_dof_name]
                # self.object_target_dof_idx.append(object_target_dof_idx)
            else:
                self.num_object_dofs = 0

        self.num_dofs_with_object = self.num_shadow_hand_dofs + self.num_object_dofs

        # get object and goal poses
        poses = []

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.15)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )

        for object_type, object_asset_file, start_pose in zip(
            self.env_task_order, object_asset_files, self.start_poses
        ):
            object_start_pose = gymapi.Transform()

            allegro_hand_dof_start_pos = to_torch(allegro_hand_dof_default_pos, device=self.device)
            if start_pose:
                object_start_pose.p = gymapi.Vec3(*start_pose["pos"])
                # TODO: handle multiple hand_start_pose transforms per object
                if "hand_start_pos" in start_pose:
                    # allegro_hand_start_pose = gymapi.Transform()
                    #     allegro_hand_start_pose.p = gymapi.Vec3(*start_pose["hand_start_pos"])
                    #     allegro_hand_start_pose.r = gymapi.Vec3(*start_pose["hand_start_rot"])
                    # if "hand_start_qpos" in start_pose:
                    translate = to_torch(start_pose["hand_start_pos"], device=self.device)
                    rot = get_euler_xyz(to_torch(start_pose["hand_start_rot"], device=self.device))
                    allegro_hand_dof_start_pos = torch.cat([translate, rot, allegro_hand_dof_start_pos[6:]])
            else:
                object_start_pose.p.x = -0.12
                object_start_pose.p.y = -0.08  # should roughly be half height of object
                object_start_pose.p.z = 0.124

            object_start_pose.r.w = 1.0
            object_start_pose.r.x = 0.0
            object_start_pose.r.y = 0.0
            object_start_pose.r.z = 0.0

            goal_start_pose = gymapi.Transform()
            goal_start_pose.p = object_start_pose.p + self.goal_displacement

            goal_start_pose.p.z -= 0.04

            init_pose_offset = gymapi.Vec3(*get_axis_params(0.25, self.up_axis_idx))
            init_rotation_offset = (
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.5 * np.pi)
                * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.97 * np.pi)
                * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.25 * np.pi)
            )
            allegro_hand_frame_origin = init_transform = gymapi.Transform()
            init_transform.p = init_pose_offset
            init_transform.r = init_rotation_offset

            if self.set_start_pos:
                allegro_hand_frame_origin = gymapi.Transform()
                allegro_hand_frame_origin.p = gymapi.Vec3(*(allegro_hand_dof_start_pos[:3].cpu().numpy().tolist()))
                allegro_hand_frame_origin.r = gymapi.Quat.from_euler_zyx(
                    allegro_hand_dof_start_pos[5], allegro_hand_dof_start_pos[4], allegro_hand_dof_start_pos[3]
                )  # * init_rotation_offset
                allegro_hand_dof_start_pos[:6] = 0.0
                allegro_hand_frame_origin = init_transform * allegro_hand_frame_origin
            else:
                allegro_hand_frame_origin = init_transform

            # allegro_hand_dof_start_pos[:6] = 0.0
            print("self.shadow_hand_dof_default_pos before append", self.shadow_hand_dof_default_pos)
            print("appending allegro_hand_dof_start_pos", allegro_hand_dof_start_pos)
            self.shadow_hand_dof_default_pos.append(allegro_hand_dof_start_pos)
            poses.append((object_start_pose, goal_start_pose, allegro_hand_frame_origin))

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + max(self.num_object_bodies)
        max_agg_shapes = self.num_shadow_hand_shapes + max(self.num_object_shapes)
        if self.load_goal_asset:
            max_agg_bodies += max(self.num_object_bodies)
            max_agg_shapes += max(self.num_object_shapes)
        
        # # compute aggregate size (from Krishnan)
        # max_agg_bodies = self.num_shadow_hand_bodies + max(self.num_object_bodies) + max(self.num_goal_bodies)
        # max_agg_shapes = self.num_shadow_hand_shapes + max(self.num_object_shapes) + max(self.num_goal_shapes)
        # print("shadow_hand_dof_default_pos", self.shadow_hand_dof_default_pos) 
        # print("shadow_hand_dof_default_pos len", len(self.shadow_hand_dof_default_pos)) 
        # self.shadow_hand_dof_default_pos = torch.stack(tuple(self.shadow_hand_dof_default_pos), dim=0).repeat(
        #     (self.num_envs // len(self.object_assets), 1)
        # )

        print("self.shadow_hand_dof_default_pos", self.shadow_hand_dof_default_pos)
        self.shadow_hand_dof_default_pos = torch.stack(tuple(self.shadow_hand_dof_default_pos), dim=0).repeat(
            (self.num_envs // len(self.object_assets), 1)
        )
        # self.shadow_hand_dof_default_pos[:, :6] = 0s
        self.shadow_hands = []
        self.object_actor_handles = []
        self.object_rb_masses = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        self.fingertips = list(
            filter(
                lambda x: x.endswith("link_3"),
                self.gym.get_asset_rigid_body_dict(allegro_hand_asset).keys(),
            )
        )
        self.fingertip_indices = [
            self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self.fingertips
        ]
        bodies_per_env = np.concatenate([[0], np.cumsum(self.num_object_bodies) + np.cumsum(self.num_goal_bodies)])
        self.shadow_hand_rb_handles = [
            self.num_shadow_hand_bodies * i + np.arange(self.num_shadow_hand_bodies) + bodies_per_env[i]
            for i in range(len(self.num_object_bodies))
        ]
        self.object_rb_handles = [
            list(
                range(
                    self.num_shadow_hand_bodies * (i + 1) + sum(self.num_object_bodies[:i]),
                    self.num_shadow_hand_bodies * (i + 1) + sum(self.num_object_bodies[: i + 1]),
                )
            )
            for i in range(len(self.num_object_bodies))
        ]
        for i in range(num_envs):
            object_asset, object_dof_props, goal_asset = (
                self.object_assets[i % len(self.object_assets)],
                self.object_dof_props[i % len(self.object_dof_props)],
                self.goal_assets[i % len(self.goal_assets)],
            )
            object_start_pose, goal_start_pose, allegro_hand_frame_origin = poses[i % len(poses)]
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(
                env_ptr, allegro_hand_asset, allegro_hand_frame_origin, "hand", i, -1, 0
            )
            self.hand_start_states.append(
                [
                    allegro_hand_frame_origin.p.x,
                    allegro_hand_frame_origin.p.y,
                    allegro_hand_frame_origin.p.z,
                    allegro_hand_frame_origin.r.x,
                    allegro_hand_frame_origin.r.y,
                    allegro_hand_frame_origin.r.z,
                    allegro_hand_frame_origin.r.w,
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
            self.x_link_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "translation_x")
            self.y_link_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "translation_y")
            self.z_link_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "translation_z")

            self.xrot_link_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "rotation_x")
            self.yrot_link_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "rotation_y")
            self.zrot_link_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, "allegro_mount")

            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.bs:
            #     for ft_handle in self.fingertip_handles:
            #         env_sensors = []
            #         env_sensors.append(self.gym.create_force_sensor(env_ptr, ft_handle, sensor_pose))
            #         self.sensors.append(env_sensors)

            #     self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 1)
            self.gym.set_actor_dof_properties(env_ptr, object_handle, object_dof_props)
            self.object_actor_handles.append(object_handle)
            rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            if self.object_mass_base_only:
                rb_props[0].mass = self.object_mass
            else:
                total_mass = sum([prop.mass for prop in rb_props])
                for prop in rb_props:
                    prop.mass *= self.object_mass / total_mass
            if i < len(self.object_assets):
                self.object_rb_masses.append([prop.mass for prop in rb_props])
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
            if goal_asset:
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
                if goal_asset:
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

        # create cameras
        if self.use_image_obs:
            self.camera_spec_dict = dict()
            self.get_default_camera_specs()
            if self.camera_spec_dict:
                # tactile cameras created along with other cameras in create_camera_actors
                self.create_camera_actors()

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] += 0.04  # lifted up 4 centimeters from initial pose
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.shadow_hand_rb_handles = to_torch(self.shadow_hand_rb_handles, dtype=torch.long, device=self.device)

        # number of bodies vary per env depending on object type, this gives us what to add to each rb_handle to get global index
        if self.num_objects > 1:
            env_rb_counts = [self.gym.get_env_rigid_body_count(env_ptr) for env_ptr in self.envs[: self.num_objects]]
            self.env_num_bodies = to_torch(
                [0] + np.cumsum(env_rb_counts)[:-1].tolist(),
                dtype=torch.long,
                device=self.device,
            )
            self.object_rb_handles = torch.cat(
                [
                    to_torch(object_rb_handle, device=self.device, dtype=torch.long)
                    for object_rb_handle, num_bodies in zip(self.object_rb_handles, self.env_num_bodies)
                ]
            )
            self.fingertip_indices = to_torch(self.fingertip_indices, device=self.device, dtype=torch.long)
            self.fingertip_indices = (self.env_num_bodies[:, None] + self.fingertip_indices[None, :]).flatten()
            # concatenate and then turn the concatenated list of object_rb_masses into a torch tensor
            self.object_rb_masses = to_torch(np.concatenate(self.object_rb_masses), device=self.device)
        else:
            self.env_num_bodies = self.gym.get_env_rigid_body_count(self.envs[0])
            self.object_rb_handles = self.object_rb_handles[0]
            self.object_rb_masses = self.object_rb_masses[0]

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.fingertip_indices = to_torch(self.fingertip_indices, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def reset_target_pose(self, env_ids, apply_reset=False):
        # print("resetting goals in reset_target_pose at", env_ids)
        if isinstance(self.object_target_dof_pos, torch.Tensor) and len(self.object_target_dof_pos) > 1:
            object_target_dof = self.object_target_dof_pos.repeat(self.num_envs // self.num_objects).unsqueeze(-1)[
                env_ids
            ]
        else:
            object_target_dof = self.object_target_dof_pos  # TODO: resample goal dof state here

        # sets goal object position and rotation and dof pos
        if self.load_goal_asset:
            self.goal_states[env_ids, 0:7] = self.goal_init_state[env_ids, 0:7]
            self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = (
                self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
            )
            self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
            self.goal_dof_state[env_ids, :, 0] = object_target_dof

            goal_indices = self.goal_object_indices[env_ids].to(torch.int32)
            # reset goal task pose
            self.prev_targets[
                env_ids,
                self.num_dofs_with_object : self.num_dofs_with_object + self.num_object_dofs,
            ] = object_target_dof
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(goal_indices),
                len(env_ids),
            )
            # print("set_dof_position_target_tensor_indexed in reset_target_pose with result: ", result)

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(goal_indices),
                len(env_ids),
            )
            # print("set_dof_state_tensor_indexed in reset_target_pose with result: ", result)

            # zeroes velocities
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
                self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
            )

            if apply_reset and self.load_goal_asset:
                # print("applying reset in reset_target_pose at", env_ids)
                goal_object_indices = self.goal_object_indices[env_ids].to(torch.long)
                if self.prev_bufs is None:
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.root_state_tensor),
                        gymtorch.unwrap_tensor(goal_object_indices),
                        len(env_ids),
                    )
                    # print("set_actor_root_state_tensor_indexed in reset_target_pose with result: ", result)
                else:
                    self.prev_bufs["prev_actor_root_state_tensor"][goal_object_indices] = self.root_state_tensor[
                        goal_object_indices
                    ]

        self.reset_goal_buf[env_ids] = 0

    def get_or_sample_noise_scale(self, param: Union[float, dict]):
        if isinstance(param, float):
            noise_scale = param
        elif isinstance(param, dict):
            self.last_step = self.gym.get_frame_count(self.sim)
            sched_type = param["schedule"]
            if sched_type == "linear":
                sched_steps = param["schedule_steps"]
                sched_scaling = 1 / sched_steps * min(self.last_step, sched_steps)
            elif sched_type == "constant":
                sched_scaling = float(self.last_step > param.get("schedule_steps", 0))
            low, high = param["range"]
            noise_scale = low + (high - low) * sched_scaling
        else:
            raise ValueError(f"Invalid reset_position_noise: {param} must be float or dict")
        return noise_scale

    def reset_idx(self, env_ids, goal_env_ids=None):
        # generate random values
        rand_floats = torch_rand_float(
            -1.0,
            1.0,
            (len(env_ids), self.num_shadow_hand_dofs * 2 + 13),
            device=self.device,
        )  # * 0

        # reset start object target poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        if self.num_objects > 1:
            rb_env_ids = torch.unique(to_torch(env_ids // self.num_objects, device=self.device, dtype=torch.long))
        else:
            rb_env_ids = env_ids
        self.rb_forces[rb_env_ids, :, :] = 0.0

        # reset object position/orientation
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        pose_noise_scale = self.get_or_sample_noise_scale(self.reset_position_noise)

        # TODO: disentangle object pose noise and dof_pose noise
        object_pose_noise = pose_noise_scale * rand_floats[:, :13]
        object_pose_noise[:, 1] = 0.0  # no noise in y-dim
        # z-dim noise must be negative
        object_pose_noise[:, 2] = torch.clamp(object_pose_noise[:, 2], 0, torch.inf)
        object_pose_noise[:, 3:] = 0.0
        self.root_state_tensor[self.object_indices[env_ids]] += object_pose_noise
        rand_floats = rand_floats[:, 13:] * 0

        # reset object velocity
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        object_indices = [self.object_indices[env_ids]]
        if self.load_goal_asset:
            object_indices += [self.goal_object_indices[env_ids]]
            if goal_env_ids is not None:
                object_indices += [
                    self.goal_object_indices[goal_env_ids],  # TODO: remove these if goal asset not added
                ]
        object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        if self.prev_bufs is None:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(object_indices),
                len(object_indices),
            )
            # print("set_actor_root_state_tensor_indexed in reset_idx with result: ", result)
        else:
            self.prev_bufs["prev_actor_root_state_tensor"][object_indices] = self.root_state_tensor[
                object_indices
            ]

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        # reset shadow hand
        # delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        # delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        delta_min = (self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos)[env_ids]
        delta_max = (self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos)[env_ids]

        print("env_ids", env_ids)
        print("len(env_ids)", len(env_ids))
        print("self.shadow_hand_dof_default_pos[env_ids].shape", self.shadow_hand_dof_default_pos[env_ids].shape)
        print("rand_floats[:, : self.num_shadow_hand_dofs].shape", rand_floats[:, : self.num_shadow_hand_dofs].shape)
        print("delta_min.shape", delta_min.shape)
        print("delta_max.shape", delta_max.shape)
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, : self.num_shadow_hand_dofs]
        # print("rand_delta.shape", rand_delta.shape)

        pos_noise_scale = self.get_or_sample_noise_scale(self.reset_dof_pos_noise)
        vel_noise_scale = self.get_or_sample_noise_scale(self.reset_dof_vel_noise)

        pos = self.shadow_hand_dof_default_pos[env_ids] + pos_noise_scale * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = (
            self.shadow_hand_dof_default_vel
            + vel_noise_scale * rand_floats[:, self.num_shadow_hand_dofs : self.num_shadow_hand_dofs * 2]
        )
        self.prev_targets[env_ids, : self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_shadow_hand_dofs] = pos

        hand_indices = (self.hand_indices[env_ids] // 2).to(torch.int32)  # TODO: remove this hack

        if self.prev_bufs is None:
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(hand_indices),
                len(env_ids),
            )
            # print("set_dof_state_tensor_indexed in reset_idx")

            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(hand_indices),
                len(env_ids),
            )
            # print("set_dof_position_target_tensor_indexed in reset_idx")

        else:
            print("hand_indices", hand_indices)
            # print("self.prev_bufs[prev_dof_state_tensor].shape", self.prev_bufs["prev_dof_state_tensor"].shape)
            # print("self.prev_bufs[prev_targets].shape", self.prev_bufs["prev_targets"].shape)
            # print("self.dof_state.shape", self.dof_state.shape)
            # print("self.prev_targets.shape", self.prev_targets.shape)
            self.prev_bufs["prev_dof_state_tensor"][hand_indices] = self.dof_state[hand_indices]
            self.prev_bufs["prev_targets"][hand_indices] = self.prev_targets[hand_indices]

        # print("env_ids", env_ids)
        # print("progress_buf before zeroing", self.progress_buf)
        self.progress_buf[env_ids] = 0
        # print("progress_buf after zeroing", self.progress_buf)
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def reset(self):
        if (self.reset_buf == 1).all() and (self.progress_buf == 0).all():
            self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        if self.use_image_obs:
            IsaacGymCameraBase.compute_observations(self)
        ret = VecTask.reset(self)
        return ret

    def pre_physics_step(self, actions):
        self.extras = {}
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # self.reset_idx([1,2,3]) # TODO: for debugging only, remove this line!!

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            # print("resetting envs at", env_ids, "with goals at", goal_env_ids)
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        self.assign_act(self.actions)

        if self.prev_bufs is not None:
            # setting state tensors to the values in self.prev_bufs
            self.gym.set_actor_root_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_bufs["prev_actor_root_state_tensor"]),
            )

            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_bufs["prev_dof_state_tensor"]),
            )

            self.gym.set_dof_position_target_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_bufs["prev_targets"]),
            )

        # get rb forces
        force_noise_scale = self.get_or_sample_noise_scale(self.force_scale)
        if force_noise_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn_like(
                    self.rb_forces[force_indices, self.object_rb_handles, :],
                    device=self.device,
                )
                * self.object_rb_masses.unsqueeze(-1).unsqueeze(0)
                * force_noise_scale
            )

            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                None,
                gymapi.LOCAL_SPACE,
            )

    def assign_act(self, actions):
        if self.debug_zero_actions:
            if self.use_relative_control:
                actions *= 0
            else:
                actions = unscale(
                    self.cur_targets[:, self.actuated_dof_indices],
                    self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                    self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
                )
                self.actions = actions
        if self.use_relative_control:
            targets = (
                self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * actions
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                targets,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
        else:
            # self.move_hand_pos()
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                actions,
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

        if self.prev_bufs is None:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
            # print("set_dof_position_target_tensor in assign_act with result: ", result)
        else:
            self.prev_bufs["prev_targets"] = self.cur_targets

    def post_physics_step(self):
        # print("self.progress_buf before +=", self.progress_buf)
        self.progress_buf += 1
        # print("self.progress_buf after +=", self.progress_buf)

        self.compute_observations()
        self.compute_reward()
        # print('reward', self.current_rew_dict)
        self.extras["consecutive_successes"] = self.reward_extras["consecutive_successes"].mean()
        self.extras["goal_dist"] = torch.norm(
            self.current_obs_dict["object_pos"] - self.current_obs_dict["goal_pos"],
            p=2,
            dim=-1,
        )
        self.extras["hand_dist"] = torch.norm(
            self.current_obs_dict["hand_palm_pos"] - self.current_obs_dict["object_pos"],
            p=2,
            dim=-1,
        )
        self.extras["fingertip_dist"] = torch.norm(
            self.current_obs_dict["fingertip_pos"] - self.current_obs_dict["object_pos"].unsqueeze(1),
            p=2,
            dim=-1,
        ).sum(-1)
        self.extras["full_hand_dist"] = self.extras["hand_dist"] + self.extras["fingertip_dist"]

        self.extras["success"] = self._check_success().flatten()

        if self.debug_rewards and (self.gym.get_frame_count(self.sim) % ((self.max_episode_length) // 10) == 0):
            total_rew = self.rew_buf.clone()
            for key, rew in self.current_rew_dict.items():
                norm_rew = rew / total_rew
                print(
                    f"{key} reward (min/max), [mean+/-std]: ({norm_rew.min()}, {norm_rew.max()}), [{norm_rew.mean()} +/- {norm_rew.std()}]"
                )

        if self.print_success_stat and self.reset_buf.sum() > 0:
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

        if self.viewer and self.debug_viz:
            self.debug_visualization()

    def compute_reward(self):
        self.current_rew_dict = self.get_reward_dict(
            self.reward_params, self.current_obs_dict, self.actions, self.reward_extras
        )
        if self.log_reward_info:
            for k in self.current_rew_dict:
                self.extras[f"r_{k}"] = self.current_rew_dict[k]

        for key, value in self.current_rew_dict.items():
            if value.view(-1).shape[0] != self.num_envs:
                assert False, f"Reward dict key '{key}' has incorrect shape: {value.view(-1).shape}"
        reward = torch.cat([v.view(-1, 1) for v in self.current_rew_dict.values()], dim=-1).sum(dim=-1)
        self.extras["task_dist"] = (
            (
                self.current_obs_dict["goal_dof_pos"]
                - self.current_obs_dict["object_dof_pos"][:, self.object_target_dof_idx]
            )
            .abs()
            .flatten()
        )
        self.reset_goal_buf = new_successes = torch.where(
            self.current_rew_dict["reach_bonus"] > 0,
            torch.ones_like(self.reset_goal_buf),
            self.reset_goal_buf,
        )
        new_successes = new_successes.float()
        self.successes += new_successes
        resets = torch.where(
            self.current_rew_dict["fall_penalty"] < 0,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        resets = resets | torch.where(
            self.progress_buf >= self.max_episode_length,
            torch.ones_like(self.reset_buf),
            resets,
        )
        if self.max_consecutive_successes > 0:
            resets = torch.where(
                self.successes >= self.max_consecutive_successes,
                torch.ones_like(resets),
                resets,
            )
            reward = torch.where(
                self.progress_buf >= self.max_episode_length,
                reward + 0.5 * self.current_rew_dict["fall_penalty"],
                reward,
            )

        num_resets = resets.sum()
        self.reset_buf[:] = resets

        finished_cons_successes = torch.sum(self.successes * resets.float(), dim=-1)
        self.consecutive_successes = torch.where(
            num_resets > 0,
            self.av_factor * finished_cons_successes / num_resets + (1.0 - self.av_factor) * self.consecutive_successes,
            self.consecutive_successes,
        )
        self.rew_buf[:] = reward

    @staticmethod
    def get_reward_dict(reward_params, obs_dict, actions, reward_extras):
        if "actions" not in obs_dict:
            reward_extras["actions"] = actions
        rew_dict = {}
        for k, (cost_fn, rew_terms, rew_scale) in reward_params.items():
            rew_args = []
            for arg in rew_terms:
                if isinstance(arg, (tuple, list, dict)):
                    rew_args += [arg[k] for k in arg]
                elif isinstance(arg, str) and arg in reward_extras:
                    rew_args.append(reward_extras[arg])
                elif isinstance(arg, str) and arg in obs_dict:
                    rew_args.append(obs_dict[arg])
                elif isinstance(arg, (float, np.ndarray)):
                    rew_args.append(arg)
                else:
                    raise TypeError("Invalid argument for reward function {}, ('{}')".format(k, arg))
            v = cost_fn(*rew_args) * rew_scale
            rew_dict[k] = v.view(-1)
        return rew_dict

    def _check_success(self):
        obs_dict = self.current_obs_dict
        task_dist = self.extras.get(
            "task_dist",
            torch.abs(obs_dict["object_dof_pos"] - obs_dict["goal_dof_pos"]),
        )
        return torch.where(
            torch.abs(task_dist) <= self.success_tolerance,
            torch.ones_like(self.reset_goal_buf),
            torch.zeros_like(self.reset_goal_buf),
        )

    def debug_visualization(self):
        # draw axes on target object
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # vec3 color red
        color = gymapi.Vec3(0.85, 0.1, 0.1)
        for env in map(lambda x: self.envs[x], self.get_hand_in_contact().nonzero()):
            self.gym.draw_env_rigid_contacts(self.viewer, env, color, 1.0, True)

        for i in range(self.num_envs):
            targetx = (
                (
                    self.current_obs_dict["hand_palm_pos"][i]
                    + quat_apply(
                        self.current_obs_dict["hand_palm_quat"][i],
                        to_torch([1, 0, 0], device=self.device) * 0.2,
                    )
                )
                .cpu()
                .numpy()
            )
            targety = (
                (
                    self.current_obs_dict["hand_palm_pos"][i]
                    + quat_apply(
                        self.current_obs_dict["hand_palm_quat"][i],
                        to_torch([0, 1, 0], device=self.device) * 0.2,
                    )
                )
                .cpu()
                .numpy()
            )
            targetz = (
                (
                    self.current_obs_dict["hand_palm_pos"][i]
                    + quat_apply(
                        self.current_obs_dict["hand_palm_quat"][i],
                        to_torch([0, 0, 1], device=self.device) * 0.2,
                    )
                )
                .cpu()
                .numpy()
            )

            p0 = (
                self.current_obs_dict["hand_palm_pos"][i].cpu().numpy()
            )  # + self.goal_displacement_tensor.cpu().numpy()
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

    def compute_observations(self, skip_manipulability=False):
        # print("refreshing (compute_observations in post_physics_step)")
        prev_hand_dof_pos = self.shadow_hand_dof_pos.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        #     self.gym.refresh_force_sensor_tensor(self.sim)
        #     self.gym.refresh_dof_force_tensor(self.sim)

        if self.num_objects > 1:
            palm_index = [self.palm_index + b for b in self.env_num_bodies]
        else:
            palm_index = self.palm_index
        obs_dict = {} if not self.use_dict_obs else self.obs_dict
        obs_dict["hand_dof_pos"] = self.shadow_hand_dof_pos
        obs_dict["hand_joint_pos"] = unscale(
            self.shadow_hand_dof_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        obs_dict["hand_joint_pos_init"] = self.shadow_hand_dof_default_pos
        if self.use_relative_control:
            obs_dict["hand_joint_pos_delta"] = self.actions
        else:
            obs_dict["hand_joint_pos_delta"] = self.actions - unscale(
                prev_hand_dof_pos,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )

        obs_dict["hand_joint_pos_target"] = unscale(
            self.cur_targets[:, self.actuated_dof_indices],
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )

        obs_dict["hand_joint_vel"] = self.vel_obs_scale * self.shadow_hand_dof_vel
        obs_dict["object_pose"] = self.root_state_tensor[self.object_indices, 0:7]
        obs_dict["object_pos"] = self.root_state_tensor[self.object_indices, 0:3]
        obs_dict["object_quat"] = self.root_state_tensor[self.object_indices, 3:7]
        obs_dict["goal_pose"] = self.goal_states[:, 0:7]
        obs_dict["goal_pos"] = self.goal_states[:, 0:3]
        obs_dict["goal_quat"] = self.goal_states[:, 3:7]
        obs_dict["object_lin_vel"] = self.root_state_tensor[self.object_indices, 7:10]
        obs_dict["object_ang_vel"] = self.vel_obs_scale * self.root_state_tensor[self.object_indices, 10:13]

        obs_dict["object_dof_pos"] = self.object_dof_pos.view(self.num_envs, -1)
        obs_dict["object_dof_pos_scaled"] = unscale(
            obs_dict["object_dof_pos"].view(self.num_envs // self.num_objects, self.num_objects, -1),
            self.object_dof_lower_limits,
            self.object_dof_upper_limits,
        ).view(self.num_envs, -1)

        if isinstance(self.object_target_dof_pos, torch.Tensor):
            object_target_dof = (
                self.object_target_dof_pos.unsqueeze(0)
                .repeat(self.num_envs // len(self.object_target_dof_pos), 1)
                .unsqueeze(-1)
                .to(self.device)
            )
        else:
            object_target_dof = self.object_target_dof_pos * torch.ones_like(
                obs_dict["object_dof_pos"], device=self.device
            )

        obs_dict["goal_dof_pos"] = object_target_dof.view(self.num_envs, -1)
        obs_dict["goal_dof_pos_scaled"] = unscale(
            obs_dict["goal_dof_pos"].view(self.num_envs // self.num_objects, self.num_objects, -1),
            self.object_dof_lower_limits,
            self.object_dof_upper_limits,
        ).view(self.num_envs, -1)

        obs_dict["hand_init_pos"] = self.hand_init_pos
        obs_dict["hand_init_quat"] = self.hand_init_quat
        obs_dict["hand_pos"] = self.root_state_tensor[self.hand_indices + 1, 0:3]
        obs_dict["hand_quat"] = self.root_state_tensor[self.hand_indices + 1, 3:7]
        # open and append hand_pos and hand_quat to a npz file for debugging
        # if os.path.exists("hand_pos_quat.npz"):
        #     d = np.load("hand_pos_quat.npz")
        #     hand_pos = np.concatenate([d['hand_pos'], obs_dict["hand_pos"].cpu().numpy()])
        #     hand_quat = np.concatenate([d['hand_quat'], obs_dict["hand_quat"].cpu().numpy()])
        # else:
        #     hand_pos = obs_dict["hand_pos"].cpu().numpy()[:1]
        #     hand_quat = obs_dict["hand_quat"].cpu().numpy()[:1]
        # np.savez("hand_pos_quat.npz", hand_pos=hand_pos, hand_quat=hand_quat)

        obs_dict["hand_palm_pos"] = self.rigid_body_states[:, palm_index, 0:3].view(self.num_envs, -1)
        obs_dict["hand_palm_quat"] = self.rigid_body_states[:, palm_index, 3:7].view(self.num_envs, -1)
        obs_dict["hand_palm_vel"] = self.vel_obs_scale * self.rigid_body_states[:, palm_index, 7:10].view(
            self.num_envs, -1
        )
        obs_dict["fingertip_pose_vel"] = self.rigid_body_states[:, self.fingertip_indices][:, :, 0:10].view(
            self.num_envs, -1, 10
        )  # n_envs x 4 x 10
        obs_dict["fingertip_pos"] = obs_dict["fingertip_pose_vel"][:, :, 0:3]
        obs_dict["fingertip_rot"] = obs_dict["fingertip_pose_vel"][:, :, 3:7]
        obs_dict["fingertip_vel"] = obs_dict["fingertip_pose_vel"][:, :, 7:10]

        # get thumb, index, middle, ring pos/rot
        obs_dict["thumb_pos"] = obs_dict["fingertip_pos"][:, 0]
        obs_dict["thumb_rot"] = obs_dict["fingertip_rot"][:, 0]
        obs_dict["index_pos"] = obs_dict["fingertip_pos"][:, 1]
        obs_dict["index_rot"] = obs_dict["fingertip_rot"][:, 1]
        obs_dict["middle_pos"] = obs_dict["fingertip_pos"][:, 2]
        obs_dict["middle_rot"] = obs_dict["fingertip_rot"][:, 2]
        obs_dict["ring_pos"] = obs_dict["fingertip_pos"][:, 3]
        obs_dict["ring_rot"] = obs_dict["fingertip_rot"][:, 3]

        obs_dict["actions"] = self.actions
        obs_dict["hand_joint_pos_err"] = self.prev_targets[:, self.actuated_dof_indices] - obs_dict["hand_joint_pos"]

        obs_dict["object_type"] = (
            to_torch(
                np.concatenate(
                    [
                        [
                            SUPPORTED_PARTNET_OBJECTS.index(otype)
                            for _ in self.object_instance.get(otype, self.asset_files_dict[otype])
                        ]
                        for otype in self.object_type
                    ]
                ),
                device=self.device,
            )
            .repeat(self.num_envs // self.num_objects)
            .unsqueeze(-1)
        )

        obs_dict["object_instance"] = (
            to_torch(self.env_instance_order, device=self.device)
            .repeat(self.num_envs // self.num_objects)
            .unsqueeze(-1)
        )

        object_instance_one_hot = torch.nn.functional.one_hot(
            obs_dict["object_instance"].to(torch.int64),
            num_classes=self.cfg["env"].get("numObjectInstances", NUM_OBJECT_INSTANCES),
        ).squeeze(-2)
        object_type_one_hot = torch.nn.functional.one_hot(
            obs_dict["object_type"].to(torch.int64),
            num_classes=NUM_OBJECT_TYPES,
        ).squeeze(-2)
        obs_dict["object_instance_one_hot"] = object_instance_one_hot.to(self.device)
        obs_dict["object_type_one_hot"] = object_type_one_hot.to(self.device)

        # placeholder replacing manipulability calculation with just the buffers would be returned
        self.prev_bufs = {
        "prev_actor_root_state_tensor": self.root_state_tensor.clone(),
        "prev_dof_state_tensor": self.dof_state.clone(),
        "prev_rigid_body_tensor": self.rigid_body_states.clone(),
        "prev_targets": self.prev_targets.clone(),
        }

        self.current_obs_dict = obs_dict
        if not self.use_dict_obs:
            # for key in self.obs_keys:
            #     if key in obs_dict:
            #         self.obs_dict[key][:] = obs_dict[key]
            obs_tensor = obs_dict_to_tensor(obs_dict, self.obs_keys, self.num_envs, self.device)
            # check obs shape
            assert (
                obs_tensor.shape[-1] == self.num_obs_dict[self.obs_type]
            ), f"Obs shape {obs_tensor.shape} not correct!"
            self.obs_buf[:] = obs_tensor
        if self.use_image_obs:
            IsaacGymCameraBase.compute_observations(self)

    def get_default_camera_specs(self):
        camera_specs = self.cfg["env"].get("camera_spec", {"hand_camera": dict(width=64, height=64)})
        self.camera_spec_dict = {}
        for k in camera_specs:
            camera_spec = camera_specs[k]
            camera_config = {
                "name": k,
                "is_body_camera": camera_spec.get("is_body_camera", False),
                "actor_name": camera_spec.get("actor_name", "hand"),
                "attach_link_name": camera_spec.get("attach_link_name", "palm_link"),
                "use_collision_geometry": True,
                "width": camera_spec.get("width", 64),
                "height": camera_spec.get("height", 64),
                "image_size": [camera_spec.get("width", 64), camera_spec.get("height", 64)],
                "image_type": "rgb",
                "horizontal_fov": 90.0,
                # "position": [-0.1, 0.15, 0.15],
                # "rotation": [1, 0, 0, 0],
                "near_plane": 0.1,
                "far_plane": 100,
                "camera_pose": camera_spec.get("camera_pose", [[0.0, -0.35, 0.2], [0.0, 0.0, 0.85090352, 0.52532199]]),
            }
            self.camera_spec_dict[k] = OmegaConf.create(camera_config)
        self.current_obs_dict = obs_dict
        if not self.use_dict_obs:
            # for key in self.obs_keys:
            #     if key in obs_dict:
            #         self.obs_dict[key][:] = obs_dict[key]
            obs_tensor = obs_dict_to_tensor(obs_dict, self.obs_keys, self.num_envs, self.device)
            # check obs shape
            assert (
                obs_tensor.shape[-1] == self.num_obs_dict[self.obs_type]
            ), f"Obs shape {obs_tensor.shape} not correct!"
            self.obs_buf[:] = obs_tensor
        if self.use_image_obs:
            IsaacGymCameraBase.compute_observations(self)

    def get_default_camera_specs(self):
        camera_specs = self.cfg["env"].get("camera_spec", {"hand_camera": dict(width=64, height=64)})
        self.camera_spec_dict = {}
        for k in camera_specs:
            camera_spec = camera_specs[k]
            camera_config = {
                "name": k,
                "is_body_camera": camera_spec.get("is_body_camera", False),
                "actor_name": camera_spec.get("actor_name", "hand"),
                "attach_link_name": camera_spec.get("attach_link_name", "palm_link"),
                "use_collision_geometry": True,
                "width": camera_spec.get("width", 64),
                "height": camera_spec.get("height", 64),
                "image_size": [camera_spec.get("width", 64), camera_spec.get("height", 64)],
                "image_type": "rgb",
                "horizontal_fov": 90.0,
                # "position": [-0.1, 0.15, 0.15],
                # "rotation": [1, 0, 0, 0],
                "near_plane": 0.1,
                "far_plane": 100,
                "camera_pose": camera_spec.get("camera_pose", [[0.0, -0.35, 0.2], [0.0, 0.0, 0.85090352, 0.52532199]]),
            }
            self.camera_spec_dict[k] = OmegaConf.create(camera_config)

class ArticulateMinTaskCamera(ArticulateMinTask):
    dict_obs_cls: bool = False

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
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        assert self.use_image_obs, "ArticulateTaskCamera requires use_image_obs to be True"
        if not isinstance(self.obs_space, spaces.Dict):
            tmp_obs_space = self.obs_space
            space_dict = {"obs": tmp_obs_space}
            for camera_name, camera_spec in self.camera_spec_dict.items():
                space_dict[camera_name] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        camera_spec.width,
                        camera_spec.height,
                        3,
                    ),
                    dtype=np.uint8,
                )
            self.obs_space = spaces.Dict(space_dict)

        # IsaacGymCameraBase.__init__(self, self.camera_spec_dict)

    def create_camera_actors(self):
        """Overrides IsaacGymCameraBase.create_camera_actors
        to create multiple camera handles if needed"""
        import copy
        from scipy.spatial.transform import Rotation

        if hasattr(self, "camera_handles_list"):
            return
        if not self.cfg["env"].get("vary_camera_pose", False):
            return super().create_camera_actors()
        self.camera_handles_list = []
        self.camera_tensors_list = []
        if self.camera_spec_dict:
            camera_name = list(self.camera_spec_dict.keys())[0]
            init_camera_spec = self.cfg["env"]["camera_spec"][camera_name]
        else:
            camera_name = "hand_camera"
            init_camera_spec = self.cfg["env"]["cameraSpec"]["fixed_camera"]
        pos, rot = init_camera_spec.get("camera_pose", [[0.0, -0.35, 0.2], [0.0, 0.0, 0.85090352, 0.52532199]])
        R = Rotation.from_quat(rot)
        self.camera_spec_dicts = []
        # create a line of cameras from initial position looking in same direction
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            camera_spec = copy.deepcopy(self.camera_spec_dict)
            cameras_per_axis = self.num_envs // 3
            axis = "xyz"[int(i / cameras_per_axis)]
            fov = 60
            c = fov / cameras_per_axis
            # rotate camera by 5 degrees around pitch axis
            new_rot = R * Rotation.from_euler(axis, -fov / 2 + (i % cameras_per_axis) * c, degrees=True)
            camera_spec[camera_name]["camera_pose"] = [pos, [float(x) for x in new_rot.as_quat()]]
            env_camera_handles = self.setup_env_cameras(env_ptr, camera_spec)
            self.camera_spec_dicts.append(camera_spec)
            self.camera_handles_list.append(env_camera_handles)
            self.camera_tensors_list.append(
                self.create_tensors_for_env_cameras(env_ptr, env_camera_handles, camera_spec)
            )