from gym import Wrapper
from omegaconf import OmegaConf
from isaacgym import gymtorch
from isaacgym import gymapi

import numpy as np
import os
import torch


class IsaacGymCameraBase:
    """
    Base class for integrating with IsaacGymCamera

    Must call create_camera_actors() before using
    """

    height = 256
    width = 256

    # def __init__(self, camera_spec, cfg_task=None):
    #     self.cfg_task = cfg_task
    #     cam_pos = camera_spec["camera_pose"]["position"]
    #     cam_quat = camera_spec["camera_pose"]["rotation"]
    #     self.znear = camera_spec["near_plane"]
    #     self.fov_x = camera_spec["horizontal_fov"]
    #     camera_config = {
    #         "name": "hand_camera",
    #         "is_body_camera": False,  # set to True to have camera move with hand
    #         "actor_name": "hand",
    #         "image_size": [self.height, self.width],
    #         "image_type": "rgb",
    #         # 'image_type': 'depth',
    #         "horizontal_fov": float(self.fov_x),
    #         "camera_pose": [cam_pos, cam_quat.tolist()],
    #         "near_plane": self.znear,
    #         "use_collision_geometry": True,
    #         "attach_link_name": "palm_link",
    #     }
    #     camera_config = OmegaConf.create(camera_config)
    #     self.camera_spec_dict = {camera_config["name"]: camera_config}
    #     if self.camera_spec_dict:
    #         # tactile cameras created along with other cameras in create_camera_actors
    #         self.create_camera_actors()
    #
    # super().__init__(env)

    def create_camera_actors(self):
        self.camera_handles_list = []
        self.camera_tensors_list = []
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
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
        self.obs_dict["rgb"] = cameras["hand_camera"]

    def get_camera_image_tensors_dict(self):
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
        return camera_image_tensors_dict

        # if self.nominal_tactile:
        #     for k in self.nominal_tactile:
        #         # import ipdb; ipdb.set_trace()
        #         depth_image = self.nominal_tactile[k] - camera_image_tensors_dict[k]  # depth_image_delta
        #         # camera_image_tensors_dict[k] = depth_image  # visualize diff
        #         # taxim_render_all = self.taxim_gelsight.render_batch(depth_image)
        #         taxim_render_all = self.taxim_gelsight.render_tensorized(depth_image)
        #         # taxim_render = self.taxim_gelsight.render(depth_image.cpu().numpy()[0])
        #         camera_image_tensors_dict[f"{k}_taxim"] = taxim_render_all

        #         # Optionally subsample tactile image
        #         ssr = self.cfg_task.env.tactile_subsample_ratio
        #         camera_image_tensors_dict[k] = camera_image_tensors_dict[k][:, ::ssr, ::ssr]
        #         camera_image_tensors_dict[f"{k}_taxim"] = camera_image_tensors_dict[f"{k}_taxim"][:, ::ssr, ::ssr]
        # else:
        #     pass
