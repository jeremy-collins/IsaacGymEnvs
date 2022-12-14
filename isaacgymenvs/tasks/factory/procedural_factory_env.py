
import hydra
import omegaconf
import os
import torch

from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_nut_bolt_screw import FactoryEnvNutBoltScrew
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.utils import torch_jit_utils


class FactoryTaskNutBoltScrewFriction(FactoryEnvNutBoltScrew, FactoryABCTask):


    def set_nut_bolt_friction(self, nut_bolt_friction):
        for env_ptr, nut_handle, bolt_handle in zip(self.env_ptrs, self.nut_handles, self.bolt_handles):
            nut_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, nut_handle)
            nut_shape_props[0].friction = nut_bolt_friction
            nut_shape_props[0].rolling_friction = 0.0  # default = 0.0
            nut_shape_props[0].torsion_friction = 0.0  # default = 0.0
            nut_shape_props[0].restitution = 0.0  # default = 0.0
            nut_shape_props[0].compliance = 0.0  # default = 0.0
            nut_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, nut_handle, nut_shape_props)

            bolt_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, bolt_handle)
            bolt_shape_props[0].friction = nut_bolt_friction
            bolt_shape_props[0].rolling_friction = 0.0  # default = 0.0
            bolt_shape_props[0].torsion_friction = 0.0  # default = 0.0
            bolt_shape_props[0].restitution = 0.0  # default = 0.0
            bolt_shape_props[0].compliance = 0.0  # default = 0.0
            bolt_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, bolt_handle, bolt_shape_props)

