# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
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

import hydra

from omegaconf import DictConfig, OmegaConf


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict["params"]["config"]
    if cfg.get("full_experiment_name"):
        train_cfg["full_experiment_name"] = cfg.full_experiment_name

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
            )
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    from isaacgymenvs.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    from isaacgymenvs.utils.rlgames_utils import (
        RLGPUEnv,
        RLGPUEnvAlgoObserver,
        RLGPUAlgoObserver,
        MultiObserver,
        ComplexObsRLGPUEnv,
    )
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    from isaacgymenvs.learning import actor_network_builder
    import isaacgymenvs

    time_now = datetime.now()
    time_str = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        if (
            cfg.checkpoint
            and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(cfg.checkpoint)), "config.yaml"))
            and cfg.load_config
        ):
            load_config = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(cfg.checkpoint)), "config.yaml"))
            cfg.task = load_config.task
            cfg.train = load_config.train

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )

        return envs

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
        },
    )

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = False
    if hasattr(ige_env_cls, "dict_obs_cls"):
        dict_cls = ige_env_cls.dict_obs_cls  # or cfg.task.env.get("useDictObs", False)

    if dict_cls:
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        if actor_net_cfg.name == "actor_critic_dict":
            input_keys = set(cfg.train.params.network.input_preprocessors.keys())
            if "obsDims" in cfg.task.env:
                env_keys = set(cfg.task.env.obsDims.keys())
                assert (
                    len(input_keys - env_keys) == 0
                ), f"Input keys must be a subset of env keys, missing {input_keys-env_keys}"
            else:
                assert not cfg.task.env.use_dict_obs
                env_keys = set(["obs"] + list(input_keys))

            obs_spec["obs"] = {
                "names": list(env_keys),
                "concat": False,
                "space_name": "observation_space",
            }

        else:
            obs_spec["obs"] = {
                "names": list(actor_net_cfg.inputs.keys()),
                "concat": not actor_net_cfg.name == "complex_net",
                "space_name": "observation_space",
            }
            if "central_value_config" in cfg.train.params.config:
                critic_net_cfg = cfg.train.params.config.central_value_config.network
                obs_spec["states"] = {
                    "names": list(critic_net_cfg.inputs.keys()),
                    "concat": not critic_net_cfg.name == "complex_net",
                    "space_name": "state_space",
                }

        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs),
        )
    else:
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
        )

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUEnvAlgoObserver()]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder("amp_continuous", lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder(
            "amp_continuous", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
        )
        model_builder.register_model(
            "continuous_amp",
            lambda network, **kwargs: amp_models.ModelAMPContinuous(network),
        )
        model_builder.register_network("amp", lambda **kwargs: amp_network_builder.AMPBuilder())
        model_builder.register_network("actor_critic_dict", lambda **kwargs: actor_network_builder.A2CBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join(
            "runs", cfg.train.params.config.name + "_{date:%d-%H-%M-%S}".format(date=time_now)
        )

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()
