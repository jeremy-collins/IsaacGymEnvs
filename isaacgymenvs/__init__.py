import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict


def maybe_register(name, func):
    try:
        OmegaConf.register_new_resolver(name, func)
    except ValueError:
        print(f"{name} already registered")
    return


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)


def map_dict(items, keys):
    assert isinstance(keys, (list, ListConfig)), f"keys must be a list"
    assert isinstance(items, (list, dict, ListConfig, DictConfig)), f"items must be a list or dict"
    if isinstance(items, (dict, DictConfig)):
        return dict([(k, items[k]) for k in keys])
    else:
        return dict([(k, item) for k, item in zip(keys, items)])


maybe_register("dict", map_dict)


def custom_eval(x):
    import numpy as np  # noqa
    import torch  # noqa

    return eval(x)


maybe_register("eval", custom_eval)
maybe_register("device", lambda x, y: x.to(y))


def resolve_child(default, node, arg):
    """
    Attempts to get a child node parameter `arg` from `node`. If not
        present, the return `default`
    """
    if arg in node:
        return node[arg]
    else:
        return default


OmegaConf.register_new_resolver("resolve_child", resolve_child)


def make(
    seed: int,
    task: str,
    num_envs: int,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg: DictConfig = None,
):
    from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

    # create hydra config if no config passed in
    if cfg is None:
        # reset current hydra config if already parsed (but not passed in here)
        if HydraConfig.initialized():
            task = HydraConfig.get().runtime.choices["task"]
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        with initialize(config_path="./cfg"):
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            cfg_dict = omegaconf_to_dict(cfg.task)
            cfg_dict["env"]["numEnvs"] = num_envs
    # reuse existing config
    else:
        cfg_dict = omegaconf_to_dict(cfg.task)

    create_rlgpu_env = get_rlgames_env_creator(
        seed=seed,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )
    return create_rlgpu_env()
