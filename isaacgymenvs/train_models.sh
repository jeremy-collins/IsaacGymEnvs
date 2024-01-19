
python train.py task=ArticulateTaskShaped train=AllegroHandGraspPPO experiment=NoManip task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=2048 -m
python train.py task=ManipulabilityArticulateTaskShaped train=AllegroHandGraspPPO experiment=WithManip task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=2048 -m