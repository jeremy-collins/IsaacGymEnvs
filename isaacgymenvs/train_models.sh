# python train.py task=ArticulateTaskShaped train=AllegroHandGraspPPO experiment=ArticulateTaskShaped task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=750 -m
# python train.py task=ArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ArticulateTaskSpray1 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=750 -m
# python train.py task=ManipulabilityArticulateTaskShaped train=AllegroHandGraspPPO experiment=ManipulabilityArticulateTaskShaped task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=750 -m
python train.py task=ManipulabilityVectorizedArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityVectorizedArticulateTaskSpray1 task.rewards.manipulability_reward_vectorized.scale=10 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16368 train.params.config.max_epochs=1000 -m
python train.py task=ArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ArticulateTaskSpray1 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=1000 -m
python train.py task=ManipulabilityRandVecArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityRandVecArticulateTaskSpray1 task.rewards.manipulability_reward_vec.scale=100 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=1000 -m