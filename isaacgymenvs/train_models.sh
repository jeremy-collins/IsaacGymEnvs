# python train.py task=ArticulateTaskShaped train=AllegroHandGraspPPO experiment=ArticulateTaskShaped task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=750 -m
# python train.py task=ArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ArticulateTaskSpray1 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=750 -m
# python train.py task=ManipulabilityArticulateTaskShaped train=AllegroHandGraspPPO experiment=ManipulabilityArticulateTaskShaped task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=750 -m

# python train.py task=ManipulabilityVectorizedArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityVectorizedArticulateTaskSpray1 task.rewards.manipulability_reward_vectorized.scale=1 task.rewards.hand_dist.scale=-50. wandb_activate=false headless=true num_envs=16368 train.params.config.max_epochs=1000 -m
# python train.py task=ManipulabilityVectorizedArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityVectorizedArticulateTaskSpray1 task.rewards.manipulability_reward_vectorized.scale=1 task.rewards.hand_dist.scale=-50. wandb_activate=false headless=true num_envs=16368 train.params.config.max_epochs=1000 -m
# python train.py task=ManipulabilityVectorizedArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityVectorizedArticulateTaskSpray1 task.rewards.manipulability_reward_vectorized.scale=1 task.rewards.hand_dist.scale=-50. wandb_activate=false headless=true num_envs=16368 train.params.config.max_epochs=1000 -m
python train.py task=ManipulabilityVectorizedArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityVectorizedArticulateTaskSpray1_2-18 task.rewards.manipulability_reward_vectorized.scale=1 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=false num_envs=16368 train.params.config.max_epochs=1000 task.env.resetDofPosRandomInterval=0. -m

python train.py task=ManipulabilityVectorizedNegCostArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityVectorizedNegCostArticulateTaskSpray1_3-25_0 task.rewards.manipulability_neg_cost_vectorized.scale=1. task.rewards.hand_dist.scale=-50. wandb_activate=true headless=false num_envs=16368 train.params.config.max_epochs=1000 task.env.resetDofPosRandomInterval=0. -m

# python train.py \
#   task=ManipulabilityVectorizedArticulateTaskScissors \
#   train=AllegroHandGraspPPO \
#   experiment=ManipulabilityVectorizedArticulateTaskScissors \
#   task.rewards.manipulability_reward_vectorized.scale=1 \
#   task.rewards.hand_dist.scale=-50. \
#   wandb_activate=true \
#   headless=false \
#   num_envs=16368 \
#   train.params.config.max_epochs=1000 \
#   +task.env.hand_init_path=allegro_hand_dof_default_pos_scissors_closer.npy \
#   task.env.resetDofPosRandomInterval=0. \
#   task.env.useRelativeControl=true \
#   -m


# python train.py task=ArticulateTaskScissorsNew train=AllegroHandGraspPPONew experiment=ArticulateTaskScissorsNew task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16368 train.params.config.max_epochs=1000 -m
# python train.py task=ArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ArticulateTaskSpray1 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16368 train.params.config.max_epochs=1000 -m
# python train.py task=ManipulabilityRandVecArticulateTaskSpray1 train=AllegroHandGraspPPO experiment=ManipulabilityRandVecArticulateTaskSpray1 task.rewards.manipulability_reward_vec.scale=100 task.rewards.hand_dist.scale=-50. wandb_activate=true headless=True num_envs=16384 train.params.config.max_epochs=1000 -m