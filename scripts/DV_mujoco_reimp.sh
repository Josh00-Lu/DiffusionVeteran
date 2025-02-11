tasks="halfcheetah-medium-expert-v2 halfcheetah-medium-replay-v2 halfcheetah-medium-v2 hopper-medium-expert-v2 hopper-medium-replay-v2 hopper-medium-v2 walker2d-medium-expert-v2 walker2d-medium-replay-v2 walker2d-medium-v2"
model="veteran"
pipeline="mujoco"

for task in $tasks; do
tags="default"
echo "start training"
experiment="train"
python pipelines/${model}_d4rl_${pipeline}.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="train" \
    task=$task \
    enable_wandb=0

experiment="train_expected_value"
python pipelines/${model}_d4rl_${pipeline}.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="train_expected_value" \
    task=$task \
    enable_wandb=0

echo "start inference"
experiment="inference"
python pipelines/${model}_d4rl_${pipeline}.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="inference" \
    task=$task \
    enable_wandb=0
done