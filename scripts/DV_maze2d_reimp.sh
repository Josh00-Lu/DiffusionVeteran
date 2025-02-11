tasks="maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1"
model="veteran"
pipeline="maze2d"

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

echo "start inference"
experiment="inference"
python pipelines/${model}_d4rl_${pipeline}.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="inference" \
    task=$task \
    enable_wandb=0
done
