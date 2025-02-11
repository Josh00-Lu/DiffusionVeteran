tasks="kitchen-partial-v0 kitchen-mixed-v0"
model="veteran"
pipeline="kitchen"

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

