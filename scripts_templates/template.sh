task="kitchen-partial-v0" # change task with desired ones
pipeline="kitchen" # set corresponding environment
model="veteran"

planner_net="transformer" # ["transformer", "unet"]
guidance_type="MCSS" # ["MCSS", "cfg", "cg"]
pipeline_type="separate" # ["separate", "joint"]

# Inference settings here
stride=4 
planner_d_model=256
planner_depth=2
# ... more configs

python pipelines/${model}_d4rl_${pipeline}.py \
    mode=inference \
    task.stride=$stride \
    task.planner_d_model=$planner_d_model

python pipelines/${model}_d4rl_${pipeline}.py \
    mode=inference \
    task.stride=$stride \
    task.planner_d_model=$planner_d_model
