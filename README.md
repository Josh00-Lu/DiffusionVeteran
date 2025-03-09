# What Makes a Good Diffusion Planner for Decision Making?

<p align="center">
·
<a href="https://openreview.net/pdf?id=7BQkXXM8Fy">Paper</a>
·
<a href="#">Code</a>
·
<a href="https://openreview.net/forum?id=7BQkXXM8Fy">OpenReview</a>
</p>

This repository contains the PyTorch implementation of *"What Makes a Good Diffusion Planner for Decision Making?"* (ICLR 2025, Spotlight)

<p align="center">
    <br>
    <img src="figures/framework.png" width="85%"/>
    <br>
<p>

## 🛠️ Setup
Let's start with python 3.9. It's recommend to create a `conda` env:

### Create a new conda environment 
```shell
conda create -n dv python=3.9 mesalib glew glfw pip=23 setuptools=63.2.0 wheel=0.38.4 protobuf=3.20 -c conda-forge -y
conda activate dv
```

### Install for MuJoCo Simulator and mujoco-py (Important)
Install mujoco following the instruction [here](https://github.com/openai/mujoco-py#install-mujoco).

Alternatively, run the following script for a quick setup:
```bash
#!/bin/bash
sudo apt-get update && sudo apt-get install -y wget tar libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf cmake
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
echo $USER_DIR
wget -c "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
mkdir -p /home/$USER_DIR/.mujoco
cp mujoco210-linux-x86_64.tar.gz /home/$USER_DIR/mujoco.tar.gz
rm mujoco210-linux-x86_64.tar.gz
mkdir -p /home/$USER_DIR/.mujoco
tar -zxvf /home/$USER_DIR/mujoco.tar.gz -C /home/$USER_DIR/.mujoco
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER_DIR/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=/home/$USER_DIR/.mujoco/mujoco210" >> ~/.bashrc
```

### Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```
For PyTorch installation, refer to the official PyTorch setup guide to ensure compatibility with your hardware.


## 💻 Training & Inference
### Run Diffusion Veteran 
We provide a **single script** for easy execution of the training and inference tasksets. You can run the following commands to get started with different environments:
```bash
bash scripts/DV_antmaze_reimp.sh
bash scripts/DV_kitchen_reimp.sh
bash scripts/DV_maze2d_reimp.sh
bash scripts/DV_mujoco_reimp.sh
```

### Try Different Configurations
To experiment with different configurations, simply modify the settings in scripts_templates/train_template.sh. This file allows you to easily adjust parameters for your experiments. Once you’ve updated the configuration, run the following script:
```bash
bash scripts_templates/template.sh
```
This provides flexibility for trying different setups without needing to manually edit each script.

## 🙏 Acknowledgements
🎉 Currently, Diffusion-Veteran is offically supported by [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser). You are also welcomed to conduct unified ablation studies or standardized comparisons with the baselines in CleanDiffuser. Please see the [license](LICENSE) for further details. 

## 📚 Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{lu2025what,
  title={What Makes a Good Diffusion Planner for Decision Making?},
  author={Haofei Lu and Dongqi Han and Yifei Shen and Dongsheng Li},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=7BQkXXM8Fy}
}
```
Also consider citing these prior works that helped contribute to this project:
```bibtex
@article{janner2022planning,
  title={Planning with diffusion for flexible behavior synthesis},
  author={Janner, Michael and Du, Yilun and Tenenbaum, Joshua B and Levine, Sergey},
  journal={arXiv preprint arXiv:2205.09991},
  year={2022}
}
@article{ajay2022conditional,
  title={Is conditional generative modeling all you need for decision-making?},
  author={Ajay, Anurag and Du, Yilun and Gupta, Abhi and Tenenbaum, Joshua and Jaakkola, Tommi and Agrawal, Pulkit},
  journal={arXiv preprint arXiv:2211.15657},
  year={2022}
}
@article{dong2024cleandiffuser,
  title={Cleandiffuser: An easy-to-use modularized library for diffusion models in decision making},
  author={Dong, Zibin and Yuan, Yifu and Hao, Jianye and Ni, Fei and Ma, Yi and Li, Pengyi and Zheng, Yan},
  journal={arXiv preprint arXiv:2406.09509},
  year={2024}
}
```

## 🏷️  License
Please see the [license](LICENSE.txt) for more details.
