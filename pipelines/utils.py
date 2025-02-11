import random
import time
import uuid
import os
import json
import wandb
import wandb.sdk.data_types.video as wv
import numpy as np
import torch
from omegaconf import OmegaConf

from cleandiffuser.env.wrapper import VideoRecordingWrapper

import json

def full_qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    assert 'timeouts' in dataset
    timeout_ = []

    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        timeout_bool = bool(dataset['timeouts'][i])

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        timeout_.append(timeout_bool)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'timeouts': np.array(timeout_),
    }

def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)
    return base


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Timer:
    def __init__(self):
        self.tik = None

    def start(self):
        self.tik = time.time()

    def stop(self):
        return time.time() - self.tik
    
    
class Logger:
    """Primary logger object. Logs in wandb."""
    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / 'models')
        self._video_dir = make_dir(self._log_dir / 'videos')
        self._cfg = cfg

        wandb.init(
            config=OmegaConf.to_container(cfg),
            project=cfg.project,
            group=cfg.group,
            name=cfg.exp_name,
            id=str(uuid.uuid4()),
            mode=cfg.wandb_mode,
            dir=self._log_dir
        )
        self._wandb = wandb

    def video_init(self, env, enable=False, video_id=""):
        # assert isinstance(env.env, VideoRecordingWrapper)
        if isinstance(env.env, VideoRecordingWrapper):
            video_env = env.env
        else:
            video_env = env
        if enable:
            video_env.video_recoder.stop()
            video_filename = os.path.join(self._video_dir, f"{video_id}_{wv.util.generate_id()}.mp4")
            video_env.file_path = str(video_filename)
        else:
            video_env.file_path = None
            
    def log(self, d, category):
        assert category in ['train', 'inference']
        assert 'step' in d
        print(f"[{d['step']}]", " / ".join(f"{k} {v:.2f}" for k, v in d.items()))
        with (self._log_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": d['step'], **d}) + "\n")
        _d = dict()
        for k, v in d.items():
            _d[category + "/" + k] = v
        self._wandb.log(_d, step=d['step'])
        
    def save_agent(self, agent=None, identifier='final'):
        if agent:
            fp = self._model_dir / f'model_{str(identifier)}.pt'
        agent.save(fp)
        print(f"model_{str(identifier)} saved")

    def finish(self, agent):
        try:
            self.save_agent(agent)
        except Exception as e:
            print(f"Failed to save model: {e}")
        if self._wandb:
            self._wandb.finish()


    
    


    
