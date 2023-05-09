import gym
from gym import register
import pygame

import tensorflow as tf
from ray.rllib import TFPolicy
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from Engine import Engine
from model import EngineModel

tf.compat.v1.disable_eager_execution()

register(
    id='Engine-v0',
    entry_point='Engine:Engine',
    max_episode_steps=1000
)

version = "V5"

register_env("E", lambda _: Engine())
ModelCatalog.register_custom_model("EngineModel", EngineModel)
my_restored_policy = TFPolicy.from_checkpoint(f"result/{version}/training_models/policy_checkpoint")
CartpoleEnv = gym.make("Engine-v0")
obs = CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
done = False
while not done:
    actions = my_restored_policy.compute_single_action(obs)
    obs, rew, done, info = CartpoleEnv.step(actions[0])
    total_rev += rew
    CartpoleEnv.render()
print(total_rev)
CartpoleEnv.close()