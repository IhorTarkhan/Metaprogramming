import ray
import os


from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from gym.envs.registration import register
import gym
import ray.rllib.agents.ppo as ppo
from model import EngineModel
import matplotlib.pyplot as plt

MAX_EPISODE_STEPS = 500 * 500

ready_to_exit = False

version = "V5"

def save_graph_train(lens, rewards):
    x = range(1, len(rewards) + 1)

    plt.figure()
    plt.plot(x, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    os.makedirs(f'result/{version}')
    plt.savefig(f'result/{version}/rewards')
    plt.figure()
    plt.plot(x, lens)
    plt.xlabel('Episode')
    plt.ylabel('Mean length')
    plt.savefig(f'result/{version}/steps_per_episode')


def create_my_env():
    register(
        id='Engine-v0',
        entry_point='Engine:Engine',
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    return gym.make("Engine-v0")


ray.init(include_dashboard=False)
register_env("E", lambda config: create_my_env())
ModelCatalog.register_custom_model("EngineModel", EngineModel)

trainer = ppo.PPOTrainer(env="E",
                         config={"model": {"custom_model": "EngineModel"},
                                 'create_env_on_driver': True,
                                 'gamma': 0.99,
                                 'train_batch_size': 400,
                                 'lambda': 0.95,
                                 'entropy_coeff': 0.001
                                 # 'kl_target': 0.02,
                                 # 'clip_param': 0.3
                                 }
                         )


_rewards = []
_lens = []
while True:
    if len(_rewards) > 3000:
        break
    rest = trainer.train()
    cur_reward = rest["episode_reward_mean"]
    cur_length = rest["episode_len_mean"]
    _rewards.append(cur_reward)
    _lens.append(cur_length)
    print(f"episode: {len(_rewards)};\tavg. reward: {round(cur_reward, 2)};\tavg. episode length: {round(cur_length, 2)}")

save_graph_train(_lens, _rewards)
trainer.save(f"result/{version}/training_models/checkpoints")
default_policy = trainer.get_policy(policy_id="default_policy")
default_policy.export_checkpoint(f"result/{version}/training_models/policy_checkpoint")
ray.shutdown()
