import gym
# import gym_chess
from stable_baselines3 import PPO
# from functools import partial
import os

model_dir = "momos"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

log_dir = "dados"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


env = gym.make('LunarLander-v2')

'''
sp = env.reset()
fake = env.step
def sus(act):
    try:
        return fake(act)
    except ValueError:
        return sp, -1, True, {}
env.step = sus
'''


env.reset()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
n = 10_000
i=69
while 1:
    model.learn(total_timesteps=n, tb_log_name=f"Alcatraz_10", reset_num_timesteps=False)
    model.save(f"{model_dir}/{2**(n/10_000)*i}")
    i+=1
done = False
epis = 10

for ep in range(epis):
    ob = env.reset()
    done = False
    while not done:
        env.render()
        ob, rew, done, info = env.step(model.predict(ob)[0])
env.close()
'''
while 1:
    env.reset()
    while not done:
        env.render()
        ob, rew, done, info = env.step(env.action_space.sample())
'''
