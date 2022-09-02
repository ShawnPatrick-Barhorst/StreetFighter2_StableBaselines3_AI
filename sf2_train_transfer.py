import gym
import retro
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import time

from gym import Env
import cv2
import numpy as np
from gym.spaces import MultiBinary, Box

states = ['ryu_guile',
          'ryu_ken',
          'ryu_chunli',
          'ryu_zangief',
          'ryu_dhalsim',
          'ryu_honda',
          'ryu_ryu',
          'ryu_sagat',
          'ryu_balrog',
          'ryu_blanka',
          'ryu_vega',
          'ryu_bison']

model_name = 'a2c_mlp_world_2'

for i in range(2):
    for state in states:
        class StreetFighter(Env):
            def __init__(self):
                super().__init__()
                self.observation_space = Box(low=0, high=255, shape=(100, 128, 1), dtype=np.uint8)
                self.action_space = MultiBinary(12)
                self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED, scenario='scenario_playback', state=state)
                #self.score = 0
            
            def step(self, action):
                obs, reward, done, info = self.game.step(action)
                obs = self.preprocess(obs)
                
                # Shape reward
                reward = info['score'] - self.score 
                self.score = info['score']

                return obs, reward, done, info
            
            def render(self, *args, **kwargs): 
                self.game.render()
            


            def reset(self):
                self.previous_frame = np.zeros(self.game.observation_space.shape)
                
                # Frame delta
                obs = self.game.reset()
                obs = self.preprocess(obs)
                self.previous_frame = obs
                
                # Create initial variables
                self.score = 0

                return obs
            
            def preprocess(self, observation): 
                gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
                resize = cv2.resize(gray, (100,128), interpolation=cv2.INTER_CUBIC)
                state = np.reshape(resize, (100,128, 1))
                return state
            
            def close(self): 
                self.game.close()


        #create environment
        env = StreetFighter()
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')
        env.reset()


        #create new model or load one for training
        model = A2C.load(model_name)
        model.set_env(env)
        model.learn(total_timesteps=100000)
        model.save(model_name)
        env.close()





