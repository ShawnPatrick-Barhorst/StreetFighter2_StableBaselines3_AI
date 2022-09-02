import pygame
import retro
import retrowrapper
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import time
import sys
from pygame.locals import *

from gym import Env
import cv2
import numpy as np
from gym.spaces import MultiBinary, Box

import signal

pygame.joystick.init()
j = pygame.joystick.Joystick(0)
j.init()


#load model
gamename = "StreetFighterIISpecialChampionEdition-Genesis"
modelname = 'a2c_mlp_world_1'
model = A2C.load(modelname)

#take observation from obs1 and convert it to obs2 shape
def preprocess(observation):
    gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (100,128), interpolation=cv2.INTER_CUBIC)
    state = np.reshape(resize, (100,128, 1))
    state = np.reshape(state, (1, 100, 128))
    return state

#Custom gym class
class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(100, 128, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED, state='ryuryu', scenario='scenario_playback')
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        
        reward = info['score'] - self.score 
        self.score = info['score']

        return obs, reward, done, info
    
    def render(self, *args, **kwargs): 
        self.game.render()
    


    def reset(self):
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)

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

#Create environments
env = retrowrapper.RetroWrapper(gamename, state='ryuryu', players=2)
env2 = StreetFighter()
env2 = DummyVecEnv([lambda: env2])
env2 = VecFrameStack(env2, 4, channels_order='last')


model.set_env(env2)

obs = env.reset()

#initialize pygame display
pygame.init()
win = pygame.display.set_mode((1200,900), pygame.FULLSCREEN|pygame.SCALED)
clock = pygame.time.Clock()


#initialize input array
butts = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
action_array = [0,0,0,0,0,0,0,0,0,0,0,0]

done = False


#Macgyver Frame stack
obs_mod = preprocess(obs)
obs_stack = np.array([obs_mod, obs_mod, obs_mod, obs_mod])
obs2 = np.concatenate((obs_stack[0], obs_stack[1], obs_stack[2], obs_stack[3]))

pause = True


while not done:
    while pause:
        
        actions = set()
        
        # Display
        #decode numpy array into a displayable image
        img = pygame.image.frombuffer(obs.tobytes(), obs.shape[1::-1], "RGB")
        img = pygame.transform.scale(img, (1200,900))
        win.blit(img,(0,0))
        pygame.display.flip()
        
        # Control Events
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            
            if keys[pygame.K_RIGHT] or (j.get_axis(0) >= 0.5):
                actions.add('RIGHT')
            if keys[pygame.K_LEFT] or (j.get_axis(0) <= -0.5):
                actions.add('LEFT') 
            if keys[pygame.K_DOWN] or (j.get_axis(1) >= 0.5):
                actions.add('DOWN')
            if keys[pygame.K_UP] or (j.get_axis(1) <= -0.5):
                actions.add('UP')            
            if keys[pygame.K_z] or j.get_button(2):
                actions.add('A')
            if keys[pygame.K_x] or j.get_button(1):
                actions.add('B')
            if keys[pygame.K_c] or j.get_button(5):
                actions.add('C')
            if keys[pygame.K_a] or j.get_button(3):
                actions.add('X')
            if keys[pygame.K_s] or j.get_button(0):
                actions.add('Y')
            if keys[pygame.K_d] or j.get_button(4):
                actions.add('Z')
            if j.get_button(8):
                pygame.quit()
                env.close()
                env2.close()
                sys.exit()
            if j.get_button(9):
                pause = False
        
            for i, a in enumerate(butts):
                if a in actions:
                    action_array[i] =  1            
                else:
                    action_array[i] = 0


        a2, _ = model.predict(obs2)
        a2 = np.hstack(a2)
        act = a2.tolist() + action_array
        
        # Progress Environemt forward
        obs, _, _, _ = env.step(act)

        obs_mod = preprocess(obs)
        
        obs_stack[0] = obs_stack[1]
        obs_stack[1] = obs_stack[2]
        obs_stack[2] = obs_stack[3]
        obs_stack[3] = obs_mod
        
        #to avoid errors with multiple obs spaces at once, I used a np to reshape the original observation into a shape accepted by obs2
        #it is essentialy frame stacking but without being attatched to a model :)
        obs2 = np.concatenate((obs_stack[0], obs_stack[1], obs_stack[2], obs_stack[3]))


        clock.tick(60)

    my_square = pygame.Rect(50, 50, 50, 50)
    pygame.draw.rect(win, 'red', my_square)

    for event in pygame.event.get():
        if j.get_button(9):
                pause = True


