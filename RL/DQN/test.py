from vizdoom import *
import random
import time
import cv2
import numpy as np
from model import Agent
import torch



def train(game, episodes, eps, eps_decay, agent, input_shape, max_rew = -1e-5, resume = False, path = 'None'):
    game.set_window_visible(False)
    if resume:
        agent.qnetwork_local.load_state_dict(torch.load(path))
        agent.qnetwork_target.load_state_dict(torch.load(path))
    game.init()
    for i in range(episodes):
        game.new_episode()
        
        while not game.is_episode_finished():
            state = game.get_state().screen_buffer
            state= cv2.resize(state, (input_shape, input_shape))
            if len(state.shape)==2:
                state = np.expand_dims(state, axis=0)
            
            action = agent.get_action(state, eps)
            reward = game.make_action(action)
            done = 1 if game.is_episode_finished() else 0
            if not done:
                next_state = game.get_state().screen_buffer
                next_state = cv2.resize(next_state, (input_shape, input_shape))
                if len(next_state.shape)==2:
                    next_state = np.expand_dims(next_state, axis=0)
            else:
                next_state = np.zeros((input_channels, input_shape, input_shape))

            agent.step(state, action, reward, next_state, done)
            state = next_state
            eps*=eps_decay
            #time.sleep(0.02)
        total_rew = game.get_total_reward()  
        print (f'Episode {i} Reward {total_rew}')
        if total_rew >= max_rew:
            max_rew = total_rew
            torch.save(agent.qnetwork_local.state_dict(), f'./checkpoints/basic{total_rew}.pth')

def infer(game, agent, episodes, path):
    game.set_window_visible(True)
    game.init()
    agent.qnetwork_local.load_state_dict(torch.load(path))
    for i in range(episodes):
        game.new_episode()
        
        while not game.is_episode_finished():
            state = game.get_state().screen_buffer
            state= cv2.resize(state, (input_shape, input_shape))
            if len(state.shape)==2:
                state = np.expand_dims(state, axis=0)
            
            action = agent.get_action(state, eps=0)
            reward = game.make_action(action)
            time.sleep(0.05)
        time.sleep(2)

if __name__ == '__main__':
    game = DoomGame()
    game.load_config("./scenarios/basic.cfg")
    game.set_screen_format(ScreenFormat.GRAY8)

    input_shape = 50
    input_channels = 1
    action_size = 3
    eps = 1
    eps_decay = 0.9
    agent = Agent(action_size, input_shape, input_channels)
    max_rew = -1e5
    episodes = 5000

    train(game, episodes, eps, eps_decay, agent, input_shape)
    #infer(game, agent, episodes = 5, path = './checkpoints/basic.pth')
