# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 00:26:47 2020

@author: Abdelhamid 
"""
import gym
from Agent import Agent
from plot import plot_epi_step

if __name__ == '__main__':
    
    env     = gym.make('CartPole-v1')
    agent   = Agent(lr=10**-4, n_actions=env.action_space.n, input_dim=env.observation_space.shape, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01)
    n_games = 10000
    
    scores = []
    steps  = []
    
    for i in range(n_games):
        
        score  = 0
        cont   = 0
        done   = False
        state  = env.reset()
        
        while not done:
            
            action = agent.choose_action(state)
            n_state, reward, done, info = env.step(action)
            
            score += reward
            agent.learn(state,action,reward,n_state)
            
            state  = n_state
            cont  +=1
        scores.append(score)
        steps.append(cont)
        print("############## episode number = {} ######### number of steps = {} ############ score {}".format(i, cont, score))
        
    plot_epi_step(scores,steps)