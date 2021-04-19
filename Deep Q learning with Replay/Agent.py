# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:34:24 2020

@author: Abdelhamid 
"""
import numpy as np
import torch as T
from DuelDeepQnetwork import DuelDeepQnetwork
from Buffer import Buffer

class Agent(object):
    def __init__(self, gamma, lamda, epsilon, lr, input_dims, n_actions, max_size, check_dir, replace, batch_size,eps_dec,eps_min):
        
        self.gamma      = gamma
        self.lamda      = lamda
        self.epsilon    = epsilon
        self.lr         = lr
        self.input_dims = input_dims
        self.n_actions  = n_actions
        self.max_size   = max_size
        self.check_dir  = check_dir
        self.replace    = replace
        self.batch_size = batch_size
        self.eps_min    = eps_min
        self.eps_dec    = eps_dec
        
        self.learn_step_cntr = 0
        
        self.memory = Buffer(self.max_size, self.input_dims)
        
        self.model = DuelDeepQnetwork(self.lr, self.n_actions, self.input_dims)
        
    def choose_action(self, state):
        self.model.eval()
        if np.random.random()> self.epsilon:
            state   = T.from_numpy(np.array([state], copy=False, dtype=np.float32)).to(self.model.device)
            actions = self.model.forward(state)
            action     = T.argmax(actions).item()
            
        else:
            possible_actions = [i for i in range(self.n_actions)]
            action           = np.random.choice(possible_actions)
        self.model.train()    
        return action
    
    def learn(self):
        if self.memory.current_mem < self.batch_size:
            return
        
        self.model.optimizer.zero_grad()
        
        indices = np.arange(self.batch_size)
        
        states, actions, rewards, n_states, terminal = self.sample_memory()
        
        Q_pred       = self.model.forward(states)
        N_Q_pred      = self.model.forward(n_states)
        
        max_actions   = T.argmax(N_Q_pred, dim=1)
        N_Q_pred[terminal] = 0.0
        
        Q_pred   = Q_pred[indices,actions]
        N_Q_pred = N_Q_pred[indices,max_actions]
        
        '''########################################### compute loss #################################################'''
        truth    = rewards + self.gamma*N_Q_pred
        loss           = self.model.loss(truth,Q_pred).to(self.model.device)
        loss.backward()
        
        self.model.optimizer.step()
        self.learn_step_cntr += 1
        
        self.decrement_epsilon()
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
                        
            
    def store_transitions(self, state, action, reward, n_state, done):
        self.memory.store_transitions(state, action, reward, n_state, done)
        
    def sample_memory(self):
        
        states, actions, rewards, n_states, terminal = self.memory.sample_buffer(self.batch_size)
         
        states    = T.tensor(states).to(self.model.device)
        actions   = T.tensor(actions).to(self.model.device)
        rewards   = T.tensor(rewards).to(self.model.device)
        n_states  = T.tensor(n_states).to(self.model.device)
        terminal  = T.tensor(terminal).to(self.model.device)
        
        return states, actions, rewards, n_states, terminal
    
    def save_models(self):
        self.model.save_checkpoint()

    def load_models(self):
        self.model.load_checkpoint()
        
