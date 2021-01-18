# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:34:24 2020

@author: Abdelhamid 
"""
import numpy as np
import torch as T
from DeepQnetwork import DeepQnetwork
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
        
        self.eval_model = DeepQnetwork(self.lr, self.n_actions, self.input_dims, self.check_dir, 'eval_model_0')
        self.next_model = DeepQnetwork(self.lr, self.n_actions, self.input_dims, self.check_dir, 'next_model_0')
        
    def choose_action(self, state):
        self.eval_model.eval()
        if np.random.random()> self.epsilon:
            state   = T.tensor([state],dtype=T.float).to(self.eval_model.device)
            
            actions = self.eval_model.forward(state)
            action  = T.argmax(actions).item()
            
        else:
            possible_actions = [i for i in range(self.n_actions)]
            action           = np.random.choice(possible_actions)
        self.eval_model.train()    
        return action
    
    def learn(self):
        if self.memory.current_mem < self.batch_size:
            return
        
        self.eval_model.optimizer.zero_grad()
        self.update_model_weights()
        
        indices = np.arange(self.batch_size)
        
        states, actions, rewards, n_states, terminal = self.sample_memory()
        
        '''#################### choose best action for next states accorfing to eval model ##########################'''
        Q_n_state_eval = self.eval_model.forward(n_states)
        n_max_actions  = T.argmax(Q_n_state_eval, dim=1)
        '''##########################################################################################################'''
        
        Q_n_state_next = self.next_model.forward(n_states)
        Q_n_state_next[terminal] = 0.0
        
        '''########################################### compute loss #################################################'''
        Q_state_pred   = self.eval_model.forward(states)[indices, actions]
        Q_target       = rewards + self.gamma*Q_n_state_next[indices, n_max_actions]
        loss           = self.eval_model.loss(Q_target,Q_state_pred).to(self.eval_model.device)
        loss.backward()
        
        self.eval_model.optimizer.step()
        self.learn_step_cntr += 1
        
        self.decrement_epsilon()
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
                               
    def update_model_weights(self):
        if (self.replace is not None) and (self.learn_step_cntr % self.replace ==0):
            self.next_model.load_state_dict(self.eval_model.state_dict())
            
    def store_transitions(self, state, action, reward, n_state, done):
        self.memory.store_transitions(state, action, reward, n_state, done)
        
    def sample_memory(self):
        
        states, actions, rewards, n_states, terminal = self.memory.sample_buffer(self.batch_size)
         
        states    = T.tensor(states).to(self.eval_model.device)
        actions   = T.tensor(actions).to(self.eval_model.device)
        rewards   = T.tensor(rewards).to(self.eval_model.device)
        n_states  = T.tensor(n_states).to(self.eval_model.device)
        terminal  = T.tensor(terminal).to(self.eval_model.device)
        
        return states, actions, rewards, n_states, terminal
    
    def save_models(self):
        self.eval_model.save_checkpoint()
        self.next_model.save_checkpoint()

    def load_models(self):
        self.eval_model.load_checkpoint()
        self.next_model.load_checkpoint()
        
