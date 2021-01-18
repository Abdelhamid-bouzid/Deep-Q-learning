# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:00:45 2020

@author: Abdelhamid
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelDeepQnetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(DuelDeepQnetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        
        self.fc1 = nn.Linear(fc_input_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.V   = nn.Linear(512, 1)
        self.A   = nn.Linear(512, n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self,state):
        conv1      = F.relu(self.conv1(state))
        conv2      = F.relu(self.conv2(conv1))
        conv3      = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1      = F.relu(self.fc1(conv_state))
        flat2      = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A
        
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))        
        
    def save_checkpoint(self, path):
        print('#################### saving: '+ path +'  checkpoint ######################')
        T.save(self.state_dict(), path)
        
    def load_checkpoint(self, path):
        print('#################### loading: '+ path +'  checkpoint ######################')
        self.load_state_dict(T.load(path))
