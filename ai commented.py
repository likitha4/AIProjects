# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:23:24 2020

@author: HOME
"""
#libraries

import numpy as np
import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable 

#architecture of neural network
class Network(nn.Module):
    
    def __init__(self, input_size,nb_action):
         super(Network,self).__init__()
         self.input_size=input_size
         self.nb_action= nb_action
         self.fc1=nn.Linear(input_size, 30) #input size and hidden neurons 30
         self.fc2=nn.Linear(30,nb_action) #hidden neurons 30 and outputactions 3
         
    def forward(self,state):   #forward function
        x = F.relu(self.fc1(state)) #relu function on input-hidden layer
        q_values=self.fc2(x)
        return q_values
    
#implementing experience replay(considering more states in past along with current to find next Long term memory)
    
class replaymemory(object):
    
    def __init__(self,capacity):
        self.capacity= capacity #max no.of transitions we want to have in our memory of events
        self.memory= []  #memory contains last 100 events
        
    def push(self, event):
         self.memory.append(event) #events or transitions are appended
         if len(self.memory)>self.capacity:
             del self.memory[0]
             
    def sample(self, batchsize): #samples are previous events with values
        samples=zip(*random.sample(self.memory,batch_size))
        return map(lambda x: Variable(torch.cat(x,0)),samples)#we get list of batches,each batch-pytorch variable
        
  # if list=((1,2,3),(4,5,6)) then zip(*list)=((1,4)(2,3),(5,6))
        #pytorch:-tensor and gradient
   
 #implementing deep q-learning 
    #unsqueeze returns tensor with the dimension mentioned in the parenthesis
    #tensor is the same as numpy array
    
class Dqn():
     #nb_action=left,right,straight,gamma-delay coefficient
    def __init__(self,input_size,nb_action,gamma):
        self.gamma=gamma
        self.reward_window=[]#sliding window with mean of last 100 rewards used to evaluate performance
        self.model=Network(input_size,nb_action)
        self.memory=replaymemory(100000)
        self.optimizer=optim.Adam(self.model.parameters(),lr=0.001) #object of Adam class,(learningrate)lr=agent to explore
        self.last_state=torch.Tensor(input_size).unsqueeze(0)
        self.last_action=0
        self.last_reward=0
     
    def select_action(self,state):
        probs=F.softmax(self.model(Variable(state,volatile=True))*7) #multiplying with 7 to get high q values
        # softmax([1,2,3]) = [0.04,0.11,0.85]=>softmax([1,2,3]*3)=[0,0.02,0.98]
        action=probs.multinomial()# it gives random draw
        return action.data[0,0]
        
    def learn(self,batch_state,batch_next_state,batch_reward,batch_action):
        outputs=self.model(batch_state).gather(1,batch_action).unsqueeze(1).squeeze(1)
        # to make batch state and batch action of same dimension we use unsqueeze
        #to get the action which is chosen
        next_outputs=self.model(batch_next_state).detach().max(1)[0]
        #we get max of all q values of the next state represented by index 0 according to all the actions represented by index 1
        target=self.gamma*next_outputs + batch_reward
        td_loss=F.smooth_l1_loss(outputs,target) #smooth_l1_loss is the loss function of Q Learning
        self.optimizer.zero_grad()#reintialise  from one iteration to other in the loop of stochastic gradient descent 
        td_loss.backward(retain_variables=True)#back propagation
        self.optimizer.step() #updates the weights 
             
    def update(self,reward,new_signal):
        new_state=torch.Tensor(new_signal).float().unsqueeze(0) #all the states are torch tensors
        self.memory.push((self.last_state,new_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward]))) 
        #converts simple 0 and 1 to tensors
        action=self.select_action(new_state)
        if len(self.memory.memory)>100:
            batch_state,batch_next_state,batch_reward,batch_action=self.memory.sample(100) #collecting random batches
            self.learn(batch_state,batch_next_state,batch_reward,batch_action)#learning the random batches
        self.last_action=action
        #reached new state now
        self.last_state=new_state
        self.last_reward=reward
        self.reward_window.append(reward)#reward window has fixed size 
        if len(reward_window)>1000:
            del self.reward_window[0]
        return action
     
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)#mean of all the rewards in the reward window
        #1 is added so that denominator can never be zero 
            
        
    def save(self):
        #saving the model(neural network, optimizer,last weights)
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    }, 'last_brain1.pth')
             
    def load(self):
          
         #loading the saved model
          if os.path.isfile('last_brain1.pth'):
              print("=>loading checkpoint...")
              checkpoint=torch.load('last_brain1.pth')
              self.model.load_state_dict(checkpoint['state_dict'])
              self.optimizer.load_state_dict(checkpoint['optimizer'])
              print("done")
        
          else:
              print("no check point found...")

             
             
             
             
             
             
             
             
             
             
             
             
             
             