import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np
import random
import torch.nn.functional as F

BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 8        # how often to update the network
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class QNetwork(nn.Module):
    def __init__(self, action_size, input_shape, input_channels):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 8, kernel_size=3, stride = 2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size = 3, stride = 2), nn.BatchNorm2d(16), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 3, stride = 1), nn.BatchNorm2d(32), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1), nn.BatchNorm2d(32), nn.ReLU())
        out_size = self.calc_size(input_channels, input_shape, self.conv1, self.conv2, self.conv3, self.conv4)
        self.fc1 = nn.Sequential(nn.Linear(out_size, 32), nn.ReLU())
        self.out = nn.Linear(32,action_size)
    
    def calc_size(self, input_channels, input_shape, *layers):
        x = torch.rand(1, input_channels, input_shape, input_shape)
        for layer in layers:
            x = layer(x)
        out_size = x.size(1)*x.size(2)*x.size(3)
        return out_size
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.out(x)
        return x

class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add_exp(self, state, action, reward, next_state, done):
        state = np.array(state)/255
        next_state = np.array(next_state)/255
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample_batch(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.Tensor(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.Tensor(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.Tensor(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.Tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)

class Agent():

    def __init__(self, action_size, input_shape, input_channels):
    
        self.input_shape = input_shape
        self.action_size = action_size

        self.qnetwork_local = QNetwork(action_size, input_shape, input_channels).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
    
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        action = np.argmax(np.array(action))
    
        self.memory.add_exp([state], action, reward, [next_state], done)
        
        self.t_step +=1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                for i in range(5):
                    experiences = self.memory.sample_batch()
                    self.learn(experiences, GAMMA)

    def get_action(self, state, eps):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.Tensor(state/255).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            action_idx = np.argmax(action_values.cpu().data.numpy())
            action = np.zeros((self.action_size,))
            action[action_idx] = 1
            return list(action)
        else:
            action_idx = random.choice(np.arange(self.action_size))
            action = np.zeros((self.action_size,))
            action[action_idx] = 1
            return list(action)
        

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next, _ = torch.max(self.qnetwork_local(next_states).detach(), dim = 1, keepdim = True)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    
        Q_expected = torch.gather(self.qnetwork_local(states), dim = 1, index = actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    a = torch.rand(64,3)
    a, _ = torch.max(a, dim = 1, keepdim = True)
    print(a.size())