import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import pdb

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon, num_actions, device):
        if random.random() > epsilon:
            # pdb.set_trace()
            state  = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.forward(Variable(state, volatile=True)).max(1)[1]
            return action.data[0] + 1
        else:
            return random.randrange(1,num_actions+1)
