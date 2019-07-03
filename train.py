import math
import random
from collections import namedtuple, deque
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from environment import StochasticMDP
from model import Net
from misc import ReplayBuffer
from tqdm import tqdm 

import pdb

random.seed(3)
dev = 'cuda:' + str(sys.argv[1])
device = torch.device(dev if torch.cuda.is_available() else 'cpu')
print(device)

env = StochasticMDP()

num_goals    = env.num_states
num_actions  = env.num_actions

controller        = Net(2*num_goals, num_actions).to(device)
target_controller = Net(2*num_goals, num_actions).to(device)
for p in target_controller.parameters():
   p.requires_grad = False

meta_controller        = Net(num_goals, num_goals).to(device)
target_meta_controller = Net(num_goals, num_goals).to(device)
for p in target_meta_controller.parameters():
    p.requires_grad = False


optimizer      = optim.Adam(controller.parameters())
meta_optimizer = optim.Adam(meta_controller.parameters())

replay_buffer      = ReplayBuffer(10000)
meta_replay_buffer = ReplayBuffer(10000)

def to_onehot(x):
    oh = np.zeros(6)
    oh[x - 1] = 1.
    return oh

def update(model, target_model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return
    # pdb.set_trace()
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state      = Variable(torch.FloatTensor(state)).to(device)
    next_state = Variable(torch.FloatTensor(next_state), volatile=True).to(device)
    action     = Variable(torch.LongTensor(action)).to(device)
    reward     = Variable(torch.FloatTensor(reward)).to(device)
    done       = Variable(torch.FloatTensor(done)).to(device)
    
    q_value = model(state)
    q_value = q_value.gather(1, action.unsqueeze(1)-1).squeeze(1)
    
    next_q_value     = target_model(next_state).max(1)[0]
    expected_q_value = reward + 0.99 * next_q_value * (1 - done)
   
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    #num_frames = 100000
    num_frames = int(sys.argv[2])
    frame_idx  = 1

    state = env.reset()
    done = False
    all_rewards = []
    episode_reward = 0
    final_rewards = []
    last_idx = 0
    #for frame_idx in tqdm(range(1,num_frames+1)):
    while frame_idx < num_frames:
        # pdb.set_trace()
        target_controller.load_state_dict(controller.state_dict())
        target_meta_controller.load_state_dict(meta_controller.state_dict())
        
        goal = meta_controller.act(state, epsilon_by_frame(frame_idx), num_goals, device)
        onehot_goal  = to_onehot(goal)
        
        meta_state = state
        extrinsic_reward = 0
        
        while not done and goal != np.argmax(state):
            # pdb.set_trace()
            #print(frame_idx)
            goal_state  = np.concatenate([state, onehot_goal])
            action = controller.act(goal_state, epsilon_by_frame(frame_idx), num_actions, device)
            next_state, reward, done, _ = env.step(action)

            episode_reward   += reward
            extrinsic_reward += reward
            intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0

            replay_buffer.push(goal_state, action, intrinsic_reward, np.concatenate([next_state, onehot_goal]), done)
            state = next_state
            
            update(controller, target_controller, optimizer, replay_buffer, 32)
            update(meta_controller, target_meta_controller, meta_optimizer, meta_replay_buffer, 32)
            frame_idx += 1
            
            if frame_idx % 1000 == 0:
                #pdb.set_trace()
                print("frame_idx: " + str(frame_idx) + "/" + str(num_frames) + " (" +str(frame_idx*100.0/num_frames) + "%)" + " :: rewards: " + str(np.mean(all_rewards[last_idx:])))
                #last_idx = len
                final_rewards.append(np.mean(all_rewards))
                last_idx = len(all_rewards)
                #pdb.set_trace()
                # clear_output(True)
                # n = 100 #mean reward of last 100 episodes
                # plt.figure(figsize=(20,5))
                # plt.title(frame_idx)
                # plt.plot([np.mean(all_rewards[i:i + n]) for i in range(0, len(all_rewards), n)])
                # plt.show()

        meta_replay_buffer.push(meta_state, goal, extrinsic_reward, state, done)
            
        if done:
            state = env.reset()
            done  = False
            all_rewards.append(episode_reward)
            episode_reward = 0      
    final_rewards = np.asarray(final_rewards)
    np.save('final_rewards', final_rewards)

if __name__ == '__main__':
    train()
