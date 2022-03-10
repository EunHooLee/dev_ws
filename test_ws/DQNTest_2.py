from turtle import forward
import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():

    def __init__(self):
        self.buffer = collections.deque()
        self.batch_size = 32
        self.size_limit = 50000

    def put(self, data):
        self.buffer.append(data)
        if self.size_limit < len(self.buffer):
            self.buffer.popleft()
    
    def sample(self, n):
        return random.sample(self.buffer, n)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilion):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilion:
            return random.randint(0,1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, gamma, optimizer, batch_size):
    for i in range(10):
        batch = memory.sample(batch_size)
        s_1st, a_1st, r_1st, s_prime_1st, done_mask_1st = [], [], [], [], []

        for transition in batch:
            s, a, r, s_prime, done_mask = transition
            s_1st.append(s)
            a_1st.append([a])
            r_1st.append([r])
            s_prime_1st.append(s_prime)
            done_mask_1st.append([done_mask])
        
        s,a,r,s_prime,done_mask = torch.tensor(s_1st, dtype=torch.float), \
            torch.tensor(a_1st), torch.tensor(r_1st), torch.tensor(s_prime_1st, dtype=torch.float),\
                torch.tensor(done_mask_1st)
        q_out = q(s) # 32 x 2
        q_a = q_out.gather(1,a) # 행동한 action의 값만을 골라냄, 1 : 차원
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(target, q_a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    # state_dict()는 model의 wight 정보를 dictionary 형태로 갖고있다.
    memory = ReplayBuffer()

    avg_t = 0
    gamma = 0.98
    batch_size = 32
    optimizer = optim.Adam(q.parameters(), lr=0.0005)

    for n_epi in range(10000):
        epsilon = max(0.01,0.08-0.01*(n_epi/200))
        s = env.reset()

        for t in range(600):
            a = q.sample_action(torch.from_numpy(s).float(), epsilion=epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/200.0,s_prime, done_mask))
            s = s_prime 

            if render:
                env.render()

            if done:
                break
    if memory.size() > 2000:
        train(q,q_target, memory, gamma, optimizer, batch_size)
    
    if n_epi%20 == 0 and n_epi !=0:
        q_target.load_state_dict(q.state_dict())
        print('# of episode: {}, Avg timestep: {:.1f}, buffer size: {} epsilion: {:.1f}%'.foramt(
            n_epi, avg_t/20.0, memory.size(), epsilon*100
        ))


if __name__ == '__main__':
    main()