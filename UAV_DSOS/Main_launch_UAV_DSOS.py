"""
Arguments for UAV downlink environment:
EP_MAX :         Maximum Number of episodes/rounds.
LR_A :           Learning rate for actor.
LR_C :           Learning rate for critic.
TAU:             Parameter for soft replacement.
MEMORY_CAPACITY: Size of the memory.
BATCH:           Size of batch (the agent randomly takes a batch at each step as the training data).
"""

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from DSOS_core.UAV_env import Comm
from DSOS_core.DRL_DSOS import Actor, Critic

EP_MAX = 400
LR_A, LR_C, TAU = 0.00003, 0.00002, 0.01
MEMORY_CAPACITY, BATCH = 10000, 128


#-----------------------------------core of DSOS algorithm-----------------------------------
"""
Arguments:
ini_ue :          Initial users in each cluster.
s_dim :           Dimension of the states.
a_dim :           Dimension of the actions.
u_dim :           Maximum number of users in each cluster.
EP_LEN :          Maximum frames/steps in each episode/round. 
E:                The matrix for recording energy
R:                The matrix for recording Rewards
"""
env = Comm()
ini_ue, s_dim, a_dim, u_dim, EP_LEN = env.U_ini, env.s_dimension, env.a_dimension, env.max_num_user, env.T_max
actor = Actor(n_features=s_dim, n_actions=a_dim, lr=LR_A)
critic = Critic(n_features=s_dim, lr=LR_C)

E, R = np.zeros([EP_MAX, EP_LEN]), np.zeros([EP_MAX, EP_LEN])
all_ep_r, all_ep_e, all_ep_sd = [], [], []

for round in range(EP_MAX):
    s, ue, demands = env.reset(ini_ue)
    buffer_s, buffer_a, buffer_s_, buffer_r, buffer_td_e = [], [], [], [], []
    ep_r, ep_e, t, sum_e, sum_d = 0, 0, 0, 0, 0
    random.seed(132)
    t1 = time.time()
    for t in range(EP_LEN):
        a = actor.choose_action(s)
        s_, r, fe, sum_e_, sum_d_, done = env.step(ue, a, s, sum_e, sum_d, t, round)
        td_error = critic.get_td_error(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        E[round, t], R[round, t] = fe, r
        buffer_s.append(s), buffer_a.append(a), buffer_s_.append(s_), buffer_r.append(r), buffer_td_e.append(td_error)
        if len(buffer_td_e) > BATCH:
            buffer_s.pop(0), buffer_s_.pop(0), buffer_a.pop(0), buffer_r.pop(0), buffer_td_e.pop(0)
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            bs, bs_, ba, br, btde = np.vstack(buffer_s), np.vstack(buffer_s_), np.vstack(buffer_a), np.vstack(buffer_r), np.vstack(buffer_td_e)
            actor.learn(bs, ba, btde)  # true_gradient = grad[logPi(s,a) * td_error]
            critic.learn(bs, br, bs_)
        s, sum_e, sum_d, ep_r, t = s_, sum_e_, sum_d_, ep_r + r, t + 1
        if done == 1:
            print('Running time: ', time.time() - t1, 'Step: ', t)
            break
        #print('Episode:', round, 'Step:', t, 'Reward: %f' % r, ' Energy: %f' % fe)
    if len(all_ep_r):
        ep_r = all_ep_r[-1] * 0.9 + ep_r * 0.1
    if len(all_ep_e):
        ep_e = all_ep_e[-1] * 0.9 + ep_e * 0.1
    all_ep_r.append(ep_r)
    all_ep_e.append(sum_e)
    data_state = np.reshape(s[0, u_dim * u_dim: u_dim * u_dim + u_dim], [1, u_dim])
    sd_ratio = (np.sum(demands) - np.sum(data_state)) / np.sum(demands)
    if len(all_ep_sd):
        sd_ratio = all_ep_sd[-1] * 0.9 + sd_ratio * 0.1
    all_ep_sd.append(sd_ratio)
    print('Episode:', round, ' Epi_Reward: %f' % ep_r, ' Epi_Energy: %f' % sum_e, 'SD_ratio: %f' % sd_ratio)


if os.path.isdir('./figures'):
    pass
else:
    os.mkdir('./figures')

test_index = np.linspace(0, EP_MAX - 1, EP_MAX)
plt.figure(1)
plt.plot(test_index, all_ep_r, color='red', linewidth=1.0, linestyle='-')
plt.title("Rewards")
plt.grid(True)
plt.savefig('./figures/rewards_episodes.png')
plt.figure(2)
plt.plot(test_index, all_ep_e, color='blue', linewidth=1.0, linestyle='-')
plt.title("Energy")
plt.grid(True)
plt.savefig('./figures/energy_episodes.png')
plt.figure(3)
plt.plot(test_index, all_ep_sd, color='green', linewidth=1.0, linestyle='-')
plt.title("Supply/Demands ratio")
plt.grid(True)
plt.savefig('./figures/sd_ratio_episodes.png')
plt.show()
