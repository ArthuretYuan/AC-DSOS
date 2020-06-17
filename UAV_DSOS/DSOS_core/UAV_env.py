"""
Arguments for UAV downlink environment:
N :             Number of clusters, fixed at 3.
K :             Maximum number of users.
K_n :           Maximum number of each cluster.
ini_UE:         Initial number users for each cluster. size=[1,3]
T_max:          Time limitation (in frame). UAV need to complete the service within this time.
B:              System bandwidth
Pmax:           Transmission power
var_noise:      System noise
"""

import numpy as np
import math
import random
import time
import sys
import itertools
import Data_generation as dg
import scipy.io as sio                     # import scipy.io for .mat file I/O
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

N, K, K_n, B, T_L, P_max = 3, 9, 3, 1, 280, 1
U_ini = np.array([3, 3, 3])
var_noise = 0.1

class Comm(object):
    def __init__(self):
        '''
        self.a_dimension:   The dimension of the action
        self.s_dimension:   The dimension of the state
        self.ini_UE:        The initial users for each cluster
        '''
        self.a_dimension = 1
        self.s_dimension = K * K + K + 1
        self.max_num_user = K
        self.T_max = T_L
        self.U_ini = U_ini
        self.G = []
        self.tic = 0
        self.num_G = 0
        self.t_max_cluster = np.zeros([1, N])
    def reset(self, u_ini):
        '''
        def reset(self, u_ini):
                            A method for resetting the system states (to initial states)
        Inputs:
        u_ini:              Initial number of users for each cluster. size=[1,3]
        Outputs:
        state_ini:          Initial state, including channel, remaining demands and current hovering cluster
        UE:                 New users information for each cluster in current episode
        b_ini:              The initial demands for all the users in current episode

        Variables:
        UE:                 New users information for each cluster in current episode, i.e., after arrival and departure.
        G:                  Group list. A list with the size of 2 ** UE[0], where each element is a candidate user group.
        h_ini:              A channel coefficients matrix. size = [K, K].
        b_ini:              The initial demands for all the users. size = [1, K].
        b_ini_cluster:      The total demands for each cluster. size = [1, N]
        c_ini:              The cluster that UAV is hovering at.
        state_ini:          The initial state. size = [1, K * (K + 1) + 1]
        '''
        time.sleep(0.1)
        # np.random.seed(1)  # initial state seed
        UE = np.array([0, 0, 0])
        for n in range(N):
            UE_arrive = np.random.poisson(lam=0)
            UE_depart = np.random.poisson(lam=0)
            UE[n] = u_ini[n] + UE_arrive - UE_depart
            UE[n] = np.clip(UE[n], 0, K_n)
        self.G = [None]
        for k in range(UE[0]):
            l = list(itertools.combinations(np.arange(0, UE[0]), k + 1))
            m = [list(x) for x in l]
            self.G = self.G + m
        self.num_G = 2 ** UE[0]

        h_ini, b_ini, b_ini_cluster, c_ini = np.zeros([K, K]), np.zeros([1, K]), np.zeros(N), 0
        for n in range(N):
            h_ini[K_n * n: K_n * n + UE[n], K_n * n: K_n * n + UE[n]] = np.random.randint(1, 5, size=[UE[n], UE[n]])
            b_ini[0, K_n * n: K_n * n + UE[n]] = np.random.randint(1, 5, size=[1, UE[n]])
            b_ini_cluster[n] = np.sum(b_ini[0, K_n * n: K_n * n + UE[n]])
        self.t_max_cluster = self.T_max * b_ini_cluster / np.sum(b_ini_cluster)

        h_ini = dg.beta_generation(K, h_ini)
        b_ini = dg.demands_generation(K, b_ini)

        state_ini = np.append(np.reshape(h_ini, [1, K * K]), np.reshape(b_ini, [1, K]))
        state_ini = np.append(state_ini, [c_ini])
        state_ini = state_ini[np.newaxis, :]
        return state_ini, UE, b_ini

    def step(self, UE, action, s_current, total_e, total_d, step,  epi):
        '''
        def step(self, UE, action, s_current, total_e, total_d, step,  epi):
                            A method for taking actions, obtaining a new state, and receiving all the infomations

        Inputs:
        UE:                 New users information for each cluster in current episode
        action:             Selected action from the actor
        s_current:          Current state
        total_e:            Total energy consumed
        total_d:            Total demands transmitted

        Outputs:
        s_, reward, frame_energy, total_e, total_d, done
        s_:                 Next state
        reward:             Received reward value
        frame_energy:       Energy consumed at the current frame
        total_e:            Total energy consumed (next)
        total_d:            Total demands transmitted (next)
        done:               Whether terminate the learning earlier for the current episode
        '''

        done = 0
        act = np.around((action+2)/4 * (self.num_G - 1))
        act = np.clip(act, 0, self.num_G - 1)
        act = act.astype(np.int64)

        # -------------------------------collect the state--------------------------------------------------------------
        h_current = np.reshape(s_current[0, 0: K * K], [K, K])
        b_current = np.reshape(s_current[0, K * K: K * K + K], [1, K])
        c_current = int(s_current[0, K * K + K])
        h_cluster = h_current[K_n * c_current: K_n * c_current + UE[c_current],
                    K_n * c_current: K_n * c_current + UE[c_current]]
        b_cluster = b_current[0, K_n * c_current: K_n * c_current + UE[c_current]] * 1000
        k_cluster = UE[c_current]

        # ------------------------------------search space restriction--------------------------------------------------
        u_selected = self.G[act]
        satisfied_id = np.argwhere(b_cluster == 0)
        satisfied = []
        for i in satisfied_id:
            satisfied.append(int(i))
            if u_selected is not None:
                if i in u_selected:
                    u_selected.remove(i)
        # ------------------------------calculate_data_and_energy-------------------------------------------------------
        r, e = np.zeros([1, k_cluster]), np.zeros([1, k_cluster])
        if u_selected is None:
            r[0, :] = np.zeros(k_cluster)
            e[0, :] = np.zeros(k_cluster)
        else:
            beta = np.zeros([k_cluster, k_cluster])
            beta[u_selected, u_selected] = h_cluster[u_selected, u_selected]
            for k_u in range(k_cluster):
                if k_u in satisfied:
                    r[0, k_u] = 0
                    e[0, k_u] = 0
                elif k_u in u_selected:
                    s = var_noise
                    for i_u in range(k_cluster):
                        if i_u != k_u and i_u in u_selected:
                            s = s + beta[k_u, i_u] * P_max
                    r[0, k_u] = B * math.log2(1 + beta[k_u, k_u] * P_max / s)
                    e[0, k_u] = beta[k_u, k_u] * P_max * math.log2(len(u_selected) + 1) * 1  # power_discount
                else:
                    r[0, k_u] = 0
                    e[0, k_u] = 0
        r_current = r[0]
        com_energy = np.sum(e)
        hov_energy = 2
        frame_energy = com_energy + hov_energy

        # ----------------------------------environment state transfer--------------------------------------------------
        h_next = np.zeros([K, K])
        random.seed(133)
        for l in range(K):
            for k in range(K):
                some_list = [h_current[l, k], h_current[l, k] + 0.1, h_current[l, k] - 0.1]
                probabilities = [0.996, 0.002, 0.002]
                h_next[l, k] = np.clip(dg.random_pick(some_list, probabilities), 0, 1)
        b_cluster_next = np.clip(b_cluster - r_current, 0, 500)
        b_current[0, K_n * c_current: K_n * c_current + UE[c_current]] = b_cluster_next / 1000

        total_d = total_d + np.sum(r_current)
        total_e = total_e + frame_energy
        energy_efficiency = np.sum(r_current) / frame_energy ** 1.2
        # energy_efficiency = total_d / total_e

        if np.sum(b_cluster_next) > 1e-10 and step < self.T_max - 1:
            reward = energy_efficiency
        elif np.sum(b_cluster_next) <= 1e-10 and c_current < 2:
            time_allocation = step - self.tic
            print('hovering_time_cluster1: %d' % time_allocation)
            c_current = c_current + 1
            self.tic = step
            self.G = [None]
            for k in range(UE[c_current]):
                l = list(itertools.combinations(np.arange(0, UE[c_current]), k + 1))
                m = [list(x) for x in l]
                self.G = self.G + m
            self.num_G = 2 ** UE[c_current]
            reward = energy_efficiency
        elif np.sum(b_cluster_next) <= 1e-10 and c_current == 2:
            time_allocation = step - self.tic
            print('hovering_time_cluster2: %d' % time_allocation)
            self.tic = 0
            done = 1
            reward = energy_efficiency
        elif step == self.T_max - 1:
            print('timeout')
            self.tic = 0
            done = 1
            reward = energy_efficiency

        c_next = c_current
        b_next = b_current
        s_ = np.append(np.reshape(h_next, [1, K * K]), b_next)
        s_ = np.append(s_, [c_next])
        s_ = s_[np.newaxis, :]
        return s_, reward, frame_energy, total_e, total_d, done
