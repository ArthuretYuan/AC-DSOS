# -*- coding: utf-8 -*-
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

N, K, K_n, B, T_max, P_max = 3, 9, 3, 1, 3000, 1
UE = np.array([2, 3, 2])
var_noise = 0.1

np.random.seed(1364)
H_sequential, Demands, Demands_cluster = np.zeros([T_max, K, K]), np.zeros([K_n, N]), np.zeros(N)
for n in range(N):
    H_sequential[0, K_n * n: K_n * n + UE[n], K_n * n: K_n * n + UE[n]] = np.random.randint(1, 5, size=[UE[n], UE[n]])
    Demands[0: UE[n], n] = dg.demands_generation(UE[n], np.random.randint(1, 5, size=[1, UE[n]]))
    Demands_cluster[n] = np.sum(Demands[:, n])
H_sequential[0, :, :] = dg.beta_generation(K, H_sequential[0, :, :])

for t in range(T_max-1):
    h_current = H_sequential[t, :, :]
    h_next = np.zeros([K, K])
    for l in range(K):
        for k in range(K):
            if h_current[l, k] >= 0.1:
                some_list = [h_current[l, k], h_current[l, k] + 0.1, h_current[l, k] - 0.1]
                probabilities = [0.996, 0.002, 0.002]
                h_next[l, k] = np.clip(dg.random_pick(some_list, probabilities), 0, 1)
    H_sequential[t+1] = h_next
    t = t + 1
#---------------------------------------------------------------------------------

for n in range(N):
#while ep < 100:
    Num_u = UE[n]
    remain_demand = Demands[:, n] * 1000

    total_transmitted = np.zeros(Num_u)
    total_energy = 0
    ep = 0

    U_remain = np.arange(0, Num_u)
    U_remain = list(U_remain)
    satisfied = 1
    satisfied_list = []

    t1 = time.time()
    while satisfied:
        print("sat_list:", satisfied_list)
        for k in satisfied_list:
            if k in U_remain:
                U_remain.remove(k)


        #for k in range(len(satisfied_list)):
        #    U_remain.remove(satisfied_list[k])
        K_u = len(U_remain)
        print("remaining users:", U_remain)

        beta = H_sequential[ep, K_n * n: K_n * n + UE[n], K_n * n: K_n * n + UE[n]]


        if K_u > 2:
            sumkj = 1000
            for j in range(K_u):
                for k in range(K_u):
                    if k > j:
                        sumkj_temp = beta[j, k] + beta[k, j]
                        if sumkj_temp < sumkj:
                            sumkj = sumkj_temp
                            group = [j, k]
            print("group (U>2):", group)
        else:
            group = U_remain
            print("group (U<=2):", group)


        ep = ep + 1
        print("episode:", ep, "cluster:", n)

        e = np.zeros(UE[n])
        r = np.zeros(UE[n])

        for k_u in range(UE[n]):
            if k_u in group:
                s = var_noise
                for i_u in range(K):
                    if i_u != k_u and i_u in group:
                        s = s + beta[k_u, i_u] * P_max
                e[k_u] = beta[k_u, k_u] * P_max * math.log2(len(group) + 1) * 1  # power_discount
                r[k_u] = B * math.log2(1 + beta[k_u, k_u] * P_max / s)
                remain_demand[k_u] = remain_demand[k_u] - r[k_u]
                if remain_demand[k_u] < 0:
                    satisfied_list.append(k_u)
            else:
                e[k_u] = 0
                r[k_u] = 0
        if len(satisfied_list) == Num_u:
            satisfied = 0
        #print(np.sum(r, axis=1))
        #print(np.sum(e))
