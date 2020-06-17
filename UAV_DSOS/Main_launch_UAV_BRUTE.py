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
import itertools
import Data_generation as dg
import OFFLINE_core.Optimizer_BLP_gurobi as sol

N, K, K_n, B, T_max, P_max = 3, 9, 3, 1, 80, 1
UE = np.array([2, 3, 2])
var_noise = 0.1

np.random.seed(1364)
H_sequential = np.zeros([T_max, K, K])
Demands = np.zeros([K_n, N])
Demands_cluster = np.zeros(N)
for n in range(N):
    H_sequential[0, K_n * n: K_n * n + UE[n], K_n * n: K_n * n + UE[n]] = np.random.randint(1, 5, size=[UE[n], UE[n]])
    Demands[0: UE[n], n] = dg.demands_generation(UE[n], np.random.randint(1, 5, size=[1, UE[n]]))
    Demands_cluster[n] = np.sum(Demands[:, n])
t_max_cluster = np.round(T_max * Demands_cluster / np.sum(Demands_cluster))
print(t_max_cluster)
if np.sum(t_max_cluster) > T_max:
    t_max_cluster[-1] = t_max_cluster[-1] - (np.sum(t_max_cluster) - T_max)
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

count_non = 0

G = [None]
for k in range(K_n):
    l = list(itertools.combinations(np.arange(0, K_n), k + 1))
    m = [list(x) for x in l]
    G = G + m
G_size = len(G)

#t1 = time.time()
e = np.zeros([T_max, G_size, K_n, N])
r = np.zeros([T_max, G_size, K_n, N])
for t in range(T_max):
    for n in range(N):
        for g in range(G_size):
            if g == 0:
                r[t, g, :, n] = np.zeros(K_n)
                e[t, g, :, n] = np.zeros(K_n)
            else:
                beta = H_sequential[t, K_n * n: K_n * (n+1), K_n * n: K_n * (n+1)]
                for k_u in range(K_n):
                    if (beta[k_u, k_u] < 0.001) and (k_u in G[g]):
                        e[t, g, :, n] = 0
                        r[t, g, :, n] = 0
                    elif k_u in G[g]:
                        s = var_noise
                        for i_u in range(K_n):
                            if i_u != k_u and i_u in G[g]:
                                s = s + beta[k_u, i_u] * P_max
                        e[t, g, k_u, n] = beta[k_u, k_u] * P_max * math.log2(len(G[g]) + 1) * 1  # power_discount
                        r[t, g, k_u, n] = B * math.log2(1 + beta[k_u, k_u] * P_max / s)
                    else:
                        e[t, g, k_u, n] = 0
                        r[t, g, k_u, n] = 0

e_cn = np.sum(e, axis=2)
d_n = r * 4

feasible_obj = []
feasible_sol = []
for t0 in range(10, T_max):
    comm_energy_0 = sol.BLP_solver(t0, G_size, e_cn[0:t0, :, 0], d_n[0:t0, :, :, 0], K_n, Demands[:,0])
    if comm_energy_0 is None:
        continue
    remain0 = T_max - t0
    for t1 in range(15, remain0):
        comm_energy_1 = sol.BLP_solver(t1, G_size, e_cn[t0:t0+t1, :, 1], d_n[t0:t0+t1, :, :, 1], K_n, Demands[:,1])
        if comm_energy_1 is None:
            continue
        remain1 = remain0 - t1
        for t2 in range(10, remain1):
            comm_energy_2 = sol.BLP_solver(t2, G_size, e_cn[t0+t1:t0+t1+t2, :, 2], d_n[t0+t1:t0+t1+t2, :, :, 2], K_n, Demands[:,2])
            if comm_energy_2 is None:
                continue
            remain2 = remain0 - t2
            obj = comm_energy_0 + comm_energy_1 + comm_energy_2 + 2 * (t0 + t1 + t2)
            time_allocation = [t0, t1, t2]
            feasible_obj.append(obj)
            feasible_sol.append(time_allocation)

print('min_energy:', min(feasible_obj))
print('time_allocation:', feasible_sol[feasible_obj.index(min(feasible_obj))])
