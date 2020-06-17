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
H_sequential, Demands, Demands_cluster = np.zeros([T_max, K, K]), np.zeros([K_n, N]), np.zeros(N)
for n in range(N):
    H_sequential[0, K_n * n: K_n * n + UE[n], K_n * n: K_n * n + UE[n]] = np.random.randint(1, 5, size=[UE[n], UE[n]])
    Demands[0: UE[n], n] = dg.demands_generation(UE[n], np.random.randint(1, 5, size=[1, UE[n]]))
    Demands_cluster[n] = np.sum(Demands[:, n])
t_max_cluster = np.round(T_max * Demands_cluster / np.sum(Demands_cluster))
print("t_max=", t_max_cluster)
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
    print(h_next)
    t = t + 1

count_non = 0
for n in range(N):

    G = [None]
    for k in range(UE[n]):
        l = list(itertools.combinations(np.arange(0, UE[n]), k + 1))
        m = [list(x) for x in l]
        G = G + m
    G_size = len(G)

    #t1 = time.time()
    e = np.zeros([int(t_max_cluster[n]), G_size, UE[n]])
    r = np.zeros([int(t_max_cluster[n]), G_size, UE[n]])
    for t_temp in range(int(t_max_cluster[n])):
        if n > 0:
            t = t_temp + int(t_max_cluster[n-1])
        else:
            t = t_temp
        for g in range(G_size):
            if g == 0:
                r[t_temp, g, :] = np.zeros(UE[n])
                e[t_temp, g, :] = np.zeros(UE[n])
            else:
                beta = H_sequential[t, K_n * n: K_n * n + UE[n], K_n * n: K_n * n + UE[n]]
                for k_u in range(UE[n]):
                    if k_u in G[g]:
                        s = var_noise
                        for i_u in range(K):
                            if i_u != k_u and i_u in G[g]:
                                s = s + beta[k_u, i_u] * P_max
                        e[t_temp, g, k_u] = beta[k_u, k_u] * P_max * math.log2(len(G[g]) + 1) * 1  # power_discount
                        r[t_temp, g, k_u] = B * math.log2(1 + beta[k_u, k_u] * P_max / s)
                    else:
                        e[t_temp, g, k_u] = 0
                        r[t_temp, g, k_u] = 0

    e_cn, d_n, tau, p_h = np.sum(e, axis=2), r * 4, 0.618, 2
    comm_energy_test = sol.BLP_solver(int(t_max_cluster[n]), G_size, e_cn, d_n, UE[n], Demands[:, n])
    t_min_cluster = 0
    comm_energy_test = None
    while comm_energy_test is None:
        t_min_cluster = t_min_cluster + 1
        comm_energy_test = sol.BLP_solver(t_min_cluster, G_size, e_cn, d_n, UE[n], Demands[:, n])

    print("e=", comm_energy_test)
    print("t=", t_min_cluster)

    a, b = t_min_cluster, int(t_max_cluster[n])
    u, v = math.ceil(b - tau * (b - a)), math.ceil(a + tau * (b - a))
    while np.abs(b - a) != 1:
        comm_energy_u = sol.BLP_solver(u, G_size, e_cn, d_n, UE[n], Demands[:, n])
        comm_energy_v = sol.BLP_solver(v, G_size, e_cn, d_n, UE[n], Demands[:, n])
        hove_energy_u = p_h * u
        hove_energy_v = p_h * v
        total_energy_u = comm_energy_u + hove_energy_u
        total_energy_v = comm_energy_v + hove_energy_v
        if total_energy_u < total_energy_v:
            b = v
            v = u
            u = math.ceil(b - tau * (b - a))
        else:
            a = u
            u = v
            v = math.ceil(b - tau * (b - a))
    comm_energy = comm_energy_v
    hove_energy = hove_energy_v
    total_energy = total_energy_v
    t_n = v
    print('cluster: %d' % n, 'time_allocation: %d' % t_n, 'communication_energy: %f' % comm_energy, 'hovering_energy: %f' % hove_energy, 'total_energy: %f' % total_energy)
