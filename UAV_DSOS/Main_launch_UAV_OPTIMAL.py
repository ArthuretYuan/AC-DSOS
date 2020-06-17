# -*- coding: utf-8 -*-
import numpy as np
import math
import itertools
import Data_generation as dg
import OFFLINE_core.Optimizer_OPT_gurobi as sol

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

optimal_energy = sol.BLP_solver(T_max, G_size, N, e_cn, d_n, K_n, Demands)
print("optimal_energy=", optimal_energy)
