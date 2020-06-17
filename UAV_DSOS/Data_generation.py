"""
These codes are for generating data:
function random_pick:      selection based on fixed probability
channel_generation:        generate channel gain from a finite quantified set + MMSE precoding
beta_generation:           generate channel coefficients beta (after MMSE)
demands_generation:        generate users' demands from a finite quantified set
rayleigh_generation:       generate channel gain by rayleigh fading + MMSE precoding
"""

import numpy as np
np.set_printoptions(suppress=True)

def random_pick(some_list, probabilities):
    x = np.random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:break
    return item

def channel_generation(K, H_state):
    np.random.seed(333)
    channel_state_map = np.random.rand(5, 9) * 0.1\
                        + np.tile(np.array([2, 2.05, 2.1, 2.15, 2.25, 2.3, 2.35, 2.4, 2.45]), (5, 1)) + 1
    location_state_map = np.array([0.90, 0.92, 0.94, 0.96, 0.98])
    '''channel generation'''
    H = np.zeros([K, 5])
    for k in range(K):
        if H_state[k] == 0.0:
            H[k, :] = np.zeros([1, 5])
        else:
            H[k, :] = channel_state_map[:, int(H_state[k] - 1)]
    '''MMSE precoding'''
    H_conjh = H.conj().T
    H_temp = np.dot(H, H_conjh) + np.eye(K, K) * 0.1
    H_temp_inv = np.linalg.inv(H_temp)
    H_c = np.dot(H_conjh, H_temp_inv)
    beta = np.dot(H, H_c)
    beta = abs(beta) ** 2

    return H, beta

def beta_generation(K, Beta_state):
    '''beta state mapping'''
    beta_state_map = np.array([0, 0.6, 0.7, 0.8, 0.9, 1.0])
    beta = np.zeros([K, K])
    for i in range(K):
        for j in range(K):
            beta[i, j] = beta_state_map[int(Beta_state[i, j])]
    return beta

def demands_generation(K, Demands_state):
    '''demands state mapping'''
    demands_state_map = np.array([0, 0.1, 0.11, 0.12, 0.13, 0.14])
    '''beta generation'''
    demands = np.zeros([1, K])
    for i in range(K):
        demands[0, i] = demands_state_map[int(Demands_state[0, i])]
    return demands

def rayleigh_generation(K, H_state, dist_state):
    coefficient_state_map = np.array([1.31, 1.35, 1.39, 1.43, 1.47]) / np.sqrt(2)
    Var = 1e-8
    '''channel generation'''
    H = np.zeros([5, K])
    for l in range(5):
        for k in range(K):
            if H_state[l, k] == 0.0:
                H[l, k] = 0
            else:
                H[l, k] = coefficient_state_map[int(H_state[l, k] - 1)] * (dist_state[k]**(-3))/Var

    H = np.transpose(H)
    '''MMSE precoding'''
    H_conjh = H.conj().T
    H_temp = np.dot(H, H_conjh) + np.eye(K, K) * 0.1
    H_temp_inv = np.linalg.inv(H_temp)
    H_c = np.dot(H_conjh, H_temp_inv)
    beta = np.dot(H, H_c)
    beta = abs(beta) ** 2
    return H, beta

