'''
Description:
function BLP_solver:   the solver for binary linear programming problem (by gurobi)
function LP_solver:    the solver for linear programming problem (by gurobi)
function DPro_solver:  the solver for binary linear programming problem (by dynamic programming)
function FPTAS_solver: the solver for binary linear programming problem (by TPTAS)
'''

import numpy as np
import gurobipy

def BLP_solver(t_n, G_size, e_cn, d_n, K_n, Demands):

    MODEL = gurobipy.Model("Energy_min_BLP")
    MODEL.setParam('LogToConsole', 0)  # decativate processing log
    MODEL.setParam('NonConvex', 2)

    x = MODEL.addVars(t_n, G_size, vtype=gurobipy.GRB.BINARY, name='x')  # create variables
    MODEL.update()  # update variables' environment
    MODEL.setObjective(gurobipy.quicksum(e_cn[i, g] * x[i, g] for i in range(t_n) for g in range(G_size)), gurobipy.GRB.MINIMIZE)
    for k in range(K_n):
        MODEL.addConstr(gurobipy.quicksum(d_n[i, g, k] * x[i, g] for i in range(t_n) for g in range(G_size)) >= Demands[k]*1000)
    for i in range(t_n):
        MODEL.addConstr(gurobipy.quicksum(x[i, g] for g in range(G_size)) <= 1)
    MODEL.optimize()  # operate BLP model
    #print(MODEL.getVars()) # output the optimal decision
    if MODEL.SolCount == 0:
        comm_energy = None
    else:
        comm_energy = MODEL.objVal
    return comm_energy

def LP_solver(t_n, G_size, e_cn, d_n, K_n, Demands):

    MODEL = gurobipy.Model("Energy_min_LP")
    MODEL.setParam('LogToConsole', 0) # decativate processing log

    Demands, d_n = Demands.astype(np.int32), d_n.astype(np.int32)
    x = MODEL.addVars(t_n, G_size, lb=0.0, ub=1.0, vtype=gurobipy.GRB.CONTINUOUS, name='x')
    MODEL.update()
    MODEL.setObjective(gurobipy.quicksum(e_cn[i, g] * x[i, g] for i in range(t_n) for g in range(G_size)), gurobipy.GRB.MINIMIZE)
    for k in range(K_n):
        MODEL.addConstr(gurobipy.quicksum(d_n[i, g, k] * x[i, g] for i in range(t_n) for g in range(G_size)) >= Demands[k]*1000)
    for i in range(t_n):
        MODEL.addConstr(gurobipy.quicksum(x[i, g] for g in range(G_size)) <= 1)
    MODEL.optimize()
    #print(MODEL.getVars()) # output the optimal decision
    if MODEL.SolCount == 0:
        comm_energy = None
    else:
        comm_energy = MODEL.objVal
    return comm_energy

def DPro_solver(t_n, G_size, e_cn, d_n, K_n, Demands):

    E = np.reshape(e_cn[0:t_n, :], [t_n, G_size])  # energy consumed for each item (Value)
    Q = np.reshape(Demands[0: K_n], [K_n, 1])*1000  # minimum demands for each knapsack (Weight limit)
    R = np.reshape(d_n[0:t_n, :, :], [t_n, G_size, K_n])  # (Weight for each item)

    Q,R = Q.astype(np.int32), R.astype(np.int32)
    Q = np.squeeze(Q)
    Q = Q.tolist()
    Shape = [t_n] + Q

    E_min = 1000 * np.ones(Shape)
    if K_n == 2:
        E_min[:, 0, 0] = np.zeros([t_n,])
        for i in range(t_n-1):  # for each item
            alarm = 0
            for q1 in range(Q[0])[::-1]:
                for q2 in range(Q[1])[::-1]:
                    E_min_g = E_min[i][q1][q2]
                    for g in range(G_size):
                        x1 = int(np.maximum(q1 - R[i, g, 0], 0))
                        x2 = int(np.maximum(q2 - R[i, g, 1], 0))
                        E_min_g = np.minimum(E_min_g, E_min[i][x1][x2] + E[i, g])
                    E_min[i + 1][q1][q2] = E_min_g
                    if E_min[i + 1][q1][q2] < 999 and alarm == 0:
                        alarm = 1
    if K_n == 3:
        E_min[0][0][0][0] = 0
        for g in range(G_size):  # for each item
            for q1 in range(R[i, g, 0]+1):
                for q2 in range(R[i, g, 1]+1):
                    for q3 in range(R[i, g, 2] + 1):
                        E_min[g + 1][q1][q2][q3] = np.minimum(E_min[g][q1][q2][q3], E[i, g])
            for q1 in range(R[i, g, 0]+1, Q[0]):
                for q2 in range(R[i, g, 1]+1, Q[1]):
                    for q3 in range(R[i, g, 2]+1, Q[2]):
                        E_min[g + 1][q1][q2][q3] = np.minimum(E_min[g][q1][q2][q3], E_min[g][q1 - R[i, g, 0]][q2 - R[i, g, 1]][q3 - R[i, g, 2]] + E[i, g])

def FPTAS_solver(t_n, G_size, e_cn, d_n, K_n, Demands):
    E = np.reshape(e_cn[0:t_n, :], [t_n, G_size])  # energy consumed for each item (Value)
    Q = np.reshape(Demands[0: K_n], [K_n, 1]) * 1000  # minimum demands for each knapsack (Weight limit)
    R = np.reshape(d_n[0:t_n, :, :], [t_n, G_size, K_n])  # (Weight for each item)

    b = np.zeros(K_n)
    eps = 0.0001
    for k in range(K_n):
        # b[k] = np.maximum(np.floor(np.max(R[:, :, k])/((1+1/eps) * t_n * G_size)), 1)
        b[k] = 2
        R[:, :, k] = R[:, :, k]/b[k]
        Q[k, 0] = Q[k, 0]/b[k]

    Q, R = Q.astype(np.int32), R.astype(np.int32)
    Q = np.squeeze(Q)
    Q = Q.tolist()
    Shape = [t_n] + Q

    E_min = 1000 * np.ones(Shape)
    if K_n == 2:
        E_min[:, 0, 0] = np.zeros([t_n, ])
        for i in range(t_n - 1):  # for each item
            alarm = 0
            for q1 in range(Q[0])[::-1]:
                for q2 in range(Q[1])[::-1]:
                    E_min_g = E_min[i][q1][q2]
                    for g in range(G_size):
                        x1 = int(np.maximum(q1 - R[i, g, 0], 0))
                        x2 = int(np.maximum(q2 - R[i, g, 1], 0))
                        E_min_g = np.minimum(E_min_g, E_min[i][x1][x2] + E[i, g])
                    E_min[i + 1][q1][q2] = E_min_g
                    if E_min[i + 1][q1][q2] < 999 and alarm == 0:
                        alarm = 1
    if K_n == 3:
        E_min[0][0][0][0] = 0
        for g in range(G_size):  # for each item
            for q1 in range(R[i, g, 0] + 1):
                for q2 in range(R[i, g, 1] + 1):
                    for q3 in range(R[i, g, 2] + 1):
                        E_min[g + 1][q1][q2][q3] = np.minimum(E_min[g][q1][q2][q3], E[i, g])
            for q1 in range(R[i, g, 0] + 1, Q[0]):
                for q2 in range(R[i, g, 1] + 1, Q[1]):
                    for q3 in range(R[i, g, 2] + 1, Q[2]):
                        E_min[g + 1][q1][q2][q3] = np.minimum(E_min[g][q1][q2][q3],
                                                              E_min[g][q1 - R[i, g, 0]][q2 - R[i, g, 1]][q3 - R[i, g, 2]] + E[i, g])