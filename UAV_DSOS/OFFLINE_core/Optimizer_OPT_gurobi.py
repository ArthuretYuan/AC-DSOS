
import numpy as np
import gurobipy

def BLP_solver(T_L, G_size, N, e_cn, d_n, K_n, Demands):
    MODEL = gurobipy.Model("Energy_min_BLP")
    MODEL.setParam('LogToConsole', 0)  # decativate processing log
    MODEL.setParam('NonConvex', 2)

    x = MODEL.addVars(T_L, G_size, N, lb=0.0, ub=1.0, vtype=gurobipy.GRB.BINARY, name='x')
    v = MODEL.addVars(T_L, N + 1, lb=0.0, ub=1.0, vtype=gurobipy.GRB.BINARY, name='v')
    MODEL.update()
    MODEL.setObjective(gurobipy.quicksum(gurobipy.quicksum(v[i, n] * e_cn[i, g, n] * x[i, g, n] for i in range(T_L) for n in range(N)) for g in range(G_size))
                       + gurobipy.quicksum(v[i, n] * 2 for i in range(T_L) for n in range(N)), gurobipy.GRB.MINIMIZE)

    for n in range(N):
        for k in range(K_n):
            MODEL.addConstr(gurobipy.quicksum(v[i, n] * d_n[i, g, k, n] * x[i, g, n] for i in range(T_L) for g in range(G_size)) >= Demands[k, n]*1000)
    for i in range(T_L):
        MODEL.addConstr(gurobipy.quicksum(x[i, g, n] for g in range(G_size) for n in range(N)) <= 1)
    for i in range(T_L):
        MODEL.addConstr(gurobipy.quicksum(v[i, n] for n in range(N + 1)) <= 1)

    MODEL.addConstr(v[0, 0] == 1)
    for i in range(0, T_L - 1):
        MODEL.addConstr(v[i, 0] <= 10 * (v[i + 1, 0] + v[i + 1, 1]))
        MODEL.addConstr(v[i, 1] <= 10 * (v[i + 1, 1] + v[i + 1, 2]))
        MODEL.addConstr(v[i, 2] <= 10 * (v[i + 1, 2] + v[i + 1, 3]))
        MODEL.addConstr(v[i, 3] <= 10 * v[i + 1, 3])

    MODEL.optimize()
    #print(MODEL.getVars())
    time_allocation = np.zeros([1, N])
    for n in range(N):
        for i in range(T_L):
            time_allocation[0, n] = time_allocation[0, n] + v[i, n].X
    print(time_allocation)
    if MODEL.SolCount == 0:
        comm_energy = None
    else:
        comm_energy = MODEL.objVal
    return comm_energy