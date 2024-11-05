import numpy as np

class Problem:

    def __init__(self, objectives, dim, UBLB, equality_constraints, inequality_constraints, www):
        self.objectives = objectives  # 目标函数列表
        self.num_obj = len(objectives)  # 目标函数的数量
        self.dim = dim  # 决策变量的数量
        if UBLB.shape[0] == 2 and UBLB.size == 2:
            self.lb = UBLB[1] * np.ones(dim)
            self.ub = UBLB[0] * np.ones(dim)
        elif UBLB.shape[0] == 2 and UBLB.shape[1] == dim:
            self.lb = UBLB[1, :]
            self.ub = UBLB[0, :]
        else:
            raise ValueError("UBLB 的形状不正确。应为 (2,dim) 或 (2, 1)。")
        self.equality_constraints = equality_constraints  # 等式约束条件列表
        self.inequality_constraints = inequality_constraints  # 不等式约束条件列表
        self.num_constr = len(equality_constraints) + len(inequality_constraints)  # 约束条件的数量
        # if UBLB.shape[0] == 2 and UBLB.size == 2:
        #     self.lb = UBLB[0] * np.ones(population.shape[1])
        #     self.ub = UBLB[1] * np.ones(population.shape[1])
        # elif UBLB.shape[0] == 2 and UBLB.shape[1] == population.shape[1]:
        #     self.lb = UBLB[0, :]
        #     self.ub = UBLB[1, :]
        # else:
        #     raise ValueError("UBLB 的形状不正确。应为 (2,dim) 或 (2, 1)。")
        self.www = www#平衡数量级
    
    def evaluate(self, x):
        obj_w = []
        for obj_fun in self.objectives:
            obj_w.append(obj_fun(x))
        obj_w = np.array(obj_w)
        obj_w = np.dot(obj_w, self.www)
        return obj_w

