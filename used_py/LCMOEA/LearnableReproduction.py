import torch
import numpy as np
from LCMOEA.population import Population
from tqdm import tqdm
class LearnableReproduction:
    def __init__(self, M1, M2, P, problem, alpha, r, device):
        """
        可学习的繁殖过程，生成子代种群 Q
        :param M1: 训练好的模型 M1
        :param M2: 训练好的模型 M2
        :param P: 父代种群实例
        :param problem: 问题实例
        :param alpha: 控制 y1 和 y2 对 xc 的影响程度的参数
        :param device: 设备
        """
        self.M1 = M1
        self.M2 = M2
        self.P = P.pop  # 从种群实例中获取种群数据
        self.problem = problem
        self.alpha = alpha 
        self.r = r
        
        self.lb = problem.lb  # 从问题实例中获取变量下界
        self.ub = problem.ub  # 从问题实例中获取变量上界
        self.dim = problem.dim  # 从问题实例中获取决策变量维度
        self.device = device
        self.N = len(P) # 种群规模

    def polynomial_mutation(self, x, eta=20):
        """
        多项式变异操作
        :param x: 需要变异的解
        :param eta: 变异分布指数
        :return: 变异后的解
        """
        x = np.copy(x)
        dim = len(x)
        for i in range(dim):
            if np.random.rand() <= 1.0 / dim:
                delta1 = (x[i] - self.lb[i]) / (self.ub[i] - self.lb[i])
                delta2 = (self.ub[i] - x[i]) / (self.ub[i] - self.lb[i])
                rand = np.random.rand()
                mut_pow = 1.0 / (eta + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    if val < 0:
                        val = 0  # 防止负数
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    if val < 0:
                        val = 0  # 防止负数
                    delta_q = 1.0 - val ** mut_pow
                x[i] = x[i] + delta_q * (self.ub[i] - self.lb[i])
                x[i] = np.clip(x[i], self.lb[i], self.ub[i])
        return x
    
    def denorm_y(self, y):
        """
        反归一化解 y
        :param y: 归一化解 y
        :return: 反归一化解 y
        """
        return y * (self.ub - self.lb + 1e-8) + self.lb

    def reproduce(self):
        """
        执行可学习的繁殖过程，生成子代种群 Q
        :return: 子代种群 Q
        """
        # 初始化子代种群 Q
        Q = Population(self.problem, 0)

        # 将 P 转换为张量
        P_tensor = torch.tensor(self.P, dtype=torch.float32).to(self.device)

        for i in tqdm(range(self.N), desc="Reproduction"):
            x = P_tensor[i]  # 当前解 x

            # 计算 y1 和 y2
            y1 = self.M1(x).detach().cpu().numpy()
            y1 = self.denorm_y(y1)
            # y1 = np.clip(y1, self.lb, self.ub)
            y2 = self.M2(x).detach().cpu().numpy()
            y2 = self.denorm_y(y2)
            # y2 = np.clip(y2, self.lb, self.ub)

            # 从 P 中随机选择两个不同的解 xd1 和 xd2，且 x ≠ xd1 ≠ xd2
            idxs = list(range(self.N))
            idxs.remove(i)
            xd_indices = np.random.choice(idxs, size=2, replace=False)
            xd1 = self.P[xd_indices[0]]
            xd2 = self.P[xd_indices[1]]

            # 计算子代解 xc
            # 第2项促进向UPF的快速收敛
            # 第3项加速向CPF的收敛
            # 最后一项增强搜索多样性以防止群体陷入局部最优
            
            xc = x.cpu().numpy() + \
                self.alpha * (y1 - x.cpu().numpy()) + \
                (0.8 - self.alpha) * (y2 - x.cpu().numpy()) + \
                self.r * (xd1 - xd2)

            # 多项式变异
            xc = self.polynomial_mutation(xc)
            # xc_new = self.polynomial_mutation(xc)
            # xc_new = np.clip(xc_new, self.lb, self.ub)
            xc = np.clip(xc, self.lb, self.ub)
            
            # # 贪婪选择 
            # xc_new_obj = self.problem.evaluate(xc_new)
            # xc_obj = self.problem.evaluate(xc)
            # if xc_new_obj <= xc_obj:
            #     xc = xc_new

            # 将 xc 添加到子代种群 Q 中
            Q.append_pop(xc)

        return Q