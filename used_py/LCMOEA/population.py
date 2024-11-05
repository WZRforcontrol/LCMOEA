import numpy as np
import sympy
# from pyDOE import lhs

def is_prime(n):
    """
    判断一个数是否为素数
    :param n: 待判断的数
    :return: 是否为素数
    """
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

class Population:

    def __init__(self, problem, N_init):
        self.problem = problem
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.N_init = N_init
        self.pop = np.empty((0, self.dim))  # 存储种群中的个体
        self.objectives = problem.objectives
        self.equality_constraints = problem.equality_constraints
        self.inequality_constraints = problem.inequality_constraints
        self.num_obj = len(self.objectives)
        self.constraint_violations = None  # 约束违反程度
        self.obj_val = None  # 目标值
        self.norm_obj = None  # 归一化目标值
        self.norm_pop = None  # 归一化种群
        self.constr_violations = None  # 约束违反程度

    def __len__(self):
        return len(self.pop)  # 返回种群中个体的数量

    def __iter__(self):
        return iter(self.pop)  # 返回种群的迭代器

    def append_pop(self, new_individual):
        """
        向种群中添加一个新的个体
        :param new_individual: 新的个体，形状应为 (self.dim,)
        """
        new_individual = np.atleast_2d(new_individual)
        if new_individual.shape[1] != self.dim:
            raise ValueError(f"new_individual 的形状应为 (self.dim,)")
        
        self.pop = np.vstack([self.pop, new_individual])  # 向种群中添加一个新的个体

    def gps_init(self):
        # """
        # 通过佳点集初始化策略初始化第一个搜索代理种群
        # :return: 初始化的种群
        # """
        
        # population = np.zeros((self.N_init, self.dim))
        # prime_number_min = self.dim * 2 + 3

        # # 找到 (prime_number_min - 3) / 2 >= dim 的最小素数 prime_number_min
        # while True:
        #     if is_prime(prime_number_min):
        #         break
        #     else:
        #         prime_number_min += 1

        # for i in range(self.N_init):
        #     for j in range(self.dim):
        #         r = (2 * np.cos(2 * np.pi * (j+1) / prime_number_min) * (i+1) ) % 1  # 对应维度的 r
        #         population[i, j] = self.lb[j] + r * (self.ub[j] - self.lb[j])
        
        # 生成 m x d 的矩阵，其中每一行都是 [1, 2, ..., m]
        temp1 = np.arange(1, self.N_init + 1).reshape(-1, 1) * np.ones((1, self.dim))
        
        # 生成 [1, 2, ..., d]
        ind = np.arange(1, self.dim + 1)
        
        # 生成素数列表，范围是 [0, 100 * d]
        prime = list(sympy.sieve.primerange(0, 100 * self.dim))
        
        # 找到第一个大于等于 (2 * d + 3) 的素数的索引
        idx = np.where(np.array(prime) >= (2 * self.dim + 3))[0]
        
        # 计算 temp2
        temp2 = (2 * np.pi * ind) / prime[idx[1]]
        temp2 = 2 * np.cos(temp2)
        temp2 = np.ones((self.N_init, 1)) * temp2
        
        # 计算佳点集
        gd = temp1 * temp2
        gd = np.mod(gd, 1)
        
        # 将佳点集映射到 [lb, ub] 范围内
        population = self.lb + gd * (self.ub - self.lb)
        
        # """
        # 通过拉丁超立方采样（LHS）初始化第一个搜索代理种群
        # """
        # # 使用 LHS 生成均匀分布的样本点
        # lhs_samples = lhs(self.dim, samples=self.N_init)
        
        # # 将样本点映射到决策变量的上下界范围内
        # population = self.lb + lhs_samples * (self.ub - self.lb)

        # # 使用 numpy 生成均匀分布的样本点
        # uniform_samples = np.random.uniform(0, 1, (self.N_init, self.dim))
        
        # # 将样本点映射到决策变量的上下界范围内
        # population = self.lb + uniform_samples * (self.ub - self.lb)
 

        self.pop = population

    def merge(self, other_population):
        """
        合并两个种群
        :param other_population: 另一个种群
        """
        if not isinstance(other_population, Population):
            raise ValueError("other_population 必须是 Population 类型")
        self.extend(other_population.pop)

    def extend(self, new_individuals):
        """
        扩展种群，加入新的个体
        :param new_individuals: 新的个体，形状应为 (n, self.dim)
        """
        new_individuals = np.atleast_2d(new_individuals)
        if new_individuals.shape[1] != self.dim:
            raise ValueError(f"new_individuals 的形状应为 (n, {self.dim})")
        
        self.pop = np.vstack([self.pop, new_individuals])  # 扩展种群，加入新的个体

    def norm_sol(self):
        '''
        归一化种群中的所有解
        '''
        self.norm_pop = (self.pop - self.lb) / (self.ub - self.lb + 1e-8)  # N x dim
        
    def denorm_sol(self):
        '''
        反归一化种群中的所有解
        '''
        if self.norm_pop is None:
            raise ValueError("归一化种群不存在，请先调用 norm_sol 方法。")
        self.pop = self.norm_pop * (self.ub - self.lb + 1e-8) + self.lb  # N x dim

    def compute_obj_val(self):
        '''
        计算种群中所有解的目标值
        '''
        obj_val = np.zeros((len(self), self.num_obj))
        for i in range(len(self)):
            x = self.pop[i]
            for j, obj_func in enumerate(self.objectives):
                obj_val[i, j] = obj_func(x)
        self.obj_val = obj_val  # N x num_obj

    def normalize_objectives(self):
        '''
        归一化目标值F'(x)
        '''
        z_min = np.min(self.obj_val, axis=0)  # num_obj
        z_max = np.max(self.obj_val, axis=0)  # num_obj

        self.norm_obj = (self.obj_val - z_min) / (z_max - z_min + 1e-8)  # N x num_obj

    def compute_constraint_violations(self):
        '''
        计算种群中所有解的约束违反程度
        分别处理等式约束 h(x) == 0 和不等式约束 g(x) <= 0
        '''
        constraint_violations = np.zeros(len(self))
        for i in range(len(self)):
            x = self.pop[i]
            total_violation = 0.0
            # # 检查是否在上下界范围内，并计算偏离程度
            # if np.any(x < self.lb) or np.any(x > self.ub):
            #     total_violation += 1e6
            lower_violation = np.sum(np.maximum(0, self.lb - x))
            upper_violation = np.sum(np.maximum(0, x - self.ub))
            total_violation += 2*lower_violation + 2*upper_violation
            # 处理等式约束 h(x) == 0
            for cons_func in self.equality_constraints:
                violation = abs(cons_func(x))  # 绝对值表示违反程度
                total_violation += violation
            # 处理不等式约束 g(x) <= 0
            for cons_func in self.inequality_constraints:
                violation = cons_func(x)
                total_violation += max(0, violation)  # 违反程度为正值
            constraint_violations[i] = total_violation
        self.constr_violations = constraint_violations  # N