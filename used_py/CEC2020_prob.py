import numpy as np

class CMOP2Problem:
    def __init__(self, D=25):
        """
        初始化CMOP2问题实例。

        参数：
        - D: 决策变量的维度，默认为25。
        """
        self.dim = D  # 决策变量维度
        self.UBLB = np.array([1.1, 0])
        self.objectives = [self.objective1, self.objective2]  # 目标函数列表
        self.inequality_constraints = [self.constraint]  # 不等式约束列表
        self.equality_constraints = []  # 等式约束列表（此问题中为空）
        self.www = np.array([1, 1])  # 目标函数的权重向量
    
    def g_func(self, x):
        """
        计算辅助函数 g(x)。

        参数：
        - x: 决策变量向量。

        返回值：
        - g: 计算得到的 g(x) 值。
        """
        D = self.dim
        z = 1 - np.exp(-10 * (x[1:] - (np.arange(2, D+1) - 1)/D)**2)
        g = 1 + np.sum(1.5 + (0.1 / D) * z**2 - 1.5 * np.cos(2 * np.pi * z))
        return g

    def objective1(self, x):
        """
        计算第一个目标函数值。

        参数：
        - x: 决策变量向量。

        返回值：
        - f1: 第一个目标函数值。
        """
        g = self.g_func(x)
        return g * x[0]
    
    def objective2(self, x):
        """
        计算第二个目标函数值。

        参数：
        - x: 决策变量向量。

        返回值：
        - f2: 第二个目标函数值。
        """
        g = self.g_func(x)
        f1 = self.objective1(x)
        value = 1.1**2 - (f1 / g)**2
        if value < 0:
            value = 0  # 防止负数
        return g * np.sqrt(value)
    
    def LA(self, A, B, C, D_param, theta):
        """
        计算约束条件中的 LA 函数。

        参数：
        - A, B, C, D_param: LA 函数的参数。
        - theta: 角度参数。

        返回值：
        - LA 函数的计算结果。
        """
        t = theta ** C
        return A * np.cos(B * t) ** D_param
    
    def constraint(self, x):
        """
        计算约束条件值。

        参数：
        - x: 决策变量向量。

        返回值：
        - constraint_value: 约束条件的计算结果，应满足 constraint_value <= 0。
        """
        f1 = self.objective1(x)
        f2 = self.objective2(x)
        theta = np.arctan2(f2, f1)
        term1 = (f1**2) / (1 + self.LA(0.15, 6, 4, 10, theta))**2
        term2 = (f2**2) / (1 + self.LA(0.75, 6, 4, 10, theta))**2
        return term1 + term2 - 1


class CMOP3Problem:
    def __init__(self, D=25):
        """
        Initialize an instance of the CMOP3 problem.

        Parameters:
        - D: The dimension of decision variables, default is 25.
        """
        self.dim = D  # Dimension of decision variables
        self.lower_bounds = np.zeros(self.dim)  # Lower bounds for decision variables
        self.upper_bounds = np.ones(self.dim)   # Upper bounds for decision variables
        self.UBLB = np.array([1, 0])
        self.objectives = [self.objective1, self.objective2]  # List of objective functions
        self.inequality_constraints = [self.constraint1, self.constraint2, self.constraint3]  # List of inequality constraints
        self.equality_constraints = []  # List of equality constraints (empty for this problem)
        self.www = np.array([1, 1])  # Weights for the objective functions

    def g_func(self, x):
        """
        Compute the auxiliary function g(x).

        Parameters:
        - x: Decision variable vector.

        Returns:
        - g: The computed value of g(x).
        """
        D = self.dim
        M = 2  # Number of objectives
        z = 1 - np.exp(-10 * (x[M-1:] - ((np.arange(M, D+1) - 1) / D))**2)
        g = 1 + np.sum(1.5 + (0.1 / D) * z**2 - 1.5 * np.cos(2 * np.pi * z))
        return g

    def objective1(self, x):
        """
        Compute the first objective function value.

        Parameters:
        - x: Decision variable vector.

        Returns:
        - f1: The value of the first objective function.
        """
        g = self.g_func(x)
        return g * x[0] ** self.dim

    def objective2(self, x):
        """
        Compute the second objective function value.

        Parameters:
        - x: Decision variable vector.

        Returns:
        - f2: The value of the second objective function.
        """
        g = self.g_func(x)
        f1 = self.objective1(x)
        return g * (1 - (f1 / g) ** 2)

    def constraint1(self, x):
        """
        Compute the first constraint value.

        Parameters:
        - x: Decision variable vector.

        Returns:
        - constraint_value: The value of the first constraint (should be <= 0).
        """
        f1 = self.objective1(x)
        f2 = self.objective2(x)
        return -1 * (2 - 4 * f1 ** 2 - f2) * (2 - 8 * f1 ** 2 - f2)

    def constraint2(self, x):
        """
        Compute the second constraint value.

        Parameters:
        - x: Decision variable vector.

        Returns:
        - constraint_value: The value of the second constraint (should be <= 0).
        """
        f1 = self.objective1(x)
        f2 = self.objective2(x)
        return (2 - 2 * f1 ** 2 - f2) * (2 - 16 * f1 ** 2 - f2)

    def constraint3(self, x):
        """
        Compute the third constraint value.

        Parameters:
        - x: Decision variable vector.

        Returns:
        - constraint_value: The value of the third constraint (should be <= 0).
        """
        f1 = self.objective1(x)
        f2 = self.objective2(x)
        return (1 - f1 ** 2 - f2) * (1.2 - 1.2 * f1 ** 2 - f2)
