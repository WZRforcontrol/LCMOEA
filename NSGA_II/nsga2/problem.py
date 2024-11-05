from nsga2.individual import Individual
import random


class Problem:

    def __init__(self, objectives, num_of_variables, variables_range, expand=True, same_range=False):
        self.num_of_objectives = len(objectives)  # 目标函数的数量
        self.num_of_variables = num_of_variables  # 变量的数量
        self.objectives = objectives  # 目标函数列表
        self.expand = expand  # 是否展开目标函数的参数
        self.variables_range = []  # 变量的取值范围
        if same_range:
            # 如果所有变量的取值范围相同
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            # 如果变量的取值范围不同
            self.variables_range = variables_range

    def generate_individual(self):
        # 生成一个个体
        individual = Individual()
        # 随机生成个体的特征值
        individual.features = [random.uniform(*x) for x in self.variables_range]
        return individual

    def calculate_objectives(self, individual):
        # 计算个体的目标值
        if self.expand:
            # 如果展开参数
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            # 如果不展开参数
            individual.objectives = [f(individual.features) for f in self.objectives]