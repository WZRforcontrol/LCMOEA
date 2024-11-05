class Population:

    def __init__(self):
        self.population = []  # 存储种群中的个体
        self.fronts = []  # 存储种群中的前沿

    def __len__(self):
        return len(self.population)  # 返回种群中个体的数量

    def __iter__(self):
        return self.population.__iter__()  # 返回种群的迭代器

    def extend(self, new_individuals):
        self.population.extend(new_individuals)  # 扩展种群，加入新的个体

    def append(self, new_individual):
        self.population.append(new_individual)  # 向种群中添加一个新的个体