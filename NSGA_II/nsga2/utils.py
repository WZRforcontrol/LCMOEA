from nsga2.population import Population
import random

class NSGA2Utils:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):
        # 初始化NSGA2工具类
        self.problem = problem
        self.num_of_individuals = num_of_individuals  # 个体数量
        self.num_of_tour_particips = num_of_tour_particips  # 锦标赛参与者数量
        self.tournament_prob = tournament_prob  # 锦标赛概率
        self.crossover_param = crossover_param  # 交叉参数
        self.mutation_param = mutation_param  # 变异参数

    def create_initial_population(self):
        # 创建初始种群
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)
        return population

    def fast_nondominated_sort(self, population):
        # 快速非支配排序
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        # 计算拥挤距离
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    def crowding_operator(self, individual, other_individual):
        # 拥挤操作符
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        # 创建子代
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        # 交叉操作
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        num_of_features = len(child1.features)
        genes_indexes = range(num_of_features)
        for i in genes_indexes:
            beta = self.__get_beta()
            x1 = (individual1.features[i] + individual2.features[i]) / 2
            x2 = abs((individual1.features[i] - individual2.features[i]) / 2)
            child1.features[i] = x1 + beta * x2
            child2.features[i] = x1 - beta * x2
        return child1, child2

    def __get_beta(self):
        # 获取beta值
        u = random.random()
        if u <= 0.5:
            return (2 * u) ** (1 / (self.crossover_param + 1))
        return (2 * (1 - u)) ** (-1 / (self.crossover_param + 1))

    def __mutate(self, child):
        # 变异操作
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            u, delta = self.__get_delta()
            if u < 0.5:
                child.features[gene] += delta * (child.features[gene] - self.problem.variables_range[gene][0])
            else:
                child.features[gene] += delta * (self.problem.variables_range[gene][1] - child.features[gene])
            if child.features[gene] < self.problem.variables_range[gene][0]:
                child.features[gene] = self.problem.variables_range[gene][0]
            elif child.features[gene] > self.problem.variables_range[gene][1]:
                child.features[gene] = self.problem.variables_range[gene][1]

    def __get_delta(self):
        # 获取delta值
        u = random.random()
        if u < 0.5:
            return u, (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        return u, 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))

    def __tournament(self, population):
        # 锦标赛选择
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        # 根据概率选择
        if random.random() <= prob:
            return True
        return False