from nsga2.utils import NSGA2Utils
from nsga2.population import Population
from tqdm import tqdm

class Evolution:

    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, crossover_param=2, mutation_param=5):
        # 初始化NSGA2工具类
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param,
                                mutation_param)
        self.population = None  # 初始种群为空
        self.num_of_generations = num_of_generations  # 进化的代数
        self.on_generation_finished = []  # 每代结束时的回调函数列表
        self.num_of_individuals = num_of_individuals  # 每代的个体数量

    def evolve(self):
        # 创建初始种群
        self.population = self.utils.create_initial_population()
        # 快速非支配排序
        self.utils.fast_nondominated_sort(self.population)
        # 计算每个前沿的拥挤距离
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        # 创建子代
        children = self.utils.create_children(self.population)
        returned_population = None
        # 进行多代进化
        for i in tqdm(range(self.num_of_generations)):
            # 将子代加入当前种群
            self.population.extend(children)
            # 快速非支配排序
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()  # 新的种群
            front_num = 0
            # 按前沿依次加入新的种群，直到达到个体数量限制
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            # 对当前前沿进行拥挤距离计算和排序
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            # 加入剩余的个体
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
            returned_population = self.population  # 保存当前种群
            self.population = new_population  # 更新种群
            # 快速非支配排序
            self.utils.fast_nondominated_sort(self.population)
            # 计算每个前沿的拥挤距离
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            # 创建新的子代
            children = self.utils.create_children(self.population)
        # 返回最优前沿
        return returned_population.fronts[0]