class Individual(object):

    def __init__(self):
        self.rank = None  # 个体的等级
        self.crowding_distance = None  # 拥挤距离
        self.domination_count = None  # 支配计数
        self.dominated_solutions = None  # 被支配的解
        self.features = None  # 个体的特征
        self.objectives = None  # 个体的目标值

    def __eq__(self, other):
        # 判断两个个体是否相等，基于特征进行比较
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def dominates(self, other_individual):
        # 判断当前个体是否支配另一个个体
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)