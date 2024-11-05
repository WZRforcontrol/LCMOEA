import numpy as np
from LCMOEA.population import Population
from tqdm import tqdm
class ClusteringAidedSelection:
    def __init__(self, P, Q, problem, w):
        """
        环境选择过程，基于聚类的方法
        :param P: 父代种群
        :param Q: 子代种群
        :param problem: 问题实例
        :param w: 权重参数
        """
        self.P = P
        self.Q = Q
        self.problem = problem
        self.w = w
        self.N = len(P)  # 种群规模

    def compute_theta(self, v1, v2):
        dot_product = np.abs(np.dot(v1, v2))
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return theta

    def select(self):
        """
        执行聚类辅助的环境选择过程
        :return: 更新后的种群 P_new
        """
        # 合并 P 和 Q，形成 U
        U = Population(self.problem, 0)
        U.merge(self.P)
        U.merge(self.Q)

        # 计算 U 中所有解的目标值
        U.compute_obj_val()

        # 归一化目标值 F'(x)
        U.normalize_objectives()
        normalized_objectives = U.norm_obj

        # 计算 U 中所有解的约束违反程度
        U.compute_constraint_violations()
        constraint_violations = U.constr_violations

        # 初始化簇 c = (c1, ..., c2N)，每个解 xi ∈ U 都是一个簇 ci
        clusters = []
        for i in range(len(U)):
            cluster = {'indices': [i], 'centroid': normalized_objectives[i]} # 簇的索引和中心
            clusters.append(cluster)

        # 进行 N 轮层次聚类
        for _ in tqdm(range(self.N), desc=f"Clustering"):
            # 获取非空簇的索引
            cluster_indices = [i for i in range(len(clusters)) if len(clusters[i]['indices']) > 0] # 非空簇的索引, 初始为0,1,2,...,2N
            num_clusters = len(cluster_indices)
            if num_clusters <= self.N: # 簇的数量小于等于 N，停止聚类，控制簇的数量
                break

            # 计算所有簇之间的 θ
            min_theta = None
            pair_to_merge = None # 待合并的簇对u, h
            for idx_u in range(len(cluster_indices)):
                i = cluster_indices[idx_u]
                ci = clusters[i]
                for idx_h in range(idx_u + 1, len(cluster_indices)):
                    j = cluster_indices[idx_h]
                    cj = clusters[j]
                    # 计算簇 ci 和 cj 的 θ
                    theta = self.compute_theta(ci['centroid'], cj['centroid'])
                    if min_theta is None or theta < min_theta: # 更新最小的θ值和待合并的簇对
                        min_theta = theta
                        pair_to_merge = (i, j)

            # 合并簇
            if pair_to_merge is not None:
                u, h = pair_to_merge
                cu = clusters[u]
                ch = clusters[h]
                # 合并簇 c^u = c^u + c^h
                cu['indices'].extend(ch['indices'])
                # 更新簇中心 cc^u
                indices = cu['indices']
                ccu = np.mean(normalized_objectives[indices], axis=0)
                cu['centroid'] = ccu
                # 清空簇 ch
                ch['indices'] = []
                ch['centroid'] = None
            else:
                # 没有可合并的簇
                break

        # 从每个非空簇中选择代表解，形成更新后的种群 P_new
        P_new = Population(self.problem, 0)

        for cluster in tqdm(clusters, desc="Selecting"):
            if len(cluster['indices']) == 0:
                continue
            indices = cluster['indices']
            solutions = U.pop[indices]
            c_violations = constraint_violations[indices]
            # 判断簇中解的可行性
            feasible = c_violations == 0
            num_feasible = np.sum(feasible)
            if num_feasible == len(solutions):
                # 所有解均可行，选择目标性能最好的解
                sum_norm_obj = np.sum(normalized_objectives[indices], axis=1)
                best_index = np.argmin(sum_norm_obj)
                x_best = solutions[best_index]
            elif num_feasible == 0:
                # 所有解均不可行，使用综合指标 CI(x)
                sum_norm_obj = np.sum(normalized_objectives[indices], axis=1) # 按行求和
                rank_obj = np.argsort(sum_norm_obj) # 按照目标值排序
                rank_cv = np.argsort(c_violations) # 按照约束违反程度排序
                rank_obj_positions = np.empty_like(rank_obj) # 创建一个和rank_obj形状相同的数组
                rank_cv_positions = np.empty_like(rank_cv) # 创建一个和rank_cv形状相同的数组
                rank_obj_positions[rank_obj] = np.arange(len(solutions)) # 按照目标值排序的索引
                rank_cv_positions[rank_cv] = np.arange(len(solutions)) # 按照约束违反程度排序的索引
                CI = self.w * rank_obj_positions + (1 - self.w) * rank_cv_positions # 综合指标
                best_index = np.argmin(CI) # 选择综合指标最小的解
                x_best = solutions[best_index] # 最优解
            else:
                # 可行和不可行解的混合，使用 CI(x)
                sum_norm_obj = np.sum(normalized_objectives[indices], axis=1)
                rank_obj = np.argsort(sum_norm_obj)
                rank_cv = np.argsort(c_violations)
                rank_obj_positions = np.empty_like(rank_obj)
                rank_cv_positions = np.empty_like(rank_cv)
                rank_obj_positions[rank_obj] = np.arange(len(solutions))
                rank_cv_positions[rank_cv] = np.arange(len(solutions))
                CI = self.w * rank_obj_positions + (1 - self.w) * rank_cv_positions
                best_index = np.argmin(CI)
                x_best = solutions[best_index]
            P_new.append_pop(x_best)

        return P_new