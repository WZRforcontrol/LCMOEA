将下列代码转换成matlab的
LCMOEA_test.py
import torch
from LCMOEA_py import LCMOEA, problem as problem_module
from CEC2020_prob import CMOP2Problem

def main():
    # 创建问题实例
    probl = CMOP2Problem() 
    problem = problem_module.Problem(probl.objectives, probl.dim, probl.UBLB, probl.equality_constraints, probl.inequality_constraints, probl.www)
    # 设置参数
    N = 50  # 种群规模
    max_iter = 100000  # 最大评价次数
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 实例化 LCMOEA，设置横纵坐标范围
    # xlim = (0, 1.5)
    # ylim = (0, 1.5)
    # xlim = (0, 25)
    # ylim = (0, 25)
    # algorithm = LCMOEA.LCMOEA(problem, N, max_iter, device, True, xlim=xlim, ylim=ylim)
    
    # 运行算法
    algorithm = LCMOEA.LCMOEA(problem, N, max_iter, device, is_plot=True)
    pareto_front = algorithm.run()

    # # 输出结果
    # print("Pareto front solutions:")
    # for solution in pareto_front:
    #     print(solution)
    
if __name__ == "__main__":
    main()

CEC2020_prob.py
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
        return g * np.sqrt(1.1**2 - (f1 / g)**2)
    
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

ClusteringAidedSelection.py
import numpy as np
from LCMOEA_py.population import Population
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
LCMOEA.py
import numpy as np
from LCMOEA_py.problem import Problem
from LCMOEA_py.population import Population
from LCMOEA_py.MLP_Training import MLP_Training
from LCMOEA_py.LearnableReproduction import LearnableReproduction
from LCMOEA_py.ClusteringAidedSelection import ClusteringAidedSelection
import matplotlib.pyplot as plt
import os
import json
import random
import cv2

class LCMOEA:
    def __init__(self, probl, N, max_iter, device, 
                 hidden_sizes = [10], num_epochs = 100, learning_rate = 0.001, model_dir='models',
                 is_plot=False,image_dir='images/LCMOEA_img', video_dir = 'video', xlim=None, ylim=None):
        '''
        :param pro: 问题实例
        :param N: 种群规模
        :param max_iter: 最大迭代次数
        :param device: 设备
        :param is_plot: 是否绘制结果图
        :param model_dir: 模型保存目录
        :param xlim: 横坐标范围 (min, max)
        :param ylim: 纵坐标范围 (min, max)
        '''
        if not isinstance(probl, Problem):
            raise TypeError("probl 参数必须是 Problem 类的实例")
        self.probl = probl
        self.N = N
        self.device = device
        
        self.hidden_sizes = hidden_sizes
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
                  
        self.is_plot = is_plot
        self.image_dir = image_dir
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        self.video_dir = video_dir
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        self.xlim = xlim  # 横坐标范围
        self.ylim = ylim  # 纵坐标范围
            
        self.fronts = None  # 有约束pareto前沿
        self.FE_max = max_iter  # 最大函数评价次数
        self.avg_constraint_violations = []  # 存储每一代的平均约束违反值
        self.best_obj_values = []  # 存储每一代的最优目标值
        self.avg_obj_values = []  # 存储每一代的平均目标值
        


    def save_problem(self):
        '''保存问题实例'''
        problem_path = os.path.join(self.model_dir, 'problem.json')
        problem_data = {
            'lb': self.probl.lb.tolist(),
            'ub': self.probl.ub.tolist(),
            'num_obj': self.probl.num_obj,
            'num_constr': self.probl.num_constr
        }
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        with open(problem_path, 'w') as f:
            json.dump(problem_data, f)

    def load_problem(self):
        '''加载问题实例'''
        problem_path = os.path.join(self.model_dir, 'problem.json')
        if os.path.exists(problem_path):
            with open(problem_path, 'r') as f:
                problem_data = json.load(f)
            if (problem_data['lb'] == self.probl.lb.tolist() and
                problem_data['ub'] == self.probl.ub.tolist() and
                problem_data['num_obj'] == self.probl.num_obj and
                problem_data['num_constr'] == self.probl.num_constr):
                return True
        return False

    def run(self):
        
        # 1. 初始化种群
        P = Population(self.probl, self.N)  # 创建种群
        P.gps_init()  # 通过佳点集初始化策略初始化第一个搜索代理种群
        self.fronts = P
        if self.is_plot and self.probl.num_obj == 2:
            self.plot_pareto(0)

        # 2. 主循环
        FE = 0
        use_existing_models = self.load_problem()
        if use_existing_models:
            print("Using existing models.")
        else:
            self.save_problem()

        while FE <= self.FE_max:
            print(f"Evaluating counter {FE}")
            # 3. 训练模型 M1 和 M2 by Algorithm 2 with MLP_Training.py
            trainer = MLP_Training(P, self.probl, self.device, self.hidden_sizes, self.num_epochs, self.learning_rate,model_dir=self.model_dir, use_existing_models=use_existing_models)
            M1, M2 = trainer.train_models()
            
            # 4. 可学习的繁殖，生成子代种群 Q by Algorithm 3 with LearnableReproduction.py
            if FE <= 0.5*self.FE_max:
                alpha = 0.5
            else:
                # alpha = 0
                alpha = 0.5 - 0.5*(FE - 0.5*self.FE_max)/(0.5*self.FE_max)# 逐渐减小 alpha，也许更好
            reproducer = LearnableReproduction(M1, M2, P, self.probl, alpha, self.device)
            Q = reproducer.reproduce()
            
            # 5. 环境选择，选择下一代种群 P by Algorithm 4 with ClusteringAidedSelection.py
            # 计算 w 的值
            if FE < 0.4 * self.FE_max:
                w = 1.0
            elif FE > 0.6 * self.FE_max:
                w = 0.1
            else:
                w = -4.5 * FE / self.FE_max + 2.8
            Selection = ClusteringAidedSelection(P, Q, self.probl, w)
            P = Selection.select()
            self.fronts = P
            P.compute_constraint_violations()
            avg_violation = np.mean(P.constr_violations)
            self.avg_constraint_violations.append(avg_violation)
            
            # 计算每一代的最优目标值和平均目标值
            P.compute_obj_val()
            best_obj_values = np.min(P.obj_val, axis=0)
            avg_obj_values = np.mean(P.obj_val, axis=0)
            self.best_obj_values.append(best_obj_values)
            self.avg_obj_values.append(avg_obj_values)
            
            # 6. 更新函数评价次数 FE，假设每个子代解的评价计为一次 FE
            FE += len(P)

            # 保存每轮的图像
            if self.is_plot and self.probl.num_obj <= 2:
                self.plot_pareto(FE)
        
        # 输出有约束pareto前沿
        self.plot_pareto_end()

        # 生成视频
        if self.is_plot and self.probl.num_obj == 2:
            self.create_video()

        # 绘制平均约束违反值图像
        self.plot_avg_constraint_violations()

        # 绘制目标函数的最优解和平均值
        if self.is_plot and self.probl.num_obj <= 4:
            self.plot_obj_values()

        # 8. 输出有约束pareto前沿
        return P
    
    def plot_pareto(self, FE):
        '''
        绘制有约束pareto前沿
        '''
        if self.fronts is None:
            print("No fronts to plot.")
            return
        
        self.fronts.compute_obj_val()
        objectives = self.fronts.obj_val
        plt.figure()
        
        # 随机颜色
        colors = [plt.cm.tab20(random.randint(0, 19)) for _ in range(len(objectives))]
        
        plt.scatter(objectives[:, 0], objectives[:, 1], c=colors, edgecolor=colors, alpha=0.7, s = 7)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front at FE={FE}')
        plt.grid(True)
        
        # 设置横纵坐标范围
        if self.xlim:
            plt.xlim(self.xlim)
        if self.ylim:
            plt.ylim(self.ylim)
        
        plt.savefig(f'{self.image_dir}/pareto_front/pareto_front_{FE}.png', format='png', dpi=300)
        plt.close()
        
    def plot_pareto_end(self):
        '''
        绘制有约束pareto前沿
        '''
        if self.fronts is None:
            print("No fronts to plot.")
            return
        
        self.fronts.compute_obj_val()
        objectives = self.fronts.obj_val
        plt.figure()
        
        # 随机颜色
        colors = [plt.cm.tab20(random.randint(0, 19)) for _ in range(len(objectives))]
        
        plt.scatter(objectives[:, 0], objectives[:, 1], c=colors, edgecolor=colors, alpha=0.7, s = 7)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front at the end')
        plt.grid(True)
        
        plt.savefig(f'{self.image_dir}/pareto_front_end.svg', format='svg')
        plt.close()

    def plot_avg_constraint_violations(self):
        '''
        绘制平均约束违反值图像
        '''
        plt.figure()
        plt.plot(self.avg_constraint_violations, c = '#58508d')
        plt.xlabel('Generation')
        plt.ylabel('Average Constraint Violation')
        plt.title('Average Constraint Violation per Generation')
        plt.grid(True)
        plt.savefig(f'{self.image_dir}/avg_constraint_violations.svg', format='svg')
        plt.close()

    def plot_obj_values(self):
        '''
        绘制每一代的最优解和平均目标值
        '''
        num_generations = len(self.best_obj_values)
        num_objs = self.probl.num_obj

        fig, axs = plt.subplots(num_objs, 1, figsize=(12, 4 * num_objs))
        for i in range(num_objs):
            # 绘制最优解和平均值
            axs[i].plot(range(num_generations), [gen[i] for gen in self.best_obj_values],label='Best', c = '#925eb0')
            axs[i].plot(range(num_generations), [gen[i] for gen in self.avg_obj_values], label='Average', c = '#7e99f4')
            axs[i].set_title(f'Objective {i+1}')
            axs[i].set_xlabel('Generation')
            axs[i].set_ylabel(f'Objective {i+1}')
            axs[i].grid(True)
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(f'{self.image_dir}/obj_values.svg', format='svg')
        plt.close()

    def create_video(self):
        '''
        将保存的图像合成为视频
        '''
        images = [img for img in os.listdir(f'{self.image_dir}/pareto_front') if img.endswith(".png")]
        images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按照FE排序

        if not images:
            print("No images to create video.")
            return

        frame = cv2.imread(os.path.join(f'{self.image_dir}/pareto_front', images[0]))
        if frame is None:
            print(f"Error reading the first image: {os.path.join(f'{self.image_dir}/pareto_front', images[0])}")
            return

        height, width, layers = frame.shape

        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

        video_path = os.path.join(self.video_dir, 'pareto_front_evolution.mp4')
        # 设置帧率为 30，使用无损压缩
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for image in images:
            image_path = os.path.join(self.image_dir, 'pareto_front', image)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error reading image: {image_path}")
                continue
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
        print(f"Video saved at {video_path}")

LearnableReproduction.py
import torch
import numpy as np
from LCMOEA_py.population import Population
from tqdm import tqdm
class LearnableReproduction:
    def __init__(self, M1, M2, P, problem, alpha, device):
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
            r = np.random.uniform(0, 1)
            xc = x.cpu().numpy() + \
                self.alpha * (y1 - x.cpu().numpy()) + \
                (0.5 - self.alpha) * (y2 - x.cpu().numpy()) + \
                r * (xd1 - xd2)

            # 多项式变异
            xc_new = self.polynomial_mutation(xc)
            xc_new = np.clip(xc_new, self.lb, self.ub)
            
            # 贪婪选择 
            xc_new_obj = self.problem.evaluate(xc_new)
            xc_obj = self.problem.evaluate(xc)
            if xc_new_obj <= xc_obj:
                xc = xc_new

            # 将 xc 添加到子代种群 Q 中
            Q.append_pop(xc)

        return Q
MLP_Training.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import itertools
from math import comb
from tqdm import tqdm
import os

# 定义单层感知机模型（MLP）
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.LeakyReLU())  # 使用 LeakyReLU 激活函数
            current_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.Sigmoid()) # 使用 Sigmoid 激活函数
        
        # 将所有层组合成一个顺序容器
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MLP_Training:
    def __init__(self, P, problem, device, hidden_size, num_epochs, learning_rate, model_dir, use_existing_models=False):
        '''
        P: 种群实例, N x dim, N为种群大小, dim为决策变量个数
        problem: 问题实例
        device: 训练设备, 'cpu' 或 'cuda'
        num_epochs: 训练轮数
        learning_rate: 学习率
        model_dir: 模型保存目录
        use_existing_models: 是否使用已存在的模型
        '''
        self.P = P  # N x dim
        self.dim = P.dim  # 决策变量的数量
        self.probl = problem # 问题实例
        self.device = device
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.use_existing_models = use_existing_models
        
        self.lb = problem.lb  # 下界
        self.ub = problem.ub  # 上界
        self.num_obj = problem.num_obj # 目标函数的数量
        self.num_constr = problem.num_constr # 约束条件的数量
        self.objectives = problem.objectives  # 目标函数列表
        self.equality_constraints = problem.equality_constraints  # 等式约束条件列表
        self.inequality_constraints = problem.inequality_constraints  # 不等式约束条件列表
        self.N = len(P)  # 种群大小        

        self.num_refvec = 0  # 参考向量的数量
        self.M1 = None  # MLP模型M1
        self.M2 = None  # MLP模型M2
        self.reference_vectors = None  # 参考向量集合 V

    def model_init(self):
        '''模型初始化
        它们的输入/输出层有 dim 个神经元，
        并且只有一个包含 K=10 个神经元的隐藏层
        这样使模型训练的成本可承受且收益还算值得
        '''
        self.M1 = MLP(self.dim, self.dim, self.hidden_size).to(self.device)
        self.M2 = MLP(self.dim, self.dim, self.hidden_size).to(self.device)

    def generate_reference_vectors(self):
        '''
        使用Das and Dennis方法生成均匀分布在单纯形上的参考向量
        '''
        p = self._get_division_number() # 估计分割数p
        self.reference_vectors = self._das_dennis_reference_vectors(p)  # K x m
        self.num_refvec = self.reference_vectors.shape[0]  # 更新N为参考向量的数量

    def _get_division_number(self):
        '''
        根据种群大小N和目标维度num_obj,估计分割数p
        '''
        # 使用组合数计算，可以调整使得K接近N
        p = 0
        K = 0
        while K < self.N:
            p += 1
            K = self._calculate_combinations(p, self.num_obj)
        return p

    def _calculate_combinations(self, p, m):
        '''
        计算组合数 C(p + m - 1, m - 1)
        '''
        return comb(p + m - 1, m - 1)

    def _das_dennis_reference_vectors(self, p):
        '''
        生成Das and Dennis参考向量
        '''
        indices = [i for i in range(p + 1)]# 0,1,2,...,p
        combins = list(itertools.combinations_with_replacement(indices, self.num_obj - 1))# 生成组合,如(0,0),(0,1),(0,2),...,(p,p)
        vectors = []
        for c in combins: 
            vec = [c[0]]
            for i in range(1, len(c)):
                vec.append(c[i] - c[i - 1])
            vec.append(p - c[-1])
            vector = np.array(vec) / p
            vectors.append(vector)
        vectors = np.array(vectors)
        return vectors  # num_refvec x num_obj

    def compute_theta(self, F_x_normalized, v):
        '''
        计算theta(x, v)
        F_x_normalized: N x num_obj 的归一化目标值矩阵
        v: 参考向量，形状为 (num_obj,)
        返回值: theta值,形状为 (N,)
        '''
        dot_product = np.abs(np.dot(F_x_normalized, v))
        norm_F_x = np.linalg.norm(F_x_normalized, axis=1)
        norm_v = np.linalg.norm(v)
        cos_theta = dot_product / (norm_F_x * norm_v + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
        theta = np.arccos(cos_theta)
        return theta  # N x 1

    def fit(self, x_index, v):
        '''
        计算fit(x) = sum_{i=1}^num_obj v_i * f_i(x)
        x_index: 解在种群中的索引
        v: 参考向量
        '''
        return np.dot(v, self.P.obj_val[x_index])

    def save_model(self, model, model_name):
        '''保存模型'''
        model_path = os.path.join(self.model_dir, model_name)
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, model_name):
        '''加载模型'''
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(self.device)
            return model
        return None

    def train_models(self):
        '''
        训练M1和M2模型
        '''
        if self.use_existing_models:
            self.M1 = self.load_model(MLP(self.dim, self.dim, self.hidden_size), 'M1.pth')
            self.M2 = self.load_model(MLP(self.dim, self.dim, self.hidden_size), 'M2.pth')
        else:
            self.model_init()
        self.generate_reference_vectors()# 生成参考向量
        self.P.norm_sol()# 归一化种群
        if self.P.obj_val is None:
            self.P.compute_obj_val()# 计算目标值
        self.P.normalize_objectives()# 归一化目标值
        self.P.compute_constraint_violations()# 计算约束违反程度

        # 将数据转换为PyTorch张量
        population_tensor = torch.tensor(self.P.norm_pop, dtype=torch.float32).to(self.device)

        M1_optimizer = optim.Adam(self.M1.parameters(), lr=self.learning_rate)
        M2_optimizer = optim.Adam(self.M2.parameters(), lr=self.learning_rate)

        criterion = nn.MSELoss()# 损失函数
        
        # 初始化TensorBoard记录器
        # writer = SummaryWriter()

        for epoch in tqdm(range(self.num_epochs), desc=f"MLP Training"):
            # epoch_loss_M1 = 0.0
            # epoch_loss_M2 = 0.0
            for i in range(self.num_refvec):
                v = self.reference_vectors[i]  # 参考向量v_i 即针对问题i的参考向量

                # 计算所有解与参考向量的theta值
                theta_values = self.compute_theta(self.P.norm_obj, v)  # N

                # 找到与v_i最接近的两个不同的解
                sorted_indices = np.argsort(theta_values)# 按theta值排序
                x1_index = sorted_indices[0]# 最接近的解
                # 寻找下一个不同的解
                x2_index = None
                for idx in sorted_indices[1:]:
                    if idx != x1_index:
                        x2_index = idx
                        break
                if x2_index is None:
                    x2_index = (x1_index + 1) % self.N  # 若所有解都相同，则选择下一个解

                x1 = population_tensor[x1_index]
                x2 = population_tensor[x2_index]

                # 计算fit值
                fit_x1 = self.fit(x1_index, v)
                fit_x2 = self.fit(x2_index, v)

                # 对于M1（忽略约束任务）
                if fit_x1 >= fit_x2:
                    x_input_M1 = x1.unsqueeze(0)
                    x_label_M1 = x2.unsqueeze(0)
                else:
                    x_input_M1 = x2.unsqueeze(0)
                    x_label_M1 = x1.unsqueeze(0)

                # 更新M1
                self.M1.train() # 训练模式
                M1_optimizer.zero_grad()# 梯度清零
                output_M1 = self.M1(x_input_M1)# 前向传播
                loss_M1 = criterion(output_M1, x_label_M1)# 计算损失
                loss_M1.backward()# 反向传播
                M1_optimizer.step()# 更新参数
                # epoch_loss_M1 += loss_M1.item()

                # 计算约束违反程度
                cnu_x1 = self.P.constr_violations[x1_index]
                cnu_x2 = self.P.constr_violations[x2_index]

                # 对于M2（可行性优先任务）
                if cnu_x1 > cnu_x2:
                    x_input_M2 = x1.unsqueeze(0)
                    x_label_M2 = x2.unsqueeze(0)
                elif cnu_x1 < cnu_x2:
                    x_input_M2 = x2.unsqueeze(0)
                    x_label_M2 = x1.unsqueeze(0)
                else:
                    # 若约束违反程度相等，则比较fit值
                    if fit_x1 >= fit_x2:
                        x_input_M2 = x1.unsqueeze(0)
                        x_label_M2 = x2.unsqueeze(0)
                    else:
                        x_input_M2 = x2.unsqueeze(0)
                        x_label_M2 = x1.unsqueeze(0)

                # 更新M2
                self.M2.train()
                M2_optimizer.zero_grad()
                output_M2 = self.M2(x_input_M2)
                loss_M2 = criterion(output_M2, x_label_M2)
                loss_M2.backward()
                M2_optimizer.step()
                # epoch_loss_M2 += loss_M2.item()
            
            # 记录每一轮的损失
            # writer.add_scalar('Loss/M1', epoch_loss_M1 / self.num_refvec, epoch)
            # writer.add_scalar('Loss/M2', epoch_loss_M2 / self.num_refvec, epoch)

        # 保存模型
        self.save_model(self.M1, 'M1.pth')
        self.save_model(self.M2, 'M2.pth')

        # 返回训练好的模型
        return self.M1, self.M2
        
population.py
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

problem.py
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


