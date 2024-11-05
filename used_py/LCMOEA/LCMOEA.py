import numpy as np
import torch
from LCMOEA.problem import Problem
from LCMOEA.population import Population
from LCMOEA.MLP_Training import MLP_Training
from LCMOEA.LearnableReproduction import LearnableReproduction
from LCMOEA.ClusteringAidedSelection import ClusteringAidedSelection
import matplotlib.pyplot as plt
import os
import json
import random
import cv2

class LCMOEA:
    def __init__(self, probl, N = 50, max_iter = 50000, device = torch.device("cpu"), FE_k = 0.5,
                 hidden_sizes = [10], num_epochs = 100, learning_rate = 0.001, model_dir='models',
                 is_plot=False,image_dir='images', video_dir = 'video', xlim=None, ylim=None):
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
        self.FE_k = FE_k
        
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
            if FE <= self.FE_k*self.FE_max:
                alpha = 0.8
            else:
                alpha = 0.2 + 0.6 * (1 - (FE - (1 - self.FE_k)*self.FE_max)/((1 - self.FE_k)*self.FE_max))**2
            r = np.random.uniform(0, 1.3)
            reproducer = LearnableReproduction(M1, M2, P, self.probl, alpha, r, self.device)
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
            avg_violation = np.sum(P.constr_violations)
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
            if self.is_plot and self.probl.num_obj == 2:
                self.plot_pareto(FE)
        
        # 输出有约束pareto前沿
        self.plot_pareto_end()

        # 绘制平均约束违反值图像
        self.plot_avg_constraint_violations()

        # 绘制目标函数的最优解和平均值
        if self.is_plot and self.probl.num_obj <= 4:
            self.plot_obj_values()

        # 生成视频
        if self.is_plot and self.probl.num_obj == 2:
            self.create_video()

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