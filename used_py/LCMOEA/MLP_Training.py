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