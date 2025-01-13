classdef MLP < handle
% MLP network for LCMOEA
%------------------------------- Information ------------------------------
% Author: Z.R.Wang
% Email: wangzhanran@stumail.ysu.edu.cn
% Affiliation: Intelligent Dynamical Systems Research Group, 
% Department of Mechanical Design, Yanshan University, China
%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    properties
        % M1  % 忽略约束任务网络
        % M2  % 约束可行性任务网络
        Problem % 问题
        hidden_size % 隐藏层大小
        num_layers % 神经网络层数
        num_epochs  % 训练轮数
        lr % 学习率
        Population  % 种群
        ref_vec % 参考向量
        data_size % 数据集大小
        
        % M1 与 M2 网络的权重
        M1_hidden_weights 
        M1_output_weights
        M2_hidden_weights
        M2_output_weights

        % 添加损失记录属性
        M1_loss_history = [];  % M1网络的训练损失历史
        M2_loss_history = [];  % M2网络的训练损失历史

        lr_decay = 0.95; % 可选学习率衰减因子
    end
    
    methods
        function obj = MLP(Problem, Population, hidden_size, epochs, lr, ref_vec, num_layers)
            % 初始化基础属性和网络结构大小
            obj.Problem = Problem;
            obj.Population = Population;
            obj.hidden_size = hidden_size; % 隐藏层大小
            obj.num_epochs = epochs; % 训练轮数
            obj.lr = lr; % 学习率
            obj.ref_vec = ref_vec; % 参考向量
            obj.num_layers = num_layers; % 神经网络层数
            obj.data_size = size(ref_vec, 1); % 数据集大小
        end

        function Model_init(obj)
            obj.M1_hidden_weights = cell(1,obj.num_layers);
            obj.M2_hidden_weights = cell(1,obj.num_layers);
            % M1 与 M2 隐藏层与输出层的权重初始化
            for i = 1:obj.num_layers
                if i == 1
                    input_size = obj.Problem.D;
                else
                    input_size = obj.hidden_size;
                end
                obj.M1_hidden_weights{i} = randn(input_size, obj.hidden_size);
                obj.M2_hidden_weights{i} = randn(input_size, obj.hidden_size);
            end
            obj.M1_output_weights = randn(obj.hidden_size, obj.Problem.D);
            obj.M2_output_weights = randn(obj.hidden_size, obj.Problem.D);
        end
        
        function train_models(obj)
            [x_1_M1, x_2_M1, x_1_M2, x_2_M2] = obj.prepare_training_data();
            % 保存初始学习率
            initial_lr = obj.lr;
            % 训练网络
            obj.train_net_M1(x_1_M1, x_2_M1);
            % 恢复学习率
            obj.lr = initial_lr;
            obj.train_net_M2(x_1_M2, x_2_M2);
            % 恢复学习率
            obj.lr = initial_lr;
        end

        function train_net_M1(obj, inputs, targets)
            % constraint-ignored任务(M1)训练
            for e = 1:obj.num_epochs
                epoch_loss = 0;
                for i = 1:obj.data_size
                    % 从(x1, x2)获取(x, x')作为M1训练对
                    x = inputs(i,:);
                    t = targets(i,:);
                    % 前向传播并计算MSE
                    [outputs, hidden_outputs] = obj.forward_M1(x);
                    % 计算输出层误差并更新权重
                    output_err = (outputs - t) .* sigmoid_derivative(outputs);
                    output_grad = hidden_outputs{end}' * output_err;
                    obj.M1_output_weights = obj.M1_output_weights - obj.lr * output_grad;

                    % 反向传播计算隐藏层误差
                    hidden_err = cell(1, obj.num_layers);
                    hidden_err{end} = (output_err * obj.M1_output_weights') .* ...
                        leaky_relu_derivative(hidden_outputs{end});
                    for j = obj.num_layers-1:-1:1
                        hidden_err{j} = (hidden_err{j+1} * obj.M1_hidden_weights{j+1}') .* ...
                            leaky_relu_derivative(hidden_outputs{j});
                    end
                    
                    % 计算隐藏层梯度并更新权重
                    for j = 1:obj.num_layers
                        if j == 1
                            hidden_input = x;
                        else
                            hidden_input = hidden_outputs{j-1};
                        end
                        % 更新M1网络参数
                        obj.M1_hidden_weights{j} = obj.M1_hidden_weights{j} - ...
                            obj.lr * hidden_input' * hidden_err{j};
                    end
                    epoch_loss = epoch_loss + mean((outputs - t).^2);
                end
                obj.M1_loss_history(end+1) = epoch_loss / obj.data_size;
                % 在每轮末尾加入学习率衰减
                obj.lr = obj.lr * obj.lr_decay;
            end
        end
        
        function train_net_M2(obj, inputs, targets)
            % 约束可行性任务(M2)训练
            for e = 1:obj.num_epochs
                epoch_loss = 0;
                for i = 1:obj.data_size
                    % 从(x1, x2)获取(x, x')作为M2训练对
                    x = inputs(i,:);
                    t = targets(i,:);
                    % 前向传播并计算MSE
                    [outputs, hidden_outputs] = obj.forward_M2(x);
                    % 计算输出层误差并更新权重
                    output_err = (outputs - t) .* sigmoid_derivative(outputs);
                    output_grad = hidden_outputs{end}' * output_err;
                    obj.M2_output_weights = obj.M2_output_weights - obj.lr * output_grad;

                    % 反向传播计算隐藏层误差
                    hidden_err = cell(1, obj.num_layers);
                    hidden_err{end} = (output_err * obj.M2_output_weights') .* ...
                        leaky_relu_derivative(hidden_outputs{end});
                    for j = obj.num_layers-1:-1:1
                        hidden_err{j} = (hidden_err{j+1} * obj.M2_hidden_weights{j+1}') .* ...
                            leaky_relu_derivative(hidden_outputs{j});
                    end
                    
                    % 计算隐藏层梯度并更新权重
                    for j = 1:obj.num_layers
                        if j == 1
                            hidden_input = x;
                        else
                            hidden_input = hidden_outputs{j-1};
                        end
                        % 更新M2网络参数
                        obj.M2_hidden_weights{j} = obj.M2_hidden_weights{j} - ...
                            obj.lr * hidden_input' * hidden_err{j};
                    end
                    epoch_loss = epoch_loss + mean((outputs - t).^2);
                end
                obj.M2_loss_history(end+1) = epoch_loss / obj.data_size;
                % 在每轮末尾加入学习率衰减
                obj.lr = obj.lr * obj.lr_decay;
            end
        end

        % 添加绘制损失曲线的方法
        function plot_loss(obj)
            figure('Name', 'Training Loss');
            
            % 创建子图1：M1网络损失
            subplot(2,1,1);
            plot(1:length(obj.M1_loss_history), obj.M1_loss_history, 'b-', 'LineWidth', 1.5);
            title('M1 Network Training Loss');
            xlabel('Epoch');
            ylabel('Loss');
            grid on;
            
            % 创建子图2：M2网络损失
            subplot(2,1,2);
            plot(1:length(obj.M2_loss_history), obj.M2_loss_history, 'r-', 'LineWidth', 1.5);
            title('M2 Network Training Loss');
            xlabel('Epoch');
            ylabel('Loss');
            grid on;
        end

        function [output, hidden_outputs] = forward_M1(obj, input)
            % 针对单个输入的M1前向传播
            hidden_outputs = cell(1,obj.num_layers);
            for i = 1:obj.num_layers
                if i == 1
                    h_in = input * obj.M1_hidden_weights{i};
                else
                    h_in = hidden_outputs{i-1} * obj.M1_hidden_weights{i};
                end
                hidden_outputs{i} = leaky_relu(h_in);  % 改为LeakyReLU
            end
            output = sigmoid(hidden_outputs{end}*obj.M1_output_weights);
        end

        function [output, hidden_outputs] = forward_M2(obj, input)
            % 针对单个输入的M2前向传播
            hidden_outputs = cell(1,obj.num_layers);
            for i = 1:obj.num_layers
                if i == 1
                    h_in = input * obj.M2_hidden_weights{i};
                else
                    h_in = hidden_outputs{i-1} * obj.M2_hidden_weights{i};
                end
                hidden_outputs{i} = leaky_relu(h_in);  % 改为LeakyReLU
            end
            output = sigmoid(hidden_outputs{end}*obj.M2_output_weights);
        end

        function y = ReproductionM1(obj, inputs)
            % 归一化输入数据
            Lower = obj.Problem.lower;
            Upper = obj.Problem.upper;
            inputs = (inputs - repmat(Lower, size(inputs, 1), 1)) ./ repmat(Upper - Lower, size(inputs, 1), 1);
            % 前向传播计算输出
            [y, ~] = obj.forward_M1(inputs);
            % 反归一化输出数据
            y = y .* repmat(Upper - Lower, size(y, 1), 1) + repmat(Lower, size(y, 1), 1);
        end

        function y = ReproductionM2(obj, inputs)
            % 归一化输入数据
            Lower = obj.Problem.lower;
            Upper = obj.Problem.upper;
            inputs = (inputs - repmat(Lower, size(inputs, 1), 1)) ./ repmat(Upper - Lower, size(inputs, 1), 1);
            % 前向传播计算输出
            [y, ~] = obj.forward_M2(inputs);
            % 反归一化输出数据
            y = y .* repmat(Upper - Lower, size(y, 1), 1) + repmat(Lower, size(y, 1), 1);
        end

        function [x_1_M1, x_2_M1, x_1_M2, x_2_M2] = prepare_training_data(obj)
            % 将Population中的个体与参考向量进行关联，生成训练对
            N = obj.Problem.N;
            % 准备训练数据
            x_1_M1 = zeros(length(obj.ref_vec), obj.Problem.D);
            x_2_M1 = zeros(length(obj.ref_vec), obj.Problem.D);
            x_1_M2 = zeros(length(obj.ref_vec), obj.Problem.D);
            x_2_M2 = zeros(length(obj.ref_vec), obj.Problem.D);
     
            PopObj = obj.Population.objs;
            zmin = min(PopObj,[],1);
            zmax = max(PopObj,[],1);
            PopObj = (PopObj - repmat(zmin,N,1))./(repmat(zmax-zmin,N,1));% 优劣解距法
            Popcon = obj.Population.cons;
            w_con = ones(1, size(Popcon, 2));% 约束权重
            Popcon = Popcon * w_con';
            INDEX = obj.Association(PopObj); % 关联个体索引
            
            for i = 1:size(obj.ref_vec,1)
                v = obj.ref_vec(i,:); % 参考向量
                x_idx = INDEX(i,:); % 关联的个体索引

                % M1训练对
                if v*PopObj(x_idx(1),:)' >= v*PopObj(x_idx(2),:)'
                    x_1_M1(i,:) = obj.Population(x_idx(1)).dec;
                    x_2_M1(i,:) = obj.Population(x_idx(2)).dec;
                else
                    x_1_M1(i,:) = obj.Population(x_idx(2)).dec;
                    x_2_M1(i,:) = obj.Population(x_idx(1)).dec;
                end

                % M2训练对
                if Popcon(x_idx(1)) > Popcon(x_idx(2))
                    x_1_M2(i,:) = obj.Population(x_idx(1)).dec;
                    x_2_M2(i,:) = obj.Population(x_idx(2)).dec;
                elseif Popcon(x_idx(1)) < Popcon(x_idx(2))
                    x_1_M2(i,:) = obj.Population(x_idx(2)).dec;
                    x_2_M2(i,:) = obj.Population(x_idx(1)).dec;
                else
                    if v*PopObj(x_idx(1),:)' >= v*PopObj(x_idx(2),:)'
                        x_1_M2(i,:) = obj.Population(x_idx(1)).dec;
                        x_2_M2(i,:) = obj.Population(x_idx(2)).dec;
                    else
                        x_1_M2(i,:) = obj.Population(x_idx(2)).dec;
                        x_2_M2(i,:) = obj.Population(x_idx(1)).dec;
                    end
                end
            end

            % %% 在归一化之前绘制训练数据的目标空间关联向量,测试数据集合格与否
            % PopObj_x1 = obj.Problem.CalObj(x_1_M1);
            % PopObj_x2 = obj.Problem.CalObj(x_2_M1);
            % 
            % % 创建新图窗
            % figure('Name', 'Training Data Association Vectors');
            % 
            % % 根据目标维度选择绘图方式
            % if size(PopObj_x1, 2) == 2
            %     % 2D绘图
            %     for i = 1:size(x_1_M1, 1)
            %         quiver(PopObj_x1(i,1), PopObj_x1(i,2), ...
            %               PopObj_x2(i,1)-PopObj_x1(i,1), ...
            %               PopObj_x2(i,2)-PopObj_x1(i,2), ...
            %               0, 'b-', 'LineWidth', 1);
            %         hold on;
            %     end
            %     xlabel('f1');
            %     ylabel('f2');
            % elseif size(PopObj_x1, 2) == 3
            %     % 3D绘图
            %     for i = 1:size(x_1_M1, 1)
            %         quiver3(PopObj_x1(i,1), PopObj_x1(i,2), PopObj_x1(i,3), ...
            %                PopObj_x2(i,1)-PopObj_x1(i,1), ...
            %                PopObj_x2(i,2)-PopObj_x1(i,2), ...
            %                PopObj_x2(i,3)-PopObj_x1(i,3), ...
            %                0, 'b-', 'LineWidth', 1);
            %         hold on;
            %     end
            %     xlabel('f1');
            %     ylabel('f2');
            %     zlabel('f3');
            %     view(45, 45);
            % end
            % grid on;
            % title('Training Data Association Vectors (Before Normalization)');
            % hold off;
            
            %% 归一化
            Lower = obj.Problem.lower;
            Upper = obj.Problem.upper;
            x_1_M1 = (x_1_M1 - repmat(Lower,size(x_1_M1,1),1)) ./ repmat(Upper-Lower,size(x_1_M1,1),1);
            x_2_M1 = (x_2_M1 - repmat(Lower,size(x_2_M1,1),1)) ./ repmat(Upper-Lower,size(x_2_M1,1),1);
            x_1_M2 = (x_1_M2 - repmat(Lower,size(x_1_M2,1),1)) ./ repmat(Upper-Lower,size(x_1_M2,1),1);
            x_2_M2 = (x_2_M2 - repmat(Lower,size(x_2_M2,1),1)) ./ repmat(Upper-Lower,size(x_2_M2,1),1);
        end

        function INDEX = Association(obj,PopObj)
            % 根据角度最小匹配选择两个个体与目标向量关联
            %% 为每个参考向量关联 2 个候选解
            Cosine = pdist2(obj.ref_vec, PopObj, 'cosine');% 计算参考向量与群体目标之间的余弦相似度
            % 对于每个参考向量，找出角度最小的两个个体
            [~, sorted_indices] = sort(Cosine, 2); % 沿（参考向量）行排序
            INDEX = sorted_indices(:, 1:min(2,size(PopObj,1))); % 取前两个最接近的个体
        end
    end
end

% sigmoid 激活函数
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

% sigmoid 导数
function y = sigmoid_derivative(x)
    y = sigmoid(x) .* (1-sigmoid(x));
end

% 添加LeakyReLU激活函数
function y = leaky_relu(x)
    alpha = 0.01;  % LeakyReLU的斜率参数
    y = max(x, alpha * x);
end

% 添加LeakyReLU导数
function y = leaky_relu_derivative(x)
    alpha = 0.01;  % LeakyReLU的斜率参数
    y = ones(size(x));
    y(x < 0) = alpha;
end