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
        M1  % 忽略约束任务网络
        M2  % 约束可行性任务网络
        Problem % 问题
        hidden_size % 隐藏层大小
        num_layers % 神经网络层数
        num_epochs  % 训练轮数
        lr % 学习率
        Population  % 种群
        ref_vec % 参考向量
        
        % M1 与 M2 网络的权重
        M1_hidden_weights
        M1_output_weights
        M2_hidden_weights
        M2_output_weights

        % Adam超参数及优化所需动量矩阵
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-8;
        M1_m_hidden
        M1_v_hidden
        M1_m_out
        M1_v_out
        M2_m_hidden
        M2_v_hidden
        M2_m_out
        M2_v_out
    end
    
    methods
        function obj = MLP(Problem, Population, hidden_size, epochs, lr, ref_vec, num_layers)
            % 初始化基础属性和网络结构大小
            obj.Problem = Problem;
            obj.Population = Population;
            obj.hidden_size = hidden_size;
            obj.num_epochs = epochs;
            obj.lr = lr;
            obj.ref_vec = ref_vec;
            obj.num_layers = num_layers; 
        end

        function Model_init(obj)
            obj.M1_hidden_weights = cell(1,obj.num_layers);
            obj.M2_hidden_weights = cell(1,obj.num_layers);
            obj.M1_m_hidden = cell(1,obj.num_layers);
            obj.M1_v_hidden = cell(1,obj.num_layers);
            obj.M2_m_hidden = cell(1,obj.num_layers);
            obj.M2_v_hidden = cell(1,obj.num_layers);
            % 此处为 M1 与 M2 隐藏层与输出层的权重初始化
            for i = 1:obj.num_layers
                if i == 1
                    input_size = obj.Problem.D;
                else
                    input_size = obj.hidden_size;
                end
                obj.M1_hidden_weights{i} = randn(input_size, obj.hidden_size);
                obj.M2_hidden_weights{i} = randn(input_size, obj.hidden_size);
                obj.M1_m_hidden{i} = zeros(size(obj.M1_hidden_weights{i}));
                obj.M1_v_hidden{i} = zeros(size(obj.M1_hidden_weights{i}));
                obj.M2_m_hidden{i} = zeros(size(obj.M2_hidden_weights{i}));
                obj.M2_v_hidden{i} = zeros(size(obj.M2_hidden_weights{i}));
            end
            obj.M1_output_weights = randn(obj.hidden_size, obj.Problem.D);
            obj.M2_output_weights = randn(obj.hidden_size, obj.Problem.D);
            % 初始化Adam优化中需要的动量与二阶动量矩阵
            obj.M1_m_out = zeros(size(obj.M1_output_weights));
            obj.M1_v_out = zeros(size(obj.M1_output_weights));
            obj.M2_m_out = zeros(size(obj.M2_output_weights));
            obj.M2_v_out = zeros(size(obj.M2_output_weights));
        end
        
        function train_models(obj)
            [x_1_M1, x_2_M1, x_1_M2, x_2_M2] = obj.prepare_training_data();
            % 准备数据并分别调用 M1 与 M2 的训练方法
            obj.train_net_M1(x_1_M1, x_2_M1);
            obj.train_net_M2(x_1_M2, x_2_M2);
        end

        function train_net_M1(obj, inputs, targets)
            % 针对 M1 网络进行 Adam 梯度更新的训练循环
            for e = 1:obj.num_epochs
                [output, hidden_outputs] = obj.forward_M1(inputs);
                batch_size = size(inputs,1);
                out_err = (output - targets) .* sigmoid_derivative(output);
                % 将误差除以 batch_size 后再计算梯度
                out_err = out_err ./ batch_size;
                % Adam更新输出层权重
                [obj.M1_m_out, obj.M1_v_out, out_delta] = obj.AdamUpdate(obj.M1_m_out, obj.M1_v_out, ...
                    hidden_outputs{end}'*out_err, e);
                obj.M1_output_weights = obj.M1_output_weights - obj.lr*out_delta;

                hidden_err = cell(1, obj.num_layers);
                hidden_err{end} = (out_err * obj.M1_output_weights') .* ...
                    sigmoid_derivative(hidden_outputs{end});
                for i = obj.num_layers-1:-1:1
                    hidden_err{i} = hidden_err{i+1} * obj.M1_hidden_weights{i+1}' .* ...
                        sigmoid_derivative(hidden_outputs{i});
                end
                for i = obj.num_layers:-1:1
                    if i == 1
                        hidden_input = inputs;
                    else
                        hidden_input = hidden_outputs{i-1};
                    end
                    hidden_err{i} = hidden_err{i} ./ batch_size;
                    [obj.M1_m_hidden{i}, obj.M1_v_hidden{i}, h_delta] = ...
                        obj.AdamUpdate(obj.M1_m_hidden{i}, obj.M1_v_hidden{i}, hidden_input'*hidden_err{i}, e);
                    obj.M1_hidden_weights{i} = obj.M1_hidden_weights{i} - obj.lr*h_delta;
                    % 针对每一层隐藏层进行梯度更新
                end
            end
        end

        function train_net_M2(obj, inputs, targets)
            % 针对 M2 网络进行 Adam 梯度更新的训练循环
            for e = 1:obj.num_epochs
                [output, hidden_outputs] = obj.forward_M2(inputs);
                batch_size = size(inputs,1);
                out_err = (output - targets) .* sigmoid_derivative(output);
                out_err = out_err ./ batch_size;
                [obj.M2_m_out, obj.M2_v_out, out_delta] = ...
                    obj.AdamUpdate(obj.M2_m_out, obj.M2_v_out, hidden_outputs{end}'*out_err, e);
                obj.M2_output_weights = obj.M2_output_weights - obj.lr*out_delta;

                hidden_err = cell(1, obj.num_layers);
                hidden_err{end} = (out_err * obj.M2_output_weights') .* ...
                    sigmoid_derivative(hidden_outputs{end});
                for i = obj.num_layers-1:-1:1
                    hidden_err{i} = hidden_err{i+1} * obj.M2_hidden_weights{i+1}' .* ...
                        sigmoid_derivative(hidden_outputs{i});
                end
                for i = obj.num_layers:-1:1
                    if i == 1
                        hidden_input = inputs;
                    else
                        hidden_input = hidden_outputs{i-1};
                    end
                    hidden_err{i} = hidden_err{i} ./ batch_size;
                    [obj.M2_m_hidden{i}, obj.M2_v_hidden{i}, h_delta] = ...
                        obj.AdamUpdate(obj.M2_m_hidden{i}, obj.M2_v_hidden{i}, hidden_input'*hidden_err{i}, e);
                    obj.M2_hidden_weights{i} = obj.M2_hidden_weights{i} - obj.lr*h_delta;
                end
            end
        end

        function [m, v, delta] = AdamUpdate(obj, m_old, v_old, grad, t)
            % Adam优化算法的核心更新函数
            beta1t = obj.beta1^t;
            beta2t = obj.beta2^t;
            m = obj.beta1*m_old + (1-obj.beta1)*grad;
            v = obj.beta2*v_old + (1-obj.beta2)*(grad.^2);
            m_hat = m./(1-beta1t);
            v_hat = v./(1-beta2t);
            delta = m_hat./(sqrt(v_hat)+obj.epsilon);
        end

        function [output, hidden_outputs] = forward_M1(obj, input)
            % 计算 M1 前向传播各层激活值
            hidden_outputs = cell(1,obj.num_layers);
            for i = 1:obj.num_layers
                if i == 1
                    h_in = input * obj.M1_hidden_weights{i};
                else
                    h_in = hidden_outputs{i-1} * obj.M1_hidden_weights{i};
                end
                hidden_outputs{i} = sigmoid(h_in);
            end
            output = sigmoid(hidden_outputs{end}*obj.M1_output_weights);
        end

        function [output, hidden_outputs] = forward_M2(obj, input)
            % 计算 M2 前向传播各层激活值
            hidden_outputs = cell(1,obj.num_layers);
            for i = 1:obj.num_layers
                if i == 1
                    h_in = input * obj.M2_hidden_weights{i};
                else
                    h_in = hidden_outputs{i-1} * obj.M2_hidden_weights{i};
                end
                hidden_outputs{i} = sigmoid(h_in);
            end
            output = sigmoid(hidden_outputs{end}*obj.M2_output_weights);
        end

        function y = ReproductionM1(obj, inputs)
            % 归一化输入数据
            Lower = obj.Problem.lower;
            Upper = obj.Problem.upper;
            inputs = (inputs - repmat(Lower, size(inputs, 1), 1)) ./ repmat(Upper - Lower, size(inputs, 1), 1);
            % 前向传播计算输出
            [tmp, ~] = obj.forward_M1(inputs);
            % 反归一化输出数据
            y = tmp .* repmat(Upper - Lower, size(tmp, 1), 1) + repmat(Lower, size(tmp, 1), 1);
        end

        function y = ReproductionM2(obj, inputs)
            % 归一化输入数据
            Lower = obj.Problem.lower;
            Upper = obj.Problem.upper;
            inputs = (inputs - repmat(Lower, size(inputs, 1), 1)) ./ repmat(Upper - Lower, size(inputs, 1), 1);
            % 前向传播计算输出
            [tmp, ~] = obj.forward_M2(inputs);
            % 反归一化输出数据
            y = tmp .* repmat(Upper - Lower, size(tmp, 1), 1) + repmat(Lower, size(tmp, 1), 1);
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
            
            for i = 1:length(obj.ref_vec)
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
                
                % % Visualization for M1 training pairs
                % figure;
                % quiver(0, 0, v(1), v(2), 'r', 'LineWidth', 2); % Plot reference vector v
                % hold on;
                % % Get objective values for both individuals in the training pair
                % obj1 = PopObj(x_idx(1),:);
                % obj2 = PopObj(x_idx(2),:);
                % quiver(obj1(1), obj1(2), obj2(1)-obj1(1), obj2(2)-obj1(2), 'b', 'LineWidth', 2); % Plot objective values vector
                % legend('Reference Vector v', 'Training Pair Objectives');
                % grid on;
                % hold off;

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
            Lower = obj.Problem.lower;
            Upper = obj.Problem.upper;
            x_1_M1 = (x_1_M1 - repmat(Lower,size(x_1_M1,1),1)) ./ repmat(Upper-Lower,size(x_1_M1,1),1);
            x_2_M1 = (x_2_M1 - repmat(Lower,size(x_2_M1,1),1)) ./ repmat(Upper-Lower,size(x_2_M1,1),1);
            x_1_M2 = (x_1_M2 - repmat(Lower,size(x_1_M2,1),1)) ./ repmat(Upper-Lower,size(x_1_M2,1),1);
            x_2_M2 = (x_2_M2 - repmat(Lower,size(x_2_M2,1),1)) ./ repmat(Upper-Lower,size(x_2_M2,1),1);
        end

        function INDEX = Association(obj,PopObj)
            %% 根据角度最小匹配选择两个个体与目标向量关联
            Cosine = pdist2(obj.ref_vec, PopObj, 'cosine');% 计算参考向量与群体目标之间的余弦相似度
            Angles = acos(1-Cosine);% 将余弦相似度转换为角度（以弧度为单位）
            % 对于每个参考向量，找出角度最小的两个个体
            [~, sorted_indices] = sort(Angles, 2); % 按行排序
            INDEX = sorted_indices(:, 1:2); % 取前两个最接近的个体并转置
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