classdef MLP_Training
    properties
        P  % 种群实例
        dim  % 决策变量维度
        probl  % 问题实例
        device  % 设备（在 MATLAB 中可忽略）
        hidden_size  % 隐藏层大小
        num_epochs  % 训练轮数
        learning_rate  % 学习率
        model_dir  % 模型保存目录
        use_existing_models  % 是否使用已有模型
        lb  % 下界
        ub  % 上界
        num_obj  % 目标函数数量
        num_constr  % 约束数量
        objectives  % 目标函数列表
        equality_constraints  % 等式约束列表
        inequality_constraints  % 不等式约束列表
        N  % 种群大小
        num_refvec  % 参考向量数量
        M1  % 神经网络模型 M1
        M2  % 神经网络模型 M2
        reference_vectors  % 参考向量集合 V
    end

    methods
        function obj = MLP_Training(P, problem, device, hidden_size, num_epochs, learning_rate, model_dir, use_existing_models)
            if nargin < 8
                use_existing_models = false;
            end
            obj.P = P;
            obj.dim = P.dim;
            obj.probl = problem;
            obj.device = device;  % 在 MATLAB 中可忽略
            obj.hidden_size = hidden_size;
            obj.num_epochs = num_epochs;
            obj.learning_rate = learning_rate;
            obj.model_dir = model_dir;
            obj.use_existing_models = use_existing_models;
            obj.lb = problem.lb;
            obj.ub = problem.ub;
            obj.num_obj = problem.num_obj;
            obj.num_constr = problem.num_constr;
            obj.objectives = problem.objectives;
            obj.equality_constraints = problem.equality_constraints;
            obj.inequality_constraints = problem.inequality_constraints;
            obj.N = size(P.pop, 1);
            obj.num_refvec = 0;
            obj.M1 = [];
            obj.M2 = [];
            obj.reference_vectors = [];
        end

        function obj = model_init(obj)
            % 初始化神经网络模型 M1 和 M2
            input_size = obj.dim;
            output_size = obj.dim;

            % 创建 M1
            obj.M1 = feedforwardnet(obj.hidden_size);
            obj.M1 = configure(obj.M1, zeros(input_size, 1), zeros(output_size, 1));
            obj.M1.trainParam.epochs = obj.num_epochs;
            obj.M1.trainParam.lr = obj.learning_rate;

            % 创建 M2
            obj.M2 = feedforwardnet(obj.hidden_size);
            obj.M2 = configure(obj.M2, zeros(input_size, 1), zeros(output_size, 1));
            obj.M2.trainParam.epochs = obj.num_epochs;
            obj.M2.trainParam.lr = obj.learning_rate;
        end

        function obj = generate_reference_vectors(obj)
            % 生成参考向量
            p = obj.get_division_number();
            obj.reference_vectors = obj.das_dennis_reference_vectors(p);
            obj.num_refvec = size(obj.reference_vectors, 1);
        end

        function p = get_division_number(obj)
            % 根据种群大小和目标维度估计分割数 p
            p = 0;
            K = 0;
            while K < obj.N
                p = p + 1;
                K = nchoosek(p + obj.num_obj - 1, obj.num_obj - 1);
            end
        end

        function vectors = das_dennis_reference_vectors(obj, p)
            % 生成 Das 和 Dennis 方法的参考向量
            m = obj.num_obj;
            combinations = nchoosek(0:p, m - 1);
            vectors = [];
            for i = 1:size(combinations, 1)
                c = combinations(i, :);
                vec = [c(1), diff(c), p - c(end)];
                vectors = [vectors; vec / p];
            end
        end

        function theta = compute_theta(~, F_x_normalized, v)
            % 计算 theta 值
            dot_product = abs(F_x_normalized * v');
            norm_F_x = vecnorm(F_x_normalized, 2, 2);
            norm_v = norm(v);
            cos_theta = dot_product ./ (norm_F_x * norm_v + 1e-8);
            cos_theta = min(max(cos_theta, -1.0), 1.0);
            theta = acos(cos_theta);
        end

        function fit_val = fit(obj, x_index, v)
            % 计算适应度值
            fit_val = v * obj.P.obj_val(x_index, :)';
        end

        function obj = train_models(obj)
            % 训练 M1 和 M2 模型
            if obj.use_existing_models
                % 如果有现有模型，可以在此加载
                % obj.M1 = load('M1.mat');
                % obj.M2 = load('M2.mat');
            else
                obj = obj.model_init();
            end

            obj = obj.generate_reference_vectors();
            obj.P = obj.P.norm_sol();
            if isempty(obj.P.obj_val)
                obj.P = obj.P.compute_obj_val();
            end
            obj.P = obj.P.normalize_objectives();
            obj.P = obj.P.compute_constraint_violations();

            population_data = obj.P.norm_pop';

            for epoch = 1:obj.num_epochs
                for i = 1:obj.num_refvec
                    v = obj.reference_vectors(i, :);

                    % 计算所有解与参考向量的 theta 值
                    theta_values = obj.compute_theta(obj.P.norm_obj, v);

                    % 找到与 v 最接近的两个不同的解
                    [~, sorted_indices] = sort(theta_values);
                    x1_index = sorted_indices(1);
                    x2_index = sorted_indices(2);

                    x1 = population_data(:, x1_index);
                    x2 = population_data(:, x2_index);

                    % 计算适应度值
                    fit_x1 = obj.fit(x1_index, v);
                    fit_x2 = obj.fit(x2_index, v);

                    % 对于 M1（忽略约束任务）
                    if fit_x1 >= fit_x2
                        x_input_M1 = x1;
                        x_label_M1 = x2;
                    else
                        x_input_M1 = x2;
                        x_label_M1 = x1;
                    end

                    % 更新 M1
                    obj.M1 = train(obj.M1, x_input_M1, x_label_M1);

                    % 计算约束违反程度
                    cnu_x1 = obj.P.constr_violations(x1_index);
                    cnu_x2 = obj.P.constr_violations(x2_index);

                    % 对于 M2（可行性优先任务）
                    if cnu_x1 > cnu_x2
                        x_input_M2 = x1;
                        x_label_M2 = x2;
                    elseif cnu_x1 < cnu_x2
                        x_input_M2 = x2;
                        x_label_M2 = x1;
                    else
                        % 若约束违反程度相等，则比较适应度值
                        if fit_x1 >= fit_x2
                            x_input_M2 = x1;
                            x_label_M2 = x2;
                        else
                            x_input_M2 = x2;
                            x_label_M2 = x1;
                        end
                    end

                    % 更新 M2
                    obj.M2 = train(obj.M2, x_input_M2, x_label_M2);
                end
            end

            % 保存模型
            % save('M1.mat', 'obj.M1');
            % save('M2.mat', 'obj.M2');
        end
    end
end
