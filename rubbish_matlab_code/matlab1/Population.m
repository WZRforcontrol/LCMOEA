classdef Population
    properties
        problem  % 问题实例
        dim  % 决策变量维度
        lb  % 下界
        ub  % 上界
        N_init  % 初始种群大小
        pop  % 种群矩阵
        objectives  % 目标函数列表
        equality_constraints  % 等式约束列表
        inequality_constraints  % 不等式约束列表
        num_obj  % 目标函数数量
        constr_violations  % 约束违反程度
        obj_val  % 目标值
        norm_obj  % 归一化目标值
        norm_pop  % 归一化种群
    end

    methods
        function obj = Population(problem, N_init)
            obj.problem = problem;
            obj.dim = problem.dim;
            obj.lb = problem.lb;
            obj.ub = problem.ub;
            obj.N_init = N_init;
            obj.pop = [];  % 初始化为空
            obj.objectives = problem.objectives;
            obj.equality_constraints = problem.equality_constraints;
            obj.inequality_constraints = problem.inequality_constraints;
            obj.num_obj = length(obj.objectives);
            obj.constr_violations = [];
            obj.obj_val = [];
            obj.norm_obj = [];
            obj.norm_pop = [];
        end

        function obj = append_pop(obj, new_individual)
            % 添加新的个体到种群
            obj.pop = [obj.pop; new_individual];
        end

        function obj = gps_init(obj)
            % 使用佳点集方法初始化种群
            N = obj.N_init;
            d = obj.dim;
            population = zeros(N, d);
            prime_number_min = 2 * d + 3;

            % 找到最小的素数
            while ~isprime(prime_number_min)
                prime_number_min = prime_number_min + 1;
            end

            for i = 1:N
                for j = 1:d
                    r = mod(2 * cos(2 * pi * j / prime_number_min) * i, 1);
                    population(i, j) = obj.lb(j) + r * (obj.ub(j) - obj.lb(j));
                end
            end

            obj.pop = population;
        end

        function obj = compute_obj_val(obj)
            % 计算种群中所有个体的目标值
            N = size(obj.pop, 1);
            obj.obj_val = zeros(N, obj.num_obj);
            for i = 1:N
                x = obj.pop(i, :)';
                for j = 1:obj.num_obj
                    obj.obj_val(i, j) = obj.objectives{j}(x);
                end
            end
        end

        function obj = normalize_objectives(obj)
            % 归一化目标值
            z_min = min(obj.obj_val);
            z_max = max(obj.obj_val);
            obj.norm_obj = (obj.obj_val - z_min) ./ (z_max - z_min + 1e-8);
        end

        function obj = compute_constraint_violations(obj)
            % 计算约束违反程度
            N = size(obj.pop, 1);
            obj.constr_violations = zeros(N, 1);
            for i = 1:N
                x = obj.pop(i, :)';
                total_violation = 0;

                % 检查变量是否在上下界内
                lower_violation = sum(max(0, obj.lb - x));
                upper_violation = sum(max(0, x - obj.ub));
                total_violation = total_violation + 2 * (lower_violation + upper_violation);

                % 等式约束
                for k = 1:length(obj.equality_constraints)
                    violation = abs(obj.equality_constraints{k}(x));
                    total_violation = total_violation + violation;
                end

                % 不等式约束
                for k = 1:length(obj.inequality_constraints)
                    violation = obj.inequality_constraints{k}(x);
                    total_violation = total_violation + max(0, violation);
                end

                obj.constr_violations(i) = total_violation;
            end
        end

        function obj = norm_sol(obj)
            % 归一化种群解
            obj.norm_pop = (obj.pop - obj.lb') ./ (obj.ub' - obj.lb' + 1e-8);
        end
    end
end
