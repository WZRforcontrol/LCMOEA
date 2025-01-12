classdef LearnableReproduction
    properties
        M1  % 训练好的模型 M1
        M2  % 训练好的模型 M2
        P  % 父代种群实例
        problem  % 问题实例
        alpha  % 参数 alpha
        lb  % 下界
        ub  % 上界
        dim  % 决策变量维度
        N  % 种群规模
    end

    methods
        function obj = LearnableReproduction(M1, M2, P, problem, alpha)
            obj.M1 = M1;
            obj.M2 = M2;
            obj.P = P.pop;  % 获取种群数据
            obj.problem = problem;
            obj.alpha = alpha;
            obj.lb = problem.lb;
            obj.ub = problem.ub;
            obj.dim = problem.dim;
            obj.N = size(P.pop, 1);
        end

        function x = polynomial_mutation(obj, x, eta)
            % 多项式变异操作
            if nargin < 3
                eta = 20;
            end
            for i = 1:obj.dim
                if rand <= 1 / obj.dim
                    delta1 = (x(i) - obj.lb(i)) / (obj.ub(i) - obj.lb(i));
                    delta2 = (obj.ub(i) - x(i)) / (obj.ub(i) - obj.lb(i));
                    rand_val = rand;
                    mut_pow = 1 / (eta + 1);
                    if rand_val < 0.5
                        xy = 1 - delta1;
                        val = 2 * rand_val + (1 - 2 * rand_val) * (xy^(eta + 1));
                        delta_q = val^(mut_pow) - 1;
                    else
                        xy = 1 - delta2;
                        val = 2 * (1 - rand_val) + 2 * (rand_val - 0.5) * (xy^(eta + 1));
                        delta_q = 1 - val^(mut_pow);
                    end
                    x(i) = x(i) + delta_q * (obj.ub(i) - obj.lb(i));
                    x(i) = min(max(x(i), obj.lb(i)), obj.ub(i));
                end
            end
        end

        function y = denorm_y(obj, y_norm)
            % 反归一化解 y
            y = y_norm .* (obj.ub - obj.lb + 1e-8) + obj.lb;
        end

        function Q = reproduce(obj)
            % 执行可学习的繁殖过程，生成子代种群 Q
            Q = Population(obj.problem, 0);

            for i = 1:obj.N
                x = obj.P(i, :)';

                % 计算 y1 和 y2
                y1_norm = obj.M1(x);
                y1 = obj.denorm_y(y1_norm);
                y2_norm = obj.M2(x);
                y2 = obj.denorm_y(y2_norm);

                % 从 P 中随机选择两个不同的解 xd1 和 xd2
                idxs = 1:obj.N;
                idxs(i) = [];
                xd_indices = randperm(length(idxs), 2);
                xd1 = obj.P(idxs(xd_indices(1)), :)';
                xd2 = obj.P(idxs(xd_indices(2)), :)';

                % 计算子代解 xc
                r = rand;
                xc = x + obj.alpha * (y1 - x) + (0.5 - obj.alpha) * (y2 - x) + r * (xd1 - xd2);

                % 多项式变异
                xc_new = obj.polynomial_mutation(xc);
                xc_new = min(max(xc_new, obj.lb), obj.ub);

                % 贪婪选择
                xc_new_obj = obj.problem.evaluate(xc_new);
                xc_obj = obj.problem.evaluate(xc);
                if xc_new_obj <= xc_obj
                    xc = xc_new;
                end

                % 添加到子代种群 Q 中
                Q = Q.append_pop(xc');
            end
        end
    end
end
