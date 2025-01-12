classdef CMOP2Problem
    properties
        dim  % 决策变量维度
        UBLB  % 上下界
        objectives  % 目标函数列表
        inequality_constraints  % 不等式约束列表
        equality_constraints  % 等式约束列表
        www  % 目标函数的权重向量
    end

    methods
        function obj = CMOP2Problem(D)
            % 初始化CMOP2问题实例
            if nargin < 1
                D = 25;
            end
            obj.dim = D;
            obj.UBLB = [1.1; 0];
            obj.objectives = {@obj.objective1, @obj.objective2};
            obj.inequality_constraints = {@obj.constraint};
            obj.equality_constraints = {};
            obj.www = [1; 1];
        end

        function g = g_func(obj, x)
            % 计算辅助函数 g(x)
            D = obj.dim;
            z = 1 - exp(-10 * (x(2:end) - ((2:D)' - 1)/D).^2);
            g = 1 + sum(1.5 + (0.1 / D) * z.^2 - 1.5 * cos(2 * pi * z));
        end

        function f1 = objective1(obj, x)
            % 计算第一个目标函数值
            g = obj.g_func(x);
            f1 = g * x(1);
        end

        function f2 = objective2(obj, x)
            % 计算第二个目标函数值
            g = obj.g_func(x);
            f1 = obj.objective1(x);
            f2 = g * sqrt(1.1^2 - (f1 / g)^2);
        end

        function LA_val = LA(obj, A, B, C, D_param, theta)
            % 计算约束条件中的 LA 函数
            t = theta.^C;
            LA_val = A * (cos(B * t)).^D_param;
        end

        function constr_value = constraint(obj, x)
            % 计算约束条件值
            f1 = obj.objective1(x);
            f2 = obj.objective2(x);
            theta = atan2(f2, f1);
            term1 = (f1^2) / (1 + obj.LA(0.15, 6, 4, 10, theta))^2;
            term2 = (f2^2) / (1 + obj.LA(0.75, 6, 4, 10, theta))^2;
            constr_value = term1 + term2 - 1;
        end
    end
end
