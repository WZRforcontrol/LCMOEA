classdef Problem
    properties
        objectives  % 目标函数列表
        num_obj  % 目标函数数量
        dim  % 决策变量数量
        lb  % 下界
        ub  % 上界
        equality_constraints  % 等式约束列表
        inequality_constraints  % 不等式约束列表
        num_constr  % 约束数量
        www  % 目标函数的权重向量
    end

    methods
        function obj = Problem(objectives, dim, UBLB, equality_constraints, inequality_constraints, www)
            obj.objectives = objectives;
            obj.num_obj = length(objectives);
            obj.dim = dim;
            if size(UBLB, 1) == 2 && numel(UBLB) == 2
                obj.lb = UBLB(2) * ones(dim, 1);
                obj.ub = UBLB(1) * ones(dim, 1);
            elseif size(UBLB, 1) == 2 && size(UBLB, 2) == dim
                obj.lb = UBLB(2, :)';
                obj.ub = UBLB(1, :)';
            else
                error('UBLB 的形状不正确。应为 (2,dim) 或 (2, 1)。');
            end
            obj.equality_constraints = equality_constraints;
            obj.inequality_constraints = inequality_constraints;
            obj.num_constr = length(equality_constraints) + length(inequality_constraints);
            obj.www = www;
        end

        function obj_w = evaluate(obj, x)
            obj_values = zeros(obj.num_obj, 1);
            for i = 1:obj.num_obj
                obj_values(i) = obj.objectives{i}(x);
            end
            obj_w = obj_values' * obj.www;
        end
    end
end
