classdef ClusteringAidedSelection
    properties
        P  % 父代种群
        Q  % 子代种群
        problem  % 问题实例
        w  % 权重参数
        N  % 种群规模
    end

    methods
        function obj = ClusteringAidedSelection(P, Q, problem, w)
            obj.P = P;
            obj.Q = Q;
            obj.problem = problem;
            obj.w = w;
            obj.N = size(P.pop, 1);
        end

        function theta = compute_theta(~, v1, v2)
            % 计算两个向量之间的角度 theta
            dot_product = abs(dot(v1, v2));
            norm_v1 = norm(v1);
            norm_v2 = norm(v2);
            cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8);
            cos_theta = min(max(cos_theta, -1.0), 1.0);
            theta = acos(cos_theta);
        end

        function P_new = select(obj)
            % 执行聚类辅助的环境选择过程
            % 合并 P 和 Q，形成 U
            U = Population(obj.problem, 0);
            U = U.merge(obj.P);
            U = U.merge(obj.Q);

            % 计算 U 中所有解的目标值
            U = U.compute_obj_val();

            % 归一化目标值
            U = U.normalize_objectives();
            normalized_objectives = U.norm_obj;

            % 计算约束违反程度
            U = U.compute_constraint_violations();
            constraint_violations = U.constr_violations;

            % 初始化簇
            clusters = cell(size(U.pop, 1), 1);
            for i = 1:size(U.pop, 1)
                clusters{i}.indices = i;
                clusters{i}.centroid = normalized_objectives(i, :);
            end

            % 进行 N 轮层次聚类
            while length(clusters) > obj.N
                min_theta = inf;
                pair_to_merge = [];

                % 计算所有簇之间的 theta
                for idx_u = 1:length(clusters)
                    ci = clusters{idx_u};
                    for idx_h = idx_u + 1:length(clusters)
                        cj = clusters{idx_h};
                        theta = obj.compute_theta(ci.centroid, cj.centroid);
                        if theta < min_theta
                            min_theta = theta;
                            pair_to_merge = [idx_u, idx_h];
                        end
                    end
                end

                % 合并簇
                if ~isempty(pair_to_merge)
                    u = pair_to_merge(1);
                    h = pair_to_merge(2);
                    clusters{u}.indices = [clusters{u}.indices; clusters{h}.indices];
                    clusters{u}.centroid = mean(normalized_objectives(clusters{u}.indices, :), 1);
                    clusters(h) = [];
                else
                    break;
                end
            end

            % 从每个簇中选择代表解，形成新的种群 P_new
            P_new = Population(obj.problem, 0);

            for i = 1:length(clusters)
                indices = clusters{i}.indices;
                solutions = U.pop(indices, :);
                c_violations = constraint_violations(indices);

                % 判断簇中解的可行性
                feasible = c_violations == 0;
                num_feasible = sum(feasible);
                if num_feasible == length(solutions)
                    % 所有解均可行，选择目标性能最好的解
                    sum_norm_obj = sum(normalized_objectives(indices, :), 2);
                    [~, best_index] = min(sum_norm_obj);
                    x_best = solutions(best_index, :);
                elseif num_feasible == 0
                    % 所有解均不可行，使用综合指标 CI(x)
                    sum_norm_obj = sum(normalized_objectives(indices, :), 2);
                    [~, rank_obj] = sort(sum_norm_obj);
                    [~, rank_cv] = sort(c_violations);
                    rank_obj_positions = zeros(length(solutions), 1);
                    rank_cv_positions = zeros(length(solutions), 1);
                    rank_obj_positions(rank_obj) = 1:length(solutions);
                    rank_cv_positions(rank_cv) = 1:length(solutions);
                    CI = obj.w * rank_obj_positions + (1 - obj.w) * rank_cv_positions;
                    [~, best_index] = min(CI);
                    x_best = solutions(best_index, :);
                else
                    % 可行和不可行解的混合，使用 CI(x)
                    sum_norm_obj = sum(normalized_objectives(indices, :), 2);
                    [~, rank_obj] = sort(sum_norm_obj);
                    [~, rank_cv] = sort(c_violations);
                    rank_obj_positions = zeros(length(solutions), 1);
                    rank_cv_positions = zeros(length(solutions), 1);
                    rank_obj_positions(rank_obj) = 1:length(solutions);
                    rank_cv_positions(rank_cv) = 1:length(solutions);
                    CI = obj.w * rank_obj_positions + (1 - obj.w) * rank_cv_positions;
                    [~, best_index] = min(CI);
                    x_best = solutions(best_index, :);
                end

                P_new = P_new.append_pop(x_best);
            end
        end
    end
end
