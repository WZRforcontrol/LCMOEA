function Population = ClusteringAidedSelection(Problem, Population, Offspring, deg_pro)
% 聚类辅助的环境选择算法
%------------------------------- 基本信息 ------------------------------
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
    %% 种群合并    
    Merging = [Population,Offspring]; 
    N_all = length(Merging);  

    Popdecs = Merging.decs;
    Popobjs = Merging.objs;
    zmin = min(Popobjs,[],1);
    zmax = max(Popobjs,[],1);
    Popobjs = (Popobjs - repmat(zmin,N_all,1))./(repmat(zmax-zmin,N_all,1));% 优劣解距法
    Popcons = Merging.cons;
    w_con = ones(1, size(Popcons, 2));% 约束权重
    Popcons = Popcons * w_con';
    
    %% 基于余弦相似度的聚类
    % 归一化目标值到单位超球面
    Popobjs = Popobjs./vecnorm(Popobjs,2,2);
    
    % 初始化簇
    N_clusters = N_all;  % 初始簇数量
    class = struct('clusters', [], 'centroid', []);
    class = repmat(class, 1, N_all);
    for i = 1 : N_all
        class(i).clusters = i;  % 初始化簇
        class(i).centroid = Popobjs(i, :);  % 初始化质心
    end

    for i = 1: Problem.N
        % 获取有效的类（非空的centroid）的索引
        validIdx = ~cellfun('isempty', {class.clusters});
        % 将所有有效的centroid组成一个矩阵
        centroids = vertcat(class(validIdx).centroid);
        % 计算余弦相似度
        dist = pdist2(centroids, centroids, "cosine");
        dist(logical(eye(size(dist)))) = inf;  % 对角线元素设为inf
        [~, minIndex] = min(dist(:));
        [c_u, c_h] = ind2sub(size(dist), minIndex);
        % 从有效子集中的索引转换为全集中的索引
        valid_indices = find(validIdx);
        c_u = valid_indices(c_u);
        c_h = valid_indices(c_h);
        
        % 合并簇
        class(c_u).clusters = [class(c_u).clusters, class(c_h).clusters];
        class(c_u).centroid = mean(Popobjs(class(c_u).clusters, :), 1);
        class(c_h).clusters = [];
        class(c_h).centroid = [];
    end
    
    
    %% 计算自适应权重w
    if deg_pro < 0.4
        w = 1.0;              % 进化初期偏重目标函数
    elseif deg_pro > 0.6
        w = 0.1;              % 进化后期偏重约束满足
    else
        w = -4.5 * deg_pro + 2.8;  % 中期线性变化
    end
    
    %% 环境选择
    next = [];
    for i = 1 : N_all
        clusters = class(i).clusters;
        centroid = class(i).centroid;
        
        if isempty(clusters)
            continue;
        end
        
        if isscalar(clusters)
            next = [next, clusters];
            continue;
        else
            % 获取当前簇中的解
            cluster_objs = Popobjs(clusters,:);
            cluster_objs = cluster_objs*centroid';  % 计算目标函数值
            cluster_cons = Popcons(clusters);
            
            % 判断簇中解的可行性
            feasible = cluster_cons == 0;
            num_feasible = sum(feasible);
            
            if num_feasible == length(clusters)
                % 所有解均可行，选择目标性能最好的解
                [~, best_idx] = min(cluster_objs);
                next = [next, clusters(best_idx)];
            else
                % 计算目标函数排名和约束违反度排名
                [~, obj_rank] = sort(cluster_objs);
                [~, cv_rank] = sort(cluster_cons);
                
                % 计算排名位置
                obj_positions = zeros(size(obj_rank));
                cv_positions = zeros(size(cv_rank));
                obj_positions(obj_rank) = 1:length(clusters);
                cv_positions(cv_rank) = 1:length(clusters);
                
                % 计算综合指标CI
                CI = w * obj_positions + (1-w) * cv_positions;
                [~, best_idx] = min(CI);
                next = [next, clusters(best_idx)];
            end
        end
    end
    
    % 生成下一代种群
    Population = Merging(next);
end