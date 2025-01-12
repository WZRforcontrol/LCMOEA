function vectors = das_dennis_ref_vecs(N, M)
    % 生成均匀分布的Das and Dennis参考向量
    % 输入:
    %   N - 种群数，也是向量个数
    %   M - 目标函数维度
    % 输出:
    %   vectors - 生成的参考向量矩阵

    % 计算p值
    p = N^(1/(M-1)) - 1;
    p = round(p);

    % 生成组合
    indices = 0:p;
    combins = nchoosek(repmat(indices, 1, M-1), M-1);

    vectors = [];
    for i = 1:size(combins, 1)
        c = combins(i, :);
        vec = [c(1)];
        for j = 2:length(c)
            vec = [vec, c(j) - c(j-1)];
        end
        vec = [vec, p - c(end)];
        vector = vec / p;
        vectors = [vectors; vector];
    end

    % 归一化
    vectors = vectors ./ vecnorm(vectors, 2, 2);

    % 确保参考向量在空间上角度均匀分布
    vectors = unique(vectors, 'rows');
end