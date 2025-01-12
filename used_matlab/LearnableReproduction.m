function Offspring = LearnableReproduction(Problem, Population, MLPs, deg_pro)
% Learnable reproduction
%------------------------------- Information ------------------------------
% Author: Z.R.Wang
% Email: wangzhanran@stumail.ysu.edu.cn
% Affiliation: Intelligent Dynamical Systems Research Group, 
% Department of Mechanical Design, Yanshan University, China
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    if deg_pro <= 0.5
        alpha = 0.8;
    else
        alpha = 0.2;
    end
    Offspring = zeros(Problem.N,Problem.D);
    for xi = 1:Problem.N
        x = Population(xi).dec;
        y_1 = MLPs.ReproductionM1(x);
        y_2 = MLPs.ReproductionM2(x);
        other_indices = [1:xi-1, xi+1:Problem.N];
        selected = randperm(length(other_indices), 2);
        p1 = other_indices(selected(1));
        p2 = other_indices(selected(2));
        % eq7
        % x_c = x + alpha*(y_1-x) + (0.5-alpha)*(y_2-x) + ...
        %     rand*(Population(p1).dec - Population(p2).dec);
        x_c = x + alpha*(y_1-x) + (0.5-alpha)*(y_2-x);
        % % 多项式变异
        % proM = 1; disM = 20;
        % Lower = Problem.lower;
        % Upper = Problem.upper;
        % x_c   = min(max(x_c,Lower),Upper);
        % Site  = rand(1,Problem.D) < proM/Problem.D;
        % mu    = rand(1,Problem.D);
        % temp  = Site & mu <= 0.5;
        % x_c(temp) = x_c(temp) + (Upper(temp)-Lower(temp)).*...
        %     ((2.*mu(temp) + (1 - 2.*mu(temp)).*...
        %     (1 - (x_c(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)) - 1);
        % temp = Site & mu > 0.5;
        % x_c(temp) = x_c(temp) + (Upper(temp)-Lower(temp)).*...
        %     (1 - (2.*(1-mu(temp)) + 2.*(mu(temp)-0.5).*...
        %     (1 - (Upper(temp)-x_c(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
        Offspring(xi,:) = x_c;
        % Offspring(xi,:) = y_1;
    end
    Offspring = Problem.Evaluation(Offspring);
end