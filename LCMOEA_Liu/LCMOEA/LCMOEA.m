classdef LCMOEA < ALGORITHM
% <2024> <multi> <real/integer> <constrained>
% Learning-Aided Constrained Multiobjective Evolutionary Algorithm
% k --- 2 ---  

%------------------------------- Reference --------------------------------
% Liu, S., Wang, Z., Lin, Q., Li, J., & Tan, K. C, Learning-aided 
% evolutionary search and selection for scaling-up constrained 
% multiobjective optimization, IEEE Transactions on Evolutionary 
% Computation, 2024
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            k = Algorithm.ParameterSet(2);
            Population     = Problem.Initialization();
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);% 生成参考向量
            [z,znad]      = deal(min(Population.objs),max(Population.objs));% 种群目标值的下限和上限
            CV = sum(max(0,Population.cons),2);
            Archive       = Population(CV == 0);% 可行域中的个体
            Zmin       = min(Population.objs,[],1);
            %% Optimization
            while Algorithm.NotTerminated(Archive) 
                a          = Problem.FE/Problem.maxFE;% 进度
                % 个人感觉author在玩赖，所以改了一下
                % if length(Archive) == Problem.N
                %     Nt = floor((1-a)*Problem.N);
                %     % 交配池，初期更多的在整个种群中采样，后期更多的在可行域种群中采样
                %     MatingPool = [Population(randsample(Problem.N,Nt)),Archive(randsample(Problem.N,Problem.N-Nt))];
                %     [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPool,Zmin);
                %     Offspring = OperatorDE(Problem,Mate1,Mate2,Mate3);% 差分进化+多项式变异
                % else
                    % Offspring  = LearnableDE(Problem, Population, V, k);% 鲁棒性较差
                    % Offspring  = LearnableDE2(Problem, Population, V, k, a);
                    Offspring  = LearnableDE3(Problem, Population, V, k);% 不错
                    % Offspring  = LearnableDE4(Problem, Population, V, k, a);%和2差不多
                % end
                cv_offspring = sum(max(0,Offspring.cons),2);
                feasible_Offspring = Offspring(cv_offspring == 0);
                Zmin = min([Zmin;Offspring.objs],[],1);
                %
                if a < 0.25
                    Population = EnvironmentalSelection_TDEA([Population,Offspring],V,Problem.N,z,znad);
                    %Population = EnvironmentalSelection_Clustering2([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                    % Population = Clustering([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                else
                    Population = EnvironmentalSelection_Clustering2([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                end
                if length(Archive) + length(feasible_Offspring) <= Problem.N
                    Archive = [Archive,feasible_Offspring];
                else
                    Archive = EnvironmentalSelection_Clustering3([Archive,feasible_Offspring],Problem.N,z,znad,Problem.N,a);
                end
            end
        end
    end
end