classdef LCMOEA < ALGORITHM
% <2024> <multi> <real/integer> <constrained>
% Learning-Aided Constrained Multiobjective Evolutionary Algorithm
% hidden_size --- 10 --- Number of neurons in the hidden layer
% epochs --- 100 --- Number of epochs
% lr --- 0.01 --- Learning rate
% num_layers --- 1 --- Number of layers

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
%------------------------------- Information ------------------------------
% Author: Z.R.Wang
% Email: wangzhanran@stumail.ysu.edu.cn
% Affiliation: Intelligent Dynamical Systems Research Group, 
% Department of Mechanical Design, Yanshan University, China
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            % k = Algorithm.ParameterSet(2);
            hidden_size = Algorithm.ParameterSet(10);
            epochs = Algorithm.ParameterSet(100);
            lr = Algorithm.ParameterSet(0.01);
            num_layers = Algorithm.ParameterSet(1);

            %% Initial
            Population = Problem.Initialization();% 原始初始化
            % Population = GPS_init(Problem);% 佳点集初始化
            ref_vec = UniformPoint(Problem.N,Problem.M,"MUD");% 生成参考向量
            ref_vec = ref_vec ./ vecnorm(ref_vec, 2, 2);
            % ref_vec = das_dennis_ref_vecs(Problem.N,Problem.M);% 生成参考向量
            MLPs = MLP(Problem, Population, hidden_size, epochs, lr, ref_vec, num_layers);
            MLPs.Model_init();
            % [z,znad]      = deal(min(Population.objs),max(Population.objs));% 种群目标值的下限和上限
            % CV = sum(max(0,Population.cons),2);
            % Archive       = Population(CV == 0);% 可行域中的个体
            % Zmin       = min(Population.objs,[],1);
            %% Optimization
            while Algorithm.NotTerminated(Population) 
                deg_pro = Problem.FE/Problem.maxFE;% 进度
                % Algorithm 2 Training
                MLPs.train_models();
                % Algorithm 3 LearnableReproduction
                Offspring  = LearnableReproduction(Problem, Population, MLPs, deg_pro);
                % Algorithm 4 ClusteringAidedSelection
                Population = ClusteringAidedSelection(Problem, Population, Offspring, deg_pro);
            end
        end
    end
end