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
                if deg_pro <= 0.5
                    alpha = 0.5;
                else
                    alpha = 0.0;
                end
                Offspring  = LearnableReproduction(Problem, Population, MLPs, alpha);
                % % 个人感觉author在玩赖，所以改了一下
                % % if length(Archive) == Problem.N
                % %     Nt = floor((1-a)*Problem.N);
                % %     % 交配池，初期更多的在整个种群中采样，后期更多的在可行域种群中采样
                % %     MatingPool = [Population(randsample(Problem.N,Nt)),Archive(randsample(Problem.N,Problem.N-Nt))];
                % %     [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPool,Zmin);
                % %     Offspring = OperatorDE(Problem,Mate1,Mate2,Mate3);% 差分进化+多项式变异
                % % else
                %     % Offspring  = LearnableDE(Problem, Population, V, k);% 鲁棒性较差
                %     % Offspring  = LearnableDE2(Problem, Population, V, k, a);
                %     % Offspring  = LearnableDE3(Problem, Population, V, k);% 不错
                %     % Offspring  = LearnableDE4(Problem, Population, V, k, a);%和2差不多
                % % end
                
                % Algorithm 4 ClusteringAidedSelection
                cv_offspring = sum(max(0,Offspring.cons),2);
                feasible_Offspring = Offspring(cv_offspring == 0);
                Zmin = min([Zmin;Offspring.objs],[],1);
                %
                if deg_pro < 0.25
                    Population = EnvironmentalSelection_TDEA([Population,Offspring],ref_vec,Problem.N,z,znad);
                    %Population = EnvironmentalSelection_Clustering2([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                    % Population = Clustering([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                else
                    Population = EnvironmentalSelection_Clustering2([Population,Offspring],Problem.N,z,znad,Problem.N,deg_pro);
                end
                % if length(Archive) + length(feasible_Offspring) <= Problem.N
                %     Archive = [Archive,feasible_Offspring];
                % else
                %     Archive = EnvironmentalSelection_Clustering3([Archive,feasible_Offspring],Problem.N,z,znad,Problem.N,a);
                % end
                % Population = Offspring;
                % Population = Problem.Evaluation(Population);
            end
        end
    end
end