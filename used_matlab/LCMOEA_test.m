% LCMOEA_test.m

% 创建问题实例
probl = CMOP2Problem();
problem = Problem(probl.objectives, probl.dim, probl.UBLB, probl.equality_constraints, probl.inequality_constraints, probl.www);

% 设置参数
N = 50;  % 种群规模
max_iter = 100000;  % 最大评价次数

% 实例化 LCMOEA，设置是否绘图
algorithm = LCMOEA(problem, N, max_iter, 'is_plot', true);

% 运行算法
pareto_front = algorithm.run();

% 输出结果
disp('Pareto front solutions:');
disp(pareto_front);
