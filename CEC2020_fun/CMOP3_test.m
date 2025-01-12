clc; clear; close all;

% 获取Pareto前沿
PF = CMOP3('PF');

% 绘制目标空间中的Pareto前沿
figure;
hold on;
scatter(PF(:,1), PF(:,2), 'b*'); % 绘制Pareto前沿
xlabel('Objective 1');
ylabel('Objective 2');
title('Objective Space');
legend('Pareto Front');
axis tight; % 自动调整xlim和ylim，使其紧密包围数据
hold off;