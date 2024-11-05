import torch
from LCMOEA import LCMOEA, problem as problem_module
from CEC2020_prob import CMOP2Problem, CMOP3Problem

def main():
    # 创建问题实例
    probl = CMOP3Problem() 
    problem = problem_module.Problem(probl.objectives, probl.dim, probl.UBLB, probl.equality_constraints, probl.inequality_constraints, probl.www)
    # 设置参数
    N = 50  # 种群规模
    max_iter = 400000  # 最大评价次数
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 实例化 LCMOEA，设置横纵坐标范围
    xlim = (0, 2)
    ylim = (0, 2)
    # xlim = (0, 25)
    # ylim = (0, 25)
    # algorithm = LCMOEA.LCMOEA(problem, N, max_iter, device, True, xlim=xlim, ylim=ylim)
    
    # 运行算法
    algorithm = LCMOEA.LCMOEA(problem, N = N, max_iter = max_iter, device = device, FE_k = 0.2, is_plot=True, xlim=xlim, ylim=ylim)
    pareto_front = algorithm.run()

    # # 输出结果
    # print("Pareto front solutions:")
    # for solution in pareto_front:
    #     print(solution)
    
if __name__ == "__main__":
    main()
    # & E:/Anaconda/evn/wzrPytorch/python.exe e:/Anaconda/Projects/Optimal/LCMOEA/used_py/LCMOEA_test.py