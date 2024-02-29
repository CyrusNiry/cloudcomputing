import numpy as np
from functions import *
import time
from scipy.stats import norm
import sys
import os
import matplotlib.pyplot as plt
times_lists = {
    'data12': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    'data36': [3, 6, 9, 12, 15, 18, 21, 24],
    'data48': [3, 6, 9, 12, 15, 18]
}



def cal_ave_cost(res_f):
    total_cost = 0
    exp_num = 0
    for line in res_f:
        cost = line.strip()
        if cost:
            cost = float(cost)
            exp_num += 1
            total_cost += cost
    if exp_num < 50:
        print(res_f.name)
    return total_cost / exp_num

def f(i, s, l):
    if s >= delta[i]:
        return f_state[i, delta[i], l]
    elif f_state[i, s, l] == -1:
        if l < L[i]:
            res = min(R+beta[i,:]*f(i,s+1,l+1)+(1-beta[i,:])*f(i,s+1,l))
        else:
            res = min(R+f(i,s+1,l))
        f_state[i, s, l] = res
        return res
    else:
        return f_state[i, s, l]




for data_type in times_lists:
    params = load_params(f'./params/{data_type}.pickle')

    # 1. initialize
    ## resources
    J = int(params['J'])
    resources = params['resources']
    R = params['R']
    V = params['V']

    ## orders
    I = int(params['I'])
    arr_lambda = params['arr_lambda']
    L = params['L']
    delta = params['delta']
    W = params['W']

    ## success rate
    beta = params['beta']

    ## time param.
    # tau = 60 * 10
    tau = 1 * 30 * 24 * 60
    K2 = 1 / 8

    ## other param.
    beta_lower = 0.001
    beta_upper = 0.999

    f_state = np.zeros((I, max(delta)+1, max(L)+1)) - 1
    for i in range(I):
        for s in range(delta[i], max(delta)+1):
            for l in range(L[i]+1):
                f_state[i, s, l] = (L[i] - l) * min([(R[j]+W[i])/beta[i, j] for j in range(J)])


    # Lower Bound per time period
    LB_ = 0
    for i in range(I):
        LB_ += arr_lambda[i] * f(i, 0, 0)
    regrets = []
    LBRs = []
    LBRs_c = []
    times_list = np.array(times_lists[data_type])
    for times in times_list:
        T = times * tau
        res_f = open(f'./res/costs_{times}_{data_type}', 'r')
        ave_cost = cal_ave_cost(res_f)
        res_f.close()
        res_f = open(f'./res/costs_clairvoyant_{times}_{data_type}', 'r')
        ave_cost_c = cal_ave_cost(res_f)
        res_f.close()
        ave_regret = ave_cost - ave_cost_c
        regrets.append(ave_regret)
        LB = T * LB_
        LB_rate = (ave_cost - LB) / LB
        LB_rate_c = (ave_cost_c - LB) / LB
        # print(ave_cost_c, LB, data_type, times)
        LBRs.append(LB_rate)
        LBRs_c.append(LB_rate_c)
        # ln_R.append(np.log(ave_regret))


    plt.title(f'Regrets lambda~U[{data_type[4]}, {data_type[5]}]')
    plt.plot(times_list*tau, regrets)
    plt.show()
    plt.title(f'(Cost-LB)/LB lambda~U[{data_type[4]}, {data_type[5]}]')
    plt.plot(times_list*tau, LBRs)
    plt.show()
    plt.title(f'(Clarvoyant_Cost-LB)/LB lambda~U[{data_type[4]}, {data_type[5]}]')
    plt.plot(times_list*tau, LBRs_c)
    plt.show()

    