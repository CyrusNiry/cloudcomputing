import numpy as np
from functions import *
import time
from scipy.stats import norm
import sys
import os

args = sys.argv
# if len(args) == 2:
#     file_path = args[1]
#     create_params(file_path)
times = int(args[2])
param_path = './params/' + args[1]
exp_tag = args[1].split('.')[0]
params = load_params(param_path)

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
# times = 12
T = times * tau
K2 = 1 / 8
eta = int(np.ceil(np.log(T) / K2))

## other param.
beta_lower = 0.001
beta_upper = 0.999

## open file recording costs
if not os.path.exists('./res'):
    os.mkdir('./res')


def experiment():
    ## t = 1
    orders = Orders(L, delta, W, arr_lambda)
    orders.arrive()
    cost = 0
    cost_l = 0
    v = np.zeros((I, J))
    n = np.zeros((I, J))

    # Optimal policy for Clairvoyant Problem
    f_state = np.zeros((I, max(delta)+1, max(L)+1)) - 1
    j_state = np.zeros((I, max(delta)+1, max(L)+1)) - 1
    lambda_state = np.zeros((I, max(delta)+1, max(L))) - 1
    for i in range(I):
        for s in range(delta[i], max(delta)+1):
            for l in range(L[i]+1):
                f_state[i, s, l] = (L[i] - l) * min([(R[j]+W[i])/beta[i, j] for j in range(J)])
                j_state[i, s, l] = np.argmin((R+W[i])/beta[i, :])

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

    def j_(i, s, l):
        if s >= delta[i]:
            return j_state[i, delta[i], l]
        elif j_state[i, s, l] == -1:
            if l < L[i]:
                res = np.argmin(R+beta[i,:]*f(i,s+1,l+1)+(1-beta[i,:])*f(i,s+1,l))
            else:
                res = np.argmin(R+f(i,s+1,l))
            j_state[i, s, l] = res
            return res
        else:
            return j_state[i, s, l]
        
    def lambda_(i, s, l):
        if lambda_state[i, s, l] == -1:
            if s == 0 and l == 0:
                res = arr_lambda[i]
            else:
                if 1 <= s and l == 0:
                    res = (1 - beta[i, int(j_(i,s-1,l))]) * lambda_(i, s-1, l)
                elif (s >= 1 and s <= L[i] - 1) and l == s:
                    res = beta[i, int(j_(i,s-1,l-1))] * lambda_(i, s-1, l-1)
                elif (l >= 1 and l <= min(L[i],s) - 1) and s >= 2:
                    res = beta[i, int(j_(i,s-1,l-1))] * lambda_(i, s-1, l-1) + (1 - beta[i, int(j_(i,s-1,l))]) * lambda_(i, s-1, l)
            lambda_state[i, s, l] = res
            return lambda_state[i, s, l]
        else:
            return lambda_state[i, s, l]


    # Formal steps
    last_time = time.time()
    M = np.zeros(J)
    miu = np.zeros(J)
    for i in range(I):
        for l in range(L[i]):
            for s in range(l, delta[i]):
                j = int(j_(i, s, l))
                miu[j] += lambda_(i, s, l)
    for j in range(J):
        for i in range(I):
            temp = 0
            if j == np.argmin((R+W[i])/beta[i, :]):
                for l in range(L[i]):
                    for l_prime in range(l+1):
                        temp += lambda_(i, delta[i], l_prime)
            miu[j] += temp / beta[i, j]
        M[j] = miu[j] + np.sqrt(miu[j]) * norm.ppf((V[j]-R[j])/V[j])

    m = np.zeros(J)
    lambda_state = np.zeros((I, max(delta)+1, max(L))) - 1

    for t in range(2, T+1):
        if t % int(tau/20) == 0:
            current_time = time.time()
            print(f'Currently t={t}. It costs {current_time-last_time}s from the last record')
            last_time = current_time
        orders.pass_one_period()
        x = orders.record_sparse_x()
        u = np.zeros(J)
        for coor in x:
            i, s, l = coor
            j = int(j_(i, s, l))
            u[j] += x[coor]
        for j in range(J):
            m[j] = u[j] - M[j] if u[j] - M[j] > 0 else 0
            # hiring cost
            # TODO m and M is not an integer now
            cost += M[j] * resources[j].R + m[j] * resources[j].V
        cost_t, success_record = orders.serve(beta=beta, j_=j_)
        orders.arrive()
        cost += cost_t
        # res_file.write(f'{t} {cost}\n')

    print('There are')
    for i in range(I):
        print(f'  {len(orders.order_list[i])} orders incompleted of type {i}.')
    print('\n---------------------\n')
    print('Starting the remaining periods...')
    print(f'Currently t={t} and it has already reached T={T}.')
    print('There are')
    for i in range(I):
        print(f'  {len(orders.order_list[i])} orders incompleted of type {i}.')

    while len(orders) > 0:
        m = np.zeros(J)
        t += 1
        if t % 10 == 0:
            print('There are')
            for i in range(I):
                print(f'  {len(orders.order_list[i])} orders incompleted of type {i}.')
        orders.pass_one_period()
        x = orders.record_sparse_x()
        for coor in x:
            i, s, l = coor
            j = int(j_(i, s, l))
            m[j] += x[coor]
        # hiring cost
        # TODO m and M is not integer now
        for j in range(J):
            cost += m[j] * resources[j].V
        # waiting cost
        cost_t, success_record = orders.serve(beta=beta, j_=j_)
        cost += cost_t
        # res_file.write(f'{t} {cost}\n')

    print('Completed!')
    return cost

for exp_num in range(50):
    cost = experiment()
    res_file = open(f'./res/costs_clairvoyant_{times}_{exp_tag}', 'a+')
    res_file.write(str(cost)+'\n')
    res_file.close()