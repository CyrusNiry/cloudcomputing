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
# times = 15
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
    est_beta = np.zeros((I, J))
    v = np.zeros((I, J))
    n = np.zeros((I, J))

    # ---------------------
    # 2. Exploration Phase
    print('Starting the exploration phase...')
    for j in range(J):
        x = np.zeros(I)
        start_t = time.time()
        for t in range(j*eta+2, (j+1)*eta+2):
            orders.pass_one_period()
            x_t = orders.assign_resource_for_exploration()
            x += x_t
            cost_t, success_record = orders.serve(beta=beta, j=j)
            orders.arrive()
            # waiting cost
            cost += cost_t
            # hiring cost and update
            cost += x_t.sum() * resources[j].V
            # record completions
            if success_record is not None:
                for i, j in success_record:
                    v[i, j] += success_record[(i, j)]
            # res_file.write(f'{t} {cost}\n')
        end_t = time.time()
        
        # estimate
        start_x = time.time()
        # for i in range(I):
        #     for t in range(j*eta+2, (j+1)*eta+2):
        #         for s in range(t-1):
        #             for l in range(min(L[i]-1, s)+1):
        #                 n[i, j] += x[(i, t, s, l)]
        for i in range(I):
            n[i, j] = x[i]
            est_beta[i, j] = v[i, j] / n[i, j]
            est_beta[i, j] = max(min(est_beta[i, j], beta_upper), beta_lower)
            # n_ij = x[i, j*eta+2:(j+1)*eta+2, :t-1, :min(L[i]-1, s)+1].sum()
            # est_beta[i, j] = v[i, j] / n_ij
            
        end_x = time.time()
        print(f'j:{j}, t:{t}, t_cost:{end_t-start_t}, x_cost:{end_x-start_x}')

        print(beta[:, j])
        print(est_beta[:, j])
        # print(f'j:{j}, t:{t}')
    print(f'After the exploration phase(t={t}),there are')
    for i in range(I):
        print(f'  {len(orders.order_list[i])} orders incompleted of type {i}.')

    # ---------------------
    # 3. Exploitation Phase
    print('\n---------------------\n')
    print('Starting the exploitation phase...')
    # First get f_i(s,l) and j`(i,s,l) when s >= delta_i
    f_state = np.zeros((I, max(delta)+1, max(L)+1)) - 1
    j_state = np.zeros((I, max(delta)+1, max(L)+1)) - 1
    for i in range(I):
        for s in range(delta[i], max(delta)+1):
            for l in range(L[i]+1):
                f_state[i, s, l] = (L[i] - l) * min([(R[j]+W[i])/est_beta[i, j] for j in range(J)])
                j_state[i, s, l] = np.argmin((R+W[i])/est_beta[i, :])

    def f(i, s, l):
        if s >= delta[i]:
            return f_state[i, delta[i], l]
        elif f_state[i, s, l] == -1:
            if l < L[i]:
                res = min(R+est_beta[i,:]*f(i,s+1,l+1)+(1-est_beta[i,:])*f(i,s+1,l))
            else:
                res = 0
            f_state[i, s, l] = res
            return res
        else:
            return f_state[i, s, l]

    def j_(i, s, l):
        if s >= delta[i]:
            return j_state[i, delta[i], l]
        elif j_state[i, s, l] == -1:
            if l < L[i]:
                res = np.argmin(R+est_beta[i,:]*f(i,s+1,l+1)+(1-est_beta[i,:])*f(i,s+1,l))
            j_state[i, s, l] = res
            return res
        else:
            return j_state[i, s, l]
        
    def lambda_(i, h, s, l):
        if lambda_state[i, h, s, l] == -1:
            if s == 0 and l == 0:
                res = arr_lambda[i]
            else:
                if 1 <= s and l == 0:
                    res = (1 - est_beta_[i, int(j_(i,s-1,l)), h]) * lambda_(i, h, s-1, l)
                elif (s >= 1 and s <= L[i] - 1) and l == s:
                    res = est_beta_[i, int(j_(i,s-1,l-1)), h] * lambda_(i, h, s-1, l-1)
                elif (l >= 1 and l <= L[i] - 1) and s >= l + 1:
                    res = est_beta_[i, int(j_(i,s-1,l-1)), h] * lambda_(i, h, s-1, l-1) + (1 - est_beta_[i, int(j_(i,s-1,l)), h]) * lambda_(i, h, s-1, l)
            lambda_state[i, h, s, l] = res
            return lambda_state[i, h, s, l]
        else:
            return lambda_state[i, h, s, l]

    # Formal steps
    H = int((T-1-J*eta)/tau)
    miu_hat = np.zeros((J, H+1))
    # M = np.zeros((J, H))
    # m = np.zeros((J, tau))
    last_time = time.time()
    est_beta_ = np.zeros((I, J, H))
    v_ = np.zeros((I, J, H))
    n_ = np.zeros((I, J, H))
    lambda_state = np.zeros((I, H, max(delta)+1, max(L))) - 1

    for h in range(H):
        for t in range(J*eta+h*tau+2, J*eta+(h+1)*tau+2):
            M = np.zeros(J)
            m = np.zeros(J)
            if t % int(tau/20) == 0:
                current_time = time.time()
                print(f'Currently t={t}. It costs {current_time-last_time}s from the last record')
                last_time = current_time
            orders.pass_one_period()
            x = orders.record_sparse_x()
            u = np.zeros(J)
            # x = np.transpose(x.nonzero())
            for coor in x:
                i, s, l = coor
                j = int(j_(i, s, l))
                u[j] += x[coor]
                n_[i, j, h] += x[coor]
            for j in range(J):
                M[j] = miu_hat[j, h] + np.sqrt(miu_hat[j, h]) * norm.ppf((V[j]-R[j])/V[j])
                # print('M', M[j, h], miu_hat[j, h], np.sqrt(miu_hat[j, h]), norm.ppf((V[j]-R[j])/V[j]))
                m[j] = u[j] - M[j] if u[j] - M[j] > 0 else 0
                # hiring cost
                # TODO m and M is not an integer now
                cost += M[j] * resources[j].R + m[j] * resources[j].V
            cost_t, success_record = orders.serve(beta=beta, j_=j_)
            orders.arrive()
            if success_record is not None:
                for i, j in success_record:
                    v_[i, j, h] += success_record[(i, j)]
            # waiting cost
            cost += cost_t
            # res_file.write(f'{t} {cost}\n')
            

            
        # Update the truncated mean estimator of beta_ij
        for i in range(I):
            for j in range(J):
                est_beta_[i, j, h] = v_[i, j, :h+1].sum() / n_[i, j, :h+1].sum() if v_[i, j, :h+1].sum() != 0 else 0
                # if v_[i, j, :h+1].sum() != 0:
                #     print(f'({i},{j}):',v_[i, j, :h+1].sum()/n_[i, j, :h+1].sum(), beta[i, j])
                est_beta_[i, j, h] = max(min(beta_upper, est_beta_[i, j, h]), beta_lower)
        
        # Compute miu_hat

        for i in range(I):
            for l in range(L[i]):
                for s in range(l, delta[i]):
                    j = int(j_(i, s, l))      
                    miu_hat[j, h+1] += lambda_(i, h, s, l)
        for j in range(J):
            for i in range(I):
                temp = 0
                if j == np.argmin((R+W[i])/est_beta[i, :]):
                    for l in range(L[i]):
                        for l_prime in range(l + 1):
                            temp += lambda_(i, h, delta[i], l_prime)
                miu_hat[j, h+1] += temp / est_beta_[i, j, h]
            # print('miu_hat',miu_hat[j, h+1])
        print('There are')
        for i in range(I):
            print(f'  {len(orders.order_list[i])} orders incompleted of type {i}.')
        print(f'--------h:{h}, t:{t}--------')
    print('\n---------------------\n')
    print('Starting the remaining periods...')
    print(f'Currently t={t} and it should reach T={T} next.')
    t_ = t
    for t in range(t_+1, T+1):
        m = np.zeros(J)
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
        orders.arrive()
        # res_file.write(f'{t} {cost}\n')
        
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
    res_file = open(f'./res/costs_{times}_{exp_tag}', 'a+')
    res_file.write(str(cost)+'\n')
    res_file.close()
