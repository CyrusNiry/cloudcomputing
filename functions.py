import numpy as np
import pandas as pd
import pickle
import datetime
import os

FAMILY_NAME = ['general purpose', 'compute optimized', 'memory optimized', 'Instance with local SSDs', 'network enhanced', 'big data', 'shared memory']

class Resource:
    def __init__(self, index, family):
        self.index = index
        self.family = family
    
    def set_cost(self, R, V):
        self.R = R
        self.V = V

class Order:
    def __init__(self, family, arr_time, total_step, ddl, cost):
        self.family = family
        self.arr_time = arr_time
        self.total_step = total_step
        self.ddl = ddl
        self.cost = cost
        self.completed_step = 0

class Orders:
    def __init__(self, L, delta, W, arr_lambda):
        assert L.shape == delta.shape == W.shape
        self.I = len(L)

        self.t = 1

        self.L = L
        self.delta = delta
        self.W = W
        self.arr_lambda = arr_lambda
        self.order_list = {}

        for i in range(self.I):
            self.order_list[i] = []

    def __len__(self):
        res = 0
        for i in range(self.I):
            res += len(self.order_list[i])
        return res

    def arrive_for_one_type(self, order_type, arr_num):
        i = order_type
        for _ in range(arr_num):
            self.order_list[i].append(Order(i, self.t, self.L[i], self.delta[i], self.W[i]))

    def arrive(self):
        for i in range(self.I):
            arr_num = np.random.poisson(self.arr_lambda[i], 1)[0]
            self.arrive_for_one_type(i, arr_num)

    def pass_one_period(self):
        self.t += 1

    def record_sparse_x(self):
        x = {}
        for i in range(self.I):
            for order_index in range(len(self.order_list[i])):
                order = self.order_list[i][order_index]
                s = self.t - 1 - order.arr_time
                l = order.completed_step
                x[(i,s,l)] = x.get((i,s,l), 0) + 1
        return x

    def assign_resource_for_exploration(self):
        # only hire on-demand resources in the exploration phase
        x = np.zeros(self.I)
        for i in range(self.I):
            for order in self.order_list[i]:
                x[i] += 1
        return x
        # x = np.zeros((self.I, self.t-1, max(self.L)))
        # for i in range(self.I):
        #     for order_index in self.incompleted_order_index[i]:
        #         order = self.order_list[i][order_index]
        #         s = self.t - 1 - order.arr_time
        #         l = order.completed_step
        #         x[i,s,l] += 1
        # return x
    
    def serve(self, beta, j = None, j_ = None):
        success_record = {}
        cost = 0
        for i in range(self.I):
            delete_index = []
            for order_index in range(len(self.order_list[i])):
                order = self.order_list[i][order_index]
            # for order in self.order_list[i]:
            #     if order.completed_step < order.total_step:
                s = self.t - 1 - order.arr_time
                l = order.completed_step
                if j_ is not None:
                    j = int(j_(i, s, l))
                serve_step = np.random.binomial(1, beta[i,j])
                if (i, j) not in success_record:
                    success_record[(i, j)] = 0
                # print(f'({i,j}), {beta[i,j]}, {serve_step}')
                success_record[(i, j)] += serve_step
                order.completed_step += serve_step
                if order.completed_step >= order.total_step:
                    delete_index.append(order_index)
                else:
                    if s > order.ddl:
                        cost += order.cost
            for index in sorted(delete_index, reverse=True):
                del self.order_list[i][index]
        return cost, success_record

    def cal_waiting_cost(self):
        cost = 0
        for i in range(self.I):
            for order in self.order_list[i]:
                s = self.t - 1 - order.arr_time
                if s > order.ddl:
                    cost += order.cost
        return cost

def create_params(file_path, res_path, arr_lambda_l=1, arr_lambda_u=2):
    params = {}
    df = pd.read_excel(file_path, header=None)

    # read the excel file
    resources = []
    R = []
    V = []
    for index, row in df.iterrows():
        resource = Resource(index, int(row[2])-1)
        resource.set_cost(row[1], row[0])
        R.append(row[1]) # reserved
        V.append(row[0]) # on-demande
        resources.append(resource)
    R = np.array(R)
    V = np.array(V)

    # 1. initialize
    ## resources
    J = len(resources)
    B = np.random.uniform(0, 1, J)
    B.sort()

    ## orders
    I = len(FAMILY_NAME)
    arr_lambda = np.random.uniform(arr_lambda_l,  arr_lambda_u, I)
    L = np.random.randint(25, 50, I)
    delta = np.random.randint(75, 150, I)
    W = np.random.uniform(df.min()[1], df.max()[1], I)
    D = np.random.uniform(0, 2, I)

    ## success rate
    beta = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            beta[i,j] = B[j] ** D[i] if i == resources[j].family else (B[j]/2) ** D[i]

    params['resources'] = resources
    params['R'] = R
    params['V'] = V
    params['J'] = J
    params['I'] = I
    params['arr_lambda'] = arr_lambda
    params['L'] = L
    params['delta'] = delta
    params['W'] = W
    params['beta'] = beta
    with open(res_path, 'wb') as f:
        pickle.dump(params, f)
        

def load_params(param_path):
    if not os.path.exists(param_path):
        raise FileExistsError('You should create parameters first!')
    with open(param_path, 'rb') as f:
        params = pickle.load(f)
    return params
