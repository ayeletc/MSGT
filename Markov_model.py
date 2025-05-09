import math
import random
import numpy as np
from utils import *
from HMM import HMM
from matplotlib import pyplot as plt

INIT_PROB_N500_K3 = np.array([0.85, 0.003, 0.02, 0.004, 0.07, 0.002, 0.05, 0.001])
INIT_PROB_N1024_K8 = np.array([0.85, 0.003, 0.02, 0.004, 0.07, 0.002, 0.05, 0.001])
INIT_PROB_N10000_K13 = np.array([0.85, 0.003, 0.02, 0.004, 0.07, 0.002, 0.05, 0.001])
TRANS_MAT_N500_K3 = np.array([
                            [0.998, 0.002],
                            [0.05,  0.95],# 0.15,0.85
                            [0.999, 0.001],
                            [0.35,   0.65], #0.5,0.5
                            [0.998, 0.002],
                            [0.75,  0.25], 
                            [0.999, 0.001],
                            [0.95,   0.05]])
TRANS_MAT_N1024_K8 = np.array([
                                [0.996, 0.004],
                                [0.05,  0.95],
                                [0.999, 0.001],
                                [0.45,  0.55],
                                [0.998, 0.002],
                                [0.75,  0.25],
                                [0.999, 0.001],
                                [0.9,   0.1]])

TRANS_MAT_N10000_K13 = np.array([
                            [0.9996, 0.0004],
                            [0.05,  0.95],
                            [0.999, 0.001],
                            [0.05,  0.95],
                            [0.998, 0.002],
                            [0.15,  0.85],
                            [0.999, 0.001],
                            [0.9,   0.1]])
class Markov_model: 
    def __init__(self, memory_time_steps, init_prob, trans_mat):
        assert trans_mat.shape[0]==2**memory_time_steps, "Transition matrix does not match memory time steps"
        assert trans_mat.shape[1] == 2, "Transition matrix has more than 2 coloums"

        self.memory_time_steps = memory_time_steps
        self.init_prob = init_prob
        self.trans_mat = trans_mat
        self.possible_states = ["{0:b}".format(st).zfill(self.memory_time_steps) for st in range(int(2**self.memory_time_steps))]
        pass
    
    def sample_population(self, N, max_bad=np.inf, debug=False):
        iter = 0
        U = np.zeros((1,N), dtype=np.uint8)
        while np.sum(U) != max_bad:
            init_state = np.random.choice(self.possible_states, 1, p=self.init_prob)[0]
            U[0,:self.memory_time_steps] = [int(st) for st in init_state]
            assert N >= self.memory_time_steps, "N is too small"
            for ii in range(self.memory_time_steps, N):
                prev_st = bitlist2int(list(U[0,ii-self.memory_time_steps:ii]))
                next_st = np.random.choice([0,1], 1, p=self.trans_mat[prev_st,:])[0]
                U[0,ii] = int(next_st)
            # print(np.sum(U))
            if debug:
                iter += 1
        if debug:
            return U, iter
        else:
            return U
    

    # def calculate_lower_bound_markov(self, N, Pe=0.0)
    # def calc_Pw(self, permute, DD2, DND1)        
    def calc_Pw(self, N, permute, DD2, DND1):
        ts = self.memory_time_steps
        U = np.zeros((N,), np.uint8)
        U[permute] = 1
        
        # initial state
        first_state = bitlist2int(list(U[:ts]))
        obs = []
        for ii in range(ts):            
            if ii in DND1:
                obs.append(0)
            elif ii in DD2:
                obs.append(1)
            else: 
                obs.append(2)
        possible_states = find_matching_states_to_obs(obs)
        denominator = np.sum([self.init_prob[int(st,2)] for st in possible_states])
        Pw = self.init_prob[first_state] / denominator

        # next states:
        for ii in range(ts, N):
            if (ii not in DND1) and (ii not in DD2):
                prev_st = bitlist2int(list(U[ii-ts:ii]))
                Pw *= self.trans_mat[prev_st, U[ii]]
        return Pw

    def model_as_hmm(self):
        return HMM(states=self.possible_states, init_prob=self.init_prob,
                   trans_mat=self.trans_mat, ts=self.memory_time_steps)

    def model_as_hmm_with_1step_memory(self):
        n_states = len(self.possible_states)
        even_states = [st for st in range(n_states) if np.mod(st,2) == 0]
        odd_states = [st for st in range(n_states) if np.mod(st,2) == 1]
        
        # calc 2x1 initial probabilities
        prob_init0 = np.sum([self.init_prob[st] for st in even_states])
        prob_init1 = np.sum([self.init_prob[st] for st in odd_states])
            
        init_prob_1step = np.array([prob_init0, prob_init1])

        # calc 2x2 transition matrix
        P0even, P1even = 0, 0
        P0odd, P1odd = 0, 0
        for st in even_states:
            P0even += self.trans_mat[st,0]
            P1even += self.trans_mat[st,1]
        Peven = P0even+P1even
        for st in odd_states:
            P0odd += self.trans_mat[st,0]
            P1odd += self.trans_mat[st,1]
        Podd = P0odd+P1odd
        trans_mat_1step = np.array([[P0even/Peven, P1even/Peven],
                                   [P0odd/Podd, P1odd/Podd]])
        assert(np.sum(trans_mat_1step[0,:]) == 1)
        assert(np.sum(trans_mat_1step[1,:]) == 1)
        
        return HMM(states=['0','1'], init_prob=init_prob_1step, 
                   trans_mat=trans_mat_1step, ts=1)

def find_matching_states_to_obs(obs):
    possible_states = ['']
    for idx, s in enumerate(obs):
        if s != 2:
            possible_states = [ps+str(obs[idx]) for ps in possible_states]
        else:
            new_possible_states = []
            for ps in possible_states:
                new_possible_states.append(ps+'0')
                new_possible_states.append(ps+'1')
            possible_states = new_possible_states
    return possible_states


def sample_population_for_N500_K3_ts3(N, K, markov_model=None):
    if markov_model is None:
        init_prob = INIT_PROB_N500_K3
        trans_mat = TRANS_MAT_N500_K3
        N = 500
        K = 3
        ts = 3
        markov_model = Markov_model(ts, init_prob, trans_mat)
    return markov_model.sample_population(N,K), markov_model

def sample_population_for_N1024_K8_ts3(N, K, markov_model=None):
    if markov_model is None:
        init_prob = INIT_PROB_N1024_K8
        trans_mat = TRANS_MAT_N1024_K8
        N = 1024
        K = 8
        ts = 3
        markov_model = Markov_model(ts, init_prob, trans_mat)
    return markov_model.sample_population(N,K), markov_model

def sample_population_for_N10000_K13_ts3(N, K, markov_model=None):
    if markov_model is None:
        init_prob = INIT_PROB_N10000_K13
        trans_mat = TRANS_MAT_N10000_K13
        N = 10000
        K = 13
        ts = 3
        markov_model = Markov_model(ts, init_prob, trans_mat)
    return markov_model.sample_population(N,K), markov_model

def test_sample_population(N=500):
    if N == 500:
        K = 3
        init_prob = INIT_PROB_N500_K3
        trans_mat = TRANS_MAT_N500_K3
    elif N == 1024:
        K = 8
        init_prob = INIT_PROB_N1024_K8
        trans_mat = TRANS_MAT_N1024_K8
    elif N == 10000:
        K = 13
        init_prob = INIT_PROB_N10000_K13
        trans_mat = TRANS_MAT_N10000_K13
    else:
        print('error')
    
    ts = 3
    nmc = 100
    markov_model = Markov_model(ts, init_prob, trans_mat)
    s = np.zeros((nmc,))
    for ii in range(nmc):
        U, iter = markov_model.sample_population(N, K, debug=True)
        s[ii] = iter
    
    plt.hist(s)
    plt.show()
    pass

if __name__ == '__main__':
    test_sample_population(N=10000)
    pass