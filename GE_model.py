import math
import random
import numpy as np
from utils import *
from HMM import HMM


class GE_model:
    def __init__(self, s, q, pi_B):
        self.s = s
        self.q = q
        self.pi_B = pi_B
        self.probabilities_to_bad_dict = self.calc_conditional_probability_GE() 
        self.num_of_permutations = None

    def calculate_num_of_permutations_by_entropy(self, K, T, nPD):
        p = np.log(2) / K
        # prob_error_DD = 1-p*(1-p)**(nPD-1)
        # entropy_error_DD = -prob_error_DD * np.log2(prob_error_DD) - (1-prob_error_DD) * np.log2(1-prob_error_DD)
        prob_error_COMA = 1-p*(1-p)**K
        entropy_error_COMA = -prob_error_COMA * np.log2(prob_error_COMA) - (1-prob_error_COMA) * np.log2(1-prob_error_COMA)
        self.num_of_permutations = np.ceil(2 ** (T * entropy_error_COMA)).astype(np.int64)

    def sample_gilbert_elliot_channel(self, N, max_bad=np.inf):
        # for GT with fixed num of K : if there are more than max_bad bad items, return false and don't complete the chain

        # P = np.array([[1-self.q, self.q], [self.s, 1-self.s]])
        # pi_G = self.s/(self.s+self.q) # probability of being in State G
        # pi_B = self.q/(self.s+self.q) # probability of being in State B
    
        channel_statef = np.zeros((N,))
        channel_stater = np.zeros((N,))
        
        goodf = random.random() > self.pi_B
        goodr = random.random() > self.pi_B          
        
        num_of_bad = 0
        for ii in range(N):
            # set goodf = 1 and goodr = 1 if next step is bad (=erasure/defective)
            if goodf == 1 and goodr == 1:
                goodf = random.random() > self.q  
                goodr = random.random() > self.q  
            elif goodf == 1 and goodr == 0:
                goodf = random.random() > self.q  
                goodr = random.random() > 1-self.s
            elif goodf == 0 and goodr == 1:
                goodf = random.random() > 1-self.s 
                goodr = random.random() > self.q 
            elif goodf == 0 and goodr == 0:
                goodf = random.random() > 1-self.s 
                goodr = random.random() > 1-self.s

            channel_statef[ii] = goodf
            channel_stater[ii] = goodr
            if goodf == 0:
                num_of_bad += 1
                if num_of_bad > max_bad:
                    return None, None
        return 1-channel_statef,1-channel_stater


    def calc_conditional_probability_GE(self):
        probabilities_to_bad_dict = {}
        # given previous item is defective
        probabilities_to_bad_dict['previous_is_defective'] = 1-self.s
        # given previous item is not defective
        probabilities_to_bad_dict['previous_is_not_defective'] = self.q
        # no given prior about previous item
        probabilities_to_bad_dict['first_item'] = self.pi_B
        probabilities_to_bad_dict['no_prior_given'] = self.pi_B# * (1-self.s) + (1-self.pi_B) * self.q
        return probabilities_to_bad_dict

    def get_conditional_probability_GE_old_didnot_work(self, item, DD2, DND1):
        if item-1 in DD2:
            # given previous item is defective
            return self.probabilities_to_bad_dict['previous_is_defective']
        elif item-1 in DND1:
            # given previous item is not defective
            return self.probabilities_to_bad_dict['previous_is_not_defective']
        else:
            # no given prior about previous item
            if item == 0: # first item, the probabiity is not conditional
                return self.probabilities_to_bad_dict['first_item']
            else:
                return self.probabilities_to_bad_dict['no_prior_given']

    def get_conditional_probability_GE(self, item, DD2, DND1):
        if item in DD2:
            return 1
        
        elif item in DND1:
            return 0
        
        else: # status is unknown
            if item == 0 or ( item > 0 and item-1 not in DD2 and item-1 not in DND1 ): # no prior 
                return self.probabilities_to_bad_dict['no_prior_given']

            elif item-1 in DD2:
                return self.probabilities_to_bad_dict['previous_is_defective']
            
            elif item-1 in DND1:
                    return self.probabilities_to_bad_dict['previous_is_not_defective']
            
        return None
    
    def calculate_lower_bound_GE(self, N, Pe=0.0):
        # calculate lb by joint entropy
        # H[U1,...,UN] = H(U1)+H(U2|U1)+...+H(UN|UN-1)
        return self.calculate_entropy(N) * (1-Pe)
        # lb = 0
        # for ii in range(1,N):

    def calculate_entropy2(self, N, K):
        H = lambda x: -x*np.log2(x)-(1-x)*np.log2(1-x)
        return (N-K) * H(self.q) + K * H(self.s)
    
    def calculate_entropy(self, N):
        H = lambda x: -x*np.log2(x)-(1-x)*np.log2(1-x)
        return N * ( (1-self.pi_B) * H(self.q) + self.pi_B * H(self.s))

    def sort_comb_by_priors_GE_cut_by_entropy(self, N, K, T, nPD, DD2, DND1, unknowns, permutation_factor=50):
        '''
        1. calculate Np the number of permutations we want to check - 2^( T*H(Perror_dd) )
        1.1. calculate Perror_dd

        2. define array of probabilities of permutations Pw 
            and array Mp (Np x K_left) of the permutation with the highest Pw
        
        3. calculate the permutations one by one, for each one:
        3.1. calculate Pw
        3.2 if this Pw high enough put it in the array Mp of the permutations (keep both arrays sorted)
        '''
        K_left = K - len(DD2)
        # if K_left == 1:
        #     permute =
        #     return Pw, high_prob_permutations
        ''' 
        Perror in DD:
        (1-p) + p*(Phidden)
        Phidden = 1-P(there are no more PDs on) = 1-P( (#PD-1) items are off ) = 1-(1-p)^(#PD-1)
        '''
        num_of_permutations_binomial = math.comb(len(unknowns), K_left)
        if self.num_of_permutations is None:
            self.calculate_num_of_permutations_by_entropy(K, T, nPD)
        # print('num_of_permutations_binomial', num_of_permutations_binomial)
        # if num_of_permutations_binomial < 500 and num_of_permutations_binomial > self.num_of_permutations:
        #     save_permutations = num_of_permutations_binomial
        # else:
        #     save_permutations = np.min([self.num_of_permutations, num_of_permutations_binomial])
        if permutation_factor == -1:
            num_permutations_to_save = num_of_permutations_binomial
        else:
            if num_of_permutations_binomial > self.num_of_permutations and num_of_permutations_binomial > self.num_of_permutations * permutation_factor :
                num_permutations_to_save = permutation_factor*self.num_of_permutations
            else:
                num_permutations_to_save = num_of_permutations_binomial#self.num_of_permutations
        Pw = np.zeros((num_permutations_to_save,))
        high_prob_permutations = np.zeros((num_permutations_to_save, K_left))
        num_of_iterations_in_sort = num_of_permutations_binomial
        # if num_of_permutations_binomial < self.num_of_permutations:
        #     # built the iterator on all the possible options and sort
        # else:
        # calculate permutations one by one:
        iterable = unknowns
        r = K_left
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            print('Do something1')
            return
        indices = list(range(r))
        comb = tuple(pool[i] for i in indices)
        # print(comb)
        prob_permute = self.calc_Pw_fixed(N, comb, DD2, DND1)
        Pw, high_prob_permutations = add_new_value_and_symbol_keep_sort(Pw, high_prob_permutations, prob_permute, comb)
        # iteration = 1
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                # print('#iterations in sort = ', iteration)
                return high_prob_permutations.astype(np.int64), Pw, num_of_iterations_in_sort
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            comb = tuple(pool[i] for i in indices)
            # print(comb)
            prob_permute = self.calc_Pw_fixed(N, comb, DD2, DND1)
            Pw, high_prob_permutations = add_new_value_and_symbol_keep_sort(Pw, high_prob_permutations, prob_permute, comb)
            # iteration += 1        

    def calc_Pw(self, permute, DD2, DND1):        
        # probability of the first item in the permutation:
        if permute[0] in DD2:
            Pw = 1
        elif permute[0] in DND1:
            Pw = 0
        else:
            Pw = self.pi_B
        # multiply transition probabilities
        for jj in range(1,len(permute)):
            p_item_is_defective_given_previous = self.get_conditional_probability_GE(permute[jj-1], DD2, DND1)
            Pw *= p_item_is_defective_given_previous
        return Pw

    def calc_Pw_fixed(self, N, permute, DD2, DND1):  # take in account all n items
        permute = list(permute) + DD2
        probability_per_item = np.zeros((N,))
        for item in range(N):
            if item in permute:
                probability_per_item[item] = self.get_conditional_probability_GE(item, DD2, DND1)
            else:
                probability_per_item[item] = 1 - self.get_conditional_probability_GE(item, DD2, DND1)
        Pw = np.prod(probability_per_item)
        return Pw

    def model_as_hmm(self, K, T, nPD, p, ver_states=True):
        if ver_states:
            states = np.array(['non_defective', 'defective'])
        else:
            states = np.array(['0', '1'])
        init_prob = np.array([1-self.pi_B, self.pi_B])
        trans_mat = np.array([  [1-self.q, self.q], 
                                [self.s, 1-self.s]  ])
        P_PD_notDefective = (1-p*(1-p)**(K-1))**T # probability to be PD given its not defective (occlusion in COMA)
        P_PD_defective = (1-p*(1-p)**(nPD-1))**T # probability to be PD given its defective (after DD algo)
        emit_mat = np.array([[1-P_PD_notDefective,   0,                  P_PD_notDefective], 
                            [0,                      1-P_PD_defective,   P_PD_defective]])
        hmm_model = HMM(states=states, init_prob=init_prob, trans_mat=trans_mat, emit_mat=emit_mat)
        return hmm_model
    
    def model_as_hmm_with_long_memory(self, K, T, nPD, p, ts=2):
        if ts == 1:
            return self.model_as_hmm(K, T , nPD, p)
        ## set states
        n_states = int(2**ts)
        hmm_1ts = self.model_as_hmm(K, T, nPD, p)
        states_idx = np.arange(n_states)
        states_binary = [bin(x)[2:].zfill(ts) for x in states_idx]
        # states_names = 
        
        ## calculate transition matrix:
        prev_trans_mat = hmm_1ts.trans_mat
        for cur_ts in range(2,ts+1): # if ts==2 do one iteration
            cur_n_states = int(2**cur_ts)
            cur_trans_mat = np.zeros((cur_n_states, 2))

            for half in ['top','bottom']:
                if half == 'top':
                    row = 0
                else:
                    row = 1
                aux_mat = np.zeros((int(cur_n_states/2),2))
                for ii in range(aux_mat.shape[0]):
                    for jj in range(2):
                            if ii < aux_mat.shape[0]/2:
                                aux_mat[ii,jj] = hmm_1ts.trans_mat[row,0]
                            else:
                                aux_mat[ii,jj] = hmm_1ts.trans_mat[row,1]
                if half == 'top':
                    cur_trans_mat[:int(cur_n_states/2), :] = aux_mat * prev_trans_mat
                else:
                    cur_trans_mat[int(cur_n_states/2):, :] = aux_mat * prev_trans_mat
            
            prev_trans_mat = cur_trans_mat

        cur_trans_mat = cur_trans_mat / np.sum(cur_trans_mat, axis=1)[:, np.newaxis]
        
        ## calculate initial step
        init_prob = np.ones((n_states,))
        init_prob[:int(n_states/2)] = hmm_1ts.init_prob[0]
        init_prob[int(n_states/2):] = hmm_1ts.init_prob[1]
        for ii, state in enumerate(states_binary):
            for t in range(ts-1):
                s1 = int(state[t])
                s2 = int(state[t+1])
                init_prob[ii] *= hmm_1ts.trans_mat[s1, s2]

        return HMM(states=states_binary, init_prob=init_prob, trans_mat=prev_trans_mat, ts=ts, trans_mat_1step=hmm_1ts.trans_mat, emit_mat=None)


    def model_as_hmm_with_2_steps_memory(self, K, T, nPD, p):
        states = np.array(['non_defective | non_defective', 'non_defective | defective', 
                            'defective | non_defective', 'defective | defective'])
        init_prob_1step = np.array([1-self.pi_B, self.pi_B]) 
        
        a = 1-self.q
        b = self.q
        c = self.s
        d = 1-self.s
        trans_mat_1step =   np.array([[1-self.q, self.q], 
                                    [self.s, 1-self.s]])
        trans_mat_2steps =  np.array([[a*a,  a*b,   b*c,    b*d],
                                    [c*a,   c*b,   d*c,    d*d],
                                    [a**2,  a*b,   b*c,    b*d],
                                    [c*a,   c*b,   c*d,    d*d]])
        init_prob_1step =   np.array([1-self.pi_B, self.pi_B])
        init_prob_2steps =  np.array([init_prob_1step[0]*a, init_prob_1step[0]*b,
                                        init_prob_1step[1]*c, init_prob_1step[1]*d])
                                    
        P_PD_notDefective = (1-p*(1-p)**(K-1))**T # probability to be PD given its not defective (after DND algo)
        P_PD_defective = (1-p*(1-p)**(nPD-1))**T # probability to be PD given its defective (after DD algo)
        emit_mat = np.array([[1-P_PD_notDefective,   0,                  P_PD_notDefective], 
                            [0,                      1-P_PD_defective,   P_PD_defective]])
        hmm_model = HMM(states=states, init_prob=init_prob_2steps, trans_mat=trans_mat_2steps,
                        init_prob_1step=init_prob_1step, trans_mat_1step=trans_mat_1step, emit_mat=None)
        return hmm_model

    
    def parse_2step_to_1step(self, seq_2step):
        n2 = seq_2step.shape[0]
        n1 = int(n2*2)
        seq_1step = np.zeros((n1,))
        for ii in range(n2):
            seq_1step[ii*2:ii*2+2] = convert_int_to_base(seq_2step[ii], 2)
        return seq_1step

    if __name__ == '__main__':
        pass
    
