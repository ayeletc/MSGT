from itertools import count
import numpy as np
import heapq

from sklearn.covariance import oas


class HMM:
    def __init__(self, states, init_prob, trans_mat, init_prob_1step=None, ts=1, trans_mat_1step=None, emit_mat=None):
        self.states = states
        self.init_prob = init_prob
        self.trans_mat = trans_mat
        self.ts = ts
        self.emit_mat = emit_mat
        # The following 2 field are relevant for viterbi algo using 2 time steps memory  
        self.trans_mat_1step = trans_mat_1step
        self.init_prob_1step = init_prob_1step

    def list_viterbi_algo_parallel_with_long_memory(self, obs, top_k):
        '''
        List Viterbi Algorithm for GT with Markovian priors.
        Assumption: Population with T items, was sampled from a Markovian model with S states.
        Inputs:
        - obs: Tx1 np.array, each element must be in {0,1,2}, where:
                            - 0 indicates the item is non-defective
                            - 1 indicates the item is defective
                            - 2 indicates that the item is possibly defective.
        - top_k: 1x1 int - LVA outputs the top_k most likely trajectories (top_k >= 1)

        Outputs:
        - path: top_KxT np.array, includes the top_k most likely population vectors.
        - path_probs: top_KxT np.array, the probabilities along the top_k most likely paths.
        - ml_prob: TxSxtop_k np.array of float, 
                    auxilliary array to save the higher top_k probailities for each state at each time step.
        - ml_prev_state: TxSxtop_k np.array of integers, 
                    auxilliary array to save the previous state for each state at each time step in the top_k most likely paths.
        '''
        n_states = self.trans_mat.shape[0]
        T = obs.shape[0]
        ml_prob = np.zeros((T, n_states, top_k))
        # ml_traj --> argmax 
        ml_prev_state = np.zeros((T, n_states, top_k), dtype=np.int16)
        # The ranking of multiple paths through a state
        rank = np.zeros((T, n_states, top_k), dtype=np.int16)

        # Initialize tt=ts-1 with init_prob, if obs = 222, 
        o = obs[:self.ts]
        possible_states = self.find_matching_states_to_obs(o, partially_obs=False)
        denomenator = np.sum([self.init_prob[int(st, 2)] for st in possible_states])
        for st in self.states:
            # k=0
            st_idx = int(st, 2)
            ml_prev_state[self.ts-1, st_idx, 0] = st_idx
            if st in possible_states:
                ml_prob[self.ts-1, st_idx, 0] = self.init_prob[st_idx] / denomenator
        
        for st_idx in range(n_states):
            # k>0
            for k in range(1, top_k):
                ml_prob[self.ts-1, st_idx, k] = 0.0
                ml_prev_state[self.ts-1, st_idx, k] = st_idx

        # Go forward calculating top k scoring paths
        # for each state s1 from previous state s2 at time step t 
        for tt in range(self.ts, T):
            ml_prob, ml_prev_state, rank = self.single_time_step_viterbi_with_single_memory(tt, top_k, obs[tt-self.ts+1:tt+1], ml_prob, ml_prev_state, rank)
        # Put all the last (tt=T) items on the stack
        h = []
        # Get all the top_k from all the states
        for s1 in range(n_states):
            for k in range(top_k):
                prob = ml_prob[T-1, s1, k] 
                # Sort by the probability, but retain what state it came from and the k
                heapq.heappush(h, (prob, s1, -k))

        # Then get sorted by the probability including its state and top_k
        h_sorted = [heapq.heappop(h) for _ in range(len(h))]
        h_sorted.reverse() # notice the sort of k is reversed - maybe we need to push -k instead of k ?

        # init blank path
        path = np.zeros((top_k, T), int)
        path_probs = np.zeros((top_k, T), float)

        # Now backtrack for k and each time step
        for k in range(top_k):
            # The maximum probability and the state it came from
            max_prob = h_sorted[k][0]
            state = h_sorted[k][1]
            rank_k = -h_sorted[k][2]
            # Assign to output arrays 
            path_probs[k, -1] = max_prob
            path[k, -1] = state # np.mod(state,2)

            # Then from T-1 down to 0 store the correct sequence for t+1
            for tt in reversed(range(self.ts-1,T-1)):
                # The next state and its rank
                next_state = path[k, tt+1]
                # Get the new state
                path[k, tt] = ml_prev_state[tt+1, next_state, rank_k]
                path_probs[k,tt] = ml_prob[tt+1, next_state, rank_k]
                # Get the correct ranking for the next phi
                rank_k = rank[tt+1, next_state, rank_k]
            # pass
            # a=5
            # print('old path:', path[k,:])
            path[k,:self.ts] = [int(b) for b in bin(int(path[k, tt]))[2:].zfill(self.ts)]
            # print('put ', [int(b) for b in bin(int(path[k, tt]))[2:].zfill(self.ts)])
            # print('new path:', path[k,:])
            # pass
        path = np.mod(path,2)
        return path, path_probs, ml_prob, ml_prev_state

    def list_viterbi_algo_parallel_with_deter(self, obs, top_k):
        '''
        Input:
        - obs - np.array Tx1 - observations durint T time steps. each obs is an index in [1,2,...,MAX_OBS_IDX]
        - top_k = int 1x1 - number of the best trajectories through the trellis
        Output:
        - map_trajectories - np.array Txtop_k - most probable trajectories
        - map_probabilities - np.array 1xtop_k - probabilities of the trajectories found
        
        Assuming: 
        1. state 0 is non_defective, state 1 is defective
        2. obs=0 means DND and obs=1 is DD
        '''
        
        n_states = np.shape(self.trans_mat)[0]
        T = np.shape(obs)[0]
        # assert (top_k <= np.power(n_states, T)), "k < n_states ^ top_k"
        # ml_prob --> highest probability of any path that reaches state ii
        ml_prob = np.zeros((T, n_states, top_k))
        # ml_traj --> argmax 
        ml_prev_state = np.zeros((T, n_states, top_k), dtype=np.int16)
        # The ranking of multiple paths through a state
        rank = np.zeros((T, n_states, top_k), dtype=np.int16)
        
        # Initialize tt = 0
        if obs[0] == 0 or obs[0] == 1:
            ml_prob[0, obs[0], 0] = 1.0
            ml_prob[0, 1-obs[0], 0] = 0.0
            for ii in range(n_states):
                ml_prev_state[0, ii, 0] = ii
            
            for ii in range(n_states):
                # Set the other options to 0 initially
                for k in range(1, top_k):
                    ml_prob[0, ii, k] = 0.0
                    ml_prev_state[0, ii, k] = ii
        else:
            for ii in range(n_states):
                ml_prob[0, ii, 0] = self.init_prob[ii] 
                ml_prev_state[0, ii, 0] = ii
                # Set the other options to 0 initially
                for k in range(1, top_k):
                    ml_prob[0, ii, k] = 0.0
                    ml_prev_state[0, ii, k] = ii

        # Go forward calculating top k scoring paths
        # for each state s1 from previous state s2 at time step t
        for tt in range(1, T):
            for s1 in range(n_states):
                h = []
                for s2 in range(n_states):
                    for k in range(top_k):
                        if obs[tt] == s1:# obsereved 0 and next state is 0 or obsereved 1 and next state is 1
                            prob = ml_prob[tt - 1, s2, k] # transition probability = 1 for DD&defecgive of DND and not defective
                        elif (obs[tt] == 1 and s1 == 0) or (obs[tt] == 0 and s1 == 1): # probability to be defective when observed DND = 0 
                            prob = 0
                        else: # prob to be defective/non defective when observed unknown obs==2 
                            prob = ml_prob[tt - 1, s2, k] * self.trans_mat[s2, s1]# * self.emit_mat[s1, obs[t]]
                        prev_state = s2
                        # Push the probability and state that led to it
                        heapq.heappush(h, (prob, prev_state))

                # Get the sorted list (by probabilities), descending order 
                h_sorted = [heapq.heappop(h) for _ in range(len(h))]
                h_sorted.reverse()
                # We need to keep a ranking if a path crosses a state more than once
                rank_dict = dict()
                # Retain the top k scoring paths and their probability and rankings
                for k in range(top_k):
                    ml_prob[tt, s1, k] = h_sorted[k][0]
                    ml_prev_state[tt, s1, k] = h_sorted[k][1]
                    state = h_sorted[k][1]
                    if state in rank_dict:
                        rank_dict[state] += 1
                    else:
                        rank_dict[state] = 0
                    rank[tt, s1, k] = rank_dict[state]

        # Put all the last (tt=T) items on the stack
        h = []
        # Get all the top_k from all the states
        for s1 in range(n_states):
            for k in range(top_k):
                prob = ml_prob[T-1, s1, k] 

                # Sort by the probability, but retain what state it came from and the k
                heapq.heappush(h, (prob, s1, -k))

        # Then get sorted by the probability including its state and top_k
        h_sorted = [heapq.heappop(h) for _ in range(len(h))]
        h_sorted.reverse() # notice the sort of k is reversed - maybe we need to push -k instead of k ?

        # init blank path
        path = np.zeros((top_k, T), int)
        path_probs = np.zeros((top_k, T), float)

        # Now backtrack for k and each time step
        for k in range(top_k):
            # The maximum probability and the state it came from
            max_prob = h_sorted[k][0]
            state = h_sorted[k][1]
            rank_k = -h_sorted[k][2]
            # Assign to output arrays 
            path_probs[k][-1] = max_prob
            path[k][-1] = state

            # Then from T-1 down to 0 store the correct sequence for t+1
            for tt in range(T - 2, -1, -1):
                # The next state and its rank
                next_state = path[k][tt+1]
                # Get the new state
                p = ml_prev_state[tt+1][next_state][rank_k]
                # Pop into output array
                path[k][tt] = p
                # Get the correct ranking for the next phi
                rank_k = rank[tt + 1][next_state][rank_k]

        return path, path_probs, ml_prob, ml_prev_state

    def find_matching_states_to_obs(self, obs, partially_obs=False):
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
        if partially_obs: # for initialization when we still have non ts previous observations
            while(len(possible_states[0]) < self.ts):
                new_possible_states = []
                for ps in possible_states:
                        new_possible_states.append('0'+ps)
                        new_possible_states.append('1'+ps)
                possible_states = new_possible_states
        return possible_states
    
    def single_time_step_viterbi_with_single_memory(self, tt, top_k, prev_obs, ml_prob, ml_prev_state, rank):
        # for initialization when we still have non ts previous observations
        # we consider the last observation and we check which last states are possible given obs[tt-1], and then we calculate 
        partially_obs = len(prev_obs) != self.ts
        possible_s2 = self.find_matching_states_to_obs(prev_obs, partially_obs=partially_obs) 
        for s2_idx, s2 in enumerate(self.states):
            h = []
            for s1_idx, s1 in enumerate(self.states):                
                # check what are the possible previsous states according the obs[tt-1]
                if s2 in possible_s2 and (s2[0:self.ts-1] == s1[-(self.ts-1):] or self.ts == 1): 
                    # meaning: if the previous obs was 011 we can continue with 011 because this is not a valid sequence.
                    # however, if we don't consider long memory and ts =1, we don't need to check that there is a sequence.
                    valid_state = True
                else:
                    valid_state = False
                for k in range(top_k):
                    if valid_state:
                        # if self.ts == 1 and prev_obs[-1] in [0,1]: # deterministic case
                        if prev_obs[-1] in [0,1]: # deterministic case
                            # if prev_obs[-1] == s2_idx:
                            if prev_obs[-1] == int(s2[-1]):
                                prob = ml_prob[tt-1, s1_idx, k]
                            else: # the other state gets prob 0, but it won't be in the valid states 
                                prob = 0
                                print('broken')
                        else: # not deterministic 
                            prob = ml_prob[tt-1, s1_idx, k] * self.trans_mat[s1_idx, int(s2[-1])] # long memory
                            # prob = ml_prob[tt-1, s1_idx, k] * self.trans_mat_1step[int(np.mod(s1_idx, 2)), int(s2[-1])] # 1 step memory only 
                    else:
                        prob = 0
                    # Push the probability and state that led to it
                    heapq.heappush(h, (prob, s1_idx, -k)) 
                    # reasonable sort, that priorize 0 over 7
            
            # Get the sorted list (by probabilities), descending order 
            h_sorted = [heapq.heappop(h) for _ in range(len(h))]
            h_sorted.reverse()
            # # We need to keep a ranking if a path crosses a state more than once
            # rank_dict = dict()
            # Retain the top k scoring paths and their probability and rankings
            for k in range(top_k):
                ml_prob[tt, s2_idx, k] = h_sorted[k][0]
                ml_prev_state[tt, s2_idx, k] = h_sorted[k][1]

                # state = h_sorted[k][1]
                # if state in rank_dict:
                #     rank_dict[state] += 1
                # else:
                #     rank_dict[state] = 0
                # rank[tt, s2_idx, k] = rank_dict[state]
                rank[tt, s2_idx, k] = -h_sorted[k][2]

        return ml_prob, ml_prev_state, rank

def test_viterbi_algo():
    init_prob = np.array([0.8, 0.2])
    trans_mat = np.array([[0.7, 0.3],[0.4, 0.6]])
    emit_mat = np.array([[0.2, 0.4, 0.4],[0.5, 0.4, 0.1]])
    states = np.array(['hot', 'cold'])
    model = HMM(states=states, init_prob=init_prob, trans_mat=trans_mat, emit_mat=emit_mat)
    obs = np.array([2,0,2])
    model.viterbi_algo(obs=obs)


if __name__ == '__main__':
    test_viterbi_algo()
    pass