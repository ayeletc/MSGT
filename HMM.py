from itertools import count
import numpy as np
import heapq

# from sklearn.covariance import oas


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
        n_states = self.trans_mat.shape[0]
        T = obs.shape[0]
        ml_prob = np.zeros((T+1, n_states, top_k))
        # ml_traj --> argmax 
        ml_prev_state = np.zeros((T+1, n_states, top_k), dtype=np.int16)
        # The ranking of multiple paths through a state
        rank = np.zeros((T+1, n_states, top_k), dtype=np.int16)

        # Initialize tt = 0,...,ts-1 
        # in tt=0 we only have the initial probabilities
        for ii in range(n_states):
            ml_prev_state[0, ii, 0] = ii
            ml_prob[0, ii, 0] = self.init_prob[ii]
        for ii in range(n_states):
            # Set the other options to 0 initially
            for k in range(1, top_k):
                ml_prev_state[0, ii, k] = ii
                ml_prob[0, ii, k] = 0.0    

        # for t=1,...,ts-1 we should predict the previous state using 1,2,... memory steps 
        for tt in range(1,self.ts):
            o = [obs[tt-1]] if tt==1 else obs[:tt]
            ml_prob, ml_prev_state, rank = self.single_time_step_viterbi_with_single_memory(tt, top_k, o, ml_prob, ml_prev_state, rank)

        # Go forward calculating top k scoring paths
        # for each state s1 from previous state s2 at time step t 
        for tt in range(self.ts, T+1):
            ml_prob, ml_prev_state, rank = self.single_time_step_viterbi_with_single_memory(tt, top_k, obs[tt-self.ts:tt], ml_prob, ml_prev_state, rank)
            '''
            possible_s2 = self.find_matching_states_to_obs(obs[tt-self.ts:tt])
            for s2_idx, s2 in enumerate(self.states):
                h = []
                for s1_idx, s1 in enumerate(self.states):
                    if s2 in possible_s2 and s2[:-1]== s1[1:]:
                        valid_state = True
                    else:
                        valid_state = False
                    for k in range(top_k):
                        if valid_state:
                            prob = ml_prob[tt-1, s1_idx, k] * self.trans_mat[s1_idx, int(s2[-1])]
                            # heapq.heappush(h, (prob, s1_idx))
                        # TRY: if prob=0 dont push to the heap, 
                        # we should always have at least 1 possible state and then should at least
                        # have top_k probabilities in the heap
                        else:
                            prob = 0      
                        heapq.heappush(h, (prob, s1_idx))                    
                            
                # Get the sorted list (by probabilities), descending order 
                h_sorted = [heapq.heappop(h) for _ in range(len(h))]
                h_sorted.reverse()
                # We need to keep a ranking if a path crosses a state more than once
                rank_dict = dict()
                # Retain the top k scoring paths and their probability and rankings
                for k in range(top_k):
                    ml_prob[tt, s2_idx, k] = h_sorted[k][0]
                    ml_prev_state[tt, s2_idx, k] = h_sorted[k][1]
                    state = h_sorted[k][1]
                    if state in rank_dict:
                        rank_dict[state] += 1
                    else:
                        rank_dict[state] = 0
                    rank[tt, s2_idx, k] = rank_dict[state]
            '''
        # Put all the last (tt=T) items on the stack
        h = []
        # Get all the top_k from all the states
        for s1 in range(n_states):
            for k in range(top_k):
                prob = ml_prob[T, s1, k] 
                # Sort by the probability, but retain what state it came from and the k
                heapq.heappush(h, (prob, s1, -k))

        # Then get sorted by the probability including its state and top_k
        h_sorted = [heapq.heappop(h) for _ in range(len(h))]
        h_sorted.reverse() # notice the sort of k is reversed - maybe we need to push -k instead of k ?

        # init blank path
        path = np.zeros((top_k, T+1), int)
        path_probs = np.zeros((top_k, T+1), float)

        # Now backtrack for k and each time step
        for k in range(top_k):
            # The maximum probability and the state it came from
            max_prob = h_sorted[k][0]
            state = h_sorted[k][1]
            rank_k = -h_sorted[k][2]
            # Assign to output arrays 
            path_probs[k][-1] = max_prob
            path[k][-1] = state#np.mod(state,2)

            # Then from T-1 down to 0 store the correct sequence for t+1
            for tt in reversed(range(1,T)):
                # The next state and its rank
                next_state = path[k][tt+1]
                # Get the new state
                p = ml_prev_state[tt+1][next_state][rank_k]
                # Pop into output array
                path[k][tt] = p#np.mod(p,2)
                # Get the correct ranking for the next phi
                rank_k = rank[tt+1][next_state][rank_k]

        return np.mod(path[:,1:],2), path_probs[:,1:], ml_prob[1:,:,:], ml_prev_state[1:,:,:]

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
            # # check if the observation is deterministic in case we only check 1 time step
            # if self.ts == 1 and prev_obs in ['0','1']:
            #     for k in range(top_k):
            #         prob = ml_prob[tt-1, int(prev_obs), k]
            #         heapq.heappush(h, (prob, int(prev_obs)))
            #         heapq.heappush(h, (0, 1-int(prev_obs)))
            # else:
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
                            if prev_obs[-1] == s2_idx:
                                prob = ml_prob[tt-1, s1_idx, k]
                            else: # the other state gets prob 0, but it won't be in the valid states 
                                prob = 0
                        else: # not deterministic 
                            prob = ml_prob[tt-1, s1_idx, k] * self.trans_mat[s1_idx, int(s2[-1])]
                    else:
                        prob = 0
                    # Push the probability and state that led to it
                    heapq.heappush(h, (prob, s1_idx)) 
                    # reasonable sort, that priorize 0 over 7
            
            # Get the sorted list (by probabilities), descending order 
            h_sorted = [heapq.heappop(h) for _ in range(len(h))]
            h_sorted.reverse()
            # We need to keep a ranking if a path crosses a state more than once
            rank_dict = dict()
            # Retain the top k scoring paths and their probability and rankings
            for k in range(top_k):
                ml_prob[tt, s2_idx, k] = h_sorted[k][0]
                ml_prev_state[tt, s2_idx, k] = h_sorted[k][1]
                state = h_sorted[k][1]
                if state in rank_dict:
                    rank_dict[state] += 1
                else:
                    rank_dict[state] = 0
                rank[tt, s2_idx, k] = rank_dict[state]
        # ===Check if all the probs are 0===
        # probs = [p for (p,_) in h]
        # if np.sum(probs) == 0:
        # for ii in range(self.T+1):
        # if np.sum(ml_prob[tt,:,:])==0:        
        #     print('all probs are 0 in {}'.format(tt))
        #     pass
        return ml_prob, ml_prev_state, rank

if __name__ == '__main__':
    pass