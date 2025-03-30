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

    def set_transitions_probabilities_for_2steps(self):
        '''
        States:
            0: dnd+dnd   (00)
            1: dnd+dd    (01)
            2: dd+dnd    (10)
            3: dd+dd     (11)
        '''
        pi_B = self.init_prob_1step[1]
        determinisitic_obs = [0,1,3,4]
        # mapping deterministic observations to states:
        determinisitic_obs_cur_state_dict = {0:0, # dnd+dnd
                                            1:1,  # dnd+dd
                                            3:2,  # dd+dnd
                                            4:3}  # dd+dd

        ambiguity_obs = [2, 5, 6, 7, 8]

        # map observations to the possible current states
        ambiguity_obs_cur_state_dict = {2: [0, 1], # obs dnd+unknown    => dnd+dnd / dnd+dd
                                        5: [2, 3], # obs dd+unknown     => dd+dnd  / dd+dd
                                        6: [0, 2], # obs unknown+dnd    => dnd+dnd / dd+dnd
                                        7: [1, 3]} # obs unknown+dd     => dnd+dd  / dd+dd
        
        # map observations to the transition probabilities to the possible states in ambiguity_obs_cur_state_dict
        # for example: when we get observation 5, the state may be either 2 with probability self.trans_mat_1step[1,0]
        # or it may be 3 with probability self.trans_mat_1step[1,1]
        # ambiguity_obs_prob_dict = { 2: [self.trans_mat_1step[0,0],                  self.trans_mat_1step[0,1]], 
        #                             5: [self.trans_mat_1step[1,0],                  self.trans_mat_1step[1,1]],
        #                             6: [self.trans_mat_1step[0,0],                  (pi_B/(1-pi_B))*self.trans_mat_1step[1,0]],
        #                             7: [(pi_B/(1-pi_B))*self.trans_mat_1step[1,0],  self.trans_mat_1step[1,1]]} 
        
        semi_ambiguity_obs_in_2nd_state_prob_dict = { 2: [self.trans_mat_1step[0,0],        self.trans_mat_1step[0,1]], 
                                                    5: [self.trans_mat_1step[1,0],          self.trans_mat_1step[1,1]]}
        '''
        In observations 6,7 we have ambiguity in the 1st state (unknown+___)
        Therefore we set the transition probability according to the previous state
        For example: 
        Obs 6 is unknown+dnd and may indicate either state 0 or 2. If the previous state was 0 or 2 (00/10 that end with 0), 
        then the probability that the current state is 0 equals to Ptrans0->0, and current state is 2 w.p. Ptrans0->1. 
        That explains the first 2 rows in this dictionary.
        If the preivous state was 1 or 3 (ends with 1) then w.p. Ptrans1->0 the current state is 00 (state 0), 
        and w.p. Ptrans1->1 the current state is 10 (state 2). That explains the rows 3,4 in the dictionary.
        
        Exactly the same happens with obs 7 (with the same probabilities).
        semi_ambiguity_obs_in_obs6_prob_dict = {0:  [self.trans_mat_1step[0,0],       self.trans_mat_1step[0,1]],
                                                2:  [self.trans_mat_1step[0,0],       self.trans_mat_1step[0,1]],
                                                1:  [self.trans_mat_1step[1,0],       self.trans_mat_1step[1,1]],
                                                3:  [self.trans_mat_1step[1,0],       self.trans_mat_1step[1,1]]}
        
        semi_ambiguity_obs_in_obs7_prob_dict = {0:  [self.trans_mat_1step[0,0],       self.trans_mat_1step[0,1]],
                                                2:  [self.trans_mat_1step[0,0],       self.trans_mat_1step[0,1]],
                                                1:  [self.trans_mat_1step[1,0],       self.trans_mat_1step[1,1]],
                                                3:  [self.trans_mat_1step[1,0],       self.trans_mat_1step[1,1]]}
        '''

        semi_ambiguity_obs_in_1st_state_prob_dict = {0:  [self.trans_mat_1step[0,0],       self.trans_mat_1step[0,1]],
                                                    2:  [self.trans_mat_1step[0,0],       self.trans_mat_1step[0,1]],
                                                    1:  [self.trans_mat_1step[1,0],       self.trans_mat_1step[1,1]],
                                                    3:  [self.trans_mat_1step[1,0],       self.trans_mat_1step[1,1]]}

        return determinisitic_obs, determinisitic_obs_cur_state_dict, \
                ambiguity_obs, ambiguity_obs_cur_state_dict, \
                semi_ambiguity_obs_in_2nd_state_prob_dict, semi_ambiguity_obs_in_1st_state_prob_dict
    '''
    # Classic Parallel LVA Decoder using heaps and rankings
    def list_viterbi_algo_parallel(self, obs, topK):
        if topK == 1:
            return self.viterbi_algo(obs)

        nStates = np.shape(self.emit_mat)[0]
        T = np.shape(obs)[0]
        # assert (topK <= np.power(nStates, T)), "k < nStates ^ topK"
        # ml_prob --> highest probability of any path that reaches state ii
        ml_prob = np.zeros((T, nStates, topK))
        # ml_traj --> argmax 
        ml_prev_state = np.zeros((T, nStates, topK), dtype=np.int16)
        # The ranking of multiple paths through a state
        rank = np.zeros((T, nStates, topK), dtype=np.int16)

        for ii in range(nStates):
            ml_prob[0, ii, 0] = self.init_prob[ii] #* self.emit_mat[ii, obs[0]]
            ml_prev_state[0, ii, 0] = ii

            # Set the other options to 0 initially
            for k in range(1, topK):
                ml_prob[0, ii, k] = 0.0
                ml_prev_state[0, ii, k] = ii

        # Go forward calculating top k scoring paths
        # for each state s1 from previous state s2 at time step t
        for t in range(1, T):
            for s1 in range(nStates):
                h = []
                for s2 in range(nStates):
                    for k in range(topK):
                        prev_state = s2
                        prob = ml_prob[t - 1, s2, k] * self.trans_mat[s2, s1]# * self.emit_mat[s1, obs[t]]
                        # Push the probability and state that led to it
                        heapq.heappush(h, (prob, prev_state))

                # Get the sorted list (by probabilities), descending order 
                h_sorted = [heapq.heappop(h) for _ in range(len(h))]
                h_sorted.reverse()
                # We need to keep a ranking if a path crosses a state more than once
                rankDict = dict()
                # Retain the top k scoring paths and their phi and rankings
                for k in range(topK):
                    ml_prob[t, s1, k] = h_sorted[k][0]
                    ml_prev_state[t, s1, k] = h_sorted[k][1]
                    state = h_sorted[k][1]
                    if state in rankDict:
                        rankDict[state] += 1
                    else:
                        rankDict[state] = 0
                    rank[t, s1, k] = rankDict[state]

        # Put all the last items on the stack
        h = []
        # Get all the topK from all the states
        for s1 in range(nStates):
            for k in range(topK):
                prob = ml_prob[T - 1, s1, k]

                # Sort by the probability, but retain what state it came from and the k
                heapq.heappush(h, (prob, s1, k))

        # Then get sorted by the probability including its state and topK
        h_sorted = [heapq.heappop(h) for _ in range(len(h))]
        h_sorted.reverse()

        # init blank path
        path = np.zeros((topK, T), int)
        path_probs = np.zeros((topK, T), float)

        # Now backtrack for k and each time step
        for k in range(topK):
            # The maximum probability and the state it came from
            max_prob = h_sorted[k][0]
            state = h_sorted[k][1]
            rankK = h_sorted[k][2]
            # Assign to output arrays
            path_probs[k][-1] = max_prob
            path[k][-1] = state

            # Then from T-1 down to 0 store the correct sequence for t+1
            for t in range(T - 2, -1, -1):
                # The next state and its rank
                nextState = path[k][t+1]
                # Get the new state
                p = ml_prev_state[t+1][nextState][rankK]
                # Pop into output array
                path[k][t] = p
                # Get the correct ranking for the next phi
                rankK = rank[t + 1][nextState][rankK]

        return path, path_probs, ml_prob, ml_prev_state
    '''

    def list_viterbi_algo_parallel_with_deter_2steps(self, obs, top_k):
        ''' 
        Input:
        - obs - np.array Tx1 - observations durint T time steps. each obs is an index in [1,2,...,MAX_OBS_IDX]
        - top_k = int 1x1 - number of the best trajectories through the trellis
        Output:
        - map_trajectories - np.array Txtop_k - most probable trajectories
        - map_probabilities - np.array 1xtop_k - probabilities of the trajectories found
        
        Assuming: 
        1. state 0 is [non_defective, non_defective], state 1 is [non_defective, defective], 
            state 2 is [defective, non_defective], state 3 is [defective, defective], 
        2. obs=0,...,8. 0 <=> [dnd, dnd], 1<=>[dnd, dd], 3<=>[dnd, unknown], ...,8<=>[unknown, unknown]

        '''
        if top_k == 1:
            return self.viterbi_algo_adjusted_to_GE(obs)

        n_states = np.shape(self.trans_mat)[0]
        T = np.shape(obs)[0]
        # assert (top_k <= np.power(n_states, T)), "k < n_states ^ top_k"
        # ml_prob --> highest probability of any path that reaches state ii
        ml_prob = np.zeros((T, n_states, top_k))
        # ml_traj --> argmax 
        ml_prev_state = np.zeros((T, n_states, top_k), dtype=np.int16)
        # The ranking of multiple paths through a state
        rank = np.zeros((T, n_states, top_k), dtype=np.int16)
        
        determinisitic_obs, determinisitic_obs_cur_state_dict, \
        ambiguity_obs, ambiguity_obs_cur_state_dict, \
        semi_ambiguity_obs_in_2nd_state_prob_dict, semi_ambiguity_obs_in_1st_state_prob_dict = \
            self.set_transitions_probabilities_for_2steps()

        # Initialize tt = 0
        if obs[0] in determinisitic_obs:
            cur_state = determinisitic_obs_cur_state_dict[obs[0]]
            ml_prob[0, cur_state, 0] = 1.0 # and the probability to other state equals 0
            for ii in range(n_states):
                # TODO: I think all top K should pass in the deterministic state
                ml_prev_state[0, ii, 0] = ii

            for ii in range(n_states):
                # Set the other options to 0 initially
                for k in range(1, top_k):
                    ml_prob[0, ii, k] = 0.0
                    ml_prev_state[0, ii, k] = ii
            
        elif obs[0] == 8: # full ambiguity
            for ii in range(n_states):
                ml_prob[0, ii, 0] = self.init_prob[ii] 
                ml_prev_state[0, ii, 0] = ii
                # Set the other options to 0 initially
                for k in range(1, top_k):
                    ml_prob[0, ii, k] = 0.0
                    ml_prev_state[0, ii, k] = ii
            
        elif obs[0] in [6,7]: # semi-ambiguity in 1st state
            possible_states_list = ambiguity_obs_cur_state_dict[obs[0]]
            # probabilities_list_1st_state_ambiguity = semi_ambiguity_obs_in_1st_state_prob_dict[obs[0]]
            for st in possible_states_list:
                ml_prob[0, st, 0] = self.init_prob[st] / \
                                    (self.init_prob[possible_states_list[0]]+self.init_prob[possible_states_list[1]])
                ml_prev_state[0, st, 0] = st
            
            for ii in range(n_states):
                if st in possible_states_list:
                    continue # already configured
                ml_prev_state[0, ii, 0] = ii
            
            for k in range(1, top_k):
                    ml_prob[0, ii, k] = 0.0
                    ml_prev_state[0, ii, k] = ii
        
        elif obs[0] in [2,5]: # semi-ambiguity in 2nd state
            possible_states_list = ambiguity_obs_cur_state_dict[obs[0]]
            probabilities_list_2nd_state_ambiguity = semi_ambiguity_obs_in_2nd_state_prob_dict[obs[0]]
            for ii, st in enumerate(possible_states_list):
                ml_prob[0, st, 0] = probabilities_list_2nd_state_ambiguity[ii]
                ml_prev_state[0, st, 0] = st
            
            for ii in range(n_states):
                if st in possible_states_list:
                    continue # already configured
                ml_prev_state[0, ii, 0] = ii

            for k in range(1, top_k):
                    ml_prob[0, ii, k] = 0.0
                    ml_prev_state[0, ii, k] = ii
        
        # Go forward calculating top k scoring paths
        # for each state s1 from previous state s2 at time step t
        for tt in range(1, T):
            for s1 in range(n_states): # current state
                h = []
                
                if obs[tt] in determinisitic_obs:
                    for s2 in range(n_states):
                        for k in range(top_k):
                            if s1 == determinisitic_obs_cur_state_dict[obs[tt]]:
                                prob = 1.0 * ml_prob[tt - 1, s2, k]
                            else:
                                prob = 0.0
                            prev_state = s2
                            heapq.heappush(h, (prob, prev_state))
                
                elif obs[tt] == 8:
                    for s2 in range(n_states):
                        for k in range(top_k):
                            prob = ml_prob[tt - 1, s2, k] * self.trans_mat[s2, s1]
                            prev_state = s2
                            heapq.heappush(h, (prob, prev_state))

                elif obs[tt] in [2, 5]:
                    possible_states_list = ambiguity_obs_cur_state_dict[obs[tt]]
                    probabilities_list = semi_ambiguity_obs_in_2nd_state_prob_dict[obs[tt]]
                    for s2 in range(n_states): # previous state
                        for k in range(top_k):
                            if s1 in possible_states_list:
                                prob = ml_prob[tt - 1, s2, k] * probabilities_list[possible_states_list.index(s1)]
                            else:
                                prob = 0.0
                            prev_state = s2 # not s1?
                            heapq.heappush(h, (prob, prev_state))
                
                elif obs[tt] in [6, 7]:
                    possible_states_list = ambiguity_obs_cur_state_dict[obs[tt]]
                    for s2 in range(n_states): # previous state
                        probabilities_list = semi_ambiguity_obs_in_1st_state_prob_dict[s2]
                        for k in range(top_k):
                            if s1 in possible_states_list:
                                prob = ml_prob[tt - 1, s2, k] * probabilities_list[possible_states_list.index(s1)]
                            else:
                                prob = 0.0
                            prev_state = s2 # not s1?
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
        h_sorted.reverse()

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


    def list_viterbi_algo_parallel_with_long_memory(self, obs, top_k):
        n_states = self.trans_mat.shape[0]
        T = obs.shape[0]
        ml_prob = np.zeros((T+1, n_states, top_k))
        # ml_traj --> argmax 
        ml_prev_state = np.zeros((T+1, n_states, top_k), dtype=np.int16)
        # The ranking of multiple paths through a state
        rank = np.zeros((T+1, n_states, top_k), dtype=np.int16)

        # Initialize tt = 0,...,ts-1 
        # TODO Im not sure about this initialization, maybe I need
        # to shorten T to be T-ts and then I can initialize only 1 step here
        # I need to make sure that the states are not clashing 
        
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
        # for now implement only 1 time step back, TODO later more than that
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
            # Assign to output arrays - yul - canceled
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
        # if top_k == 1:
        #     return self.viterbi_algo_adjusted_to_GE(obs)

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
            # Assign to output arrays - yul - canceled
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
    '''
    def viterbi_algo(self, obs):
        
        # Input:
        # - obs - np.array Tx1 - observations durint T time steps. each obs is an index in [1,2,...,MAX_OBS_IDX]
        # Output:
        # - map_trajectories - np.array Tx1 - most probable trajectories
        # - map_probabilities - np.array 1x1 - probabilities of the trajectories found
        
        V = [{}]
        for st_idx, st in enumerate(self.states):
            # print(st)
            # print(V)
            d = {"prob": self.init_prob[st_idx], #* self.emit_mat[st_idx][obs[0]], 
                "prev": None}
            V[0][st] = d
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            # print(' === t = {} === '.format(t))
            V.append({})
            for st_idx, st in enumerate(self.states):
                max_tr_prob = V[t - 1][self.states[0]]["prob"] *  \
                                self.trans_mat[0][st_idx] #*  \
                                # self.emit_mat[st_idx][obs[t]] 
                # print('calc P(' + st + ', obs{} | '.format(obs[t]) + self.states[0] + ')  = {} and set as max'.format(max_tr_prob))
                prev_st_selected = self.states[0]

                prev_st_idx = 1
                for prev_st in self.states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * \
                                self.trans_mat[prev_st_idx][st_idx] #* \
                                # self.emit_mat[st_idx][obs[t]] # consider P(st1, obs1|st0)
                    # print('calc P(' + st + ', obs{} | '.format(obs[t]) + prev_st + ') = {}'.format(tr_prob))
                    if tr_prob > max_tr_prob:
                        # print('change max')
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                    
                    prev_st_idx += 1
                
                max_prob = max_tr_prob
                V[t][st] = {"prob": max_prob, 
                            "prev": prev_st_selected}
                # print()

        # for line in self.dptable(V):
        #     print(line)

        T = obs.shape[0]
        # map_trajectories = np.zeros((T))
        # map_probabilities = np.zeros((1))
        opt = []
        max_prob = 0.0
        best_st = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        # set outputs
        replacements = {"hot":0, "cold":1, "defective":1, "non_defective":0}
        replacer = replacements.get  # For faster gets.
        opt = [replacer(n, n) for n in opt]

        map_trajectories = opt
        map_probabilities = max_prob
        
        # print ("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)
        
        return map_trajectories, map_probabilities
    '''

    def viterbi_algo_adjusted_to_GE(self, obs):
        ''' 
        Input:
        - obs - np.array Tx1 - observations durint T time steps. each obs is an index in [1,2,...,MAX_OBS_IDX]
        Output:
        - map_trajectories - np.array Tx1 - most probable trajectories
        - map_probabilities - np.array 1x1 - probabilities of the trajectories found
        
        Assuming: 
        1. state 0 is non_defective, state 1 is defective
        2. obs=0 means DND and obs=1 is DD

        '''
        T = obs.shape[0]
        V = [{}]
        # Initialize, no prev state
        if obs[0] == 0 or obs[0] == 1:
            V[0][self.states[obs[0]]] = {"prob": 1, "prev": None}
            V[0][self.states[1-obs[0]]] = {"prob": 0, "prev": None}
        else:
            for st_idx, st in enumerate(self.states):
                V[0][st] = {"prob": self.init_prob[st_idx], "prev": None}

        # Run Viterbi when t > 0
        for tt in range(1, T+1):
            # print(' === t = {} === '.format(t))
            V.append({})
            for st_idx, st in enumerate(self.states):                
                max_tr_prob = -1
                prev_st_selected = self.states[0]
                for prev_st_idx, prev_st in enumerate(self.states):
                    if obs[tt-1] == prev_st_idx:# and prev_st_idx or obs[t] == 1:
                        tr_prob = 1
                    elif (obs[tt-1] == 1 and prev_st_idx == 0) or (obs[tt-1] == 0 and prev_st_idx == 1):
                        tr_prob = 0
                    else:
                        tr_prob = V[tt - 1][prev_st]["prob"] * self.trans_mat[prev_st_idx][st_idx]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                
                max_prob = max_tr_prob
                V[tt][st] = {"prob": max_prob, "prev": prev_st_selected}
            
        opt = []
        max_prob = 0.0
        best_st = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        # set outputs
        replacements = {"hot":0, "cold":1, "defective":1, "non_defective":0}
        replacer = replacements.get  # For faster gets.
        opt = [replacer(n, n) for n in opt]

        map_trajectories = opt[:-1]
        map_probabilities = max_prob
        
        # print ("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)
        
        return map_trajectories, map_probabilities

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
                    heapq.heappush(h, (prob, s1_idx)) # yul: TODO: flip the s1_idx to -s1_idx to make a more 
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
'''
    def viterbi_algo_one_traj(self, obs):
        #  Basic Viterbi, return 1 trajectory only
        # Input:
        # - obs - np.array Tx1 - observations durint T time steps. each obs is an index in [1,2,...,MAX_OBS_IDX]
        # - n_best_traj - int 1x1 - number of best trajectories we want to find, default is 1
        
        V = [{}]
        for st_idx, st in enumerate(self.states):
            print(st)
            print(V)
            d = {"prob": self.init_prob[st_idx] * self.emit_mat[st_idx][obs[0]], 
                        "prev": None}
            V[0][st] = d
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            print(' === t = {} === '.format(t))
            V.append({})
            for st_idx, st in enumerate(self.states):
                max_tr_prob = V[t - 1][self.states[0]]["prob"] *  \
                                self.trans_mat[0][st_idx] *  \
                                self.emit_mat[st_idx][obs[t]] 
                print('calc P(' + st + ', obs{} | '.format(obs[t]) + self.states[0] + ')  = {} and set as max'.format(max_tr_prob))
                prev_st_selected = self.states[0]

                prev_st_idx = 1
                for prev_st in self.states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * \
                                self.trans_mat[prev_st_idx][st_idx] * \
                                self.emit_mat[st_idx][obs[t]] # consider P(st1, obs1|st0)
                    print('calc P(' + st + ', obs{} | '.format(obs[t]) + prev_st + ') = {}'.format(tr_prob))
                    if tr_prob > max_tr_prob:
                        print('change max')
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                    
                    prev_st_idx += 1
                
                max_prob = max_tr_prob
                V[t][st] = {"prob": max_prob, 
                            "prev": prev_st_selected}
                print()

        for line in self.dptable(V):
            print(line)

        opt = []
        max_prob = 0.0
        best_st = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        print ("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)

    def dptable(self, V):
        # Print a table of steps from dictionary
        yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
        for state in V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state] ["prob"]) for v in V)
'''


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