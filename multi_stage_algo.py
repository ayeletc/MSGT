import os
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime
import itertools
import random
import numpy.matlib
from sample_population import *
from plotters import *
from calc_bounds_and_num_of_tests import *
from Markov_model import *
import scipy.io


# pid = os.getpid()
# os.system('sudo renice -n -20 -p ' + str(pid))
#%% Config simulation
N                   = 500
vecK                = [3]
nmc                 = 10
enlarge_tests_num_by_factors = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] #[0.25, 0.5, 0.75, 1, 1.25] #[0.5, 0.25, 0.5, 0.75, 1, 1.5]#[0.75, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2]# [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3, 1.7, 2] #[0.85, 0.9, 0.95, 1, 1.25, 1.5, 1.75, 2]#
Tbaseline           = 'GE' # options: 'ML', 'lb_no_priors', 'GE'
third_step_type     = 'viterbi+MAP' # options: ['MLE', 'MAP_for_GE', 'viterbi', 'viterbi+MAP']
save_raw            = False
save_fig            = False
save_path           = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
is_plot             = False
do_third_step       = True
is_sort_comb_by_priors = True
add_dd_based_prior  = False
debug_mode          = False 
plot_status_DD      = False

### probabilistic model config ###
sample_method       = 'GE'  # options: 'onlyPu', 'indicative'

### code config ###
code_type           = 'near_constant' # options: 'bernoulli', 'typical_bernoulli', 'near_constant'
delta_typical_cols  = 0.1 
delta_typical_rows  = 0.1

### viterbi config ###
init_paths_number    = 500
max_paths_for_lva    = 500
step_in_lva_paths    = 500
viterbi_time_steps   = 1 #1/2/3
max_iteration_for_map = 1e6
gamma_dict = {'0.5': 12,
              '0.6': 6,
              '0.7': 6,
              '0.8': 6,
              '0.9': 6,
              '1.0': 6,
              '1.1': 6,
              '1.2': 6,
              }

### seed ###
random.seed(123)
np.random.seed(123)

invalid = -1
all_permutations = []
vecTs = []
print('===== code = {} ====='.format(code_type))
if do_third_step:
    print('===== Last step decoder: {} ====='.format(third_step_type))
else:
    print('===== Stop after DD ====')
if third_step_type == 'viterbi+MAP':
    print('===== max paths in lva = {}  <=>  max lva iterations = {} ====='.format(max_paths_for_lva, int((max_paths_for_lva-init_paths_number)/step_in_lva_paths)+1))
    print('===== viterbi time steps = {} ====='.format(viterbi_time_steps))
## Initialize counters
numOfK = len(vecK)
viterbi_fail_try_full_map = False
num_of_test_scale = len(enlarge_tests_num_by_factors)
count_DND1 = np.zeros((numOfK, num_of_test_scale, nmc))
count_PD1 = np.zeros((numOfK, num_of_test_scale, nmc))
count_DD2 = np.zeros((numOfK, num_of_test_scale, nmc))
count_DND3 = np.zeros((numOfK, num_of_test_scale))
count_PD3 = np.zeros((numOfK, num_of_test_scale))
count_unknown2 = np.zeros((numOfK, num_of_test_scale, nmc))
count_success_DD_exact = np.zeros((numOfK, num_of_test_scale, nmc))
count_success_DD_non_exact = np.zeros((numOfK, num_of_test_scale, nmc))
count_success_exact_third_step = np.zeros((numOfK, num_of_test_scale, nmc))
count_success_non_exact_third_step = np.zeros((numOfK, num_of_test_scale, nmc))
count_not_detected = np.zeros((numOfK, num_of_test_scale))
count_not_detected_no_valid_option = np.zeros((numOfK, num_of_test_scale, nmc))
expected_notDetected = np.zeros((numOfK, num_of_test_scale))
expected_DD = np.zeros((numOfK, num_of_test_scale))
expected_PD = np.zeros((numOfK, num_of_test_scale))
expected_unknown = np.zeros((numOfK, num_of_test_scale))
bound_PD_DND = np.zeros((numOfK, num_of_test_scale))
bound_DD_DD = np.zeros((numOfK, num_of_test_scale))

#%% Start simulation
for idxK in range(numOfK):
    K = vecK[idxK]
    print('K = ' + str(K))
    # For each K calculate number of test according the Tml and scale factor
    ge_model = None
    markov_model = None
    if sample_method == 'GE': # create the ge model 
        _, ge_model = sample_population_gilbert_elliot_channel(N, K, ge_model, debug=False)
    elif sample_method == 'Markov':
            if N == 500:
                _, markov_model = sample_population_for_N500_K3_ts3(N, K, markov_model)
            elif N == 1024:
                _, markov_model = sample_population_for_N1024_K8_ts3(N, K, markov_model)
            elif N == 10000:
                _, markov_model = sample_population_for_N10000_K13_ts3(N, K, markov_model)
            else:
                print('Sampling method is not defined for the given N,K')
    vecT = calculate_vecT_for_K(K, N, enlarge_tests_num_by_factors, Tbaseline=Tbaseline,
                ge_model=ge_model)
    vecTs.append(vecT)
    time_start = time.time()
    for idxT in range(num_of_test_scale):
        T = np.int16(vecT[idxT])
        print('T = {} || factor = {}'.format(T, enlarge_tests_num_by_factors[idxT]))
        overflow_const = 1 #10^T

        p = 1-2**(-1/K) # options: 1/K, log(2)/K, 1-2**(-1/K)
        expected_PD[idxK, idxT] = K + (N-K) * (1-p*(1-p)**K)**T 
        nPD = expected_PD[idxK, idxT]
        expected_DD[idxK, idxT]  = K*(1-(1-p*(1-p)**(nPD-1))**T)#nPD*p_defective*(1-(1-p*(1-p)**(nPD-1))**T) # version1 - p_defective appears once
        expected_notDetected[idxK, idxT] = K - expected_DD[idxK, idxT]

        alpha = 1.4 * enlarge_tests_num_by_factors[idxT]
        fn = N ** (-0.5*alpha*(1-np.log(2)/K))
        bound_PD_DND[idxK, idxT] = K + (N-K) * fn
        gn = N ** (-0.5*alpha*(1-np.log(2)/K) ** (fn * N))
        bound_DD_DD[idxK, idxT] = K * (1-gn)

        count_success_DD_exact_vec_nmc = np.zeros((nmc,))
        count_success_DD_non_exact_vec_nmc = np.zeros((nmc,))
        if sample_method == 'GE':
            ge_model.calculate_num_of_permutations_by_entropy(K, T, nPD)
        if code_type == 'near_constant':
            L = np.int16(np.round(p*T))
            rows_idx = np.arange(T)

        for nn in tqdm(range(nmc), desc='T ' + str(T)):
        # for nn in range(nmc):
            if debug_mode:
                print('nn', nn)
            ## Sample
            if sample_method == 'onlyPu':
                U, Pu, Pw = sample_population_no_corr(N, K)
            elif sample_method == 'indicative':
                U, Pu, Pw, num_of_distractions = sample_population_indicative(N, K)
            elif sample_method == 'GE':
                U, ge_model = sample_population_gilbert_elliot_channel(N, K, ge_model, debug=debug_mode)
            elif sample_method == 'Markov':
                    U, markov_model = sample_population_for_N500_K3_ts3(N, K, markov_model)
            true_defective_set = np.where(U == 1)[1].tolist()
            ## 1. Definitely Not Defective
            # Encoder - bernoulli 
            if 'bernoulli' in code_type:
                X =  np.multiply(np.random.uniform(0,1,(T, N)) < p,1) # iid testing matrix
            
            if code_type == 'typical_bernoulli':
                all_X_typical = False
                while not all_X_typical:
                    non_typical_cols = set(np.arange(N))
                    while non_typical_cols:
                        non_typical_cols_list = list(non_typical_cols)
                        X[:,non_typical_cols_list] = np.multiply(np.random.uniform(0,1,(T, len(non_typical_cols))) < p,1)
                        sum_cols = np.sum(X[:,non_typical_cols_list], axis=0)
                        dist_cols = np.abs(sum_cols / T - p)
                        typical_cols = np.array(non_typical_cols_list)[np.where(dist_cols < delta_typical_cols)[0]]
                        non_typical_cols = non_typical_cols.difference(set(typical_cols))
                        
                    # verify that the rows are also typical 
                    sum_rows = np.sum(X, axis=1)
                    dist_rows = np.abs(sum_rows / N - p)
                    non_typical_row = [r for r in dist_rows if r > delta_typical_rows]
                    all_X_typical = len(non_typical_row) == 0

                non_typical_cols = list(non_typical_cols)

            elif code_type == 'near_constant':
                X = np.zeros((T, N)).astype(np.uint8)
                for ii in range(N):
                    X[random.choices(population=rows_idx, k=L),ii] = 1

            tested_mat = X*U
            Y = np.sum(tested_mat, 1) > 0 

            X_mark_occlusion = np.zeros((X.shape[0]+1, X.shape[1]))
            X_mark_occlusion[1:, :] = X
            X_mark_occlusion[0,true_defective_set] = 5 
            
            ## Decoder
            # 1. DND
            PD1 = np.arange(N)
            DND1 = []
            for ii in range(T):
                if len(PD1) <= K:
                    break 
                if Y[ii] == 0:
                    for jj in PD1:
                        # iter_until_detection_CoMa_and_DD[idxK, idxT] += 1
                        if X[ii,jj] == 1: # definitely not defected
                            PD1 = PD1[PD1 != jj]
                            DND1 += [jj]
            count_DND1[idxK, idxT, nn] = len(DND1)
            count_PD1[idxK, idxT, nn] = len(PD1)
            
            if len(PD1) <= K: # all the PD are DD - all defective found
                count_DD2[idxK, idxT, nn] = len(PD1)
                count_success_DD_exact[idxK, idxT, nn] += 1
                count_success_DD_non_exact[idxK, idxT, nn] += 1
                continue
            ## 2. Definite Defective
            DD2 = []
            for ii in range(T):
                if Y[ii] == 1 and np.sum(X[ii,PD1]) == 1: # only 1 item among the PD equals 1 and the rest equal 0
                    jj = np.where(X[ii,PD1] == 1)[0][0] # find the definite defective item index in PD1 array
                    defective = PD1[jj]
                    if defective not in DD2: # add jj only if jj is not already detected as DD
                        DD2 += [defective]
                        X_mark_occlusion[ii+1, defective] = 4
                elif Y[ii]==1: # occlusion - mark
                    participating = PD1[np.where(X[ii,PD1] ==1)[0]]
                    X_mark_occlusion[ii+1, participating] = 2
                    defective_occluded = [e for e in participating if e in true_defective_set]
                    X_mark_occlusion[ii+1, defective_occluded] = 3
            
            count_DD2[idxK, idxT, nn] = len(DD2)

            if len(DD2) >= K: # all defective found
                count_success_DD_exact[idxK, idxT, nn] += 1
                count_success_DD_non_exact[idxK, idxT, nn] +=1
                continue
            
            count_not_detected_defectives = K-len(DD2)
            count_success_DD_non_exact[idxK, idxT, nn] += (len(DD2) / K)
            unknown2 = [e for e in PD1 if e not in DD2]#PD1[PD1 not in DD2][0]
            count_unknown2[idxK, idxT, nn] = len(unknown2)

            ## Define HMM
            if is_plot and plot_status_DD:
                plot_status_before_third_step(N, K, T, enlarge_tests_num_by_factors[idxT], PD1, DD2, true_defective_set) 

            observations = 2*np.ones((N,)).astype(np.int8) # observations[PD1] = 2
            observations[DD2] = 1
            observations[DND1] = 0
            

            if not do_third_step:
                continue
            estU = np.zeros(U.shape)
            if third_step_type == 'viterbi':
                map_trajectory, map_probabilities = hmm_model.viterbi_algo_adjusted_to_GE(observations)
                
                map_trajectory = np.array(map_trajectory)
                estU[0,map_trajectory == 1] = 1
                # senity check - is DD2 in the most likely path?
                if len(DD2) > 0 and len(list(set(DD2) - set(np.where(map_trajectory == 1)[0]))) != 0:
                    print('{} DD2 not in path'.format(nn)) 

            elif third_step_type == 'viterbi+MAP':
                possible_combination_found = False
                skip_viterbi_paths_options = False
                if debug_mode:
                    print('start lva')
                top_k = init_paths_number
                while not possible_combination_found and top_k <= max_paths_for_lva:
                    if viterbi_time_steps == 1:
                        if sample_method == 'GE':
                            hmm_model = ge_model.model_as_hmm(K, T, len(PD1), p, ver_states=False)
                            path_trajs, path_probs, ml_prob, ml_traj = hmm_model.list_viterbi_algo_parallel_with_deter(observations, top_k=top_k)
                        elif sample_method == 'Markov':
                            # hmm_model_1step_memory = markov_model.model_as_hmm_with_1step_memory()
                            hmm_model = markov_model.model_as_hmm_with_1step_memory()
                            path_trajs, path_probs, ml_prob, ml_traj = hmm_model.list_viterbi_algo_parallel_with_deter(observations, top_k=top_k)
                        else: 
                            print('Not defined yet')# remove dup rows
                        unique_rows = np.unique(path_trajs, axis=0)
                        paths = unique_rows
                        if debug_mode:
                            print(paths.shape[0], ' unique paths')
                    else:
                        if sample_method == 'GE':
                            hmm_model = ge_model.model_as_hmm_with_long_memory(K,T,len(PD1), p, ts=viterbi_time_steps)
                        elif sample_method == 'Markov':
                            hmm_model = markov_model.model_as_hmm()
                        path_trajs2, path_probs2, ml_prob2, ml_traj2 = hmm_model_2steps_2.list_viterbi_algo_parallel_with_long_memory(observations, top_k=top_k)
                        paths = np.copy(path_trajs2)
                        
                        # remove dup rows
                        unique_rows = np.unique(paths, axis=0)
                        if debug_mode and unique_rows.shape[0] != paths.shape[0]:
                            print('check')
                        paths = unique_rows
                    
                    count_detections_for_non_exact_recovery = []
                    
                    # prepare k-defective optional sets
                    optional_sets_list = []
                    for ii in range(paths.shape[0]):
                        detected_defective_set = np.where(paths[ii]==1)[0]
                        if detected_defective_set.shape[0] >= gamma_dict[str(enlarge_tests_num_by_factors[idxT])]:#N*0.9:
                            # Too many potential combinations, skip the viterbi option
                            skip_viterbi_paths_options = True
                            continue
                        
                        elif detected_defective_set.shape[0] >= K:
                            # reasonable number of combinations, find the options
                            detected_defective_set_minus_DD = [e for e in detected_defective_set if e not in DD2] 
                            possible_kleft_combinations = prepare_nchoosek_comb(detected_defective_set_minus_DD, K-len(DD2))
                            possible_k_combinations = []
                            for c in possible_kleft_combinations:
                                c = (list(c) + DD2)
                                possible_k_combinations.append(c)
                            
                            for comb in possible_k_combinations:
                                list_of_false_positive_items = [item for item in detected_defective_set if (item not in unknown2) and (item not in DD2)]
                                if (not DD2 or len(list(set(DD2) - set(comb))) == 0) and not list_of_false_positive_items: # the comc include all the DD2 and does not include dnd1
                                    optional_sets_list.append(list(comb))
                        else: # no path with as least K defectives:
                            pass
                            
                            if len(count_detections_for_non_exact_recovery) <= len(detected_defective_set):
                                count_detections_for_non_exact_recovery = list(detected_defective_set)

                    if optional_sets_list: 
                        possible_combination_found = True
                    else:
                        top_k += step_in_lva_paths
                
                if possible_combination_found:
                    # keep only unique combinations, remove dups
                    optional_sets_ar = np.array(optional_sets_list)
                    optional_sets_ar = np.unique(optional_sets_ar, axis=0).astype(np.uint16)#.tolist()
                    viterbi_fail_try_full_map = False 
                    # num_comb_after_lva[idxK, idxT,nn]
                
                elif top_k > max_paths_for_lva:
                    # didn't find valid options using viterbi
                    # check if going over all the options is possible:
                    num_of_true_set_options_in_step3 = int(scipy.special.comb(len(unknown2), K-len(DD2)))
                    viterbi_fail_try_full_map = False # initialization
                    estU = np.zeros(U.shape)
                    estU[0,count_detections_for_non_exact_recovery] = 1
                    detected_defectives = np.where(estU==1)[1] # may be errornous detection
                    not_detected = set(true_defective_set)-set(detected_defectives)
                    num_of_correct_detection = K-len(not_detected)
                    count_success_non_exact_third_step[idxK, idxT, nn] += (num_of_correct_detection-len(DD2))/K 
                    continue
                else:
                    pass

                if debug_mode:
                    print('start map, #options=', optional_sets_ar.shape[0])
                # MAP
                ## 1st option - iterative MAP:
                apriori = invalid*np.ones((optional_sets_ar.shape[0],))
                for comb_idx, comb in enumerate(optional_sets_ar): 
                    comb = comb.tolist()
                    U_forW = np.zeros((1,N))
                    U_forW[0,list(set(comb + DD2))] = 1
                    
                    X_forW = X*U_forW
                    Y_forW = np.sum(X_forW, 1) > 0
                    if (Y_forW != Y).any():
                        if debug_mode and set(comb+DD2) == set(true_defective_set):
                            print('Yw!=Y')
                        continue
                    if sample_method == 'GE':
                        Pw_map = ge_model.calc_Pw(N, comb, DD2, DND1)
                    elif sample_method == 'Markov':
                        Pw_map = markov_model.calc_Pw(N, comb, DD2, DND1)
                    P_X_Sw = p ** np.sum(X_forW == 1)
                    apriori[comb_idx] = Pw_map * P_X_Sw
                # elapsed_iter_map = time.time() - t_iter_map
                # print('elapsed time in iterative MAP: {} [sec]'.format(elapsed_iter_map))
                
                max_likelihood_W = np.argmax(apriori)
                estU = np.zeros(U.shape)
                estU[0,optional_sets_ar[max_likelihood_W,:].tolist() + DD2] = 1
                
            elif third_step_type == 'MAP_for_GE':
                if sample_method != 'GE':
                    print('MAP estimator is currently implemented for GE only')
                    quit()
                
                all_permutations = np.array(list(itertools.combinations(unknown2, count_not_detected_defectives)))
                num_of_permutations = all_permutations.shape[0]
                apriori = invalid*np.ones((num_of_permutations,1)) 
                for comb in range(num_of_permutations):                            
                    permute = all_permutations[comb,:].tolist()
                    U_forW = np.zeros((1,N))
                    U_forW[0,permute + DD2] = 1
                    X_forW = X*U_forW
                    Y_forW = np.sum(X_forW, 1) > 0

                    if (Y_forW != Y).any():
                        if debug_mode and set(permute+DD2) == set(true_defective_set):
                            print('Yw!=Y')
                        continue
                    
                    Pw = ge_model.calc_Pw(N, permute, DD2, DND1)
                    P_X_Sw = p ** np.sum(X_forW == 1)
                    apriori[comb] = Pw * P_X_Sw
                    if set(permute+DD2) == set(true_defective_set) and debug_mode:
                        print('true defective set prior: prior(W*) = ' + str(apriori[comb,0]))
                
                max_likelihood_W = np.argmax(apriori)
                if debug_mode:
                    print('chosen defective set prior: prior(estW) = ' + str(apriori[max_likelihood_W,0]))
                estU = np.zeros(U.shape)
                estU[0, all_permutations[max_likelihood_W,:]] = 1  
                estU[0, DD2] = 1

            elif third_step_type == 'MLE':
                try:
                    all_permutations3_no_prior = np.array(list(itertools.combinations(unknown2, count_not_detected_defectives)))
                    num_of_all_permutations3_no_prior = all_permutations3_no_prior.shape[0]
                except:
                    print('could not find permutations')
                    continue
                # we want to find estW: estW = argmax{P(Y|W)}
                max_likelihood_W = None
                min_error_counter = np.inf
                # try each permutation w: 
                for w, permute in enumerate(all_permutations3_no_prior):
                    # calculate Y for this permutation Y|W=w
                    U_forW = np.zeros((1,N))
                    U_forW[0,permute.tolist()+DD2] = 1
                    X_forW = X*U_forW
                    Y_forW = np.sum(X_forW, 1) > 0
                    # evaluate Y
                    error_counter = np.sum(Y_forW != Y)
                    if error_counter < min_error_counter:
                        min_error_counter = error_counter
                        max_likelihood_W = w
                        if error_counter == 0: # Yw = Y
                            break 
                estU = np.zeros(U.shape)
                estU[0, all_permutations3_no_prior[max_likelihood_W,:]] = 1  
                estU[0, DD2] = 1
                
            if np.sum(U != estU) == 0:
                count_success_exact_third_step[idxK, idxT, nn] += 1
                count_success_non_exact_third_step[idxK, idxT, nn] += (K-len(DD2))/K
                if viterbi_fail_try_full_map:
                    print('here')
            else:
                # count only the items detected in the 3rd step 
                detected_defectives = np.where(estU==1)[1] # may be errornous detection
                not_detected = set(true_defective_set)-set(detected_defectives)
                num_of_correct_detection = K-len(not_detected)
                count_success_non_exact_third_step[idxK, idxT, nn] += (num_of_correct_detection-len(DD2))/K 
        
    elapsed = time.time() - time_start            
    print('It took {:.3f}[min]'.format(elapsed/60))
    
# Normalize success and counters

count_success_DD_exact = np.sum(count_success_DD_exact, axis=2) * 100/nmc
count_success_exact_third_step = np.sum(count_success_exact_third_step, axis=2) * 100/nmc 
count_success_exact_tot = count_success_DD_exact + count_success_exact_third_step
count_success_DD_non_exact = np.sum(count_success_DD_non_exact, axis=2) * 100/nmc 
count_success_non_exact_third_step = np.sum(count_success_non_exact_third_step, axis=2) * 100/nmc 
count_success_non_exact_tot = count_success_DD_non_exact + count_success_non_exact_third_step

print('count_success_exact_tot', count_success_exact_tot)
print('count_success_exact_non_exact_tot', count_success_non_exact_tot)
count_DND1_avg = np.sum(count_DND1, axis=2) / nmc
count_PD1_avg = np.sum(count_PD1, axis=2) / nmc
count_DD2_avg = np.sum(count_DD2, axis=2) / nmc
count_unknown2_avg = np.sum(count_unknown2, axis=2) / (nmc - count_success_DD_exact*nmc/100) 
expected_unknown = expected_PD - expected_DD
count_not_detected = np.matlib.repmat(np.array(vecK), num_of_test_scale,1).T - count_DD2_avg
if sample_method == 'GE':
    q = ge_model.q
    s = ge_model.s
    pi_B = ge_model.pi_B
    del markov_model
elif sample_method == 'Markov':
    del ge_model
if do_third_step and 'viterbi' in third_step_type:
        trans_mat = hmm_model.trans_mat
        init_prob = hmm_model.init_prob
# Make resutls directory
results_dir_path = None

typical_label = '_nottypical'
if code_type == 'typical_bernoulli':
    typical_label = '_typical'

third_step_label = third_step_type
if not do_third_step:
    third_step_label = 'None'

viterbi_label = ''
if third_step_label == 'viterbi+MAP':
    viterbi_label =  '_max_paths_for_lva' + str(max_paths_for_lva) 
    viterbi_label = viterbi_label + '_' + str(viterbi_time_steps) + 'steps'

if save_fig or save_raw:    
    print('Save...')
    time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    experiment_str = 'N' + str(N) + '_K1_' + str(vecK[0]) + '_nmc' + str(nmc) + '_thirdStep_' + third_step_label + viterbi_label + '_' + code_type + '_Tbaseline_' +  Tbaseline + '_'
    results_dir_path = os.path.join(save_path, 'countPDandDD_' + experiment_str + time_str)
    os.mkdir(results_dir_path)

#%% Visualize
if is_plot:
    plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1_avg, enlarge_tests_num_by_factors, nmc, count_DD2_avg, sample_method, 
                        Tbaseline, code_type, results_dir_path)
    plot_expected_DD(vecK, expected_DD, count_DD2_avg, vecT, enlarge_tests_num_by_factors, results_dir_path)
    plot_expected_PD(vecK, expected_PD, count_PD1_avg, vecT, enlarge_tests_num_by_factors, results_dir_path)
    plot_expected_unknown(vecK, expected_unknown, count_unknown2_avg, vecT, enlarge_tests_num_by_factors, results_dir_path)
    plot_expected_not_detected(vecK, expected_notDetected, count_not_detected, vecT, enlarge_tests_num_by_factors, results_dir_path)
    plot_expected_unknown_avg(vecK, expected_unknown, count_PD1_avg - count_DD2_avg, vecT, 
                            enlarge_tests_num_by_factors, results_dir_path)
    plot_Psuccess_vs_T(vecTs, count_success_DD_exact, count_success_exact_tot, vecK, N, nmc, third_step_label, sample_method, 
                        Tbaseline, enlarge_tests_num_by_factors, typical_label, delta_typical_cols,
                        results_dir_path, exact=True)
    plot_Psuccess_vs_T(vecTs, count_success_DD_non_exact, count_success_non_exact_tot, vecK, N, nmc, third_step_label, sample_method, 
                        Tbaseline, enlarge_tests_num_by_factors, typical_label,delta_typical_cols,
                        results_dir_path, exact=False)
    
#%% Save
if save_raw:
    fullRawPath = os.path.join(results_dir_path, 'workspace.mat')
    all_variables_names = dir()
    variables_to_save = [var for var in all_variables_names if var not in dont_include_variables]
    save_workspace(fullRawPath, variables_to_save, globals())
    save_code_dir(results_dir_path)
