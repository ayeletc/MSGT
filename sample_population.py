# from tkinter.tix import Tree
# from xmlrpc.client import boolean
from locale import normalize
from GE_model import GE_model
from audioop import reverse
import numpy as np
import pandas as pd
import random
from utils import *
import matplotlib.pyplot as plt


def sample_population_ISI_m1(N, K, cyclic=True, sample_only_consecutive=False):
    # Define the correlation between the items using transition probabilities from one item to the next item:
    P_trans = np.random.rand(N,) # this vector is given as a prior

    # Sample the model:
    # Randomize initial probabilities to each item to be defective - unknown
    Pu0 = np.random.rand(N,) 
    
    ''' 
    The probability of the i-th item to be defective is given by:
    P(the i-th item is defective) = Pu0[i] + Pu0[i-1] * P_trans[i-1 => i]
                                = Pu0[i] + Pu0[i-1] * trans_prob[i-1]
    '''

    coeff_mat = np.eye(N)
    # if not cyclic:
    for ii in range(N-1):
        coeff_mat[ii, ii+1] = P_trans[ii]
    if cyclic:
        coeff_mat[-1, 0] = P_trans[N-1]
    
    if sample_only_consecutive:
        init_item = random.randint(0,N-1)
        defectives = np.mod(np.arange(init_item, init_item+K), N)
        U = np.zeros((1,N))
        U[0,defectives] = 1
        
    else:
        # Calculate the probability of each item to be defective:
        Pdefective = Pu0 @ coeff_mat

        # Normalize Pu0 so sum(Pd) = K
        Pu0 *= (K/np.sum(Pdefective))
        
        # Calculate the probability of each item to be defective:
        # Pd = np.zeros((N,))
        # for ii in range(1,N):
        #     Pd[ii] = Pu0[ii] + Pu0[ii-1] * P_trans[ii-1]
        # if cyclic:
        #     Pd[0] = Pu0[0] + Pu0[N-1] * P_trans[N-1]
        # else:
        #     Pd[0] = Pu0[0]
        Pdefective = Pu0 @ coeff_mat

        
        U = choose_defective_given_probability_to_be_defecive(N, K, Pdefective)
    Pu = Pu0[np.newaxis, :]
    return U, Pu, coeff_mat
    # return U, W, Pu, coeff_mat, Pw
    
def choose_defective_given_probability_to_be_defecive(N, K, P_defective):
    # Choose the K items with the max probability
    Pd_series = pd.Series(np.squeeze(P_defective))
    maxKIdx = Pd_series.nlargest(K).index.values.tolist()
    U = np.zeros((1, N))
    U[0, maxKIdx] = 1
    return U

def sample_population_ISI(N, K, m, all_permutations, isi_type='asymmetric', calc_Pw=False, calc_Pu=True):

    # ====== Generate infection coefficients matrix ======
    # P(Ui) = sum{ a_ij * P(Uj)}
    # a_ij ≠ 0 for m different items 
    
    if isi_type == 'symmetric':
    # Generate a (N div m) blocks matrix
    # each item is affected by its m-members family
    
        coeff_mat = np.zeros((N,N))
        # fill N div m blocks
        curIdx = 1
        while curIdx <= N-m:
            for ii in range(m):
                tempcoeff_mat = np.sort(np.squeeze(rand_array_fixed_sum(1, m, 1)))
                relevantIdx = list(set(np.arange(curIdx,curIdx+m)) - set([curIdx+ii-1]))
                coeff_mat[curIdx + ii-1, relevantIdx] = tempcoeff_mat
            curIdx = curIdx + m + 1
        # fill the rest mod(N, m) lines 
        while curIdx <= N:
            tempcoeff_mat = np.squeeze(rand_array_fixed_sum(1, m, 1))
            relevantIdx = list(set(np.arange(curIdx,N)) - set([curIdx]))
            anotherLegalAffected = np.arange(1,curIdx)
            relevantIdx = [relevantIdx, randsample(anotherLegalAffected , m-len(relevantIdx))]
            coeff_mat[curIdx, relevantIdx] = tempcoeff_mat
            curIdx = curIdx + 1
            
    elif isi_type == 'asymmetric':
    # each item is affected by the m-items before
        coeff_mat = np.zeros((N,N))
        for ii in range(m,N):
            coeff_mat[ii, ii-m:ii] = np.sort(np.squeeze(rand_array_fixed_sum(1, m, 1)))
        # fill the m-first rows in cyclic way
        for ii in range(m):
            tempcoeff_mat = np.squeeze(rand_array_fixed_sum(1, m, 1))
            if len(np.arange(np.mod(N+ii-m, N+1), np.mod(N+ii-1,N+1))) == m:
                coeff_mat[ii, np.mod(N+ii-m, N+1):np.mod(N+ii-1,N+1)] = tempcoeff_mat[ii]
            else:
                numOfCoeffInTheEnd = N-np.mod(N+ii-m, N+1)#len(np.arange(np.mod(N+ii-m, N+1), N))
                coeff_mat[ii, np.arange(np.mod(N+ii-m, N+1), N)] = tempcoeff_mat[np.arange(numOfCoeffInTheEnd)]
                numOfCoeffInTheBeginning = m - numOfCoeffInTheEnd
                coeff_mat[ii, np.arange(numOfCoeffInTheBeginning)] = tempcoeff_mat[numOfCoeffInTheEnd:]
            
    elif isi_type == 'general':
        coeff_mat = np.zeros((N,N))
        for ii in range(N):
            coeff_mat[ii, :] = np.squeeze(rand_array_fixed_sum(1, N, 1)) # random vector with length N and sum 1
    
    # ======= Sample population =======
    # choose the first defective item
    firstDefectiveItem = random.randint(0, N-1)
    U = spread_infection_using_corr_mat(N, K, coeff_mat, firstDefectiveItem)
    defectiveItems = np.where(U == 1)[0]
    if not all_permutations:
        W = None
    else:
        W = np.where(defectiveItems == all_permutations)[0]
        if not W:
            raise('Error: permutation does not exist')
        
    # ======= Calculate W and U distribution if needed =======
    Pu = np.ones((N,1))/N
    Pw = None#np.zeros((numOfW ,1))
    if calc_Pw and all_permutations:
        Pw = calculatePw(N, K, coeff_mat, all_permutations)
    if calc_Pu:
        Pu = calculatePu(N, K, coeff_mat)
    return U, W, Pu, coeff_mat, Pw

def sample_population_no_corr(N, K, calcPw=False, exactly_k_defectives=True):
    # ====== Generate Pu ======
    # bernoulli probabilities
    Pu = rand_array_fixed_sum(N,1,K).T
    
    # ======= Sample population =======
    U = np.ones((1,N)) # initialize U
    if exactly_k_defectives:
        while(np.sum(U) != K):
            U = np.random.rand(N,1).T
            U = U <= Pu
    else:
        U = np.random.rand(N,1).T
        U = U <= Pu
        
    # ======= Calculate W distribution
    #  if needed =======
    if calcPw:
        Pw = [] # I cant use permutations unless I throw sampling where sum(U)≠K
    else:
        Pw = []
    return  U, Pu, Pw
    

def spread_infection_using_corr_mat(N, K, coeff_mat, firstDefectiveItem):
# sample population
# N - population size
# K - the number of the defective items
# coeffMat - NxN matrix where the i-th row includes the infection rate from
#           the i-th item to evryone else
# firstDefectiveItem - index of the first item that is known to be
#                       defective
# probabilityToDefective - Nx1 vector, the i-th element is the probaility
#                           of the i-th item to be defective after the 
#                           infection spread
# U - Nx1 boolean vector of the items. 1 = defective, 0 = not defective

    probabilityToDefective = np.zeros((N,1))
    probabilityToDefective[firstDefectiveItem,0] = 1

    # spread the infection to the other items in steps
    currentInfectionStep = [firstDefectiveItem] # a list of the defective 
                        # items that may affect others in this current step
    checkedItems = [] # list of items that could be affected
    while len(checkedItems) < N and currentInfectionStep:
        nextInfectionLevel = []
        for ii in range(len(currentInfectionStep)):
            currentDefectiveItem = currentInfectionStep[ii]
            # find all items the current item affect that were not checked
            # yet and are not going to be checked in the current infection
            # step
            relevantColumn = coeff_mat[:,currentDefectiveItem]
            maybeInfectedList = np.where(relevantColumn != 0)[0]
            maybeInfectedList = list(set(maybeInfectedList) - set(checkedItems + currentInfectionStep)) 
            probabilityToDefective[maybeInfectedList,0] = probabilityToDefective[maybeInfectedList,0]  + coeff_mat[maybeInfectedList, currentDefectiveItem] * probabilityToDefective[currentDefectiveItem,0]
            nextInfectionLevel = nextInfectionLevel + maybeInfectedList
         
        checkedItems = list(set(checkedItems + currentInfectionStep))
        currentInfectionStep = list(set(nextInfectionLevel))
    
    
    # # Choose the K items with the max probability
    # probabilityToDefective_series = pd.Series(np.squeeze(probabilityToDefective))
    # maxKIdx = probabilityToDefective_series.nlargest(K).index.values.tolist()
    # U = np.zeros((1, N))
    # U[0, maxKIdx] = 1
    # return U
    U = choose_defective_given_probability_to_be_defecive(N, K, probabilityToDefective)
    return U
    
def calculatePw(N, K, coeff_mat, all_permutations):
    pass

def calculatePu(N, K, coeff_mat):

    # N - population size
# K - the number of the defective items
# coeffMat - NxN matrix where the i-th row includes the infection rate from
#           the i-th item to evryone else
# allPermutations - nchoosek x K matrix in which each row is a possible
#                   permutation of K items in population of size N
#                   permutation is K indices of the defective items.
    Pu = np.zeros((N, 1))
    for ii in range(N):
        # for each item as the initial defective item
        # spread the infection and get W
        firstDefectiveItem = ii
        U_ii = spread_infection_using_corr_mat(N, K, coeff_mat, firstDefectiveItem)
        defective_idx = np.where(U_ii == 1)[0]
        Pu = Pu + U_ii.T
    Pu = Pu / (N * K) # normalization to sum(Pu)=1
    return Pu

def sample_population_indicative(N, K, distractions_rate=0.1, max_prob_distraction=0.5):
    population = set(np.arange(N))
    defectives_idx = random.sample(set(population), K)
    U = np.zeros((1,N))
    U[0,defectives_idx] = 1
    Pu = np.zeros((1,N))
    Pu[0,defectives_idx] = 1
    Pw = []
    # add distractions - non-defective items with small prior
    num_of_distractions = np.floor(distractions_rate * N).astype(np.int64)
    distractions_idx = random.sample(population - set(defectives_idx), num_of_distractions)
    distractions = max_prob_distraction*np.random.random((num_of_distractions,1)) # uniform in [0,max_prob_distraction]
    Pu[:,distractions_idx] = distractions.T
    Pu = Pu / np.sum(Pu)
    return U, Pu, Pw, num_of_distractions
    
def sort_comb_by_priors(all_permutations, Pu, overflow_const=10):
    # calculate Pw
    Pw = np.ones((1,all_permutations.shape[0]))
    for ii,permute in enumerate(all_permutations):
        if not (Pu[0,permute] == 0).any():
            Pw[0,ii] = np.prod(Pu[0,permute]*overflow_const)
            if Pw[0,ii] != 0:
                pass
        else:
            Pw[0,ii] = 0

    # sort all permutations by their probabilities Pw
    Pw_idx = Pw.argsort()[0,:][::-1] # descending order, first has the highest probability
    Pw_sorted = Pw[0,Pw_idx]
    if Pw_sorted[0] == 0:
        print('overflow?')
        pass
    all_permutations_sorted = all_permutations[Pw_idx,:]
    return all_permutations_sorted, Pw_sorted

def sort_comb_by_priors_ISI_m1(all_permutations, Pu, coeff_mat, DD2, debug_correct_permute):
    Pw = np.ones((all_permutations.shape[0],))
    N = Pu.shape[1]
    for ii, permute in enumerate(all_permutations):
        permute = list(permute) + DD2
        sorted_permute = sorted(permute)
        
        list_a, list_b, valid_sequenc = split_list_into_2_sequence(sorted_permute, 0, N)
        if not valid_sequenc:
            Pw[ii] == 0
            continue
        else: # if there is a cyclic consecutive, we reorder the sorted permute. 
            # for example: a = [1,2], b = [7, 8, 9, 10] and the new sorted_permute = [7,8,9,10,1,2]
           sorted_permute = list_b + list_a

        Pw[ii] = Pu[0,sorted_permute[0]]
        if Pw[ii] == 0:
            continue
        for jj in range(1, len(sorted_permute)):
            if coeff_mat[sorted_permute[jj-1], sorted_permute[jj]] == 0:
                Pw[ii] == 0
                continue
            Pw[ii] *= coeff_mat[sorted_permute[jj-1], sorted_permute[jj]]
            
    ''' 
    N = Pu.shape[1]
    Pw = np.ones((1,all_permutations.shape[0]))
    for ii, permute in enumerate(all_permutations):
        permute = list(permute) + DD2
        if N-1 not in permute: # then with m=1, only a consecutive numbers can be a defective set
            sorted_permute = sorted(permute)
            if sorted_permute == list(range(np.min(permute), np.max(permute)+1)):
                Pw[0,ii] = Pu[0,sorted_permute[0]]
                for item in sorted_permute[:-1]:
                    Pw[0,ii] *= coeff_mat[item, item+1]
                
            else:
                Pw[0,ii] = 0
        else: 
            list_a, list_b, num_of_sequences = split_list_into_2_sequence(sorted_permute)
            if num_of_sequences <= 2 and list_a[-1] == N:
                # calc Pw
                Pw[0,ii] = Pu[0,sorted_permute[0]]
                for item in sorted_permute[:-1]:
                    if item == N-1:
                        Pw[0,ii] *= coeff_mat[item, 0]
                    else:
                        Pw[0,ii] *= coeff_mat[item, item+1]
            else:
                Pw[0,ii] = 0    
                '''
    # sort all permutations by their probabilities Pw
    Pw_idx = Pw.argsort()[::-1] # descending order, first has the highest probability
    Pw_sorted = Pw[Pw_idx]
    if Pw_sorted[0] == 0:
        print('overflow?')
        pass
    all_permutations_sorted = all_permutations[Pw_idx,:]
    return all_permutations_sorted, Pw_sorted



def sample_population_gilbert_elliot_channel(N, K, ge_model, epsilon=0.01, debug=False):
    if ge_model is None:
        # q = 10*1/N
        # q defines the transition probability 0=>1
        # s define the  transition probability 1=>0
        
        s=K/N; 
        if s < 0.1:
            s = 0.2 # for N=500, K=10, s=0.1 will output only 1 burst, s=0.15 will output 1 or 2 bursts, 0.2=> even 3 bursts
        if N < 200:
            q = 2*1/N
        elif N > 200:
            q = 0.5*1/N#0.01#200*1/N
        if K/N > 0.3:
            s = s/10
        # s = 0.1
        # q = epsilon*s/(1-epsilon)
        pi_B = q/(s+q)
        '''
        pi_B = K/N
        q = 3/N
        s = q/pi_B-q
        '''
        '''
        pi_B = K/N
        b = 3 # num of bursts
        s = 1/(K/b)
        q = pi_B*s/(1-pi_B)
        '''
        '''
        s = 0.449
        q = 0.001
        pi_B = q/(s+q)
        '''
        # if N == 500 and K == 10:
        if K < 5:
            b = 0.5
        else:
            b = 2
        q = b/N
        s = b/K
        pi_B = q/(s+q)
        ge_model = GE_model(s, q, pi_B)
     
    
    num_of_iter = 0
    num_defective_sampled = 0
    while num_defective_sampled != K:
        num_of_iter += 1
        channel_statef, _  = ge_model.sample_gilbert_elliot_channel(N, max_bad=K) 
        if channel_statef is None: 
            # there were too much good (bad after inversion) items
            continue
        num_defective_sampled = np.sum(channel_statef)
        U = np.zeros((1,N))
        U[0, :] = channel_statef
    if debug:
        print('num_of_iter (until U with K defectives found)', num_of_iter)
    return U, ge_model  
    
        
# def sample_population_gilbert_elliot_channel(N, K, ge_model, epsilon=0.01, debug=False):
#     if ge_model is None:
#         # q = 10*1/N
#         s=K/N; 
#         if s < 0.1:
#             s = 0.2 # for N=500, K=10, s=0.1 will output only 1 burst, s=0.15 will output 1 or 2 bursts, 0.2=> even 3 bursts
#         if N < 200:
#             q = 2*1/N
#         elif N > 200:
#             q = 0.5*1/N#0.01#200*1/N
#         if K/N > 0.3:
#             s = s/10
#         # s = 0.1
#         # q = epsilon*s/(1-epsilon)
#         pi_B = q/(s+q)
#         ge_model = GE_model(s, q, pi_B)
    
#     num_of_iter = 0
#     num_defective_sampled = 0
#     while num_defective_sampled != K:
#         num_of_iter += 1
#         channel_statef, _  = ge_model.sample_gilbert_elliot_channel(N, max_bad=K) 
#         if channel_statef is None: 
#             # there were too much good (bad after inversion) items
#             continue
#         num_defective_sampled = np.sum(channel_statef)
#         U = np.zeros((1,N))
#         U[0, :] = channel_statef
#     if debug:
#         print('num_of_iter (until U with K defectives found)', num_of_iter)
#     return U, ge_model

def test_sample_population_gilbert_elliot_channel():
    N = 500
    K = 10
    ge_model = None
    U, _ = sample_population_gilbert_elliot_channel(N, K, ge_model, debug=True)
    # plt.figure()
    # plt.stem(U)
    # plt.show()
    participating_items = np.where(U == 1)[1]
    print('#K = ' + str(np.sum(U)))
    print('participating_items', participating_items)

def test_sample_population_gilbert_elliot_channel_count_bursts():
    N = 500
    K = 10
    nmc = 500
    count_num_of_bursts = np.zeros((nmc,))
    for nn in range(nmc):   
        ge_model = None
        U, _ = sample_population_gilbert_elliot_channel(N, K, ge_model, debug=False)
        participating_items = np.where(U == 1)[1]
        num_of_bursts = 1
        for ii,item in enumerate(participating_items[:-1]):
            if item != participating_items[ii+1] - 1:
                num_of_bursts += 1
        count_num_of_bursts[nn] = num_of_bursts

    plt.figure()
    plt.hist(count_num_of_bursts)
    plt.xlabel('#bursts')
    plt.title('#bursts, {} iterations'.format(nmc))
    plt.show()
    print('#bursts = ' + str(np.sum(U)))
    print('participating_items', participating_items)

def tune_sample_population_gilbert_elliot_channel():
    N = 100
    K = 10    
    nmc = 1000
    # for s in np.linspace(0.05, 0.5, 10):
    #     for pi_B_factor in np.linspace(2, 100, 20):
    s_list = [0.05, 0.1, 0.2]
    pi_B_factors = [1]

    store_seq = np.zeros((len(s_list)*len(pi_B_factors)*nmc, N))
    for idx_s, s in enumerate(s_list):
        for idx_piB, pi_B_factor in enumerate(pi_B_factors):
            prob_defective = K/N # = πB
            pi_B = prob_defective / pi_B_factor # /10 addition to N=100, K=10
            # Forward GE channel
            q = pi_B*s/(1-pi_B) # since eps_B q/(q+s) = eps, and eps_B = 1
            ge_model = GE_model(s, q, pi_B)
            for nn in range(nmc):
                channel_statef, _  = ge_model.sample_gilbert_elliot_channel(N) 
                store_seq[idx_s*nmc+idx_piB*nmc + nn,:] = channel_statef


    plt.figure()
    plt.imshow(store_seq)    
    plt.show()
    
def test_sample_population_gilbert_elliot_channel_hist():
    N = 100
    K = 10
    prob_defective = K/N  # = πB
    pi_B = prob_defective /10
    s = 0.1 # if N=100 and K=10, s=0.1 and pi_B = prob_defective +0.1 good!
    # Forward GE channel
    q = pi_B*s/(1-pi_B) # since eps_B q/(q+s) = eps, and eps_B = 1    
    
    nmc = 10000
    effective_Ks = np.zeros((nmc,))
    ge_model = GE_model(s, q, pi_B)
    for nn in range(nmc):
        channel_statef, _  = ge_model.sample_gilbert_elliot_channel(N) 
        num_defective_sampled = np.sum(channel_statef)
        effective_Ks[nn] = num_defective_sampled

    plt.figure()
    plt.hist(effective_Ks, bins=list(range(N)))
    plt.show()
    
def test_sample_population_indicative():
    ## show that each item has the same probability to be defective
    nmc = 1000
    N = 100
    K = 10
    totU = np.zeros((1,N))
    sumPu = np.zeros((1,nmc))
    for ii in range(nmc):
        U, Pu, _ = sample_population_indicative(N, K)
        totU = totU + U
        sumPu[0,ii] = np.sum(Pu)
    
    totU = totU * 100 / nmc
    plt.figure
    plt.plot(np.arange(N), totU.T)
    plt.xlabel('item')
    plt.ylabel('#time_it_was_defective')
    plt.ylim([0,100])
    plt.grid(True)
    
    # plt.hist(sumPu)
    plt.show()
    

def test_sample_population_no_corr():
    ## show that each item has the same probability to be defective
    nmc = 1000
    N = 100
    K = 10
    totU = np.zeros((1,N))
    sumPu = np.zeros((1,nmc))
    for ii in range(nmc):
        U, Pu, _ = sample_population_no_corr(N, K)
        totU = totU + U
        sumPu[0,ii] = np.sum(Pu)
    
    totU = totU * 100 / nmc
    plt.figure
    plt.plot(np.arange(N), totU.T)
    plt.xlabel('item')
    plt.ylabel('#times\_it\_was\_defective')
    plt.ylim([0,100])
    plt.grid(True)
    plt.show()
#     figurehistogram(sumPu)

def test_sample_population_ISI():
    # show that each item has the same probability to be defective
    nmc = 10000
    N = 100
    K = 5
    m = 20
    all_permutations = None
    totU = np.zeros((1,N))
    # sumPu = np.zeros
    # ((1,nmc))
    for ii in range(nmc):
        U, W, Pu, _, _ = sample_population_ISI(N, K, m, all_permutations, isi_type='asymmetric', calc_Pw=True, calc_Pu=False)
        totU = totU + U
        # sumPu[0,ii] = np.sum(Pu)
    
    totU = totU * 100 / nmc
    plt.figure
    plt.plot(np.arange(N), totU.T)
    plt.xlabel('item')
    plt.ylabel('#time\_it\_was\_defective')
    plt.ylim([0,100])
    plt.grid(True)
    plt.show()


def test_sample_population_ISI_m1():
    # show that each item has the same probability to be defective
    nmc = 10000
    N = 100
    K = 5
    m = 1
    all_permutations = None
    totU = np.zeros((1,N))
    # sumPu = np.zeros
    # ((1,nmc))
    for ii in range(nmc):
        U, Pu, coeff_mat = sample_population_ISI_m1(N, K, cyclic=True)
        totU = totU + U
        # sumPu[0,ii] = np.sum(Pu)
    
    totU = totU * 100 / nmc
    plt.figure
    plt.plot(np.arange(N), totU.T)
    plt.xlabel('item')
    plt.ylabel('#time it was defective')
    plt.ylim([0,100])
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    # sample_population_no_corr()
    # test_sample_population_ISI()
    # test_sample_population_indicative()
    # test_sample_population_ISI_m1()
    # test_sample_population_gilbert_elliot_channel_hist()
    # tune_sample_population_gilbert_elliot_channel()
    # test_sample_population_gilbert_elliot_channel()
    test_sample_population_gilbert_elliot_channel_count_bursts()
    pass