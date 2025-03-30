import numpy as np
from sample_population import *
import math 


Pemax_theor = 0.4#0.it 05:0.1:0.95 # for N = 500, K = 5, possible T range for DND+MAP: (35, ∞)
Pe = 0.05


def calculate_vecT_for_K(K, N, enlarge_tests_num_by_factors, Tbaseline='ML', Pe=0.05, 
                        sample_method='onlyPu', ge_model=None, Pu=None, coeff_mat=None):
    vecT = []
    if Tbaseline == 'ML': # was ML
        vecT = (np.ceil(1.4 * K * np.log2(N) * np.array(enlarge_tests_num_by_factors))).tolist() # ceil((1-Pemax_theor) * K * log(N/K))
    elif Tbaseline == 'lb_no_priors':
        vecT = (np.ceil(K * np.log(N/K) * np.array(enlarge_tests_num_by_factors))).tolist()
    elif Tbaseline == 'lb_with_priors' and sample_method == 'ISI':
        vecT = (np.ceil(calculate_lower_bound_ISI_m1(N, Pu, coeff_mat, Pe) * np.array(enlarge_tests_num_by_factors))).tolist()
    elif Tbaseline == 'GE':
        vecT = (np.ceil(np.log2(math.comb(N,K))/ge_model.calculate_lower_bound_GE(N)) * np.array(enlarge_tests_num_by_factors)).tolist()
        

    return vecT


def calc_entropy_binary_RV(p):
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def calc_entropy_y_given_x_binary_RV(px, conditional_py1x1, conditional_py1x0):
    # for x=1:
    H = -px * (conditional_py1x1*np.log2(conditional_py1x1) + (1-conditional_py1x1)*np.log2(1-conditional_py1x1))
    # for x=0:
    if conditional_py1x0 != 0:
        H -= (1-px) * (conditional_py1x0*np.log2(conditional_py1x0) + (1-conditional_py1x0)*np.log2(1-conditional_py1x0))
    return H


def test_calculate_lower_bound_GE():
    N = 100
    K = 10
    Pe = 0.05
    # calc Pu0?
    
    U, ge_model = sample_population_gilbert_elliot_channel2(N, K, ge_model=None)
    lb_GE = ge_model.calculate_lower_bound_GE(N, Pe=Pe)
    lb_no_corr = np.log2(float(math.comb(N, K)))
    print('lb_GE', lb_GE)
    print('lb_no_corr', lb_no_corr)
    print('upper bound ML', np.ceil(1.4 * K * np.log(N)/np.log(2)))
    pass

def calculate_lower_bound_ISI_m1(N, Pu, coeff_mat, Pe):
    '''
    Lower bound:
    T ≥ (1-Pe) * H(U) = (1-Pe) *  [ H(U1) + H(U2|U1) + H(U3|U1, U2) + ... + H(Uk|U1, U2, ..., Uk-1) ]

    In ISI with m=1: H(Uk|U1, U2, ..., Uk-1) = H(Uk|Uk-1)
    '''
    # The probability of each item to be defective:
    Pdefective =  Pu @ coeff_mat
    lb = calc_entropy_binary_RV(Pdefective[0,0])
    for ii in range(1,N):
        conditional_py1x1 = coeff_mat[ii-1, ii]#+ Pu[0,ii]
        conditional_py1x0 = Pu[0,ii] 
        lb += calc_entropy_y_given_x_binary_RV(Pu[0,ii-1], conditional_py1x1, conditional_py1x0)
    return (1-Pe) * lb

def test_calculate_lower_bound_ISI_m1():
    # nmc = 10000
    N = 100
    K = 10
    Pe = 0
    _, Pu, coeff_mat = sample_population_ISI_m1(N, K, cyclic=True, sample_only_consecutive=False)
    lb_ISI = calculate_lower_bound_ISI_m1(N, Pu, coeff_mat, Pe)
    lb_no_corr = np.log2(math.comb(N, K))
    print('lb_ISI', lb_ISI)
    print('lb_no_corr', lb_no_corr)
    print('upper bound ML', np.ceil(1.4 * K * np.log(N)/np.log(2)))

if __name__ ==  '__main__':
    # test_calculate_lower_bound_ISI_m1()
    test_calculate_lower_bound_GE()

