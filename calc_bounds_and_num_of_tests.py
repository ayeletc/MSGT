import numpy as np
from sample_population import *
import math 

def calculate_vecT_for_K(K, N, enlarge_tests_num_by_factors, Tbaseline='ML', 
                         ge_model=None):
    return (np.ceil(1.4 * K * np.log2(N) * np.array(enlarge_tests_num_by_factors))).tolist() 
