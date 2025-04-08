import os
from math import perm
import numpy as np
from scipy.io import savemat, loadmat
from itertools import combinations
import bisect 
import zipfile


def single_map(comb, N, DND1, DD2, X, Y, p, ge_model):
    comb = comb.tolist()
    U_forW = np.zeros((1,N))
    U_forW[0,list(set(comb + DD2))] = 1
    
    X_forW = X*U_forW
    Y_forW = np.sum(X_forW, 1) > 0
    if (Y_forW != Y).any():
        return 0
    Pw_map = ge_model.calc_Pw_fixed(N, comb, DD2, DND1)
    # Pw_vietrbi = TODO:get the path probability and compare to the map's
    P_X_Sw = p ** np.sum(X_forW == 1)
    return Pw_map * P_X_Sw

def single_map_test(comb):
    return np.sum(comb)

def bitlist2int(bits_list):
    integer = 0
    for bit in bits_list:
        integer = (integer << 1) | bit
    return integer

def rand_array_fixed_sum(n1,n2, fixed_sum):
    if 'fixed_sum' not in locals():
        fixed_sum = 1
    mat = np.random.rand(n1,n2)#[0]
    return mat * fixed_sum / np.sum(mat)

def split_list_into_2_sequence(list, min_item, max_item):
    a = []
    b = []
    add_to_list = 'a'
    num_of_sequences = 1
    valid_sequenc = True
    for ii, item in enumerate(list[:-1]):
        if add_to_list == 'b':
            b.append(item)
            if list[ii] - list[ii-1] == 1:
                num_of_sequences += 1
                valid_sequenc = False
                break
            
            
        else:
            a.append(item)
            if list[ii+1] - list[ii] != 1:
                add_to_list = 'b'
                num_of_sequences += 1
    if add_to_list == 'a':
        a.append(list[-1])
    else:
        b.append(list[-1])
    if num_of_sequences > 2:
        valid_sequenc = False
    elif num_of_sequences == 2:
        if a[0] != min_item or b[-1] != max_item:
            valid_sequenc = False
    return a, b, valid_sequenc

def compute_HammingDistance(X):
    return (X[:, None, :] != X).sum(2)

dont_include_variables = ['np', 'scipy', 'scipy.io', 'numpy', 'pd', 'matplotlib', 'zipfile', \
                        'yaml', 'time', 'tqdm', 'math', 'itertools', 'random', 'go', 'px', \
                        'datetime', 'os', 'plt', 'binomial', 'shelve', 'reverse', 'bisect', \
                        'plot_DD_vs_K_and_T', 'plot_expected_DD', 'plot_expected_PD', 'plot_expected_unknown', \
                        'plot_expected_not_detected', 'plot_expected_unknown_avg', 'plot_Psuccess_vs_T', 'plot_and_save', \
                        'save_workspace', 'load_workspace', 'rand_array_fixed_sum', 'split_list_into_2_sequence', \
                        'sample_population_no_corr', 'sample_population_ISI', 'sample_population_ISI+m1', \
                        'spread_infection_using_corr_mat', \
                        'hmm_model', 'hmm_model_2steps', 'hmm_model_2steps_2', \
                        'calculatePu', 'calculatePw', 'test_sample_population_no_corr', 'test_sample_population_ISI', \
                        'test_sample_population_ISI_m1', 'sample_population_indicative', 'sort_comb_by_priors', \
                        'sort_comb_by_priors_ISI_m1' , \
                        'calculate_lower_bound_ISI_m1', 'calc_entropy_y_given_x_binary_RV', 'calc_entropy_binary_RV', \
                        'test_calculate_lower_bound_ISI_m1', \
                        'not_detected', 'ge_model', 'perm', 'combinations', 'permutations'\
                        'num_of_false_positive_in_DD2', 'enlarge_tests_num_by_factors', 'count_not_detected_defectives', \
                        '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__','__spec__', 'fig']

def save_workspace(filename, names_of_spaces_to_save, dict_of_values_to_save):
    '''
        filename = location to save workspace.
        names_of_spaces_to_save = use dir() from parent to save all variables in previous scope.
            -dir() = return the list of names in the current local scope
        dict_of_values_to_save = use globals() or locals() to save all variables.
            -globals() = Return a dictionary representing the current global symbol table.
            This is always the dictionary of the current module (inside a function or method,
            this is the module where it is defined, not the module from which it is called).
            -locals() = Update and return a dictionary representing the current local symbol table.
            Free variables are returned by locals() when it is called in function blocks, but not in class blocks.

        Example of globals and dir():
            >>> x = 3 #note variable value and name bellow
            >>> globals()
            {'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', 'x': 3, '__doc__': None, '__package__': None}
            >>> dir()
            ['__builtins__', '__doc__', '__name__', '__package__', 'x']
    '''
    print('save_workspace')
    mydic = {}
    for key in names_of_spaces_to_save:
        # print(key, len(key))
        # if len(key) >= 31:
        #     print(key)
        try:
            mydic[key] = dict_of_values_to_save[key]
        except TypeError:
            pass
    
    savemat(filename, mydic, long_field_names=True)

def load_workspace(filename):
    '''
        filename = location to load workspace.
    '''
    mat = loadmat(filename)
    ignore_var = ['__header__', '__version__', '__globals__', 'plot_DD_vs_K_and_T', 'save_workspace', 'load_workspace', 'rand_array_fixed_sum', 'permutations', 'add_new_value_and_symbol_keep_sort']
    var_dict = {}
    for key in mat.keys():
        if key in ignore_var:
            continue
        if mat[key].dtype in ['<U3','<U8', '<U6', '<U2']:
            str_ar = mat[key]
            var_dict[key] = [string.replace(' ', '') for string in str_ar]
            if key in ['method_DD', 'sampleMethod', 'Tbaseline']:
                var_dict[key] = var_dict[key][0]
            # print(var_dict[key])
        else:
            var_dict[key] = mat[key].squeeze()
    return var_dict
## after getting this var_dict, put it in the globals:
# for key in var_dict.keys():
#     globals()[key] = var_dict[key]

 
def save_code_dir(output_path, code_dir_path=os.path.dirname(os.path.realpath(__file__))):
#def save_code_dir(output_path, code_dir_path=r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/code'):
    files_names_list = ['multi_stage_algo.py', 'GE_model.py', 'HMM.py', 'utils.py', 'plotters.py', 
                        'calc_bounds_and_num_of_tests.py', 'sample_population.py']
    with zipfile.ZipFile(os.path.join(output_path, 'code_dir.zip'), 'w') as zipMe:        
        for file in files_names_list:
            zipMe.write(os.path.join(code_dir_path, file), compress_type=zipfile.ZIP_DEFLATED)
    
def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    permute = tuple(pool[i] for i in indices[:r])
    print(permute)
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                permute = tuple(pool[i] for i in indices[:r])
                print(permute)
                break
        else:
            return
    
def my_combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    comb = tuple(pool[i] for i in indices)
    print(comb)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        comb = tuple(pool[i] for i in indices)
        print(comb)
    print('f')
    return (3, 4)

def add_new_value_and_symbol_keep_sort(values_arr, symbols_arr, value, symbol):
    # find the new idx:
    if value > values_arr[0]:
        idx = 0
    else:
        idx = values_arr.shape[0]-1
        for ii in range(1,values_arr.shape[0]):
            if value > values_arr[ii] and value <= values_arr[ii-1]:
                idx = ii
                break
    # idx = values_arr.searchsorted(-value) # minus for descending order
    if idx < values_arr.shape[0]:
        values_arr = np.concatenate((values_arr[:idx], [value], values_arr[idx:-1]))
        symbols_arr = np.concatenate((symbols_arr[:idx], [symbol], symbols_arr[idx:-1]))
    
    return values_arr, symbols_arr

def prepare_nchoosek_comb(array, k):
    return list(combinations(array, k))

def convert_int_to_base(n, base):
    # for example: n=17, base=3 return [1,2,2]
    sign = '-' if n<0 else ''
    n = abs(n)
    if n < base:
        return n
    s = ''
    while n != 0:
        s = str(n%base) + s
        n = n//base
    return list(map(int, sign+s))

if __name__ == '__main__':
    # db_path=r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw/countPDandDD_N20_nmc500_methodDD_Sum_typical_Tbaseline_ML_02082022_224856.mat'
    # var_dict = load_workspace(db_path)
    # for key in var_dict.keys():
    #     globals()[key] = var_dict[key]
    # permutations([1, 2, 3, 4, 5, 6, 7], 2)
    a = combinations([1, 2, 3, 4, 5, 6, 7], 1)
    pass

    

