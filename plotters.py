import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots #TODO add success heatmap/graph
from plotly.offline import plot #TODO check this out https://stackoverflow.com/a/58848335
from matplotlib import pyplot as plt
from utils import * 


marker_symbols = ["circle", "x", "star","asterisk"]
curve_colors = px.colors.qualitative.Plotly

def plot_status_before_third_step(N, K, T, test_factor, PD1, DD2, true_defective_set):
    DD_n = np.zeros((N,))
    PD_n = np.zeros((N,))
    true_defective_n = np.zeros((N,))
    DD_n[DD2,] = 0.75
    PD_n[PD1,] = 0.5
    true_defective_n[true_defective_set] = 1
    vecN = np.arange(N)+1

    plt.figure()
    plt.plot(vecN, PD_n, 'b', label='PD')
    plt.plot(vecN, DD_n, 'r', label='DD')
    plt.plot(vecN, true_defective_n, 'k', label='the defective set')
    plt.title('N={}, K={}, T={}={}*T_ML'.format(N,K,T,test_factor))
    plt.legend()
    plt.show()

def plot_DD_non_exact_Ps_vs_min_and_avg_hamming_dist(N, K, T, test_factor, count_success_DD_non_exact_vec_nmc, hamming_dist_avg, hamming_dist_min):
    fig, axs = plt.subplots(2)
    axs[0].scatter(hamming_dist_avg, count_success_DD_non_exact_vec_nmc, s=100, alpha=0.5)
    axs[0].grid(True)
    axs[0].set_xlabel('min hamming distance')
    axs[0].set_ylabel('Ps')
    axs[0].set_title('Ps vs. avg(Hamming dist) in Testing matrix \n' + r'N={}, K={}, T={}={}*T_ML'.format(N,K,T,test_factor),wrap=True)

    # axs[1].scatter(hamming_dist_min, count_success_DD_non_exact_vec_nmc, s=100, alpha=0.5)
    # axs[1].grid(True)
    # axs[1].set_xlabel('min hamming distance')
    # axs[1].set_ylabel('Ps')
    # axs[1].set_title('Ps vs. min(Hamming dist) in Testing matrix \n' + r'N={}, K={}, T={}={}*T_ML'.format(N,K,T,test_factor),wrap=True)
    heatmap, xedges, yedges = np.histogram2d(hamming_dist_min, count_success_DD_non_exact_vec_nmc)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = axs[1].imshow(heatmap.T, extent=extent, origin='lower', interpolation='nearest')
    axs[1].set_aspect(5)
    axs[1].set_xlabel('min hamming distance')
    axs[1].set_ylabel('Ps')
    axs[1].set_title('Ps vs. min(Hamming dist) in Testing matrix \n' + r'N={}, K={}, T={}={}*T_ML'.format(N,K,T,test_factor),wrap=True)
    plt.show()


def plot_DD_exact_Ps_vs_min_and_avg_hamming_dist(N, K, T, test_factor, count_success_DD_exact_vec_nmc, hamming_dist_avg, hamming_dist_min):
    hamming_dist_min_items = list(set(hamming_dist_min))
    hamming_dist_min_hist = np.zeros((int(np.max(hamming_dist_min_items)),2))

    for ii in range(len(count_success_DD_exact_vec_nmc)):
            hamming_dist_min_hist[int(hamming_dist_min[ii]-1), int(count_success_DD_exact_vec_nmc[ii])] += 1
    hamming_dist_min_hist[[0,1],:] = hamming_dist_min_hist[[1,0],:] # swap 0,1 for imshow
    fig, axs = plt.subplots(2)
    axs[0].scatter(hamming_dist_avg, count_success_DD_exact_vec_nmc, s=100, alpha=0.5)
    axs[0].grid(True)
    axs[0].set_xlabel('min hamming distance')
    axs[0].set_ylabel('success')
    axs[0].set_title('success vs. avg(Hamming dist) in Testing matrix \n' + r'N={}, K={}, T={}={}*T_ML'.format(N,K,T,test_factor),wrap=True)
    heatmap = axs[1].imshow(hamming_dist_min_hist[int(np.min(hamming_dist_min_items)):,:].T, interpolation='nearest')
    # axs[1].set_aspect(5)
    axs[1].grid(True)
    axs[1].set_title('success vs. min(Hamming dist) in Testing matrix \n' + r'N={}, K={}, T={}={}*T_ML'.format(N,K,T,test_factor),wrap=True)
    default_axis_x = np.arange(int(np.max(hamming_dist_min_items)-np.min(hamming_dist_min_items)))
    new_axis_x = np.arange(int(np.min(hamming_dist_min_items)), int(np.max(hamming_dist_min_items)),2)
    new_axis_x = [str(item) for item in new_axis_x]
    axs[1].set_xticklabels(new_axis_x, fontdict=None, minor=False)
    # axs[1].set_yticklabels([1,0], fontdict=None, minor=False)
    plt.colorbar(heatmap, ax=axs[1])
    axs[1].set_xlabel('min hamming distance')
    axs[1].set_ylabel('success')
    plt.show()

def plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1, enlarge_tests_num_by_factors, nmc, count_DD2, sample_method, Tbaseline, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=count_DD2[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='DD(2), T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=count_PD1[:,idxT] - count_DD2[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='Unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))
    fig.add_trace(go.Scatter(x=vecK, y=vecK,
                            mode='lines+markers',
                            marker_line_color="white", marker_color="white",
                            name='K'))

    fig.update_layout(title= sample_method + '<br>DD vs. K after CoMa and DD <br>\
                            N = ' + str(N) + ', T=T_{' + Tbaseline + '}*[' + str(enlarge_tests_num_by_factors) + '] <br>\
                            ' + ', iterations=' + str(nmc),
                        xaxis_title='K',
                        yaxis_title='#DD',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')
    # fig.show()
    plot_and_save(fig, fig_name='DD_vs_K_and_T', results_dir_path=results_dir_path)

    
def plot_expected_DD(vecK, expected_DD, real_DD, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_DD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected DD, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_DD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated DD, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected DD vs. simulated DD',
                        xaxis_title='K',
                        yaxis_title='#DD(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_DD', results_dir_path=results_dir_path)

def plot_expected_PD(vecK, expected_PD, real_PD, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_PD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected PD, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))

        fig.add_trace(go.Scatter(x=vecK, y=real_PD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated PD, T=' + str(enlarge_tests_num_by_factors[idxT])+ 'T = ' + str(T)),)

    fig.update_layout(title='Expected PD vs. simulated PD',
                        xaxis_title='K',
                        yaxis_title='#PD(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_PD', results_dir_path=results_dir_path)
    
def plot_expected_unknown(vecK, expected_unknown, real_unknown, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected unknown vs. simulated unknown',
                        xaxis_title='K',
                        yaxis_title='#unknown(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_unknown', results_dir_path=results_dir_path)

def plot_expected_not_detected(vecK, expected_not_detected, real_not_detected, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_not_detected[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected not detected, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_not_detected[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated not detected, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected not detected vs. simulated not detected',
                        xaxis_title='K',
                        yaxis_title='#not_detected',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_not_detedcted', results_dir_path=results_dir_path)

def plot_expected_unknown_avg(vecK, expected_unknown, real_unknown, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected unknown vs. simulated unknown (averaged)',
                        xaxis_title='K',
                        yaxis_title='#unknown(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_unknown_average', results_dir_path=results_dir_path)

def plot_Psuccess_vs_T(vecTs, count_success_DD, count_success_Tot, vecK, N, nmc, third_step_label, sample_method, Tbaseline, 
                        enlarge_tests_num_by_factors, results_dir_path, exact=True):
    if exact:
        comment = 'exact analysis'
    else:
        comment = 'non-exact analysis'
    fig = go.Figure()
    for idxK,K in enumerate(vecK):
        vecT = vecTs[idxK]
        fig.add_trace(go.Scatter(x=vecT, y=count_success_DD[idxK,:], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxK], marker_color=curve_colors[-idxK],
                                hovertemplate='%{y:.3f}',
                                name='Psuccess using DD, K=' + str(K)))
        
        fig.add_trace(go.Scatter(x=vecT, y=count_success_Tot[idxK,:], 
                                mode='lines+markers',
                                line_dash='dash',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxK], marker_color=curve_colors[-idxK],
                                hovertemplate='%{y:.3f}',
                                name='Psuccess Tot, K=' + str(K)))

    fig.update_layout(title= 'Third step: ' + third_step_label + ' || ' + sample_method + ' || ' + \
                            '<br> Probability of success vs. T  || ' + comment + '<br>\
                            N = ' + str(N) + \
                            ' || K = ' + str(vecK) + \
                            ' || T = T_{' + Tbaseline + '}*' + str(enlarge_tests_num_by_factors) + \
                            ' || #iterations = ' + str(nmc),
                        xaxis_title='T',
                        yaxis_title='#Ps [%]',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')
    fig.update_yaxes(range=[0, 100])
    plot_and_save(fig, fig_name='Ps_vs_T', exact=exact, results_dir_path=results_dir_path)

def plot_and_save(fig, fig_name, exact=True, results_dir_path=None):
    if exact:
        fig_name = fig_name + '_exact_analysis'
    else:
        fig_name = fig_name + '_non_exact_analysis'

    if results_dir_path is not None:
        fig.write_html(os.path.join(results_dir_path, fig_name+'.html'), auto_open=True)
    else:
        fig.show()

def mark_hidden_in_DD(X, save_path=r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/temp_res/'):

    pass

if __name__ == '__main__':
    db_path=r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw/countPDandDD_N100_nmc1000_methodDD_Sum_typical_Tbaseline_ML_07082022_092256.mat'
    var_dict = load_workspace(db_path)
    for key in var_dict.keys():
        globals()[key] = var_dict[key]
    plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1, enlarge_tests_num_by_factors, nmc, count_DD2, sample_method, Tbaseline)
    