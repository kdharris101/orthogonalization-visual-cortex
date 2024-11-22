# -*- coding: utf-8 -*-
#%%
"""
Created on Mon Feb 14 15:07:16 2022

@author: Samuel
"""


import numpy as np
import pandas as pd
import sklearn.metrics as skm
from os.path import join
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, kurtosis, linregress
from joblib import Parallel,delayed
from tqdm import tqdm


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# plt.style.use('seaborn')

# results_dir = 'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

subjects = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']
subject_unique = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B',
            'SF171107', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']
expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]

trained = [False, True, False, True, False, True, False, True, False, True]

# Load most up-to-date saved results
# file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
#                    str(expt_nums[i]), 
#                    'stim_decoding_results_replacement*'))[-1] 
#             for i in range(len(subjects))]

# file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
#                     str(expt_nums[i]), 
#                     'stim_decoding_results_all_cells_2022-01-24*'))[-1] 
#             for i in range(len(subjects))]

# file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
#                     str(expt_nums[i]), 
#                     'stim_decoding_results_all_cells_2022*'))[-1] 
#             for i in range(len(subjects))]

file_paths_all = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
                    str(expt_nums[i]), 
                    'stim_decoding_results_45 and 90 only_with cell stats_set 1600 cells_1000 repeats_*'))[-1] 
            for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel/OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%%

df_weights = pd.DataFrame()

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode_all = np.load(file_paths_all[i],allow_pickle = True)[()]
           
    n_cells = decode_all['n_cells']
    n_repeats = decode_all['n_repeats']
    
    repeat_inds = decode_all['cell_pool_ind']
    
    selectivity = decode_all['OSI'][repeat_inds]
    ori_pref = decode_all['mean_pref_ori'][repeat_inds]
    
    stim = decode_all['stim']
    trials = decode_all['trials_test'][:,repeat_inds]
    trials_z = decode_all['trials_z_test'][:,repeat_inds]
    
    ave_45 = trials_z[stim==45,...].mean(0)
    ave_90 = trials_z[stim==90,...].mean(0)

    # Recalculate d-prime by average mean and var across directions with same orientation
    stim_dir = decode_all['stim_dir']
    
    mu = np.array([trials[stim_dir==s,:].mean(0) 
                         for s in np.unique(stim_dir)])
    # mu_diff = np.diff(mu,axis=0).flatten()
    stim_ori = np.array([45,90,45,90])
    mu = np.array([mu[stim_ori==s,...].mean(0) for s in np.unique(stim_ori)])
    
    
    mu_diff = np.diff(mu,axis = 0).squeeze()
    
    var = np.array([trials[stim_dir==s,:].var(0) 
                         for s in np.unique(stim_dir)])
    
    var = np.array([var[stim_ori==s,...].mean(0) for s in np.unique(stim_ori)])
    
    sigma = np.sqrt(var.mean(0))
    
    # sigma = np.sqrt(np.array([x_test[stim_ori_test==s,:].var(0) 
    #                      for s in np.unique(stim_ori_test)]).mean(0))
    
    d_prime = mu_diff/sigma

    # CV is for max mean response
    max_mu = np.argmax(mu,axis = 0)    
    m,n = mu.shape[1:]
    I,J = np.ogrid[:m,:n]
    mu = mu[max_mu,I,J]
    var = var[max_mu,I,J]

    cv = np.sqrt(var)/mu

    # For each repeat, find contribution of each cell to decision function and then sort cells by this
    # Use average for 45 and 90 (should be almost identical since activity is z-scored)
    
    weights = decode_all['model_weights']

    df_45 = ave_45 * weights.T
    prop_df_45 = df_45/df_45.sum(axis=1,keepdims=True)
    df_90 = ave_90 * weights.T
    prop_df_90 = df_90/df_90.sum(axis=1,keepdims=True)
       
    mean_prop_df = (np.abs(prop_df_45) + np.abs(prop_df_90))/2

    s_ind = np.argsort(mean_prop_df,axis=1)[:,::-1]
    m = np.arange(prop_df_45.shape[0])[:,None]
    
    mean_prop_df = mean_prop_df[m,s_ind]
    
    prop_df_45 = prop_df_45[m,s_ind]
    prop_df_90 = prop_df_90[m,s_ind]
    
    prop_df_45_cumsum = prop_df_45.cumsum(axis=1)
    prop_df_90_cumsum = prop_df_90.cumsum(axis=1)
    
    prop_df_90_neg = prop_df_90*-1
    prop_df_90_cumsum_neg = prop_df_90_cumsum*-1
    
    
      
    repeat_label = np.tile(np.arange(n_repeats).reshape(-1,1),(1,n_cells))
    cell_num = np.tile(np.arange(n_cells).reshape(1,-1),(n_repeats,1))
  
    df_weights = pd.concat([
        df_weights,pd.DataFrame({'subject' : np.tile(subject_unique[i],(n_cells,n_repeats)).flatten(),
                                 'trained' : np.tile(trained[i],(n_cells,n_repeats)).flatten(),
                                 'repeat_num' : repeat_label.flatten(),
                                 'cell_num' : cell_num.flatten()+1,
                                 'selectivity' : selectivity[m,s_ind].flatten(),
                                 'ori_pref' : ori_pref[m,s_ind].flatten(),
                                 'model_weight' : decode_all['model_weights'].T[m,s_ind].flatten(),
                                 'cv' : cv[m,s_ind].flatten(),
                                 'd_prime' : d_prime[m,s_ind].flatten(),
                                 'ave_45_resp' : ave_45[m,s_ind].flatten(),
                                 'ave_90_resp' : ave_90[m,s_ind].flatten(),
                                 'df_45' : df_45[m,s_ind].flatten(),
                                 'df_90' : df_90[m,s_ind].flatten(),
                                 'prop_df_45' : prop_df_45.flatten(),
                                 'prop_df_90' : prop_df_90.flatten(),
                                 'prop_df_45_sum' : prop_df_45_cumsum.flatten(),
                                 'prop_df_90_sum' : prop_df_90_cumsum.flatten(),
                                 'mean_prop_df' : mean_prop_df.flatten(),
                                 'mean_prop_df_sum' : mean_prop_df.cumsum(axis=1).flatten(),
                                 'prop_df_90_neg' : prop_df_90_neg.flatten(),
                                 'prop_df_90_sum_neg' : prop_df_90_cumsum_neg.flatten()})], ignore_index=True)


df_weights['pref_bin'] = pd.cut(df_weights.ori_pref,np.linspace(-11.25,180-11.25,9),
                                labels = np.ceil(np.arange(0,180,22.5))).astype('category')
df_weights['r_bin'] = pd.cut(df_weights.selectivity,np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5)).astype('category')

df_weights['prop_df_90']

df_weights['abs_weight'] = np.abs(df_weights.model_weight)

# df_weights = df_weights.sort_values(['subject','trained','repeat_num','abs_weight'], ascending = False)

# df_weights = df_weights.reset_index(drop=True)

#%%

def pair_points(ax):
    
    # p0 = ax.get_children()[0].get_offsets().data
    # p1 = ax.get_children()[1].get_offsets().data
    
    # breakpoint()
    p = [i.get_offsets().data for i in ax.get_children() if type(i) is type(ax.get_children()[0])]
    
    points = []
    i = 0
    for i in range(len(p)):
        if len(p[i]) > 0:
            print(len(p[i]))
            points.append(p[i])
        
        if len(points) == 2:
            break
    
    # breakpoint()
    
    for i in range(len(points[0])):
        ax.plot([points[0][i,0], points[1][i,0]],[points[0][i,1], points[1][i,1]],'-k')


 #%% For each expt, plot proportion of decision variable as function of include cells (ordered from lowest weight to highest)

# def weight_norm(df,ind,r):
#     ind_r = np.logical_and(ind, df.repeat_num==r)
#     weights = df[ind_r].model_weight.to_numpy()
    
#     resps_45 = df[ind_r].ave_45_resp.to_numpy()
#     resps_90 = df[ind_r].ave_90_resp.to_numpy()
               
#     prop_resp_45 = resps_45 * weights
#     prop_resp_45 /= prop_resp_45.sum()
#     prop_resp_45_sum = np.cumsum(prop_resp_45)
      
#     prop_resp_90 = resps_90 * weights
#     prop_resp_90 /= prop_resp_90.sum()
#     prop_resp_90_sum = np.cumsum(prop_resp_90)
  
#     # df.loc[ind_r,'prop_resp_45'] = prop_resp_45
#     # df.loc[ind_r,'prop_resp_90'] = prop_resp_90
#     # df.loc[ind_r,'prop_resp_45_sum'] = prop_resp_45_sum
#     # df.loc[ind_r,'prop_resp_90_sum'] = prop_resp_90_sum
#     # # breakpoint()
#     # df.loc[ind_r,'rel_weight'] = np.abs(weights)/np.abs(weights).max()
#     # df.loc[ind_r,'prop_of_cells'] = (np.arange(np.sum(ind_r))+1)/np.sum(ind_r)
    
#     rel_weight = np.abs(weights)/np.abs(weights).max()
#     prop_of_cells = (np.arange(np.sum(ind_r))+1)/np.sum(ind_r)
    
#     return prop_resp_45,prop_resp_90,prop_resp_45_sum, prop_resp_90_sum,rel_weight,prop_of_cells


# results = []

# for s,t in zip(subject_unique,trained):

#         ind = np.logical_and(df_weights.subject.to_numpy() == s, df_weights.trained.to_numpy() == t)
        
#         for r in np.arange(n_repeats):
#             print(r)
#             ind_r = np.logical_and(ind, df_weights.repeat_num.to_numpy()==r)
#             weights = df_weights[ind_r].model_weight.to_numpy()
            
#             resps_45 = df_weights[ind_r].ave_45_resp.to_numpy()
#             resps_90 = df_weights[ind_r].ave_90_resp.to_numpy()
                       
#             prop_resp_45 = resps_45 * weights
#             prop_resp_45 /= prop_resp_45.sum()
#             prop_resp_45_sum = np.cumsum(prop_resp_45)
            
            
#             prop_resp_90 = resps_90 * weights
#             prop_resp_90 /= prop_resp_90.sum()
#             prop_resp_90_sum = np.cumsum(prop_resp_90)
          
    
#             df_weights.loc[ind_r,'prop_resp_45'] = prop_resp_45
#             df_weights.loc[ind_r,'prop_resp_90'] = prop_resp_90
#             df_weights.loc[ind_r,'prop_resp_45_sum'] = prop_resp_45_sum
#             df_weights.loc[ind_r,'prop_resp_90_sum'] = prop_resp_90_sum
#             # breakpoint()
#             df_weights.loc[ind_r,'rel_weight'] = np.abs(weights)/np.abs(weights).max()
#             df_weights.loc[ind_r,'prop_of_cells'] = (np.arange(np.sum(ind_r))+1)/np.sum(ind_r)
        
#         # results.append(Parallel(n_jobs=10)(delayed(weight_norm)(df_weights,ind,r) for r in tqdm(np.arange(n_repeats))))


#%% Use pandas functions


# df_weights['w_dot_x_45'] = df_weights['ave_45_resp']*df_weights['model_weight']
# df_weights['w_dot_x_90'] = df_weights['ave_90_resp']*df_weights['model_weight']


# # Max is 1
# df_weights['w_dot_x_45_norm'] = df_weights.groupby(
#                                     ['subject','trained','repeat_num'],
#                                     as_index = False)['w_dot_x_45'].transform(
#                                                         lambda x : x/np.sum(x))
                              
# # Max is -1                                    
# df_weights['w_dot_x_90_norm'] = df_weights.groupby(
#                                     ['subject','trained','repeat_num'],
#                                     as_index = False)['w_dot_x_90'].transform(
#                                                         lambda x : -1*x/np.sum(x))
                                        
# df_weights['w_dot_x_90_norm_pos'] = df_weights.w_dot_x_90_norm * -1

# df_weights['mean_w_dot'] = df_weights[['w_dot_x_45_norm','w_dot_x_90_norm_pos']].mean(axis=1)
                                        
                                        
# # df_weights['w_dot_x_45_norm_sum'] = df_weights.groupby(
# #                                     ['subject','trained','repeat_num'],
# #                                     as_index = False)['w_dot_x_45_norm'].transform(np.cumsum)

# # df_weights['w_dot_x_90_norm_sum'] = df_weights.groupby(
# #                                     ['subject','trained','repeat_num'],
# #                                     as_index = False)['w_dot_x_90_norm'].transform(np.cumsum)

# df_weights['rel_weight'] = df_weights.groupby(
#                                     ['subject','trained','repeat_num'],
#                                     as_index = False)['abs_weight'].transform(lambda x : x/x.sum())

# df_weights['prop_of_cells'] = df_weights.groupby(
#                                     ['subject','trained','repeat_num'],
#                                     as_index = False)['abs_weight'].transform(lambda x : (np.arange(len(x))+1)/len(x))

# df_weights['cell_num'] = df_weights.groupby(
#                                     ['subject','trained','repeat_num'],
#                                     as_index = False)['abs_weight'].transform(lambda x : np.arange(len(x))+1)


#%% Look at slope of proportion of decision function for naive vs proficient

n_cells = 1600 # Look only at first n_cells

df_prop_weights = df_weights[df_weights.cell_num <= n_cells].copy()

df_prop_weights = df_prop_weights.groupby(['subject','trained','cell_num'],as_index=False).mean()

# Look at difference of prop of df for 1st, 5th, 10th, and 20th cell

# cell_nums = [1,5,10,20,100,1000]
cell_nums = np.arange(1,101)

p_values = np.zeros(len(cell_nums))

for i,c in enumerate(cell_nums):
    
    p_values[i] = ttest_rel(df_prop_weights[np.logical_and(df_prop_weights.cell_num==c,
                                                   df_prop_weights.trained==False)].mean_prop_df,
                    df_prop_weights[np.logical_and(df_prop_weights.cell_num==c,
                                                   df_prop_weights.trained==True)].mean_prop_df)[1]
    print(p_values[i])



#%%

# For each experiment, find proportion of decision function as cell number (sorted by contribution to dec. fun.)

df_prop_weights = df_weights.copy()

df_prop_weights = df_prop_weights.groupby(['subject','trained','cell_num'],as_index=False).mean()

# Apply absolute value so that negative values (close to zero) can be logged
df_prop_weights['prop_df_45'] = df_prop_weights.prop_df_45.abs()
df_prop_weights['prop_df_90'] = df_prop_weights.prop_df_90.abs()

df_prop_weights['prop_df_45_log'] = df_prop_weights.prop_df_45.apply(np.log)
df_prop_weights['prop_df_90_log'] = df_prop_weights.prop_df_45.apply(np.log)
df_prop_weights['mean_prop_df_log'] = df_prop_weights.mean_prop_df.apply(np.log)
df_prop_weights['cell_num_log'] = df_prop_weights.cell_num.apply(np.log)

df_prop_weights['mean_percent_df'] = df_prop_weights.mean_prop_df*100
df_prop_weights['mean_percent_df_log'] = np.log(df_prop_weights.mean_percent_df)

# Get slope and R2 of loglog plots

df_slope = pd.DataFrame()

n_cells = 100

df_slope['slope'] = df_prop_weights[df_prop_weights.cell_num<=n_cells].groupby(['subject','trained']).apply(
    lambda x: linregress(x.cell_num_log,x.mean_percent_df_log)[0])
df_slope['r2'] = df_prop_weights[df_prop_weights.cell_num<=n_cells].groupby(['subject','trained']).apply(
    lambda x: linregress(x.cell_num_log,x.mean_percent_df_log)[2]**2)
df_slope = df_slope.reset_index()
    

# stim_colors = [(0,1,0), (1,0,1)]

cond_colors = sns.color_palette('colorblind',2)

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":2,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    labelpad = 2

    # Linear scale
    f,ax = plt.subplots(1,2,figsize=(2.5,1.5),sharey=False,sharex=False)


    sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'mean_percent_df',
                   hue = 'trained', errorbar=('se',1), palette = 'colorblind',
                   legend = False, linestyle = '-', ax = ax[1])
    
       
    # sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'prop_df_90',
    #                hue = 'trained', errorbar=('se',1), palette = 'colorblind',
    #                legend = False, linestyle = '--', ax = ax[0])
     
    # ax0_2 = ax[0].twinx().twiny()
    ax[1].set_xlabel('Cell number', labelpad = labelpad)
    ax[1].set_ylabel('% of discriminant', labelpad = labelpad)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xticks([1,10,100,1000])
    # ax[1].set_yticks([10**x for x in [0,-2,-4,-6,-8]])
    ax[1].set_yticks([10**x for x in [-1,0,1]])
    # ax[1].set_yticks([10**x for x in [0,-2,-4]])

    # ax[0].set_xticks([1,800,1600])
    
    ax[1].set_xlim([1,200])
    # ax[1].set_ylim([ax[1].get_yticks().min(),1])
    ax[1].set_ylim([0.1,10])
    # ax[1].vlines(100,*ax[0].get_ylim())
    # sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'mean_prop_df',
    #                hue = 'trained', errorbar=('se',1), palette = 'colorblind',
    #                legend = False, linestyle = '-', ax = ax0_2)
    
    # ax0_2.set_xlabel('Cell number')
    # ax0_2.set_ylabel('Proportion of dec. fun.')
    # ax0_2.set_xlim([1,100])
    # # ax0_2.set_xticks([1,800,1600])
    # ax0_2.set_xscale('linear')
    # ax0_2.set_yscale('linear')
    # sns.despine(ax=ax0_2, offset = 3)
    
    
    # sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'mean_prop_df',
    #                hue = 'trained', errorbar=('se',1), palette = 'colorblind',
    #                legend = False, linestyle = '-', ax = ax[0])
    
    
    # sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'prop_df_45_sum',
    #               hue = 'trained', errorbar=('se',1), palette = 'colorblind',
    #               legend = False, linestyle = '-', ax = ax[1])
    
    # sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'prop_df_90_sum',
    #               hue = 'trained', errorbar=('se',1), palette = 'colorblind',
    #               legend = False, linestyle = '--', ax = ax[1])
    
    # ax[1].set_xlabel('Cell number')
    # ax[1].set_ylabel('Cumulative sum of d. fun.')
    # ax[1].set_yscale('log')
    # ax[1].set_ylim([0.08,1])
    # ax[1].set_yticks([0.1,1])
    # ax[1].set_xscale('log')
    # ax[1].set_xticks([1,10,100,1000])
    # ax[1].set_xlim([1,1600])
    
    
    sns.lineplot(data = df_prop_weights[df_prop_weights.cell_num <= 20], 
                 x = 'cell_num', y = 'mean_percent_df',
                 hue = 'trained', errorbar=('se',1), palette = 'colorblind',
                 legend = False, linestyle = '-', ax = ax[0], zorder = 10)
    
    ax[0].set_xlabel('Cell number', labelpad = labelpad)
    ax[0].set_ylabel('% of discriminant', labelpad = labelpad)
    ax[0].set_xticks([1,5,10,15,20])
    ax[0].set_xlim([1,20])
    ax[0].set_yticks([0,5,10])
    
    pc = PatchCollection([Rectangle((5,0),6,0.05)], facecolor='grey', alpha=0.25,
                         edgecolor=None, zorder = 0)
    
    ax[0].add_collection(pc)
    
    # for i,c in enumerate([1,5,10,20]):
    #     if p_values[i] < 0.05:
    #         ax[0].text(c,0.1,'*', horizontalalignment = 'center')
    #     else:
    #         ax[0].text(c,0.1,'n.s.',  horizontalalignment = 'center')

    
    for a in ax:
        a.set_box_aspect(1)
        sns.despine(ax=a, offset = 3)


    f.tight_layout()
                                
    
    # # log-log scale
    # f,ax = plt.subplots(1,1)

    # sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'prop_df_45_sum',
    #                style = 'trained', errorbar=None, color = stim_colors[0],
    #                legend = False, style_order = [True,False], ax = ax)
     
    # sns.lineplot(data = df_prop_weights, x = 'cell_num', y = 'prop_df_90_sum',
    #                style = 'trained', errorbar=None, color = stim_colors[1],
    #                legend = False, style_order = [True,False], ax = ax)
     
    # ax.set_xlabel('Cell number')
    # ax.set_ylabel('Proportion of d. fun.')
    # ax.set_xticks([1,10,100,1000])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    # ax.set_box_aspect(1)
    # sns.despine(ax=ax, offset = 3)
    
    # f.tight_layout()
    
    # Plot slopes and R2
    
    # f,a = plt.subplots(1,2)
    
    
    # sns.stripplot(data = df_slope, x = 'trained', y = 'slope',
    #               ax = a[0], hue = 'trained')
    
    # sns.stripplot(data = df_slope, x = 'trained', y = 'r2',
    #               ax = a[1], hue = 'trained')
    
    
    # for a in a:
    #     pair_points(a)       
    #     sns.despine(ax = a)
    #     a.legend_.set_visible(False)
    
    
                      
    
#%% Plot cell selectivity and cv as function of model weight, averaged over repeats


df_plot = df_weights.copy()

df_plot['d_abs'] = df_plot.d_prime.abs()

df_plot = df_plot.groupby(['subject','trained','cell_num'],as_index=False).mean()


f, ax = plt.subplots(1,2, sharex=True, sharey=True)
# sns.scatterplot(data = df_plot[df_plot.trained==False], x = 'cv',
#                 y = 'selectivity', hue = 'rel_weight', ax = ax[0],
#                 palette = 'magma_r',legend = False, vmin = 0, vmax = 0.09,
#                 s = 10)
# sns.scatterplot(data = df_plot[df_plot.trained==True], x = 'cv',
#                 y = 'selectivity', hue = 'rel_weight', ax = ax[1],
#                 palette = 'magma_r', legend = False, vmin = 0, vmax = 0.09,
#                 s = 10)

sns.scatterplot(data = df_plot[df_plot.trained==False], x = 'cv',
                y = 'selectivity', hue = 'df_45', ax = ax[0],
                palette = 'magma_r',legend = False, vmin = 0, vmax = 0.14,
                s = 10)
sns.scatterplot(data = df_plot[df_plot.trained==True], x = 'cv',
                y = 'selectivity', hue = 'df_45', ax = ax[1],
                palette = 'magma_r', legend = False, vmin = 0, vmax = 0.14,
                s = 10)

norm = plt.Normalize(0, 0.14)
sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
sm.set_array([])

f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.825, 0.22, 0.008, 0.5])
plt.colorbar(sm, cax = cbar_ax, label = 'Proportion of dec. fun.')

ax[0].set_ylim([0,1])
ax[1].set_ylim([0,1])

ax[0].set_xlim([0.75,2])
ax[1].set_xlim([0.75,2])

# f, ax = plt.subplots(1,2, sharex=True, sharey=True)


# sns.scatterplot(data = df_plot[df_plot.trained==False], x = 'd_abs',
#                 y = 'ori_pref', hue = 'rel_weight', ax = ax[0],
#                 palette='magma_r', legend = False, vmin = 0, vmax = 0.09)

# sns.scatterplot(data = df_plot[df_plot.trained==True], x = 'd_abs',
#                 y = 'ori_pref', hue = 'rel_weight', ax = ax[1],
#                 palette='magma_r', legend = False, vmin = 0, vmax = 0.09)

# f.subplots_adjust(right=0.8)
# cbar_ax = f.add_axes([0.825, 0.22, 0.008, 0.5])
# plt.colorbar(sm, cax = cbar_ax, label = 'Model weight')


#%%

# For each experiment, find proportion of decision function as a set proportion of cells

df_prop_weights = df_weights.copy()

# one percent intervals
bins = np.linspace(0,1,1001)

df_prop_weights['cell_prop_bins'] = pd.cut(df_prop_weights.prop_of_cells,bins,labels=np.linspace(0.01,1,1000))
df_prop_weights['rel_weight_bins'] = pd.cut(df_prop_weights.rel_weight,bins,labels=np.linspace(0.01,1,1000))


df_prop_weights = df_prop_weights.groupby(['subject','trained','repeat_num','rel_weight_bins'],as_index=False).sum()
df_prop_weights = df_prop_weights.groupby(['subject','trained','rel_weight_bins'],as_index=False).mean()

df_prop_weights['prop_resp_45_sum'] = df_prop_weights.groupby(['subject','trained'])['w_dot_x_45_norm'].transform(np.cumsum)
df_prop_weights['prop_resp_90_sum'] = df_prop_weights.groupby(['subject','trained'])['w_dot_x_90_norm'].transform(np.cumsum)

df_prop_weights['prop_resp_sum_ave'] = (df_prop_weights.prop_resp_45_sum + df_prop_weights.prop_resp_90_sum)/2
df_prop_weights['prop_resp_ave'] = (df_prop_weights.w_dot_x_45_norm + df_prop_weights.w_dot_x_90_norm)/2

plt.figure()
sns.lineplot(data = df_prop_weights, x = 'rel_weight_bins', y = 'prop_resp_sum_ave',
              hue = 'trained', errorbar=('se',1), palette = 'colorblind',
              legend = False)

plt.figure()
sns.lineplot(data = df_prop_weights, x = 'rel_weight_bins', y = 'prop_resp_ave',
              hue = 'trained', errorbar=('se',1), palette = 'colorblind',
              legend = False)
        

#%%

# For each experiment, find proportion of decision function as a set proportion of cells

df_prop_weights = df_weights.copy()

# one percent intervals
bins = np.linspace(0,1,101)

df_prop_weights['cell_prop_bins'] = pd.cut(df_prop_weights.prop_of_cells,bins,labels=np.linspace(0.01,1,100))
df_prop_weights['rel_weight_bins'] = pd.cut(df_prop_weights.rel_weight,bins,labels=np.linspace(0.01,1,100))

df_prop_weights = df_prop_weights.groupby(['subject','trained','repeat_num','cell_prop_bins'],as_index=False).sum()
df_prop_weights = df_prop_weights.groupby(['subject','trained','cell_prop_bins'],as_index=False).mean()

df_prop_weights['prop_resp_45_sum'] = df_prop_weights.groupby(['subject','trained'])['w_dot_x_45_norm'].transform(np.cumsum)
df_prop_weights['prop_resp_90_sum'] = df_prop_weights.groupby(['subject','trained'])['w_dot_x_90_norm'].transform(np.cumsum)

df_prop_weights['prop_resp_sum_ave'] = (df_prop_weights.prop_resp_45_sum + df_prop_weights.prop_resp_90_sum)/2
df_prop_weights['prop_resp_ave'] = (df_prop_weights.w_dot_x_45_norm + df_prop_weights.w_dot_x_90_norm)/2

cond_colors = sns.color_palette('colorblind',2)

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":2,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams['figure.dpi'] = 100

    
    f,ax = plt.subplots(1,2,figsize=(2.5,1.25),sharey=False,sharex=False)
    sns.lineplot(data = df_prop_weights, x = 'cell_prop_bins', y = 'prop_resp_ave',
              hue = 'trained', errorbar=('se',1), palette = 'colorblind',
              legend = False, ax = ax[0])
    ax[0].set_ylim([0,0.7])
    # ax[0].set_aspect('equal')
    
    sns.despine(trim = True, offset=2, ax = ax[0])
    ax[0].set_xlabel('Prop. of neurons')
    ax[0].set_ylabel('Prop. of d.f.')
    
    ind_n = np.logical_and(df_prop_weights.cell_prop_bins==1,df_prop_weights.trained==False)
    ind_t = np.logical_and(df_prop_weights.cell_prop_bins==1,df_prop_weights.trained)
    ax[0].scatter(1,df_prop_weights[ind_n].prop_resp_ave.to_numpy().mean(),color = cond_colors[0],marker='_')
    ax[0].scatter(1,df_prop_weights[ind_t].prop_resp_ave.to_numpy().mean(),color = cond_colors[1],marker='_')
    
    
    sns.scatterplot(x = df_prop_weights[ind_n].prop_resp_ave.to_numpy(),
                y = df_prop_weights[ind_t].prop_resp_ave.to_numpy(), ax=ax[1])
    ax[1].set_xticks(np.arange(0,0.7,0.1))
    ax[1].set_yticks(np.arange(0,0.7,0.1))
    ax[1].plot([0,0.6],[0,0.6],'--k')
    # ax[1].set_aspect('equal')
    
    ax[1].set_xlabel('Prop. of d.f. (Naive)')
    ax[1].set_ylabel('Prop. of d.f. (Proficient)')
    
    sns.despine(trim = True,offset=0, ax = ax[1])

    f.tight_layout()


#%%

n_cells = np.zeros(len(subject_unique))

for i,(s,t) in enumerate(zip(subject_unique,trained)):
    ind = np.logical_and(df_weights.subject==s,df_weights.trained==t)
    n_cells[i] = ind.sum()

prop_last = df_prop_weights[df_prop_weights.cell_prop_bins==1].prop_resp_ave.to_numpy()

plt.plot(n_cells,prop_last,'.')


#%%

n_cells = np.zeros(len(subject_unique))
max_weight = np.zeros(len(subject_unique))

for i,(s,t) in enumerate(zip(subject_unique,trained)):
    ind = np.logical_and(df_weights.subject==s,df_weights.trained==t)
    n_cells[i] = ind.sum()
    max_weight[i] = df_weights[ind].abs_weight.max()
    

sns.scatterplot(x=n_cells, y = max_weight, hue = trained)

#%%

sns.lineplot(data = df_weights, x = 'prop_of_cells', y = 'prop_resp_45',
             hue = 'trained', errorbar = ('se',1))
        
#%% Plot model weight by orientation preference, selectivity, cv, d_prime

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'd_prime', palette = 'vlag', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue d_prime.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'cv', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue cv.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'ave_cv', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue ave cv.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'selectivity', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue selectivity.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'selectivity', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by selectivity_hue ori pref.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'd_prime', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by d-prime_hue ori pref.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'cv', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by cv_hue ori pref.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ave_cv', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ave cv_hue ori pref.svg'),format='svg')


#%% CV vs selectivity with model weight and orientation preference

sns.relplot(data = df_weights, col = 'subject', row = 'trained', 
            kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
            x = 'cv', y = 'selectivity', palette = 'hls',
            legend = False, ec = 'k')

plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_size weight.svg'),format='svg')


sns.relplot(data = df_weights, col = 'subject', row = 'trained', 
            kind = 'scatter', hue = 'rel_weight',
            x = 'cv', y = 'selectivity', legend = False,
            ec = 'k')

plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue weight.svg'),format='svg')


#%% sparseness of weights

weight_sparseness = np.zeros(len(subjects))

c = 0
for s in df_weights.subject.unique():
    for t in df_weights.Trained.unique():
        ind = np.logical_and(df_weights.subject==s,df_weights.Trained==t)
        
        weight_sparseness[c] = kurtosis(np.abs(df_weights[ind].model_weight))
        c += 1
        

df_sparseness = pd.DataFrame({'sparseness' : weight_sparseness,
                              'subject' : subjects,
                              'trained' : trained})


sns.catplot(data = df_sparseness, x = 'trained', y = 'sparseness')


#%% Proportion of of weights above 1, 5, 10, 100

weight_sparseness = np.zeros(len(subjects))

df_sparseness = pd.DataFrame(columns = ['above_1','above_5','above_10',
                                        'above_100', 'subject', 'trained'])

c = 0
for s in df_weights.subject.unique():
    for t in df_weights.Trained.unique():
        ind = np.logical_and(df_weights.subject==s,df_weights.Trained==t)
        
        
        d = {'above_' + str(i) : np.sum(df_weights[ind].model_weight > i)/np.sum(ind)
             for i in [1,5,10,100]}
        
        d['subject'] = s
        d['trained'] = t
        
        df_sparseness = pd.concat([df_sparseness,pd.DataFrame(d,index = None)])
        




sns.catplot(data = df_sparseness, x = 'trained', y = 'sparseness')


#%% Plot model weights by cell selectivity and orientation preference

df_plot = df_weights.groupby(['subject','Trained','pref_bin','r_bin'], as_index = False).mean()

sns.relplot(data = df_plot, x = 'pref_bin', y = 'model_weight', col = 'r_bin', 
            kind = 'line', errorbar = ('se',1), hue = 'trained', palette = 'colorblind',
            col_wrap = 5)

#%% Scatter plots of model weight by selectivity and CoV


df_weights['Mean ori. pref.'] = pd.cut(df_weights.ori_pref,np.linspace(-11.25,180-11.25,9),
                                labels = np.ceil(np.arange(0,180,22.5))).astype(int)
bins = np.linspace(0,1,21)
labels = 0.5*(bins[1:]+bins[:-1])
df_weights['r_bin'] = pd.cut(df_weights.selectivity,bins,
                     labels = labels)
bins = np.linspace(0,6,41)
labels = 0.5*(bins[1:]+bins[:-1])
df_weights['cv_bin'] = pd.cut(df_weights.cv, bins,
                              labels = labels)

# bins = np.linspace(-150,150,41)
bins = np.insert(np.insert(np.arange(-20,25,5), 0, -1000), len(np.arange(-20,25,5))+1, 1000)
# labels = 0.5*(bins[1:]+bins[:-1])
# df_weights['weight_bin'] = pd.cut(df_weights.model_weight,bins, labels = labels)
df_weights['weight_bin'] = pd.cut(df_weights.model_weight,bins,labels = np.arange(len(bins)-1))

df_plot = pd.DataFrame.copy(df_weights)

ind = np.logical_or(df_plot['Mean ori. pref.'] == 45, df_plot['Mean ori. pref.'] == 90)
df_plot = df_plot[ind].groupby(['subject','Trained','Mean ori. pref.', 'weight_bin'], 
                               as_index = False).mean()
# df_plot_r = df_plot[ind].groupby(['subject','Trained','Mean ori. pref.','r_bin'], as_index = False).mean()
# df_plot_cv = df_plot[ind].groupby(['subject','Trained','Mean ori. pref.','cv_bin'], as_index = False).mean()



ori_cp = sns.hls_palette(8)

ori_colors = {o:c for o,c in zip(np.unique(df_weights['Mean ori. pref.']),ori_cp)}
marker_style = {False : 'x',
                True : 'o'}

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":2,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams['figure.dpi'] = 100

    f,ax = plt.subplots(1,4,figsize = (5.5,2.6))
    
    # sns.scatterplot(data = df_weights, y = 'selectivity', x = 'model_weight', hue = 'Mean ori. pref.',
    #                 style = 'Trained', palette = ori_cp,
    #                 style_order = [True,False], legend = False, edgecolor = 'k',
    #                 ax = ax[0], rasterized = True, zorder = 2, facecolor = None)
    
    ax[0].scatter(x = df_weights[df_weights.Trained==False].model_weight, 
                  y = df_weights[df_weights.Trained==False].selectivity,
                  ec = df_weights.loc[df_weights.Trained==False,'Mean ori. pref.'].map(ori_colors),
                  marker = 'x', fc = None)
    ax[0].scatter(x = df_weights[df_weights.Trained].model_weight, 
                  y = df_weights[df_weights.Trained].selectivity,
                  ec = df_weights.loc[df_weights.Trained,'Mean ori. pref.'].map(ori_colors),
                  marker = 'o', fc = "none")
    
    # sns.scatterplot(data = df_weights, y = 'cv', x = 'model_weight', hue = 'Mean ori. pref.',
    #                 style = 'Trained', palette = ori_cp,
    #                 style_order = [True,False], legend = False, ec = 'k',
    #                 ax = ax[1], rasterized = True, zorder = 2, facecolor = None)
    
    
    ax[1].scatter(x = df_weights[df_weights.Trained==False].model_weight, 
                  y = df_weights[df_weights.Trained==False].cv,
                  ec = df_weights.loc[df_weights.Trained==False,'Mean ori. pref.'].map(ori_colors),
                  marker = 'x', fc = None)
    ax[1].scatter(x = df_weights[df_weights.Trained].model_weight, 
                  y = df_weights[df_weights.Trained].cv,
                  ec = df_weights.loc[df_weights.Trained,'Mean ori. pref.'].map(ori_colors),
                  marker = 'o', fc = "none")
    
    
    # ax[1].legend([1,2,5,10], frameon = False, fancybox = False,
    #                     ncol = 1, loc = 4, labelspacing = 0.1,
    #                     handletextpad = 0,
    #                     bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
    
      
    ax[0].set_xlabel('Model weight')
    ax[1].set_xlabel('Model weight')
    ax[0].set_ylabel('Selectivity')
    ax[1].set_ylabel('Coefficient of variation')
    ax[0].set_xlim([-150,150])
    ax[1].set_xlim([-150,150])
    ax[1].set_yticks([])
    ax[1].set_ylim([-0.1,5])
    ax[1].set_yticks([0,2.5,5])
    # ax[0].axvline(0.6, zorder = 0, color = 'k', linestyle = '--')
    # ax[1].axvline(0.5, zorder = 0, color = 'k', linestyle = '--')
    ax[0].set_aspect(1.0/ax[0].get_data_ratio(), adjustable='box')
    ax[1].set_aspect(1.0/ax[1].get_data_ratio(), adjustable='box')
    
    sns.despine(offset = 3, trim = True, ax = ax[0])
    sns.despine(offset = 3, trim = True, ax = ax[1])
    
    
    
    
    lp = sns.lineplot(data = df_plot, y = 'selectivity', 
                 x = 'weight_bin', hue = 'Trained', ax = ax[2],
                 palette = 'colorblind', hue_order = [False,True],
                 errorbar = ('se',1),
                 style = 'Mean ori. pref.')
    # ax[2].set_xlim([-150,150])
    # ax[2].set_xlim([0,11])
    ax[2].set_ylim([0,1])
    # bins = np.linspace(0,1,21)
    # labels = 0.5*(bins[1:]+bins[:-1])
    ax[2].set_yticks([0,0.5,1])
    # ax[2].set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    # ax[2].xaxis.set_tick_params(rotation=35)
    ax[2].set_aspect(1.0/ax[2].get_data_ratio(), adjustable='box')
    ax[2].set_xlabel('Model weight')
    ax[2].set_ylabel('Selectivity')
    ax[2].legend_.set_visible(False)
    
    sns.despine(offset = 3, trim = True, ax = ax[2])
    
    lp = sns.lineplot(data = df_plot, y = 'cv', 
                 x = 'weight_bin', hue = 'Trained', ax = ax[3],
                 palette = 'colorblind', hue_order = [False,True],
                  errorbar = ('se',1),
                 style = 'Mean ori. pref.')
    # ax[3].set_xlim([-150,150])
    # ax[3].set_xlim([0,11])
    ax[3].set_ylim([0,2])
    ax[3].set_yticks([0,1,2])
  
    ax[3].set_aspect(1.0/ax[3].get_data_ratio(), adjustable='box')
    ax[3].set_xlabel('Model weight')
    ax[3].set_ylabel('Coefficient of variation')
    ax[3].legend_.set_visible(False)
    
    
    # ax[3].set_ylabel('')
    # ax[3].set_yticks([])
    sns.despine(offset = 3, trim = True, ax = ax[3])

    
    f.tight_layout()

