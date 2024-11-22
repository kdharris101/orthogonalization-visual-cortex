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
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, kurtosis

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# plt.style.use('seaborn')

# results_dir = 'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']

subjects = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B',
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

file_paths_subsets = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), 
                    'stim_decoding_results_stepwise_selection_45 and 90 only_with cell stats_set features 10 cells_from varying pool sizes_1000 repeats_*'))[-1] 
            for i in range(len(subjects))]

file_paths_all = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), 
                    'stim_decoding_results_stepwise_selection_45 and 90 only_all cells*'))[-1] 
            for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel/OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%%

# For each repeat, calculate prop of df by cell

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode_subsets = np.load(file_paths_subsets[i],allow_pickle = True)[()]
    decode_all = np.load(file_paths_all[i],allow_pickle = True)[()]
    
    pool_sizes = decode_subsets['pool_sizes']
          
    stim = decode_subsets['stim']
    
    pred_stim = decode_subsets['pred_stim']
    
    features = decode_subsets['features']
       
    num_trials, num_features, num_pools, num_repeats = pred_stim.shape
        
    accu = pred_stim == np.tile(stim.reshape(-1,1,1,1), (1,num_features,num_pools,num_repeats))
    accu = accu.sum(0)/len(stim)
    
    prob_stim = decode_subsets['prob_stim']
    prob_stim = prob_stim.reshape(num_trials,2,-1)
    
    log_loss = np.zeros((num_features,num_pools,num_repeats))
    log_loss = log_loss.reshape(-1,)
    
    for l in range(prob_stim.shape[2]):
        log_loss[l] = skm.log_loss(stim,prob_stim[:,:,l])
    
    log_loss = log_loss.reshape(num_features,num_pools,num_repeats)
    
    if i == 0:
        accu_all = np.zeros((len(subjects), num_features, num_pools+1))
        OSI_all = np.copy(accu_all)
        mean_resp_all = np.copy(accu_all)
        std_resp_all = np.copy(accu_all)
        lt_sparseness_all = np.copy(accu_all)
        log_loss_all = np.copy(accu_all)
        model_weights_all = np.copy(accu_all)
        skew_weights_all = np.zeros((len(subjects),num_pools+1))
        
        accu_all_full_pool = np.zeros((len(subjects), num_pools))
        log_loss_all_full_pool = np.zeros((len(subjects), num_pools))
        
    
    accu_all[i,:,:num_pools] = accu.mean(-1)
    OSI_all[i,:,:num_pools] = decode_subsets['OSI'][features].mean(-1)
    mean_resp_all[i,:,:num_pools] = decode_subsets['mean_pref'][features].mean(-1)
    std_resp_all[i,:,:num_pools] = decode_subsets['std_pref'][features].mean(-1)
    lt_sparseness_all[i,:,:num_pools] = decode_subsets['lt_sparseness'][features].mean(-1)
    log_loss_all[i,:,:num_pools] = log_loss.mean(-1)
    
    # Model weights for all features/cells
    weights = np.zeros((num_features,num_pools,num_repeats))
    weights_expt = decode_subsets['model_weights'][-1,...]
    for p in range(num_pools):
        for r in range(num_repeats):
            weights[:,p,r] = weights_expt[p,r]
    
    # weights = np.concatenate(weights.ravel(),axis=0)
    # weights = weights.reshape(num_repeats,num_pools,num_features)
    # weights = np.moveaxis(weights,(0,1,2),(2,1,0))
    model_weights_all[i,:,:num_pools] = np.abs(weights).mean(-1)
    skew_weights_all[i,:num_pools] = kurtosis(np.abs(weights).mean(-1),axis = 0)

    
    accu = decode_all['pred_stim'] == np.tile(stim.reshape(-1,1), (1,num_features))
    accu = accu.sum(0)/len(stim)
    
    prob_stim = decode_all['prob_stim']
   
    log_loss = np.zeros(num_features)
   
    for l in range(prob_stim.shape[2]):
        log_loss[l] = skm.log_loss(stim,prob_stim[:,:,l])
      
    features = decode_all['features']
    accu_all[i,:,-1] = accu
    OSI_all[i,:,-1] = decode_all['OSI'][features]
    mean_resp_all[i,:,-1] = decode_all['mean_pref'][features]
    std_resp_all[i,:,-1] = decode_all['std_pref'][features]
    lt_sparseness_all[i,:,-1] = decode_all['lt_sparseness'][features]
    log_loss_all[i,:,-1] = log_loss
    model_weights_all[i,:,-1] =  np.abs(decode_all['model_weights'])
    
    # accu = decode_subsets['pred_stim_full_pool'] == np.tile(stim.reshape(-1,1,1), (1,num_pools,num_repeats))
    # accu = accu.sum(0)/len(stim)
    
    # accu_all_full_pool[i,:] = accu.mean(-1)
    
    # log_loss = np.zeros((num_pools,num_repeats))
    # log_loss = log_loss.reshape(-1,)
    
    # prob_stim = decode_subsets['prob_stim_full_pool']
    # prob_stim = prob_stim.reshape(num_trials,2,-1)
    
    # for l in range(prob_stim.shape[2]):
    #     log_loss[l] = skm.log_loss(stim,prob_stim[...,l])
        
    # log_loss = log_loss.reshape(num_pools,num_repeats)
    # log_loss_all_full_pool[i,:] = log_loss.mean(-1)
    

#%% Plot model weights by pool size

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

df_results = pd.DataFrame({'Cells in pool' : cell_pool_labels.flatten(),
                           'Cell number' : num_cells_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'Selectivity' : OSI_all.flatten(),
                           'Mean' : mean_resp_all.flatten(),
                           'STD' : std_resp_all.flatten(),
                           'CV' : std_resp_all.flatten()/mean_resp_all.flatten(),
                           'Model weight' : model_weights_all.flatten()})



# f, ax = plt.subplots(1,2)

# sns.scatterplot(data = df_results[df_results['Cells in pool'] != 'all'], 
#                 x = 'Model weight', 
#                 y = 'Selectivity',
#                 hue = 'Trained',
#                 ax = ax[0],
#                 style = 'Subject')


# ind = np.logical_and(df_results['Cells in pool'] != 'all', df_results['Cell number'] <= 3)


f,ax = plt.subplots(1,5, figsize = (10,2))

for i,c in enumerate(['all',400,200,100,50]):
    
    ind = df_results['Cells in pool'] == c
    
    # if i == 0:
    #     ind = np.logical_and(ind,df_results['Cell number']==1)
    #     sp = sns.stripplot(data = df_results[ind],
    #                   x = 'Trained',
    #                   y = 'Model weight',
    #                   hue = 'Trained',
    #                   palette = 'colorblind',
    #                   ax = ax[i])
    #     sp.legend_.set_visible(False)
    # else:
    sns.lineplot(data = df_results[ind],
                 x = 'Cell number',
                 y = 'Model weight',
                 hue = 'Trained',
                 palette = 'colorblind',
                 errorbar = ('se',1),
                 ax = ax[i],
                 legend = False)
    ax[i].set_xticks([1,10])
    ax[i].set_ylim([0,60])
        
    sns.despine(ax=ax[i])


f.tight_layout()
        
        
f,ax = plt.subplots(1,5, figsize = (10,2))

for i,c in enumerate(['all',400,200,100,50]):
    
    ind = df_results['Cells in pool'] == c
    
    # if i == 0:
    #     ind = np.logical_and(ind,df_results['Cell number']==1)
    #     sp = sns.stripplot(data = df_results[ind],
    #                   x = 'Trained',
    #                   y = 'Selectivity',
    #                   hue = 'Trained',
    #                   palette = 'colorblind',
    #                   ax = ax[i])
    #     sp.legend_.set_visible(False)
    # else:
    sns.lineplot(data = df_results[ind],
                 x = 'Cell number',
                 y = 'Selectivity',
                 hue = 'Trained',
                 palette = 'colorblind',
                 errorbar = ('se',1),
                 ax = ax[i],
                 legend = False)
    ax[i].set_xticks([1,10])
    ax[i].set_ylim([0,0.8])

        
    sns.despine(ax=ax[i])
    

f.tight_layout()
        
        
f,ax = plt.subplots(1,5, figsize = (10,2))

for i,c in enumerate(['all',400,200,100,50]):
    
    ind = df_results['Cells in pool'] == c
    
    # if i == 0:
    #     ind = np.logical_and(ind,df_results['Cell number']==1)
    #     sp = sns.stripplot(data = df_results[ind],
    #                   x = 'Trained',
    #                   y = 'CV',
    #                   hue = 'Trained',
    #                   palette = 'colorblind',
    #                   ax = ax[i])
    #     sp.legend_.set_visible(False)
    # else:
    sns.lineplot(data = df_results[ind],
                 x = 'Cell number',
                 y = 'CV',
                 hue = 'Trained',
                 palette = 'colorblind',
                 errorbar = ('se',1),
                 ax = ax[i],
                 legend = False)
    
    ax[i].set_xticks([1,10])
    ax[i].set_ylim([0,3.5])

        
    sns.despine(ax=ax[i])
    
    
    
f.tight_layout()



# sns.relplot(data = df_results, 
#                 x = 'Model weight', 
#                 y = 'Selectivity',
#                 hue = 'Trained',
#                 col = 'Cells in pool',
#                 kind = 'scatter',
#                 style = 'Cell number',
#                 palette = 'colorblind',
#                 col_order = ['all',400,200,100,50])

# sns.relplot(data = df_results, 
#                 x = 'Model weight', 
#                 y = 'CV',
#                 hue = 'Trained',
#                 col = 'Cells in pool',
#                 kind = 'scatter',
#                 style = 'Cell number',
#                 palette = 'colorblind',
#                 col_order = ['all',400,200,100,50])

# sns.relplot(data = df_results,
#             x = 'Cell number',
#             y = 'Model weight',
#             hue = 'Trained',
#             kind = 'line',
#             col = 'Cells in pool',
#             errorbar = ('se',1),
#             palette = 'colorblind',
#             col_order = ['all',400,200,100,50])

# sns.relplot(data = df_results,
#             x = 'Cell number',
#             y = 'Selectivity',
#             hue = 'Trained',
#             kind = 'line',
#             col = 'Cells in pool',
#             errorbar = ('se',1),
#             palette = 'colorblind',
#             col_order = ['all',400,200,100,50])

# sns.relplot(data = df_results,
#             x = 'Cell number',
#             y = 'CV',
#             hue = 'Trained',
#             kind = 'line',
#             col = 'Cells in pool',
#             errorbar = ('se',1),
#             palette = 'colorblind',
#             col_order = ['all',400,200,100,50])


#%% 

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

df_results = pd.DataFrame({'Cells in pool' : cell_pool_labels.flatten(),
                           'Cell number' : num_cells_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'Selectivity' : OSI_all.flatten(),
                           'Mean' : mean_resp_all.flatten(),
                           'STD' : std_resp_all.flatten(),
                           'CV' : std_resp_all.flatten()/mean_resp_all.flatten(),
                           'Model weight' : model_weights_all.flatten()})


f,ax = plt.subplots(1,5, figsize = (10,2))


for i,c in enumerate(['all',400,200,100,50]):
    
    ind = np.logical_and(df_results['Cells in pool'] == c, df_results['Cell number']==1)
    
    ind_n = np.logical_and(ind,df_results.Trained==False)
    ind_t = np.logical_and(ind,df_results.Trained)
    
    # lims = [np.min(df_results[ind].Selectivity),np.max(df_results[ind].Selectivity)]
    # print(lims)
        
    ax[i].scatter(df_results[ind_n].Selectivity.to_numpy(),
                  df_results[ind_t].Selectivity.to_numpy())
    
    ax[i].set_xlim([0.5,0.9])
    ax[i].set_ylim([0.5,0.9])
       
    ax[i].set_xlabel('Naive selectivity')
    ax[i].set_ylabel('Proficient selectivity')
    
    ax[i].plot([0.5,0.9],[0.5,0.9], '--k')
    
    sns.despine(ax = ax[i])
    
f.tight_layout()

    
    
f,ax = plt.subplots(1,5, figsize = (10,2))


for i,c in enumerate(['all',400,200,100,50]):
    
    ind = np.logical_and(df_results['Cells in pool'] == c, df_results['Cell number']==1)
    
    ind_n = np.logical_and(ind,df_results.Trained==False)
    ind_t = np.logical_and(ind,df_results.Trained)
    
    # lims = [np.min(df_results[ind].Selectivity),np.max(df_results[ind].Selectivity)]
    # print(lims)
        
    ax[i].scatter(df_results[ind_n].CV.to_numpy(),
                  df_results[ind_t].CV.to_numpy())
    
    ax[i].set_xlim([0,1])
    ax[i].set_ylim([0,1])
    
    ax[i].set(aspect = 'equal')
    ax[i].set_xlabel('Naive CV')
    ax[i].set_ylabel('Proficient CV')
    sns.despine(ax = ax[i])
    ax[i].plot([0,1],[0,1],'--k')
    
f.tight_layout()

    

f,ax = plt.subplots(1,5, figsize = (10,2))


for i,c in enumerate(['all',400,200,100,50]):
    
    ind = np.logical_and(df_results['Cells in pool'] == c, df_results['Cell number']==1)
    
    ind_n = np.logical_and(ind,df_results.Trained==False)
    ind_t = np.logical_and(ind,df_results.Trained)
    
    # lims = [np.min(df_results[ind].Selectivity),np.max(df_results[ind].Selectivity)]
    # print(lims)
        
    ax[i].scatter(df_results.loc[ind_n,'Model weight'].to_numpy(),
                  df_results.loc[ind_t,'Model weight'].to_numpy())
    
    ax[i].set_xlim([0,100])
    ax[i].set_ylim([0,100])
    
    ax[i].set(aspect = 'equal')
    ax[i].set_xlabel('Naive model weight')
    ax[i].set_ylabel('Proficient model weight')
    sns.despine(ax = ax[i])
    ax[i].plot([0,100],[0,100],'--k')
    
f.tight_layout()

    
f,ax = plt.subplots(1,5, figsize = (10,2))

for i,c in enumerate(['all',400,200,100,50]):
    
    ind = np.logical_and(df_results['Cells in pool'] == c, df_results['Cell number']==1)
    
    ind_n = np.logical_and(ind,df_results.Trained==False)
    ind_t = np.logical_and(ind,df_results.Trained)
    
    # lims = [np.min(df_results[ind].Selectivity),np.max(df_results[ind].Selectivity)]
    # print(lims)
        
    sns.scatterplot(x = df_results.loc[ind,'Model weight'].to_numpy(),
                    y = df_results.loc[ind,'CV'].to_numpy(),
                    hue = df_results.loc[ind,'Trained'].to_numpy(),
                    ax = ax[i], legend = False)
    
    ax[i].set_xlabel('Model weight')
    ax[i].set_ylabel('CV')
    
    sns.despine(ax = ax[i])

f.tight_layout()



f,ax = plt.subplots(1,5, figsize = (10,2))

for i,c in enumerate(['all',400,200,100,50]):
    
    ind = np.logical_and(df_results['Cells in pool'] == c, df_results['Cell number']==1)
    
    ind_n = np.logical_and(ind,df_results.Trained==False)
    ind_t = np.logical_and(ind,df_results.Trained)
    
    # lims = [np.min(df_results[ind].Selectivity),np.max(df_results[ind].Selectivity)]
    # print(lims)
        
    sns.scatterplot(x = df_results.loc[ind,'Model weight'].to_numpy(),
                    y = df_results.loc[ind,'Selectivity'].to_numpy(),
                    hue = df_results.loc[ind,'Trained'].to_numpy(),
                    ax = ax[i], legend = False)
    ax[i].set_xlabel('Model weight')
    ax[i].set_ylabel('Selectivity')
    
    sns.despine(ax = ax[i])

f.tight_layout()

#%% Look at cell metrics of first cell for all pools

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

df_results = pd.DataFrame({'Cells in pool' : cell_pool_labels.flatten(),
                           'Cell number' : num_cells_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'Selectivity' : OSI_all.flatten(),
                           'Mean' : mean_resp_all.flatten(),
                           'STD' : std_resp_all.flatten(),
                           'CV' : std_resp_all.flatten()/mean_resp_all.flatten(),
                           'Model weight' : model_weights_all.flatten()})


ind = df_results['Cell number'] == 1

sns.relplot(data = df_results[ind], x = 'Model weight', y = 'Selectivity',
            hue = 'Trained', size = 'CV', col = 'Cells in pool')

sns.relplot(data = df_results[ind], x = 'Model weight', y = 'CV', size = 'Selectivity',
            hue = 'Trained', col = 'Cells in pool')

sns.catplot(data = df_results[ind], x = 'Cells in pool', y = 'CV', hue = 'Trained',
            kind = 'bar', ci = 95)




#%% Line plots of logloss, OSI, mean response, std response, by pool size and training

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

# cell_pool_labels = np.tile(np.array([10,20,40,80]).reshape(1,-1,1),(10,1,10))

df_results = pd.DataFrame({'Accuracy' : accu_all.flatten()*100,
                           'Log loss' : log_loss_all.flatten(),
                           'Number of cells' : num_cells_labels.flatten(),
                           'Cells in pool' : cell_pool_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'Selectivity' : OSI_all.flatten(),
                           'Mean' : mean_resp_all.flatten(),
                           'STD' : std_resp_all.flatten(),
                           'Lifetime sparseness' : lt_sparseness_all.flatten()})

df_results['CV'] = df_results['STD']/df_results['Mean']

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":0.5,
                                        "ytick.major.size":0.5,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":4,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    palette = sns.color_palette('colorblind',2)

    f0,ax0 = plt.subplots(6,n_pools-1, sharey = False, figsize = (5.5,6))
    
    uni_pools = df_results['Cells in pool'].unique()
    uni_pools = uni_pools[-1::-1]
    
    for i in range(1,n_pools):
        ind = df_results['Cells in pool']==uni_pools[i]
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Accuracy', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[0,i], legend = False,
                     palette = palette)
        ax0[0,i].set_ylim([50,105])
        ax0[0,i].plot([1,10],[100,100],'--k')
        # ax0[0,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[0,i].set_title(uni_pools[i])

        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Log loss', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[1,i], legend = False,
                     palette = palette)
        ax0[1,i].set_ylim([-0.01,0.5])
        sns.lineplot(data = df_results[ind], x = 'Number of cells', y = 'Selectivity',
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[2,i], legend = False,
                     palette = palette)
        ax0[2,i].set_ylim([0,1])
        sns.lineplot(data = df_results[ind], x = 'Number of cells', y = 'Mean',
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[3,i], legend = False,
                     palette = palette)
        ax0[3,i].set_ylim([0,1])
        sns.lineplot(data = df_results[ind], x = 'Number of cells', y = 'STD',
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[4,i], legend = False,
                     palette = palette)
        ax0[4,i].set_ylim([0,1])
        sns.lineplot(data = df_results[ind], x = 'Number of cells', y = 'CV',
                      hue = 'Trained', errorbar = ('se',1), ax = ax0[5,i], legend = False,
                      palette = palette)
        ax0[5,i].set_ylim([0,4])
        
        if i > 0:
            for r in range(6):
                if r < 5:
                    ax0[r,i].set_ylabel('')
                    ax0[r,i].set_xlabel('')
                    ax0[r,i].set_xticks([])
                    ax0[r,i].set_yticks([])
                    sns.despine(ax = ax0[r,i], left = True, bottom = True)

                else:
                    ax0[r,i].set_ylabel('')
                    ax0[r,i].set_yticks([])
                    ax0[r,i].set_xticks([1,5,10])
                    sns.despine(ax = ax0[r,i], left = True, trim = True)

        else:
            for r in range(6):
                if r < 5:
                    ax0[r,i].set_xlabel('')
                    ax0[r,i].set_xticks([])
                    sns.despine(ax = ax0[r,i], bottom = True)

                else:
                    ax0[r,i].set_xticks([1,5,10])
                    sns.despine(ax=ax0[r,i], trim = True)
        
    for a in ax0.flat:
        a.set_box_aspect(1)
     
            
    # Scatter plot of accuracy differences

    f1,ax1 = plt.subplots(2,2, sharex = True, sharey = True)
    
    sns.set_palette('colorblind')
    
    for i in range(n_pools):
        sns.set_palette('colorblind')
        
        for f in [1,2,5,10]:
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            sns.scatterplot(x = df_results.loc[ind_N,'Accuracy'].to_numpy(),
                            y = df_results.loc[ind_T,'Accuracy'].to_numpy(),
                            ax = ax1.flat[i], legend = False)
            
            _,p = ttest_rel(df_results.loc[ind_N,'Accuracy'].to_numpy(),df_results.loc[ind_T,'Accuracy'].to_numpy())
            
           
            print(p)
                      
        sns.despine(ax = ax1.flat[i])
        ax1.flat[i].set_ylabel('Proficient accuracy')
        ax1.flat[i].set_xlabel('Naïve accuracy')
        ax1.flat[i].set_box_aspect(1)
        ax1.flat[i].set_title(uni_pools[i])
        ax1.flat[i].set_xlim([0.825,1.025])
        ax1.flat[i].set_ylim([0.825,1.025])
        ax1.flat[i].plot([0.825,1],[0.825,1],'k--')
        
        
#%% Line plots of accuracy and log loss

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

# cell_pool_labels = np.tile(np.array([10,20,40,80]).reshape(1,-1,1),(10,1,10))

df_results = pd.DataFrame({'Accuracy' : accu_all.flatten()*100,
                           'Log loss' : log_loss_all.flatten(),
                           'Number of cells' : num_cells_labels.flatten(),
                           'Cells in pool' : cell_pool_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'Selectivity' : OSI_all.flatten(),
                           'Mean' : mean_resp_all.flatten(),
                           'STD' : std_resp_all.flatten(),
                           'Lifetime sparseness' : lt_sparseness_all.flatten()})

df_results['CV'] = df_results['STD']/df_results['Mean']

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


    palette = sns.color_palette('colorblind',2)

    f0,ax0 = plt.subplots(2,n_pools, sharey = False, sharex = False, figsize = (4.75,2.4))
    
    uni_pools = df_results['Cells in pool'].unique()
    uni_pools = uni_pools[-1::-1]
    
    for i in range(n_pools):
        ind = df_results['Cells in pool']==uni_pools[i]
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Accuracy', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[0,i], legend = False,
                     palette = palette)
        ax0[0,i].set_ylim([50,105])
        ax0[0,i].plot([1,10],[100,100],'--k')
        ax0[0,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        sns.lineplot(data = df_results[ind], x = 'Number of cells', y = 'Log loss',
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[1,i], legend = False,
                     palette = palette)
        ax0[1,i].set_ylim([-0.01,0.5])
        
        if i > 0:
            for r in range(2):
                if r < 1:
                    ax0[r,i].set_ylabel('')
                    ax0[r,i].set_xlabel('')
                    ax0[r,i].set_xticks([])
                    ax0[r,i].set_yticks([])
                    sns.despine(ax = ax0[r,i], left = True, bottom = True)

                else:
                    ax0[r,i].set_ylabel('')
                    ax0[r,i].set_yticks([])
                    ax0[r,i].set_xticks([1,10])
                    sns.despine(ax = ax0[r,i], left = True, trim = True)

        else:
            for r in range(2):
                if r < 1:
                    ax0[r,i].set_xlabel('')
                    ax0[r,i].set_xticks([])
                    sns.despine(ax = ax0[r,i], bottom = True)

                else:
                    ax0[r,i].set_xticks([1,10])
                    sns.despine(ax=ax0[r,i], trim = True)
        
    for a in ax0.flat:
        a.set_box_aspect(1)
     

    f1,ax1 = plt.subplots(2,2, sharex = True, sharey = True)
    
    sns.set_palette('colorblind')
    
    for i in range(n_pools-1):
        sns.set_palette('colorblind')
        
        for f in [1,2,5,10]:
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            sns.scatterplot(x = df_results.loc[ind_N,'Log loss'].to_numpy(),
                            y = df_results.loc[ind_T,'Log loss'].to_numpy(),
                            ax = ax1.flat[i], legend = False)
            
            _,p = ttest_rel(df_results.loc[ind_N,'Log loss'].to_numpy(),df_results.loc[ind_T,'Log loss'].to_numpy())

            print(p)
                      
        sns.despine(ax = ax1.flat[i])
        ax1.flat[i].set_ylabel('Proficient log loss')
        ax1.flat[i].set_xlabel('Naive log loss')
        ax1.flat[i].set_box_aspect(1)
        ax1.flat[i].set_title(uni_pools[i])
        ax1.flat[i].set_xlim([-0.05,0.4])
        ax1.flat[i].set_ylim([-0.05,0.4])
        ax1.flat[i].plot([0,0.4],[0,0.4],'k--')


#%% Line and scatter plots of accuracy only

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

# cell_pool_labels = np.tile(np.array([10,20,40,80]).reshape(1,-1,1),(10,1,10))

df_results = pd.DataFrame({'Accuracy' : accu_all.flatten()*100,
                           'Number of cells' : num_cells_labels.flatten(),
                           'Cells in pool' : cell_pool_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'CV' : std_resp_all.flatten()/mean_resp_all.flatten()})


point_colors = sns.color_palette('flare', 10)
# point_colors = sns.color_palette('Set1', 10)
point_colors = [point_colors[i] for i in [0,1,4,9]]
# point_colors_m = {s:c for s,c in zip([1,2,5,10],point_colors)}

cond_colors = sns.color_palette('colorblind',2)
cond_colors_m = {False : cond_colors[0],
                 True : cond_colors[1]}

point_styles = [r'$1$',r'$2$',r'$5$',r'$10$']
point_sizes = [2,4,10,20]
    
palette = sns.color_palette('colorblind',2)

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
                                        "lines.linewidth":0.75,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    f0,ax0 = plt.subplots(4,n_pools, sharey = False, sharex = False, figsize = (5.5,6))
    
    uni_pools = df_results['Cells in pool'].unique()
    uni_pools = uni_pools[-1::-1]
    
    for i in range(n_pools):
        
        # Accuracy plots
        ind = df_results['Cells in pool']==uni_pools[i]
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Accuracy', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[0,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'Accuracy', 
                        ax = ax0[0,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
           
        ax0[0,i].set_ylim([50,105])
        ax0[0,i].plot([1,10],[100,100],'--k', zorder = 0)
        ax0[0,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[0,i].set_xlim([0.5,12])
        ax0[0,i].set_xticks([1,10])
        
        if i > 0:
            ax0[0,i].scatter(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(), edgecolor = palette[0], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].scatter(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(), edgecolor = palette[1], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].errorbar(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[0::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[0,i].errorbar(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[1::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'Accuracy'].to_numpy(),
                            y = df_results.loc[ind_T,'Accuracy'].to_numpy(),
                            ax = ax0[1,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[1,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[1,i].set_ylabel('Proficient accuracy (% correct)')
        ax0[1,i].set_xlabel('Naïve accuracy (% correct)')
        ax0[1,i].set_box_aspect(1)
        ax0[1,i].set_title(uni_pools[i])
        ax0[1,i].set_xlim([80,102])
        ax0[1,i].set_ylim([80,102])
        ax0[1,i].plot([80,100],[80,100],'k--', zorder = 0)
        ax0[1,i].set_title('')
            
        if i > 0:
            sns.despine(ax=ax0[0,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[1,i],left = True, trim = True, offset = 3)
            ax0[0,i].set_ylabel('')
            ax0[0,i].set_yticks([])
            ax0[1,i].set_ylabel('')
            ax0[1,i].set_yticks([])
            ax0[1,i].set_xticks(np.linspace(80,100,5))
        else:
            sns.despine(ax=ax0[0,i], trim = True, offset = 3)
            sns.despine(ax=ax0[1,i], trim = True, offset = 3)
            ax0[1,i].set_xticks(np.linspace(80,100,5))
            ax0[1,i].set_yticks(np.linspace(80,100,5))
        
        
        # CV plots
        
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'CV', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[2,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'CV', 
                        ax = ax0[2,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
        
        ax0[2,i].set_ylim([0,3])
        # ax0[2,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[2,i].set_xlim([0.5,12])
        ax0[2,i].set_xticks([1,10])
        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'CV'].to_numpy(),
                            y = df_results.loc[ind_T,'CV'].to_numpy(),
                            ax = ax0[3,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[3,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[3,i].set_ylabel('Proficient CV')
        ax0[3,i].set_xlabel('Naïve CV')
        # ax0[3,i].set_box_aspect(1)
        ax0[3,i].set_title(uni_pools[i])
        ax0[3,i].set_xlim([0,3])
        ax0[3,i].set_ylim([0,3])
        ax0[3,i].plot([0,3],[0,3],'k--', zorder = 0)
        ax0[3,i].set_title('')
        
        if i > 0:
            sns.despine(ax=ax0[2,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[3,i],left = True, trim = True, offset = 3)
            ax0[2,i].set_ylabel('')
            ax0[2,i].set_yticks([])
            ax0[3,i].set_ylabel('')
            ax0[3,i].set_yticks([])
            ax0[3,i].set_xticks(np.linspace(0,3,3))
        else:
            sns.despine(ax=ax0[2,i], trim = True, offset = 3)
            sns.despine(ax=ax0[3,i], trim = True, offset = 3)
            ax0[3,i].set_xticks(np.linspace(0,3,3))
            ax0[3,i].set_yticks(np.linspace(0,3,3))
        
        
    for a in ax0.flat:
        a.set_box_aspect(1)
     
            
    # f0.tight_layout()


#%% Line and scatter plots of accuracy and logloss

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

# cell_pool_labels = np.tile(np.array([10,20,40,80]).reshape(1,-1,1),(10,1,10))

df_results = pd.DataFrame({'Accuracy' : accu_all.flatten()*100,
                           'Number of cells' : num_cells_labels.flatten(),
                           'Cells in pool' : cell_pool_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'Log loss' : log_loss_all.flatten()})


point_colors = sns.color_palette('flare', 10)
# point_colors = sns.color_palette('Set1', 10)
point_colors = [point_colors[i] for i in [0,1,4,9]]
# point_colors_m = {s:c for s,c in zip([1,2,5,10],point_colors)}

cond_colors = sns.color_palette('colorblind',2)
cond_colors_m = {False : cond_colors[0],
                 True : cond_colors[1]}

point_styles = [r'$1$',r'$2$',r'$5$',r'$10$']
point_sizes = [2,4,10,20]
    
palette = sns.color_palette('colorblind',2)

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
                                        "lines.linewidth":0.75,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    f0,ax0 = plt.subplots(4,n_pools, sharey = False, sharex = False, figsize = (4.75,5))
    
    uni_pools = df_results['Cells in pool'].unique()
    uni_pools = uni_pools[-1::-1]
    
    for i in range(n_pools):
        
        # Accuracy plots
        ind = df_results['Cells in pool']==uni_pools[i]
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Accuracy', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[0,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'Accuracy', 
                        ax = ax0[0,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
           
        ax0[0,i].set_ylim([50,105])
        ax0[0,i].plot([1,10],[100,100],'--k', zorder = 0)
        ax0[0,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[0,i].set_xlim([0.5,12])
        ax0[0,i].set_xticks([1,10])
        ax0[0,i].yaxis.set_label_coords(x=-0.35,y=0.5)
        ax0[0,i].xaxis.set_label_coords(x=0.45,y=-0.3)


        
        if i > 0:
            ax0[0,i].scatter(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(), edgecolor = palette[0], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].scatter(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(), edgecolor = palette[1], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].errorbar(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[0::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[0,i].errorbar(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[1::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[0,i].text(11.25,88, 'all', horizontalalignment='center')

        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'Accuracy'].to_numpy(),
                            y = df_results.loc[ind_T,'Accuracy'].to_numpy(),
                            ax = ax0[1,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[1,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[1,i].set_ylabel('Proficient accuracy (% correct)')
        ax0[1,i].yaxis.set_label_coords(x=-0.35,y=0.45)
        ax0[1,i].set_xlabel('Naive accuracy (% correct)')
        ax0[1,i].xaxis.set_label_coords(x=0.45,y=-0.3)
        ax0[1,i].set_box_aspect(1)
        ax0[1,i].set_title(uni_pools[i])
        ax0[1,i].set_xlim([80,102])
        ax0[1,i].set_ylim([80,102])
        ax0[1,i].plot([80,100],[80,100],'k--', zorder = 0)
        ax0[1,i].set_title('')
            
        if i > 0:
            sns.despine(ax=ax0[0,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[1,i],left = True, trim = True, offset = 3)
            ax0[0,i].set_ylabel('')
            ax0[0,i].set_yticks([])
            ax0[1,i].set_ylabel('')
            ax0[1,i].set_yticks([])
            ax0[1,i].set_xticks(np.linspace(80,100,5))
        else:
            sns.despine(ax=ax0[0,i], trim = True, offset = 3)
            sns.despine(ax=ax0[1,i], trim = True, offset = 3)
            ax0[1,i].set_xticks(np.linspace(80,100,5))
            ax0[1,i].set_yticks(np.linspace(80,100,5))
        
        
        # logloss plots
        
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Log loss', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[2,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'Log loss', 
                        ax = ax0[2,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
        
        ax0[2,i].set_ylim([-0.01,0.4])
        ax0[2,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[2,i].set_xlim([0.5,12])
        ax0[2,i].set_xticks([1,10])
        ax0[2,i].yaxis.set_label_coords(x=-0.35,y=0.5)
        ax0[2,i].xaxis.set_label_coords(x=0.45,y=-0.3)
        
        
        if i > 0:
            ax0[2,i].scatter(11,np.fliplr(log_loss_all_full_pool)[0::2,i-1].mean(), edgecolor = palette[0], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[2,i].scatter(11.5,np.fliplr(log_loss_all_full_pool)[1::2,i-1].mean(), edgecolor = palette[1], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[2,i].errorbar(11,np.fliplr(log_loss_all_full_pool)[0::2,i-1].mean(),
                              np.fliplr(log_loss_all_full_pool)[0::2,i-1].std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[2,i].errorbar(11.5,np.fliplr(log_loss_all_full_pool)[1::2,i-1].mean(),
                              np.fliplr(log_loss_all_full_pool)[1::2,i-1].std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[2,i].text(11.25,0.18, 'all', horizontalalignment='center')
        
        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'Log loss'].to_numpy(),
                            y = df_results.loc[ind_T,'Log loss'].to_numpy(),
                            ax = ax0[3,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[3,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[3,i].set_ylabel('Proficient log loss')
        ax0[3,i].yaxis.set_label_coords(x=-0.35,y=0.45)
        ax0[3,i].set_xlabel('Naive log loss')
        ax0[3,i].xaxis.set_label_coords(x=0.45,y=-0.3)


        # ax0[3,i].set_box_aspect(1)
        ax0[3,i].set_title(uni_pools[i])
        ax0[3,i].set_xlim([-0.015,0.5])
        ax0[3,i].set_ylim([-0.015,0.5])
        ax0[3,i].plot([0,0.4],[0,0.4],'k--', zorder = 0)
        ax0[3,i].set_title('')
        
        if i > 0:
            sns.despine(ax=ax0[2,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[3,i],left = True, trim = True, offset = 3)
            ax0[2,i].set_ylabel('')
            ax0[2,i].set_yticks([])
            ax0[3,i].set_ylabel('')
            ax0[3,i].set_yticks([])
            ax0[3,i].set_xticks(np.linspace(0,0.4,3))
        else:
            sns.despine(ax=ax0[2,i], trim = True, offset = 3)
            sns.despine(ax=ax0[3,i], trim = True, offset = 3)
            ax0[3,i].set_xticks(np.linspace(0,0.4,3))
            ax0[3,i].set_yticks(np.linspace(0,0.4,3))
        
        
    for a in ax0.flat:
        a.set_box_aspect(1)
     
            
    f0.tight_layout()

#%% Keep all repeats

df = pd.DataFrame()


for i,f in enumerate(file_paths_subsets):

    print('Loading ' + f)
    decode = np.load(f,allow_pickle = True)[()]
    
    pool_sizes = decode['pool_sizes']
          
    stim = decode['stim']
    
    pred_stim = decode['pred_stim']
    
    features = decode['features']
       
    num_trials, num_features, num_pools, num_repeats = pred_stim.shape
        
    accu = pred_stim == np.tile(stim.reshape(-1,1,1,1), (1,num_features,num_pools,num_repeats))
    accu = accu.sum(0)/len(stim)
    
    d_s = {'accuracy' : accu.ravel(),
           'OSI' : decode['OSI'][features].ravel(),
           'mean' : decode['mean_pref'][features].ravel(),
           'std' : decode['std_pref'][features].ravel(),
           'lt_sparseness' : decode['lt_sparseness'][features].ravel(),
           'feature_number' : np.tile(np.arange(1,num_features+1).reshape(-1,1,1), 
                                    (1,num_pools,num_repeats)).ravel(),
           'pool_size' : np.tile(np.array(decode['pool_sizes']).reshape(1,-1,1), 
                                 (num_features,1,num_repeats)).ravel(),
           'mean_pref_ori' : decode['mean_pref_ori'][features].ravel(),
           # 'pref_blank' : decode['pref_blank'][features].ravel(),
           'subject' : np.tile(subjects[i], (num_features,num_pools,num_repeats)).ravel(),
           'trained' : np.tile(trained[i], (num_features,num_pools,num_repeats)).ravel(),
           'repeat_num' : np.tile(np.arange(num_repeats).reshape(1,1,-1), (num_features,num_pools,1)).ravel(),
           'CV' : decode['std_pref'][features].ravel()/decode['mean_pref'][features].ravel()
           }
        
    
    df = pd.concat([df,pd.DataFrame(d_s)])

df = df.reset_index(drop=True)

#%% 


# df_plot = df[df.subject=='SF180613'].reset_index(drop=True)
df_plot = df.copy()

pool_sizes = df_plot.pool_size.unique()

colors = sns.color_palette('colorblind',2)

y = ['accuracy','mean_pref_ori','OSI']
y_labels = ['Accuracy','Ori. pref. (deg)','Selectivity']
y_lims = [[0.4,1],[-11.25,180-11.25],[0,1]]
y_ticks = [[0,0.5,1],np.arange(0,180,22.5), [0,0.5,1]]
y_tick_labels = [y_ticks[0], np.ceil(y_ticks[1]).astype(int), y_ticks[2]]
xbins = np.arange(num_features)
bins = [[xbins,np.arange(0,1.1,0.1)],[xbins,np.linspace(-11.25,180-11.25,9)],[xbins,np.arange(0,1.1,0.1)]]
discrete_flag = [[True,False],[True,False],[True,False]]
vmin = [0,0,0]
vmax = [1,0.45,0.45]    

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":8,"axes.titlesize":8,
                                        "axes.labelsize":8,
                                        "axes.linewidth":1,
                                        'xtick.labelsize':6,
                                        'ytick.labelsize':6,
                                        "xtick.major.width":1,
                                        "ytick.major.width":1,
                                        "xtick.major.size":5,
                                        "ytick.major.size":5,
                                        "patch.linewidth":1,
                                        "lines.markersize":4,
                                        "lines.linewidth":1,
                                        "legend.fontsize":6,
                                        "legend.title_fontsize":6}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    fig = plt.figure()
    
    subfigs = fig.subfigures(3,1)
                                      
    for f in range(3):
        
        ax = subfigs[f].subplots(2,num_pools, sharex = False)
        
        for i,p in enumerate(pool_sizes):
            ind_n = np.logical_and(df_plot.trained==False, df_plot.pool_size == p)
            ind_t = np.logical_and(df_plot.trained, df_plot.pool_size == p)
            
            
            if i == 3:
                cbar = True
            else:
                cbar = False
                
                      
            cbar_ax = subfigs[f].add_axes([0.93, 0.555, 0.01, 0.3])
            sns.histplot(data = df_plot[ind_n], x = 'feature_number', y = y[f],
                        bins = bins[f], color = colors[0], ax = ax[0,i], weights = 1/(num_repeats*len(subjects)/2),
                        discrete = discrete_flag[f], vmin = vmin[f], vmax = vmax[f],
                        cbar = cbar, cbar_ax = cbar_ax)
            cbar_ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False,
                                labelbottom = False, labeltop = False)
            cbar_ax.tick_params(axis = 'y', which = 'both', left = False,
                                labelleft = False)
            cbar_ax = subfigs[f].add_axes([0.93, 0.129, 0.01, 0.3])
            sns.histplot(data = df_plot[ind_t], x = 'feature_number', y = y[f],
                        bins = bins[f], color = colors[1], ax = ax[1,i], weights = 1/(num_repeats*len(subjects)/2),
                        discrete = discrete_flag[f], vmin = vmin[f], vmax = vmax[f],
                        cbar = cbar, cbar_ax = cbar_ax)
            cbar_ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False,
                                labelbottom = False, labeltop = False)
            cbar_ax.tick_params(axis = 'y', which = 'both', left = False,
                                labelleft = False)
            if i == 0:
                # ax[0,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
                # ax[1,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
                # if f != 1:
                #     ax[0,i].set_ylabel(y_labels[f])
                #     ax[1,i].set_ylabel(y_labels[f])
                # else:
                #     ax[0,i].set_ylabel(y_labels[f], labelpad = 2.8)
                #     ax[1,i].set_ylabel(y_labels[f], labelpad = 2.8)
                ax[0,i].set_yticks(y_ticks[f])
                ax[1,i].set_yticks(y_ticks[f])
                ax[0,i].set_yticklabels(y_tick_labels[f])
                ax[1,i].set_yticklabels(y_tick_labels[f])
        
            else:
                ax[0,i].set_yticks([])
                ax[1,i].set_yticks([])
                
            if f != 2:
                ax[0,i].set_xticks([])
                ax[1,i].set_xticks([])
                ax[0,i].set_xlabel('')
                ax[1,i].set_xlabel('')
            else:
                ax[0,i].set_xlabel('')
                ax[0,i].set_xticks([])
                ax[1,i].set_xlabel('Cell number')
                ax[0,i].set_xlim([1,num_features])
                ax[1,i].set_xlim([1,num_features])
                ax[1,i].set_xticks([1,5,10])
            
            
            ax[0,i].set_ylim(y_lims[f])
            ax[1,i].set_ylim(y_lims[f])
            ax[0,i].set_xlim([0.5,10.5])
            ax[1,i].set_xlim([0.5,10.5])
            ax[0,i].set_ylabel('')
            ax[1,i].set_ylabel('')
            
       
        subfigs[f].supylabel(y_labels[f])
            
            
            # if f < 2:
            #     if i == 0:
            #         sns.despine(ax=ax[0,i], bottom = True, trim = True)
            #         sns.despine(ax=ax[1,i], bottom = True, trim = True)
            #     else:
            #         sns.despine(ax=ax[0,i], bottom = True, left = True, trim = True)
            #         sns.despine(ax=ax[1,i], bottom = True, left = True, trim = True)
            # else:
            #     if i == 0:
            #         sns.despine(ax=ax[0,i], bottom = True, trim = True)
            #         sns.despine(ax=ax[1,i], trim = True)
            #     else:
            #         sns.despine(ax=ax[0,i], bottom = True, left = True, trim = True)
            #         sns.despine(ax=ax[1,i], left = True, trim = True)
            
    
    
    # fig = plt.figure(constrained_layout=True)
    
    # subfigs = fig.subfigures(3,1)
    
    # colors = sns.color_palette('colorblind',2)
    
    # y = ['mean','std','CoV']
    # y_labels = ['Mean resp.', 'STD resp.', 'Cov resp.']
    # bins = [[xbins,np.linspace(0,1,40)],[xbins,np.linspace(0,1,40)],[xbins,np.linspace(0,1,40)]]
                                          
    # for f in range(3):
        
    #     ax = subfigs[f].subplots(2,num_pools)
        
    #     for i,p in enumerate(pool_sizes):
    #         ind_n = np.logical_and(df_plot.trained==False, df_plot.pool_size == p)
    #         ind_t = np.logical_and(df_plot.trained, df_plot.pool_size == p)
        
            
    #         sns.histplot(data = df_plot[ind_n], x = 'feature_number', y = y[f],
    #                     bins = bins[f], color = colors[0], ax = ax[0,i], weights = 0.0002,
    #                     discrete = (True,False))
    #         sns.histplot(data = df_plot[ind_t], x = 'feature_number', y = y[f],
    #                     bins = bins[f], color = colors[1], ax = ax[1,i], weights = 0.0002,
    #                     discrete = (True,False))
            
    #         if i == 0:
    #             # ax[0,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
    #             # ax[1,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
    #             ax[0,i].set_ylabel(y_labels[f])
    #             ax[1,i].set_ylabel(y_labels[f])
        
    #         else:
    #             ax[0,i].set_yticks([])
    #             ax[1,i].set_yticks([])
    #             ax[0,i].set_ylabel('')
    #             ax[1,i].set_ylabel('')
                
    #         if f != 5:
    #             ax[0,i].set_xticks([])
    #             ax[1,i].set_xticks([])
    #             ax[0,i].set_xlabel('')
    #             ax[1,i].set_xlabel('')
    #         else:
    #             ax[0,i].set_xticks([])
    #             ax[0,i].set_xlabel('')
    #             ax[1,i].set_xlabel('Feature number')
    #             ax[0,i].set_xlim([1,num_features])
    #             ax[1,i].set_xlim([1,num_features])
    
    
#%% 


# df_plot = df[df.subject=='SF180613'].reset_index(drop=True)
df_plot = df.copy()

pool_sizes = df_plot.pool_size.unique()

colors = sns.color_palette('colorblind',2)

y = ['accuracy','mean_pref_ori', 'OSI', 'CV']
y_labels = ['Accuracy','Ori. pref. (deg)','Selectivity', 'Coef. of variation']
y_lims = [[0.4,1],[-11.25,180-11.25],[0,1], [0,2]]
y_ticks = [[0,0.5,1],np.arange(0,180,22.5), [0,0.5,1], [0,1,2]]
y_tick_labels = [y_ticks[0], np.ceil(y_ticks[1]).astype(int), y_ticks[2], y_ticks[3]]
xbins = np.arange(num_features)
bins = [[xbins,np.arange(0,1.1,0.1)],[xbins,np.linspace(-11.25,180-11.25,9)],[xbins,np.arange(0,1.1,0.1)],
        [xbins,np.arange(0,2.1,0.2)]]
discrete_flag = [[True,False],[True,False],[True,False], [True,False]]
vmin = [0,0,0,0]
vmax = [1,0.45,0.45,0.6]    

pool_sizes = pool_sizes[-1::-1]

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


    fig = plt.figure(figsize = (4.75,6))
    
    subfigs = fig.subfigures(len(y),1)
                                      
    for f in range(len(y)):
        
        ax = subfigs[f].subplots(2, num_pools, sharex = False)
        
        for i,p in enumerate(pool_sizes):
            ind_n = np.logical_and(df_plot.trained==False, df_plot.pool_size == p)
            ind_t = np.logical_and(df_plot.trained, df_plot.pool_size == p)
            
            
            if i == 3:
                cbar = True
            else:
                cbar = False
                
                      
            cbar_ax = subfigs[f].add_axes([0.93, 0.555, 0.01, 0.3])
            sns.histplot(data = df_plot[ind_n], x = 'feature_number', y = y[f],
                        bins = bins[f], color = colors[0], ax = ax[0,i], weights = 1/(num_repeats*len(subjects)/2),
                        discrete = discrete_flag[f], vmin = vmin[f], vmax = vmax[f],
                        cbar = cbar, cbar_ax = cbar_ax)
            cbar_ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False,
                                labelbottom = False, labeltop = False)
            cbar_ax.tick_params(axis = 'y', which = 'both', left = False,
                                labelleft = False)
            cbar_ax = subfigs[f].add_axes([0.93, 0.129, 0.01, 0.3])
            sns.histplot(data = df_plot[ind_t], x = 'feature_number', y = y[f],
                        bins = bins[f], color = colors[1], ax = ax[1,i], weights = 1/(num_repeats*len(subjects)/2),
                        discrete = discrete_flag[f], vmin = vmin[f], vmax = vmax[f],
                        cbar = cbar, cbar_ax = cbar_ax)
            cbar_ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False,
                                labelbottom = False, labeltop = False)
            cbar_ax.tick_params(axis = 'y', which = 'both', left = False,
                                labelleft = False)
            if i == 0:
                # ax[0,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
                # ax[1,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
                # if f != 1:
                #     ax[0,i].set_ylabel(y_labels[f])
                #     ax[1,i].set_ylabel(y_labels[f])
                # else:
                #     ax[0,i].set_ylabel(y_labels[f], labelpad = 2.8)
                #     ax[1,i].set_ylabel(y_labels[f], labelpad = 2.8)
                ax[0,i].set_yticks(y_ticks[f])
                ax[1,i].set_yticks(y_ticks[f])
                ax[0,i].set_yticklabels(y_tick_labels[f])
                ax[1,i].set_yticklabels(y_tick_labels[f])
        
            else:
                ax[0,i].set_yticks([])
                ax[1,i].set_yticks([])
                
            if f != len(y)-1:
                ax[0,i].set_xticks([])
                ax[1,i].set_xticks([])
                ax[0,i].set_xlabel('')
                ax[1,i].set_xlabel('')
            else:
                ax[0,i].set_xlabel('')
                ax[0,i].set_xticks([])
                ax[1,i].set_xlabel('Cell number')
                ax[0,i].set_xlim([1,num_features])
                ax[1,i].set_xlim([1,num_features])
                ax[1,i].set_xticks([1,10])
            
            
            ax[0,i].set_ylim(y_lims[f])
            ax[1,i].set_ylim(y_lims[f])
            ax[0,i].set_xlim([0.5,10.5])
            ax[1,i].set_xlim([0.5,10.5])
            ax[0,i].set_ylabel('')
            ax[1,i].set_ylabel('')
            
            if f == 0:
                ax[0,i].set_title(p)
       
        subfigs[f].supylabel(y_labels[f])   
            
            # if f < 2:
            #     if i == 0:
            #         sns.despine(ax=ax[0,i], bottom = True, trim = True)
            #         sns.despine(ax=ax[1,i], bottom = True, trim = True)
            #     else:
            #         sns.despine(ax=ax[0,i], bottom = True, left = True, trim = True)
            #         sns.despine(ax=ax[1,i], bottom = True, left = True, trim = True)
            # else:
            #     if i == 0:
            #         sns.despine(ax=ax[0,i], bottom = True, trim = True)
            #         sns.despine(ax=ax[1,i], trim = True)
            #     else:
            #         sns.despine(ax=ax[0,i], bottom = True, left = True, trim = True)
            #         sns.despine(ax=ax[1,i], left = True, trim = True)
            
    
    
    # fig = plt.figure(constrained_layout=True)
    
    # subfigs = fig.subfigures(3,1)
    
    # colors = sns.color_palette('colorblind',2)
    
    # y = ['mean','std','CoV']
    # y_labels = ['Mean resp.', 'STD resp.', 'Cov resp.']
    # bins = [[xbins,np.linspace(0,1,40)],[xbins,np.linspace(0,1,40)],[xbins,np.linspace(0,1,40)]]
                                          
    # for f in range(3):
        
    #     ax = subfigs[f].subplots(2,num_pools)
        
    #     for i,p in enumerate(pool_sizes):
    #         ind_n = np.logical_and(df_plot.trained==False, df_plot.pool_size == p)
    #         ind_t = np.logical_and(df_plot.trained, df_plot.pool_size == p)
        
            
    #         sns.histplot(data = df_plot[ind_n], x = 'feature_number', y = y[f],
    #                     bins = bins[f], color = colors[0], ax = ax[0,i], weights = 0.0002,
    #                     discrete = (True,False))
    #         sns.histplot(data = df_plot[ind_t], x = 'feature_number', y = y[f],
    #                     bins = bins[f], color = colors[1], ax = ax[1,i], weights = 0.0002,
    #                     discrete = (True,False))
            
    #         if i == 0:
    #             # ax[0,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
    #             # ax[1,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
    #             ax[0,i].set_ylabel(y_labels[f])
    #             ax[1,i].set_ylabel(y_labels[f])
        
    #         else:
    #             ax[0,i].set_yticks([])
    #             ax[1,i].set_yticks([])
    #             ax[0,i].set_ylabel('')
    #             ax[1,i].set_ylabel('')
                
    #         if f != 5:
    #             ax[0,i].set_xticks([])
    #             ax[1,i].set_xticks([])
    #             ax[0,i].set_xlabel('')
    #             ax[1,i].set_xlabel('')
    #         else:
    #             ax[0,i].set_xticks([])
    #             ax[0,i].set_xlabel('')
    #             ax[1,i].set_xlabel('Feature number')
    #             ax[0,i].set_xlim([1,num_features])
    #             ax[1,i].set_xlim([1,num_features])
    
#%% 


# df_plot = df[df.subject=='SF180613'].reset_index(drop=True)
df_plot = df.copy()

pool_sizes = df_plot.pool_size.unique()

colors = sns.color_palette('colorblind',2)

y = ['mean_pref_ori', 'OSI', 'CV']
y_labels = ['Ori. pref. (deg)','Selectivity', 'Coef. of variation']
y_lims = [[-11.25,180-11.25],[0,1], [0,2]]
y_ticks = [np.arange(0,180,22.5), [0,0.5,1], [0,1,2]]
y_tick_labels = [np.ceil(y_ticks[0]).astype(int), y_ticks[1], y_ticks[2]]
xbins = np.arange(num_features)
bins = [[xbins,np.linspace(-11.25,180-11.25,9)],[xbins,np.arange(0,1.1,0.1)],
        [xbins,np.arange(0,2.1,0.2)]]
discrete_flag = [[True,False],[True,False], [True,False]]
vmin = [0,0,0]
vmax = [0.45,0.45,0.6]    

pool_sizes = pool_sizes[-1::-1]

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


    fig = plt.figure(figsize = (4.75,6))
    
    subfigs = fig.subfigures(len(y),1)
                                      
    for f in range(len(y)):
        
        ax = subfigs[f].subplots(2, num_pools, sharex = False)
        
        for i,p in enumerate(pool_sizes):
            ind_n = np.logical_and(df_plot.trained==False, df_plot.pool_size == p)
            ind_t = np.logical_and(df_plot.trained, df_plot.pool_size == p)
            
            
            if i == 3:
                cbar = True
            else:
                cbar = False
                
                      
            cbar_ax = subfigs[f].add_axes([0.93, 0.555, 0.01, 0.3])
            sns.histplot(data = df_plot[ind_n], x = 'feature_number', y = y[f],
                        bins = bins[f], ax = ax[0,i], 
                        weights = 1/(num_repeats*len(subjects)/2),
                        discrete = discrete_flag[f], vmin = vmin[f], vmax = vmax[f],
                        cbar = cbar, cbar_ax = cbar_ax,
                        palette = 'viridis')
            cbar_ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False,
                                labelbottom = False, labeltop = False)
            cbar_ax.tick_params(axis = 'y', which = 'both', left = False,
                                labelleft = False)
            cbar_ax = subfigs[f].add_axes([0.93, 0.129, 0.01, 0.3])
            sns.histplot(data = df_plot[ind_t], x = 'feature_number', y = y[f],
                        bins = bins[f], ax = ax[1,i], 
                        weights = 1/(num_repeats*len(subjects)/2),
                        discrete = discrete_flag[f], vmin = vmin[f], vmax = vmax[f],
                        cbar = cbar, cbar_ax = cbar_ax,
                        palette = 'viridis')
            cbar_ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False,
                                labelbottom = False, labeltop = False)
            cbar_ax.tick_params(axis = 'y', which = 'both', left = False,
                                labelleft = False)
            if i == 0:
                # ax[0,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
                # ax[1,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
                # if f != 1:
                #     ax[0,i].set_ylabel(y_labels[f])
                #     ax[1,i].set_ylabel(y_labels[f])
                # else:
                #     ax[0,i].set_ylabel(y_labels[f], labelpad = 2.8)
                #     ax[1,i].set_ylabel(y_labels[f], labelpad = 2.8)
                ax[0,i].set_yticks(y_ticks[f])
                ax[1,i].set_yticks(y_ticks[f])
                ax[0,i].set_yticklabels(y_tick_labels[f])
                ax[1,i].set_yticklabels(y_tick_labels[f])
        
            else:
                ax[0,i].set_yticks([])
                ax[1,i].set_yticks([])
                
            if f != len(y)-1:
                ax[0,i].set_xticks([])
                ax[1,i].set_xticks([])
                ax[0,i].set_xlabel('')
                ax[1,i].set_xlabel('')
            else:
                ax[0,i].set_xlabel('')
                ax[0,i].set_xticks([])
                ax[1,i].set_xlabel('Cell number')
                ax[0,i].set_xlim([1,num_features])
                ax[1,i].set_xlim([1,num_features])
                ax[1,i].set_xticks([1,10])
            
            
            ax[0,i].set_ylim(y_lims[f])
            ax[1,i].set_ylim(y_lims[f])
            ax[0,i].set_xlim([0.5,10.5])
            ax[1,i].set_xlim([0.5,10.5])
            ax[0,i].set_ylabel('')
            ax[1,i].set_ylabel('')
            
            if f == 0:
                ax[0,i].set_title(p)
       
        subfigs[f].supylabel(y_labels[f])   
            
            # if f < 2:
            #     if i == 0:
            #         sns.despine(ax=ax[0,i], bottom = True, trim = True)
            #         sns.despine(ax=ax[1,i], bottom = True, trim = True)
            #     else:
            #         sns.despine(ax=ax[0,i], bottom = True, left = True, trim = True)
            #         sns.despine(ax=ax[1,i], bottom = True, left = True, trim = True)
            # else:
            #     if i == 0:
            #         sns.despine(ax=ax[0,i], bottom = True, trim = True)
            #         sns.despine(ax=ax[1,i], trim = True)
            #     else:
            #         sns.despine(ax=ax[0,i], bottom = True, left = True, trim = True)
            #         sns.despine(ax=ax[1,i], left = True, trim = True)
            
    
    
    # fig = plt.figure(constrained_layout=True)
    
    # subfigs = fig.subfigures(3,1)
    
    # colors = sns.color_palette('colorblind',2)
    
    # y = ['mean','std','CoV']
    # y_labels = ['Mean resp.', 'STD resp.', 'Cov resp.']
    # bins = [[xbins,np.linspace(0,1,40)],[xbins,np.linspace(0,1,40)],[xbins,np.linspace(0,1,40)]]
                                          
    # for f in range(3):
        
    #     ax = subfigs[f].subplots(2,num_pools)
        
    #     for i,p in enumerate(pool_sizes):
    #         ind_n = np.logical_and(df_plot.trained==False, df_plot.pool_size == p)
    #         ind_t = np.logical_and(df_plot.trained, df_plot.pool_size == p)
        
            
    #         sns.histplot(data = df_plot[ind_n], x = 'feature_number', y = y[f],
    #                     bins = bins[f], color = colors[0], ax = ax[0,i], weights = 0.0002,
    #                     discrete = (True,False))
    #         sns.histplot(data = df_plot[ind_t], x = 'feature_number', y = y[f],
    #                     bins = bins[f], color = colors[1], ax = ax[1,i], weights = 0.0002,
    #                     discrete = (True,False))
            
    #         if i == 0:
    #             # ax[0,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
    #             # ax[1,i].set_yticks(np.ceil(np.arange(0,180,22.5)))
    #             ax[0,i].set_ylabel(y_labels[f])
    #             ax[1,i].set_ylabel(y_labels[f])
        
    #         else:
    #             ax[0,i].set_yticks([])
    #             ax[1,i].set_yticks([])
    #             ax[0,i].set_ylabel('')
    #             ax[1,i].set_ylabel('')
                
    #         if f != 5:
    #             ax[0,i].set_xticks([])
    #             ax[1,i].set_xticks([])
    #             ax[0,i].set_xlabel('')
    #             ax[1,i].set_xlabel('')
    #         else:
    #             ax[0,i].set_xticks([])
    #             ax[0,i].set_xlabel('')
    #             ax[1,i].set_xlabel('Feature number')
    #             ax[0,i].set_xlim([1,num_features])
    #             ax[1,i].set_xlim([1,num_features])

#%% Calculation proportion of repeats with each cell stat (binned) - plot averages as heatmaps

df_plot = df.copy()

df_plot['pref_bin'] = pd.cut(df_plot.mean_pref_ori, np.linspace(-11.25,180-11.25,9), 
                             labels = np.ceil(np.arange(0,180,22.5)).astype(int))
df_plot['r_bin'] = pd.cut(df_plot.OSI, np.linspace(0,1,11),
                          labels = np.arange(10))
df_plot['cv_bin'] = pd.cut(df_plot.CV, np.linspace(0,2,11))


df_pref = df_plot.groupby(['subject','trained','feature_number','pool_size']
                     )['pref_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()
df_pref = df_pref.groupby(['trained','feature_number','pool_size','level_4'], as_index = False).mean()
df_r = df_plot.groupby(['subject','trained','feature_number','pool_size']
                     )['r_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()
df_cv = df_plot.groupby(['subject','trained','feature_number','pool_size']
                     )['cv_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()

pool_sizes = df_pref.pool_size.unique()




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
        
        fig = plt.figure(figsize = (4.75,6))
        
        subfigs = fig.subfigures(3,1)
    
        ax = subfigs[0].subplots(2, 4, sharex = False)
        cbar_ax = subfigs[0].add_axes([0.93, 0.35, 0.01, 0.3])

        for i,p in enumerate(pool_sizes[::-1]):
        
            df_pref_n = pd.pivot_table(df_pref[np.logical_and(df_pref.pool_size==p,
                                                              df_pref.trained==False)],
                                       values = 'prop_repeats',
                                       index = 'level_4',
                                       columns = 'feature_number')
            df_pref_t = pd.pivot_table(df_pref[np.logical_and(df_pref.pool_size==p,
                                                              df_pref.trained)],
                                       values = 'prop_repeats',
                                       index = 'level_4',
                                       columns = 'feature_number')
            
            if i == 3:
                cbar = True
            else:
                cbar = False
            
            sns.heatmap(df_pref_n, ax = ax[0,i], 
                        vmax = np.max([df_pref_t.to_numpy().max(),df_pref_t.to_numpy().max()]), 
                        vmin = 0, cbar = cbar, cbar_ax = cbar_ax)
            sns.heatmap(df_pref_t, ax = ax[1,i], 
                        vmax = np.max([df_pref_t.to_numpy().max(),df_pref_t.to_numpy().max()]), 
                        vmin = 0, cbar = cbar, cbar_ax = cbar_ax)
            # sns.heatmap(df_pref_n, ax = ax[0,i])
            # sns.heatmap(df_pref_t, ax = ax[1,i])
            
            if i > 0:
                ax[0,i].set_yticks([])
                ax[0,i].set_ylabel('')
                ax[1,i].set_yticks([])
                ax[1,i].set_ylabel('')
            else:
                ax[0,i].set_ylabel('Naïve')
                ax[1,i].set_ylabel('Proficient')
                
            ax[0,i].set_xticks([])
            ax[0,i].set_xlabel('')
            ax[1,i].set_xticks([])
            ax[1,i].set_xlabel('')
            
            ax[0,i].set_title(p)
            
        
        ax = subfigs[1].subplots(2, 4, sharex = False)
        cbar_ax = subfigs[1].add_axes([0.93, 0.35, 0.01, 0.3])
        
        for i,p in enumerate(pool_sizes[::-1]):

            df_r_n = pd.pivot_table(df_r[np.logical_and(df_r.pool_size==p,
                                                              df_r.trained==False)], 
                                    values = 'prop_repeats',
                                    index = 'level_4',
                                    columns = 'feature_number')
            df_r_t = pd.pivot_table(df_r[np.logical_and(df_r.pool_size==p,
                                                              df_r.trained)], 
                                    values = 'prop_repeats',
                                    index = 'level_4',
                                    columns = 'feature_number')
            if i == 3:
                cbar = True
            else:
                cbar = False
            
            sns.heatmap(df_r_n, ax = ax[0,i], 
                        vmax = np.max([df_r_t.to_numpy().max(),df_r_t.to_numpy().max()]), 
                        vmin = 0, cbar = cbar, cbar_ax = cbar_ax)
            sns.heatmap(df_r_t, ax = ax[1,i], 
                        vmax = np.max([df_r_t.to_numpy().max(),df_r_t.to_numpy().max()]), 
                        vmin = 0, cbar = cbar, cbar_ax = cbar_ax)
            # sns.heatmap(df_r_n, ax = ax[0,i])
            # sns.heatmap(df_r_t, ax = ax[1,i])
            
            if i > 0:
                ax[0,i].set_yticks([])
                ax[0,i].set_ylabel('')
                ax[1,i].set_yticks([])
                ax[1,i].set_ylabel('')
            else:
                ax[0,i].set_ylabel('Naïve')
                ax[1,i].set_ylabel('Proficient')
                ax[0,i].set_yticks([0,10])
                ax[0,i].set_yticklabels([0,1])
                ax[1,i].set_yticks([0,10])
                ax[1,i].set_yticklabels([0,1])
                
            ax[0,i].set_xticks([])
            ax[0,i].set_xlabel('')
            ax[1,i].set_xticks([])
            ax[1,i].set_xlabel('')

        ax = subfigs[2].subplots(2, 4, sharex = False)
        cbar_ax = subfigs[2].add_axes([0.93, 0.35, 0.01, 0.3])
        
        for i,p in enumerate(pool_sizes[::-1]):
            
            df_cv_n = pd.pivot_table(df_cv[np.logical_and(df_cv.pool_size==p,
                                                              df_cv.trained==False)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')
            df_cv_t = pd.pivot_table(df_cv[np.logical_and(df_cv.pool_size==p,
                                                              df_cv.trained)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')
            
            if i == 3:
                cbar = True
            else:
                cbar = False
                
            sns.heatmap(df_cv_n, ax = ax[0,i], 
                        vmax = np.max([df_r_t.to_numpy().max(),df_r_t.to_numpy().max()]), 
                        vmin = 0, cbar = cbar, cbar_ax = cbar_ax)
            sns.heatmap(df_cv_t, ax = ax[1,i], 
                        vmax = np.max([df_r_t.to_numpy().max(),df_r_t.to_numpy().max()]), 
                        vmin = 0, cbar = cbar, cbar_ax = cbar_ax)
            # sns.heatmap(df_cv_n, ax = ax[0,i])
            # sns.heatmap(df_cv_t, ax = ax[1,i])
            
            if i > 0:
                ax[0,i].set_yticks([])
                ax[0,i].set_ylabel('')
                ax[1,i].set_yticks([])
                ax[1,i].set_ylabel('')
            else:
                ax[0,i].set_ylabel('Naïve')
                ax[1,i].set_ylabel('Proficient')
                ax[0,i].set_yticks([0,10])
                ax[0,i].set_yticklabels([0,2])
                ax[1,i].set_yticks([0,10])
                ax[1,i].set_yticklabels([0,2])
                
            ax[0,i].set_xticks([])
            ax[0,i].set_xlabel('')
            ax[1,i].set_xlabel('Cell number')
                
        subfigs[0].supylabel('Mean ori. pref.')
        subfigs[1].supylabel('Selectivity')
        subfigs[2].supylabel('Coefficient of variation')
        
    
#%% Calculation proportion of repeats with each cell stat (binned) - plot histograms for first cell

df_plot = df.copy()

df_plot['pref_bin'] = pd.cut(df_plot.mean_pref_ori, np.linspace(-11.25,180-11.25,9), 
                             labels = np.ceil(np.arange(0,180,22.5)).astype(int))
df_plot['r_bin'] = pd.cut(df_plot.OSI, np.linspace(0,1,11),
                          labels = np.arange(10))
df_plot['cv_bin'] = pd.cut(df_plot.CV, np.linspace(0,2,11))


df_pref = df_plot.groupby(['subject','trained','feature_number','pool_size']
                     )['pref_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()
df_pref = df_pref.groupby(['trained','feature_number','pool_size','level_4'], as_index = False).mean()
df_r = df_plot.groupby(['subject','trained','feature_number','pool_size']
                     )['r_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()
df_cv = df_plot.groupby(['subject','trained','feature_number','pool_size']
                     )['cv_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()

pool_sizes = df_pref.pool_size.unique()


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
        
        fig = plt.figure(figsize = (6,6))
        
        subfigs = fig.subfigures(3,1)
    
        ax = subfigs[0].subplots(1, 4, sharex = False)

        for i,p in enumerate(pool_sizes[::-1]):
        
            y_n = pd.pivot_table(df_pref[np.logical_and(df_pref.pool_size==p,
                                                              df_pref.trained==False)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')[1]
            y_t = pd.pivot_table(df_pref[np.logical_and(df_pref.pool_size==p,
                                                              df_pref.trained)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')[1]
            
            # x = pd.pivot_table(df_pref[np.logical_and(df_pref.pool_size==p,
            #                                                   df_pref.trained)],
            #                          values = 'prop_repeats',
            #                          index = 'level_4',
            #                          columns = 'feature_number').index.to_numpy()
            # x = [x_a[i].left for i in range(len(x_a))]
            # x.append(x_a[-1].right)
            
            ax[i].stairs(y_n, np.linspace(-11.25,180-11.25,9), color = cond_colors[0])
            ax[i].stairs(y_t, np.linspace(-11.25,180-11.25,9), color = cond_colors[1])
            
            if i > 0:
                ax[i].set_yticks([])
                ax[i].set_ylabel('')
                sns.despine(ax=ax[i],left=True)
            else:
                ax[i].set_ylabel('Proportion of repeats')
                sns.despine(ax=ax[i])

                
            ax[i].set_xlabel('Mean ori. pref.')
            ax[i].set_xticks(np.arange(0,180,22.5))
            ax[i].set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
            
            ax[i].set_title(p)
            
        
        ax = subfigs[1].subplots(1, 4, sharex = False)
        
        for i,p in enumerate(pool_sizes[::-1]):

            y_n = pd.pivot_table(df_r[np.logical_and(df_r.pool_size==p,
                                                              df_r.trained==False)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')[1]
            y_t = pd.pivot_table(df_r[np.logical_and(df_r.pool_size==p,
                                                              df_r.trained)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')[1]
            
            # x_a = pd.pivot_table(df_r[np.logical_and(df_r.pool_size==p,
            #                                                   df_r.trained)],
            #                          values = 'prop_repeats',
            #                          index = 'level_4',
            #                          columns = 'feature_number').index.to_numpy()
            # x = [x_a[i].left for i in range(len(x_a))]
            # x.append(x_a[-1].right)
            
            ax[i].stairs(y_n, np.linspace(0,1,11), color = cond_colors[0])
            ax[i].stairs(y_t, np.linspace(0,1,11), color = cond_colors[1])

            
            if i > 0:
                ax[i].set_yticks([])
                ax[i].set_ylabel('')
                sns.despine(ax=ax[i],left=True)
            
            else:
               ax[i].set_ylabel('Proportion of repeats')
               sns.despine(ax=ax[i])

                
            ax[i].set_xlabel('Selectivity')


        ax = subfigs[2].subplots(1, 4, sharex = False)
        
        for i,p in enumerate(pool_sizes[::-1]):
            
            y_n = pd.pivot_table(df_cv[np.logical_and(df_cv.pool_size==p,
                                                              df_cv.trained==False)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')[1]
            y_t = pd.pivot_table(df_cv[np.logical_and(df_cv.pool_size==p,
                                                              df_cv.trained)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number')[1]
            
            x_a = pd.pivot_table(df_cv[np.logical_and(df_cv.pool_size==p,
                                                              df_cv.trained)],
                                     values = 'prop_repeats',
                                     index = 'level_4',
                                     columns = 'feature_number').index.to_numpy()
            x = [x_a[i].left for i in range(len(x_a))]
            x.append(x_a[-1].right)
            
            ax[i].stairs(y_n, x, color = cond_colors[0])
            ax[i].stairs(y_t, x, color = cond_colors[1])
            
            if i > 0:
                ax[i].set_yticks([])
                ax[i].set_ylabel('')
                sns.despine(ax=ax[i],left = True)
            else:
                ax[i].set_ylabel('Proportion of repeats')
                sns.despine(ax=ax[i])
                
                
            ax[i].set_xlabel('CV')
            
        
        
        # fig.tight_layout() 
        # subfigs[0].supylabel('Mean ori. pref.')
        # subfigs[1].supylabel('Selectivity')
        # subfigs[2].supylabel('Coefficient of variation')

#%% Plot 1d histograms for each cell stat, where color is both condition and feature number

cond_colors = sns.color_palette('colorblind',n_colors=2)

n_colors = sns.light_palette(cond_colors[0],n_colors=10, reverse = True)
t_colors = sns.light_palette(cond_colors[1],n_colors=10, reverse = True)

df_plot = df.copy()

df_plot['pref_bin'] = pd.cut(df_plot.mean_pref_ori, np.linspace(-11.25,180-11.25,9), 
                             labels = np.ceil(np.arange(0,180,22.5)).astype(int))
df_plot['r_bin'] = pd.cut(df_plot.OSI, np.linspace(0,1,11),
                          labels = np.arange(10))
df_plot['cv_bin'] = pd.cut(df_plot.CV, np.linspace(0,2,11),
                           labels = np.arange(10))


df_pref = df_plot.groupby(['subject','trained','feature_number', 'pool_size']
                     )['pref_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()
# df_pref = df_pref.groupby(['trained','feature_number','pool_size','level_4'], as_index = False).mean()


ind = df_pref.feature_number <= 3
sns.catplot(data = df_pref[ind], x = 'level_4', y = 'prop_repeats', hue='trained', 
            linewidth = 0.5, edgecolor = 'k',
            palette = 'colorblind', ci = 68, row = 'feature_number', col = 'pool_size',
            kind = 'bar')

df_r = df_plot.groupby(['subject','trained','feature_number', 'pool_size']
                     )['r_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()
# df_r = df_r.groupby(['trained','feature_number','level_4'], as_index = False).mean()

ind = df_r.feature_number <= 3
sns.catplot(data = df_r[ind], x = 'level_4', y = 'prop_repeats', hue='trained', 
            linewidth = 0.5, edgecolor = 'k',
            palette = 'colorblind', ci = 68, row = 'feature_number', col = 'pool_size',
            kind = 'bar')

df_cv = df_plot.groupby(['subject','trained','feature_number','pool_size']
                     )['cv_bin'].value_counts(normalize=True).rename('prop_repeats').reset_index()
# df_cv = df_cv.groupby(['trained','feature_number','level_4'], as_index = False).mean()

ind = df_cv.feature_number <= 3
sns.catplot(data = df_cv[ind], x = 'level_4', y = 'prop_repeats', hue='trained', 
            linewidth = 0.5, edgecolor = 'k',
            palette = 'colorblind', ci = 68, row = 'feature_number', col = 'pool_size',
            kind = 'bar')

#%% 


cond_colors = sns.color_palette('colorblind',n_colors=2)

n_colors = sns.light_palette(cond_colors[0],n_colors=10, reverse = True)
t_colors = sns.light_palette(cond_colors[1],n_colors=10, reverse = True)

df_plot = df.copy()

df_plot['pref_bin'] = pd.cut(df_plot.mean_pref_ori, np.linspace(-11.25,180-11.25,9), 
                             labels = np.ceil(np.arange(0,180,22.5)).astype(int))
df_plot['r_bin'] = pd.cut(df_plot.OSI, np.linspace(0,1,11),
                          labels = np.arange(10))
df_plot['cv_bin'] = pd.cut(df_plot.CV, np.linspace(0,2,11))

df_plot = df_plot.pivot_table(index = ['subject','trained','pref_bin','r_bin',
                                       'pool_size','feature_number'],
                              aggfunc = 'size').reset_index()

df_plot[0] = df_plot[0]/1000

df_plot = df_plot.groupby(['trained','pref_bin','r_bin','pool_size',
                           'feature_number'], as_index = False).mean()

fig = plt.figure()

c = 1
for f in range(1,4):
    for p in [50,100,200,400]:
        ax = fig.add_subplot(3,4,c, projection = '3d')
        ind = np.logical_and(df_plot.pool_size == p,df_plot.feature_number==f)
        ind_n = np.logical_and(ind,df_plot.trained==False)
        ind_t = np.logical_and(ind,df_plot.trained)
        df_s_n = df_plot[ind_n].copy()
        df_s_n = df_s_n.pivot_table(index='pref_bin',columns = 'r_bin', values=0)
        z = df_s_n.to_numpy()
        xv,yv = np.meshgrid(np.arange(10),np.arange(0,180,22.5))
        # ax.plot_surface(xv,yv,z, color = cond_colors[0], alpha = 0.5,
        #                 linewidth=0, rstride=1, cstride=1)
        ax.plot_wireframe(xv,yv,z, color = cond_colors[0],rstride=1, cstride=1,
                          linewidth=0.5)
        
        df_s_t = df_plot[ind_t].copy()
        df_s_t = df_s_t.pivot_table(index='pref_bin',columns = 'r_bin', values=0)
        z = df_s_t.to_numpy()
        # ax.plot_surface(xv,yv,z, color = cond_colors[1], alpha = 0.5,
        #                 linewidth = 0, rstride=1, cstride=1)
        ax.plot_wireframe(xv,yv,z, color = cond_colors[1],rstride=1, cstride=1,
                          linewidth=0.5)

        ax.set_zlim([0,0.2])
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        
        c += 1
        
