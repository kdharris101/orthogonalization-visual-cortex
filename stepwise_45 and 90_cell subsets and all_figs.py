# -*- coding: utf-8 -*-
#%% Initialize

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
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from seaborn import axes_style
import seaborn.objects as so
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

trained = ['Naive', 'Proficient', 'Naive', 'Proficient', 'Naive', 'Proficient', 'Naive', 'Proficient', 'Naive', 'Proficient']

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
                    'stim_decoding_results_stepwise_selection_45 and 90 only_with cell stats_set features 10 cells_from varying pool sizes_*000 repeats_*'))[-1] 
            for i in range(len(subjects))]

file_paths_all = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), 
                    'stim_decoding_results_stepwise_selection_45 and 90 only_all cells*'))[-1] 
            for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel/OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%%

# For each repeat and cell number, calculate accuracy, log loss, mean response, std response, 
# cv of response, response index, proportion of decision function (only calculated for full model)

df_sw = pd.DataFrame()
df_full = pd.DataFrame()
df_trials = pd.DataFrame()
df_features = pd.DataFrame()
df_features_all = pd.DataFrame()
df_prob_stim = pd.DataFrame()

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode_subsets = np.load(file_paths_subsets[i],allow_pickle = True)[()]
    
    pool_sizes = decode_subsets['pool_sizes']
          
    stim = decode_subsets['stim']
    
    pred_stim = decode_subsets['pred_stim']
    
    features = decode_subsets['features']
       
    num_trials, num_features, num_pools, num_repeats = pred_stim.shape
        
    accu = pred_stim == np.tile(stim.reshape(-1,1,1,1), (1,num_features,num_pools,num_repeats))
    accu = accu.sum(0)/len(stim)
    
    accu = accu.mean(-1)
    
    prob_stim = decode_subsets['prob_stim']
    prob_diff = np.squeeze(np.diff(prob_stim,axis=1))
    prob_stim = prob_stim.reshape(num_trials,2,-1)
    
    log_loss = np.zeros((num_features,num_pools,num_repeats))
    log_loss = log_loss.reshape(-1,)
    
    for l in range(prob_stim.shape[2]):
        log_loss[l] = skm.log_loss(stim,prob_stim[:,:,l])
    
    log_loss = log_loss.reshape(num_features,num_pools,num_repeats)
    log_loss = log_loss.mean(-1)
   
    # Average responses for all neurons
    trials = decode_subsets['trials_test'][:,features.flatten()]
    mu = np.array([trials[stim==s,:].mean(0) for s in np.unique(stim)])
    mu_diff = np.diff(mu,axis=0)
    sigma = np.array([trials[stim==s,:].std(0) for s in np.unique(stim)])
    resp_index = ((np.diff(mu,axis=0))/mu.sum(0)).flatten()
    mu_pref = mu[np.argmax(mu,axis=0),np.arange(mu.shape[1])]
    sigma_pref = sigma[np.argmax(mu,axis=0),np.arange(mu.shape[1])]
    cv = sigma_pref/mu_pref
    
    mu_pref_stim = np.unique(stim)[np.argmax(mu,axis=0)]
    
    mu_pref = np.nanmean(mu_pref.reshape(num_features,num_pools,num_repeats),axis=-1)
    sigma_pref = np.nanmean(sigma_pref.reshape(num_features,num_pools,num_repeats),axis=-1)
    cv = np.nanmean(cv.reshape(num_features,num_pools,num_repeats),axis=-1)
    
    resp_index = np.diff(mu,axis=0)/mu.sum(0)
    resp_index = resp_index.reshape(num_features,num_pools,num_repeats)
    resp_index = np.nanmean(np.abs(resp_index),axis=-1)
    
    trials_z = decode_subsets['trials_test_z'][:,features.flatten()]
    mu_z = np.array([trials_z[stim==s,:].mean(0) for s in np.unique(stim)])
    mu_z = mu_z.reshape(2,num_features,num_pools,num_repeats)
    
    # Model weights for all features/cells
    weights = np.zeros((num_features,num_pools,num_repeats))
    weights_expt = decode_subsets['model_weights'][-1,...]
    for p in range(num_pools):
        for r in range(num_repeats):
            weights[:,p,r] = weights_expt[p,r]
    
    # Get proportion of decision function for each cell
    df = mu_z*weights
    df = np.transpose(df,[1,0,2,3])
    
    prop_df = df / df.sum(0)
    prop_df = prop_df.mean(1).squeeze()
    prop_df = prop_df.mean(-1)
    
    accu_full = decode_subsets['pred_stim_full_pool'] == np.tile(stim.reshape(-1,1,1), (1,num_pools,num_repeats))
    accu_full = accu_full.sum(0)/len(stim)
    
    accu_full = accu_full.mean(-1)
    
    log_loss_full = np.zeros((num_pools,num_repeats))
    log_loss_full = log_loss_full.reshape(-1,)
    
    prob_stim = decode_subsets['prob_stim_full_pool']
    prob_stim = prob_stim.reshape(num_trials,2,-1)
    
    for l in range(prob_stim.shape[2]):
        log_loss_full[l] = skm.log_loss(stim,prob_stim[...,l])
        
    log_loss_full = log_loss_full.reshape(num_pools,num_repeats)
    log_loss_full = log_loss_full.mean(-1)
    
        
    print('Loading ' + subjects[i])
    decode_all = np.load(file_paths_all[i],allow_pickle = True)[()]
    
    stim = decode_all['stim']
    
    pred_stim = decode_all['pred_stim']
    
    accu_all = pred_stim == np.tile(stim.reshape(-1,1), (1,num_features))
    accu_all = accu_all.sum(0)/len(stim)
    
    log_loss_all = np.zeros(num_features)
    
    prob_stim = decode_all['prob_stim']
    prob_diff_all = np.squeeze(np.diff(prob_stim,axis=1))
        
    for l in range(prob_stim.shape[2]):
        log_loss_all[l] = skm.log_loss(stim,prob_stim[...,l])
        
        
    # Add pool size of all cells to arrays
    accu = np.hstack([accu,accu_all.reshape(-1,1)])
    log_loss = np.hstack([log_loss,log_loss_all.reshape(-1,1)])
    pool_sizes = np.insert(np.array(pool_sizes,dtype='object').reshape(1,-1),4,'all')
    # Just add zeros for these so the arrays are all the same shape
    new_arrs =  [np.hstack([r,np.zeros_like(accu_all.reshape(-1,1))]) 
                 for r in (prop_df,mu_pref,sigma_pref,cv,resp_index)]    
    prop_df, mu_pref, sigma_pref, cv, resp_index = new_arrs
    

    df_sw = pd.concat([df_sw,pd.DataFrame({'accuracy' : accu.flatten()*100,
                                           'log_loss' : log_loss.flatten(),
                                           'prop_df' : prop_df.flatten(),
                                           'mu' : mu_pref.flatten(),
                                           'sigma' : sigma_pref.flatten(),
                                           'cv' : cv.flatten(),
                                           'resp_index' : resp_index.flatten(),
                                           'subject' : np.repeat(subjects[i],resp_index.size),
                                           'trained' : np.repeat(trained[i],resp_index.size),
                                           'pool_size' : np.tile(pool_sizes,(num_features,1)).flatten(),
                                           'cell_num' : np.tile(np.arange(num_features).reshape(-1,1),(1,len(pool_sizes))).flatten()+1})],
                      ignore_index=True)
     

    df_full = pd.concat([df_full,pd.DataFrame({'accuracy' : accu_full.flatten()*100,
                                               'log_loss' : log_loss_full.flatten(),
                                               'subject' : np.repeat(subjects[i],accu_full.size),
                                               'trained' : np.repeat(trained[i],accu_full.size),
                                               'pool_size' : np.array(decode_subsets['pool_sizes']).flatten()})], 
                        ignore_index=True)
        

    cell_labels = np.repeat(np.arange(decode_subsets['mean_pref'].shape[0]).reshape(1,-1), len(decode_subsets['trials_test']), axis = 0)
    subject_labels = np.tile(subjects[i], decode_subsets['trials_test'].shape)
    cond_labels = np.tile(trained[i], decode_subsets['trials_test'].shape)
    stim = np.repeat(decode_subsets['stim'].reshape(-1,1), cell_labels.shape[1], axis = 1)

    df_trials = pd.concat([df_trials,pd.DataFrame({'trials_test' : decode_subsets['trials_test'].flatten(),
                                                   'stim_ori_test' : stim.flatten(),
                                                   'cell' : cell_labels.flatten(),
                                                   'subject' : subject_labels.flatten(),
                                                   'trained' : cond_labels.flatten()})], 
                          ignore_index=True)
    
    subject_labels = np.tile(subjects[i], (num_features, num_pools, num_repeats))
    cond_labels = np.tile(trained[i], (num_features, num_pools, num_repeats))
    feat_labels = np.tile(np.arange(num_features).reshape(-1,1,1), (1,num_pools,num_repeats))
    pool_labels = np.tile(np.arange(num_pools).reshape(1,-1,1), (num_features,1,num_repeats))
    repeat_labels = np.tile(np.arange(num_repeats).reshape(1,1,-1), (num_features, num_pools, 1))
    
    df_features = pd.concat([df_features,pd.DataFrame({'feature' : decode_subsets['features'].flatten(),
                                                       'subject' : subject_labels.flatten(),
                                                       'trained' : cond_labels.flatten(),
                                                       'feature_num' : feat_labels.flatten(),
                                                       'pool' : pool_labels.flatten(),
                                                       'repeat_num' : repeat_labels.flatten(),
                                                       'pref_stim' : mu_pref_stim.flatten()})], 
                            ignore_index=True)
    
    
    trials = decode_subsets['trials_test'][:,decode_all['features'].flatten()]
    mu = np.array([trials[decode_all['stim']==s,:].mean(0) for s in np.unique(decode_all['stim'])])    
    mu_pref_stim = np.unique(stim)[np.argmax(mu,axis=0)]
    
    
    df_features_all = pd.concat([df_features_all, pd.DataFrame({'feature' : decode_all['features'],
                                                                'subject' : np.repeat(subjects[i],decode_all['n_features']),
                                                                'trained' : np.repeat(trained[i],decode_all['n_features']),
                                                                'feature_num' : np.arange(decode_all['n_features']),
                                                                'pref_stim' : mu_pref_stim})],
                                ignore_index=True)
    
    
    subject_labels = np.tile(subjects[i], (num_trials,num_features, num_pools, num_repeats))
    cond_labels = np.tile(trained[i], (num_trials,num_features, num_pools, num_repeats))
    feat_labels = np.tile(np.arange(num_features).reshape(1,-1,1,1), (num_trials,1,num_pools,num_repeats))
    pool_labels = np.tile(np.arange(num_pools).reshape(1,1,-1,1), (num_trials, num_features,1,num_repeats))
    repeat_labels = np.tile(np.arange(num_repeats).reshape(1,1,1,-1), (num_trials, num_features, num_pools, 1))
    
    df_prob_stim = pd.concat([df_prob_stim,pd.DataFrame({'prob_diff' : prob_diff.flatten(),
                                                         'subject' : subject_labels.flatten(),
                                                         'pool' : pool_labels.flatten(),
                                                         'num_features' : feat_labels.flatten(),
                                                         'repeat_num' : repeat_labels.flatten(),
                                                         'trained' : cond_labels.flatten()})],
                             ignore_index=True)
    
    subject_labels = np.tile(subjects[i], (num_trials, num_features))
    cond_labels = np.tile(trained[i], (num_trials,num_features))
    feat_labels = np.tile(np.arange(num_features).reshape(1,-1), (num_trials,1))
    
    df_prob_stim = pd.concat([df_prob_stim,pd.DataFrame({'prob_diff' : prob_diff_all.flatten(),
                                                         'subject' : subject_labels.flatten(),
                                                         'num_features' : feat_labels.flatten(),
                                                         'pool' : np.repeat(['all'], prob_diff_all.size),
                                                         'trained' : cond_labels.flatten()})],
                             ignore_index=True)
    
    

#%% Plot distributions of difference in stim probability

df_plot = df_prob_stim.copy()
df_plot['prob_diff'] = df_plot.prob_diff.abs()

sns.displot(data = df_plot[df_plot.num_features==0], x = 'prob_diff', kind = 'hist', col = 'pool', col_order = ['all',3,2,1,0],
            row = 'trained', stat = 'probability')





#%% Group all trials by pool and trained condition, make sure all responses to preferred orientation are grouped together

df_resps = pd.DataFrame()
pools = np.hstack([np.array(['all'],dtype=object),np.arange(num_pools)])

for s in np.unique(subjects):

    for t in np.unique(trained):

        for pi,p in enumerate(pools):
            
            if p == 'all':
                feature = df_features_all[(df_features_all.trained==t) & (df_features_all.subject == s)
                                            & (df_features_all.feature_num==0)].feature.to_numpy()[0]
                pref_stim = df_features_all[(df_features_all.trained==t) & (df_features_all.subject == s)
                                            & (df_features_all.feature_num==0)].pref_stim.to_numpy()[0]
                
                trials = df_trials[(df_trials.trained==t) & (df_trials.subject == s) & (df_trials.cell==feature)].trials_test.to_numpy()
                stim = df_trials[(df_trials.trained==t) & (df_trials.subject == s) & (df_trials.cell==feature)].stim_ori_test.to_numpy()
                
                pref_on_trial = stim == pref_stim

            else:
                features = df_features[(df_features.trained==t) & (df_features.subject == s)
                                            & (df_features.feature_num==0) & (df_features.pool == p)].feature.to_numpy()
                pref_stim = df_features[(df_features.trained==t) & (df_features.subject == s)
                                            & (df_features.feature_num==0) & (df_features.pool == p)].pref_stim.to_numpy()
                
                trials = df_trials[(df_trials.trained==t) & (df_trials.subject == s)].trials_test.to_numpy()
                trials = trials.reshape(-1,df_trials[(df_trials.trained==t) & (df_trials.subject == s)].cell.max()+1)
                trials = trials[:,features.flatten()]
            
                stim = df_trials[(df_trials.trained==t) & (df_trials.subject == s)].stim_ori_test.to_numpy()
                stim = stim.reshape(-1,df_trials[(df_trials.trained==t) & (df_trials.subject == s)].cell.max()+1)
                stim = stim[:,features.flatten()]

                pref_on_trial = stim == pref_stim
                
            
            df_resps = pd.concat([df_resps, pd.DataFrame({'pref_on_trial' : pref_on_trial.flatten(),
                                                          'trials' : trials.flatten(),
                                                          'trained' : np.repeat(t,trials.size),
                                                          'pool' : np.repeat(p,trials.size)})],
                                 ignore_index = True)
            

df_resps['pool'] = df_resps.pool.astype(str)
df_resps['pref_on_trial'] = df_resps.pref_on_trial.astype(str)

#%% Plot distributions for each pool size

style = {**axes_style('ticks'),
         'font.size':5,'axes.titlesize':5,
         'axes.labelsize':5,
         'axes.linewidth':0.5,
         'xtick.labelsize':5,
         'ytick.labelsize':5,
         'xtick.major.width':0.5,
         'ytick.major.width':0.5,
         'xtick.major.size':3,
         'ytick.major.size':3,
         'patch.linewidth':0.5,
         'lines.markersize':2,
         'lines.linewidth':0.5,
         'legend.fontsize':5,
         'legend.title_fontsize':5,
         'axes.spines.left': True,
         'axes.spines.bottom': True,
         'axes.spines.right': False,
         'axes.spines.top': False,
         'font.sans-serif' : 'Helvetica',
         'font.family' : 'sans-serif'}

fig_size = (2,1)


# (
#     so.Plot(df_resps, x = 'trials')
#     .layout(size = fig_size, engine = 'tight')
#     .facet(col = 'pool', row = 'trained',order={"col": ['all','3','2','1','0']})
#     .add(so.Bars(), so.Hist(stat = 'probability', common_norm = False, bins = np.linspace(0,3,100)),
#         legend = False, color = 'pref_on_trial')
#     .label(y = '', x = '', title = '')
#     .theme(style)
#     .share(x = True, y = True)
#     #.limit(x = (0,3),y = (0,0.1))
# )

f,a = plt.subplots(1,5)

for t in df_resps.trained.unique():
    for ip, p in enumerate(['all','3','2','1','0']):
        ind = (df_resps.trained==t) & (df_resps.pool == p)
        if t == 'Naive':
            fill = False
        else:
            fill = True
            
        sns.histplot(data = df_resps[ind], x = 'trials', stat = 'probability', kde = True,
                     hue = 'pref_on_trial', bins = np.linspace(0,3,100), fill = fill,
                     ax = a.flatten()[ip], legend = False)
        
        a.flatten()[ip].set_ylim([0,0.5])
        sns.despine(ax=a.flatten()[ip])
            



# sns.displot(data = df_resps, x = 'trials', kind = 'hist', col = 'pool', row = 'trained', col_order = ['all','3','2','1','0'],
#             hue = 'pref_on_trial', stat = 'probability', common_norm = False, bins = np.linspace(0,3,100))



#%% Look at response distributions for "best" cells for different pool sizes

# Pick random best cells for each pool for a mouse

i_subject = 'SF180613'

df_resps = pd.DataFrame()
pools = np.hstack([np.array(['all'],dtype=object),np.arange(num_pools)])

for t in np.unique(trained):

    for pi,p in enumerate(pools):
        
        if p == 'all':
            feature = df_features_all[(df_features_all.subject == i_subject) & (df_features_all.trained==t) 
                                      & (df_features_all.feature_num==0)].feature.to_numpy()[0]

        else:
            repeat_num = np.random.choice(np.arange(num_repeats),1)[0]
            feature = df_features[(df_features.subject == i_subject) & (df_features.trained==t) 
                                        & (df_features.feature_num==0) & (df_features.repeat_num==repeat_num) 
                                        & (df_features.pool == p)].feature.to_numpy()[0]


        df = df_trials[(df_trials.subject==i_subject) & (df_trials.trained == t) & (df_trials.cell == feature)].copy()
        
        df.loc[:,'pool'] = p
        

        df_resps = pd.concat([df_resps, df], ignore_index=True)

df_resps['trained'] = df_resps['trained'].astype('str')



style = {**axes_style('ticks'),
         'font.size':5,'axes.titlesize':5,
         'axes.labelsize':5,
         'axes.linewidth':0.5,
         'xtick.labelsize':5,
         'ytick.labelsize':5,
         'xtick.major.width':0.5,
         'ytick.major.width':0.5,
         'xtick.major.size':3,
         'ytick.major.size':3,
         'patch.linewidth':0.5,
         'lines.markersize':2,
         'lines.linewidth':0.5,
         'legend.fontsize':5,
         'legend.title_fontsize':5,
         'axes.spines.left': True,
         'axes.spines.bottom': True,
         'axes.spines.right': False,
         'axes.spines.top': False,
         'font.sans-serif' : 'Helvetica',
         'font.family' : 'sans-serif'}

fig_size = (5,1.4)



(
    so.Plot(df_resps, x = 'trials_test')
    .layout(size = fig_size, engine = 'tight')
    .facet(col = 'pool')
    .add(so.Bars(), so.Hist(stat = 'probability', common_norm = False),
        legend = False, color = 'trained')
    .label(y = 'Proportion of trials', x = 'Response')
    .theme(style)
)

#%% Look at response distributions for "best" cells for different pool sizes

# Pick random best cells for each pool for a mouse

i_subject = 'SF180613'

df_resps = pd.DataFrame()
pools = np.hstack([np.array(['all'],dtype=object),np.arange(num_pools)])

for t in np.unique(trained):

    for pi,p in enumerate(pools):
        
        if p == 'all':
            feature = df_features_all[(df_features_all.subject == i_subject) & (df_features_all.trained==t) 
                                      & (df_features_all.feature_num==0)].feature.to_numpy()[0]

        else:
            repeat_num = np.random.choice(np.arange(num_repeats),1)[0]
            feature = df_features[(df_features.subject == i_subject) & (df_features.trained==t) 
                                        & (df_features.feature_num==0) & (df_features.repeat_num==repeat_num) 
                                        & (df_features.pool == p)].feature.to_numpy()[0]


        df = df_trials[(df_trials.subject==i_subject) & (df_trials.trained == t) & (df_trials.cell == feature)].copy()
        
        df.loc[:,'pool'] = p
        

        df_resps = pd.concat([df_resps, df], ignore_index=True)

df_resps['trained'] = df_resps['trained'].astype('str')


style = {**axes_style('ticks'),
         'font.size':5,'axes.titlesize':5,
         'axes.labelsize':5,
         'axes.linewidth':0.5,
         'xtick.labelsize':5,
         'ytick.labelsize':5,
         'xtick.major.width':0.5,
         'ytick.major.width':0.5,
         'xtick.major.size':3,
         'ytick.major.size':3,
         'patch.linewidth':0.5,
         'lines.markersize':2,
         'lines.linewidth':0.5,
         'legend.fontsize':5,
         'legend.title_fontsize':5,
         'axes.spines.left': True,
         'axes.spines.bottom': True,
         'axes.spines.right': False,
         'axes.spines.top': False,
         'font.sans-serif' : 'Helvetica',
         'font.family' : 'sans-serif'}

fig_size = (5,1.4)



(
    so.Plot(df_resps, x = 'trials_test')
    .layout(size = fig_size, engine = 'tight')
    .facet(col = 'pool')
    .add(so.Bars(), so.Hist(stat = 'probability', common_norm = False),
        legend = False, color = 'trained')
    .label(y = 'Proportion of trials', x = 'Response')
    .theme(style)
)


#%% Find significant differences for feature metrics

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def make_sig_boxes(p,pad,x_range,y,h):
    
    mask = p < 0.05
    mask_diff = np.diff(mask,axis=0,prepend=False,append=False)
    x_ind = np.where(mask_diff==True)[0]
    
    if len(x_ind) == 0:
        return None
    
    borders = np.array(list(zip(x_ind[::2], x_ind[1::2]-1)))
    borders = x_range[borders].astype(float)
    borders[:,0] = borders[:,0] - pad
    borders[:,1] = borders[:,1] + pad
    
    boxes = [Rectangle((b[0],y),width = np.diff(b), height = h) for b in borders]
    
    return boxes
    
    

to_test = ['accuracy','log_loss','prop_df','resp_index','mu','sigma','cv']
pools = df_sw.pool_size.unique()[3::-1]

pvals = np.zeros((10,len(pools),len(to_test)))


for it,t in enumerate(to_test):
    for ip,p in enumerate(pools):
        for ic,c in enumerate(df_sw.cell_num.unique()):
            pvals[ic,ip,it] = ttest_rel(df_sw.loc[(df_sw.trained=='Naive')&(df_sw.cell_num==c)&(df_sw.pool_size==p),t],
                                        df_sw.loc[(df_sw.trained=='Proficient')&(df_sw.cell_num==c)&(df_sw.pool_size==p),t])[1]
        








#%% Plot accuracy, log loss, prop of dec fun, mean, std, cv as function of cell num


y = ['accuracy','log_loss','prop_df','resp_index','mu','sigma','cv']
ylabel = ['Accuracy (% correct)','Log loss', 'Prop. of dec. fun.','Resp. index',
          'Mean resp', 'Std of resp', 'Coef. of var.']
ylims = [(50,100),(0,0.34),(0,0.6),(0.3,1),(0.38,1),(0.2,0.7),(0.2,1.5)]

yticks = [np.linspace(y[0],y[1],3) for y in ylims]


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":2,
                                        "ytick.major.size":2,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":4,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    # Exclude all cell pool
    pools = df_sw.pool_size.unique()[3::-1]

    f,a = plt.subplots(7,len(pools), sharex = False, sharey = False,
                       figsize = (4.75,5.5))
    
    
    for i,p in enumerate(pools):
        
        for r in range(len(y)):
        
            sns.lineplot(data = df_sw[df_sw.pool_size==p], x = 'cell_num', y = y[r],
                         errorbar = ('se',1), hue = 'trained', ax = a[r,i],
                         palette = 'colorblind', legend = False)
            a[r,i].set_ylim(ylims[r])
            a[r,i].set_yticks(yticks[r])
            a[r,i].set_ylabel(ylabel[r])
            
            # if r < 0:
            #     a[r,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if r == 0:
                a[r,i].set_title('Neuron pool size - ' + str(int(p)))
        
            if i == 0:
                if r < 6:
                    a[r,i].set_xticks([])
                    a[r,i].set_xlabel('')
                    sns.despine(ax=a[r,i],bottom=True,offset=3,trim=True)
                    
                else:
                    a[r,i].set_xticks([1,10])
                    a[r,i].set_xlabel('Number of cells')
                    sns.despine(ax=a[r,i],offset=3,trim=True)
                    
            if i > 0:
                if r < 6:
                    a[r,i].set_xticks([])
                    a[r,i].set_yticks([])
                    a[r,i].set_xlabel('')
                    a[r,i].set_ylabel('')
                    sns.despine(ax=a[r,i],bottom=True,left=True,offset=3,trim=True)
                   
                else:
                    a[r,i].set_xticks([1,10])
                    a[r,i].set_xlabel('Number of cells')
                    a[r,i].set_yticks([])
                    a[r,i].set_ylabel('')
                    sns.despine(ax=a[r,i],left=True,offset=3,trim=True)

            # Add significance boxes
            # boxes = make_sig_boxes(pvals[:,i,r],0.05,np.arange(1,11),(ylims[r])[0],ylims[r][1])
            
            # if boxes is not None:
            #     pc = PatchCollection(boxes, facecolor='grey', alpha=0.2,
            #                 edgecolor=None)
                
            #     a[r,i].add_collection(pc)    
            
            # Add signifiance stars
            
            inds = np.arange(1,11)[pvals[:,i,r]<0.05]
            
            for pt in inds:
                
                a[r,i].text(pt,ylims[r][1]*0.9, '*',{'horizontalalignment' : 'center'})

    
    f.tight_layout()



#%% Plot accuracy only with accuracy of full pool and for full pool size (all cells recorded)

y = ['accuracy']
ylabel = ['Accuracy (% correct)']
ylims = [(50,105)]

yticks = [np.arange(50,110,10)]

cond_colors = sns.color_palette('colorblind',2)
cond_colors_m = {False : cond_colors[0],
                 True : cond_colors[1]}

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":2,
                                        "ytick.major.size":2,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":2,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"
    
    pools = df_sw.pool_size.unique()[::-1]


    f,a = plt.subplots(len(y), len(pools), sharex = False, sharey = False,
                       figsize = (4.75,1.1))
    
    a = a.reshape(1,-1)
    
    
    for i,p in enumerate(pools):
        
        for r in range(len(y)):
        
            sns.lineplot(data = df_sw[df_sw.pool_size==p], x = 'cell_num', y = y[r],
                         errorbar = ('se',1), hue = 'trained', ax = a[r,i],
                         palette = 'colorblind', legend = False)
            sns.scatterplot(data = df_sw[df_sw.pool_size==p], x = 'cell_num', y = y[r],
                            palette = 'colorblind',
                            ax = a[r,i],
                            ec = df_sw.loc[df_sw.pool_size==p,'trained'].map(cond_colors_m), 
                            fc = 'none',
                            zorder = 1,
                            linewidth = 0.25,
                            legend = False)
            a[r,i].set_ylim(ylims[r])
            a[r,i].set_yticks(yticks[r])
            a[r,i].set_ylabel(ylabel[r])
            a[r,i].plot([1,10],[100,100],'--k', zorder = 0)
            a[r,i].set_xlim([0.5,12])
            a[r,i].set_xticks([1,10])
            
            
            if i > 0:
                ind_n = (df_full.trained==False) & (df_full.pool_size == p)
                ind_t = (df_full.trained==True) & (df_full.pool_size == p)
                a[r,i].scatter(11,df_full[ind_n].accuracy.mean(), edgecolor = cond_colors[0], marker = 'o',
                              facecolor = 'none', zorder = 1, linewidths = 0.25)
                a[r,i].scatter(11.5,df_full[ind_t].accuracy.mean(), edgecolor = cond_colors[1], marker = 'o',
                              facecolor = 'none', zorder = 1, linewidths = 0.25)
                a[r,i].errorbar(11,df_full[ind_n].accuracy.mean(),
                                  df_full[ind_n].accuracy.std()/np.sqrt(5),
                                  capsize = 0.5, color = 'k', zorder = 0,
                                  linewidth = 0.5)
                a[r,i].errorbar(11.5,df_full[ind_t].accuracy.mean(),
                                     df_full[ind_t].accuracy.std()/np.sqrt(5),
                                     capsize = 0.5, color = 'k', zorder = 0,
                                     linewidth = 0.5)
            
                a[r,i].text(11.25,88,'all',horizontalalignment='center')
            
            # if r < 0:
            #     a[r,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if r == 0:
                # a[r,i].set_title('Neuron pool size - ' + str(p), pad = 0)
                a[r,i].text(5.5,105, 'Neuron pool size - ' + str(p), horizontalalignment = 'center')

            a[r,i].set_xticks([1,10])
            a[r,i].set_xlabel('Number of cells')
            sns.despine(ax=a[r,i],trim=True)
                    
            if i > 0:
                    a[r,i].set_yticks([])
                    a[r,i].set_ylabel('')
                    sns.despine(ax=a[r,i],left=True,trim=True)

                
    
    f.tight_layout()

#%% Line and scatter plots of accuracy only

cond_colors = sns.color_palette('colorblind',2)
cond_colors_m = {False : cond_colors[0],
                 True : cond_colors[1]}


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

