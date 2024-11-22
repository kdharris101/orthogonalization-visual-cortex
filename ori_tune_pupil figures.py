#%%
# -*- coding: utf-8 -*-


"""
Created on Thu Oct  1 13:35:14 2020

Neural and pupil trial responses

@author: Samuel Failor
"""

from os.path import join
import glob
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import mannwhitneyu, levene, ranksums
from scipy.stats import gmean
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.weightstats import ttest_ind

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'


# results_dir = r'H:/OneDrive for Business/Results'
results_dir = r'C:/Users/Samuel/OneDrive - University College London/Results'
# results_dir = '/mnt/c/Users/Samuel/OneDrive - University College London/Results'


subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']
subjects = ['SF170620B', 'SF170620B', 'SF170905B', 'SF171107',
            'SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613']
expt_dates = ['2017-07-04', '2017-12-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 1, 7, 3, 1, 3, 1, 5]

trained = [False, True, True, False, True, False, True, False, True]

# subjects_file = ['M170620B_SF', 'SF170620B',
#             'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
#             'SF180613']
# subjects = ['SF170620B', 'SF170620B', 'SF171107',
#             'SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613']
# expt_dates = ['2017-07-04', '2017-12-21', '2017-11-17', '2018-04-04', 
#               '2018-06-13', '2018-09-21','2018-06-28', '2018-12-12']

# expt_nums = [5, 7, 7, 3, 1, 3, 1, 5]

# trained = [False, True, False, True, False, True, False, True]

file_paths = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), r'_'.join([subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), 'orientation tuning_norm by pref ori_[2020]*'])))[-1] 
                                            for i in range(len(subjects_file))]
file_paths_pupil = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), r'_'.join([subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), 'orientation tuning_pupil and face resps_[2020]*'])))[-1] 
                                            for i in range(len(subjects_file))]
file_paths_convexity = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), r'_'.join([subjects_file[i],
                    'trial_convexity.npy'])))[-1] for i in range(1,len(subjects_file),2)]


single_cell_metrics = ['cell_plane','pref_ori_all','r','th','v_x', 
                       'v_y','r_all','th_all','v_x_all','v_y_all']


#%%

df_scm = pd.DataFrame(columns = single_cell_metrics 
                      + ['subject'] + ['trained'])

peri_pupil = np.zeros(len(subjects), dtype = object)
peri_dpupil = np.copy(peri_pupil)
peri_move = np.copy(peri_pupil)
peri_dmove = np.copy(peri_pupil)
peri_svd = np.copy(peri_pupil)
peri_dsvd = np.copy(peri_pupil)
stim_ori_all = np.copy(peri_pupil)
condition_all = np.copy(peri_pupil)
subjects_all = np.copy(peri_pupil)

for i,(f,fp) in enumerate(zip(file_paths,file_paths_pupil)):
    
    print('Loading ' + f)
    expt = np.load(f, allow_pickle = True)[()]
         
    g_cells = np.logical_and(expt['cell_plane'] > 0, expt['V1_ROIs'] == 1)
    
    # train_ind = np.repeat(
    #             np.isin(np.unique(trial_num),expt['train_ind']).reshape(-1,1),
    #                          expt['trial_resps'][:,g_cells].shape[1], axis = 1)
    
    test_ind = expt['test_ind']
    
    stim_ori = expt['stim_ori'][test_ind]
    blank_ind = np.isinf(stim_ori)
    stim_ori = stim_ori[~blank_ind]
    
    trials = expt['trial_resps'][:,g_cells]
    trials = trials[test_ind,:]
    trials = trials[~blank_ind,:]
    
    stim_ori = np.repeat(stim_ori.reshape(-1,1), trials.shape[1], axis = 1)
    
    stim_dir = expt['stim_dir'][test_ind]
    stim_dir = stim_dir[~blank_ind]
    stim_dir = np.repeat(stim_dir.reshape(-1,1), trials.shape[1], axis = 1)
    
    trial_num = np.repeat(np.arange(trials.shape[0]).reshape(-1,1),trials.shape[1], axis = 1)
    
    tc = np.load(file_paths_convexity[i], allow_pickle=True)[:,0]
    tc = np.repeat(tc.reshape(-1,1), g_cells.sum(), axis = 1)
    
    if i == 0:
        trials_dict = {'trial_resps' : trials.flatten(),
                       'stim_ori' : stim_ori.flatten(),
                       'stim_dir' : stim_dir.flatten(),
                       'subject' : np.repeat(subjects[i], trials.size),
                       'trained' : np.repeat(trained[i],trials.size),
                       'cell' : np.repeat(np.arange(g_cells.sum()).reshape(1,-1),trials.shape[0]),
                       'trial_num' : trial_num.flatten(),
                       'r' : np.repeat(expt['r_all'][g_cells].reshape(1,-1),
                                       stim_ori.shape[0],axis = 0).flatten(),
                       'pref' : np.repeat(expt['th_all'][g_cells].reshape(1,-1),
                                          stim_ori.shape[0],axis = 0).flatten(),
                    #    'train_ind' : train_ind.flatten(),
                       'tc' : tc.flatten()}
    else:
        cell_nums = np.arange(trials_dict['cell'].max()+1,
                              trials_dict['cell'].max()+g_cells.sum()+1)
        cell_nums = np.repeat(cell_nums.reshape(1,-1),
                              expt['trial_resps'].shape[0], axis = 0)
        trials_dict['trial_resps'] = np.concatenate(
            (trials_dict['trial_resps'], trials.flatten()))
        trials_dict['subject'] = np.concatenate((trials_dict['subject'],
                                 np.repeat(subjects[i], 
                                 trials.size)))
        trials_dict['trained'] = np.concatenate((trials_dict['trained'],
                                 np.repeat(trained[i], 
                                 trials.size)))
        trials_dict['stim_ori'] = np.concatenate((trials_dict['stim_ori'],stim_ori.flatten()))
        trials_dict['stim_dir'] = np.concatenate((trials_dict['stim_dir'],stim_dir.flatten()))
        trials_dict['cell'] = np.concatenate((trials_dict['cell'],cell_nums.flatten()))
        trials_dict['trial_num'] = np.concatenate((trials_dict['trial_num'],
                                                   trial_num.flatten()))
        trials_dict['r'] = np.concatenate((trials_dict['r'],np.repeat(
            expt['r_all'][g_cells].reshape(1,-1),stim_ori.shape[0],axis=0).flatten()))
        trials_dict['pref'] = np.concatenate((trials_dict['pref'],np.repeat(
            expt['th_all'][g_cells].reshape(1,-1),stim_ori.shape[0],axis=0).flatten()))
        # trials_dict['train_ind'] = np.concatenate((trials_dict['train_ind'],
        #                                            train_ind.flatten()))
        trials_dict['tc'] = np.concatenate([trials_dict['tc'],tc.flatten()])                                     
                                                
    # Save single cell metrics and append to dataframe
    single_cell_values = {s : expt[s][g_cells,None] for s in single_cell_metrics}
    
    for s in single_cell_metrics:
        if type(single_cell_values[s]) == np.ndarray:
            single_cell_values[s] = single_cell_values[s].reshape(-1,)
        else:
            single_cell_values[s] = np.array(single_cell_values[s]).reshape(-1,)
    
    single_cell_values['subject'] = np.repeat(subjects[i], 
                                    len(expt['cell_plane'][g_cells]))
    single_cell_values['trained'] = np.repeat(trained[i], 
                                    len(expt['cell_plane'][g_cells]))
       
    df_scm = df_scm.append(pd.DataFrame(single_cell_values),
                           ignore_index = True)
    
    # Load pupil trials
    
    print('Loading ' + fp)
    exptp = np.load(fp, allow_pickle = True)[()]
           
    if i == 0:
        trials_dict_pupil = {'dpupil' : exptp['dpupil'],
                             'pupil' : exptp['post_pupil'],
                             'post_pupil' : exptp['post_trial_pupil'],
                             'post_dpupil' : exptp['post_trial_dpupil'],
                             'pre_pupil' : exptp['pre_pupil'],
                             'dsvd' : exptp['dsvd'],
                             'svd' : exptp['post_svd'],
                             'post_svd' : exptp['post_trial_svd'],
                             'post_dsvd' : exptp['post_trial_dsvd'],
                             'pre_svd' : exptp['pre_svd'],
                             'dmotion' : exptp['dmotion'],
                             'motion' : exptp['post_motion'],
                             'post_motion' : exptp['post_trial_motion'],
                             'pre_motion' : exptp['pre_motion'],
                             'subject' : np.repeat(subjects[i], len(exptp['dpupil'])),
                             'trained' : np.repeat(trained[i], len(exptp['dpupil'])),
                             'stim_ori' : expt['stim_ori'],
                             'stim_dir' : expt['stim_dir']}
    else:
        trials_dict_pupil['dpupil'] = np.concatenate([trials_dict_pupil['dpupil'],
                                                    exptp['dpupil']])
        trials_dict_pupil['pupil'] = np.concatenate([trials_dict_pupil['pupil'],
                                                    exptp['post_pupil']])
        trials_dict_pupil['post_pupil'] = np.concatenate([trials_dict_pupil['post_pupil'],
                                                    exptp['post_trial_pupil']])
        trials_dict_pupil['post_dpupil'] = np.concatenate([trials_dict_pupil['post_dpupil'],
                                                    exptp['post_trial_dpupil']])
        trials_dict_pupil['pre_pupil'] = np.concatenate([trials_dict_pupil['pre_pupil'],
                                                    exptp['pre_pupil']])
        trials_dict_pupil['dsvd'] = np.concatenate([trials_dict_pupil['dsvd'],
                                                    exptp['dsvd']], axis = 0)
        trials_dict_pupil['svd'] = np.concatenate([trials_dict_pupil['svd'],
                                                    exptp['post_svd']], axis = 0)
        trials_dict_pupil['post_svd'] = np.concatenate([trials_dict_pupil['post_svd'],
                                                    exptp['post_trial_svd']], axis = 0)
        trials_dict_pupil['post_dsvd'] = np.concatenate([trials_dict_pupil['post_dsvd'],
                                                    exptp['post_trial_dsvd']], axis = 0)
        trials_dict_pupil['pre_svd'] = np.concatenate([trials_dict_pupil['pre_svd'],
                                                    exptp['pre_svd']], axis = 0)    
        
        trials_dict_pupil['dmotion'] = np.concatenate([trials_dict_pupil['dmotion'],
                                                    exptp['dmotion']], axis = 0)
        trials_dict_pupil['motion'] = np.concatenate([trials_dict_pupil['motion'],
                                                    exptp['post_motion']], axis = 0)
        trials_dict_pupil['post_motion'] = np.concatenate([trials_dict_pupil['post_motion'],
                                                    exptp['post_trial_motion']], axis = 0)
        trials_dict_pupil['pre_motion'] = np.concatenate([trials_dict_pupil['pre_motion'],
                                                    exptp['pre_motion']], axis = 0)
        
        
        
        trials_dict_pupil['subject'] = np.concatenate([trials_dict_pupil['subject'],
                                np.repeat(subjects[i], len(exptp['dpupil']))])                                            
        trials_dict_pupil['trained'] = np.concatenate([trials_dict_pupil['trained'],
                                np.repeat(trained[i], len(exptp['dpupil']))])
        trials_dict_pupil['stim_ori'] = np.concatenate([trials_dict_pupil['stim_ori'],
                                                    expt['stim_ori']])
        trials_dict_pupil['stim_dir'] = np.concatenate([trials_dict_pupil['stim_dir'],
                                                    expt['stim_dir']])                                                
       
        # # Pupil over time    
        # if i == 0:
        #     trials_dict_pupil_ot = {'dpupil' : exptp['dpupil'],
        #                             'pupil' : exptp['post_pupil'],
        #                             'post_pupil' : exptp['post_trial_pupil'],
        #                             'post_dpupil' : exptp['post_trial_dpupil'],
        #                             'pre_pupil' : exptp['pre_pupil'],
        #                             'dsvd' : exptp['dsvd'],
        #                             'svd' : exptp['post_svd'],
        #                             'post_svd' : exptp['post_trial_svd'],
        #                             'post_dsvd' : exptp['post_trial_dsvd'],
        #                             'pre_svd' : exptp['pre_svd'],
        #                             'subject' : np.repeat(subjects[i], len(exptp['dpupil'])),
        #                             'trained' : np.repeat(trained[i], len(exptp['dpupil'])),
        #                             'stim_ori' : expt['stim_ori'],
        #                             'stim_dir' : expt['stim_dir']}
        # else:
        #     trials_dict_pupil_ot['dpupil'] = np.concatenate([trials_dict_pupil_ot['dpupil'],
        #                                                 exptp['dpupil']])
        #     trials_dict_pupil_ot['pupil'] = np.concatenate([trials_dict_pupil_ot['pupil'],
        #                                                 exptp['post_pupil']])
        #     trials_dict_pupil_ot['post_pupil'] = np.concatenate([trials_dict_pupil_ot['post_pupil'],
        #                                                 exptp['post_trial_pupil']])
        #     trials_dict_pupil_ot['post_dpupil'] = np.concatenate([trials_dict_pupil_ot['post_dpupil'],
        #                                                 exptp['post_trial_dpupil']])
        #     trials_dict_pupil_ot['pre_pupil'] = np.concatenate([trials_dict_pupil_ot['pre_pupil'],
        #                                                 exptp['pre_pupil']])
        #     trials_dict_pupil_ot['dsvd'] = np.concatenate([trials_dict_pupil_ot['dsvd'],
        #                                                 exptp['dsvd']], axis = 0)
        #     trials_dict_pupil_ot['svd'] = np.concatenate([trials_dict_pupil['svd'],
        #                                                 exptp['post_svd']], axis = 0)
        #     trials_dict_pupil_ot['post_svd'] = np.concatenate([trials_dict_pupil_ot['post_svd'],
        #                                                 exptp['post_trial_svd']], axis = 0)
        #     trials_dict_pupil_ot['post_dsvd'] = np.concatenate([trials_dict_pupil_ot['post_dsvd'],
        #                                                 exptp['post_trial_dsvd']], axis = 0)
        #     trials_dict_pupil_ot['pre_svd'] = np.concatenate([trials_dict_pupil_ot['pre_svd'],
        #                                                 exptp['pre_svd']], axis = 0)
            
            
            
        #     trials_dict_pupil_ot['subject'] = np.concatenate([trials_dict_pupil_ot['subject'],
        #                             np.repeat(subjects[i], len(exptp['dpupil']))])                                            
        #     trials_dict_pupil_ot['trained'] = np.concatenate([trials_dict_pupil_ot['trained'],
        #                             np.repeat(trained[i], len(exptp['dpupil']))])
        #     trials_dict_pupil['stim_ori'] = np.concatenate([trials_dict_pupil_ot['stim_ori'],
        #                                                 expt['stim_ori']])
        #     trials_dict_pupil['stim_dir'] = np.concatenate([trials_dict_pupil_ot['stim_dir'],
        #                                                 expt['stim_dir']]) 
        
    peri_pupil[i] = exptp['peri_stim_pupil']
    peri_dpupil[i] = exptp['peri_stim_dpupil']
    peri_move[i] = exptp['peri_stim_move']
    peri_dmove[i] = exptp['peri_stim_dmove']
    peri_svd[i] = exptp['peri_stim_svd']
    peri_dsvd[i] = exptp['peri_stim_dsvd']
    stim_ori_all[i] = exptp['stim_ori']
    stim_ori_all[i][stim_ori_all[i] == np.inf] = -1
    condition_all[i] = np.repeat(trained[i],len(exptp['stim_ori']))
    subjects_all[i] = np.repeat(subjects[i],len(exptp['stim_ori']))
        
df_scm['pref_bin'] = pd.cut(df_scm.th, np.linspace(-11.25,180-11.25,9),
                        labels = np.arange(0,180,22.5))
df_scm['r_bin'] = pd.cut(df_scm.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

#%% Plot peri-stim for stim and condition - pupil


df_peri_pupil = pd.DataFrame({
    'pupil': np.concatenate(peri_dpupil,axis=0).flatten(),
    'stim' : np.repeat(np.concatenate(stim_ori_all)[:,None],151,axis=1).flatten()
    .astype(int).astype(str),
    'time' : np.repeat(np.linspace(-1,4,151)[None,:],
                       len(np.concatenate(stim_ori_all)),axis = 0).flatten(),
    'condition' : np.repeat(np.concatenate(condition_all)[:,None],151,
                            axis=1).flatten(),
    'subject' : np.repeat(np.concatenate(subjects_all)[:,None],151,
                            axis=1).flatten()})

df_peri_pupil_plot = df_peri_pupil.groupby(['stim','time','condition','subject']).mean().reset_index()
# df_all = df_peri_pupil[df_peri_pupil.stim != -1].groupby(['time','condition','subject']).mean().reset_index()
# df_all['stim'] = 'All'

# df_peri_pupil_plot = df_peri_pupil_plot.append(df_all)

# f,ax = plt.subplots(1,1)

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5, 
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":5,
                                        "ytick.major.size":5,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":4}):

    g_pupil  = sns.relplot(data = df_peri_pupil_plot[df_peri_pupil_plot.stim != -1], 
                          x = 'time', y = 'pupil', 
                          hue = 'condition', col = 'stim', kind = 'line',
                          errorbar = ('se',1), col_wrap = 4, palette = 'colorblind',
                          col_order = ['0','23','45','68','90','113','135','158'],
                          legend = True,zorder = 2)
    g_pupil._fig.set_size_inches(4.75,2)
    plt.xticks(np.arange(-1,5,1))
    plt.yticks(np.linspace(-0.2,0.2,5))
    
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection 
    def line_and_box(**kws):
        ax = plt.gca()
        ax.plot([0,0],[-0.2-0.02,0.2], '--k', zorder = 3)
        rec = Rectangle([0.5,-0.2-0.02],2.5,0.4+0.02)
        collection = PatchCollection([rec], facecolor = 'k', alpha = 0.05,
                                     zorder = 0)
        ax.add_collection(collection)
        
    g_pupil.map(line_and_box)
    
    plt.ylim([-0.2-0.02,0.2])
    
    for a,s in zip(g_pupil.axes.flatten(),
                   ['0','23','45','68','90','113','135','158']):
        a.set_title(s+r'$\degree$')
        
    g_pupil.set_axis_labels('Time from stim. (s)',
                            r'$\delta$pupil area (AU)')
    
    sns.despine(trim=True)
    
    plt.tight_layout()
    plt.savefig(
        '/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/pupil_peri.svg',
        format = 'svg')


#%% Plot peri-stim for stim and condition - movement

from math import pi, sqrt, exp

def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return np.array([1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r])

peri_move_all = np.concatenate(peri_dmove,axis=0)

# smooth with half-gaussian
# gk = gauss(n=11,sigma = 1)
# gk[6:] = 0
# gk = gk/gk.sum()

# peri_move_sm = np.zeros_like(peri_move_all)

# for i,p in enumerate(peri_move_all):
#     peri_move_sm[i,:] = np.convolve(p,gk, mode = 'same')

# peri_move_sm = np.copy(peri_move_all)

df_peri_pupil = pd.DataFrame({
    # 'movement': peri_svd_all[0,...].flatten(),
    'movement' : peri_move_all.flatten(),
    # 'movement' : peri_move_sm.flatten(),
    'stim' : np.repeat(np.concatenate(stim_ori_all)[:,None],151,axis=1).flatten()
    .astype(int).astype(str),
    'time' : np.repeat(np.linspace(-1,4,151)[None,:],
                       len(np.concatenate(stim_ori_all)),axis = 0).flatten(),
    'condition' : np.repeat(np.concatenate(condition_all)[:,None],151,
                            axis=1).flatten(),
    'subject' : np.repeat(np.concatenate(subjects_all)[:,None],151,
                            axis=1).flatten()})

df_peri_pupil = df_peri_pupil.groupby(['stim','time','condition','subject']).mean().reset_index()

# f,ax = plt.subplots(1,1)

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5, 
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":5,
                                        "ytick.major.size":5,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":4}):


    g_motion = sns.relplot(data = df_peri_pupil[df_peri_pupil.stim != -1], x = 'time', y = 'movement', 
                          hue = 'condition', col = 'stim', kind = 'line',
                          errorbar = ('se',1), col_wrap = 4, palette = 'colorblind',
                          col_order = ['0','23','45','68','90','113','135','158'],
                          legend = False, zorder = 2)
    g_motion._fig.set_size_inches(4.75,2)
    plt.xticks(np.arange(-1,5,1))
    plt.yticks(np.linspace(-0.25,0.55,5))

    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection 
    def line_and_box(**kws):
        ax = plt.gca()
        ax.plot([0,0],[-0.3,0.55], '--k', zorder = 3)
        rec = Rectangle([0,-0.3],3,0.85)
        collection = PatchCollection([rec], facecolor = 'k', alpha = 0.05,
                                     zorder = 0)
        ax.add_collection(collection)
        
    g_motion.map(line_and_box)
    plt.ylim([-0.25-0.04,0.55])

    
    for a,s in zip(g_motion.axes.flatten(),
                   ['0','23','45','68','90','113','135','158']):
        a.set_title(s+r'$\degree$')
        
    g_motion.set_axis_labels('Time from stim. (s)',
                            r'$\delta$whisking (AU)') 
    
    sns.despine(trim=True)
    plt.tight_layout()

    plt.savefig(
        '/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/whisker_peri.svg',
        format = 'svg')

#%% Plot peri-stim for stim and condition - face svd

from math import pi, sqrt, exp

def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return np.array([1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r])

peri_dsvd_all = np.concatenate(peri_dsvd,axis=1)

# smooth with half-gaussian
# gk = gauss(n=11,sigma = 1)
# gk[6:] = 0
# gk = gk/gk.sum()

# peri_move_sm = np.zeros_like(peri_move_all)

# for i,p in enumerate(peri_move_all):
#     peri_move_sm[i,:] = np.convolve(p,gk, mode = 'same')

peri_move_sm = np.copy(peri_dsvd_all)

df_peri_pupil = pd.DataFrame({
    'movement': peri_dsvd_all[0,...].flatten(),
    # 'movement' : peri_dsvd_all.flatten(),
    'stim' : np.repeat(np.concatenate(stim_ori_all)[:,None],151,axis=1).flatten()
    .astype(int).astype(str),
    'time' : np.repeat(np.linspace(-1,4,151)[None,:],
                       len(np.concatenate(stim_ori_all)),axis = 0).flatten(),
    'condition' : np.repeat(np.concatenate(condition_all)[:,None],151,
                            axis=1).flatten(),
    'subject' : np.repeat(np.concatenate(subjects_all)[:,None],151,
                            axis=1).flatten()})

df_peri_pupil = df_peri_pupil.groupby(['stim','time','condition','subject']).mean().reset_index()

# f,ax = plt.subplots(1,1)

sns.relplot(data = df_peri_pupil[df_peri_pupil.stim != -1], x = 'time', y = 'movement', 
                      hue = 'condition', col = 'stim', kind = 'line',
                      ci = 68, col_wrap = 4,
                      col_order = ['0','23','45','68','90','113','135','158'],
                      estimator = None,
                      units = 'subject')       
 
#%% Plot trials sorted by pupil and stimulus

for s,t in zip(subjects,trained):
    
    ind = np.logical_and(trials_dict['subject'] == s, 
                         trials_dict['trained'] == t)
    
    # trial responses
    trial_resps = trials_dict['trial_resps'][ind]
    cells = trials_dict['cell'][ind]
    trial_resps = trial_resps.reshape(-1,len(np.unique(cells)))
    
    # cell preferences
    r = trials_dict['r'][ind].reshape(-1,len(np.unique(cells)))[0,:]
    pref = trials_dict['pref'][ind].reshape(-1,len(np.unique(cells)))[0,:]
    
    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    trial_pupil = trials_dict_pupil['dpupil'][ind]
    
    stim = trials_dict_pupil['stim_ori'][ind]
    
    sort_t = np.lexsort((trial_pupil,stim))
    
    trial_pupil = trial_pupil[sort_t]
    trial_resps = trial_resps[sort_t,:]
    stim = stim[sort_t]
     
    pref_bin = np.digitize(pref,np.linspace(-11.25,180-11.25,9))
    
    mu_pref = np.zeros((len(np.unique(pref_bin)), trial_resps.shape[0]))
    
    for oi, o in enumerate(np.unique(pref_bin)):
        ind = pref_bin == o
        mu_pref[oi,:] = trial_resps[:,ind].mean(1)
    
    sort_c = np.lexsort((r,pref_bin))

    trial_resps = trial_resps[:,sort_c].T

    fig, axs = plt.subplots(3, sharex = True)
    axs[0].plot(np.arange(len(trial_pupil)),trial_pupil, '-k')
    axs[1].imshow(trial_resps, aspect = 'auto', cmap = 'gray_r',
                  vmin = 0, vmax = 5)
    axs[0].set_title(s + ' ' + str(t))
    axs[2].plot(mu_pref.T)
   
    
#%% Plot trials sorted by comp and stimulus

for s,t in zip(subjects,trained):
    
    ind = np.logical_and(trials_dict['subject'] == s, 
                         trials_dict['trained'] == t)
    
    # trial responses
    trial_resps = trials_dict['trial_resps'][ind]
    cells = trials_dict['cell'][ind]
    trial_resps = trial_resps.reshape(-1,len(np.unique(cells)))
    
    # cell preferences
    r = trials_dict['r'][ind].reshape(-1,len(np.unique(cells)))[0,:]
    pref = trials_dict['pref'][ind].reshape(-1,len(np.unique(cells)))[0,:]
    
    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    # trial_pupil = (trials_dict_pupil['svd'][ind,:]**2).sum(1)
    # trial_pupil = trial_pupil/np.percentile(trial_pupil,95)
    trial_pupil = trials_dict_pupil['svd'][ind,0]
    
    stim = trials_dict_pupil['stim_ori'][ind]
    
    sort_t = np.lexsort((trial_pupil,stim))
    
    trial_pupil = trial_pupil[sort_t]
    trial_resps = trial_resps[sort_t,:]
    stim = stim[sort_t]
     
    pref_bin = np.digitize(pref,np.linspace(-11.25,180-11.25,9))
    
    mu_pref = np.zeros((len(np.unique(pref_bin)), trial_resps.shape[0]))
    
    for oi, o in enumerate(np.unique(pref_bin)):
        ind = pref_bin == o
        mu_pref[oi,:] = trial_resps[:,ind].mean(1)
    
    sort_c = np.lexsort((r,pref_bin))

    trial_resps = trial_resps[:,sort_c].T

    fig, axs = plt.subplots(3, sharex = True)
    axs[0].plot(np.arange(len(trial_pupil)),trial_pupil, '-k')
    axs[1].imshow(trial_resps, aspect = 'auto', cmap = 'gray_r',
                  vmin = 0, vmax = 5)
    axs[0].set_title(s + ' ' + str(t))
    axs[2].plot(mu_pref.T)    
    
#%% Plot trials sorted by face motion and stimulus

for s,t in zip(subjects,trained):
    
    ind = np.logical_and(trials_dict['subject'] == s, 
                         trials_dict['trained'] == t)
    
    # trial responses
    trial_resps = trials_dict['trial_resps'][ind]
    cells = trials_dict['cell'][ind]
    trial_resps = trial_resps.reshape(-1,len(np.unique(cells)))
    
    # cell preferences
    r = trials_dict['r'][ind].reshape(-1,len(np.unique(cells)))[0,:]
    pref = trials_dict['pref'][ind].reshape(-1,len(np.unique(cells)))[0,:]
    
    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    # trial_pupil = (trials_dict_pupil['svd'][ind,:]**2).sum(1)
    # trial_pupil = trial_pupil/np.percentile(trial_pupil,95)
    trial_pupil = trials_dict_pupil['motion'][ind]
    trial_pupil /= trial_pupil.mean()
    
    stim = trials_dict_pupil['stim_ori'][ind]
    
    sort_t = np.lexsort((trial_pupil,stim))
    
    trial_pupil = trial_pupil[sort_t]
    trial_resps = trial_resps[sort_t,:]
    stim = stim[sort_t]
     
    pref_bin = np.digitize(pref,np.linspace(-11.25,180-11.25,9))
    
    mu_pref = np.zeros((len(np.unique(pref_bin)), trial_resps.shape[0]))
    
    for oi, o in enumerate(np.unique(pref_bin)):
        ind = pref_bin == o
        mu_pref[oi,:] = trial_resps[:,ind].mean(1)
    
    sort_c = np.lexsort((r,pref_bin))

    trial_resps = trial_resps[:,sort_c].T

    fig, axs = plt.subplots(3, sharex = True)
    axs[0].plot(np.arange(len(trial_pupil)),trial_pupil, '-k')
    axs[1].imshow(trial_resps, aspect = 'auto', cmap = 'gray_r',
                  vmin = 0, vmax = 5)
    axs[0].set_title(s + ' ' + str(t))
    axs[2].plot(mu_pref.T)    
    
#%% Plot pupil by stimulus condition

# stim_pupil = np.zeros((len(subjects),9))

# for i,(s,t) in enumerate(zip(subjects,trained)):

#     ind = np.logical_and(trials_dict_pupil['subject']==s,
#                          trials_dict_pupil['trained']==t)
#     trial_pupil = trials_dict_pupil['dpupil'][ind]
    
#     stim = trials_dict_pupil['stim_ori'][ind]
      
#     stim_pupil[i,:] = np.array([trial_pupil[stim==s].mean() 
#                                                   for s in np.unique(stim)])
    
#%% Plot average pupil by stim

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

df_pupil = pd.DataFrame({'stim' : trials_dict_pupil['stim_ori'],
                         'subject' : trials_dict_pupil['subject'],
                         'trained' : trials_dict_pupil['trained'],
                         'pupil' : trials_dict_pupil['dpupil']})

df_pupil.loc[df_pupil.stim==np.inf,'stim'] = 1
df_pupil['stim'] = df_pupil['stim'].astype(int).astype(str)
df_pupil.loc[df_pupil.stim=='1','stim'] = 'Blank'

df_pupil_plot = df_pupil.groupby(['stim','subject','trained']).mean().reset_index()

df_stim = df_pupil_plot[df_pupil_plot.stim != 'Blank']
    
model = ols('pupil ~ C(stim) + C(trained) + C(trained):C(stim)', 
                data=df_stim).fit()
anova_table = sm.stats.anova_lm(model, typ=1)

# df_all = df_pupil[df_pupil.stim != 'Blank'].groupby(
#     ['subject','trained']).mean().reset_index()
# df_all['stim'] = 'All'
# df_pupil_plot = df_pupil_plot.append(df_all).reset_index()



# model = smf.mixedlm("pupil ~ C(trained) + C(stim) + C(trained):C(stim)", 
#                     df_stim, groups=df_stim["subject"])
# mdf = model.fit()
# print(mdf.summary())

xlabel = [str(o)+r'$\degree$' for o in np.unique(df_pupil[df_pupil.stim != 'Blank'].stim.astype(int))]


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

    f,pfig = plt.subplots(1,1, figsize = (4.75,1.25))
    sns.stripplot(data = df_pupil_plot[df_pupil_plot.stim != 'Blank'], 
                  x = 'stim', y = 'pupil', 
                  dodge = 0.3, hue = 'trained', palette = 'colorblind', 
                  ax = pfig, s = 2,
                  order = ['0','23','45','68','90','113','135',
                         '158'],
                  edgecolor = 'k',
                  linewidth = 0.5, zorder = 1)
    sns.pointplot(data = df_pupil_plot[df_pupil_plot.stim != 'Blank'], x = 'stim', 
                y = 'pupil', 
                dodge = 0.4, hue = 'trained', errorbar = ('se',1), 
                join = False,
                markers = '_', legend = False, palette = 'colorblind',
                capsize = 0.15, linestyles = '--', ax = pfig,
                order = ['0','23','45','68','90','113','135',
                         '158'], 
                zorder = 2,
                errwidth = 0.5,
                scale = 1.5)
    pfig.legend_.remove()
    plt.xlabel('Stimulus orientation')
    plt.ylabel(r'$\delta$pupil area (AU)')
    pfig.set_xticklabels(xlabel)
    sns.despine(trim=True)

    
    # for t in pfig.get_xticks():
    #     if t == 8:
    #         plt.text(t,0.1,'*', horizontalalignment='center')
    #     else:
    #         plt.text(t,0.1,'N.S.', horizontalalignment = 'center')
            
    plt.tight_layout()

    plt.savefig(
        '/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/pupil_ave.svg',
        format = 'svg')
        
#%% Plot average whisker movement by stim

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

df_motion = pd.DataFrame({'stim' : trials_dict_pupil['stim_ori'],
                         'subject' : trials_dict_pupil['subject'],
                         'trained' : trials_dict_pupil['trained'],
                         'whisker' : trials_dict_pupil['dmotion']})

df_motion.loc[df_motion.stim==np.inf,'stim'] = 1
df_motion['stim'] = df_motion['stim'].astype(int).astype(str)
df_motion.loc[df_motion.stim=='1','stim'] = 'Blank'

df_motion_plot = df_motion.groupby(['stim','subject','trained']).mean().reset_index()

df_stim = df_motion_plot[df_motion_plot.stim != 'Blank']

model = ols('whisker ~ C(stim) + C(trained) + C(trained):C(stim)', 
                data=df_stim).fit()
anova_table = sm.stats.anova_lm(model, typ=1)

# df_all = df_motion[df_motion.stim != 'Blank'].groupby(
#     ['subject','trained']).mean().reset_index()
# df_all['stim'] = 'All'
# df_motion_plot = df_motion_plot.append(df_all).reset_index()



# model = smf.mixedlm("pupil ~ C(trained) + C(stim) + C(trained):C(stim)", 
#                     df_stim, groups=df_stim["subject"])
# mdf = model.fit()
# print(mdf.summary())

xlabel = [str(o)+r'$\degree$' for o in np.unique(df_motion[df_motion.stim != 'Blank'].stim.astype(int))]

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

    f,pfig = plt.subplots(1,1, figsize = (4.75,1.25))
    sns.stripplot(data = df_motion_plot[df_motion_plot.stim != 'Blank'], 
                x = 'stim', y = 'whisker', 
                dodge = 0.3, hue = 'trained', palette = 'colorblind', 
                ax = pfig, s = 2,
                order = ['0','23','45','68','90','113','135',
                         '158'],
                zorder = 1,
                edgecolor = 'k',
                linewidth = 0.5)
    sns.pointplot(data = df_motion_plot[df_motion_plot.stim != 'Blank'], x = 'stim', 
                y = 'whisker', 
                dodge = 0.4, hue = 'trained', errorbar =('se',1), join = False,
                markers = '_', legend = False, palette = 'colorblind',
                capsize = 0.15, linestyles = '--', ax = pfig,
                order = ['0','23','45','68','90','113','135',
                         '158'],
                zorder = 2,
                errwidth = 0.5,
                scale = 1.5)
    pfig.legend_.remove()
    plt.xlabel('Stimulus orientation')
    plt.ylabel(r'$\delta$whisking (AU)')
    pfig.set_xticklabels(xlabel)
    pfig.set_ylim([-0.2,0.4])
    pfig.set_yticks(np.linspace(-0.2,0.4,7))

    sns.despine(trim = True)

    
    # for t in pfig.get_xticks():
    #     if t == 8:
    #         plt.text(t,0.1,'*', horizontalalignment='center')
    #     else:
    #         plt.text(t,0.1,'N.S.', horizontalalignment = 'center')
            
    plt.tight_layout()

    plt.savefig(
        '/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/whisker_ave.svg',
        format = 'svg')
        
#%% Plot comps by stimulus condition

stim_pupil = np.zeros((len(subjects),9))

for i,(s,t) in enumerate(zip(subjects,trained)):

    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    # trial_pupil = (trials_dict_pupil['svd'][ind,:]**2).sum(1)
    # trial_pupil = trials_dict_pupil['motion'][ind]
    trial_pupil = trials_dict_pupil['dsvd'][ind,0]

    # trial_pupil /= np.percentile(trial_pupil,95)
    
    stim = trials_dict_pupil['stim_ori'][ind]
      
    stim_pupil[i,:] = np.array([trial_pupil[stim==s].mean() 
                                                  for s in np.unique(stim)])
    
#%% Plot 

stim_label = np.repeat(np.unique(stim).reshape(1,-1),
                       len(subjects), axis = 0)
trained_label = np.repeat(np.array(trained).reshape(-1,1), 
                          len(np.unique(stim_label)),axis = 1)

df_plot = pd.DataFrame(data = np.concatenate([stim_label.flatten()[:,None],
                                              trained_label.flatten()[:,None],
                                              stim_pupil.flatten()[:,None]],
                                             axis = 1), 
                       columns = ['stim', 'trained', 'motion'])

df_plot.loc[df_plot.stim==np.inf,'stim'] = 1
df_plot['stim'] = df_plot['stim'].astype(int).astype(str)
df_plot.loc[df_plot.stim=='1','stim'] = 'Blank'

pfig = sns.catplot(data = df_plot, x = 'stim', y = 'motion', 
            dodge = True, hue = 'trained', kind = 'strip')
pfig.set_axis_labels('Stimulus orientation (deg)', 'Face motion')

pfig._legend.set_title('')
pfig._legend.texts[0].set_text('Naive')
pfig._legend.texts[1].set_text('Proficient')
# plt.ylim([0.2,0.9])

p_vals = np.zeros(9)

uni_stim = np.unique(df_plot.stim)

for i,s in enumerate(uni_stim):
    
    p_vals[i] = ttest_ind(df_plot[np.logical_and(df_plot.stim==s, df_plot.trained==False)].motion,
                          df_plot[np.logical_and(df_plot.stim==s, df_plot.trained==True)].motion)[1]


for i,t in enumerate(pfig.ax.get_xticks()):
    if p_vals[i] < 0.05:
        plt.text(t,0.8,'*')
    else:
        plt.text(t,0.8,'N.S.')

#%% Plot comps by stimulus condition

stim_pupil = np.zeros((len(subjects),9))

for i,(s,t) in enumerate(zip(subjects,trained)):

    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    trial_pupil = trials_dict_pupil['dmotion'][ind]
    trial_pupil /= trial_pupil.mean()
    
    stim = trials_dict_pupil['stim_ori'][ind]
      
    stim_pupil[i,:] = np.array([trial_pupil[stim==s].mean() 
                                                  for s in np.unique(stim)])
    
#%% Plot 

stim_label = np.repeat(np.unique(stim).reshape(1,-1),
                       len(subjects), axis = 0)
trained_label = np.repeat(np.array(trained).reshape(-1,1), 
                          len(np.unique(stim_label)),axis = 1)

df_plot = pd.DataFrame(data = np.concatenate([stim_label.flatten()[:,None],
                                              trained_label.flatten()[:,None],
                                              stim_pupil.flatten()[:,None]],
                                             axis = 1), 
                       columns = ['stim', 'trained', 'face motion'])

sns.catplot(data = df_plot, x = 'stim', y = 'face motion', 
            hue = 'trained', kind = 'swarm', dodge = True)


#%% Plot sparsity vs SVs

uni_pref = np.unique(df_scm.pref_bin.to_numpy())

def pop_sparseness_ratio(resps, stim, r, pref):
    
    # breakpoint()
    top = np.logical_and(r >= 0.73, pref == stim)
    # bottom = r <= 0.28
    # ind_pref = np.logical_and(top,pref == o).to_numpy()
    ps = (resps[top].mean() 
                 / resps[np.logical_not(top)].mean())
        
    return ps

ps = np.zeros(len(subjects),dtype=object)
ps_stim = np.zeros(len(subjects), dtype = object)
ps_svs = np.zeros(len(subjects),dtype=object)

for i,(s,t) in enumerate(zip(subjects,trained)):
    
    ind = np.logical_and(trials_dict['subject'] == s, 
                         trials_dict['trained'] == t)
    ind_scm = np.logical_and(df_scm['subject'] == s,
                             df_scm['trained'] == t)
    # trial responses
    trial_resps = trials_dict['trial_resps'][ind]
    cells = trials_dict['cell'][ind]
    trial_resps = trial_resps.reshape(-1,len(np.unique(cells)))
    train_ind = trials_dict['train_ind'][ind].reshape(-1,len(np.unique(cells)))[:,0]
    trial_resps = trial_resps[train_ind,:]
    
    # cell preferences
    r = df_scm['r'][ind_scm]
    pref = df_scm['pref_bin'][ind_scm].to_numpy().astype(float)
    pref = np.ceil(pref)
    
    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    
    stim = trials_dict_pupil['stim_ori'][ind]
    stim = stim[train_ind]
    
    noblank_ind = stim != np.inf
    
    trial_resps = trial_resps[noblank_ind]
    stim = stim[noblank_ind]
  
    # SVs 
    trial_pupil = trials_dict_pupil['svd'][ind,:]
    trial_pupil = trial_pupil[train_ind]
    trial_pupil = trial_pupil[noblank_ind]
    
    ps_trials = np.zeros(len(trial_resps))
    
    for tr in range(len(trial_resps)):
        ps_trials[tr] = pop_sparseness_ratio(trial_resps[tr,:],stim[tr],
                                             r, pref)
    ps_stim[i] = stim
    ps_svs[i] = trial_pupil
    ps[i] = ps_trials
    
    
#%% Plot relationship for every experiment

n_svs = 9

for i in range(len(ps)):
    
    stim_label = np.repeat(ps_stim[i].reshape(-1,1), n_svs, axis = 1)
    stim_label = stim_label.astype('int')
    sv_label = np.repeat(np.arange(n_svs).reshape(1,-1),len(ps_stim[i]))
    ps_label = np.repeat(ps[i].reshape(-1,1), n_svs)   
    
    face_comps = ps_svs[i][:,np.arange(n_svs)]
    
    df = pd.DataFrame({'stim' : stim_label.flatten(),
                       'sv_num' : sv_label.flatten().astype(int).astype(str),
                       'ps' : ps_label.flatten(),
                       'face_comp' :face_comps.flatten()})
    df['stim'] = df['stim'].astype('category')
    
    
    fig = sns.FacetGrid(df, col = 'sv_num', col_wrap = 3)
    fig.map(sns.scatterplot, 'face_comp', 'ps')
    
    
#%% Plot sparsity vs motion

uni_pref = np.unique(df_scm.pref_bin.to_numpy())

def pop_sparseness_ratio(resps, stim, r, pref):
    
    # breakpoint()
    top = np.logical_and(r >= 0.73, pref == stim)
    # bottom = r <= 0.28
    # ind_pref = np.logical_and(top,pref == o).to_numpy()
    ps = (resps[top].mean() 
                 / resps[np.logical_not(top)].mean())
        
    return ps

ps = np.zeros(len(subjects),dtype=object)
ps_stim = np.zeros(len(subjects), dtype = object)
ps_motion = np.zeros(len(subjects),dtype=object)

for i,(s,t) in enumerate(zip(subjects,trained)):
    
    ind = np.logical_and(trials_dict['subject'] == s, 
                         trials_dict['trained'] == t)
    ind_scm = np.logical_and(df_scm['subject'] == s,
                             df_scm['trained'] == t)
    # trial responses
    trial_resps = trials_dict['trial_resps'][ind]
    cells = trials_dict['cell'][ind]
    trial_resps = trial_resps.reshape(-1,len(np.unique(cells)))
    train_ind = trials_dict['train_ind'][ind].reshape(-1,len(np.unique(cells)))[:,0]
    trial_resps = trial_resps[train_ind,:]
    
    # cell preferences
    r = df_scm['r'][ind_scm]
    pref = df_scm['pref_bin'][ind_scm].to_numpy().astype(float)
    pref = np.ceil(pref)
    
    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    
    stim = trials_dict_pupil['stim_ori'][ind]
    stim = stim[train_ind]
    
    noblank_ind = stim != np.inf
    
    trial_resps = trial_resps[noblank_ind]
    stim = stim[noblank_ind]
  
    # SVs 
    trial_pupil = trials_dict_pupil['motion'][ind]/trials_dict_pupil['motion'][ind].mean()
    trial_pupil = trial_pupil[train_ind]
    trial_pupil = trial_pupil[noblank_ind]
    
    ps_trials = np.zeros(len(trial_resps))
    
    for tr in range(len(trial_resps)):
        ps_trials[tr] = pop_sparseness_ratio(trial_resps[tr,:],stim[tr],
                                             r, pref)
    ps_stim[i] = stim
    ps_motion[i] = trial_pupil
    ps[i] = ps_trials
    
    
#%% Plot relationship for every experiment


for i in range(len(ps)):
    
    stim_label = ps_stim[i]
    stim_label = stim_label.astype('int')
    ps_label = ps[i]   
    
    df = pd.DataFrame({'stim' : ps_stim[i].astype(int),
                       'ps' : ps[i] ,
                       'motion' :ps_motion[i]})
    df['stim'] = df['stim'].astype('category')
    
    plt.figure()
    sns.scatterplot(data = df, y = 'ps', x = 'motion', 
                    hue = 'stim')
    
#%%

plt.figure()
sns.scatterplot(y = np.concatenate(ps,axis=0), x = np.concatenate(ps_motion,axis=0))
plt.xlabel('Face motion')
plt.ylabel('Trial population sparseness')

#%% Plot sparsity vs pupil

uni_pref = np.unique(df_scm.pref_bin.to_numpy())

def pop_sparseness_ratio(resps, stim, r, pref):
    
    # breakpoint()
    top = np.logical_and(r >= 0.73, pref == stim)
    # bottom = r <= 0.28
    # ind_pref = np.logical_and(top,pref == o).to_numpy()
    ps = (resps[top].mean() 
                 / resps[np.logical_not(top)].mean())
        
    return ps

ps = np.zeros(len(subjects),dtype=object)
ps_stim = np.zeros(len(subjects), dtype = object)
ps_motion = np.zeros(len(subjects),dtype=object)

for i,(s,t) in enumerate(zip(subjects,trained)):
    
    ind = np.logical_and(trials_dict['subject'] == s, 
                         trials_dict['trained'] == t)
    ind_scm = np.logical_and(df_scm['subject'] == s,
                             df_scm['trained'] == t)
    # trial responses
    trial_resps = trials_dict['trial_resps'][ind]
    cells = trials_dict['cell'][ind]
    trial_resps = trial_resps.reshape(-1,len(np.unique(cells)))
    train_ind = trials_dict['train_ind'][ind].reshape(-1,len(np.unique(cells)))[:,0]
    trial_resps = trial_resps[train_ind,:]
    
    # cell preferences
    r = df_scm['r'][ind_scm]
    pref = df_scm['pref_bin'][ind_scm].to_numpy().astype(float)
    pref = np.ceil(pref)
    
    ind = np.logical_and(trials_dict_pupil['subject']==s,
                         trials_dict_pupil['trained']==t)
    
    stim = trials_dict_pupil['stim_ori'][ind]
    stim = stim[train_ind]
    
    noblank_ind = stim != np.inf
    
    trial_resps = trial_resps[noblank_ind]
    stim = stim[noblank_ind]
  
    # SVs 
    trial_pupil = trials_dict_pupil['post_dpupil'][ind]/trials_dict_pupil['post_dpupil'][ind].mean()
    trial_pupil = trial_pupil[train_ind]
    trial_pupil = trial_pupil[noblank_ind]
    
    ps_trials = np.zeros(len(trial_resps))
    
    for tr in range(len(trial_resps)):
        ps_trials[tr] = pop_sparseness_ratio(trial_resps[tr,:],stim[tr],
                                             r, pref)
    ps_stim[i] = stim
    ps_motion[i] = trial_pupil
    ps[i] = ps_trials
    
    
#%% Plot relationship for every experiment


for i in range(len(ps)):
    
    stim_label = ps_stim[i]
    stim_label = stim_label.astype('int')
    ps_label = ps[i]   
    
    df = pd.DataFrame({'stim' : ps_stim[i].astype(int),
                       'ps' : ps[i] ,
                       'motion' :ps_motion[i]})
    df['stim'] = df['stim'].astype('category')
    
    plt.figure()
    sns.scatterplot(data = df, y = 'ps', x = 'motion', 
                    hue = 'stim')
    
#%%

plt.figure()
sns.scatterplot(y = np.concatenate(ps,axis=0), x = np.concatenate(ps_motion,axis=0))
plt.xlabel('Face motion')
plt.ylabel('Trial population sparseness')