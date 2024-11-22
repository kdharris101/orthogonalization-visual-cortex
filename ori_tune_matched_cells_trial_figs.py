#%%
import sys
sys.path.append(r'C:\Users\Samuel\OneDrive - University College London\Code\Python\Recordings')
# sys.path.append(r'C:\Users\samue\OneDrive - University College London\Code\Python\Recordings')

from importlib import reload
import glob
import twophopy as tp
import roi_retinotopy as rr
import mpeppy as mp
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
from os.path import join, split
from scipy.stats import circmean, pearsonr
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pycircstat as cs

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

results_root = r'C:\Users\Samuel\OneDrive - University College London\Results'
# results_root = r'C:\Users\samue\OneDrive - University College London\Results'


subjects = ['SF170620B','SF170905B', 'SF180515', 'SF180613']
subjects_file = [['M170620B_SF', 'SF170620B'],['M170905B_SF', 'M170905B_SF'],['SF180515', 'SF180515'],['SF180613','SF180613']]

match_files = [r'\\zubjects.cortexlab.net\Subjects\M170620B_SF\2017-07-04\suite2p\matches_for_M170620B_SF_naive_SF170620B_proficient.npy',
               r'\\zubjects.cortexlab.net\Subjects\M170905B_SF\2017-09-21\suite2p\matches_for_M170905B_SF_naive_SF170905B_proficient.npy',
               r'\\zubjects.cortexlab.net\Subjects\SF180515\2018-06-13\suite2p\matches_for_SF180515_naive_SF180515_proficient.npy',
               r'\\zubjects.cortexlab.net\Subjects\SF180613\2018-06-28\suite2p\matches_for_SF180613_naive_SF180613_proficient.npy']

expt_nums = [[5,7],[8,1],[1,3],[1,5]]
ret_expt_nums = [[6,8],[9,2],[2,4],[2,6]]


def split_path(filepath):
    parts = []
    while True:
        filepath, part = split(filepath)
        if part != "":
            parts.append(part)
        else:
            if filepath != "":
                parts.append(filepath)
            break
    parts.reverse()      
    return parts


def c_corr(a: np.ndarray, b: np.ndarray) -> float:

    """Fisher & Lee (1983); Zar (1999) eq (27.43)
    
    a, b: np.array, (n, )
        circular data in radian

    raa: float
         angular-angular correlation 
    """
    aij = np.triu(a[: ,None] - a).flatten()
    bij = np.triu(b[: ,None] - b).flatten()

    num = np.sum(np.sin(aij) * np.sin(bij))
    den = np.sqrt(np.sum(np.sin(aij) ** 2) * np.sum(np.sin(bij) ** 2))

    raa = num / den

    return raa



#%%

df_resps = pd.DataFrame()
df_trials = pd.DataFrame()
df_stats = pd.DataFrame()
ops = np.zeros((len(subjects),2),dtype=object)

for i in range(len(subjects)):
# for i in range(2,len(subjects)):
    
    # Load expts
    expts = tp.load_experiments_tracked(match_files[i],expt_nums[i],subjects_file[i])
    
    for ii in range(2):
        
        ops[i,ii] = expts[ii]['ops']
        
        # Add stimuli 
        parts = split_path(expts[ii]['rec_dir'])
        stimuli = mp.get_sequence((subjects_file[i][ii],parts[2],expt_nums[i][ii]),['ori'])
        # Transform stimuli to Rigbox/Signals convention
        stimuli = 360 - stimuli
        stimuli[stimuli==360] = 0
            
        # Correct stim
        stimuli[stimuli % 180 == 22] += 1
        stimuli[stimuli % 180 == 67] += 1
        stimuli[stimuli % 180 == 112] += 1
        stimuli[stimuli % 180 == 157] += 1

        # Set blank condition (ori = 358) to inf
        stimuli[stimuli == 358] = np.inf

        stimuli = np.ceil(stimuli)
        stimuli[stimuli==2] = np.inf
        

        expts[ii]['stim_dir'] = np.copy(stimuli)
        stimuli[~np.isinf(stimuli)] = stimuli[~np.isinf(stimuli)] % 180
        expts[ii]['stim_ori'] = stimuli
        
        expts[ii]['stim_times'] = mp.get_stim_times((subjects_file[i][ii],parts[2],expt_nums[i][ii]))
    
        # Add retinotopy
        ret_file_path = glob.glob(join(results_root, subjects_file[i][ii], parts[2], str(ret_expt_nums[i][ii]), f'{subjects_file[i][ii]}_{parts[2]}_{str(ret_expt_nums[i][ii])}_peak_pixel_retinotopy*'))[-1]
        expts[ii] = rr.add_roi_retinotopy(ret_file_path,expts[ii])
        
        print('Finding trial responses')
        expts[ii]['pre_activity'] = np.array([]).reshape((len(expts[ii]['stim_dir']), -1))
        expts[ii]['trial_resps_raw'] = np.array([]).reshape((len(expts[ii]['stim_dir']), -1))
    
        n_trials = len(expts[ii]['stim_times'][0])
        
        planes = np.unique(expts[ii]['cell_plane'])
        
        n_cells = len(expts[ii]['cell_plane'])
        
        # Isolate trial responses - make an n_trial by n_cell array
        pre_stim = -1
        post_stim = [0, 2]
        
        for p in planes:
            plane_spks = expts[ii]['spks'][expts[ii]['cell_plane'] == p, :]
            
            pre_ind = np.concatenate([np.logical_and(expts[ii]['plane_times'][p] 
                >= t + pre_stim, expts[ii]['plane_times'][p] 
                < t) for t in expts[ii]['stim_times'][0]])
            pre_ind = np.reshape(pre_ind,(n_trials, len(expts[ii]['plane_times'][p])))
        
            post_ind = np.concatenate([np.logical_and(expts[ii]['plane_times'][p] 
                >= t + post_stim[0], expts[ii]['plane_times'][p] 
                <= t + post_stim[1]) for t in expts[ii]['stim_times'][0]])
            
            post_ind = np.reshape(post_ind,(n_trials, len(expts[ii]['plane_times'][p])))
            
            pre_plane = np.divide(np.dot(pre_ind, plane_spks.T).T, 
                                np.sum(pre_ind, axis=1).T).T
            post_plane = np.divide(np.dot(post_ind, plane_spks.T).T, 
                                np.sum(post_ind, axis=1).T).T
        
            
            expts[ii]['pre_activity'] = np.concatenate([expts[ii]['pre_activity'],
                                                pre_plane], axis = 1)
            
            expts[ii]['trial_resps_raw'] = np.concatenate([expts[ii]['trial_resps_raw'],
                                                post_plane], axis = 1)
        
        # Using interpolation, create an n_cell x n_trials x n_timepoints (every 0.1 sec) array
        
        timepoints = np.linspace(-1,3,41)
        n_timepoints = len(timepoints)
        
        trial_activity = []
        
        for p in planes:
            
            interpolant = interp1d(expts[ii]['plane_times'][p], expts[ii]['spks'][expts[ii]['cell_plane']==p,:], 
                                   kind = 'nearest', bounds_error = False)
            t_timepoints = np.array([t+timepoints for t in expts[ii]['stim_times'][0]])
            trial_activity.append(interpolant(t_timepoints))
        
        
        trial_activity = np.concatenate(trial_activity,axis=0)
        
        expts[ii]['full_trial_activity_raw'] = trial_activity
        
        # Make train and test indices
        ind_all = np.arange(len(expts[ii]['stim_dir']))
        expts[ii]['train_ind'] = np.concatenate([np.where(expts[ii]['stim_ori'] == s)[0][::2] 
                                for s in np.unique(expts[ii]['stim_ori'])])
        expts[ii]['test_ind'] = np.delete(ind_all,expts[ii]['train_ind'])    
               
        # Mean respones, including blank (last)
        uni_stim = np.unique(expts[ii]['stim_ori'])
        
        expts[ii]['trial_resps_train_raw'] = expts[ii]['trial_resps_raw'][expts[ii]['train_ind'],:]
        expts[ii]['trial_resps_test_raw'] = expts[ii]['trial_resps_raw'][expts[ii]['test_ind'],:]
        
        expts[ii]['stim_ori_train'] = expts[ii]['stim_ori'][expts[ii]['train_ind']]
        expts[ii]['stim_ori_test'] = expts[ii]['stim_ori'][expts[ii]['test_ind']]
        
        expts[ii]['stim_dir_train'] = expts[ii]['stim_dir'][expts[ii]['train_ind']]
        expts[ii]['stim_dir_test'] = expts[ii]['stim_dir'][expts[ii]['test_ind']]
        
        mean_stim = np.concatenate([
            np.mean(expts[ii]['trial_resps_raw'][expts[ii]['stim_ori'] == s,:],
                axis = 0).reshape((1,-1)) for s in uni_stim])
        
        expts[ii]['scale_factor'] = np.max(mean_stim,0,keepdims=True)
        expts[ii]['blank_pref'] = np.argmax(mean_stim,0) == 8
        
        # Scale by mean preferred stim response
        expts[ii]['trial_resps'] = expts[ii]['trial_resps_raw']/expts[ii]['scale_factor']
        expts[ii]['trial_resps_train'] = expts[ii]['trial_resps_train_raw']/expts[ii]['scale_factor']
        expts[ii]['trial_resps_test'] = expts[ii]['trial_resps_test_raw']/expts[ii]['scale_factor']
        expts[ii]['pre_activity'] /= expts[ii]['scale_factor']
        
        expts[ii]['full_trial_activity'] = expts[ii]['full_trial_activity_raw']/expts[ii]['scale_factor'].T[:,None]
        
        # Use circular measures of orientation preference
    
        nb = expts[ii]['stim_ori_train'] != np.inf
        
        x = np.sum(np.cos(expts[ii]['stim_ori_train'][nb,None]*np.pi/90.)*expts[ii]['trial_resps_train'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_train'][nb,:],0)
            
        y = np.sum(np.sin(expts[ii]['stim_ori_train'][nb,None]*np.pi/90.)*expts[ii]['trial_resps_train'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_train'][nb,:],0)
            
        expts[ii]['r'] = np.sqrt(x**2 + y**2)
        expts[ii]['th'] = np.mod(11.25+np.arctan2(y, x)*90/np.pi, 180)-11.25
        expts[ii]['v_x'] = x
        expts[ii]['v_y'] = y

        nb = expts[ii]['stim_ori'] != np.inf
        
        x = np.sum(np.cos(expts[ii]['stim_ori'][nb,None]*np.pi/90.)*expts[ii]['trial_resps'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps'][nb,:],0)
            
        y = np.sum(np.sin(expts[ii]['stim_ori'][nb,None]*np.pi/90.)*expts[ii]['trial_resps'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps'][nb,:],0)
            
        expts[ii]['r_all'] = np.sqrt(x**2 + y**2)
        expts[ii]['th_all'] = np.mod(11.25+np.arctan2(y, x)*90/np.pi, 180)-11.25
        expts[ii]['v_x_all'] = x
        expts[ii]['v_y_all'] = y
        
        nb = expts[ii]['stim_ori_test'] != np.inf
        
        x = np.sum(np.cos(expts[ii]['stim_ori_test'][nb,None]*np.pi/90.)*expts[ii]['trial_resps_test'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_test'][nb,:],0)
            
        y = np.sum(np.sin(expts[ii]['stim_ori_test'][nb,None]*np.pi/90.)*expts[ii]['trial_resps_test'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_test'][nb,:],0)
            
        expts[ii]['r_test'] = np.sqrt(x**2 + y**2)
        expts[ii]['th_test'] = np.mod(11.25+np.arctan2(y, x)*90/np.pi, 180)-11.25
        expts[ii]['v_x_test'] = x
        expts[ii]['v_y_test'] = y
        
        # Use circular measures of direction preference
        
        nb = expts[ii]['stim_ori_train'] != np.inf
        
        x = np.sum(np.cos(expts[ii]['stim_dir_train'][nb,None]*np.pi/180.)*expts[ii]['trial_resps_train'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_train'][nb,:],0)
            
        y = np.sum(np.sin(expts[ii]['stim_dir_train'][nb,None]*np.pi/180.)*expts[ii]['trial_resps_train'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_train'][nb,:],0)
            
        expts[ii]['r_dir'] = np.sqrt(x**2 + y**2)
        expts[ii]['th_dir'] = np.arctan2(y, x)*180/np.pi-11.25
        expts[ii]['v_x_dir'] = x
        expts[ii]['v_y_dir'] = y

        nb = expts[ii]['stim_ori'] != np.inf
        
        x = np.sum(np.cos(expts[ii]['stim_dir'][nb,None]*np.pi/90.)*expts[ii]['trial_resps'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps'][nb,:],0)
            
        y = np.sum(np.sin(expts[ii]['stim_dir'][nb,None]*np.pi/90.)*expts[ii]['trial_resps'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps'][nb,:],0)
            
        expts[ii]['r_dir_all'] = np.sqrt(x**2 + y**2)
        expts[ii]['th_dir_all'] = np.arctan2(y, x)*180/np.pi-11.25
        expts[ii]['v_x_dir_all'] = x
        expts[ii]['v_y_dir_all'] = y
        
        nb = expts[ii]['stim_ori_test'] != np.inf
        
        x = np.sum(np.cos(expts[ii]['stim_dir_test'][nb,None]*np.pi/180.)*expts[ii]['trial_resps_test'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_test'][nb,:],0)
            
        y = np.sum(np.sin(expts[ii]['stim_dir_test'][nb,None]*np.pi/180.)*expts[ii]['trial_resps_test'][nb,:],0) \
            / np.sum(expts[ii]['trial_resps_test'][nb,:],0)
            
        expts[ii]['r_dir_test'] = np.sqrt(x**2 + y**2)
        expts[ii]['th_dir_test'] = np.mod(11.25+np.arctan2(y, x)*180/np.pi, 180)-11.25
        expts[ii]['v_x_dir_test'] = x
        expts[ii]['v_y_dir_test'] = y
        
        
        # Save expts
        # Add some code to save the new expts here
        
        n_trials, n_cells = expts[ii]['trial_resps_raw'].shape
        
        cell_num = np.repeat(np.arange(n_cells)[None,:], n_trials, axis=0)
        cell_plane = np.repeat(expts[ii]['cell_plane'][None,:], n_trials, axis=0)
        trial_num = np.repeat(np.arange(n_trials)[:,None], n_cells, axis=1)
        
        V1_ROIs = np.repeat(expts[ii]['V1_ROIs'][None,:], n_trials, axis=0)
        roi_ret = np.repeat(expts[ii]['ROI_ret'][None,:], n_trials, axis=0)
        stim_ori = np.repeat(expts[ii]['stim_ori'][:,None], n_cells, axis=1)
        stim_dir = np.repeat(expts[ii]['stim_dir'][:,None], n_cells, axis=1)
        condition = np.repeat(['naive','proficient'][ii], expts[ii]['trial_resps_raw'].size)
        expt_num = np.repeat(i+ii, expts[ii]['trial_resps_raw'].size)
        e_train_ind = np.zeros(n_trials, dtype=bool)
        train_trials = np.concatenate([np.where(expts[ii]['stim_ori']==s)[0][::2] for s in np.unique(expts[ii]['stim_ori'])])
        e_train_ind[train_trials] = True
        train_ind = np.repeat(e_train_ind[:,None], n_cells, axis=1)

        blank_pref = np.repeat(expts[ii]['blank_pref'][None,:], n_trials, axis=0)
        
        expt_dict = {
                    'trial_resps' : expts[ii]['trial_resps_raw'].ravel(),
                    'cell_num' : cell_num.ravel(),
                    'cell_plane' : cell_plane.ravel(),
                    'trial_num' : trial_num.ravel(),
                    'V1_ROI' : V1_ROIs.ravel(),
                    'ROI_ret' : roi_ret.ravel(),
                    'stim_ori' : stim_ori.ravel(),
                    'stim_dir' : stim_dir.ravel(),
                    'subject' : np.repeat(subjects[i], expts[ii]['trial_resps_raw'].size),
                    'condition' : condition.ravel(),
                    'expt_num' : expt_num.ravel(),
                    'train_ind' : train_ind.ravel(),
                    'blank_pref' : blank_pref.ravel()
                    }
    
        df_resps = pd.concat([df_resps,pd.DataFrame(expt_dict)], ignore_index=True)
        
        trials_dict =   {
                        'trial_activity' : expts[ii]['full_trial_activity'].ravel(),
                        'trial_times' : np.tile(timepoints[None,None,:], (n_cells,n_trials,1)).ravel(),
                        'cell_num' : np.repeat(cell_num.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'cell_plane' : np.repeat(cell_plane.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'trial_num' : np.repeat(trial_num.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'V1_ROI' : np.repeat(V1_ROIs.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'ROI_ret' : np.repeat(roi_ret.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'stim_ori' : np.repeat(stim_ori.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'stim_dir' : np.repeat(stim_dir.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'subject' : np.repeat(subjects[i],expts[ii]['full_trial_activity'].size),
                        'condition' : np.repeat(['naive','proficient'][ii], expts[ii]['full_trial_activity'].size),
                        'expt_num' : np.repeat(i+ii, expts[ii]['full_trial_activity'].size),
                        'train_ind' : np.repeat(train_ind.T[:,:,None], n_timepoints, axis=1).ravel(),
                        'blank_pref' : np.repeat(blank_pref.T[:,:,None], n_timepoints, axis=1).ravel(),
                        }
        
        df_trials = pd.concat([df_trials,pd.DataFrame(trials_dict)], ignore_index=True)
        
        stats_dict = {'subject' : np.repeat(subjects[i], len(expts[ii]['stat'])),
                      'condition' : np.repeat(['naive','proficient'][ii], len(expts[ii]['stat'])),
                      'cell_plane' : expts[ii]['cell_plane'],
                      'stat' : expts[ii]['stat'],
                      'blank_pref' : expts[ii]['blank_pref']
                     }
        
        df_stats = pd.concat([df_stats, pd.DataFrame(stats_dict)], ignore_index=True)
        
        
 #%%

# Save original in case you need to restore it
df_resps_original = df_resps.copy()

# blank_pref = df_resps.groupby(['cell_num','subject']).filter(lambda x: np.any(x['blank_pref'])).index

# Remove cells whose ROIs are not in V1 in both recordings
not_in_V1 = df_resps.groupby(['cell_num','subject'], observed = True).filter(lambda x: ~np.all(x['V1_ROI'])).index
# not_in_V1 = df_resps.groupby(['cell_num','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index

# near_task_stim = df_resps.groupby(['cell_num','subject']).filter(lambda x: ~np.any(x['ROI_task_ret'] > 16)).index
# df_resps = df_resps.drop(np.unique(np.concatenate([near_task_stim,not_in_V1])))
df_resps = df_resps.drop(not_in_V1)

# Scale responses by preferred stimulus
pref_resp = df_resps.groupby(['cell_num','stim_ori','condition', 'subject'], observed = True).mean().reset_index()
pref_stim = pref_resp.loc[pref_resp.groupby(['cell_num','stim_ori','condition','subject'], observed=True)['trial_resps'].idxmax(), 'stim_ori'].reset_index(drop=True)
pref_resp = pref_resp.groupby(['cell_num','condition','subject'], observed=True)['trial_resps'].max().reset_index()
pref_resp = pref_resp.rename(columns={'trial_resps':'pref_resp'})
# pref_resp['stim_with_max_resp'] = pref_stim
# pref_resp['blank_pref'] = np.isinf(pref_resp.stim_with_max_resp)

df_resps = df_resps.merge(pref_resp, on = ['cell_num','condition', 'subject'], how='left')

df_resps['trial_resps_raw'] = df_resps['trial_resps']

df_resps['trial_resps'] = df_resps['trial_resps']/df_resps['pref_resp']

df_resps['trial_resps_norm_prt'] = df_resps.groupby(['cell_num','subject','condition'], group_keys = False)['trial_resps_raw'].apply(lambda x: x/np.percentile(x,98))

# For df_trials
# Remove blank preferring
blank_pref = df_trials.groupby(['cell_num','subject'], observed = True).filter(lambda x: np.any(x['blank_pref'])).index

# Remove cells whose ROIs are not in V1 in any recordings
not_in_V1 = df_trials.groupby(['cell_num','subject'], observed = True).filter(lambda x: ~np.all(x['V1_ROI'])).index
# df_trials = df_trials.drop(np.unique(np.concatenate([blank_pref,not_in_V1])))
df_trials = df_trials.drop(np.unique(not_in_V1))


#%% Function to calculate ori tuning

def find_tuning(df_resps, train_only = True):

    ind = ~np.isinf(df_resps.stim_ori)

    if train_only:
        ind = df_resps.train_ind & ind

    df_tuning = df_resps[ind].copy()
    df_tuning['stim_ori_rad'] = df_tuning['stim_ori'] * np.pi/90
    df_tuning['exp(stim_ori_rad)*trial_resp'] = np.exp(df_tuning.stim_ori_rad*1j) * df_tuning.trial_resps

    df_tuning = df_tuning.groupby(['cell_num', 'subject', 'condition'], observed = True).agg({'exp(stim_ori_rad)*trial_resp':'sum',
                                                                                              'trial_resps' : 'sum',
                                                                                              'ROI_ret' : 'mean'})

    df_tuning['tune_vec'] = df_tuning['exp(stim_ori_rad)*trial_resp']/df_tuning['trial_resps']
    df_tuning['r'] = df_tuning.tune_vec.abs()
    df_tuning['mean_pref'] = np.mod(11.25+np.arctan2(np.imag(df_tuning.tune_vec),np.real(df_tuning.tune_vec))*90/np.pi,180)-11.25

    df_tuning = df_tuning.drop(columns = ['exp(stim_ori_rad)*trial_resp','trial_resps'])
    df_tuning = df_tuning.reset_index()

    mu_resp = df_resps[~np.isinf(df_resps.stim_ori)].groupby(['cell_num', 'subject', 'condition', 'stim_ori'], observed = True)['trial_resps'].mean().reset_index()
    df_tuning['modal_pref'] = mu_resp.loc[mu_resp.groupby(['cell_num', 'subject', 'condition'], observed = True)['trial_resps'].idxmax(),'stim_ori'].reset_index(drop=True)

    return df_tuning


#%% Cell tuning

df_tuning = find_tuning(df_resps)

pref_naive = df_tuning.loc[df_tuning.condition=='naive',['mean_pref','modal_pref','r','cell_num','subject']]
pref_naive = pref_naive.rename(columns = {'mean_pref' : 'mean_pref_naive',
                                          'modal_pref' : 'modal_pref_naive', 
                                          'r' : 'r_naive'})
df_tuning = df_tuning.merge(pref_naive, on = ['cell_num','subject'], how = 'left')

pref_proficient = df_tuning.loc[df_tuning.condition=='proficient',['mean_pref','modal_pref','r','cell_num','subject']]
pref_proficient = pref_proficient.rename(columns = {'mean_pref' : 'mean_pref_proficient',
                                                    'modal_pref' : 'modal_pref_proficient', 
                                                    'r' : 'r_proficient'})
df_tuning = df_tuning.merge(pref_proficient, on = ['cell_num', 'subject'], how = 'left')


# Bin cells by mean preference and selectivity (r)
df_tuning['pref_bin_naive'] = pd.cut(df_tuning.mean_pref_naive, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_tuning['r_bin_naive'] = pd.cut(df_tuning.r_naive, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))
df_tuning['pref_bin_proficient'] = pd.cut(df_tuning.mean_pref_proficient, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_tuning['r_bin_proficient'] = pd.cut(df_tuning.r_proficient, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

df_tuning['pref_bin'] = pd.cut(df_tuning.mean_pref, np.linspace(-11.25,180-11.25,9), labels = np.ceil(np.arange(0,180,22.5)))
df_tuning['r_bin'] = pd.cut(df_tuning.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels = np.arange(5))

# Tuning curves from test set
tuning_curves_test = df_resps.groupby(['cell_num', 'condition', 'subject', 'stim_ori', 'train_ind'], observed = True)['trial_resps'].mean().reset_index()
# Exclude training set and blank trials
tuning_curves_test = tuning_curves_test[~tuning_curves_test.train_ind & ~np.isinf(tuning_curves_test.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp'})

tuning_curves_all = df_resps.groupby(['cell_num', 'condition', 'subject', 'stim_ori'], observed = True)['trial_resps'].mean().reset_index()
tuning_curves_all = tuning_curves_all[~np.isinf(tuning_curves_all.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp_all_trials'})

df_tuning = df_tuning.merge(tuning_curves_test, on = ['cell_num', 'condition', 'subject'], how = 'left').drop(columns = 'train_ind')
df_tuning = df_tuning.merge(tuning_curves_all, on = ['cell_num', 'condition', 'subject', 'stim_ori'], how = 'left')

# df_tuning = df_tuning.astype({'pref_bin_naive' : int, 'r_bin_naive' : int, 'pref_bin_proficient' : int, 'r_bin_proficient' : int})


#%% Plot changes in modal and mean ori pref


style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

df_prefs = df_tuning.groupby(['subject','cell_num'])[['modal_pref_naive', 'modal_pref_proficient']].first().reset_index()
naive_props = df_prefs.modal_pref_naive.value_counts(normalize=True).sort_index()
proficient_props = df_prefs.modal_pref_proficient.value_counts(normalize=True).sort_index()

df_change = df_prefs.drop(['subject','cell_num'], axis=1)

df_change = df_change.groupby('modal_pref_naive').value_counts(normalize=True).reset_index()
df_change.rename({0 : 'Prop'}, axis=1, inplace=True)

x = np.concatenate([np.zeros(len(naive_props)), np.ones(len(naive_props))])
y = list(df_change.modal_pref_naive.unique())*2

df_plot = pd.DataFrame({'x' : x,
                        'y' : y,
                        'prop' : np.concatenate([naive_props,proficient_props])*100})


f = plt.figure()
a0 = f.add_subplot(1,2,1)

(
    so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'y')
    .layout(engine='tight')
    .add(so.Dot(), legend=False)
    .scale(y=so.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))),
           x=so.Continuous().tick(at=[0,1]),
           pointsize=(2,20),
           color = 'colorblind')
    .label(y='Modal orientation preference',
           x='')
    .limit(x=(-0.5,1.5))
    .theme({**style})
    .on(a0)
    .plot()
)

min_shade = 1

for i in range(len(df_change)):
    shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
    a0.plot([0,1],
           [df_change.loc[i, 'modal_pref_naive'], 
           df_change.loc[i, 'modal_pref_proficient']],
           linewidth=df_change.loc[i, 'Prop'] * 10,
           color = np.where(shade>min_shade, min_shade, shade),
           zorder = 0)

a0.set_xticklabels(['Naive','Proficient'])

sns.despine(ax=a0, trim=True)

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

df_prefs = df_tuning.groupby(['subject','cell_num'])[['pref_bin_naive', 'pref_bin_proficient']].first().reset_index()
naive_props = df_prefs.pref_bin_naive.value_counts(normalize=True).sort_index()
proficient_props = df_prefs.pref_bin_proficient.value_counts(normalize=True).sort_index()

df_change = df_prefs.drop(['subject','cell_num'], axis=1)

df_change = df_change.groupby('pref_bin_naive').value_counts(normalize=True).reset_index()
df_change.rename({0 : 'Prop'}, axis=1, inplace=True)

x = np.concatenate([np.zeros(len(naive_props)), np.ones(len(naive_props))])
y = list(df_change.pref_bin_naive.unique())*2

df_plot = pd.DataFrame({'x' : x,
                        'y' : y,
                        'prop' : np.concatenate([naive_props,proficient_props])*100})


a1 = f.add_subplot(1,2,2)

(
    so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'y')
    .layout(engine='tight')
    .add(so.Dot(), legend=False)
    .scale(y=so.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))),
           x=so.Continuous().tick(at=[0,1]),
           pointsize=(2,20),
           color = 'colorblind')
    .label(y='Mean orientation preference',
           x='')
    .limit(x=(-0.5,1.5))
    .theme({**style})
    .on(a1)
    .plot()
)


for i in range(len(df_change)):
    shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
    a1.plot([0,1],
           [df_change.loc[i, 'pref_bin_naive'], 
           df_change.loc[i, 'pref_bin_proficient']],
           linewidth=df_change.loc[i, 'Prop'] * 10,
           color = np.where(shade>min_shade,min_shade,shade),
           zorder = 0)

a1.set_xticklabels(['Naive','Proficient'])

sns.despine(ax=a1, trim=True)

f.show()

#%% Plot changes in modal and mean ori pref - flip axes


style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

df_prefs = df_tuning.groupby(['subject','cell_num'])[['modal_pref_naive', 'modal_pref_proficient']].first().reset_index()
naive_props = df_prefs.modal_pref_naive.value_counts(normalize=True).sort_index()
proficient_props = df_prefs.modal_pref_proficient.value_counts(normalize=True).sort_index()

df_change = df_prefs.drop(['subject','cell_num'], axis=1)

df_change = df_change.groupby('modal_pref_naive').value_counts(normalize=True).reset_index()
df_change.rename({0 : 'Prop'}, axis=1, inplace=True)

y = np.concatenate([np.ones(len(naive_props)), np.zeros(len(naive_props))])
x = list(df_change.modal_pref_naive.unique())*2

df_plot = pd.DataFrame({'x' : x,
                        'y' : y,
                        'prop' : np.concatenate([naive_props,proficient_props])*100})


f = plt.figure()
a0 = f.add_subplot(1,2,1)

(
    so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'x')
    .layout(engine='tight')
    .add(so.Dot(), legend=False)
    .scale(x=so.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))),
           y=so.Continuous().tick(at=[0,1]),
           pointsize=(2,20),
           color = 'colorblind')
    .label(x='Modal orientation preference',
           y='')
    .limit(y=(-0.5,1.5))
    .theme({**style})
    .on(a0)
    .plot()
)

min_shade = 1

for i in range(len(df_change)):
    shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
    a0.plot([df_change.loc[i, 'modal_pref_naive'], 
            df_change.loc[i, 'modal_pref_proficient']],
            [1,0],
            linewidth=df_change.loc[i, 'Prop'] * 10,
            color = np.where(shade>min_shade, min_shade, shade),
            zorder = 0)

a0.set_yticklabels(['Proficient','Naive'])

sns.despine(ax=a0, trim=True)

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

df_prefs = df_tuning.groupby(['subject','cell_num'])[['pref_bin_naive', 'pref_bin_proficient']].first().reset_index()
naive_props = df_prefs.pref_bin_naive.value_counts(normalize=True).sort_index()
proficient_props = df_prefs.pref_bin_proficient.value_counts(normalize=True).sort_index()

df_change = df_prefs.drop(['subject','cell_num'], axis=1)

df_change = df_change.groupby('pref_bin_naive').value_counts(normalize=True).reset_index()
df_change.rename({0 : 'Prop'}, axis=1, inplace=True)

y = np.concatenate([np.ones(len(naive_props)), np.zeros(len(naive_props))])
x = list(df_change.pref_bin_naive.unique())*2

df_plot = pd.DataFrame({'x' : x,
                        'y' : y,
                        'prop' : np.concatenate([naive_props,proficient_props])*100})


a1 = f.add_subplot(1,2,2)

(
    so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'x')
    .layout(engine='tight')
    .add(so.Dot(), legend=False)
    .scale(x=so.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))),
           y=so.Continuous().tick(at=[0,1]),
           pointsize=(2,20),
           color = 'colorblind')
    .label(x='Mean orientation preference',
           y='')
    .limit(y=(-0.5,1.5))
    .theme({**style})
    .on(a1)
    .plot()
)


for i in range(len(df_change)):
    shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
    a1.plot([df_change.loc[i, 'pref_bin_naive'], 
            df_change.loc[i, 'pref_bin_proficient']],
            [1,0],
            linewidth=df_change.loc[i, 'Prop'] * 10,
            color = np.where(shade>min_shade,min_shade,shade),
            zorder = 0)

a1.set_yticklabels(['Proficient','Naive'])

sns.despine(ax=a1, trim=True)

f.show()

#%% Plot changes in modal and mean ori pref - group by selectivity


style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False


df_prefs = df_tuning.groupby(['subject','cell_num'])[['modal_pref_naive', 'modal_pref_proficient', 'r_bin_naive']].first().reset_index()

naive_props = df_prefs.groupby('r_bin_naive').modal_pref_naive.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'modal_pref_naive' : 'proportion'}, axis=1).reset_index()
proficient_props = df_prefs.groupby('r_bin_naive').modal_pref_proficient.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'modal_pref_proficient' : 'proportion'}, axis=1).reset_index()

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,
                                        "axes.titlesize":5,
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


    f0 = plt.figure(figsize=(4.5,1.75))

    n_r_bins = len(naive_props.r_bin_naive.unique())

    min_shade = 1


    for r in range(n_r_bins):

        df_change = df_prefs[df_prefs.r_bin_naive==r].drop(['subject','cell_num','r_bin_naive'], axis=1)

        df_change = df_change.groupby('modal_pref_naive').value_counts(normalize=True).reset_index()
        df_change.rename({0 : 'Prop'}, axis=1, inplace=True)

        x = np.concatenate([np.zeros(len(naive_props[naive_props.r_bin_naive==r])), np.ones(len(naive_props[naive_props.r_bin_naive==r]))])
        y = list(df_change.modal_pref_naive.unique())*2

        df_plot = pd.DataFrame({'x' : x,
                                'y' : y,
                                'prop' : np.concatenate([naive_props[naive_props.r_bin_naive==r].proportion,
                                                        proficient_props[proficient_props.r_bin_naive==r].proportion])*100})
        
        a0 = f0.add_subplot(1,n_r_bins,r+1)

        (
            so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'y')
            .layout(engine='tight')
            .add(so.Dot(), legend=False)
            .scale(y=so.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))),
                x=so.Continuous().tick(at=[0,1]),
                pointsize=(2,10),
                color = 'colorblind')
            .label(y='Modal orientation preference',
                x='')
            .limit(x=(-0.5,1.5))
            .theme({**style})
            .on(a0)
            .plot()
        )

        for i in range(len(df_change)):
            if df_change.loc[i, 'Prop'] > 0:
                shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
                a0.plot([0,1],
                    [df_change.loc[i, 'modal_pref_naive'], 
                    df_change.loc[i, 'modal_pref_proficient']],
                    linewidth=df_change.loc[i, 'Prop']*3,
                    color = np.where(shade>min_shade, min_shade, shade),
                    zorder = 0)

        a0.set_xticklabels(['Naive','Proficient'])

        if r == 0:
            sns.despine(ax=a0, trim=True)
        else:
            sns.despine(ax=a0, trim=True, left=True)
            a0.set_yticks([])
            a0.set_ylabel('')


    df_prefs = df_tuning.groupby(['subject','cell_num'])[['pref_bin_naive', 'pref_bin_proficient', 'r_bin_naive']].first().reset_index()

    naive_props = df_prefs.groupby('r_bin_naive').pref_bin_naive.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'pref_bin_naive' : 'proportion'}, axis=1).reset_index()
    proficient_props = df_prefs.groupby('r_bin_naive').pref_bin_proficient.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'pref_bin_proficient' : 'proportion'}, axis=1).reset_index()

    f1 = plt.figure()

    n_r_bins = len(naive_props.r_bin_naive.unique())

    for r in range(n_r_bins):

        df_change = df_prefs[df_prefs.r_bin_naive==r].drop(['subject','cell_num','r_bin_naive'], axis=1)

        df_change = df_change.groupby('pref_bin_naive').value_counts(normalize=True).reset_index()
        df_change.rename({0 : 'Prop'}, axis=1, inplace=True)

        x = np.concatenate([np.zeros(len(naive_props[naive_props.r_bin_naive==r])), np.ones(len(naive_props[naive_props.r_bin_naive==r]))])
        y = list(df_change.pref_bin_naive.unique())*2

        df_plot = pd.DataFrame({'x' : x,
                                'y' : y,
                                'prop' : np.concatenate([naive_props[naive_props.r_bin_naive==r].proportion,
                                                        proficient_props[proficient_props.r_bin_naive==r].proportion])*100})
        
        a1 = f1.add_subplot(1,n_r_bins,r+1)

        (
            so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'y')
            .layout(engine='tight')
            .add(so.Dot(), legend=False)
            .scale(y=so.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))),
                x=so.Continuous().tick(at=[0,1]),
                pointsize=(2,20),
                color = 'colorblind')
            .label(y='Mean orientation preference',
                x='')
            .limit(x=(-0.5,1.5))
            .theme({**style})
            .on(a1)
            .plot()
        )

        for i in range(len(df_change)):
            
            if df_change.loc[i, 'Prop'] > 0:
                shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
                a1.plot([0,1],
                    [df_change.loc[i, 'pref_bin_naive'], 
                    df_change.loc[i, 'pref_bin_proficient']],
                    linewidth=df_change.loc[i, 'Prop'] * 10,
                    color=np.where(shade>min_shade, min_shade, shade),
                    zorder=0)

        a1.set_xticklabels(['Naive','Proficient'])

        sns.despine(ax=a1, trim=True)


    f0.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\tracked_modal_pref.svg', format='svg')

#%% test for significant orientation tuning


def mahalanobis(x, data):
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=0)
    inv_sigma = np.linalg.inv(sigma)

    x_sub_mu = x-mu

    return x_sub_mu @ inv_sigma @ x_sub_mu.T


df = df_resps[(df_resps.cell_num==1) & (df_resps.condition=='proficient') & (df_resps.subject=='SF170620B')].copy()
df_shuffle = df.copy()

n_shuffles = 2000

pref_vec = np.zeros((n_shuffles,2))

for i in range(n_shuffles):

    df_shuffle['stim_ori'] = df_shuffle.stim_ori.sample(frac=1).values

    df_t = find_tuning(df_shuffle)

    pref_vec[i,0] = np.real(df_t.tune_vec)
    pref_vec[i,1] = np.imag(df_t.tune_vec)


true_vec = find_tuning(df).tune_vec.to_numpy()

plt.scatter(x=pref_vec[:,0],y=pref_vec[:,1])
plt.scatter(x = np.real(true_vec), y=np.imag(true_vec))

true_pref = np.zeros((1,2))

true_pref[:,0] = np.real(true_vec)
true_pref[:,1] = np.imag(true_vec)

dis = mahalanobis(true_pref, pref_vec)

p = 1 - chi2.cdf(dis,2)


#%% Ori tuning significance for all cells/sessions

df_shuffle = df_resps.copy()

df_shuffle = df_resps.sort_values(by = ['cell_num','subject','condition'])

n_shuffles = 1000

df_shuffles = pd.DataFrame()

for i in range(n_shuffles):

    print(f'Shuffle {i}')

    df_shuffle['stim_ori'] = df_shuffle.groupby(['cell_num','subject','condition','train_ind'], observed = True)['stim_ori'].transform(np.random.permutation)

    df_tmp = find_tuning(df_shuffle)
   
    df_tmp['shuffle_num'] = i
   
    df_shuffles = pd.concat([df_shuffles,df_tmp], ignore_index=True)

 

#%% Get p-values - mean ori pref

from scipy.stats import chi2
from statsmodels.stats.multitest import fdrcorrection

def mahalanobis(x, data):
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=0)
    inv_sigma = np.linalg.inv(sigma)

    x_sub_mu = x-mu

    return x_sub_mu @ inv_sigma @ x_sub_mu.T


df_prefs = df_tuning.groupby(['cell_num','subject','condition'])['tune_vec'].mean().reset_index()

for i in range(df_prefs.shape[0]):
     
    s = df_prefs.iloc[i].subject
    c = df_prefs.iloc[i].cell_num
    t = df_prefs.iloc[i].condition
    
    ind_s = (df_shuffles.subject==s) & (df_shuffles.cell_num==c) & (df_shuffles.condition==t)
    
    data = df_shuffles[ind_s].tune_vec
    
    dis = mahalanobis(np.array([np.real(df_prefs.loc[i, 'tune_vec']), np.imag(df_prefs.loc[i, 'tune_vec'])]),
                        np.array([np.real(data), np.imag(data)]).T)
    
    
    df_prefs.loc[i, 'dis'] = dis
    
    df_prefs.loc[i, 'p_val_mean_pref'] = 1-chi2.cdf(dis,2)
    
    
df_tuning = df_tuning.merge(df_prefs.drop('tune_vec', axis=1), on = ['cell_num','subject','condition'], how = 'left')



#%% Using shuffles distribution, see if correlation between naive and proficient orientation pref is significant - mean pref

df_shuffles_proficient = df_shuffles[df_shuffles.condition=='proficient'].copy()

df_pref = df_tuning[df_tuning.condition=='naive'].groupby(['cell_num','subject']).agg({'mean_pref_naive' : 'first',
                                                                                       'r_bin_naive' : 'first'}).reset_index()

df_shuffles_proficient = df_shuffles_proficient.merge(df_pref, on = ['cell_num', 'subject'], how = 'left')

circ_corr_mean = np.zeros((n_shuffles,len(df_shuffles_proficient.r_bin_naive.unique())))

for r in df_shuffles_proficient.r_bin_naive.unique():        
    for s in df_shuffles_proficient.shuffle_num.unique():
        
        print(f'r bin {r} and shuffle repeat {s}')
        
        ind = (df_shuffles_proficient.r_bin_naive==r) & (df_shuffles_proficient.shuffle_num==s)
        
        circ_corr_mean[s,r] = c_corr(np.deg2rad(df_shuffles_proficient[ind].mean_pref.to_numpy())*2, 
                                     np.deg2rad(df_shuffles_proficient[ind].mean_pref_naive.to_numpy())*2)



df_pref_mean = df_tuning.groupby(['subject','cell_num'], observed=True).agg({'mean_pref_naive' : 'first',
                                                                             'mean_pref_proficient' : 'first',
                                                                             'r_bin_naive' : 'first'}).reset_index()


real_corr_mean = df_pref_mean.groupby(['r_bin_naive']).apply(lambda x: c_corr(np.deg2rad(x.mean_pref_naive.to_numpy())*2, 
                                                                              np.deg2rad(x.mean_pref_proficient.to_numpy())*2)).reset_index()


z_scores_mean = (real_corr_mean[0].to_numpy() - circ_corr_mean.mean(0))/circ_corr_mean.std(0)


#%% Using shuffles distribution, see if correlation between naive and proficient orientation pref is significant - mean pref binned

df_shuffles_proficient = df_shuffles[df_shuffles.condition=='proficient'].copy()

df_pref = df_tuning[df_tuning.condition=='naive'].groupby(['cell_num','subject']).agg({'pref_bin_naive' : 'first',
                                                                                       'r_bin_naive' : 'first'}).reset_index()

df_shuffles_proficient = df_shuffles_proficient.merge(df_pref, on = ['cell_num', 'subject'], how = 'left')

circ_corr_mean_bin = np.zeros((n_shuffles,len(df_shuffles_proficient.r_bin_naive.unique())))

for r in df_shuffles_proficient.r_bin_naive.unique():        
    for s in df_shuffles_proficient.shuffle_num.unique():
        
        print(f'r bin {r} and shuffle repeat {s}')
        
        ind = (df_shuffles_proficient.r_bin_naive==r) & (df_shuffles_proficient.shuffle_num==s)
        
        circ_corr_mean_bin[s,r] = c_corr(np.deg2rad(df_shuffles_proficient[ind].pref_bin.to_numpy())*2, 
                                         np.deg2rad(df_shuffles_proficient[ind].pref_bin_naive.to_numpy())*2)


df_pref_mean_bin = df_tuning.groupby(['subject','cell_num'], observed=True).agg({'pref_bin_naive' : 'first',
                                                                                 'pref_bin_proficient' : 'first',
                                                                                 'r_bin_naive' : 'first'}).reset_index()

real_corr_mean_bin = df_pref_mean_bin.groupby(['r_bin_naive']).apply(lambda x: c_corr(np.deg2rad(x.pref_bin_naive.to_numpy())*2, 
                                                                             np.deg2rad(x.pref_bin_proficient.to_numpy())*2)).reset_index()

z_scores_mean_bin = (real_corr_mean_bin[0].to_numpy() - circ_corr_mean_bin.mean(0))/circ_corr_mean.std(0)

#%% Using shuffles distribution, see if correlation between naive and proficient orientation pref is significant - modal pref

df_shuffles_proficient = df_shuffles[df_shuffles.condition=='proficient'].copy()

df_pref = df_tuning[df_tuning.condition=='naive'].groupby(['cell_num','subject']).agg({'modal_pref_naive' : 'first',
                                                                                       'r_bin_naive' : 'first'}).reset_index()

df_shuffles_proficient = df_shuffles_proficient.merge(df_pref, on = ['cell_num', 'subject'], how = 'left')

circ_corr_modal = np.zeros((n_shuffles,len(df_shuffles_proficient.r_bin_naive.unique())))

for r in df_shuffles_proficient.r_bin_naive.unique():        
    for s in df_shuffles_proficient.shuffle_num.unique():
        
        print(f'r bin {r} and shuffle repeat {s}')
        
        ind = (df_shuffles_proficient.r_bin_naive==r) & (df_shuffles_proficient.shuffle_num==s)
        
        circ_corr_modal[s,r] = c_corr(np.deg2rad(df_shuffles_proficient[ind].modal_pref.to_numpy())*2, 
                                np.deg2rad(df_shuffles_proficient[ind].modal_pref_naive.to_numpy())*2)



df_pref_modal = df_tuning.groupby(['subject','cell_num'], observed=True).agg({'modal_pref_naive' : 'first',
                                                                              'modal_pref_proficient' : 'first',
                                                                              'r_bin_naive' : 'first'}).reset_index()


real_corr_modal = df_pref_modal.groupby('r_bin_naive').apply(lambda x: c_corr(np.deg2rad(x.modal_pref_naive.to_numpy()*2),
                                                                              np.deg2rad(x.modal_pref_proficient.to_numpy()*2))).reset_index()


z_scores_modal = (real_corr_modal[0].to_numpy() - circ_corr_modal.mean(0))/circ_corr_modal.std(0)


#%% Plot responses over time

%matplotlib qt

df_plot = df_trials.groupby(['cell_num','subject','condition','stim_ori','trial_times'], observed = True)['trial_activity'].mean().reset_index()
df_plot = df_plot.merge(df_tuning[['cell_num', 'subject', 'condition','pref_bin_naive','r_bin_naive']], on = ['cell_num','subject','condition'], how = 'left')

# blank_pref = df_resps.groupby(['cell_num','condition','subject']).agg({'blank_pref' : 'first'}).reset_index()
# blank_pref = blank_pref[blank_pref.condition=='naive'].drop(columns='condition')
# df_plot = df_plot.merge(blank_pref, on = ['cell_num','subject'], how = 'left')

# df_plot = df_plot[(df_plot.stim_ori==90) & (df_plot.blank_pref == False)]


(
    so.Plot(df_plot[df_plot.stim_ori==45], x = 'trial_times', y = 'trial_activity', color = 'pref_bin_naive', linestyle = 'stim_ori')
    .facet(col = 'r_bin_naive', row = 'condition')
    .add(so.Lines(),so.Agg(), legend = False)
    .add(so.Band(),so.Est(errorbar=('se',1)), legend = False)
    .scale(color = 'hls')
    .show()
)

(
    so.Plot(df_plot[df_plot.stim_ori==68], x = 'trial_times', y = 'trial_activity', color = 'pref_bin_naive', linestyle = 'stim_ori')
    .facet(col = 'r_bin_naive', row = 'condition')
    .add(so.Lines(),so.Agg(), legend = False)
    .add(so.Band(),so.Est(errorbar=('se',1)), legend = False)
    .scale(color = 'hls')
    .show()
)

(
    so.Plot(df_plot[df_plot.stim_ori==90], x = 'trial_times', y = 'trial_activity', color = 'pref_bin_naive', linestyle = 'stim_ori')
    .facet(col = 'r_bin_naive', row = 'condition')
    .add(so.Lines(),so.Agg(), legend = False)
    .add(so.Band(),so.Est(errorbar=('se',1)), legend = False)
    .scale(color = 'hls')
    .show()
)


(
    so.Plot(df_plot[df_plot.stim_ori==135], x = 'trial_times', y = 'trial_activity', color = 'pref_bin_naive', linestyle = 'stim_ori')
    .facet(col = 'r_bin_naive', row = 'condition')
    .add(so.Lines(),so.Agg(), legend = False)
    .add(so.Band(),so.Est(errorbar=('se',1)), legend = False)
    .scale(color = 'hls')
    .show()
)


#%% Functions for plotting cell tuning curves and trial rasters

from matplotlib import rcParams

rcParams['image.composite_image'] = False

    
def raster_figs(df_raster, ax, s_on = 0, s_off = 2, sf=1):
    
    '''Creates a two panel figure. 
       Left panel is a heatmap of all trials sorted by stim.
       Right panel is a line plot showing average respones over time by stim.
    '''

    df_raster = df_raster[~np.isinf(df_raster.stim_ori)]
    
    df_raster['stim_ori'] = df_raster['stim_ori'].astype(int)
    
    if ax.size < 3:
        print('Three axes must be provided!')
        return
    
    df_p = pd.pivot(df_raster, index = ['stim_ori','trial_num'], columns = 'trial_times',
                    values = 'trial_activity')
    
    sns.heatmap(df_p, ax = ax.flatten()[0], cmap = 'gray_r', cbar = False, rasterized = True)
    
    ax.flatten()[0].set_yticklabels('')
    ax.flatten()[0].set_yticks([])
    ax.flatten()[0].set_ylabel(r'Stimulus orientation ($\degree$)')
    
    for _, spine in ax.flatten()[0].spines.items():
        spine.set_visible(True)
    
    uni_stim = df_raster.stim_ori.sort_values().unique()
    
    n_repeats = df_p.loc[uni_stim[0]].shape[0]
    
    y_ticks = np.arange(n_repeats/2,n_repeats/2+len(uni_stim)*n_repeats,n_repeats)
    
    ax.flatten()[0].set_yticks(y_ticks)
    ax.flatten()[0].set_yticklabels(uni_stim)
    
    x_labels = np.linspace(df_raster.trial_times.min(), df_raster.trial_times.max(), df_p.shape[1])   
    x_ticks = np.linspace(0.5,df_p.shape[1]-0.5,df_p.shape[1])
    ax.flatten()[0].set_xticks(x_ticks[
        (x_labels == df_raster.trial_times.min()) |
        (x_labels == np.mean([s_on,s_off])) |
        (x_labels == s_on) | 
        (x_labels == s_off) |
        (x_labels == df_raster.trial_times.max())])
   
    ax.flatten()[0].set_xticklabels([int(df_raster.trial_times.min()),s_on,
                                     np.mean([s_on,s_off]).astype(int),s_off,
                                     int(df_raster.trial_times.max())], rotation = 0)
    ax.flatten()[0].set_xlabel('Time rel. to stim. onset (s)')
    
    ax.flatten()[0].hlines(np.linspace(n_repeats,n_repeats*len(uni_stim),len(uni_stim)),
                           xmin=0,xmax=df_p.shape[1],colors = 'black',
                           linestyles = 'solid')
    ax.flatten()[0].vlines([x_ticks[x_labels==s_on],x_ticks[x_labels==s_off]],ymin=0,ymax=df_p.shape[0],
                           colors = 'black',
                           linestyles = 'dashed')
        
    ori_pal = sns.color_palette("husl", 8)
    sns.lineplot(data = df_raster, x = 'trial_times', y = 'trial_activity', hue = 'stim_ori',
                 palette =  ori_pal, errorbar = ('se',1), ax = ax.flatten()[1])
    
    ax.flatten()[1].set_ylabel('Normalized activity (A.U.)')
    ax.flatten()[1].set_xlabel('Time rel. to stim. onset (s)')
    
    handles, labels = ax.flatten()[1].get_legend_handles_labels()
    ax.flatten()[1].legend_.remove()
    ax.flatten()[1].legend(handles, np.ceil(np.arange(0,180,22.5)).astype(int), ncol=1, loc='upper right', 
                frameon=False, borderpad = 0, labelspacing = 0.1, 
                handletextpad = 0.4,borderaxespad = 0.1,
                columnspacing = 1)
    
    ax.flatten()[1].legend_.set_title(r'Stim. ori. ($\degree$)')
    ax.flatten()[1].legend_.set_frame_on(False)    

    ax.flatten()[1].vlines([s_on,s_off],ymin=0,ymax=max(ax.flatten()[1].get_yticks()),
                           colors = 'black',
                           linestyles = 'dashed')
    
    ax.flatten()[1].set_xticks([df_raster.trial_times.min(),s_on,
                                     np.mean([s_on,s_off]).astype(int),s_off,
                                     df_raster.trial_times.max()])
    
    # Make sure all tick labels are float so you have consistent plot sizes
    ax.flatten()[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    sns.despine(ax = ax.flatten()[1], trim = True)
        
    df_mu = df_raster[np.logical_and(df_raster.trial_times > 0, 
                                     df_raster.trial_times <= 2)].groupby(
                                         ['trial_num'], observed = True).agg({'trial_activity' : 'mean',
                                                             'stim_ori' : 'first'})
    df_mu = df_mu.groupby(['stim_ori'], observed = True)['trial_activity'].mean().reset_index()
    
    tc = df_mu.trial_activity.to_numpy()             
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])

    polar_tc_plot(ori_rad,tc,ax.flatten()[2],sf)
    
    cell_info = (df_raster.cell_num.unique()[0], df_raster.subject.unique()[0], df_raster.condition.unique()[0])

    ax.flatten()[2].set_title(cell_info)



def raster_figs_no_lineplot(df_raster, raster_ax, tc_ax, s_on=0, s_off=2, sf=1, tc_color='black'):
    
    '''Creates a two panel figure. 
       Left panel is a heatmap of all trials sorted by stim.
       Right panel is a line plot showing average respones over time by stim.
    '''

    df_raster = df_raster[~np.isinf(df_raster.stim_ori)]
    
    df_raster['stim_ori'] = df_raster['stim_ori'].astype(int)
    
    df_p = pd.pivot(df_raster, index = ['stim_ori','trial_num'], columns = 'trial_times',
                    values = 'trial_activity')
    
    vmax = np.percentile(df_p.to_numpy(),99.5)
    
    sns.heatmap(df_p, ax=raster_ax, cmap = 'gray_r', cbar = False, vmin = 0, vmax = vmax)
    
    raster_ax.set_yticklabels('')
    raster_ax.set_yticks([])
    raster_ax.set_ylabel(r'Stimulus orientation ($\degree$)')
        
    for _, spine in raster_ax.spines.items():
        spine.set_visible(True)
    
    uni_stim = df_raster.stim_ori.sort_values().unique()
    
    n_repeats = df_p.loc[uni_stim[0]].shape[0]
    
    y_ticks = np.arange(n_repeats/2,n_repeats/2+len(uni_stim)*n_repeats,n_repeats)
    
    raster_ax.set_yticks(y_ticks)
    raster_ax.set_yticklabels(uni_stim)
    
    x_labels = np.linspace(df_raster.trial_times.min(), df_raster.trial_times.max(), df_p.shape[1])   
    x_ticks = np.linspace(0.5,df_p.shape[1]-0.5,df_p.shape[1])
    raster_ax.set_xticks(x_ticks[
        (x_labels == df_raster.trial_times.min()) |
        (x_labels == np.mean([s_on,s_off])) |
        (x_labels == s_on) | 
        (x_labels == s_off) |
        (x_labels == df_raster.trial_times.max())])
   
    raster_ax.set_xticklabels([int(df_raster.trial_times.min()),s_on,
                                     np.mean([s_on,s_off]).astype(int),s_off,
                                     int(df_raster.trial_times.max())], rotation = 0)
    raster_ax.set_xlabel('Time rel. to stim. onset (s)')
    
    raster_ax.hlines(np.linspace(n_repeats,n_repeats*len(uni_stim),len(uni_stim)),
                           xmin=0,xmax=df_p.shape[1],colors = 'black',
                           linestyles = 'solid')
    raster_ax.vlines([x_ticks[x_labels==s_on],x_ticks[x_labels==s_off]],ymin=0,ymax=df_p.shape[0],
                           colors = 'black',
                           linestyles = 'dashed')
                
    df_mu = df_raster[np.logical_and(df_raster.trial_times > 0, 
                                     df_raster.trial_times <= 2)].groupby(
                                         ['trial_num'], observed = True).agg({'trial_activity' : 'mean',
                                                             'stim_ori' : 'first'})
    df_mu = df_mu.groupby(['stim_ori'], observed = True)['trial_activity'].mean().reset_index()
    
    tc = df_mu.trial_activity.to_numpy()             
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])

    polar_tc_plot(ori_rad,tc,tc_ax,sf,color=tc_color)
    
    cell_info = (df_raster.cell_num.unique()[0], df_raster.subject.unique()[0], df_raster.condition.unique()[0])

    tc_ax.set_title(cell_info)
    
def polar_tc_plot(ori,tc,ax,sf=1, color='black'):
    
    tc *= sf
    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori)])

    ax.plot(coord[:,0],coord[:,1],color=color)
    
    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax.plot(x_cir,y_cir,'k')
    ax.axis('off')
    ax.text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.plot([-1.2,1.2],[0,0],'--k',zorder = 1)
    ax.plot([0,0],[-1.2,1.2],'--k',zorder = 1)
    
    
    ax.set_aspect(1)
    

def show_cell(stat,meanImg,ax,crop_width = 25):
    
    center = stat['med']
    ypix,xpix = stat['ypix'],stat['xpix']
    
    y_range = (center[0]-int(crop_width/2),center[0]+int(crop_width/2))
    x_range = (center[1]-int(crop_width/2),center[1]+int(crop_width/2))
    
    y_range = np.clip(y_range,0,meanImg.shape[0]).astype(int)
    x_range = np.clip(x_range,0,meanImg.shape[1]).astype(int)
    
    roiMask = np.zeros_like(meanImg)
    
    meanImg = meanImg[y_range[0]:y_range[1],x_range[0]:x_range[1]]
    
    roiMask[ypix,xpix] = 1
    
    roiMask = roiMask[y_range[0]:y_range[1],x_range[0]:x_range[1]]
    
    roi_cmap = plt.get_cmap('Reds')
    roi_cmap.set_under((1,1,1,0))
    cmap_args = dict(cmap=roi_cmap, vmin=0.5, vmax=1)
    
    ax.imshow(meanImg,cmap='Greys_r',rasterized=False)
    ax.imshow(roiMask,alpha=0.4,**cmap_args,rasterized=False)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    



def display_cell_tuning(cell_num,subject,df_trials,df_stats,ops):
    
    fig = plt.figure()
    subfigs = fig.subfigures(2,1)
    # Naive plots
    top_axes = subfigs[0].subplots(1,4)
    # Proficient plots
    bottom_axes = subfigs[1].subplots(1,4)
    
    ind = (df_trials.cell_num==cell_num) & (df_trials.subject==subject) & (df_trials.condition == 'naive') & ~np.isinf(df_trials.stim_ori)
    raster_figs(df_trials[ind], ax = top_axes)
    
    ind = (df_trials.cell_num==cell_num) & (df_trials.subject==subject) & (df_trials.condition == 'proficient') & ~np.isinf(df_trials.stim_ori)
    raster_figs(df_trials[ind], ax = bottom_axes)
    
    df_stat_n = df_stats[(df_stats.subject==subject) & (df_stats.condition == 'naive')].reset_index(drop=True)
    df_stat_n = df_stat_n.iloc[cell_num]
    meanImg_n = ops[[s==subject for s in subjects].index(True),['naive','proficient'].index('naive')][df_stat_n.cell_plane]['meanImg']
    
    show_cell(df_stat_n['stat'],meanImg_n,ax=top_axes.flatten()[-1])
     
    
    df_stat_p = df_stats[(df_stats.subject==subject) & (df_stats.condition == 'proficient')].reset_index(drop=True)
    df_stat_p = df_stat_p.iloc[cell_num]
    meanImg_p = ops[[s==subject for s in subjects].index(True),['naive','proficient'].index('proficient')][df_stat_p.cell_plane]['meanImg']
    
    show_cell(df_stat_p['stat'],meanImg_p,ax=bottom_axes.flatten()[-1])
    

def display_cell_tuning_no_lineplot(cell_num,subject,df_trials,df_stats,ops,figsize=(2,2)):
    
    fig0, raster_tc_sp = plt.subplots(1,3,figsize=figsize, layout='tight')
    # subfigs = fig.subfigures(1,2, wspace=0)
    # rasters and tc
    # raster_tc_sp = subfigs[0].subplots(1,3, sharex=False, sharey=False)
    # mean images
    fig1, cell_sp = plt.subplots(2,1, sharex=True, sharey=True)
    
    cond_colors = sns.color_palette('colorblind',2)
    
    ind = (df_trials.cell_num==cell_num) & (df_trials.subject==subject) & (df_trials.condition == 'naive') & ~np.isinf(df_trials.stim_ori)
    raster_figs_no_lineplot(df_trials[ind], raster_ax = raster_tc_sp[0], tc_ax=raster_tc_sp[2], tc_color=cond_colors[0])
    
    ind = (df_trials.cell_num==cell_num) & (df_trials.subject==subject) & (df_trials.condition == 'proficient') & ~np.isinf(df_trials.stim_ori)
    raster_figs_no_lineplot(df_trials[ind], raster_ax = raster_tc_sp[1], tc_ax=raster_tc_sp[2], tc_color=cond_colors[1])
    
    raster_tc_sp[1].set_yticks([])
    raster_tc_sp[1].set_ylabel('')
    
    df_stat_n = df_stats[(df_stats.subject==subject) & (df_stats.condition == 'naive')].reset_index(drop=True)
    df_stat_n = df_stat_n.iloc[cell_num]
    meanImg_n = ops[[s==subject for s in subjects].index(True),['naive','proficient'].index('naive')][df_stat_n.cell_plane]['meanImg']
    
    show_cell(df_stat_n['stat'], meanImg_n, ax=cell_sp[0])
     
    
    df_stat_p = df_stats[(df_stats.subject==subject) & (df_stats.condition == 'proficient')].reset_index(drop=True)
    df_stat_p = df_stat_p.iloc[cell_num]
    meanImg_p = ops[[s==subject for s in subjects].index(True),['naive','proficient'].index('proficient')][df_stat_p.cell_plane]['meanImg']
    
    show_cell(df_stat_p['stat'], meanImg_p, ax=cell_sp[1])
      
        
    

#%%

%matplotlib qt

# subject = 'SF180515'
subject = 'SF180613'
# subject = 'SF170905B'

n_cells = 5

# ind = (df_tuning.subject == subject) & (df_tuning.pref_bin_naive == 68) & (df_tuning.pref_bin_proficient == 68)
# ind = (df_tuning.subject == subject) & (df_tuning.modal_pref_naive == 90) & (df_tuning.modal_pref_proficient != 90)
# ind = (df_tuning.subject == subject) & (df_tuning.r_bin_naive == 4)

ind = (df_tuning.subject == subject) & (df_tuning.modal_pref_naive == 45) & (df_tuning.modal_pref_proficient == 45) & (df_tuning.r_bin_proficient == 4)


cells = df_tuning[ind].cell_num.unique()

selection = 'random'
# selection = [262,196,156,215,159,111] # SF180613, these numbers come from order when duplicates were not removed
selection = [156,208] # SF180613
# selection = [240,203,107,533,188] # SF180515
# selection = [203,107,533,188]

if selection == 'random':
    cell_nums = np.random.choice(cells, n_cells, replace=False)
elif selection == 'all':
    cell_nums = cells
else:
    cell_nums = selection


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,
                                        "axes.titlesize":5,
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


    for c in cell_nums:

        # display_cell_tuning(c,subject,df_trials[df_trials.train_ind==False],df_stats,ops)
        display_cell_tuning_no_lineplot(c,subject,df_trials[df_trials.train_ind==False],df_stats,ops,figsize=(2.5,1.25))



#%% Tuning curves
import seaborn.objects as so
from seaborn import axes_style

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

%matplotlib qt

color = 'pref_bin_naive'
col = 'r_bin_naive'
y = 'mean_resp'

# g_cells = df_tuning.loc[(df_tuning.condition == 'naive') & df_tuning.sig_tuning, 'cell_num'].unique()

# ind = np.isin(df_tuning.cell_num,g_cells)

# df_plot = df_tuning[df_tuning.p_val < 0.05].copy()

df_plot = df_tuning.copy()

df_plot = df_plot.groupby(['cell_num','condition', 'subject', 'stim_ori', color, col], observed=True).mean().reset_index()

# df_plot = df_plot.groupby(['condition','subject','stim_ori', color, col], observed=True).mean().reset_index()

# g = sns.relplot(df_plot, x = 'stim_ori',y = 'mean_resp', hue = 'pref_bin_naive', col = 'r_bin_naive',
#             row = 'condition', kind = 'line', errorbar = ('se',1), palette = 'hls')


fig =  (
            so.Plot(df_plot, x = 'stim_ori', y = y, color = color)
            .layout(size = (20,9), engine = 'tight')
            .facet(col = col,row='condition')
            .add(so.Line(), so.Agg(),legend=False)
            .add(so.Band(), so.Est(errorbar=('se',1)), legend = False)
            .scale(color='hls',
                x = so.Continuous().tick(every=22.5))
            .theme({**style})
            .label(y = 'Response', x = 'Stimulus orientation')
            .share(x=True, y=False)
            .limit(y=(0,1.1))
            .plot()
        )


for i,a in enumerate(fig._figure.axes):
    a.set_title('')
    a.set_xticklabels(np.ceil(np.array(a.get_xticks())).astype(int))
    a.set_box_aspect(1)
    if i == 0 or i == 5:
        left = False
    else:
        left = True
        a.set_yticks([])
        
    if i < 5:
        bottom = True
        a.set_xticks([])
    else:
        bottom = False
    # if i < 5:
    #     bottom = True
    #     a.set_xticks([])
    # else:
    #     bottom = False

    sns.despine(ax=a, left=left, bottom = bottom, trim = True)

fig

#%% Tuning curves - only cells with significant tuning
import seaborn.objects as so
from seaborn import axes_style

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

%matplotlib qt

color = 'pref_bin_naive'
col = 'r_bin_naive'
# col = 'sig_tuning_naive'
y = 'mean_resp'


# g_cells = df_tuning.loc[(df_tuning.condition == 'naive') & df_tuning.sig_tuning, 'cell_num'].unique()

# ind = np.isin(df_tuning.cell_num,g_cells)

df_plot = df_tuning.copy()

df_sig_tuning_naive = df_plot.loc[df_plot.condition=='naive', ['p_val','cell_num','subject']].copy()

df_sig_tuning_naive = df_sig_tuning_naive.rename(columns = {'p_val' : 'p_val_naive'})

df_sig_tuning_proficient = df_plot.loc[df_plot.condition=='proficient', ['p_val','cell_num','subject']].copy()

df_sig_tuning_proficient = df_sig_tuning_proficient.rename(columns = {'p_val' : 'p_val_proficient'})

df_plot = df_plot.merge(df_sig_tuning_naive[['cell_num','subject','p_val_naive']], on = ['cell_num','subject'], how = 'left')
df_plot = df_plot.merge(df_sig_tuning_proficient[['cell_num','subject','p_val_proficient']], on = ['cell_num','subject'], how = 'left')

# df_plot['sig_tuning'] = (df_plot['p_val_naive'] < 0.05) & (df_plot['p_val_proficient'] < 0.05)

df_plot['sig_tuning'] = (df_plot['p_val_naive'] < 0.05) | (df_plot['p_val_proficient'] < 0.05)

df_plot = df_plot.groupby(['cell_num','condition', 'subject', 'stim_ori', color, col], observed=True).mean().reset_index()

# df_plot = df_plot.groupby(['condition','subject','stim_ori', color, col], observed=True).mean().reset_index()


# g = sns.relplot(df_plot, x = 'stim_ori',y = 'mean_resp', hue = 'pref_bin_naive', col = 'r_bin_naive',
#             row = 'condition', kind = 'line', errorbar = ('se',1), palette = 'hls')


fig0 =  (
            so.Plot(df_plot[(df_plot.sig_tuning==1)], x = 'stim_ori', y = y, color = color)
            .layout(size = (20,9), engine = 'tight')
            .facet(col = col,row='condition')
            .add(so.Line(), so.Agg(),legend=False)
            .add(so.Band(), so.Est(errorbar=('se',1)), legend = False)
            .scale(color='hls',
                x = so.Continuous().tick(every=22.5))
            .theme({**style})
            .label(y = 'Response', x = 'Stimulus orientation')
            .share(x=True, y=False)
            .limit(y=(0,1.1))
            .plot()
        )


# fig1 =   (
#             so.Plot(df_plot[(df_plot.sig_tuning==0)], x = 'stim_ori', y = y, color = color)
#             .layout(size = (20,9), engine = 'tight')
#             .facet(col = col,row='condition')
#             .add(so.Line(), so.Agg(),legend=False)
#             .add(so.Band(), so.Est(errorbar=('se',1)), legend = False)
#             .scale(color='hls',
#                 x = so.Continuous().tick(every=22.5))
#             .theme({**style})
#             .label(y = 'Response', x = 'Stimulus orientation')
#             .share(x=True, y=False)
#             .limit(y=(0,1.1))
#             .plot()
#         )

# for i,a in enumerate(fig._figure.axes):
#     a.set_title('')
#     a.set_xticklabels(np.ceil(np.array(a.get_xticks())).astype(int))
#     a.set_box_aspect(1)
#     if i == 0 or i == 5:
#         left = False
#     else:
#         left = True
#         a.set_yticks([])
#     # if i < 5:
#     #     bottom = True
#     #     a.set_xticks([])
#     # else:
#     #     bottom = False

#     sns.despine(ax=a,left=left, trim = True)

fig0
# fig1


#%% Tuning curves - use crossvalidated curves for naive, all trials for proficient (sorted by naive pref and r)
import seaborn.objects as so
from seaborn import axes_style

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

%matplotlib qt

color = 'pref_bin_naive'
col = 'r_bin_naive'
y = 'mean_resp'

df_plot = df_tuning.groupby(['cell_num','condition','subject','stim_ori', color, col], observed=True).apply(lambda x: x.mean_resp.mean() if np.all(x.condition=='Naive') else x.mean_resp_all_trials.mean()).reset_index()
df_plot.rename(columns={0:'mean_resp'}, inplace=True)


# blank_pref = df_resps.groupby(['cell_num','condition','subject']).agg({'blank_pref' : 'first'}).reset_index()
# blank_pref = blank_pref[blank_pref.condition=='naive'].drop(columns='condition')
# df_plot = df_plot.merge(blank_pref, on = ['cell_num','subject'], how = 'left')

# df_plot = df_plot[df_plot.blank_pref==False]

# df_plot = df_plot.groupby(['condition','subject','stim_ori', color, col], observed=True).mean().reset_index()


# g = sns.relplot(df_plot, x = 'stim_ori',y = 'mean_resp', hue = 'pref_bin_naive', col = 'r_bin_naive',
#             row = 'condition', kind = 'line', errorbar = ('se',1), palette = 'hls')

fig =   (
            so.Plot(df_plot, x = 'stim_ori', y = y, color = color)
            .layout(size = (20,9), engine = 'tight')
            .facet(col = col,row='condition')
            .add(so.Line(), so.Agg(),legend=False)
            .add(so.Band(), so.Est(errorbar=('se',1)), legend = False)
            .scale(color='hls',
                x = so.Continuous().tick(every=22.5))
            .theme({**style})
            .label(y = 'Response', x = 'Stimulus orientation')
            .share(x=True, y=False)
            .limit(y=(0,1.1))
            .plot()
        )


for i,a in enumerate(fig._figure.axes):
    a.set_title('')
    a.set_xticklabels(np.ceil(np.array(a.get_xticks())).astype(int))
    a.set_box_aspect(1)
    if i == 0 or i == 5:
        left = False
    else:
        left = True
        a.set_yticks([])
    # if i < 5:
    #     bottom = True
    #     a.set_xticks([])
    # else:
    #     bottom = False

    sns.despine(ax=a,left=left, trim = True)

fig


#%% Scatter plots for different cell properties, naive vs proficient

import seaborn.objects as so
from seaborn import axes_style

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style['axes.labelsize'] = 10
style['xtick.labelsize'] = 8
style['ytick.labelsize'] = 8
style['legend.fontsize'] = 8
style['legend.title_fontsize'] = 8
style['legend.frameon'] = False

# f,a = plt.subplots(1,3)

df_plot = df_tuning.groupby(['cell_num','condition', 'subject'], observed=True).agg({'mean_pref' : 'first',
                                                                                     'r' : 'first',
                                                                                     'r_naive' : 'first',
                                                                                     'r_bin_naive' : 'first',
                                                                                     'modal_pref' : 'first',
                                                                                     'modal_pref_naive' : 'first',
                                                                                     'pref_bin_naive' : 'first'}).reset_index()
df_plot_color = df_plot[df_plot.condition=='naive'][['r_naive','r_bin_naive']].reset_index(drop=True)

df_plot['mean_pref'] = df_plot['mean_pref'].apply(lambda x: x + 180 if x < 0 else x)

# blank_pref = df_resps.groupby(['cell_num','condition','subject']).agg({'blank_pref' : 'first'}).reset_index()
# blank_pref = blank_pref[blank_pref.condition=='naive'].drop(columns='condition')
# df_plot = df_plot.merge(blank_pref, on = ['cell_num','subject'], how = 'left')

# df_plot = df_plot[df_plot.blank_pref==True]

def ori_label(*x):
    return str(int(np.ceil(x[0])))

def cir_diff(abs_diff):
    if abs_diff > 90:
        abs_diff = np.abs(abs_diff - 180)
    return abs_diff

df_plot_pivot = pd.pivot(df_plot, index = ['cell_num','subject'], columns = 'condition', values = 'mean_pref').reset_index()
df_plot_pivot = pd.concat([df_plot_pivot,df_plot_color],axis=1)
df_plot_pivot['r_bin_naive'] = df_plot_pivot['r_bin_naive'].astype(int)
df_plot_pivot['sin_naive'] = np.sin(np.deg2rad(df_plot_pivot['naive']))
df_plot_pivot['sin_proficient'] = np.sin(np.deg2rad(df_plot_pivot['proficient']))
df_plot_pivot['modal_pref_naive'] = df_plot[df_plot.condition=='naive']['modal_pref'].reset_index(drop=True)
df_plot_pivot['modal_pref_proficient'] = df_plot[df_plot.condition=='proficient']['modal_pref'].reset_index(drop=True)
df_plot_pivot['modal_pref_diff'] = (df_plot_pivot['modal_pref_proficient'] - df_plot_pivot['modal_pref_naive']).abs()
df_plot_pivot['modal_pref_diff'] = df_plot_pivot['modal_pref_diff'].apply(cir_diff)

f = (
        so.Plot(df_plot_pivot, x='sin_naive', y='sin_proficient')
        .layout(size = (4,4),engine='tight')
        .add(so.Dots(), legend = False, color = 'r_bin_naive')
        .add(so.Line(color='black'), so.PolyFit(order=1))
        .theme({**style})
        .scale(color = 'viridis_r',
                x = so.Continuous().tick(every=0.5),
                y = so.Continuous().tick(every=0.5))
        .label(color = 'Naive selectivity (r)',
                x = 'Naive: sin(mean ori. pref.)',
                y = 'Proficient: sin(mean ori. pref.')
        .limit(x=(-0.05,1.05), y = (-0.05,1.05))
        .plot()
    )

for a in f._figure.axes:
    sns.despine(ax=a, trim=True)
    a.plot([0,1],[0,1],'--k')

f

df = df_plot_pivot.groupby(['subject','modal_pref_naive','r_bin_naive'])['modal_pref_diff'].mean().reset_index()

f = (
        so.Plot(df_plot_pivot, y = 'modal_pref_diff', x = 'modal_pref_naive')
        .facet(col = 'r_bin_naive')
        .add(so.Bars(),so.Agg())
        .show()
    )

# (
#     so.Plot(df_plot_pivot, x='naive', y='proficient', color = 'r_bin_naive')
#     .layout(size = (4,4),engine='tight')
#     .facet(col = 'r_bin_naive')
#     .add(so.Dots(), legend = False)
#     .add(so.Line(), so.PolyFit(order=1))
#     .theme({**style})
#     .scale(color = 'viridis_r',
#            x = so.Continuous().tick(every=22.5).label(like=ori_label),
#            y = so.Continuous().tick(every=22.5).label(like=ori_label))
#     .label(color = 'Naive selectivity (r)',
#            x = 'Naive mean ori. pref.',
#            y = 'Proficient mean ori. pref.')
#     .show()
# )



f = (
    so.Plot(df_plot_pivot, x='sin_naive', y='sin_proficient', color = 'r_bin_naive')
    .layout(size = (16.5,4),engine='tight')
    .facet(col = 'r_bin_naive')
    .add(so.Dots(), legend = False)
    .add(so.Line(color='k',linestyle='--'), so.PolyFit(order=1), legend = False)
    .theme({**style})
    .scale(color = 'viridis_r',
           x = so.Continuous().tick(every=0.5),
           y = so.Continuous().tick(every=0.5))
    .label(color = 'Naive selectivity (r)',
            x = 'Naive: sin(mean ori. pref.)',
            y = 'Proficient: sin(mean ori. pref.',
            title = '')
    .limit(x=(-0.05,1.05), y = (-0.05,1.05))
    .plot()
)

for i,a in enumerate(f._figure.axes):
    sns.despine(ax=a, trim=True)
    ind = df_plot_pivot.r_bin_naive==i
    diff = np.abs(df_plot_pivot[ind].proficient-df_plot_pivot[ind].naive)
    prop_diff = np.sum(diff>=23)/np.sum(ind)
    a.set_title(f'{int(np.round(prop_diff*100))}% with large pref. change ') 
    
f




f = (
    so.Plot(df_plot_pivot, x='naive', y='proficient', 
            #color = 'r_bin_naive'
            )
    .layout(size = (16.5,4),engine='tight')
    .facet(col = 'r_bin_naive')
    .add(so.Dots(color='grey'), legend = False)
    # .add(so.Line(color='k',linestyle='--'), so.PolyFit(order=1), legend = False)
    .theme({**style})
    .scale(#color = 'viridis_r',
           x = so.Continuous().tick(at=np.ceil(np.linspace(0,180,9))),
           y = so.Continuous().tick(at=np.ceil(np.linspace(0,180,9))))
    .label(color = 'Naive selectivity (r)',
            x = 'Naive mean ori. pref.',
            y = 'Proficient mean ori. pref.',
            title = '')
    .limit(x=(-1,181), y = (-1,181))
    .plot()
)

for i,a in enumerate(f._figure.axes):
    sns.despine(ax=a, trim=True)
    a.plot([0,90],[90,180], '--k')
    a.plot([90,180],[0,90], '--k')
    a.plot([0,180],[0,180], 'k')
    
f


df_plot_pivot = pd.pivot(df_plot, index = ['cell_num','subject'], columns = 'condition', values = 'modal_pref').reset_index()
df_plot_pivot['modal_diff'] = df_plot_pivot['proficient']-df_plot_pivot['naive']
df_plot_pivot['r_bin_naive'] = df_plot[df_plot.condition=='naive'].r_bin_naive.reset_index(drop=True)
df_plot_pivot['modal_pref_naive'] = df_plot[df_plot.condition=='naive'].modal_pref_naive.reset_index(drop=True)

(
    so.Plot(df_plot_pivot, x='naive', y='proficient', color = 'r_bin_naive')
    .layout(size = (4,4),engine='tight')
    .facet(col = 'r_bin_naive')
    .add(so.Dots(),so.Jitter(x=10,y=10), legend = False)
    .theme({**style})
    .scale(color = 'viridis_r',
           x = so.Continuous().tick(every=22.5).label(like=ori_label),
           y = so.Continuous().tick(every=22.5).label(like=ori_label))
    .label(color = 'Naive selectivity (r)',
           x = 'Naive modal ori. pref.',
           y = 'Proficient modal ori. pref.')
    .show()
)


(
    so.Plot(df_plot_pivot, x='naive', y='diff', color = 'r_bin_naive')
    .layout(size = (4,4),engine='tight')
    .facet(col = 'r_bin_naive')
    .add(so.Dots(),so.Jitter(x=10,y=10), legend = False)
    .theme({**style})
    .scale(color = 'viridis_r',
           x = so.Continuous().tick(every=22.5).label(like=ori_label),
           y = so.Continuous().tick(every=22.5).label(like=ori_label))
    .label(color = 'Naive selectivity (r)',
           x = 'Naive modal ori. pref.',
           y = 'Proficient modal ori. pref.')
    .show()
)


# Simple measure of proportion of cells who changed modal pref from naive to proficient by their naive pref
df = df_plot_pivot.groupby(['modal_pref_naive','r_bin_naive','subject']).apply(lambda x: (x.modal_diff.abs()>0).sum()/len(x)).reset_index()
df['modal_pref_naive'] = df['modal_pref_naive'].astype(int)

(
        so.Plot(df,x = 'r_bin_naive', y = 0, color = 'modal_pref_naive')
        .layout(engine='tight')
        .theme({**style})
        .add(so.Line(), so.Agg(), legend = True)
        # .scale(color = so.Continuous('husl').tick(at=np.ceil(np.arange(0,180,22.5))))
        .scale(color = 'deep')

        .label(color = 'Naive modal ori. pref',
               y = 'Proportion of neurons changing preference',
               x = 'Naive selectivity')
        .show()
)

sns.despine(ax=f._figure.axes[0])

f

# As a heatmap
df_plot_pivot = pd.pivot_table(df, index = 'modal_pref_naive', columns = 'r_bin_naive', values = 0)

f,a = plt.subplots(1,1)
sns.heatmap(df_plot_pivot, cmap = 'magma', vmin = 0, vmax = 1, square = True, ax = a, cbar_kws = {'label' : 'Prop. changed pref.'})
a.set_ylabel('Naive modal ori. pref')
a.set_xlabel('Naive selectivity')

df_plot_pivot = pd.pivot(df_plot, index = ['cell_num','subject'], columns = 'condition', values = 'modal_pref').reset_index()
df_plot_color = df_plot[df_plot.condition=='naive'].modal_pref_naive.reset_index(drop=True)


(
    so.Plot(df_plot_pivot, x='naive', y='proficient', color = df_plot_color)
    .layout(size = (4,4),engine='tight')
    .add(so.Dots(),so.Jitter(x=10,y=10), legend = False)
    .theme({**style})
     .scale(color = 'viridis_r',
           x = so.Continuous().tick(every=22.5).label(like=ori_label),
           y = so.Continuous().tick(every=22.5).label(like=ori_label))
    .label(color = 'Naive selectivity (r)',
           x = 'Naive modal ori. pref.',
           y = 'Proficient modal ori. pref.')
    .show()
)

(
    so.Plot(df_plot_pivot, x='naive', y='proficient', color = df_plot_color)
    .layout(size = (4,4),engine='tight')
    .facet(col = 'subject')
    .add(so.Dots(),so.Jitter(x=10,y=10), legend = False)
    .theme({**style})
     .scale(color = 'viridis_r',
           x = so.Continuous().tick(every=22.5).label(like=ori_label),
           y = so.Continuous().tick(every=22.5).label(like=ori_label))
    .label(color = 'Naive selectivity (r)',
           x = 'Naive modal ori. pref.',
           y = 'Proficient modal ori. pref.')
    .show()
)



df_plot_pivot['diff'] = np.abs(df_plot_pivot['proficient'] - df_plot_pivot['naive'])
df_plot_pivot['diff'] = df_plot_pivot['diff'].apply(cir_diff)

(
    so.Plot(df_plot_pivot, x = 'diff', color = 'naive')
    .facet(col ='naive', wrap = 4)
    .add(so.Bars(), so.Hist(stat = 'probability', common_norm=False))
    .theme({**style})
    .scale(color = so.Continuous('hls').tick(every=22.5).label(like='{x:.0f}'))
    .show()
)


f,a = plt.subplots(1,1)
sns.histplot(df_plot_pivot, x = 'naive', y = 'proficient', stat = 'probability', cmap = 'magma',
             bins = 8, binwidth = 22.5, binrange = (-11.25, 168.75), ax = a, cbar = True, cbar_kws = {'label' : 'Prop. of cells'})
a.set_xticks(np.linspace(0,180-22.5,8))
a.set_xticklabels(np.ceil(a.get_xticks()).astype(int))
a.set_yticks(np.linspace(0,180-22.5,8))
a.set_yticklabels(np.ceil(a.get_yticks()).astype(int))
a.set_xlabel('Naive modal ori. pref.')
a.set_ylabel('Proficient modal ori. pref.')
sns.despine(ax=a, trim=True)
a.set_box_aspect(1)

f,a = plt.subplots(1,1)
sns.displot(df_plot_pivot, x = 'naive', y = 'proficient', kind = 'hist', col = df_plot.r_bin_naive,
             bins = 8, binwidth = 22.5, binrange = (-11.25, 168.75), ax = a, common_norm = False, stat = 'probability')
a.set_xticks(np.linspace(0,180-22.5,8))
a.set_xticklabels(np.ceil(a.get_xticks()).astype(int))
a.set_yticks(np.linspace(0,180-22.5,8))
a.set_yticklabels(np.ceil(a.get_yticks()).astype(int))
a.set_xlabel('Naive modal ori. pref.')
a.set_ylabel('Proficient modal ori. pref.')


df_plot_pivot = pd.pivot(df_plot, index = ['cell_num','subject'], columns = 'condition', values = 'mean_pref').reset_index()

f,a = plt.subplots(1,1)
sns.histplot(df_plot_pivot, x = 'naive', y = 'proficient', cbar = True, stat = 'probability',
             bins = 8, binwidth = 22.5, binrange = (-11.25, 168.75), ax = a)
a.set_xticks(np.linspace(0,180-22.5,8))
a.set_xticklabels(np.ceil(a.get_xticks()).astype(int))
a.set_yticks(np.linspace(0,180-22.5,8))
a.set_yticklabels(np.ceil(a.get_yticks()).astype(int))
a.set_xlabel('Naive mean ori. pref.')
a.set_ylabel('Proficient mean ori. pref.')
sns.despine(ax=a)


df_plot_pivot = pd.pivot(df_plot, index = ['cell_num','subject'], columns = 'condition', values = 'r').reset_index()
df_plot_pivot['r_bin_naive'] = df_plot[df_plot.condition=='naive'].r_bin_naive.reset_index(drop=True)
df_plot_pivot['pref_bin_naive'] = df_plot[df_plot.condition=='naive'].pref_bin_naive.reset_index(drop=True).astype(float).apply(np.ceil).astype(int)
df_plot_pivot['diff'] = df_plot_pivot['proficient']-df_plot_pivot['naive']

(
    so.Plot(df_plot_pivot, x='naive', y='proficient')
    .layout(size = (4,4),engine='tight')
    .add(so.Dots(), legend = True)
    .add(so.Line(color = 'black'), so.PolyFit(order=1), legend = False)
    .scale(color = so.Continuous('husl').tick(at=np.ceil(np.arange(0,180,22.5))).label(like='{x:.0f}'))
    .theme({**style})
    .label(color = 'Mean ori. pref.',
           x = 'Naive selectivity',
           y = 'Proficient selectivity')
    .show()
)
plt.plot([0,1],[0,1],'--k')


f = (
        so.Plot(df_plot_pivot, x='naive', y='proficient',color = 'pref_bin_naive')
        .facet(col='pref_bin_naive', wrap = 4)
        .layout(size = (7,4),engine='tight')
        .add(so.Dots(), legend = True)
        # .add(so.Line(color='black',linestyle='--'), so.PolyFit(order=1), legend = False)
        .theme({**style})
        # .scale(color = so.Continuous('husl').tick(at=np.ceil(np.arange(0,180,22.5))).label(like='{x:.0f}'))
        .scale(color = 'deep')
        .label(color = 'Mean ori. pref.',
            x = 'Naive selectivity',
            y = 'Change in selectivity')
        .plot()
    )

for a in f._figure.axes:
    a.plot([0,1],[0,1],'--k')
    a.set_box_aspect(1)

f

f = (
        so.Plot(df_plot_pivot, x='diff', color = 'pref_bin_naive')
        .facet(col='pref_bin_naive', wrap = 4)
        .layout(size = (7,4),engine='tight')
        .add(so.Line(),so.Hist(stat='probability', common_norm = False))
        .theme({**style})
        .scale(color = 'deep')
        .label(color = 'Mean ori. pref.',
            x = 'Naive selectivity',
            y = 'Change in selectivity')
        .plot()
    )

for a in f._figure.axes:
    a.plot([0,0],[0,0.2],'--k')
    a.set_box_aspect(1)

f


f = (
        so.Plot(df_plot_pivot, x='diff')
        .facet(col='r_bin_naive')
        .layout(size = (7,4),engine='tight')
        .add(so.Line(),so.Hist(stat='probability', common_norm = False))
        .theme({**style})
        .label(color = 'Mean ori. pref.',
            x = 'Naive selectivity',
            y = 'Change in selectivity')
        .plot()
    )

for a in f._figure.axes:
    a.plot([0,0],[0,0.2],'--k')
    a.set_box_aspect(1)

f


f,a = plt.subplots(1,1)
sns.histplot(df_plot_pivot, x = 'naive', y = 'proficient', ax = a)
a.plot([0,1],[0,1],'--k')
a.set_box_aspect(1)
a.set_xlabel('Naive selectivity')
a.set_ylabel('Proficient selectivity')
sns.despine(ax=a)

sns.displot(df_plot_pivot, x = 'naive', y = 'proficient', col = 'pref_bin_naive', col_wrap = 4, kind = 'hist',
            stat = 'probability', common_norm = False)
plt.plot([0,1],[0,1],'--k')



(
    so.Plot(df_plot_pivot, x='naive', y='proficient')
    .facet(col='r_bin_naive')
    .layout(size = (4,4),engine='tight')
    .add(so.Dots(), legend = True)
    .add(so.Line(), so.PolyFit(order=1), legend = False)
    .theme({**style})
    .scale(color = so.Continuous('hls').tick(every=22.5).label(like='{x:.0f}'))
    .label(color = 'ori. pref.',
           x = 'Naive selectivity',
           y = 'Proficient selectivity')
    .show()
)

f = (
        so.Plot(df_plot_pivot, x='naive', y='diff', color = 'pref_bin_naive')
        .facet(col='pref_bin_naive', wrap = 4)
        .layout(size = (7,4),engine='tight')
        .add(so.Dots(), legend = True)
        # .add(so.Line(color='black',linestyle='--'), so.PolyFit(order=4), legend = False)
        .theme({**style})
        # .scale(color = so.Continuous('husl').tick(at=np.ceil(np.arange(0,180,22.5))).label(like='{x:.0f}'))
        .scale(color = 'deep').label(like='{x:.0f}')

        .label(color = 'Mean ori. pref.',
            x = 'Naive selectivity',
            y = 'Change in selectivity')
        .plot()
    )

for a in f._figure.axes:
    a.set_box_aspect(1)
    a.plot([0,1],[0,0],'--k')

f

(
    so.Plot(df_plot_pivot, x='naive', y='diff')
    # .facet(col='pref_bin_naive', wrap = 4)
    .layout(size = (4,4),engine='tight')
    .add(so.Dots())
    .add(so.Line(), so.PolyFit(order=1), legend = False)
    .theme({**style})
    .label(color = 'ori. pref.',
           x = 'Naive selectivity',
           y = 'Change in selectivity')
    .show()
)


(
    so.Plot(df_plot, x='r', color = 'condition')
    .layout(size = (4,4),engine='tight')
    .add(so.Line(),so.Hist(cumulative=True,common_norm = False),legend = False)
    .theme({**style})
    .show()
)



(
    so.Plot(df_plot_pivot, x='diff')
    .layout(size = (4,4),engine='tight')
    .add(so.Bars(),so.Hist(),legend = False)
    .theme({**style})
    .show()
)



#%%

import scipy.stats as stats

import seaborn.objects as so
from seaborn import axes_style

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style['axes.labelsize'] = 10
style['xtick.labelsize'] = 8
style['ytick.labelsize'] = 8
style['legend.fontsize'] = 8
style['legend.title_fontsize'] = 8
style['legend.frameon'] = False

# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y

def xp(x,e):
    y = x**e
    return y


resps = 'mean_resp'

df_plot = df_tuning.copy()

df_plot_pref = df_plot[df_plot.condition=='naive'].groupby(['cell_num','subject'])['pref_bin_naive'].first().reset_index(drop=True).astype(float).apply(np.ceil)
df_plot_modal = df_plot[df_plot.condition=='naive'].groupby(['cell_num','subject'])['modal_pref_naive'].first().reset_index(drop=True)
df_plot_r = df_plot[df_plot.condition=='naive'].groupby(['cell_num','subject'])['r_naive'].first().reset_index(drop=True)

df_plot_pivot = pd.pivot(df_plot, index = ['cell_num','subject'], columns = ['stim_ori','condition'], values = resps).reset_index()


# All cells

f,a = plt.subplots(2,4, figsize = (12,7.5))

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):

    (
        so.Plot(df_plot_pivot[s], x='naive', y='proficient', color = df_plot_pref, pointsize = df_plot_r)
        .layout(size = (4,4),engine='tight')
        .limit(x = (-0.1,1.5),
                y = (-0.1,1.5))
        .add(so.Dots(), legend = False)
        .theme({**style})
        .scale(color = so.Continuous('husl').tick(every=22.5))
        .label(color = 'Naive ori. pref.',
            x = f'Naive resp. to {s}',
            y = f'Proficient resp. to {s}')
        .on(a.ravel()[i])
        .plot()
    )
    
    
for a in a.ravel():
    a.plot([0,1.0], [0,1.0], '--k')
    a.set_box_aspect(1)    
    
f.show()


# Group by cell class

df_plot = df_tuning.copy()

df_plot = df_plot.groupby(['r_bin_naive', 'pref_bin_naive', 'condition', 'stim_ori'])['mean_resp'].mean().reset_index()

df_plot_pref = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].pref_bin_naive
df_plot_r = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].r_bin_naive

df_plot_pivot = pd.pivot(df_plot, index = ['r_bin_naive','pref_bin_naive'], columns = ['stim_ori','condition'], values = 'mean_resp').reset_index()


f,a = plt.subplots(2,4, figsize = (12,7.5))

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):

    (
        so.Plot(df_plot_pivot[s], x='naive', y='proficient', color = df_plot_pref.to_numpy(), marker = df_plot_r.to_numpy())
        .layout(size = (4,4), engine='tight')
        .limit(x = (0,1),
                y = (0,1))
        .add(so.Dots(pointsize=10), legend = False)
        .theme({**style})
        .scale(color = so.Continuous('husl').tick(every=22.5))
        .label(color = 'Naive ori. pref.',
            x = f'Naive resp. to {s}',
            y = f'Proficient resp. to {s}')
        .on(a.ravel()[i])
        .plot()
    )
    
    
for a in a.ravel():
    a.plot([0,1.0], [0,1.0], '--k')
    a.set_box_aspect(1)    

f.show()
    

# With fit

fit_params = np.zeros(8, dtype='object')
ci = np.zeros(8, dtype='object')

f,a = plt.subplots(2,4, figsize = (12,7.5))

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):
    

    # fit_params[i], covar = curve_fit(xp, df_plot_pivot[s].naive, df_plot_pivot[s].proficient, p0=1)
    fit_params[i], covar = curve_fit(pl, df_plot_pivot[s].naive, df_plot_pivot[s].proficient, 
                                 p0=[.5,.5], bounds=((0,0),(1,1)))   
    
    # sigma = np.sqrt(np.diagonal(covar))
    # dof = len(df_plot_pivot[s]['naive']) - len(fit_params[i])
    # t_score = stats.t.ppf(1 - (1 - 0.95) / 2, dof)
    # ci[i] = t_score * sigma
    
    
    (
        so.Plot(df_plot_pivot[s], x='naive', y='proficient', color=np.tile(np.ceil(np.arange(0,180,22.5)),5), marker=np.repeat(np.arange(5),8))
        .layout(size = (4,4),engine='tight')
        .limit(x = (-0.1,1.1),
                y = (-0.1,1.1))
        # .add(so.Dots(), so.Jitter(x=0.05,y=0.05), legend = False)
        # .add(so.Dots(pointsize=10), legend = False)
        .add(so.Dot(pointsize=10, alpha = 1, edgecolor='k'), legend = False)

        .theme({**style})
        .scale(color = so.Continuous('hls').tick(at=np.ceil(np.arange(0,180,22.5))),
               marker = so.Nominal(['X','P','s','v','*']))
        .label(color = 'Naive ori. pref.',
            x = f'Naive resp. to {s.astype(int)}',
            y = f'Proficient resp. to {s.astype(int)}')
        .on(a.ravel()[i])
        .plot()
    )
    
    
for i,a in enumerate(a.ravel()):
    a.plot([0,1], [0,1], '--k')
    x = np.linspace(0,1,1000)
    # a.plot(x,xp(x,*fit_params[i]),'-k')
    a.plot(x,pl(x,*fit_params[i]),'-k')
    # bound_upper = xp(x,*fit_params[i] + ci[i])
    # bound_lower = xp(x,*fit_params[i] - ci[i])
    # bound_upper = pl(x,*fit_params[i] + ci[i])
    # bound_lower = pl(x,*fit_params[i] - ci[i])
    # a.fill_between(x, bound_lower, bound_upper, color='black', alpha = 0.15)        
    sns.despine(ax=a)
    a.set_box_aspect(1)    
    
f.show()



# With fit - subject specific



df_plot = df_tuning.copy()

df_plot = df_plot.groupby(['r_bin_naive', 'pref_bin_naive', 'condition', 'stim_ori', 'subject'])['mean_resp'].mean().reset_index()

df_plot_pref = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].pref_bin_naive
df_plot_r = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].r_bin_naive

df_plot_pivot = pd.pivot(df_plot, index = ['r_bin_naive','pref_bin_naive'], columns = ['subject','stim_ori','condition'], values = 'mean_resp').reset_index()


fit_params = np.zeros((4,8), dtype='object')
ci = np.zeros((4,8), dtype='object')


for mi,m in enumerate(df_plot_pivot['subject'].unique()):

    f,a = plt.subplots(2,4, figsize = (12,7.5))

    for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):


        fit_params[mi,i], covar = curve_fit(xp, df_plot_pivot.loc[df_plot_pivot['subject']==m,s].naive, 
                                                df_plot_pivot.loc[df_plot_pivot['subject']==m,s].proficient, p0=1)
        # fit_params[mi,i], covar = curve_fit(pl, df_plot_pivot.loc[df_plot_pivot['subject']==m,s].naive, 
        #                                         df_plot_pivot.loc[df_plot_pivot['subject']==m,s].proficient, 
        #                                         p0=[.5,.5], bounds=((0,0),(1,1)))   
        
        sigma = np.sqrt(np.diagonal(covar))
        dof = len(df_plot_pivot.loc[df_plot_pivot['subject']==m,s].naive) - len(fit_params[mi,i])
        t_score = stats.t.ppf(1 - (1 - 0.95) / 2, dof)
        ci[mi,i] = t_score * sigma
        
        
        (
            so.Plot(df_plot_pivot.loc[df_plot_pivot['subject']==m,s], 
                    x='naive', y='proficient', color = df_plot_pref, pointsize = df_plot_r)
            .layout(size = (4,4),engine='tight')
            .limit(x = (-0.1,1.5),
                    y = (-0.1,1.5))
            # .add(so.Dots(), so.Jitter(x=0.05,y=0.05), legend = False)
            .add(so.Dots(), legend = False)

            .theme({**style})
            .scale(color = so.Continuous('husl').tick(at=np.ceil(np.arange(0,180,22.5))))
            .label(color = 'Naive ori. pref.',
                x = f'Naive resp. to {s.astype(int)}',
                y = f'Proficient resp. to {s.astype(int)}')
            .on(a.ravel()[i])
            .plot()
        )
        
        
    for i,a in enumerate(a.ravel()):
        a.plot([0,1], [0,1], '--k')
        x = np.linspace(0,1,1000)
        a.plot(x,xp(x,*fit_params[mi,i]),'-k')
        # a.plot(x,pl(x,*fit_params[mi,i]),'-k')
        bound_upper = xp(x,*fit_params[mi,i] + ci[mi,i])
        bound_lower = xp(x,*fit_params[mi,i] - ci[mi,i])
        # bound_upper = pl(x,*fit_params[mi,i] + ci[mi,i])
        # bound_lower = pl(x,*fit_params[mi,i] - ci[mi,i])
        # a.fill_between(x, bound_lower, bound_upper, color='black', alpha = 0.15)        
        sns.despine(ax=a)
        a.set_box_aspect(1)    
        
    f.show()



f,a = plt.subplots(2,4, figsize = (12,7.5), sharey = False)

color_pal = sns.color_palette('husl',8)

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):
# for i,s in enumerate([45,68,90]):

    df = df_plot_pivot[s].copy()
    df['diff'] = df.proficient-df.naive

    (
        so.Plot(df, x='diff')
        .layout(size = (4,4),engine='tight')
        .limit(x = (-1.2,1.2),y = (0,0.4))
        .add(so.Line(color = color_pal[i]),so.Hist(stat = 'probability', bins = 10), legend = False)
        .theme({**style})
        .scale(color = so.Continuous('hls').tick(every=22.5))
        .label(color = 'Naive ori. pref.',
            x = f'Change in response to {int(s)} ori.')
        .on(a.ravel()[i])
        .plot()
    )
    
    a.ravel()[i].plot([0,0],[0,0.4],'--k')
    if i == 0 or i == 4:
        sns.despine(ax=a.ravel()[i])
        a.ravel()[i].set_ylabel('Proportion of neurons')
    else:
        sns.despine(ax=a.ravel()[i], left=True)
        a.ravel()[i].set_yticks([])
        a.ravel()[i].set_ylabel('')
    
    a.ravel()[i].set_box_aspect()


f.show()


f,a = plt.subplots(2,4, figsize = (12,7.5), sharey = False)

color_pal = sns.color_palette('husl',8)

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):
# for i,s in enumerate([45,68,90]):

    df = df_plot_pivot[s].copy()
    df['diff'] = df.proficient-df.naive

    (
        so.Plot(df, x='diff')
        .layout(size = (4,4),engine='tight')
        .limit(x = (-1.2,1.2),y = (0,0.4))
        .add(so.Line(color = color_pal[i]),so.Hist(stat = 'probability', bins = 10), legend = False)
        .theme({**style})
        .scale(color = so.Continuous('hls').tick(every=22.5))
        .label(color = 'Naive ori. pref.',
            x = f'Change in response to {int(s)} ori.')
        .on(a.ravel()[i])
        .plot()
    )
    
    a.ravel()[i].plot([0,0],[0,0.4],'--k')
    if i == 0 or i == 4:
        sns.despine(ax=a.ravel()[i])
        a.ravel()[i].set_ylabel('Proportion of neurons')
    else:
        sns.despine(ax=a.ravel()[i], left=True)
        a.ravel()[i].set_yticks([])
        a.ravel()[i].set_ylabel('')
    
    a.ravel()[i].set_box_aspect()


f.show()



df_plot = df_tuning.copy()

df_n = df_plot[df_plot.condition=='naive'].reset_index(drop=True)
df_p = df_plot[df_plot.condition=='proficient'].reset_index(drop=True)

df_diff = df_n.copy()
df_diff['diff'] = df_p.mean_resp - df_n.mean_resp

df_diff = df_diff.groupby(['subject','stim_ori'])['diff'].mean().reset_index()

f = (
        so.Plot(df_diff, x='stim_ori', y='diff', color='stim_ori')
        .theme({**style})
        .add(so.Range(color='black'), so.Est('mean', errorbar=('se',1)), legend=False)
        .add(so.Dots(), so.Jitter(), legend=False)
        .add(so.Dash(color='black'), so.Agg('mean'), legend=False)
        .scale(color='colorblind', 
               x=so.Continuous().tick(at=np.ceil(np.arange(0,180,22.5))))
        .label(x='Stimulus orientation',
               y='Change in response')
        .plot()
)

xlim = f._figure.axes[0].get_xlim()
f._figure.axes[0].plot(xlim,[0,0],'--k')
sns.despine(ax = f._figure.axes[0], trim = True)

f




# violin plot

f,a = plt.subplots(1,1)

sns.violinplot(df_diff, x='stim_ori', y='diff', palette='colorblind', ax=a)
for s in df_plot_pivot.columns.levels[0][:8]:
    df_plot_pivot[s,'diff'] = df_plot_pivot[s,'proficient'] - df_plot_pivot[s,'naive']

xlim = a.get_xlim()
a.plot(xlim,[0,0],'--k')
sns.despine(ax = a, trim = True)







df_melt = df_plot_pivot.swaplevel(axis='columns')['diff'].melt()
df_melt['stim_ori'] = df_melt['stim_ori'].astype(int)
df_melt['subject'] = df_plot_pivot['subject']

f = (
        so.Plot(df_melt, x = 'stim_ori', y = 'value', color = 'stim_ori')
        .theme({**style})
        # .add(so.Bars(), so.Perc([25,75]))
        .add(so.Range(color = 'black'), so.Est('mean', errorbar = ('se',2)), legend = False)
        .add(so.Dots(),so.Jitter(), legend = False)
        .add(so.Dash(color = 'black'), so.Agg('mean'), legend = False)
        .scale(color = 'colorblind',
                x = so.Continuous().tick(at = np.ceil(np.arange(0,180,22.5))))
        .label(x = 'Stimulus orientation',
                y = 'Change in response')
        .plot()
    )


xlim = f._figure.axes[0].get_xlim()
f._figure.axes[0].plot(xlim,[0,0],'--k')
sns.despine(ax = f._figure.axes[0], trim = True)

f



# f,a = plt.subplots(2,8, figsize = (12,7.5))

# for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):
# # for i,s in enumerate([45,68,90]):

#     df = df_plot_pivot[s].copy()
#     df['diff'] = df.proficient-df.naive

#     (
#         so.Plot(df, x='diff', color = df_plot_modal)
#         .layout(size = (4,4),engine='tight')
#         .limit(x = (-1.3,1.3),y = (0,0.6))
#         .add(so.Line(),so.Hist(stat = 'probability', bins = 10, common_norm = False), legend = False)
#         .theme({**style})
#         .scale(color = so.Continuous('hls').tick(every=22.5))
#         .label(color = 'Naive ori. pref.',
#             x = f'Proficient - Naive resp to {s}')
#         .on(a[0,i])
#         .plot()
#     )
    
#     (
#         so.Plot(df, x='diff', color = df_plot_r)
#         .layout(size = (4,4),engine='tight')
#         .limit(x = (-1.3,1.3),y = (0,0.6))
#         .add(so.Line(),so.Hist(stat = 'probability', bins = 10, common_norm = False), legend = False)
#         .theme({**style})
#         .label(color = 'Naive ori. pref.',
#             x = f'Proficient - Naive resp to {s}')
#         .on(a[1,i])
#         .plot()
#     )

#     a[0,i].plot([0,0],[0,0.6],'--k')
#     a[1,i].plot([0,0],[0,0.6],'--k')

# f.show()


#%% Mixed effects model

import statsmodels.api as sm
import statsmodels.formula.api as smf


df_plot = df_tuning.copy()

df_n = df_plot[df_plot.condition=='naive'].reset_index(drop=True)
df_p = df_plot[df_plot.condition=='proficient'].reset_index(drop=True)

df_diff = df_n.copy()
df_diff['diff'] = df_n.mean_resp - df_p.mean_resp

md = smf.mixedlm("diff ~ C(stim_ori)", df_diff, groups=df_diff['subject'])

mdf = md.fit()

print(mdf.summary())

#45 vs 68
df_diff_45_68 = df_diff[(df_diff.stim_ori == 45) | (df_diff.stim_ori==68)]

md = smf.mixedlm("diff ~ C(stim_ori)", df_diff_45_68, groups=df_diff_45_68['subject'])

mdf = md.fit()

print(mdf.summary())

#90 vs 68
df_diff_90_68 = df_diff[(df_diff.stim_ori == 90) | (df_diff.stim_ori==68)]

md = smf.mixedlm("diff ~ C(stim_ori)", df_diff_90_68, groups=df_diff_90_68['subject'])

mdf = md.fit()

print(mdf.summary())


#45 vs 68
df_diff_45_90 = df_diff[(df_diff.stim_ori == 90) | (df_diff.stim_ori==45)]

md = smf.mixedlm("diff ~ C(stim_ori)", df_diff_45_90, groups=df_diff_45_90['subject'])

mdf = md.fit()

print(mdf.summary())

#%% Fit cells for each subject and look at convexity


def xp(x,e):
    y = x**e
    return y

# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y

fit_params = np.zeros((len(df_tuning.subject.unique()),8))

df_mean = df_tuning.groupby(['cell_num','condition','subject','stim_ori'])['mean_resp'].mean().reset_index()

for si,s in enumerate(df_mean.subject.unique()):
    for oi,o in enumerate(df_mean.stim_ori.unique()):
        fit_params[si,oi] = curve_fit(xp, 
                                      df_mean[(df_mean.subject==s) & (df_mean.stim_ori==o) & (df_mean.condition=='naive')].mean_resp,
                                      df_mean[(df_mean.subject==s) & (df_mean.stim_ori==o) & (df_mean.condition=='proficient')].mean_resp, 
                                      p0=1)[0][0]



fit_params = np.zeros((len(df_tuning.subject.unique()),8),dtype=object)

df_mean = df_tuning.groupby(['cell_num','condition','subject','stim_ori'])['mean_resp'].mean().reset_index()

for si,s in enumerate(df_mean.subject.unique()):
    for oi,o in enumerate(df_mean.stim_ori.unique()):
        fit_params[si,oi] = curve_fit(pl, 
                                      df_mean[(df_mean.subject==s) & (df_mean.stim_ori==o) & (df_mean.condition=='naive')].mean_resp,
                                      df_mean[(df_mean.subject==s) & (df_mean.stim_ori==o) & (df_mean.condition=='proficient')].mean_resp, 
                                      p0=[0.5,0.5], bounds = ((0,0),(1,1)))[0]




#%% Different method, don't it from the fit parameters specifically, but the slope of the fit to neurons with low selectivity


def xp(x,e):
    y = x**e
    return y

# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y

stim_convexity = np.zeros((len(df_tuning.subject.unique()),8))

df_mean = df_tuning.groupby(['cell_num','condition','subject','stim_ori']).agg({'r_bin_naive' : 'first',
                                                                               'pref_bin_naive' : 'first',
                                                                               'mean_resp' : 'first'}).reset_index()

for si,s in enumerate(df_mean.subject.unique()):
    for oi,o in enumerate(df_mean.stim_ori.unique()):

        ind_sub = df_mean['subject'] == s

        ind_pref = (df_mean['r_bin_naive'] == 4) & (df_mean['pref_bin_naive']==o)
        
        x_pref = df_mean[ind_sub & ind_pref & (df_mean['condition']=='naive')]['mean_resp'].mean()
        y_pref = df_mean[ind_sub & ind_pref & (df_mean['condition']=='proficient')]['mean_resp'].mean()

        top = y_pref/x_pref

        x_others = df_mean[ind_sub & (df_mean['pref_bin_naive'] != o) & (df_mean['condition']=='naive')]['mean_resp'].to_numpy().reshape(-1,1)
        y_others = df_mean[ind_sub & (df_mean['pref_bin_naive'] != o) & (df_mean['condition']=='proficient')]['mean_resp'].to_numpy().reshape(-1,1)

        bot,_,_,_ = np.linalg.lstsq(x_others,
                                    y_others,
                                    rcond = None)
        stim_convexity[si,oi] = top/bot[0][0]


#%%

from scipy.optimize import curve_fit

pref = 'modal_pref_naive'
r = 'r_bin_naive'

df_plot = df_tuning.groupby(['condition','stim_ori', pref,r],observed=True)['mean_resp'].mean().reset_index()
df_plot_pref = df_plot[df_plot.condition=='naive'][pref].reset_index(drop=True)
df_plot_r = df_plot[df_plot.condition=='naive'][r].reset_index(drop=True)


df_plot_pivot = pd.pivot(df_plot, index = [pref,r], columns = ['stim_ori','condition'], values = 'mean_resp').reset_index()


f,a = plt.subplots(2,4, figsize = (12,7.5))

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):

    (
        so.Plot(df_plot_pivot[s], x = 'naive', y = 'proficient', color = df_plot_pref, marker = df_plot_r)
        .add(so.Dots())
        .on(a.ravel()[i])
        .limit(x=(0,1),y=(0,1))
        .scale(color = so.Continuous('hls').tick(every=22.5))
        .show()
    )



# nonlinear function to sparsen
def fitfun(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y


# Fit piecewise linear
fit_params = np.zeros(8, dtype = 'object')

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):
    fit_params[i], _ = curve_fit(fitfun,df_plot_pivot[s].naive,
                                             df_plot_pivot[s].proficient,
                                             p0=[.5,.5], bounds=((0,0),(1,1)))
    
    
f,a = plt.subplots(2,4, figsize = (12,7.5))

for i,s in enumerate(np.ceil(np.arange(0,180,22.5))):

    fig = (
            so.Plot(df_plot_pivot[s], x = 'naive', y = 'proficient', color = df_plot_pref, marker = df_plot_r)
            .add(so.Dots())
            .on(a.ravel()[i])
            .limit(x=(0,1),y=(0,1))
            .scale(color = so.Continuous('hls').tick(every=22.5))
            .plot()
          )

    a.ravel()[i].plot(np.linspace(0,1,100), fitfun(np.linspace(0,1,100),*fit_params[i]),'-k')
    a.ravel()[i].plot([0,1],[0,1],'--k')

    fig.show()
    

#%%
from scipy.stats import kurtosis


df_plot = df_resps.groupby(['cell_num','subject','condition','stim_dir'],observed=True)['trial_resps'].mean().reset_index()
df_plot_pivot = pd.pivot(df_plot, index = ['cell_num','subject'], columns = ['stim_dir','condition'], values = 'trial_resps')

def pop_sparseness(fr, kind = 'treves-rolls'):

    if kind.lower() == 'treves-rolls' or  kind.lower() == 'tr':
        # Treves-Rolls
        top = (fr/fr.shape[1]).sum(1)**2
        bottom = (fr**2/fr.shape[1]).sum(1)
        s = 1 - (top/bottom)
    elif kind.lower() == 'kurtosis':
        s = kurtosis(fr,axis = 1)
    elif kind.lower() == 'active':
        sigma = fr.std(1)
        s = (fr < sigma[:,None]).sum(1)/fr.shape[1]
    return s

kind = 'kurtosis'

ps = np.zeros((2,16))

for i,s in enumerate(np.ceil(np.arange(0,360,22.5))):
    ps[0,i] = pop_sparseness(df_plot_pivot[s].naive.to_numpy()[None,:],kind)
    ps[1,i] = pop_sparseness(df_plot_pivot[s].proficient.to_numpy()[None,:],kind)
    
#%% Look at correlation of orientation preference within recordings compared to across recordings

def circ_diff(diff):
    diff[diff>90]= diff[diff>90] - 90
    diff[diff<-90] = diff[diff<-90] + 90
    return diff

def circ_diff_abs(diff):
    diff = np.abs(diff)
    diff[diff>90]= np.abs(90 - diff[diff>90])
    return diff


df = df_resps[~np.isinf(df_resps.stim_ori)].copy()

df['stim_ori_rad'] = df['stim_ori'] * np.pi/90
df['exp(stim_ori_rad)*trial_resp'] = np.exp(df.stim_ori_rad*1j) * df.trial_resps

df = df.groupby(['cell_num','condition', 'subject', 'train_ind']).agg({'exp(stim_ori_rad)*trial_resp':'sum',
                                                                       'trial_resps' : 'sum',
                                                                       'ROI_ret' : 'mean',
                                                                       'ROI_task_ret' : 'mean',
                                                                       'blank_pref' : 'first'})

df['tune_vec'] = df['exp(stim_ori_rad)*trial_resp']/df['trial_resps']
df['r'] = df.tune_vec.abs()
df['mean_pref'] = np.mod(11.25+np.arctan2(np.imag(df.tune_vec),np.real(df.tune_vec))*90/np.pi,180)-11.25


df = df.reset_index()

df_pivot = pd.pivot(df,index = ['cell_num','subject'], columns = ['condition','train_ind'], values = 'mean_pref')
df_pivot['naive_diff_abs'] = circ_diff_abs(df_pivot['naive',True] - df_pivot['naive',False])
df_pivot['proficient_diff_abs'] = circ_diff_abs(df_pivot['proficient',True] - df_pivot['proficient',False])
df_pivot['naive_diff'] = circ_diff(df_pivot['naive',True] - df_pivot['naive',False])
df_pivot['proficient_diff'] = circ_diff(df_pivot['proficient',True] - df_pivot['proficient',False])
df_pivot['cond_diff_abs'] = circ_diff_abs(df_pivot['proficient',True]-df_pivot['naive',True])
df_pivot['cond_diff'] = circ_diff(df_pivot['proficient',True]-df_pivot['naive',True])
df_pivot['r_naive'] = df[(df.condition=='naive') & (df.train_ind == True)].r.to_numpy()
df_pivot['r_proficient'] = df[(df.condition=='proficient') & (df.train_ind == True)].r.to_numpy()
df_pivot['r_bin_naive'] = pd.cut(df_pivot.r_naive, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels=np.arange(5))
df_pivot['r_bin_proficient'] = pd.cut(df_pivot.r_proficient, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels=np.arange(5))
df_pivot['pref_bin_naive'] = pd.cut(df_pivot['naive',True], np.linspace(-11.25,180-11.25,9),
                                    labels = np.ceil(np.arange(0,180,22.5)))
df_pivot['pref_bin_proficient'] = pd.cut(df_pivot['proficient',True], np.linspace(-11.25,180-11.25,9),
                                         labels = np.ceil(np.arange(0,180,22.5)))
df_pivot['blank_pref_naive'] = df[(df.condition=='naive') & (df.train_ind == True)]['blank_pref'].to_numpy().astype(int)
df_pivot['blank_pref_proficient'] = df[(df.condition=='proficient') & (df.train_ind == True)]['blank_pref'].to_numpy().astype(int)

df_pivot['naive',True] = df_pivot['naive',True].apply(lambda x: x+180 if x < 0 else x)
df_pivot['naive',False] = df_pivot['naive',False].apply(lambda x: x+180 if x < 0 else x)

df_pivot['proficient',True] = df_pivot['proficient',True].apply(lambda x: x+180 if x < 0 else x)
df_pivot['proficient',False] = df_pivot['proficient',False].apply(lambda x: x+180 if x < 0 else x)


df_pivot_trials = pd.pivot(df_trials, index = ['subject','cell_num','condition', 'train_ind', 'stim_ori','trial_num'], columns = 'trial_times', values = 'trial_activity')

cr = []

for r in np.unique(df_pivot['r_bin_naive']):
    cr.append(cs.corrcc(np.deg2rad(df_pivot[df_pivot['r_bin_naive']==r]['naive',True].to_numpy()*2),np.deg2rad(df_pivot[df_pivot['r_bin_naive']==r]['naive',False].to_numpy()*2)))



(
    so.Plot(df_pivot, x = 'r_naive', y = 'naive_diff_abs', color = 'blank_pref_naive')
    .add(so.Dots())
    .scale(color = 'colorblind')
    .show()
)


(
    so.Plot(df_pivot, x = 'r_proficient', y = 'proficient_diff_abs', color = 'blank_pref_proficient')
    .add(so.Dots())
    .scale(color = 'colorblind')
    .show()
)

(
    so.Plot(df_pivot, x = 'r_naive', y = 'naive_diff_abs', color = 'blank_pref_naive')
    .add(so.Dots())
    .scale(color = 'colorblind')
    .show()
)


(
    so.Plot(df_pivot, x = 'r_naive', y = 'cond_diff_abs')
    .add(so.Dots())
    .show()
)


(
    so.Plot(df_pivot, x = 'naive_diff', y = 'proficient_diff')
    .add(so.Dots())
)

(
    so.Plot(df_pivot, x = 'naive_diff', y = 'cond_diff', color = 'r_proficient')
    .add(so.Dots())
)


f = (
        so.Plot(df_pivot['naive'], x = True, y = False)
        .layout(size = (12,3))
        .facet(col=df_pivot['r_bin_naive'])
        .add(so.Dots())
        .limit(x = (-5,185), y = (-5,185))
        .plot()
    )

for a in f._figure.axes:
    a.plot([90,180],[0,90], '--k')
    a.plot([0,90],[90,180], '--k')

f


f = (
        so.Plot(df_pivot['proficient'], x = True, y = False)
        .layout(size = (12,3))
        .facet(col=df_pivot['r_bin_proficient'])
        .add(so.Dots())
        .limit(x = (-5,185), y = (-5,185))
        .plot()
    )

for a in f._figure.axes:
    a.plot([90,180],[0,90], '--k')
    a.plot([0,90],[90,180], '--k')

f


(
    so.Plot(df_pivot['proficient'], x = True, y = False, color = df_pivot['pref_bin_proficient'].astype(int))
    .facet(col = df_pivot['r_bin_proficient'])
    .add(so.Dots())
    # .scale(color = so.Continuous('husl').tick(at=np.arange(0,180,22.5)))
    .scale(color = 'colorblind')
    .show()
)


f = (
        so.Plot(df_pivot, x = ('naive',False), y = ('proficient',False), color = df_pivot['pref_bin_naive'].astype(int))
        .layout(size = (12,3))
        .facet(col = df_pivot['r_bin_naive'])
        .limit(x=(-5,185), y=(-5,185))
        .add(so.Dots())
        # .scale(color = so.Continuous('husl').tick(at=np.arange(0,180,22.5)))
        .scale(color = 'colorblind')
        .plot()
    )

for a in f._figure.axes:
    a.plot([90,180],[0,90], '--k')
    a.plot([0,90],[90,180], '--k')

f


(
    so.Plot(df_pivot,x='proficient_diff_abs')
    .facet(col=df_pivot['r_bin_proficient'])
    .add(so.Line(),so.Hist(stat = 'probability', common_norm=False))
    .scale(color = 'colorblind')
    .show()
)

(
    so.Plot(x = df_pivot['pref_bin_proficient'], y = df_pivot['proficient_diff_abs'])
    .facet(df_pivot['r_bin_proficient'])
    .add(so.Line(), so.Agg())
    .add(so.Band(), so.Est(errorbar=('se',1)))
    .show()
)

(
    so.Plot(df_pivot, x = ('naive',True), y = ('proficient',True))
    .facet(col=df_pivot['r_bin_naive'])
    .add(so.Dots())
    .show()
)


df_pivot = pd.pivot(df,index = ['cell_num','subject'], columns = ['condition','train_ind'], values = 'r')
df_pivot['naive_diff'] = df_pivot['naive',False] - df_pivot['naive',True]
df_pivot['proficient_diff'] = df_pivot['proficient',False] - df_pivot['proficient',True]
df_pivot['cond_diff'] = df_pivot['proficient',True]-df_pivot['naive',True]


(
    so.Plot(df_pivot['naive'], x = True, y = False)
    .add(so.Dots())
    .show()
)


(
    so.Plot(df_pivot['proficient'], x = True, y = False)
    .add(so.Dots())
    .show()
)


(
    so.Plot(df_pivot, x = ('naive',True), y = ('proficient',True))
    .add(so.Dots())
    .show()
)


(
    so.Plot(df_pivot, x = ('proficient',True), y = 'proficient_diff')
    .facet(col='subject')
    .add(so.Dots())
    .add(so.Line(),so.PolyFit(order=1))
    .limit(x=(0,1), y=(0,1))
    .show()
)


(
    so.Plot(df_pivot, x = ('naive',True), y = 'cond_diff')
    .facet(col='subject')
    .add(so.Dots())
    .add(so.Line(),so.PolyFit(order=1))
    .limit(x=(0,1))
    .show()
)


(
    so.Plot(df_pivot, x = 'naive_diff', y = 'cond_diff')
    .facet(col='subject')
    .add(so.Dots())
    .add(so.Line(),so.PolyFit(order=1))
    .show()
)


#%%


def circ_diff(diff):
    diff[diff>90]= diff[diff>90] - 90
    diff[diff<-90] = diff[diff<-90] + 90
    return diff

def circ_diff_abs(diff):
    diff = np.abs(diff)
    diff[diff>90]= np.abs(90 - diff[diff>90])
    return diff


df_t = df_resps[~np.isinf(df_resps.stim_ori)].copy()

df_t['stim_ori_rad'] = df_t['stim_ori'] * np.pi/90
df_t['exp(stim_ori_rad)*trial_resp'] = np.exp(df_t.stim_ori_rad*1j) * df_t.trial_resps

df_t = df_t.groupby(['cell_num','condition', 'subject', 'train_ind']).agg({'exp(stim_ori_rad)*trial_resp':'sum',
                                                                         'trial_resps' : 'sum',
                                                                         'ROI_ret' : 'mean',
                                                                         'ROI_task_ret' : 'mean',
                                                                         'blank_pref' : 'first'})

df_t['tune_vec'] = df_t['exp(stim_ori_rad)*trial_resp']/df_t['trial_resps']
df_t['r'] = df_t.tune_vec.abs()
df_t['mean_pref'] = np.mod(11.25+np.arctan2(np.imag(df_t.tune_vec),np.real(df_t.tune_vec))*90/np.pi,180)-11.25

df_t = df_t.reset_index()

df_m = df_resps[~np.isinf(df_resps.stim_ori)].copy()

df_m = df_m.groupby(['cell_num', 'subject', 'condition','train_ind','stim_ori'])['trial_resps'].mean().reset_index()

df_pivot = pd.pivot(df_m,index = ['cell_num','subject'], columns = ['condition','train_ind','stim_ori'], values = 'trial_resps')


df_n_diff = df_pivot['naive',False] - df_pivot['naive',True]
df_n_diff['r'] = df_t[(df_t.condition=='naive') & (df_t.train_ind == True)]['r'].to_numpy()
df_n_diff['mean_pref'] = df_t[(df_t.condition=='naive') & (df_t.train_ind == True)]['mean_pref'].to_numpy()
df_n_diff['pref_bin'] = pd.cut(df_n_diff['mean_pref'], np.linspace(-11.25,180-11.25,9), labels = np.ceil(np.arange(0,180,22.5)))

df_p_diff = df_pivot['proficient',False] - df_pivot['proficient',True]
df_p_diff['r'] = df_t[(df_t.condition=='proficient') & (df_t.train_ind == True)]['r'].to_numpy()
df_p_diff['mean_pref'] = df_t[(df_t.condition=='proficient') & (df_t.train_ind == True)]['mean_pref'].to_numpy()
df_p_diff['pref_bin'] = pd.cut(df_p_diff['mean_pref'], np.linspace(-11.25,180-11.25,9), labels = np.ceil(np.arange(0,180,22.5)))

df_n_diff = df_n_diff.sort_values(['pref_bin','r'])
df_p_diff = df_p_diff.sort_values(['pref_bin','r'])


f,a = plt.subplots(1,2)

sns.heatmap(df_n_diff.iloc[:,:8], ax = a[0], center = 0)
sns.heatmap(df_p_diff.iloc[:,:8], ax = a[1], center = 0)


#%% First half vs second half of recording


df = df_resps.copy()

trial_nums = df.groupby(['subject','condition','cell_num'])['cell_num'].rank(method = 'first', ascending = True)

df['trial_num'] = trial_nums

df['first_half'] = df.groupby(['subject','condition','cell_num'], group_keys = False)['trial_num'].apply(lambda x: x<len(x)/2)

df_t = df[~np.isinf(df.stim_ori)].copy()

df_t['stim_ori_rad'] = df_t['stim_ori'] * np.pi/90
df_t['exp(stim_ori_rad)*trial_resp'] = np.exp(df_t.stim_ori_rad*1j) * df_t.trial_resps

df_t = df_t.groupby(['cell_num','condition', 'subject', 'first_half']).agg({'exp(stim_ori_rad)*trial_resp':'sum',
                                                                          'trial_resps' : 'sum',
                                                                          'ROI_ret' : 'mean',
                                                                          'ROI_task_ret' : 'mean',
                                                                          'blank_pref' : 'first'}).reset_index()

df_t['tune_vec'] = df_t['exp(stim_ori_rad)*trial_resp']/df_t['trial_resps']
df_t['r'] = df_t.tune_vec.abs()
df_t['mean_pref'] = np.mod(11.25+np.arctan2(np.imag(df_t.tune_vec),np.real(df_t.tune_vec))*90/np.pi,180)-11.25

df_t = df_t.sort_values(['subject','cell_num'])

df_pivot = pd.pivot(df_t, index = ['subject','cell_num'], columns = ['condition','first_half'], values = 'mean_pref')
df_pivot['r_naive'] = df_t[(df_t.condition=='naive') & df_t.first_half]['r'].to_numpy()
df_pivot['r_proficient'] = df_t[(df_t.condition=='proficient') & df_t.first_half]['r'].to_numpy()
df_pivot['r_bin_naive'] = pd.cut(df_pivot.r_naive, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels=np.arange(5))
df_pivot['r_bin_proficient'] = pd.cut(df_pivot.r_proficient, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels=np.arange(5))
df_pivot['pref_bin_naive'] = pd.cut(df_pivot['naive',True], np.linspace(-11.25,180-11.25,9),
                                    labels = np.ceil(np.arange(0,180,22.5)))
df_pivot['pref_bin_proficient'] = pd.cut(df_pivot['proficient',True], np.linspace(-11.25,180-11.25,9),
                                         labels = np.ceil(np.arange(0,180,22.5)))
df_pivot['blank_pref_naive'] = df_t[(df_t.condition=='naive') & df_t.first_half]['blank_pref'].to_numpy().astype(int)
df_pivot['blank_pref_proficient'] = df_t[(df_t.condition=='naive') & df_t.first_half]['blank_pref'].to_numpy().astype(int)


icond= 'proficient'

(
    so.Plot(df_pivot[icond], x = True, y = False)
    .facet(col = df_pivot[f'r_bin_{icond}'])
    .add(so.Dots())
    .show()
)

# Look at correlation by r_bin

cr = []

for r in np.unique(df_pivot[f'r_bin_{icond}']):
    cr.append(cs.corrcc(np.deg2rad(df_pivot[df_pivot[f'r_bin_{icond}']==r][icond,True].to_numpy())*2,np.deg2rad(df_pivot[df_pivot[f'r_bin_{icond}']==r][icond,False].to_numpy())*2))


#%% Plot polar tuning curves for tracked neurons, comparing before and after training


n_cells = 1
r_ind = (df_tuning['r_bin_naive']==1) & (df_tuning['subject']=='SF170620B') & (df_tuning['pref_bin_naive']==90) & (df_tuning['pref_bin_proficient']==90)

df_cells = df_tuning[r_ind].cell_num.unique()

rand_cell = np.random.choice(df_cells, n_cells, replace=False)

fig, ax = plt.subplots(2, n_cells, sharex = True, sharey = True, squeeze=False)

for i,c in enumerate(rand_cell):

    ind_naive = r_ind & (df_tuning.cell_num==c) & (df_tuning.condition=='naive')
    tc = df_tuning[ind_naive].mean_resp.to_numpy()
    x_r = np.real(df_tuning[ind_naive].tune_vec.unique())[0]
    y_r = np.imag(df_tuning[ind_naive].tune_vec.unique())[0]
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    pref_rad = ori_rad[np.argmax(tc)]
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])
    x_rad,y_rad = np.cos(ori_rad), np.sin(ori_rad)

    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori_rad)])

    ax[0,i].plot(coord[:,0],coord[:,1],'k')
    ax[0,i].arrow(0,0, x_r, y_r, width = 0.015, length_includes_head = True,
              color = 'g')
    ax[0,i].arrow(0,0,np.cos(pref_rad)*tc.max(),np.sin(pref_rad)*tc.max(),
              width = 0.015,
              length_includes_head = True, color = 'k')
    ax[0,i].plot([-1.2,1.2],[0,0],'--k')
    ax[0,i].plot([0,0],[-1.2,1.2],'--k')
    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax[0,i].plot(x_cir,y_cir,'k')
    ax[0,i].axis('off')
    ax[0,i].text(1.3, 0, r'0$\degree$', verticalalignment='center')
    ax[0,i].text(0, 1.3, r'45$\degree$', horizontalalignment='center')
    ax[0,i].text(-1.5, 0, r'90$\degree$', verticalalignment='center')
    ax[0,i].text(0, -1.4, r'135$\degree$', horizontalalignment='center')
    ax[0,i].set_title(str(c))
    ax[0,i].set_box_aspect(1)


    ind_proficient = r_ind & (df_tuning.cell_num==c) & (df_tuning.condition=='proficient')
    tc = df_tuning[ind_proficient].mean_resp.to_numpy()
    x_r = np.real(df_tuning[ind_proficient].tune_vec.unique())[0]
    y_r = np.imag(df_tuning[ind_proficient].tune_vec.unique())[0]
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    pref_rad = ori_rad[np.argmax(tc)]
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])
    x_rad,y_rad = np.cos(ori_rad), np.sin(ori_rad)

    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori_rad)])

    ax[1,i].plot(coord[:,0],coord[:,1],'k')
    ax[1,i].arrow(0,0,x_r,y_r, width = 0.015, length_includes_head = True,
              color = 'g')
    ax[1,i].arrow(0,0,np.cos(pref_rad)*tc.max(),np.sin(pref_rad)*tc.max(),
              width = 0.015,
              length_includes_head = True, color = 'k')
    ax[1,i].plot([-1.2,1.2],[0,0],'--k')
    ax[1,i].plot([0,0],[-1.2,1.2],'--k')
    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax[1,i].plot(x_cir,y_cir,'k')
    ax[1,i].axis('off')
    ax[1,i].text(1.3, 0, r'0$\degree$', verticalalignment='center')
    ax[1,i].text(0, 1.3, r'45$\degree$', horizontalalignment='center')
    ax[1,i].text(-1.5, 0, r'90$\degree$', verticalalignment='center')
    ax[1,i].text(0, -1.4, r'135$\degree$', horizontalalignment='center')
    ax[1,i].set_title(str(c))
    ax[1,i].set_box_aspect(1)

fig.tight_layout()

# %%
