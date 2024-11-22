#%%
"""
Created on Fri Jul 10 16:32:20 2020

Processs and saves orientation tuning experiments

@author: Samuel Failor
"""

import sys
sys.path.append(r'C:\Users\samue\OneDrive - University College London\Code\Python\Recordings')


from os.path import join
from pathlib import Path
import glob
import numpy as np
import twophopy as tp
import mpeppy as mp
# import cortex_lab_utils as clu
import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import sklearn.model_selection as skms
import sklearn.metrics as skm
# import sklearn.preprocessing as skp
import scipy.stats as stats
# from scipy.interpolate import interp1d
# import scipy.io as sio
import scipy.linalg as sl
from scipy import ndimage
from skimage.measure import label


subjects = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']

expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]
expt_nums_ret = [6, 8, 9, 2, 8, 4, 2, 4, 2, 6]

# subjects = ['SF180613','SF180613']
# expt_dates = ['2018-06-28', '2018-12-12']
# expt_nums = [1,5]
# expt_nums_ret = [2,6]

# subjects = ['SF190911','SF190911']
# expt_dates = ['2019-09-20','2019-12-12']
# expt_nums = [5,3]
# expt_nums_ret = [6,2]

# subjects = ['M170620B_SF']
# expt_dates = ['2017-07-04']
# expt_nums = [5]
# expt_nums_ret = [6]


# Lab
save_root = r'C:\Users\samue\OneDrive - University College London\Results'
# Home
# save_root = r'C:/Users/Samuel/OneDrive - University College London/Results'

file_paths_ret = [glob.glob(join(save_root, subjects[i], expt_dates[i], 
            str(expt_nums_ret[i]), r'_'.join([subjects[i], expt_dates[i],
            str(expt_nums_ret[i]), 'peak_pixel_retinotopy*'])))[-1] 
                    for i in range(len(subjects))]

# ret_file_path = glob.glob(join(save_root, part[1], part[2], e, f'{part[1]}_{part[2]}_{e}_peak_pixel_retinotopy*'))



#%% Load experiment sessions, process data, and save to save_dirs

save_files = ['_'.join([s,ed,str(en),'orientation tuning_norm by pref ori',
                       str(datetime.date.today())]) 
             for s,ed,en in zip(subjects,expt_dates,expt_nums)]

save_dirs = [join(save_root, s, ed, str(en))  
             for s,ed,en in zip(subjects,expt_dates,expt_nums)]

for i,(expt_info,expt_info_sn) in enumerate(zip(zip(subjects,expt_dates,expt_nums),
                             zip(subjects,expt_dates,expt_nums_ret))):
    
    # Load experiment
    expt = tp.load_experiment(expt_info)
   
    # Load stim sequence from mpep Protocol file
    stimuli = mp.get_sequence(expt_info,['ori'])
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
    
    expt['uni_stim'], expt['stim_counts'] = np.unique(stimuli, 
                                                      return_counts = True)
    expt['stim_dir'] = np.copy(stimuli)
    stimuli[stimuli>=180] -= 180
    expt['stim_ori'] = np.copy(stimuli)
    expt['stim_repeat'] = np.kron(np.arange(0,20),
                                  np.ones(len(np.unique(stimuli))))
    # Get stim times
    expt['stim_times'] = mp.get_stim_times(expt_info)
    
    # Isolate trial responses - make an n_trial by n_cell array
    pre_stim = -1
    post_stim = [0, 2]
    
    # Loop over planes and find pre- and post-stim activity
    
    print('Loading trial responses')
    expt['pre_activity'] = np.array([]).reshape((len(expt['stim_dir']), -1))
    expt['trial_resps'] = np.array([]).reshape((len(expt['stim_dir']), -1))
    
    n_trials = len(expt['stim_times'][0])
    
    for p in range(expt['ops'][0]['nplanes']):
        plane_spks = expt['spks'][expt['cell_plane'] == p, :]
        
        pre_ind = np.concatenate([np.logical_and(expt['plane_times'][p] 
            >= t + pre_stim, expt['plane_times'][p] 
            < t) for t in expt['stim_times'][0]])
        pre_ind = np.reshape(pre_ind,(n_trials, len(expt['plane_times'][p])))
       
        post_ind = np.concatenate([np.logical_and(expt['plane_times'][p] 
            >= t + post_stim[0], expt['plane_times'][p] 
            <= t + post_stim[1]) for t in expt['stim_times'][0]])
        
        post_ind = np.reshape(post_ind,(n_trials, len(expt['plane_times'][p])))
        
        pre_plane = np.divide(np.dot(pre_ind, plane_spks.T).T, 
                              np.sum(pre_ind, axis=1).T).T
        post_plane = np.divide(np.dot(post_ind, plane_spks.T).T, 
                              np.sum(post_ind, axis=1).T).T
    
        
        expt['pre_activity'] = np.concatenate([expt['pre_activity'],
                                               pre_plane], axis = 1)
        
        expt['trial_resps'] = np.concatenate([expt['trial_resps'],
                                              post_plane], axis = 1)

    
    # Make train and test indices
    ind_all = np.arange(len(expt['stim_dir']))
    expt['train_ind'] = np.concatenate([np.where(expt['stim_ori'] == s)[0][::2] 
                            for s in np.unique(expt['stim_ori'])])
    expt['test_ind'] = np.delete(ind_all,expt['train_ind'])    
    
        
    # Mean respones, including blank (last)
    uni_stim = np.unique(expt['stim_ori'])
    
    expt['trial_resps_train_raw'] = expt['trial_resps'][expt['train_ind'],:]
    expt['trial_resps_test_raw'] = expt['trial_resps'][expt['test_ind'],:]
    
    expt['stim_ori_train'] = expt['stim_ori'][expt['train_ind']]
    expt['stim_ori_test'] = expt['stim_ori'][expt['test_ind']]
    
    expt['stim_dir_train'] = expt['stim_dir'][expt['train_ind']]
    expt['stim_dir_test'] = expt['stim_dir'][expt['test_ind']]
    
    mean_stim = np.concatenate([
        np.mean(expt['trial_resps'][expt['stim_ori'] == s,:],
            axis = 0).reshape((1,-1)) for s in uni_stim])
    
    expt['scale_factor'] = np.max(mean_stim,0,keepdims=True)
    
    # Scale by mean preferred stim response
    expt['trial_resps_raw'] = np.copy(expt['trial_resps'])
    expt['trial_resps'] /= expt['scale_factor']
    expt['trial_resps_train'] = expt['trial_resps_train_raw']/expt['scale_factor']
    expt['trial_resps_test'] = expt['trial_resps_test_raw']/expt['scale_factor']
    expt['pre_activity'] /= expt['scale_factor']

    # Find mean, std, and cv of ori response
    uni_ori = np.unique(expt['stim_ori'])   
    expt['mean_ori_train'] = np.concatenate([
        np.mean(expt['trial_resps_train'][expt['stim_ori_train'] == o,:],
            axis = 0).reshape((1,-1)) for o in uni_ori])
    
    expt['mean_ori_test'] = np.concatenate([
        np.mean(expt['trial_resps_test'][expt['stim_ori_test'] == o,:],
            axis = 0).reshape((1,-1)) for o in uni_ori])
    
    expt['mean_ori_all'] = np.concatenate([
        np.mean(expt['trial_resps'][expt['stim_ori'] == o,:],
            axis = 0).reshape((1,-1)) for o in uni_ori])
    
    uni_ori_stim = uni_ori[:-1]
    expt['pref_ori_train'] = uni_ori_stim[np.argmax(expt['mean_ori_train'][:-1,:],
                                                                     axis = 0)]
    
    expt['pref_ori_all'] = uni_ori_stim[np.argmax(expt['mean_ori_all'][:-1,:],
                                                                     axis = 0)]
    
    expt['pref_ori_test'] = uni_ori_stim[np.argmax(expt['mean_ori_test'][:-1,:],
                                                                     axis = 0)]
    expt['std_ori_test'] = np.concatenate([
        np.std(expt['trial_resps_test'][expt['stim_ori_test'] == o,:],
            axis = 0).reshape((1,-1)) for o in uni_ori])

    expt['cv_ori_test'] = np.concatenate([
        stats.variation(expt['trial_resps_test'][expt['stim_ori_test'] == o,:], 
                        axis = 0).reshape((1,-1)) for o in uni_ori])
    
    # Find mean, std, and cv of ori response
    uni_ori = np.unique(expt['stim_dir'])   
    expt['mean_dir_train'] = np.concatenate([
        np.mean(expt['trial_resps_train'][expt['stim_dir_train'] == o,:],
            axis = 0).reshape((1,-1)) for o in uni_ori])
    
    expt['mean_dir_test'] = np.concatenate([
        np.mean(expt['trial_resps_test'][expt['stim_dir_test'] == o,:],
            axis = 0).reshape((1,-1)) for o in uni_ori])
    
    expt['mean_dir_all'] = np.concatenate([
        np.mean(expt['trial_resps'][expt['stim_dir'] == o,:],
            axis = 0).reshape((1,-1)) for o in uni_ori])
    
    
    # # Find mean, std, and cv of dir response
    # uni_dir = np.unique(expt['stim_dir'])   
    # expt['mean_dir'] = np.concatenate([
    #     np.mean(expt['trial_resps'][expt['stim_dir'] == d,:],
    #         axis = 0).reshape((1,-1)) for d in uni_ori])
    
    # uni_dir_stim = uni_dir[:-1]
    # expt['pref_dir'] = uni_dir_stim[np.argmax(expt['mean_dir'][:-1,:],
    #                                                                  axis = 0)]
    # expt['std_dir'] = np.concatenate([
    #     np.std(expt['trial_resps'][expt['stim_dir'] == d,:],
    #         axis = 0).reshape((1,-1)) for d in uni_dir])

    # expt['cv_dir'] = np.concatenate([
    #     stats.variation(expt['trial_resps'][expt['stim_dir'] == d,:], 
    #                     axis = 0).reshape((1,-1)) for d in uni_dir])
      
    # Use circular measures of orientation preference
    
    nb = expt['stim_ori_train'] != np.inf
    
    x = np.sum(np.cos(expt['stim_ori_train'][nb,None]*np.pi/90.)*expt['trial_resps_train'][nb,:],0) \
        / np.sum(expt['trial_resps_train'][nb,:],0)
        
    y = np.sum(np.sin(expt['stim_ori_train'][nb,None]*np.pi/90.)*expt['trial_resps_train'][nb,:],0) \
        / np.sum(expt['trial_resps_train'][nb,:],0)
        
    expt['r'] = np.sqrt(x**2 + y**2)
    expt['th'] = np.mod(11.25+np.arctan2(y, x)*90/np.pi, 180)-11.25
    expt['v_x'] = x
    expt['v_y'] = y

    nb = expt['stim_ori'] != np.inf
    
    x = np.sum(np.cos(expt['stim_ori'][nb,None]*np.pi/90.)*expt['trial_resps'][nb,:],0) \
        / np.sum(expt['trial_resps'][nb,:],0)
        
    y = np.sum(np.sin(expt['stim_ori'][nb,None]*np.pi/90.)*expt['trial_resps'][nb,:],0) \
        / np.sum(expt['trial_resps'][nb,:],0)
        
    expt['r_all'] = np.sqrt(x**2 + y**2)
    expt['th_all'] = np.mod(11.25+np.arctan2(y, x)*90/np.pi, 180)-11.25
    expt['v_x_all'] = x
    expt['v_y_all'] = y
    
    nb = expt['stim_ori_test'] != np.inf
    
    x = np.sum(np.cos(expt['stim_ori_test'][nb,None]*np.pi/90.)*expt['trial_resps_test'][nb,:],0) \
        / np.sum(expt['trial_resps_test'][nb,:],0)
        
    y = np.sum(np.sin(expt['stim_ori_test'][nb,None]*np.pi/90.)*expt['trial_resps_test'][nb,:],0) \
        / np.sum(expt['trial_resps_test'][nb,:],0)
        
    expt['r_test'] = np.sqrt(x**2 + y**2)
    expt['th_test'] = np.mod(11.25+np.arctan2(y, x)*90/np.pi, 180)-11.25
    expt['v_x_test'] = x
    expt['v_y_test'] = y
    
    
    # Use circular measures of direction preference
    
    nb = expt['stim_ori_train'] != np.inf
    
    x = np.sum(np.cos(expt['stim_dir_train'][nb,None]*np.pi/180.)*expt['trial_resps_train'][nb,:],0) \
        / np.sum(expt['trial_resps_train'][nb,:],0)
        
    y = np.sum(np.sin(expt['stim_dir_train'][nb,None]*np.pi/180.)*expt['trial_resps_train'][nb,:],0) \
        / np.sum(expt['trial_resps_train'][nb,:],0)
        
    expt['r_dir'] = np.sqrt(x**2 + y**2)
    expt['th_dir'] = np.arctan2(y, x)*180/np.pi-11.25
    expt['v_x_dir'] = x
    expt['v_y_dir'] = y

    nb = expt['stim_ori'] != np.inf
    
    x = np.sum(np.cos(expt['stim_dir'][nb,None]*np.pi/90.)*expt['trial_resps'][nb,:],0) \
        / np.sum(expt['trial_resps'][nb,:],0)
        
    y = np.sum(np.sin(expt['stim_dir'][nb,None]*np.pi/90.)*expt['trial_resps'][nb,:],0) \
        / np.sum(expt['trial_resps'][nb,:],0)
        
    expt['r_dir_all'] = np.sqrt(x**2 + y**2)
    expt['th_dir_all'] = np.arctan2(y, x)*180/np.pi-11.25
    expt['v_x_dir_all'] = x
    expt['v_y_dir_all'] = y
    
    nb = expt['stim_ori_test'] != np.inf
    
    x = np.sum(np.cos(expt['stim_dir_test'][nb,None]*np.pi/180.)*expt['trial_resps_test'][nb,:],0) \
        / np.sum(expt['trial_resps_test'][nb,:],0)
        
    y = np.sum(np.sin(expt['stim_dir_test'][nb,None]*np.pi/180.)*expt['trial_resps_test'][nb,:],0) \
        / np.sum(expt['trial_resps_test'][nb,:],0)
        
    expt['r_dir_test'] = np.sqrt(x**2 + y**2)
    expt['th_dir_test'] = np.mod(11.25+np.arctan2(y, x)*180/np.pi, 180)-11.25
    expt['v_x_dir_test'] = x
    expt['v_y_dir_test'] = y
       
    # How excited are cells by the stimulus? 
    uni_dir = np.unique(expt['stim_dir'])[:-1]
    stim_ind = [np.equal(expt['stim_dir'], d).reshape((-1,1)) for d in uni_dir]
    stim_ind = np.concatenate(stim_ind, axis = 1)

    stim_mu = (np.dot(stim_ind.T, expt['trial_resps'])/np.sum(stim_ind, 
                                            axis = 0).reshape((-1,1)))

    pre_STD = np.std(expt['pre_activity'])
    pre_mu = np.mean(expt['pre_activity'])
    
    expt['z_stim'] = (stim_mu - pre_mu)/pre_STD
    
    # A cell is excited if a stimulus mean reasponse is 0.5 SD from baseline mean
    expt['vis_cells'] = np.any(expt['z_stim'] > 0.5, axis = 0)

      
    # Linear regression - Orientation
    model_stim = np.copy(expt['stim_ori'])
    model_stim[model_stim == np.inf] = -1
    lb = preprocessing.LabelBinarizer()
    uni_stim = np.unique(model_stim)
    lb.fit(uni_stim)
    dm = lb.transform(model_stim)
    # Remove blank column so it is intercept 
    dm = dm[:,1:] 
    
    # Average responses to each stimulus
    lr = LinearRegression()
    lr.fit(dm, expt['trial_resps'])
    pred_y_full = lr.predict(dm)
        
    skf = skms.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    
    pred_y = skms.cross_val_predict(LinearRegression(), dm,  expt['trial_resps'], 
                cv = skf.split(dm, model_stim))
    
    expt['cv_ve_ori'] = skm.r2_score(expt['trial_resps'], pred_y,
                                  multioutput = 'raw_values')
    
    expt['full_ve_ori'] = skm.r2_score(expt['trial_resps'], pred_y_full,
                                    multioutput = 'raw_values')
    
    # Add ROI retinotopy
    print('Loading pixel retinotopy ' + file_paths_ret[i])
    plane_ret = np.load(file_paths_ret[i])
    
    # Average across all planes
    plane_ret = plane_ret.mean(2)
    # Reshape to 2d array for elv and azi
    plane_ret = plane_ret.reshape(2,512,512)
    
    # Smooth averaged map
    plane_ret_sm = np.append(
    ndimage.gaussian_filter(plane_ret[0,...],50)[None,...],
    ndimage.gaussian_filter(plane_ret[1,...],50)[None,...],
    axis = 0)
    
    # Use gradient maps of elevation to find border of V1
    sx_elv = ndimage.sobel(ndimage.gaussian_filter(plane_ret[0,...],50), 
                           axis = 0, mode = 'constant')
    sy_elv = ndimage.sobel(ndimage.gaussian_filter(plane_ret[0,...],50), 
                           axis = 1, mode = 'constant')
    ang_elv = np.arctan2(sy_elv,sx_elv)
    
    # Threshold, i.e. sign map > 0 for V1
    ang_elv_th = ang_elv > 0
    
    # Only include largest patch
    # labels = label(ang_elv_th)
    # uni_labels = np.unique(labels)
    
    # label_sign = np.array([np.mean(ang_elv_th[labels==l]) 
    #                        for l in uni_labels])
    # label_sz = np.array([np.sum(labels==l) for l in uni_labels])
    
    # pos_labels = uni_labels[label_sign>0]
    
    # V1_label = pos_labels[np.argmax(label_sz[label_sign>0])]
    
    # ang_elv_th = labels == V1_label
                    
    nplanes = expt['ops'][0]['nplanes']
    
    expt['ROI_ret'] = np.empty(len(expt['cell_plane']),dtype=complex)
    expt['V1_ROIs'] = np.empty(len(expt['cell_plane']))
    
    expt['plane_ret'] = plane_ret
    expt['plane_ret_sm'] = plane_ret_sm
    expt['ang_elv'] = ang_elv
    expt['V1_map'] = ang_elv_th
    
    # Find retinotopy of each ROI and flag it as in V1 or not
    for p in range(nplanes):
        
        cell_ind = expt['cell_plane'] == p
        plane_stats = expt['stat'][cell_ind]
        ROI_med = np.array([plane_stats[c]['med'] 
                            for c in range(sum(cell_ind))]).astype(int)
        ROI_ret = np.array([plane_ret_sm[:,ROI_med[c,0], ROI_med[c,1]] 
                            for c in range(sum(cell_ind))])
        ROI_ret = ROI_ret[:,0] + ROI_ret[:,1] * 1j
        V1_flag = np.array([ang_elv_th[ROI_med[c,0], ROI_med[c,1]] 
                            for c in range(sum(cell_ind))])
        
        expt['ROI_ret'][cell_ind] = ROI_ret
        expt['V1_ROIs'][cell_ind] = V1_flag
    
    
    # Save processed data
    print('Saving results in ' + join(save_dirs[i],save_files[i]))
    
    Path(save_dirs[i]).mkdir(parents = True, exist_ok = True)
    np.save(join(save_dirs[i],save_files[i]),expt)
    
# %%
