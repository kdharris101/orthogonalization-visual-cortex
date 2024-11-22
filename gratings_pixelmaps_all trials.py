#%%
"""
Saves df/f pixel maps for all trials

@author: Samuel
"""

import sys
sys.path.append(r'C:\Users\Samuel\OneDrive - University College London\Code\Python\Recordings')
# sys.path.append(r'H:\OneDrive for Business\\Code\Python\Recordings')

from os.path import join
import datetime
import numpy as np
import twophopy as tp
import mpeppy as mp
import pixelmappy as pm
import timelinepy as tl
from sklearn.utils.extmath import randomized_svd
import time
from pathlib import Path

# subjects = ['SF180613']
# subjects_s2p = ['SF180613']
# expt_dates = ['2018-06-28']

# expt_nums = [1]

subjects = ['SF180816']
subjects_s2p = ['SF180816']
expt_dates = ['2018-09-25']
expt_nums = [4]

# results_dir = r'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

save_dir = [join(results_dir, subjects[i], expt_dates[i], 
                   str(expt_nums[i])) for i in range(len(subjects))]

save_paths = [join(save_dir[i], 'drift_pixel_maps_all_trials' 
                 + str(datetime.date.today())) for i in range(len(subjects))]


#%%

n_sv = 20000
svd_flag = False

for i in range(len(subjects)):
    
    expt_info = (subjects[i], expt_dates[i], expt_nums[i])
  
    expt_info_s2p = (subjects_s2p[i], expt_dates[i], expt_nums[i])
    
    print('Load trial stimuli')
    # Load stim sequence from mpep Protocol file
    stimuli = mp.get_sequence(expt_info,['ori'])
    # Transform stimuli to Rigbox/Signals convention
    stimuli = 360 - stimuli
    stimuli[stimuli==360] = 0
    # Change to orientation
    stimuli = stimuli % 180
    
    # Set blank condition (ori = 178) to -1
    stimuli[stimuli == 178] = -1
    
    stim_times = mp.get_stim_times(expt_info)[0]
        
    # Load suite2p ops to find number of planes
    ops = tp.load_suite2p(expt_info_s2p, plane = 0, filetype = 'ops')
    n_planes = ops['nplanes']
    
    # Get experiment numbers
    try:
        exprs = set([int(s.split('\\')[6]) for s in ops['filelist']])
    except:
        exprs = set([int(s.split('/')[-1][0]) for s in ops['filelist']])
    
    # Find the position of the experiment when it was processed by suite2p
    s2_pos = list(exprs).index(expt_nums[i])
    
    timeline = tl.load_timeline(expt_info)
    
    # Get all plane times
    plane_times = tp.get_plane_times(timeline,n_planes)
                
    # Get indices for baseline and response times      
    peri_times = (np.repeat(np.array([-1,2])[None,:], len(stim_times),0) 
                  + stim_times[:,None])
    
    for p in range(0,n_planes):
        
        print('Loading frames for plane %i'%p)
        expt_frames = pm.load_s2p_frames(expt_info_s2p, p, 1)
        frame_shape = expt_frames.shape[:-1]
        expt_frames = expt_frames.reshape(-1,expt_frames.shape[-1])
        
        if svd_flag:
            print('Denoising with SVD')
            start_time = time.time()
            [u,s,vT] = randomized_svd(expt_frames, n_sv)
            print('SVD completed in ' + str(time.time()-start_time))
            print('Reconstructing with only first %i components'%(n_sv))
            expt_frames = np.linalg.multi_dot([u,np.diag(s),vT])
        
        # Remove any negative pixel values
        expt_frames = expt_frames - np.min(expt_frames)
        
        if len(plane_times[p]) > expt_frames.shape[-1]:
            plane_times[p] = plane_times[p][:-1]
            
        # Logical indices for baseline activity and stim responses
        print('Finding indices for baseline and stim-response')
        base_ind = np.concatenate([np.logical_and(plane_times[p] 
                                                      >= peri_times[t,0], 
                               plane_times[p] < stim_times[t])[:,None]
                               for t in range(len(stim_times))], 1) 
                                                                         
        resp_ind = np.concatenate([np.logical_and(plane_times[p] 
                                                          > stim_times[t], 
                               plane_times[p] <= peri_times[t,1])[:,None]
                               for t in range(len(stim_times))], 1) 
                
        # Average activity in pre-stim and post-stim windows
        print('Getting baseline and stim-responses for each trial')
        base_activity = np.dot(expt_frames, base_ind)/np.sum(base_ind,0)
        trial_activity = np.dot(expt_frames, resp_ind)/np.sum(resp_ind,0)
        
        # Free up memory
        del expt_frames
        
        # df/f
        print('Converting trial responses to df/f')
        trial_activity = (trial_activity - base_activity)/base_activity
        
        # Return to image dims
        trial_activity = trial_activity.reshape(frame_shape + (
                                                   trial_activity.shape[-1],))
                
        # Add convexities and stimuli to the array  
        trial_activity = np.tile(trial_activity[:,:,:,None], (1,1,1,2))
        trial_activity[...,1] = stimuli
        
        Path(save_dir[i]).mkdir(parents = True, exist_ok = True)
        np.save(save_paths[i] + f'_plane_{p}',trial_activity)       
        
        # Free up memory
        del trial_activity, base_activity
    
    