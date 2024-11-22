#%%
"""
Created on Fri Jul 24 11:34:53 2020

Loads orientation tuning experiments and pixel map retinotopy and assigns
retinotopy to each ROI and flags it as whether is was located in V1

@author: Samuel Failor
"""

from os.path import join
import glob
import numpy as np
from scipy import ndimage

# results_dir = r'H:/OneDrive for Business/Results'
results_dir = r'C:\Users\Samuel\OneDrive - University College London\Results'
# results_dir = r'C:\Users\samue\OneDrive - University College London\Results'

subjects = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']



expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums_ret = [6, 8, 9, 2, 8, 4, 2, 4, 2, 6]
expt_nums_ori = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]


# subjects = ['SF180613','SF180613']
# expt_dates = ['2018-06-28', '2018-12-12']
# expt_nums_ori = [1,5]
# expt_nums_ret = [2,6]



file_paths_ret = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
            str(expt_nums_ret[i]), r'_'.join([subjects[i], expt_dates[i],
            str(expt_nums_ret[i]), 'peak_pixel_retinotopy*'])))[-1] 
                    for i in range(len(subjects))]

file_paths_ori = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
                    str(expt_nums_ori[i]), r'_'.join([subjects[i], expt_dates[i],
                    str(expt_nums_ori[i]), 'orientation tuning_norm by pref ori_[2021]*'])))[0] 
                                            for i in range(len(subjects))]



#%%

def plane_sm(plane,med_bin = 10, sigma = 60, mode = 'reflect'):
    
    plane_sm = np.zeros_like(plane)
    
    for i,p in enumerate(plane):
        plane_sm[i,...] = ndimage.median_filter(p,med_bin)
        plane_sm[i,...] = ndimage.gaussian_filter(plane_sm[i,...],sigma,
                                                  mode = mode)
        
    return plane_sm

sigma = 50

for f_ret,f_ori,expt_info_ret,expt_info_ori in zip(
    file_paths_ret, 
    file_paths_ori, 
    zip(subjects,expt_dates,expt_nums_ret),
    zip(subjects,expt_dates,expt_nums_ori)):
        
    print('Loading pixel retinotopy ' + f_ret)
    plane_ret = np.load(f_ret)
    
    print('Loading ori tune experiment ' + f_ori)
    expt = np.load(f_ori, allow_pickle = True)[()]
        
    # Average across all planes
    plane_ret = plane_ret.mean(2)
    # Reshape to 2d array for elv and azi
    plane_ret = plane_ret.reshape(2,512,512)
  
    # Smooth with function    
    plane_ret_sm = plane_sm(plane_ret)
    
    task_ret = plane_ret_sm[0,...] + (plane_ret_sm[1,...] + 80) * 1j
    task_ret = np.abs(task_ret)
    
    # Use gradient maps of elevation to find border of V1
    sx_elv = ndimage.sobel(plane_ret_sm[0,...], axis = 0, mode = 'reflect')
    sy_elv = ndimage.sobel(plane_ret_sm[0,...], axis = 1, mode = 'reflect')
    ang_elv = np.arctan2(sx_elv,sy_elv)
    
    sx_azi = ndimage.sobel(plane_ret_sm[1,...], axis = 0, mode = 'reflect')
    sy_azi = ndimage.sobel(plane_ret_sm[1,...], axis = 1, mode = 'reflect')
    ang_azi = np.arctan2(sx_azi,sy_azi)

    # Threshold, i.e. sign map > 0 for V1
    # ang_elv_th = ang_elv > 0
    
    ang_diff = np.sin(ang_elv - ang_azi)
    
    ang_th = np.zeros_like(ang_elv)

    ang_th[ang_diff > 0.31] = 1
    
    # Find largest positive region
    ang_th,_ = ndimage.label(ang_th)
    ang_th = (ang_th == (np.bincount(ang_th.flat)[1:].argmax() + 1))
                      
    nplanes = expt['ops'][0]['nplanes']
    
    expt['ROI_ret'] = np.empty(len(expt['cell_plane']), dtype='complex')
    expt['V1_ROIs'] = np.empty(len(expt['cell_plane']))
    expt['ROI_task_ret'] = np.empty(len(expt['cell_plane']))
    
    expt['plane_ret'] = plane_ret
    expt['plane_ret_sm'] = plane_ret_sm
    expt['ang_elv'] = ang_elv
    expt['ang_azi'] = ang_azi
    expt['V1_map'] = ang_th
    expt['task_ret'] = task_ret
    
    # Find retinotopy of each ROI and flag it as in V1 or not
    
    for p in range(nplanes):
        
        cell_ind = expt['cell_plane'] == p
        plane_stats = expt['stat'][cell_ind]
        ROI_med = np.array([plane_stats[c]['med'] 
                            for c in range(sum(cell_ind))]).astype(int)
        ROI_ret = np.array([plane_ret_sm[:,ROI_med[c,0], ROI_med[c,1]] 
                            for c in range(sum(cell_ind))])
        ROI_ret = ROI_ret[:,0] + ROI_ret[:,1] * 1j
        V1_flag = np.array([ang_th[ROI_med[c,0], ROI_med[c,1]] 
                            for c in range(sum(cell_ind))])
        ROI_task_ret = np.array([task_ret[ROI_med[c,0], ROI_med[c,1]] 
                            for c in range(sum(cell_ind))])
        
        expt['ROI_ret'][cell_ind] = ROI_ret
        expt['ROI_task_ret'][cell_ind] = ROI_task_ret
        expt['V1_ROIs'][cell_ind] = V1_flag
        

    print('Saving new expt file ' + f_ori)
    np.save(f_ori,expt)
    
    
# %%
