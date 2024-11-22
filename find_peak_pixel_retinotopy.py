#%%
"""
Created on Thu Jul 23 12:14:59 2020

Loads pixel rfs and finds the preferred retinotopic location of each pixel

@author: Samuel Failor
"""

import sys
sys.path.append(r'C:\Users\samue\OneDrive - University College London\Code\Python\Recordings')
# sys.path.append(r'C:\Users\Samuel\OneDrive - University College London\Code\Python\Recordings')


from os.path import join
import glob
import numpy as np
import scipy.ndimage as si
import datetime
import mpeppy as mp

results_dir = 'C:/Users/samue/OneDrive - University College London/Results'
# results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

# subjects = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
#             'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
#             'SF180613']

# expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
#               '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
#               '2018-06-28', '2018-12-12']

# expt_nums = [6, 8, 9, 2, 8, 4, 2, 4, 2, 6]

# subjects = ['SF007']
# expt_dates = ['2023-07-18']

# expt_nums = [5]

subjects = ['SF010','SF010']
expt_dates = ['2024-01-23','2024-03-28']
expt_nums = [5,4]

file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
                  str(expt_nums[i]), r'_'.join([subjects[i], 
                  expt_dates[i], str(expt_nums[i]), 
                  'pixel_retinotopy*'])))[-1] 
                                           for i in range(len(subjects))]

save_files = ['_'.join([s,ed,str(en),'peak_pixel_retinotopy',
                   str(datetime.date.today())]) 
         for s,ed,en in zip(subjects,expt_dates,expt_nums)]
save_dirs = [join(results_dir, s, ed, str(en))  
             for s,ed,en in zip(subjects,expt_dates,expt_nums)]

#%%

rf_sigma = 3

for e,(f,expt_info) in enumerate(zip(file_paths,zip(subjects,expt_dates,expt_nums))):

    print('Loading pixel rfs in ' + f)
    plane_rfs = np.load(f, allow_pickle = True)
    
    stim_info = mp.load_protocol(expt_info)

    par_names = stim_info['parnames']
    
    azi_0 = stim_info['pars'][par_names == 'x1']/10 
    azi_1 = stim_info['pars'][par_names == 'x2']/10
    
    # elv_0 = stim_info['pars'][par_names == 'y1']/10
    # elv_1 = stim_info['pars'][par_names == 'y2']/10
    
    elv_0 = -35
    elv_1 = 35
    
    # deg_per_sq = (azi_0-azi_1)/plane_rfs[0].shape[1]
    deg_per_sq = (azi_0-azi_1)/plane_rfs[0].shape[-1]


    azi_coord = np.linspace(azi_0+deg_per_sq/2, 
                     azi_1-deg_per_sq/2,plane_rfs[0].shape[1]).reshape(-1,)
    elv_coord = np.linspace(elv_0+deg_per_sq/2, 
                     elv_1-deg_per_sq/2,plane_rfs[0].shape[0]).reshape(-1,)
    
    # Find peak of each pixel RF
        
    rf_peak = np.empty((2,)+(plane_rfs[0].shape[2]*plane_rfs[0].shape[3],
                             len(plane_rfs)))
    
    # rf_peak = np.empty((2,)+(plane_rfs[0].shape[0]*plane_rfs[0].shape[1],
    #                          len(plane_rfs)))
    
    for p in range(len(plane_rfs)):
        print('Finding peak of pixel RFs in plane ' + str(p))
        for i,rf in enumerate(np.rollaxis(plane_rfs[p].reshape(
                plane_rfs[p].shape[0:2] + (-1,)),2)):
        # for i,rf in enumerate(plane_rfs[p].reshape((-1,) + plane_rfs[p].shape[2:])):
            rf_sm = si.gaussian_filter(rf,rf_sigma)
            # rf_peak[0,i,p],rf_peak[1,i,p] = np.where(rf_sm == np.amax(rf_sm))
            rf_peak[0,i,p],rf_peak[1,i,p] = np.unravel_index(np.argmax(rf_sm),rf_sm.shape)
        
    
    peak_plane_ret = np.array([elv_coord[rf_peak[0,...].astype(int)],
                               azi_coord[rf_peak[1,...].astype(int)]])    
    
    np.save(join(save_dirs[e],save_files[e]),peak_plane_ret)

    
# %%
