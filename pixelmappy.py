# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:02:28 2020

Library of functions for loading and analyzing tifs from imaging experiemnts 

@author: Samuel Failor
"""
from os.path import join
import glob
import scipy.io as sio
import numpy as np
import twophopy as tp
import timelinepy as tl
import mpeppy as mp
import cortex_lab_utils as clu
import tifffile as tf
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd
from skimage.transform import rescale
import time


def load_bin(dir_path, expt_frames, scale=1):

    file_name = join(dir_path, 'data.bin')

    # Open the file in binary mode
    with open(file_name,'rb') as f:
        # Read the data into a NumPy array
        image_frames = np.fromfile(f, dtype=np.int16)  # Change dtype according to your data
    
    image_frames = image_frames.reshape(-1,512,512)
    # image_frames = np.transpose(image_frames, [1,2,0])
    image_frames = image_frames[expt_frames,:,:]
    # image_frames = image_frames[:,:,expt_frames]
    
    if scale != 1:
        # return rescale(image_frames,[scale,scale,1])
        return rescale(image_frames,[1,scale,scale])

    else:
        return image_frames
    

def load_tiffs(dir_path, expt_frames, scale=1):

    tif_path = join(dir_path,'reg_tif')
    
    # List of all tifs
    tifs = glob.glob(join(tif_path, '*.tif'))
     
    # Get frames per tif
    frames_per_tif = np.empty(len(tifs), dtype = int)   
    for i,f in enumerate(tifs):
        with tf.TiffFile(f) as tif:
            if i == 1:
                # breakpoint()
                if len(tif.series[0].shape) > 2:
                    tif_dims = tif.series[0].shape[1:]
                else:
                    tif_dims = tif.series[0].shape
            frames_per_tif[i] = len(tif.pages)
    
    # Find first and last tif to load from tif directory
    cml_frames = np.cumsum(frames_per_tif).astype(int)
    
    first_tif = np.where((cml_frames - expt_frames[0]) >= 0)[0]
    first_tif = first_tif[(cml_frames - expt_frames[0])[first_tif].argmin()]
    
    last_tif = np.where((cml_frames - expt_frames[-1]) >= 0)[0]
    last_tif = last_tif[(cml_frames - expt_frames[-1])[last_tif].argmin()]
    
    # Index of tifs to load
    tif_ind = np.arange(first_tif,last_tif+1)
    
    # Tifs containing frames of interest
    tifs_to_load = np.array(tifs)[tif_ind]  
    
    # Find index of expt frames for each tif
    cml_frames = cml_frames[tif_ind]
    frames_per_tif = frames_per_tif[tif_ind]
        
    # Create dictionary for each tif that contains index of frames to load
    tif_dict = {f : dict() for f in tifs_to_load}
    
    for i,tif in enumerate(tifs_to_load):
        frames = np.arange(cml_frames[i]-frames_per_tif[i],
                           cml_frames[i]).astype(int)
        frame_ind = (frames >= expt_frames[0]) & (frames <= expt_frames[-1])
        tif_dict[tif]['tif_frames'] = frames[frame_ind] - (cml_frames[i]
                                                         - frames_per_tif[i])
        tif_dict[tif]['expt_frames'] = frames[frame_ind] - expt_frames[0]
        
    # Load frames 
    # breakpoint()
    # image_frames = np.zeros([round(i*scale) for i in tif_dims] + 
    #                         [len(expt_frames)], dtype = np.float32)
    image_frames = np.zeros([len(expt_frames)] + [round(i*scale) for i in tif_dims], dtype = np.float32)
    
    for t in tifs_to_load:
        # Load tifs pages for the experiment
        print('Loading ' + t)
        # pages = tf.imread(t, key = tif_dict[t]['tif_frames']).transpose(1,2,0)
        pages = tf.imread(t, key = tif_dict[t]['tif_frames'])

        
        # Rescale
        if scale != 1:
            for i,f in enumerate(tif_dict[t]['expt_frames']):
                # image_frames[...,f] = rescale(pages[...,i], scale,
                #                                   preserve_range = True,
                #                                   anti_aliasing = True)
                image_frames[f,...] = rescale(pages[i,...], scale,
                                                  preserve_range = True,
                                                  anti_aliasing = True)
        else:
            # breakpoint()
            # image_frames[...,tif_dict[t]['expt_frames']] = pages
            image_frames[tif_dict[t]['expt_frames'],...] = pages
            
        
    return image_frames


def load_s2p_frames(expt_info, plane, file_type='bin', scale=1):
    
    '''
        
    Loads tif frames for a specific experiment from tifs processed by suite2p

    Parameters
    ----------
    expt_info : tuple
        Subject, experiment date, experiment number.
    plane : int
        Plane number
    scale : float
        Image scaling factor 

    Returns
    -------
    image_frames : array
        Array containing all imaged frames, shape = (y axis, x axis, n frames).

    '''
    suite2p_path = clu.find_expt_file(expt_info,'suite2p')
    dir_path = tp.suite2p_plane_path(suite2p_path,plane)
    
    # Load suite2p metadata
    ops = tp.load_suite2p(expt_info, plane = plane, filetype = 'ops')
    
    # Get experiment numbers
    try:
        exprs, idx = np.unique([int(s.split('\\')[1]) for s 
                     in ops['filelist']], return_index=True)
        exprs = exprs[np.argsort(idx)]        
    except:
        try:
            exprs, idx = np.unique([int(s.split('/')[-1][0]) for s 
                         in ops['filelist']], return_index=True)
            exprs = exprs[np.argsort(idx)]
        except:
            exprs, idx = np.unique([int(s.split('\\')[-2][0]) for s 
                         in ops['filelist']], return_index=True)
            exprs = exprs[np.argsort(idx)]
    
    # Find the position of the experiment when it was processed by suite2p
    suite2p_pos = list(exprs).index(expt_info[2])
    
    # Frame index for the experiment
    expt_frames = tp.find_expt_frames(suite2p_pos, 
                  np.cumsum(ops['frames_per_folder']))

    if file_type == 'tif':
        image_frames = load_tiffs(dir_path, expt_frames, scale)
    elif file_type == 'bin':
        image_frames = load_bin(dir_path, expt_frames, scale)
        
    return image_frames    
 

def pixel_rfs(expt_info, plane, file_type='bin', bl_win = [-1,0], resp_win = [0.2,0.6], 
              n_dims = 100, img_scale = 1, alphas = [100]):
        
    if type(alphas) is int:
        alphas = [alphas]
    
    # Get ops for recording
    timeline = tl.load_timeline(expt_info)
    ops = tp.load_suite2p(expt_info, plane = plane, filetype = 'ops')
        
    plane_times = tp.get_plane_times(timeline, ops['nplanes'])[plane]
    
    # Load all photodiode flips (i.e. stimulus update times)
    print('Loading stim times.')
    _, update_times = mp.get_stim_times(expt_info, all_flips = True)
       
    update_times = update_times[0]
        
    # Load stimulus
    print('Loading stimulus used for receptive field mapping.')
    dirpath = clu.find_expt_file(expt_info,'root')
    # Build filepath string
    stim_path = glob.glob(join(dirpath, 'sparse_noise_stimulus*'))[-1]
    stim = sio.loadmat(stim_path, squeeze_me = True)['stim']
    stim_frames = np.concatenate([i[None,...] for i in stim['frames'][()]], 
                                 axis = 0)
    stim_shape = stim_frames.shape[1:]
    # Flatten and transpose
    stim_frames = stim_frames.reshape(stim_frames.shape[0],-1)

    sequence = stim['sequence'][()]
    
    # Only look at times when stimulus changed
    if len(update_times) > len(sequence):
        l_dif = len(update_times) - len(sequence)
        update_times = update_times[:-l_dif]
    
    stim_times = update_times[np.diff(sequence, prepend = 0) != 0]
    
    # Find periods to average responses
    if type(bl_win) is list:
        bl_win = np.array(bl_win)
    if type(resp_win) is list:
        resp_win = np.array(resp_win)
       
    bl_times = (np.repeat(bl_win.reshape(1,-1), len(stim_times), 
                            axis = 0) + stim_times.reshape(-1,1))
    resp_times = (np.repeat(resp_win.reshape(1,-1), len(stim_times), 
                            axis = 0) + stim_times.reshape(-1,1))    
    
    # Load tifs for experiment and plane
    print('Loading frames for plane ' + str(plane))
    image_frames = load_s2p_frames(expt_info, plane, file_type, img_scale)
    # Save shape for converting back to 2d
    # image_shape = image_frames.shape[:-1]
    image_shape = image_frames.shape[1:]

    # Flatten and transpose for SVD
    # image_frames = image_frames.transpose(2,0,1)
    image_frames = image_frames.reshape(image_frames.shape[0],-1)
                                          
    # Correct for possible dropped frame
    if len(plane_times) > len(image_frames):
        l_dif = len(plane_times) - len(image_frames)
        plane_times = plane_times[:-l_dif]
        
        
    # Indices for baseline activity and stim responses   
    base_ind = np.concatenate([np.logical_and(plane_times >= bl_times[t,0], 
                               plane_times < bl_times[t,1]).reshape(-1,1)
                               for t in range(len(stim_times))], axis = 1) 
                                                                         
    resp_ind = np.concatenate([np.logical_and(plane_times > resp_times[t,0], 
                               plane_times <= resp_times[t,1]).reshape(-1,1)
                               for t in range(len(stim_times))], axis = 1)     
        
    # SVD to find dimensions for regression
    print('Carrying out SVD on imaging data.')
    image_frames = zscore(image_frames)
    # [u,s,v] = svd(image_frames, full_matrices = False)
    start_time = time.time()
    [u,s,v] = randomized_svd(image_frames.T, n_dims, random_state = 1)
    print('SVD completed in ' + str(time.time()-start_time))
    
    # Consider any change in pixels
    stim_abs = np.abs(np.diff(stim_frames, axis = 0, prepend = 0))
    stim_abs[stim_abs > 1] = 1
   
    # Using indices above, get baseline subtracted stim responses
    v_bs = (np.dot(v[np.arange(n_dims),:], resp_ind)
            / np.sum(resp_ind,0)[None,:]
            - np.dot(v[np.arange(n_dims),:], base_ind)
            / np.sum(base_ind,0)[None,:])
    
    # Regression on first n_dims of v
    print('Fitting receptive fields to SVD components.')
    if len(alphas) > 1:
        lm = RidgeCV(alphas, fit_intercept = False)
    else:
        lm = Ridge(alphas[0], fit_intercept = False)    
          
    v_rf = lm.fit(stim_abs, v_bs.T).coef_
    
    # Convert v_RF to pixel receptive fields
    pixel_rf = np.linalg.multi_dot([u[:,np.arange(n_dims)], 
                                    np.diag(s[np.arange(n_dims)]),
                                    v_rf])
    
    # pixel_rf = pixel_rf.T
    # pixel_rf = pixel_rf.reshape(stim_shape + image_shape)
    pixel_rf = pixel_rf.reshape(image_shape + stim_shape)    

    return pixel_rf
    