# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:12:40 2020

Collection of functions for dealing with two-photon recordings and suite2p
outputs. 

@author: Samuel Failor
"""

from os.path import join, split
import numpy as np
import timelinepy as tl
import cortex_lab_utils as clu
import scipy.io as sio
import glob
import tifffile
from scipy.ndimage import convolve, gaussian_filter



def suite2p_plane_path(root_dir, plane = 'combined'):
    '''
    Converts tuple with experiment info to suite2p directory path on data 
    storage

    Parameters
    ----------
    root_dir : str 
        Path to root suite2p directory
    plane : int or str
        Two-photon recording plane (starts at 0) or 'combined'. If plane it not
        specified it defaults to 'combined'. If plane is -1 then path points
        to suite2p root directory.

    Returns
    -------
    filepath : str
        String of path to suite2p directory.

    '''
    
    # Build filepath string
    if type(plane) == int:
        subdir = 'plane' + str(plane)
    elif type(plane) == str:
        subdir = plane
   
    dirpath = join(root_dir, subdir)
    
    return dirpath 


def load_suite2p(expt_info_or_filepath, plane = 'combined',
                 filetype = 'spks.npy', memmap_on = False, return_path = False):  
    '''
    Loads suite2p output file. Arguments can be experiment info, plane
    number, and file type, or a string with the entire file path.

    Parameters
    ----------
    expt_info_or_filepath : tuple or str
        If tuple: (subject name, experiment date, experiment number)
        If str: filepath
    plane : int or 'combined' to load combined files
    filetype : str
        Default is 'spks.npy'
    memmap_on : bool
        If true, file is loaded with memory mapping
    return_path : bool
        If true, function returns complete filepath of loaded file as string

    Returns
    -------
    numpy.ndarray
        Numpy array of suite2p output

    '''
    
    # Adds numpy file extension if it isn't provided 
    if ('.npy' not in filetype) and ('.mat' not in filetype):
        filetype = filetype + '.npy'
    
    # breakpoint()
    
    # Check if argument is tuple or filepath string
    if ((type(expt_info_or_filepath) is tuple) | 
        (type(expt_info_or_filepath) is list)):
        filepath = suite2p_plane_path(
            clu.find_expt_file(expt_info_or_filepath,'suite2p'), plane)
        filepath = join(filepath, filetype)
    elif type(expt_info_or_filepath) is str:
        filepath = join(expt_info_or_filepath, filetype)
    
    # Set memmap to 'r', aka read-only, if True
    if memmap_on:
        memmap_on = 'r'
    else:
        memmap_on = None
        
    # breakpoint()
    # Load suite2p file
    if '.npy' in filetype:
        s2pdata = np.load(filepath, memmap_on, allow_pickle = True)[()]
    else:
        s2pdata = sio.loadmat(filepath)
    
    if return_path:
        return s2pdata,filepath
    else:
        return s2pdata
    

def get_frame_times(timeline):
    '''
    Returns frame times from timeline

    Parameters
    ----------
    timeline : numpy.ndarray
        output of load_timeline()

    Returns
    -------
    numpy.ndarray
        Array of frame times

    '''
    frames = tl.get_samples(timeline,['neuralFrames']).flatten()
    frame_times = tl.sample_times(timeline)
    # Find index where frame changes
    ind = np.diff(frames, prepend = frames[0]) > 0
    
    return frame_times[ind]
    

def get_plane_times(timeline, total_planes):
    '''
    Returns a list of numpy arrays each containing the plane times for all
    planes in the recording. 

    Parameters
    ----------
    timeline : numpy.ndarray
        output of load_timeline()
    total_planes : int
        Total number of planes in the recording

    Returns
    -------
    plane_times : list
        List of arrays containing plane times for each plane

    ''' 
    frame_times = get_frame_times(timeline)
            
    plane_times = [frame_times[p::total_planes] for p in range(total_planes)]
        
    return plane_times

def find_expt_frames(suite2p_pos, sum_frames):
    '''
    Finds frames corresponding to the experiment of interest in the suite2p
    output, based on the position of the experiment's data in the tiff stack
    '''
    
    if suite2p_pos == 0:
        expt_frames = np.arange(0,sum_frames[0])
    else:
        expt_frames = np.arange(sum_frames[suite2p_pos-1], 
                                   sum_frames[suite2p_pos])    

    return expt_frames

def load_experiment(expt_info, iscell_only = True, red_cells = False,
                    cell_outputs = ['spks'], timings = True):
    '''
    Loads outputs of suite2p data for a given experiment. 

    Parameters
    ----------
    expt_info : tuple 
        (subject name, experimenet date, and experiment number)
    iscell_only : bool, optional
        Set true to only include ROIs considered cells. The default is True.
    cell_outputs : list of strings, optional
        List the outputs you want to load (i.e. spks, F, Fneu). 
        The default is ['spks'].

    Returns
    -------
    expt_dict : dict
        Dictionary containing suite2p data and sampling times of experiment.

    '''   
    # Add basic information about the recording from ops of first plane
    print('Loading suite2p options for each plane...')
    expt_dict = {'ops': [load_suite2p(expt_info, 0, 'ops')]}
    
    # Total number of planes
    nplanes = expt_dict['ops'][0]['nplanes']
     
    '''
    Load all ops - unfortunately memmap doesn't work with dtype object so
    this is slow
    '''
    for p in range(1,nplanes):
        expt_dict['ops'].append(load_suite2p(expt_info, p, 'ops'))
    
   
    # Get experiment numbers
    # breakpoint()

    exprs, idx = np.unique([int(split(s)[0][-1]) for s in expt_dict['ops'][0]['filelist']], return_index=True)
    exprs = exprs[np.argsort(idx)]        
    
    # Find the position of the experiment when it was processed by suite2p
    suite2p_pos = list(exprs).index(expt_info[2])
    print('suite2p position ' + str(suite2p_pos))
    
    # Load iscell 
    print('Loading cell flags for ROIs...')
    iscell = [load_suite2p(expt_info, p, 'iscell')[:,0].astype(bool) for p in range(nplanes)]
    # Make single index array for later
    iscellall = np.concatenate(iscell)

    # load isred if present
    if red_cells:
        print('Assigning red cells...')
        expt_dict['isred'] = np.concatenate([load_suite2p(expt_info, p ,'isred') for p in range(nplanes)])
        
    # Load stats
    print('Loading cell stats...')
    expt_dict['stat'] = np.concatenate([load_suite2p(expt_info, p ,'stat') for p in range(nplanes)])
      
    # Remove stats and isred for ROIs not considered cells, if specified
    if iscell_only:
        expt_dict['stat'] = expt_dict['stat'][iscellall]
        if red_cells:
            expt_dict['isred'] = expt_dict['isred'][iscellall]

    # Determine which frames should be loaded from output files
    expt_frames = np.array([find_expt_frames(suite2p_pos, 
                  np.cumsum(expt_dict['ops'][p]['frames_per_folder']))
                  for p in range(nplanes)], dtype = object)
        
    # Make number of frames equal across planes
    min_frames = min(list(map(len,expt_frames)))
    for p in range(nplanes):
        if len(expt_frames[p]) > min_frames:
            expt_frames[p] = expt_frames[p][0:min_frames]
                   
    # Load suite2p outputs using memmap, only copy over frames of interest
    for f in cell_outputs:
        print('Loading cell ' + f + '...')
        if iscell_only:
            expt_dict[f] = np.concatenate(
            [np.copy(load_suite2p(expt_info, p, f, True) [iscell[p], expt_frames[p][0]:expt_frames[p][-1]+1]) for p in range(nplanes)])
        else:
            expt_dict[f] = np.concatenate(
            [np.copy(load_suite2p(expt_info, p, f, True)[:, expt_frames[p][0]:expt_frames[p][-1]+1]) for p in range(nplanes)])
    
    # Add index of plane each cell is in
    if iscell_only:
        expt_dict['cell_plane'] = np.concatenate([np.ones(sum(iscell[p]))*p for p in range(nplanes)])
    else:
        expt_dict['cell_plane'] = np.concatenate([np.ones(len(iscell[p]))*p for p in range(nplanes)])
        # Add key for iscell if non-cell rois aren't excluded
        expt_dict['iscell'] = iscell
        
    # Add plane times
    if timings:
        print('Loading plane sample times...')
        timeline = tl.load_timeline(expt_info)
        expt_dict['plane_times'] = get_plane_times(timeline, nplanes)
        
        # Make sample times same length as outputs
        for p,t in enumerate(expt_dict['plane_times']):
            expt_dict['plane_times'][p] = expt_dict['plane_times'][p][0:min_frames]
    else:
        print('Timings flag is set to FALSE, so timeline file will not be loaded.')
        
    print('Loading of experiment complete.')
    
    return expt_dict


def load_experiments_tracked(match_file, expt_num, subject=None, true_only=True, cell_outputs=['spks']):
    '''
    Creates expt dictionaries for the rois matched between recordings, as specificed in the match file

    Parameters
    ----------
    match_file : str 
        match file pattch
    expt_num: iterable
        iterable of specific experiments to load for each suite2p processed recording
    subject: iterable
        Name of subjects in case this cannot be correctly gleaned from the rec_dirs value in the match file
    true_only: bool
        only include matches flagged a true
    cell_outputs : list of strings, optional
        List the outputs you want to load (i.e. spks, F, Fneu). 
        The default is ['spks'].

    Returns
    -------
    expt_dict : list of dict
        List of dictionaries containing suite2p data and sampling times of the experiments.

    '''   
    # Chained attempts to load, to deal with network connectivity issues
    try:
        print('Loading match file...')
        matches = np.load(match_file,allow_pickle=True)[()]
    except:
        try:
            matches = np.load(match_file,allow_pickle=True)[()]
        except:
            try:
                matches = np.load(match_file,allow_pickle=True)[()]
            except:
                print('Loading match file failed after three attempts.')
                pass
                
    rec_dirs = matches['rec_dirs']
    matches = matches['matches']

    if true_only:
        matches = matches[matches.true_match]
        
        
    # check for duplicates
    
    # duplicates = [np.any(matches[matches.columns[[i,i+1]]].duplicated()) for i in np.arange(0, len(matches.columns)-2, 2)]
    
    # if np.any(duplicates):
    #     print('Duplicate matches detected. Removing...')
    
    #     for i in np.arange(0,len(matches.columns)-2, 2):
    #         # Remove duplicates, keeping the one with largest overlap
    #         matches = matches.sort_values('overlap').drop_duplicates(matches.columns[[i,i+1]], keep='last', ignore_index=True)

    roi_cols = [c for c in matches.columns if 'roi' in c]
    plane_cols = [c for c in matches.columns if 'plane' in c]
    matches.sort_values(by=plane_cols, inplace = True)
    
    planes_to_load = [matches[plane_cols[i]].unique() for i in range(len(plane_cols))]
    
    expts = []
    
    for i in range(len(roi_cols)):
                 
        expt_dict = {'stat' : np.zeros(len(matches),dtype=object),
                     'cell_plane' : np.zeros(len(matches)),
                     'plane_times' : {},
                     'rec_dir' : rec_dirs[i],
                     'ops' : {}}
        
        for o in cell_outputs:
            expt_dict[o] = np.zeros(len(matches),dtype=object)
        
        for n,p in enumerate(planes_to_load[i]):
                       
            ops = load_suite2p(join(rec_dirs[i],f'plane{p}'), filetype='ops.npy')
            expt_dict['ops'][p] = ops
            
            if n == 0:
                # Get experiment numbers for session
                exprs, idx = np.unique([int(split(s)[0][-1]) for s in ops['filelist']], return_index=True)
                exprs = exprs[np.argsort(idx)] 
        
                suite2p_pos = list(exprs).index(expt_num[i])
            
            ind = matches[plane_cols[i]].to_numpy() == p
            rois = matches.loc[ind,roi_cols[i]].to_numpy()
            expt_dict['stat'][ind] = load_suite2p(join(rec_dirs[i],f'plane{p}'), filetype='stat.npy')[rois]
            expt_dict['cell_plane'][ind] = p
            
            # Determine which frames should be loaded from output files
            expt_frames = find_expt_frames(suite2p_pos, np.cumsum(ops['frames_per_folder']))
            
            for o in cell_outputs:
                output = load_suite2p(join(rec_dirs[i],f'plane{p}'), filetype=o, memmap_on=True)[np.ix_(rois,expt_frames)]
                expt_dict[o][np.where(ind)[0]] = np.split(output,len(rois)) # Will concatenate later after ensuring number of columns is consistent
    
        # Plane times
        # Make timeline file path
        filepath = rec_dirs[i]
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
        
        if not subject:
            timeline_path = join(*parts[:3],str(expt_num[i]),f'{parts[2]}_{expt_num[i]}_{parts[1]}_Timeline.mat')
        else:
            timeline_path = join(parts[0],subject[i],parts[2],str(expt_num[i]),f'{parts[2]}_{expt_num[i]}_{subject[i]}_Timeline.mat')

        timeline = tl.load_timeline(timeline_path)
        plane_times = get_plane_times(timeline, ops['nplanes'])
        expt_dict['plane_times'] = {p : plane_times[p] for p in planes_to_load[i]}
               
        trace_lengths = [t.shape[1] for t in expt_dict[cell_outputs[0]]]

        min_frames = np.min(trace_lengths)
        
        to_be_reduced = np.where(trace_lengths > min_frames)[0]
        
        if len(to_be_reduced) > 0:
            for o in cell_outputs:
                for c in to_be_reduced:
                    expt_dict[o][c] = expt_dict[o][c][:,:min_frames]
                    
        for o in cell_outputs:
            expt_dict[o] = np.concatenate(expt_dict[o])

        for p in expt_dict['plane_times'].keys():
            expt_dict['plane_times'][p] =  expt_dict['plane_times'][p][:min_frames]
        
        expts.append(expt_dict)
             
    return expts

def update_iscell_mat(expt_info):
    '''
    Updates Fall.mat iscell field to that saved in iscell.npy. This is 
    necessary if the suite2p GUI wasn't told to save results in .mat as well
    after manual curation. 

    Parameters
    ----------
    expt_info : tuple
        (Subject, expt date, expt number).

    Returns
    -------
    None.

    '''
    
    # Add basic information about the recording from ops of first plane
    print('Loading suite2p options for each plane...')
    expt_dict = {'ops': [load_suite2p(expt_info, 0, 'ops')]}
    
    # Total number of planes
    nplanes = expt_dict['ops'][0]['nplanes']
    
    for p in range(nplanes):
        print('Loading iscell.npy')
        iscell = load_suite2p(expt_info, plane = p, filetype = 'iscell.npy')
        print('Loading Fall.mat')
        iscell_mat, mat_path = load_suite2p(expt_info,plane=p,
                                    filetype = 'Fall.mat', return_path = True)
        
        iscell_mat['iscell'] = iscell
        
        mat_path = mat_path.replace('Fall.mat','Fall_updated.mat')
        
        print('Saving ' + mat_path)
        sio.savemat(mat_path, iscell_mat)
        
        
def load_tifs(directory, nplanes=1, nchannels=1):
    
    tif_files = sorted(glob.glob(directory + '/*.tif'))
        
    data_dict = {(plane, channel): [] for plane in range(nplanes) for channel in range(nchannels)}

    for file in tif_files:
        print(f'Loading {file}')
        with tifffile.TiffFile(file) as tif:
            num_images = len(tif.pages)

            # Assign each image in the stack to the correct plane and channel
            for i in range(num_images):
                # Calculate plane and channel, considering the sequence in the stack
                # Adjust calculations to match the (plane, channel) sequence
                plane = ((i // nchannels) % nplanes)
                channel = (i % nchannels)
                
                # Append the image to the correct list in the dictionary
                data_dict[(plane, channel)].append(tif.pages[i].asarray())

    # Convert lists to numpy arrays
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key])

    return data_dict


def adaptive_threshold(image, type='mean', side = 'greater', kernel_size=21, c=5):
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if type == 'mean':
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        filtered_image = convolve(image, kernel, mode='nearest')
    elif type == 'gaussian':
        filtered_image = gaussian_filter(image, kernel_size, mode='nearest')    
    
    if side == 'greater':
        return image > (filtered_image - c)
    elif side == 'less':
        return image < (filtered_image + c)
    

    
    
    