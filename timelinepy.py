# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:09:55 2020

Functions that deal with timeline .mat (matlab) files

@author: Samuel Failor
"""

from os.path import join
from os.path import exists
import scipy.io as sio
import scipy.signal as ss
from sklearn.cluster import KMeans
import numpy as np
# import plotly.express as px
import pyqtgraph as pg
import cortex_lab_utils as clu

# def timeline_path(expt_info):
#     '''
#     Converts tuple with experiment info to timeline file path on data storage

#     Parameters
#     ----------
#     expt_info : tuple 
#         (root directory of experiment, subject name, 
#          experimenet date, experiment number)

#     Returns
#     -------
#     filepath : str
#         String of path to timeline file.

#     '''
#     # Unpack tuple
#     root_dir, subject, expt_date, expt_num = expt_info
#     # Build filepath string
#     filepath = join(root_dir, subject, expt_date, str(expt_num), expt_date
#                      + "_" + str(expt_num) + "_" + subject + "_Timeline.mat")
#     return filepath 


def load_timeline(expt_info_or_filepath):    
    '''
    Takes tuple with experimental info or string with filepath and loads the
    timeline .mat file

    Parameters
    ----------
    expt_info_or_filepath : tuple or str
        If tuple: (subject name, experiment date, experiment number)
        If str: filepath 

    Returns
    -------
    numpy.ndarray
        A matlab structure converted to a numpy array where dtypes correspond
        typically to the field names in the original structure. A bit of a 
        mess.

    '''
    # Check if argument is tuple or filepath string
    if (type(expt_info_or_filepath) is tuple) or (type(expt_info_or_filepath) 
                                                  is list):
        filepath = clu.find_expt_file(expt_info_or_filepath,'timeline')
    elif type(expt_info_or_filepath) is str:
        filepath = expt_info_or_filepath

    # Load timeline file    
    timeline = sio.loadmat(filepath, squeeze_me = True)
    return timeline["Timeline"][()]


def sample_times(timeline):
    '''
    Extracts sample times from a timeline, i.e. returns rawDAQTimestamps

    Parameters
    ----------
    timeline : numpy.ndarray
        output of load_timeline()

    Returns
    -------
    numpy.ndarray
        n x 1 array of sample times.

    '''
    return timeline['rawDAQTimestamps']


def get_input_names(timeline):
    '''
    Returns list of hardware inputs in timeline
    '''
    
    return timeline['hw']['inputs'][()]['name']


def get_samples(timeline, input_names):
    '''
    Extracts input samples from timeline object

    Parameters
    ----------
    timeline : numpy.ndarray
        output of load_timeline()
    input_names : list
        Inputs of interest in timeline data

    Returns
    -------
    timeline_data : numpy.ndarray
        n x len(input_names) array with samples of one or more specified inputs

    ''' 
    # Pre-allocate memory
    timeline_data = np.empty((len(timeline['rawDAQData']),
                              len(input_names)))
    
    inputs_in_timeline = get_input_names(timeline)
    
    # Add inputs of interest to numpy array
    for i,name in enumerate(input_names):
        timeline_data[:,i] = timeline['rawDAQData'][:,name == inputs_in_timeline].flatten()

    return timeline_data

def align_to_pd(computer_times, edge, timeline, all_flips = False):
    '''
    Function takes a tuple of arrays containing various computer times that 
    need to be aligned to photodiode flip times. For example, stimulus
    onsets and offsets. In cases where the computer time will preceed a photo-
    diode flip, this should be specified in edge as 'after', and if the timing
    follows a photodiode flips, such as in the case when a stimulus turns off,
    edge should be 'before'. 
    
    If all_flips is True, function returns all photodiode flips between two,
    and only two, event times.
    
    Parameters
    ----------
    computer_times : tuple of indexables
        Rows correspond to times, columns are events
    edge : tuple of strings
        Specifies whether flip times should come before or after computer 
        times
    timeline : numpy.ndarray
        output of load_timeline()

    Returns
    -------
    event_times : numpy.ndarray of lists 
        Event times aligned to photodiode flips.
    delta: float
        Difference between saved event times and actual event times as 
        represented in photodiode signal.

    '''
          
    # Get photodiode signal from timeline
    photodiode = get_samples(timeline,['photoDiode'])
    # Use median filter to smooth out potential noise
    #print('Median filtering photodiode trace.')
    photodiode[:,0] = ss.medfilt(photodiode[:,0],3)
    
    # Find average min and max of photodiode trace
    #print('Finding up and down states using kmeans.')
    pd_clusters = KMeans(n_clusters = 2, random_state = 0).fit(photodiode)
    pd_clusters = pd_clusters.cluster_centers_
    
    # Flip threshold is the value between min and max
    flip_threshold = np.mean(pd_clusters)
    
    # Binerize to above and below the threshold
    pd_state = (photodiode - flip_threshold) > 0
    # Flip times are where the state changes
    pd_flips = np.insert(np.abs(np.diff(pd_state[:,0].astype(int))), 0, False,
                         axis = 0)
    # Photodiode sample times
    pd_times = sample_times(timeline)
    
    # Photodiode flip times
    flip_times = pd_times[pd_flips > 0]
    
    # Pre-allocate memory
    event_times = np.empty(len(computer_times), dtype = object)
    
    # Find pd flips closest to computer event times. Flips should come either
    # after or before recorded computer times.
    
    # breakpoint()
    for e in range(len(computer_times)):
        # if np.max(flip_times) > np.max(computer_times[e]):          
        if edge[e] == 'after':
            event_times[e] = np.array([np.amin(flip_times[flip_times 
                            >= i]) for i in computer_times[e]])
            for i in range(len(computer_times[e])):
                event_times[e][i] = np.amin(flip_times[flip_times >= computer_times[e][i]])
            
        elif edge[e] == 'before':
            event_times[e] = np.array([np.amax(flip_times[flip_times 
                            <= computer_times[e][i]]) 
                              for i in range(len(computer_times[e]))])
        print('Average difference between pd flip times and computer times ' 
              + str((event_times[e] - computer_times[e]).mean()) + '. A large difference may indicate misalignment or missing pd flips.')
        # else:
        #     # Change this so that at least some computer times are corrected
        #     print('PD flips are missing, cannot align all events. Returning event times uncorrected.')
        #     return computer_times
       
    # Return all pd flips between a pair of events
    if all_flips and len(event_times[0]) == 2:
        event_times = flip_times[(flip_times >= event_times[0][0]) 
                                 & (flip_times <= event_times[0][1])]
        
    # delta = np.mean(event_times[0]-computer_times[0])

    return event_times
        
    
def get_udp(timeline):
    '''
    
    Gets udp events and times for mpep experiment from timeline

    Parameters
    ----------
    timeline : numpy.ndarray
        output of load_timeline()

    Returns
    -------
    udp : numpy.ndarray
        UDP events
    udp_times : numpy.ndarray
        UDP times

    '''
    
    udp = timeline['mpepUDPEvents']
    udp_times = timeline['mpepUDPTimes']
    
    # Remove empty elements from udp and udp_times
    ind = [type(i) == str for i in udp]
    udp, udp_times = udp[ind], udp_times[ind]
    
    return udp, udp_times


def inspect_pd_signal(timeline):
    """Opens figure that allows for inspection of photo diode signal

    Args:
        timeline_file (numpy.ndarray): output of load_timeline()
    """
    
    
    pd = np.squeeze(get_samples(timeline,['photoDiode']))
    times = sample_times(timeline)
    
    pg.plot(times, pd)  
    