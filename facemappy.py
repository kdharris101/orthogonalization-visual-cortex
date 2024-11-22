# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:05:29 2020

Functions for loading FaceMap data

@author: Samuel Failor
"""

from os.path import getmtime, getctime
import numpy as np
import timelinepy as tl
import cortex_lab_utils as clu
import choiceworldpy as cw
from scipy.io import loadmat
import datetime
import time

def load_facemap(expt_info_or_filepath):  
    '''
    Loads FaceMap output file. Arguments can be experiment info or filepath.

    Parameters
    ----------
    expt_info_or_filepath : tuple or str
        If tuple: (subject name, experiment date, experiment number)
        If str: filepath

    Returns
    -------
    numpy.ndarray
        Numpy array of facemap output

    '''
      
    # Check if argument is tuple or filepath string
    if type(expt_info_or_filepath) is tuple:
        filepath = clu.find_expt_file(expt_info_or_filepath,'facemap')
    elif type(expt_info_or_filepath) is str:
        filepath = expt_info_or_filepath
              
    # Load suite2p file
    print('Loading ' + filepath + '...')
    print("Last modified: %s" % time.ctime(getmtime(filepath)))
    print("Created: %s" % time.ctime(getctime(filepath)))
    
    facemap_data = np.load(filepath, allow_pickle = True)[()]
    
    return facemap_data

def load_frame_times(expt_info, ref = 'timeline'):
    '''
    Loads frame times for eye/face camera recording. 

    Parameters
    ----------
    expt_info : tuple or str
        If tuple: (subject name, experiment date, experiment number)
        If str: filepath
    Returns
    -------
    Frame times aligned to timeline

    '''
    
    # Load eye/face camera log file
    if type(expt_info) == str:
        eye_log = loadmat(expt_info)
    else: 
        eye_log = loadmat(clu.find_expt_file(expt_info,'eye_log'),
                          squeeze_me = True)['eyeLog']
    
   
    # breakpoint()
    
    eye_frame_times = eye_log['TriggerData'][()]['AbsTime'][:]
    eye_frame_times = np.array([t for t in eye_frame_times])
    # Add column for microseconds
    eye_frame_times = np.hstack([eye_frame_times,
                                 (eye_frame_times[:,-1]%1*1000000)[:,None]])
    # Convert to seconds
    eye_frame_times = np.array([(datetime.datetime(*t.astype(int))
                                  - datetime.datetime(1970,1,1)).total_seconds()
                                for t in eye_frame_times])
    
    
    # Load UDP times
    eye_udp_times = eye_log['udpEventTimes'][()][:]
    eye_udp_times = np.array([t for t in eye_udp_times])
    # Add column for microseconds
    eye_udp_times = np.hstack([eye_udp_times,(eye_udp_times[:,-1]%1 * 1000000)[:,None]])
    # Convert eye_udp_times to seconds
    eye_udp_times = np.array([(datetime.datetime(*t.astype(int))
                               - datetime.datetime(1970,1,1)).total_seconds() 
                              for t in eye_udp_times])
    
    if type(ref) is not str:
        
        # breakpoint()
        
        # block = cw.load_block(expt_info)[0]
        
        # expt_end = block['startDateTime']
        # expt_end = datetime.datetime.fromordinal(int(expt_end)) + datetime.timedelta(days=expt_end%1) - datetime.timedelta(days = 366)
        # expt_end = (expt_end - datetime.datetime(1970,1,1)).total_seconds()
                
        # # t_correction = eye_udp_times[0] - expt_start
        # t_correction = expt_end  - eye_udp_times[0] 
        # eye_frame_times = eye_log['TriggerData'][()]['Time'] - t_correction
        eye_frame_times = eye_log['TriggerData'][()]['Time'] + ref

    elif ref == 'timeline':
        
        # Load timeline 
        timeline = tl.load_timeline(expt_info)
        
        timeline_udp_times = timeline['mpepUDPTimes'][:]
        # Remove extra 0's from timeline
        timeline_udp_times = timeline_udp_times[np.diff(timeline_udp_times,
                                                           prepend = 0) > 0]
        # timeline_udp_times = np.insert(timeline_udp_times,0,0)  
            
        # breakpoint()
        # Excluding the first event find the average difference in time
        if len(timeline_udp_times) != 5:
            # for mpep
            t_correction = np.mean(eye_udp_times[1:] - timeline_udp_times[1:])
        else:
            # For choiceworld
            t_correction = np.mean(eye_udp_times[1:] - timeline_udp_times)
    
        eye_frame_times = eye_frame_times - t_correction
    
    
    
    
    return eye_frame_times


def load_expt(expt_info, ref = 'timeline'):
    
    expt = dict({'facemap' : None, 'frame_times' : None})
    
    print('Loading facemap data')
    expt['facemap'] = load_facemap(expt_info)
    print('Loading camera frame times')
    expt['frame_times'] = load_frame_times(expt_info, ref)
        
    return expt
    