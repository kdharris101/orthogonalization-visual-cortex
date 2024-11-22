# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:31:38 2020

General utility functions for cortex lab experiments

@author: Samuel Failor
"""

from os.path import join
from os.path import exists, isdir
from sys import platform


def expt_dirs():
    
    if platform == 'win32':
    
        dirs = [r'C:\Users\Samuel\OneDrive - University College London\Results', # Check if files are saved locally first
                r'C:\Users\samue\OneDrive - University College London\Results',
                r'//znas.cortexlab.net/Subjects/',
                r'//zubjects.cortexlab.net/Subjects/',
                r'//zserver.cortexlab.net/Data/Subjects/',
                r'//zserver.cortexlab.net/Data/trodes/',
                r'//zserver.cortexlab.net/Data/expInfo/',
                r'//zinu.cortexlab.net/Subjects/',
                r'//zaru.cortexlab.net/Subjects/',
                r'//zortex.cortexlab.net/Subjects/']
        
    elif platform == 'linux':
        
        dirs = [r'/mnt/znas/subjects/',
                r'/mnt/zubjects/subjects/',
                r'/mnt/zserver/data/Subjects/',
                r'/mnt/zserver/data/trodes/',
                r'/mnt/zserver/data/trodes/expInfo/',
                r'/mnt/znas/subjects/',
                r'/mnt/zinu/subjects/',
                r'/mnt/zaru/subjects/',
                r'/mnt/zortex/subjects']
    
    return dirs

def find_subject_dirs(subject):
    
    dirs = expt_dirs()
    
    subject_path = []
    
    for d in dirs:
        if isdir(join(d, subject)):
            subject_path.append(join(d, subject))
        
    if len(subject_path) > 0:
        return subject_path
    else:
        print('Subject directory could not be found! Be sure that ' + 
              'cortex_lab_utils.expt_dirs() includes all valid subject directories.')
        
        
def find_expt_file(expt_info,file,one_extension='None'):
    
    subject, expt_date, expt_num = expt_info
       
    file_names = {'timeline' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'Timeline.mat'])),
                  'protocol' : join(subject, expt_date, str(expt_num), 
                                  'Protocol.mat'),
                  'block' : join(subject, expt_date, str(expt_num),
                                     '_'.join([expt_date,str(expt_num),subject,
                                              'Block.mat'])),
                  'suite2p' : join(subject, expt_date, 'suite2p'),
                  'facemap' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'eye_proc.npy'])),
                  'eye_log' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'eye.mat'])),
                  'eye_video' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'eye.mj2'])),
                  'root' : join(subject,expt_date,str(expt_num)),
                  'signals' : join(subject,expt_date,str(expt_num),
                                  f'_misc_trials.{one_extension}.npy')}
                        
    file_name = file_names.get(file.lower(), 'invalid')
    
    if file_name == 'invalid':
        print('File type is invalid. Valid file types are ' 
              + str(list(file_names.keys())))
        return
                                  
    dirs = expt_dirs()

    for d in dirs:
        # print(d)
        # print(join(d,file_name))
        if exists(join(d,file_name)):
            file_path = join(d,file_name)
            # print(file_path)
            break
    
    if 'file_path' in locals():
        return file_path
    else: 
        print('File could not be found! Be sure that ' + 
              'cortex_lab_utils.expt_dirs() includes all valid directories.')