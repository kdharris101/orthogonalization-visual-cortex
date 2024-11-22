#%% 
"""
Created on Thu Feb 10 14:03:29 2022

@author: Samuel
"""


from os.path import join
import datetime
import glob
import numpy as np
import sklearn.preprocessing as skp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, make_scorer
# from sklearn.feature_selection import SequentialFeatureSelector
# from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import kurtosis
from time import process_time


# results_dir = 'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

subjects = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']
expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]


# Load most up-to-date saved results
file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i],
                    str(expt_nums[i]), r'_'.join([subjects[i], expt_dates[i],
                    str(expt_nums[i]), 'orientation tuning_norm by pref ori_[2020]*'])))[-1]
                                            for i in range(len(subjects))]
 
    
#%%


save_paths = [join(results_dir, subjects[i], expt_dates[i], 
                   str(expt_nums[i]), 'stim_decoding_results_45 and 90 only_all cells_' 
                   + str(datetime.date.today())) for i in range(len(subjects))]

for i,f in enumerate(file_paths):

    print('Loading ' + f)
    expt = np.load(f,allow_pickle = True)[()]

    stim_dir = expt['stim_dir']
    stim_dir[np.isinf(stim_dir)] = -1
    
    stim_ind = np.logical_or(stim_dir % 180 == 45, stim_dir % 180 == 90)
    stim_dir = stim_dir[stim_ind]
    stim_ori = stim_dir % 180
        
    # Index every other stimulus trial to make train and test sets
    ind_sets = np.arange(len(stim_ori))
    train_ind = np.concatenate([np.where(stim_ori == s)[0][::2]  
                            for s in np.unique(stim_ori)])
    test_ind = np.delete(ind_sets,train_ind)
    
    stim_ori_test = stim_ori[test_ind]
    stim_ori_train = stim_ori[train_ind]
    
    # Only include cells in V1 and not in the fly back plane
    g_cells = np.logical_and(expt['cell_plane'] > 0, expt['V1_ROIs'] == 1)
    # Exclude cells that don't respond to stimuli
    # all_resps = expt['trial_resps'][stim_ind,:]   
    # g_cells = np.logical_and(g_cells, all_resps[train_ind,:].sum(0) != 0)
    
    x = expt['trial_resps'][np.ix_(stim_ind,g_cells)]
       
    x_train = x[train_ind,:]
    x_test = x[test_ind,:]
    
    # Cell stats
    osi = expt['r_test'][g_cells]
    mean_ori = np.concatenate([x_test[stim_ori_test==s,:].mean(0, keepdims= True)
                                            for s in np.unique(stim_ori)], 
                                            axis = 0)
    std_ori = np.concatenate([x_test[stim_ori_test==s,:].std(0, keepdims= True)
                                            for s in np.unique(stim_ori)], 
                                            axis = 0)
    lt_sparseness = kurtosis(mean_ori, axis = 0)
    pref_ind = np.argmax(mean_ori,0)
    mean_pref = np.max(mean_ori,0)
    std_pref = std_ori[pref_ind,np.arange(std_ori.shape[1])]  
    
    # Initialize objects
    clf = LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 'auto')
       
    decode_results = {}
    
    n_labels = len(np.unique(stim_ori))
    
    pred_stim = np.empty(len(test_ind))
    prob_stim = np.empty((len(test_ind), n_labels))
    # posterior_stim = np.empty((len(test_ind), n_labels, n_features, n_pool_sizes, n_repeats))
    
    scaler = skp.StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    x_test_z = scaler.transform(x_test)
    
    # enc = skp.OneHotEncoder(sparse = False)
    
    # stim_train = enc.fit_transform(stim_ori[train_ind].reshape(-1,1))
    # stim_test = enc.transform(stim_ori[test_ind].reshape(-1,1))
    
    stim_train = stim_ori[train_ind] == 45
    stim_test = stim_ori[test_ind] == 45
 
    print(subjects[i])               
  
    # Train classifer using all cells
    
    clf.fit(x_train_z,stim_ori[train_ind])
    pred_stim = clf.predict(x_test_z)
    prob_stim = clf.predict_proba(x_test_z)
    model_weights = clf.coef_
      
    # Find cells that prefer blank
    uni_stim = np.unique(expt['stim_dir'])
    
    x_all = x = expt['trial_resps'][:,g_cells]
    
    mean_stim = np.concatenate([x_all[expt['stim_dir']==s,:].mean(0, keepdims= True)
                                            for s in uni_stim], 
                                            axis = 0)
    pref_stim = np.argmax(mean_stim,axis=0)
    pref_blank = pref_stim == 0
    
    # Calculate dprime
    mu = np.array([x_test[stim_ori_test==s,:].mean(0) 
                         for s in np.unique(stim_ori_test)])
    mu_diff = np.diff(mu,axis=0).flatten()
    
    sigma = np.sqrt(np.array([x_test[stim_ori_test==s,:].var(0) 
                         for s in np.unique(stim_ori_test)]).mean(0))
    
    d_prime = mu_diff/sigma
    
    sigma = np.array([x_test[stim_ori_test==s,:].var(0) 
                         for s in np.unique(stim_ori_test)])
    
    cv = sigma/mu
    
    # Ori prefs
    g_th = expt['th_test'][g_cells]
    g_pref = expt['pref_ori_test'][g_cells]
    g_x = expt['v_x'][g_cells]
    g_y = expt['v_y'][g_cells]
    
    decode_results= {'pred_stim' : pred_stim,
                     'prob_stim' : prob_stim,
                     'stim' : stim_ori[test_ind],
                     'stim_train' : stim_ori[train_ind],
                     'stim_dir' : stim_dir[test_ind],
                     'stim_dir_train' : stim_dir[train_ind],
                     'trials_z_test' : x_test_z,
                     'trials_test' : x_test,
                     'trials_z_train' : x_train_z,
                     'trials_train' : x_train,
                     'OSI' : osi,
                     'mean_pref' : mean_pref,
                     'std_pref' : std_pref,
                     'lt_sparseness' : lt_sparseness,
                     'modal_pref_ori' : g_pref,
                     'mean_pref_ori' : g_th,
                     'pref_blank' : pref_blank,
                     'n_cells' : len(d_prime),
                     'model_weights' : model_weights,
                     'd_prime' : d_prime,
                     'ave_cv_45_and_90' : cv.mean(0),
                     'v_x' : g_x,
                     'v_y' : g_y}

    np.save(save_paths[i], decode_results)
# %%
