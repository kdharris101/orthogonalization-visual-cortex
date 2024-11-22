#%% 
"""
@author: Samuel
"""


from os.path import join
import datetime
import glob
import numpy as np
import sklearn.preprocessing as skp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, make_scorer
from time import process_time
from joblib import Parallel,delayed
from tqdm import tqdm


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

pool_sizes = [1,10,25,50,100,200,400,800]

n_repeats = 2000

save_paths = [join(results_dir, subjects[i], expt_dates[i], 
                   str(expt_nums[i]), 'stim_decoding_results_45 and 90 only_pool_sizes_' 
                   + str(datetime.date.today())) for i in range(len(subjects))]


def classifier(x_train,x_test,y_train):
    
    clf = LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 'auto')
    try:
        clf.fit(x_train,y_train)
    except:
        print('Fit failed, skipping cell index')
        return None,None,None,None
    pred_stim = clf.predict(x_test)
    prob_stim = clf.predict_proba(x_test)
    model_weights = clf.coef_
    dec_fun = clf.decision_function(x_test)
    
    return pred_stim, prob_stim, model_weights, dec_fun

for i,f in enumerate(file_paths):
    
    decode_results = {}

    print('Loading ' + f)
    expt = np.load(f,allow_pickle = True)[()]

    stim_dir = expt['stim_dir']
    stim_dir[np.isinf(stim_dir)] = -1
    
    stim_ind = np.logical_or(stim_dir % 180 == 45, stim_dir % 180 == 90)
    stim_dir = stim_dir[stim_ind]
    stim_ori = stim_dir % 180
    
    n_labels = len(np.unique(stim_ori))

    # Index every other stimulus trial to make train and test sets
    ind_sets = np.arange(len(stim_ori))
    train_ind = np.concatenate([np.where(stim_ori == s)[0][::2]  
                            for s in np.unique(stim_ori)])
    test_ind = np.delete(ind_sets,train_ind)
    
    stim_ori_test = stim_ori[test_ind]
    stim_ori_train = stim_ori[train_ind]
        
    # Only include cells in V1 and not in the fly back plane
    g_cells = np.logical_and(expt['cell_plane'] > 0, expt['V1_ROIs'] == 1)
    cell_index = np.where(g_cells)[0]
    
    x = expt['trial_resps'][np.ix_(stim_ind,g_cells)]

    x_train = x[train_ind,:]
    x_test = x[test_ind,:]
                
    decode_results = {}
                
    scaler = skp.StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    x_test_z = scaler.transform(x_test)
    
    stim_train = stim_ori[train_ind] == 45
    stim_test = stim_ori[test_ind] == 45    

    for ip,p in enumerate(pool_sizes):
        
        pred_stim = np.empty((len(test_ind), n_repeats))
        prob_stim = np.empty((len(test_ind), n_labels, n_repeats))
        model_weights = np.zeros((p,n_repeats))
        dec_fun = np.empty((len(test_ind), n_repeats))
                
        np.random.seed(p)
        cell_inds = [np.random.choice(x.shape[1],p,replace = False) for i in range(n_repeats)]
        
        # Run in parallel to speed things up  
        parallel = Parallel(n_jobs=8)
        results = parallel(delayed(classifier)(x_train_z[:,c],x_test_z[:,c],stim_ori[train_ind]) for c in tqdm(cell_inds))  
                
        for ri,r in enumerate(results):
            pred_stim[:,ri] = r[0]
            prob_stim[...,ri] = r[1]
            model_weights[:,ri] = r[2]
            dec_fun[:,ri] = r[3]
         
        decode_results[p] = {}
        decode_results[p]['pred_stim'] = pred_stim
        decode_results[p]['prob_stim'] = prob_stim
        decode_results[p]['model_weights'] = model_weights
        decode_results[p]['decision_function'] = dec_fun
        decode_results[p]['cell_inds_rel_expt'] = [cell_index[c] for c in cell_inds]
        decode_results[p]['cell_inds_rel_gcells'] = cell_inds

    decode_results['stim_ori_test'] = stim_ori_test

    np.save(save_paths[i], decode_results)
# %%
