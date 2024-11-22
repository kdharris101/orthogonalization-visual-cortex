#%%
# 
"""
Created on Mon Feb 14 15:07:16 2022

@author: Samuel
"""


import numpy as np
import pandas as pd
import sklearn.metrics as skm
from os.path import join
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# plt.style.use('seaborn')

# results_dir = 'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

subjects = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']
expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]

trained = [False, True, False, True, False, True, False, True, False, True]

# Load most up-to-date saved results
# file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
#                    str(expt_nums[i]), 
#                    'stim_decoding_results_replacement*'))[-1] 
#             for i in range(len(subjects))]

# file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
#                     str(expt_nums[i]), 
#                     'stim_decoding_results_all_cells_2022-01-24*'))[-1] 
#             for i in range(len(subjects))]

# file_paths = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
#                     str(expt_nums[i]), 
#                     'stim_decoding_results_all_cells_2022*'))[-1] 
#             for i in range(len(subjects))]

file_paths_all = [glob.glob(join(results_dir, subjects[i], expt_dates[i], 
                    str(expt_nums[i]), 
                    'stim_decoding_results_stepwise_selection_45 and 90 only_all cells*'))[-1] 
            for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel/OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%%

df_weights = pd.DataFrame(columns = ['subject','trained','selectivity','ori_pref','model_weight'])

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode_all = np.load(file_paths_all[i],allow_pickle = True)[()]
    
    num_features = decode_all['pred_stim'].shape[1]
    
    stim = decode_all['stim']
        
    if i == 0:
        accu_all = np.zeros((len(subjects), num_features))
        OSI_all = np.copy(accu_all)
        mean_resp_all = np.copy(accu_all)
        std_resp_all = np.copy(accu_all)
        lt_sparseness_all = np.copy(accu_all)
        log_loss_all = np.copy(accu_all)
        accu_all_max_cells = np.zeros(len(subjects))
        
        
    accu = decode_all['pred_stim'] == np.tile(stim.reshape(-1,1), (1,num_features))
    accu = accu.sum(0)/len(stim)
    
    prob_stim = decode_all['prob_stim']
   
    log_loss = np.zeros(num_features)
   
    for l in range(prob_stim.shape[2]):
       log_loss[l] = skm.log_loss(stim,prob_stim[:,:,l])
      
    features = decode_all['features']
    accu_all[i,:] = accu
    OSI_all[i,:] = decode_all['OSI'][features]
    mean_resp_all[i,:] = decode_all['mean_pref'][features]
    std_resp_all[i,:] = decode_all['std_pref'][features]
    lt_sparseness_all[i,:] = decode_all['lt_sparseness'][features]
    log_loss_all[i,:] = log_loss
    
    n_cells = len(decode_all['OSI'])
    
    df_weights = pd.concat([
        df_weights,pd.DataFrame({'subject' : np.repeat(i,n_cells),
                                 'Trained' : np.repeat(trained[i],n_cells),
                                 'selectivity' : decode_all['OSI'],
                                 'ori_pref' : decode_all['mean_pref_ori'],
                                 'model_weight' : decode_all['model_weights'].flatten(),
                                 'num_cells' : np.repeat(n_cells,n_cells),
                                 'cv' : decode_all['std_pref']/decode_all['mean_pref'],
                                 'd_prime' : decode_all['d_prime']})])
    
    accu = decode_all['pred_stim_all_cells'] == stim
    
    accu_all_max_cells[i] = accu.sum()/len(stim)




df_weights['pref_bin'] = pd.cut(df_weights.ori_pref,np.linspace(-11.25,180-11.25,9),
                                labels = np.ceil(np.arange(0,180,22.5)))
df_weights['r_bin'] = pd.cut(df_weights.selectivity,np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

#%% Plot model weights by cell selectivity and orientation preference

df_plot = df_weights.groupby(['subject','Trained','pref_bin','r_bin'], as_index = False).mean()

sns.relplot(data = df_plot, x = 'pref_bin', y = 'model_weight', col = 'r_bin', 
            kind = 'line', errorbar = ('se',1), hue = 'trained', palette = 'colorblind',
            col_wrap = 5)

#%% Scatter plots of model weight by selectivity and CoV


df_weights['Mean ori. pref.'] = pd.cut(df_weights.ori_pref,np.linspace(-11.25,180-11.25,9),
                                labels = np.ceil(np.arange(0,180,22.5))).astype(int)
bins = np.linspace(0,1,21)
labels = 0.5*(bins[1:]+bins[:-1])
df_weights['r_bin'] = pd.cut(df_weights.selectivity,bins,
                     labels = labels)
bins = np.linspace(0,6,41)
labels = 0.5*(bins[1:]+bins[:-1])
df_weights['cv_bin'] = pd.cut(df_weights.cv, bins,
                              labels = labels)

# bins = np.linspace(-150,150,41)
bins = np.insert(np.insert(np.arange(-20,25,5), 0, -1000), len(np.arange(-20,25,5))+1, 1000)
# labels = 0.5*(bins[1:]+bins[:-1])
# df_weights['weight_bin'] = pd.cut(df_weights.model_weight,bins, labels = labels)
df_weights['weight_bin'] = pd.cut(df_weights.model_weight,bins,labels = np.arange(len(bins)-1))

df_plot = pd.DataFrame.copy(df_weights)

ind = np.logical_or(df_plot['Mean ori. pref.'] == 45, df_plot['Mean ori. pref.'] == 90)
df_plot = df_plot[ind].groupby(['subject','Trained','Mean ori. pref.', 'weight_bin'], 
                               as_index = False).mean()
# df_plot_r = df_plot[ind].groupby(['subject','Trained','Mean ori. pref.','r_bin'], as_index = False).mean()
# df_plot_cv = df_plot[ind].groupby(['subject','Trained','Mean ori. pref.','cv_bin'], as_index = False).mean()



ori_cp = sns.hls_palette(8)

ori_colors = {o:c for o,c in zip(np.unique(df_weights['Mean ori. pref.']),ori_cp)}
marker_style = {False : 'x',
                True : 'o'}

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":2,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams['figure.dpi'] = 100

    f,ax = plt.subplots(1,4,figsize = (5.5,2.6))
    
    # sns.scatterplot(data = df_weights, y = 'selectivity', x = 'model_weight', hue = 'Mean ori. pref.',
    #                 style = 'Trained', palette = ori_cp,
    #                 style_order = [True,False], legend = False, edgecolor = 'k',
    #                 ax = ax[0], rasterized = True, zorder = 2, facecolor = None)
    
    ax[0].scatter(x = df_weights[df_weights.Trained==False].model_weight, 
                  y = df_weights[df_weights.Trained==False].selectivity,
                  ec = df_weights.loc[df_weights.Trained==False,'Mean ori. pref.'].map(ori_colors),
                  marker = 'x', fc = None)
    ax[0].scatter(x = df_weights[df_weights.Trained].model_weight, 
                  y = df_weights[df_weights.Trained].selectivity,
                  ec = df_weights.loc[df_weights.Trained,'Mean ori. pref.'].map(ori_colors),
                  marker = 'o', fc = "none")
    
    # sns.scatterplot(data = df_weights, y = 'cv', x = 'model_weight', hue = 'Mean ori. pref.',
    #                 style = 'Trained', palette = ori_cp,
    #                 style_order = [True,False], legend = False, ec = 'k',
    #                 ax = ax[1], rasterized = True, zorder = 2, facecolor = None)
    
    
    ax[1].scatter(x = df_weights[df_weights.Trained==False].model_weight, 
                  y = df_weights[df_weights.Trained==False].cv,
                  ec = df_weights.loc[df_weights.Trained==False,'Mean ori. pref.'].map(ori_colors),
                  marker = 'x', fc = None)
    ax[1].scatter(x = df_weights[df_weights.Trained].model_weight, 
                  y = df_weights[df_weights.Trained].cv,
                  ec = df_weights.loc[df_weights.Trained,'Mean ori. pref.'].map(ori_colors),
                  marker = 'o', fc = "none")
    
    
    # ax[1].legend([1,2,5,10], frameon = False, fancybox = False,
    #                     ncol = 1, loc = 4, labelspacing = 0.1,
    #                     handletextpad = 0,
    #                     bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
    
      
    ax[0].set_xlabel('Model weight')
    ax[1].set_xlabel('Model weight')
    ax[0].set_ylabel('Selectivity')
    ax[1].set_ylabel('Coefficient of variation')
    ax[0].set_xlim([-150,150])
    ax[1].set_xlim([-150,150])
    ax[1].set_yticks([])
    ax[1].set_ylim([-0.1,5])
    ax[1].set_yticks([0,2.5,5])
    # ax[0].axvline(0.6, zorder = 0, color = 'k', linestyle = '--')
    # ax[1].axvline(0.5, zorder = 0, color = 'k', linestyle = '--')
    ax[0].set_aspect(1.0/ax[0].get_data_ratio(), adjustable='box')
    ax[1].set_aspect(1.0/ax[1].get_data_ratio(), adjustable='box')
    
    sns.despine(offset = 3, trim = True, ax = ax[0])
    sns.despine(offset = 3, trim = True, ax = ax[1])
    
    
    
    
    lp = sns.lineplot(data = df_plot, y = 'selectivity', 
                 x = 'weight_bin', hue = 'Trained', ax = ax[2],
                 palette = 'colorblind', hue_order = [False,True],
                 errorbar = ('se',1),
                 style = 'Mean ori. pref.')
    # ax[2].set_xlim([-150,150])
    # ax[2].set_xlim([0,11])
    ax[2].set_ylim([0,1])
    # bins = np.linspace(0,1,21)
    # labels = 0.5*(bins[1:]+bins[:-1])
    ax[2].set_yticks([0,0.5,1])
    # ax[2].set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    # ax[2].xaxis.set_tick_params(rotation=35)
    ax[2].set_aspect(1.0/ax[2].get_data_ratio(), adjustable='box')
    ax[2].set_xlabel('Model weight')
    ax[2].set_ylabel('Selectivity')
    ax[2].legend_.set_visible(False)
    
    sns.despine(offset = 3, trim = True, ax = ax[2])
    
    lp = sns.lineplot(data = df_plot, y = 'cv', 
                 x = 'weight_bin', hue = 'Trained', ax = ax[3],
                 palette = 'colorblind', hue_order = [False,True],
                  errorbar = ('se',1),
                 style = 'Mean ori. pref.')
    # ax[3].set_xlim([-150,150])
    # ax[3].set_xlim([0,11])
    ax[3].set_ylim([0,2])
    ax[3].set_yticks([0,1,2])
  
    ax[3].set_aspect(1.0/ax[3].get_data_ratio(), adjustable='box')
    ax[3].set_xlabel('Model weight')
    ax[3].set_ylabel('Coefficient of variation')
    ax[3].legend_.set_visible(False)
    
    
    # ax[3].set_ylabel('')
    # ax[3].set_yticks([])
    sns.despine(offset = 3, trim = True, ax = ax[3])

    
    f.tight_layout()

