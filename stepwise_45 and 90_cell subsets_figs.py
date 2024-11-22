#%%
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
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, kurtosis

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# plt.style.use('seaborn')

# results_dir = 'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']

subjects = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B',
            'SF171107', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
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

file_paths_subsets = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), 
                    'stim_decoding_results_stepwise_selection_45 and 90 only_with cell stats_set features 10 cells_from varying pool sizes_*000 repeats_*'))[-1] 
            for i in range(len(subjects))]

file_paths_all = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), 
                    'stim_decoding_results_stepwise_selection_45 and 90 only_all cells*'))[-1] 
            for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel/OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%%

# For each repeat and cell number, calculate accuracy, log loss, mean response, std response, 
# cv of response, response index, proportion of decision function (only calculated for full model)

df_sw = pd.DataFrame()
df_sw_all = pd.DataFrame()

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode_subsets = np.load(file_paths_subsets[i],allow_pickle = True)[()]
    
    pool_sizes = decode_subsets['pool_sizes']
          
    stim = decode_subsets['stim']
    
    pred_stim = decode_subsets['pred_stim']
    
    features = decode_subsets['features']
       
    num_trials, num_features, num_pools, num_repeats = pred_stim.shape
        
    accu = pred_stim == np.tile(stim.reshape(-1,1,1,1), (1,num_features,num_pools,num_repeats))
    accu = accu.sum(0)/len(stim)
    
    accu = accu.mean(-1)
    
    prob_stim = decode_subsets['prob_stim']
    prob_stim = prob_stim.reshape(num_trials,2,-1)
    
    log_loss = np.zeros((num_features,num_pools,num_repeats))
    log_loss = log_loss.reshape(-1,)
    
    for l in range(prob_stim.shape[2]):
        log_loss[l] = skm.log_loss(stim,prob_stim[:,:,l])
    
    log_loss = log_loss.reshape(num_features,num_pools,num_repeats)
    log_loss = log_loss.mean(-1)
   
    # Average responses for all neurons
    trials = decode_subsets['trials_test'][:,features.flatten()]
    mu = np.array([trials[stim==s,:].mean(0) for s in np.unique(stim)])
    sigma = np.array([trials[stim==s,:].std(0) for s in np.unique(stim)])
    resp_index = ((np.diff(mu,axis=0))/mu.sum(0)).flatten()
    mu_pref = mu[np.argmax(mu,axis=0),np.arange(mu.shape[1])]
    sigma_pref = sigma[np.argmax(mu,axis=0),np.arange(mu.shape[1])]
    cv = sigma_pref/mu_pref
    
    mu_pref = mu_pref.reshape(num_features,num_pools,num_repeats).mean(-1)
    sigma_pref = sigma_pref.reshape(num_features,num_pools,num_repeats).mean(-1)
    cv = cv.reshape(num_features,num_pools,num_repeats).mean(-1)
    
    resp_index = np.diff(mu,axis=0)/mu.sum(0)
    resp_index = resp_index.reshape(num_features,num_pools,num_repeats)
    resp_index = np.abs(resp_index).mean(-1)
    
    trials_z = decode_subsets['trials_test_z'][:,features.flatten()]
    mu_z = np.array([trials_z[stim==s,:].mean(0) for s in np.unique(stim)])
    mu_z = mu_z.reshape(2,num_features,num_pools,num_repeats)
    
    # Model weights for all features/cells
    weights = np.zeros((num_features,num_pools,num_repeats))
    weights_expt = decode_subsets['model_weights'][-1,...]
    for p in range(num_pools):
        for r in range(num_repeats):
            weights[:,p,r] = weights_expt[p,r]
    
    # Get proportion of decision function for each cell
    df = mu_z*weights
    df = np.transpose(df,[1,0,2,3])
    
    prop_df = df / df.sum(0)
    prop_df = prop_df.mean(1).squeeze()
    prop_df = prop_df.mean(-1)
    
    accu_all = decode_subsets['pred_stim_full_pool'] == np.tile(stim.reshape(-1,1,1), (1,num_pools,num_repeats))
    accu_all = accu_all.sum(0)/len(stim)
    
    accu_all = accu_all.mean(-1)
    
    log_loss_all = np.zeros((num_pools,num_repeats))
    log_loss_all = log_loss_all.reshape(-1,)
    
    prob_stim = decode_subsets['prob_stim_full_pool']
    prob_stim = prob_stim.reshape(num_trials,2,-1)
    
    for l in range(prob_stim.shape[2]):
        log_loss_all[l] = skm.log_loss(stim,prob_stim[...,l])
        
    log_loss_all = log_loss_all.reshape(num_pools,num_repeats)
    log_loss_all = log_loss_all.mean(-1)
    
    
    df_sw = pd.concat([df_sw,pd.DataFrame({'accuracy' : accu.flatten()*100,
                                           'log_loss' : log_loss.flatten(),
                                           'prop_df' : prop_df.flatten(),
                                           'mu' : mu_pref.flatten(),
                                           'sigma' : sigma_pref.flatten(),
                                           'cv' : cv.flatten(),
                                           'resp_index' : resp_index.flatten(),
                                           'subject' : np.repeat(subjects[i],resp_index.size),
                                           'trained' : np.repeat(trained[i],resp_index.size),
                                           'pool_size' : np.tile(np.array(decode_subsets['pool_sizes']).reshape(1,-1),(num_features,1)).flatten(),
                                           'cell_num' : np.tile(np.arange(num_features).reshape(-1,1),(1,num_pools)).flatten()+1})])
    
    df_sw_all = pd.concat([df_sw_all,pd.DataFrame({'accuracy' : accu_all.flatten()*100,
                                                   'log_loss' : log_loss_all.flatten(),
                                                   'subject' : np.repeat(subjects[i],accu_all.size),
                                                   'trained' : np.repeat(trained[i],accu_all.size),
                                                   'pool_size' : np.array(decode_subsets['pool_sizes']).flatten()})])
    
    
df_sw = df_sw.reset_index(drop=True)

#%% Plot accuracy, log loss, prop of dec fun, mean, std, cv as function of cell num


y = ['accuracy','log_loss','prop_df','resp_index','mu','sigma','cv']
ylabel = ['Accuracy (% correct)','Log loss', 'Prop. of dec. fun.','Resp. index',
          'Mean resp', 'Std of resp', 'Coef. of var.']
ylims = [(84,100),(0,0.34),(0,0.6),(0.3,1),(0.38,1),(0.2,0.7),(0.2,1.5)]

yticks = [np.linspace(y[0],y[1],3) for y in ylims]


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "axes.linewidth":0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":0.5,
                                        "ytick.major.size":0.5,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":4,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    f,a = plt.subplots(7,num_pools, sharex = False, sharey = False,
                       figsize = (4.75,5.5))
    
    pools = df_sw.pool_size.unique()[::-1]
    
    for i,p in enumerate(pools):
        
        for r in range(len(y)):
        
            sns.lineplot(data = df_sw[df_sw.pool_size==p], x = 'cell_num', y = y[r],
                         errorbar = ('se',1), hue = 'trained', ax = a[r,i],
                         palette = 'colorblind', legend = False)
            a[r,i].set_ylim(ylims[r])
            a[r,i].set_yticks(yticks[r])
            a[r,i].set_ylabel(ylabel[r])
            
            # if r < 0:
            #     a[r,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if r == 0:
                a[r,i].set_title('Neuron pool size - ' + str(int(p)))
        
            if i == 0:
                if r < 6:
                    a[r,i].set_xticks([])
                    a[r,i].set_xlabel('')
                    sns.despine(ax=a[r,i],bottom=True,offset=3,trim=True)
                    
                else:
                    a[r,i].set_xticks([1,5,10])
                    a[r,i].set_xlabel('Number of cells')
                    sns.despine(ax=a[r,i],offset=3,trim=True)
                    
            if i > 0:
                if r < 6:
                    a[r,i].set_xticks([])
                    a[r,i].set_yticks([])
                    a[r,i].set_xlabel('')
                    a[r,i].set_ylabel('')
                    sns.despine(ax=a[r,i],bottom=True,left=True,offset=3,trim=True)
                   
                else:
                    a[r,i].set_xticks([1,5,10])
                    a[r,i].set_xlabel('Number of cells')
                    a[r,i].set_yticks([])
                    a[r,i].set_ylabel('')
                    sns.despine(ax=a[r,i],left=True,offset=3,trim=True)

                
    
    f.tight_layout()





#%% Line and scatter plots of accuracy only

n_subjects,n_cells,n_pools = accu_all.shape

sub_labels = np.tile(np.array(subjects).reshape(-1,1,1), (1,n_cells,n_pools))
cond_labels = np.tile(np.array(trained).reshape(-1,1,1), (1,n_cells,n_pools))
num_cells_labels = np.tile(np.arange(1,n_cells+1).reshape(1,-1,1),(n_subjects,1,n_pools))
cell_pool_labels = np.tile(np.hstack([np.array(decode_subsets['pool_sizes'],dtype='object'),['all']]).reshape(1,1,-1),(n_subjects,n_cells,1))

# cell_pool_labels = np.tile(np.array([10,20,40,80]).reshape(1,-1,1),(10,1,10))

df_results = pd.DataFrame({'Accuracy' : accu_all.flatten()*100,
                           'Number of cells' : num_cells_labels.flatten(),
                           'Cells in pool' : cell_pool_labels.flatten(),
                           'Subject' : sub_labels.flatten(),
                           'Trained' : cond_labels.flatten(),
                           'CV' : std_resp_all.flatten()/mean_resp_all.flatten()})


point_colors = sns.color_palette('flare', 10)
# point_colors = sns.color_palette('Set1', 10)
point_colors = [point_colors[i] for i in [0,1,4,9]]
# point_colors_m = {s:c for s,c in zip([1,2,5,10],point_colors)}

cond_colors = sns.color_palette('colorblind',2)
cond_colors_m = {False : cond_colors[0],
                 True : cond_colors[1]}

point_styles = [r'$1$',r'$2$',r'$5$',r'$10$']
point_sizes = [2,4,10,20]
    
palette = sns.color_palette('colorblind',2)

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
                                        "lines.linewidth":0.75,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    f0,ax0 = plt.subplots(4,n_pools, sharey = False, sharex = False, figsize = (5.5,6))
    
    uni_pools = df_results['Cells in pool'].unique()
    uni_pools = uni_pools[-1::-1]
    
    for i in range(n_pools):
        
        # Accuracy plots
        ind = df_results['Cells in pool']==uni_pools[i]
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Accuracy', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[0,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'Accuracy', 
                        ax = ax0[0,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
           
        ax0[0,i].set_ylim([50,105])
        ax0[0,i].plot([1,10],[100,100],'--k', zorder = 0)
        ax0[0,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[0,i].set_xlim([0.5,12])
        ax0[0,i].set_xticks([1,10])
        
        if i > 0:
            ax0[0,i].scatter(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(), edgecolor = palette[0], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].scatter(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(), edgecolor = palette[1], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].errorbar(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[0::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[0,i].errorbar(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[1::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'Accuracy'].to_numpy(),
                            y = df_results.loc[ind_T,'Accuracy'].to_numpy(),
                            ax = ax0[1,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[1,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[1,i].set_ylabel('Proficient accuracy (% correct)')
        ax0[1,i].set_xlabel('Naïve accuracy (% correct)')
        ax0[1,i].set_box_aspect(1)
        ax0[1,i].set_title(uni_pools[i])
        ax0[1,i].set_xlim([80,102])
        ax0[1,i].set_ylim([80,102])
        ax0[1,i].plot([80,100],[80,100],'k--', zorder = 0)
        ax0[1,i].set_title('')
            
        if i > 0:
            sns.despine(ax=ax0[0,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[1,i],left = True, trim = True, offset = 3)
            ax0[0,i].set_ylabel('')
            ax0[0,i].set_yticks([])
            ax0[1,i].set_ylabel('')
            ax0[1,i].set_yticks([])
            ax0[1,i].set_xticks(np.linspace(80,100,5))
        else:
            sns.despine(ax=ax0[0,i], trim = True, offset = 3)
            sns.despine(ax=ax0[1,i], trim = True, offset = 3)
            ax0[1,i].set_xticks(np.linspace(80,100,5))
            ax0[1,i].set_yticks(np.linspace(80,100,5))
        
        
        # CV plots
        
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'CV', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[2,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'CV', 
                        ax = ax0[2,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
        
        ax0[2,i].set_ylim([0,3])
        # ax0[2,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[2,i].set_xlim([0.5,12])
        ax0[2,i].set_xticks([1,10])
        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'CV'].to_numpy(),
                            y = df_results.loc[ind_T,'CV'].to_numpy(),
                            ax = ax0[3,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[3,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[3,i].set_ylabel('Proficient CV')
        ax0[3,i].set_xlabel('Naïve CV')
        # ax0[3,i].set_box_aspect(1)
        ax0[3,i].set_title(uni_pools[i])
        ax0[3,i].set_xlim([0,3])
        ax0[3,i].set_ylim([0,3])
        ax0[3,i].plot([0,3],[0,3],'k--', zorder = 0)
        ax0[3,i].set_title('')
        
        if i > 0:
            sns.despine(ax=ax0[2,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[3,i],left = True, trim = True, offset = 3)
            ax0[2,i].set_ylabel('')
            ax0[2,i].set_yticks([])
            ax0[3,i].set_ylabel('')
            ax0[3,i].set_yticks([])
            ax0[3,i].set_xticks(np.linspace(0,3,3))
        else:
            sns.despine(ax=ax0[2,i], trim = True, offset = 3)
            sns.despine(ax=ax0[3,i], trim = True, offset = 3)
            ax0[3,i].set_xticks(np.linspace(0,3,3))
            ax0[3,i].set_yticks(np.linspace(0,3,3))
        
        
    for a in ax0.flat:
        a.set_box_aspect(1)
     
            
    # f0.tight_layout()


#%% Line and scatter plots of accuracy and logloss

point_colors = sns.color_palette('flare', 10)
# point_colors = sns.color_palette('Set1', 10)
point_colors = [point_colors[i] for i in [0,1,4,9]]
# point_colors_m = {s:c for s,c in zip([1,2,5,10],point_colors)}

cond_colors = sns.color_palette('colorblind',2)
cond_colors_m = {False : cond_colors[0],
                 True : cond_colors[1]}

point_styles = [r'$1$',r'$2$',r'$5$',r'$10$']
point_sizes = [2,4,10,20]
    
palette = sns.color_palette('colorblind',2)

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
                                        "lines.linewidth":0.75,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    f0,ax0 = plt.subplots(4,n_pools, sharey = False, sharex = False, figsize = (4.75,5))
    
    uni_pools = df_results['Cells in pool'].unique()
    uni_pools = uni_pools[-1::-1]
    
    for i in range(n_pools):
        
        # Accuracy plots
        ind = df_results['Cells in pool']==uni_pools[i]
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Accuracy', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[0,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'Accuracy', 
                        ax = ax0[0,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
           
        ax0[0,i].set_ylim([50,105])
        ax0[0,i].plot([1,10],[100,100],'--k', zorder = 0)
        ax0[0,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[0,i].set_xlim([0.5,12])
        ax0[0,i].set_xticks([1,10])
        ax0[0,i].yaxis.set_label_coords(x=-0.35,y=0.5)
        ax0[0,i].xaxis.set_label_coords(x=0.45,y=-0.3)


        
        if i > 0:
            ax0[0,i].scatter(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(), edgecolor = palette[0], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].scatter(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(), edgecolor = palette[1], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[0,i].errorbar(11,(np.fliplr(accu_all_full_pool)[0::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[0::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[0,i].errorbar(11.5,(np.fliplr(accu_all_full_pool)[1::2,i-1]*100).mean(),
                              (np.fliplr(accu_all_full_pool)[1::2,i-1]*100).std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[0,i].text(11.25,88, 'all', horizontalalignment='center')

        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'Accuracy'].to_numpy(),
                            y = df_results.loc[ind_T,'Accuracy'].to_numpy(),
                            ax = ax0[1,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[1,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[1,i].set_ylabel('Proficient accuracy (% correct)')
        ax0[1,i].yaxis.set_label_coords(x=-0.35,y=0.45)
        ax0[1,i].set_xlabel('Naive accuracy (% correct)')
        ax0[1,i].xaxis.set_label_coords(x=0.45,y=-0.3)
        ax0[1,i].set_box_aspect(1)
        ax0[1,i].set_title(uni_pools[i])
        ax0[1,i].set_xlim([80,102])
        ax0[1,i].set_ylim([80,102])
        ax0[1,i].plot([80,100],[80,100],'k--', zorder = 0)
        ax0[1,i].set_title('')
            
        if i > 0:
            sns.despine(ax=ax0[0,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[1,i],left = True, trim = True, offset = 3)
            ax0[0,i].set_ylabel('')
            ax0[0,i].set_yticks([])
            ax0[1,i].set_ylabel('')
            ax0[1,i].set_yticks([])
            ax0[1,i].set_xticks(np.linspace(80,100,5))
        else:
            sns.despine(ax=ax0[0,i], trim = True, offset = 3)
            sns.despine(ax=ax0[1,i], trim = True, offset = 3)
            ax0[1,i].set_xticks(np.linspace(80,100,5))
            ax0[1,i].set_yticks(np.linspace(80,100,5))
        
        
        # logloss plots
        
        sns.lineplot(data = df_results[ind], 
                     x = 'Number of cells', y = 'Log loss', 
                     hue = 'Trained', errorbar = ('se',1), ax = ax0[2,i], legend = False,
                     palette = palette, zorder = 2)
        sns.scatterplot(data = df_results[ind],
                        x = 'Number of cells', y = 'Log loss', 
                        ax = ax0[2,i], legend = False,
                        palette = palette,
                        ec = df_results.loc[ind,'Trained'].map(cond_colors_m), 
                        fc = 'none',
                        zorder = 1,
                        linewidth = 0.25)
        
        ax0[2,i].set_ylim([-0.01,0.4])
        ax0[2,i].set_title('Cell pool size - ' + str(uni_pools[i]))
        ax0[2,i].set_xlim([0.5,12])
        ax0[2,i].set_xticks([1,10])
        ax0[2,i].yaxis.set_label_coords(x=-0.35,y=0.5)
        ax0[2,i].xaxis.set_label_coords(x=0.45,y=-0.3)
        
        
        if i > 0:
            ax0[2,i].scatter(11,np.fliplr(log_loss_all_full_pool)[0::2,i-1].mean(), edgecolor = palette[0], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[2,i].scatter(11.5,np.fliplr(log_loss_all_full_pool)[1::2,i-1].mean(), edgecolor = palette[1], marker = 'o',
                          facecolor = 'none', zorder = 1, linewidths = 0.25)
            ax0[2,i].errorbar(11,np.fliplr(log_loss_all_full_pool)[0::2,i-1].mean(),
                              np.fliplr(log_loss_all_full_pool)[0::2,i-1].std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[2,i].errorbar(11.5,np.fliplr(log_loss_all_full_pool)[1::2,i-1].mean(),
                              np.fliplr(log_loss_all_full_pool)[1::2,i-1].std()/np.sqrt(5),
                              capsize = 0.5, color = 'k', zorder = 0,
                              linewidth = 0.5)
            ax0[2,i].text(11.25,0.18, 'all', horizontalalignment='center')
        
        
        for ff,f in enumerate([1,2,5,10]):
            # print(f)
            ind_T = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_T = ind_T & df_results['Trained']
            
            ind_N = np.logical_and(df_results['Cells in pool'] == uni_pools[i],
                                  df_results['Number of cells'] == f)
            ind_N = ind_N & ~df_results['Trained']
            
            if ff == 3:
                s = 40
            else: 
                s = 20
                
            sns.scatterplot(x = df_results.loc[ind_N,'Log loss'].to_numpy(),
                            y = df_results.loc[ind_T,'Log loss'].to_numpy(),
                            ax = ax0[3,i], legend = False, 
                            fc = 'none',
                            # marker = point_styles[ff],
                            s = point_sizes[ff],
                            ec = point_colors[ff], 
                            zorder = 1,
                            linewidth = 0.25)
        if i == 4:
            l = ax0[3,i].legend([1,2,5,10], frameon = False, fancybox = False,
                                ncol = 1, loc = 4, labelspacing = 0.1,
                                handletextpad = 0,
                                bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            
        ax0[3,i].set_ylabel('Proficient log loss')
        ax0[3,i].yaxis.set_label_coords(x=-0.35,y=0.45)
        ax0[3,i].set_xlabel('Naive log loss')
        ax0[3,i].xaxis.set_label_coords(x=0.45,y=-0.3)


        # ax0[3,i].set_box_aspect(1)
        ax0[3,i].set_title(uni_pools[i])
        ax0[3,i].set_xlim([-0.015,0.5])
        ax0[3,i].set_ylim([-0.015,0.5])
        ax0[3,i].plot([0,0.4],[0,0.4],'k--', zorder = 0)
        ax0[3,i].set_title('')
        
        if i > 0:
            sns.despine(ax=ax0[2,i],left = True, trim = True, offset = 3)
            sns.despine(ax=ax0[3,i],left = True, trim = True, offset = 3)
            ax0[2,i].set_ylabel('')
            ax0[2,i].set_yticks([])
            ax0[3,i].set_ylabel('')
            ax0[3,i].set_yticks([])
            ax0[3,i].set_xticks(np.linspace(0,0.4,3))
        else:
            sns.despine(ax=ax0[2,i], trim = True, offset = 3)
            sns.despine(ax=ax0[3,i], trim = True, offset = 3)
            ax0[3,i].set_xticks(np.linspace(0,0.4,3))
            ax0[3,i].set_yticks(np.linspace(0,0.4,3))
        
        
    for a in ax0.flat:
        a.set_box_aspect(1)
     
            
    f0.tight_layout()

