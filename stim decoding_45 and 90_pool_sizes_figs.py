# -*- coding: utf-8 -*-
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
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
# import patchworklib as pw
from scipy.stats import ttest_rel, wilcoxon, kurtosis
from sklearn.metrics.pairwise import cosine_similarity

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# plt.style.use('seaborn')

# results_dir = 'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'

subjects = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
            'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']
subject_unique = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B',
            'SF171107', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
            'SF180613']
expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]

trained = [False, True, False, True, False, True, False, True, False, True]

# Load most up-to-date saved results


file_paths_all = [glob.glob(join(results_dir, subjects[i], expt_dates[i],
                    str(expt_nums[i]),
                    'stim_decoding_results_45 and 90 only_pool_sizes*'))[-1]
            for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel/OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%%

df_accu = pd.DataFrame()

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode = np.load(file_paths_all[i],allow_pickle = True)[()]

    stim = decode['stim_ori_test']
    
    pools = list(decode.keys())
    pools.pop()

    accu = np.zeros(len(pools))
    discrim_fun = np.zeros(len(pools))
    d_prime = np.zeros(len(pools))
     
    for ip,p in enumerate(pools):
         
         pred_stim = decode[p]['pred_stim']
         dec_fun = decode[p]['decision_function']
         
         accu[ip] = ((pred_stim == stim[:,None]).sum(0)/pred_stim.shape[0]).mean()
         discrim_fun[ip] = np.abs(dec_fun).mean(1).mean()
         
         mu = np.array([dec_fun[stim==s,:].mean(0) for s in np.unique(stim)])
         var = np.array([dec_fun[stim==s,:].var(0) for s in np.unique(stim)])
         var = np.sqrt(var.sum(0))
         
         mu_diff = np.diff(mu,axis=0)
         d = mu_diff/var
         d_prime[ip] = np.abs(d).mean()
    
    df_accu = pd.concat([df_accu,pd.DataFrame({'pool' : pools,
                                               'accuracy' : accu,
                                               'discrim_fun' : discrim_fun,
                                               'd_prime' : d_prime,
                                               'subject' : np.repeat(subject_unique[i],len(pools)),
                                               'trained' : np.repeat(trained[i],len(pools))})],
                        ignore_index = True)
    
    

   

#%%

cond_colors = sns.color_palette('colorblind',2)
cond_colors_m = {False : cond_colors[0],
                 True : cond_colors[1]}

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={**axes_style('ticks'),
                                        'font.size':5,'axes.titlesize':5,
                                        'axes.labelsize':5,
                                        'axes.linewidth':0.5,
                                        'xtick.labelsize':5,
                                        'ytick.labelsize':5,
                                        'xtick.major.width':0.5,
                                        'ytick.major.width':0.5,
                                        'xtick.major.size':3,
                                        'ytick.major.size':3,
                                        'xtick.minor.width':0.25,
                                        'ytick.minor.width':0.25,
                                        'xtick.minor.size':1,
                                        'ytick.minor.size':1,
                                        'patch.linewidth':0.5,
                                        'lines.markersize':2,
                                        'lines.linewidth':0.5,
                                        'legend.fontsize':5,
                                        'legend.title_fontsize':5,
                                        'axes.spines.left': True,
                                        'axes.spines.bottom': True,
                                        'axes.spines.right': False,
                                        'axes.spines.top': False,
                                        'font.sans-serif' : 'Helvetica',
                                        'font.family' : 'sans-serif'}):


    df_plot = df_accu.copy()
    df_plot['accuracy'] = df_plot.accuracy * 100

    f,a = plt.subplots(1,1, figsize = (1.75,0.8))
    sns.lineplot(df_plot, x = 'pool', y = 'accuracy', hue = 'trained', errorbar = ('se',1), legend = True, palette = 'colorblind')
    sns.scatterplot(df_plot, x = 'pool', y = 'accuracy', 
                    ec = df_plot.trained.map(cond_colors_m), 
                    fc = 'none')
    a.set_xlabel('Pool size')
    a.set_ylabel('Accuracy (% correct)')
    a.set_ylim([47,103])
    a.set_yticks(np.linspace(50,100,6))
    a.set_xticks([1,10,25,50,100,200,400,800])
    a.set_xticklabels([1,None,None,None,100,200,400,800])
    sns.despine(ax=a, trim = True)
    # a.set_box_aspect(1)
    # f.show()


    f,a = plt.subplots(1,1, figsize = (1.75,0.8))
    sns.lineplot(df_plot, x = 'pool', y = 'accuracy', hue = 'trained', errorbar = ('se',1), legend = True, palette = 'colorblind')
    sns.scatterplot(df_plot, x = 'pool', y = 'accuracy', 
                    ec = df_plot.trained.map(cond_colors_m), 
                    fc = 'none')
    a.set_xlabel('Pool size')
    a.set_ylabel('Accuracy (% correct)')
    a.set_yticks(np.linspace(50,100,6))
    a.set_ylim([47,103])
    a.set_xscale('log')
    a.set_xticks([1,10,25,50,100,200,400,800])
    a.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    sns.despine(ax=a, trim = True)
    # a.set_box_aspect(1)
    # f.show()
    
    
    f,a = plt.subplots(1,1, figsize = (1.75,0.8))
    sns.lineplot(df_plot, x = 'pool', y = 'd_prime', hue = 'trained', errorbar = ('se',1), legend = False, palette = 'colorblind')
    sns.scatterplot(df_plot, x = 'pool', y = 'd_prime',  
                    ec = df_plot.trained.map(cond_colors_m), 
                    fc = 'none')
    a.set_xlabel('Pool size')
    a.set_ylabel('d-prime')
    a.set_yticks([0.2,2,20])
    a.set_ylim([0.1,20])
    a.set_yscale('log')
    a.set_xscale('log')
    a.set_xticks([1,10,25,50,100,200,400,800])
    a.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    a.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    sns.despine(ax=a, trim = True)
    plt.minorticks_off()
    # a.set_box_aspect(1)
    # f.show()

#%%

df_plot = df_accu.copy()

df_plot['trained'] = df_plot.trained.map({False : 'Naive',
                                          True : 'Proficient'})

df_plot = pd.pivot(df_plot, index = ['subject','pool'], columns = ['trained'],values = ['d_prime'])


f = (
     so.Plot(df_plot, x = 'Naive', y = 'Proficient', color = 'pool')
    )


#%%

def cond_label(x):
    if x:
        return 'Proficient'
    else:
        return 'Naive'


style = {**axes_style('ticks'),
         'font.size':5,'axes.titlesize':5,
         'axes.labelsize':5,
         'axes.linewidth':0.5,
         'xtick.labelsize':5,
         'ytick.labelsize':5,
         'xtick.major.width':0.5,
         'ytick.major.width':0.5,
         'xtick.major.size':3,
         'ytick.major.size':3,
         'patch.linewidth':0.5,
         'lines.markersize':2,
         'lines.linewidth':0.5,
         'legend.fontsize':5,
         'legend.title_fontsize':5,
         'axes.spines.left': True,
         'axes.spines.bottom': True,
         'axes.spines.right': False,
         'axes.spines.top': False,
         'font.sans-serif' : 'Helvetica',
         'font.family' : 'sans-serif'}


df_plot = df_accu.copy()
df_plot['accuracy'] = df_plot.accuracy * 100
df_plot['trained'] = df_plot.trained.map(cond_label)

p = (so.Plot(df_plot, 'pool', 'accuracy', color = 'trained')
     .layout(size=(1.5,1.5))
     .scale(x = so.Continuous(trans='log').tick(at=[1,10,100,1000]),
            y = so.Continuous().tick(every=10), 
            color = 'colorblind')
     .limit(y = (47,103))
     .add(so.Line(), so.Agg(), legend=False)
     .add(so.Band(), so.Est(), legend = False)
     .add(so.Dots(fill=False), so.Jitter(), legend = False)
     .label(x = 'Pool size', y = 'Accuracy (% correct)')
     .theme(style)
     .plot()
    )
  
p._figure.axes[0].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

# p._figure.axes[0].set_xticklabels(p._figure.axes[0].get_xticks().astype(int),fontdict={'fontsize' : 1})
# p._figure.axes[0].set_yticklabels(p._figure.axes[0].get_yticks().astype(int),fontdict={'fontsize' : 1})

sns.despine(ax=p._figure.axes[0], trim = True, offset = 3)

p.save(join(fig_save,'stim decoding_45 and 90_pool sizes_accuracy.svg'),format='svg')

df_plot = df_accu.copy()
df_plot['accuracy'] = df_plot.accuracy * 100
df_plot['trained'] = df_plot.trained.map(cond_label)
df_plot = pd.pivot(df_plot, columns = ['trained'],values = 'accuracy', index = ['pool','subject'])

p = (so.Plot(df_plot, 'Naive', 'Proficient', color = 'pool')
     .theme(style)
     .layout(size=(2,2))
     .add(so.Dots(fill=False), legend = False)
     .scale(color=so.Continuous(trans='log'))
     .plot()
    )
    

p._figure.axes[0].plot([50,100],[50,100],'--k', linewidth = 0.5)

p.show()

# f,a = plt.subplots(1,1)
# sns.lineplot(df_plot, x = 'pool', y = 'accuracy', hue = 'trained', errorbar = ('se',1), legend = False)
# sns.stripplot(df_plot, x = 'pool', y = 'accuracy', hue = 'trained', legend = False, fill=False)
# a.set_xlabel('Pool size')
# a.set_ylabel('Accuracy (% correct)')
# a.set_ylim([47,107])
# a.set_xscale('log')
# # sns.despine(ax=a, trim = True)
# a.legend([a.get_children()[i] for i in [0,2]], ['Naive','Proficient'], loc = 4, frameon = False)

# %%
