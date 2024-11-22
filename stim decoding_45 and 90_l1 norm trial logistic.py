#%%
"""
@author: Samuel
"""

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from os.path import join
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import seaborn as sns
import seaborn.objects as so
from scipy.stats import ttest_rel, wilcoxon, kurtosis
from scipy.special import expit

import statsmodels.api as sm
import statsmodels.formula.api as smf

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.family'] = "sans-serif"

# Add windows fonts
font_dirs = [r'C:\Windows\Fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# plt.style.use('seaborn')

# results_dir = 'H:/OneDrive for Business/Results'
results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'
# results_dir = 'C:/Users/samue/OneDrive - University College London/Results'

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
                    'stim_decoding_results_45 and 90 only_all cells*'))[-1] 
            for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft'
# fig_save = r'C:\Users\samue\OneDrive - University College London\Orthogonalization project\Figures\Draft'



#%%

df_prob_k = pd.DataFrame()

n_repeats = 2000

n_cells = 1600

# k_values = [1,10,1000,2000,3000,4000,np.e*1600,4500,4800,5000,6000,7000,8000,10000]
k_values = [2500]
n_k = len(k_values)

def trained_label(x):
    if x:
        return 'Proficient'
    else:
        return 'Naive'

# Function to generate the same random indices every time by setting seed to the interable
def random_cells(total_cells,n_cells,i):
    np.random.seed(i)
    return np.random.choice(total_cells,n_cells,replace=False)

def logistic(x,k):
    return expit(x*k)
    # return 1./(1.+np.exp(-k*x)) 

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode_all = np.load(file_paths_all[i],allow_pickle = True)[()]
    
    n_trials = len(decode_all['stim'])
    
    stim_prob = np.zeros((n_trials,n_k,n_repeats))
    l1_trial_dot = np.zeros((n_trials,n_k,n_repeats))
     
    cell_inds = [random_cells(decode_all['trials_train'].shape[1],n_cells,r) 
                 for r in range(n_repeats)]
    
    for ik,k in enumerate(k_values):
        for ir,c in enumerate(cell_inds):
        
            trials_train = decode_all['trials_train'][:,c]
            stim_train = decode_all['stim_train']
            
            mu_train = np.array([trials_train[stim_train==s,:].mean(0) for s in np.unique(stim_train)])
            # l1_mu_train = mu_train/mu_train.sum(1).reshape(-1,1)
            l1_mu_train = mu_train/np.linalg.norm(mu_train, ord=1, axis=1).reshape(-1,1)
            l1_mu_train = np.diff(l1_mu_train,axis=0)
                        
            trials_test = decode_all['trials_test'][:,c]
            
            # l1_trials_test = trials_test/trials_test.sum(1).reshape(-1,1)
            l1_trials_test = trials_test/np.linalg.norm(trials_test, ord=1, axis=1).reshape(-1,1)

                
            l1_trial_dot[:,ik,ir] = l1_mu_train @ l1_trials_test.T
            
            stim_prob[:,ik,ir] = np.squeeze(logistic(l1_trial_dot[:,ik,ir],k))
            
    
    df_prob_k  = pd.concat([df_prob_k , pd.DataFrame({'subject' : np.repeat(subject_unique[i],n_trials*n_repeats*n_k),
                                                      'subject_uni' : np.repeat(i,n_trials*n_repeats*n_k),
                                                      'trained' : np.repeat(trained[i],n_trials*n_repeats*n_k),
                                                      'stim_prob' : stim_prob.flatten(),
                                                      'x_trials' : l1_trial_dot.flatten(),
                                                      'stim' : np.tile(decode_all['stim'][:,None,None], (1,n_k,n_repeats)).flatten(),
                                                      'repeat' : np.tile(np.arange(n_repeats)[None,None,:], (n_trials,n_k,1)).flatten(),
                                                      'k_value' : np.tile(np.array(k_values)[None,:,None], (n_trials,1,n_repeats)).flatten(),
                                                      'trial' : np.tile(np.arange(n_trials)[:,None,None], (1,n_k,n_repeats)).flatten()})],
                                                      ignore_index = True)
    

df_prob_k ['trained'] = df_prob_k .trained.apply(trained_label)


#%% 
from seaborn import axes_style
import seaborn.objects as so
from sklearn.metrics import log_loss



style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style['axes.labelsize'] = 6
style['xtick.labelsize'] = 6
style['ytick.labelsize'] = 6
style['legend.fontsize'] = 6
style['legend.title_fontsize'] = 6
style['legend.frameon'] = False
style['font.family'] = 'sans-serif'
style['font.sans-serif'] = 'Helvetica'
style['figure.dpi'] = 1000


%matplotlib qt

df_plot = df_prob_k.groupby(['subject','trained','k_value','trial'], observed = True).agg({'stim' : 'first',
                                                                                           'stim_prob' : 'mean',
                                                                                           'x_trials' : 'mean'}).reset_index()

# df_plot = df_prob_k.copy()

(
    so.Plot(df_plot, x = 'k_value', y = 'stim_prob', color = 'stim', linestyle = 'trained')
    .add(so.Line(), so.Agg())
    # .add(so.Band(), so.Est(errorbar = ('se',1)))
    .scale(color = so.Nominal())
    .label(y='Probability')
    .show()
)

k = 2500

(
    so.Plot(df_plot[(df_plot.k_value==k) & (df_plot.subject=='SF180613')],x = 'stim', y = 'stim_prob', color = 'trained')
    .add(so.Dot())
    .scale(color = so.Nominal())
    .label(y='Probability')
    .show()
)

(
    so.Plot(df_plot[df_plot.k_value==k],x = 'x_trials', y = 'stim_prob', 
            color = 'trained', marker = 'stim', fill = None)
    .layout(engine='tight', size=(5,5))
    .add(so.Dots(pointsize=5), legend = False)
    .scale(color = 'colorblind')
    .label(y='Prob. choose right',
           x = 'z')
    .theme({**style})
    # .limit(x=(-0.0008,0.0008), y=(0,1))
    .show()
)



df_plot = df_prob_k .groupby(['subject','trained','k_value','trial']).mean().reset_index()
df_plot = df_plot.groupby(['subject','trained','k_value']).apply(lambda x: log_loss(x.stim==90,x.stim_prob)).reset_index()
df_plot = df_plot.rename(columns = {0 : 'log_loss'})

(
    so.Plot(df_plot, x = 'k_value', y = 'log_loss', color = 'trained')
    .add(so.Line(), so.Agg())
    # .add(so.Band(), so.Est(errorbar = ('se',1)))
    .scale(color = so.Nominal())
    .show()
)

(
    so.Plot(df_plot[df_plot.k_value==4500],x = 'trained', y = 'log_loss', color = 'trained')
    .add(so.Dot())
    .scale(color = so.Nominal())
    .show()
)



df_plot = df_prob_k .groupby(['subject','trained','k_value','trial']).mean().reset_index()
df_plot = df_plot.groupby(['subject','trained','k_value','stim']).mean().reset_index()

df_plot['prob_correct'] = df_plot.stim_prob
df_plot.loc[df_plot.stim==45, 'prob_correct'] = 1 - df_plot[df_plot.stim==45].stim_prob

df_plot['prob_correct'] = df_plot['prob_correct'] * 100

df_plot = df_plot.groupby(['subject','trained']).mean().reset_index()

p = (
        so.Plot(df_plot[df_plot.k_value==k], x='trained', y='prob_correct', group='subject')
        .layout(engine='tight', size=(1.5,1.9))
        .theme({**style})
        .add(so.Dots(color='black', pointsize=2), legend=False)
        .add(so.Line(color='black'))
        # .add(so.Dash(width=0.2), so.Agg(), legend=False)
        # .add(so.Range(), so.Est(errorbar = ('se',1)), legend=False)
        .scale(color='colorblind')
        .label(x='', y='Prob. correct choice')
        .limit(y=[55,85])
        .scale(y=so.Continuous().tick(every=10))
        .plot()
    )

sns.despine(ax=p._figure.axes[0], trim=True)

p.save(join(fig_save,'ffinhib_choice_modeling.svg'), format='svg')

#%% Vary number of cells

df_prob_cell = pd.DataFrame()

n_repeats = 2000

n_cells = [50,100,200,500,1000,1600]
n_cell_vals = len(n_cells)


def trained_label(x):
    if x:
        return 'Proficient'
    else:
        return 'Naive'

# Function to generate the same random indices every time by setting seed to the interable
def random_cells(total_cells,n_cells,i):
    np.random.seed(i)
    return np.random.choice(total_cells,n_cells,replace=False)

def logistic(x,k):
    return expit(x*k)
    # return 1./(1.+np.exp(-k*x)) 

for i in range(len(subjects)):


    print('Loading ' + subjects[i])
    decode_all = np.load(file_paths_all[i],allow_pickle = True)[()]
    
    n_trials = len(decode_all['stim'])
    
    stim_prob = np.zeros((n_trials,n_cell_vals,n_repeats))

    for ic,n_c in enumerate(n_cells):
        
        print(f'{n_c} cells')

        cell_inds = [random_cells(decode_all['trials_train'].shape[1],n_c,r) 
                    for r in range(n_repeats)]
        
        for ir,c in enumerate(cell_inds):
        
            trials_train = decode_all['trials_train'][:,c]
            stim_train = decode_all['stim_train']
            
            mu_train = np.array([trials_train[stim_train==s,:].mean(0) for s in np.unique(stim_train)])
            l1_mu_train = mu_train/mu_train.sum(1).reshape(-1,1)
            l1_mu_train = np.diff(l1_mu_train,axis=0)
                        
            trials_test = decode_all['trials_test'][:,c]
            
            l1_trials_test = trials_test/trials_test.sum(1).reshape(-1,1)
                
            l1_trial_dot = l1_mu_train @ l1_trials_test.T
            
            stim_prob[:,ic,ir] = np.squeeze(logistic(l1_trial_dot,np.e*n_c))
                
        
    df_prob_cell = pd.concat([df_prob_cell, pd.DataFrame({'subject' : np.repeat(subject_unique[i],n_trials*n_repeats*n_cell_vals),
                                                          'subject_uni' : np.repeat(i,n_trials*n_repeats*n_cell_vals),
                                                          'trained' : np.repeat(trained[i],n_trials*n_repeats*n_cell_vals),
                                                          'stim_prob' : stim_prob.flatten(),
                                                          'stim' : np.tile(decode_all['stim'][:,None,None], (1,n_cell_vals,n_repeats)).flatten(),
                                                          'repeat' : np.tile(np.arange(n_repeats)[None,None,:], (n_trials,n_cell_vals,1)).flatten(),
                                                          'n_cells' : np.tile(np.array(n_cells)[None,:,None], (n_trials,1,n_repeats)).flatten(),
                                                          'trial' : np.tile(np.arange(n_trials)[:,None,None], (1,n_cell_vals,n_repeats)).flatten()})],
                                                          ignore_index = True)
    

df_prob_cell['trained'] = df_prob_cell.trained.apply(trained_label)


#%% 

import seaborn.objects as so
from sklearn.metrics import log_loss


df_plot = df_prob_cell .groupby(['subject','trained','n_cells','stim','trial'])['stim_prob'].mean().reset_index()
df_plot = df_plot .groupby(['subject','trained','n_cells','stim'])['stim_prob'].mean().reset_index()

(
    so.Plot(df_plot, x = 'n_cells', y = 'stim_prob', color = 'stim', linestyle = 'trained')
    .add(so.Line(), so.Agg())
    # .add(so.Band(), so.Est(errorbar = ('se',1)))
    .scale(color = so.Nominal())
    .show()
)

df_plot = df_prob_cell .groupby(['subject','trained','n_cells','trial']).mean().reset_index()
df_plot = df_plot.groupby(['subject','trained','n_cells']).apply(lambda x: log_loss(x.stim==90,x.stim_prob)).reset_index()
df_plot = df_plot.rename(columns = {0 : 'log_loss'})

(
    so.Plot(df_plot, x = 'n_cells', y = 'log_loss', color = 'trained')
    .add(so.Line(), so.Agg())
    # .add(so.Band(), so.Est(errorbar = ('se',1)))
    .scale(color = so.Nominal())
    .show()
)




#%% 
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, auc, log_loss
import seaborn.objects as so


df_perf = df_dot.groupby(['subject','trained', 'repeat']).apply(lambda x: log_loss(x.stim==90,x.stim_prob))
df_perf = df_perf.groupby(['subject','trained']).mean().reset_index()

fpr, tpr, thresholds = roc_curve(df_dot[df_dot.trained=='Naive'].stim==90, df_dot[df_dot.trained=='Naive'].stim_prob)
roc_auc = auc(fpr, tpr)

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

display.plot()

#%%

df_plot = df_dot.copy()
df_plot['stim'] = df_plot.stim.astype(int)
df_plot['dot_diff'] = df_plot.l1_trial_dot_45 - df_plot.l1_trial_dot_90

df_plot['ec'] = df_plot.stim.map({45 : (0,1,0),
                            90 : (1,0,1)})
df_plot['fc'] = df_plot.ec
df_plot.loc[df_plot.trained=='Naive','fc'] = 'none'

df_mu = df_plot.groupby(['subject','trained','stim'],as_index=False).mean()

# df_mu = pd.pivot(df_mu, values = 'dot_diff', columns = ['stim','trained'], index = 'subject')

dist_45_90 = np.zeros(10)

for i,(s,t) in enumerate(zip(subject_unique,trained)):
    ind = np.logical_and(df_mu.subject==s,df_mu.trained==trained_label(t))
    
    dist_45_90[i] = np.sqrt((df_mu[ind & (df_mu.stim==45)].l1_trial_dot_45.to_numpy() - df_mu[ind & (df_mu.stim==90)].l1_trial_dot_45.to_numpy())**2 +
                            (df_mu[ind & (df_mu.stim==45)].l1_trial_dot_90.to_numpy() - df_mu[ind & (df_mu.stim==90)].l1_trial_dot_90.to_numpy())**2)


c_45 = (0,1,0)
c_90 = (1,0,1)

stim_colors = [(0,1,0),(1,0,1)]

cond_colors = sns.color_palette('colorblind',2)

bins = np.linspace(-0.0009,0.0009,41)

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
    
    
    f,ax = plt.subplots(1,3, figsize=(5,2.5), sharex = False, sharey = False)
    
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams['figure.dpi'] = 100
    
    sns.scatterplot(data = df_plot, 
                    x = 'l1_trial_dot_45',
                    y = 'l1_trial_dot_90',
                    hue = 'trained',
                    palette = 'colorblind',
                    style = 'stim',
                    ec = 'k',
                    markers = {45 : 'o',
                               90 : 'v'},
                    # ec = df_plot.ec,
                    # fc = df_plot.fc,                 
                    legend = False,
                    ax = ax[0])
    
    ax[0].set_xlim([0.0005,0.0017])
    ax[0].set_ylim([0.0005,0.0017])
    ax[0].plot([0.0005,0.0017],[0.0005,0.0017], '--k')
    ax[0].set_xlabel(r'$f \cdot f_{45}$')
    ax[0].set_ylabel(r'$f \cdot f_{90}$')
    ax[0].set_xticks(np.linspace(0.0005,0.0017,3))
    ax[0].set_yticks(np.linspace(0.0005,0.0017,3))
    
    # sns.histplot(data = df_plot[df_plot.trained=='Naive'],
    #              x = 'dot_diff',
    #              hue = 'stim',
    #              palette = stim_colors,
    #              fill = False,
    #              legend = False,
    #              bins = bins,
    #              stat = 'probability',
    #              element = 'step',
    #              ax = ax[1])
    # sns.histplot(data = df_plot[df_plot.trained=='Proficient'],
    #              x = 'dot_diff',
    #              hue = 'stim',
    #              palette = stim_colors,
    #              fill = True,
    #              legend = False,
    #              bins = bins,
    #              stat = 'probability',
    #              element = 'step',
    #              alpha = 0.5,
    #              ax = ax[1])
    
    sns.histplot(data = df_plot[df_plot.trained=='Naive'],
                 x = 'dot_diff',
                 color = cond_colors[0],
                 fill = True,
                 legend = False,
                 bins = bins,
                 stat = 'probability',
                 element = 'step',
                 ax = ax[1])
    
    sns.histplot(data = df_plot[df_plot.trained=='Proficient'],
                 x = 'dot_diff',
                 color = cond_colors[1],
                 fill = True,
                 legend = False,
                 bins = bins,
                 stat = 'probability',
                 element = 'step',
                 ax = ax[1])
    
    
    ax[1].set_xlabel(r'$f \cdot (f_{45} - f_{90})$')
    ax[1].set_xticks(np.linspace(bins[0],bins[-1],3))
    ax[1].text(-0.00055,0.15,r'$90\degree$', horizontalalignment='center')
    ax[1].text(0.00055,0.15,r'$45\degree$', horizontalalignment='center')
    
    
    sns.scatterplot(x = dist_45_90[0::2],
                    y = dist_45_90[1::2],
                    ec = 'k',
                    fc = 'k',
                    ax = ax[2])
    
    ax[2].set_xlim([0.0001,0.0009])
    ax[2].set_ylim([0.0001,0.0009])
    
    ax[2].plot([0.0001,0.0009],[0.0001,0.0009],'--k')
    ax[2].set_xlabel(r'Naive : dist. ($f \cdot f_{45}, f \cdot f_{90})$')
    ax[2].set_ylabel(r'Proficient : dist. ($f \cdot f_{45}, f \cdot f_{90})$')
    ax[2].set_xticks(np.linspace(0.0001,0.0009,3))
    ax[2].set_yticks(np.linspace(0.0001,0.0009,3))
    
    
    for a in ax:
        a.set_box_aspect(1)
        sns.despine(ax=a, trim = True, offset = 3)
    
    # ax[1].plot([0.0006,0.0016],[0.0006,0.0016], '--k')
    
    # sns.histplot(data = df_plot[df_plot.trained=='Naive'],
    #              x = 'dot_diff',
    #              hue = 'stim',
    #              palette = stim_colors,
    #              legend = False,
    #              ax = ax[0,1])
    # sns.histplot(data = df_plot[df_plot.trained=='Proficient'],
    #              x = 'dot_diff',
    #              hue = 'stim',
    #              palette = stim_colors,
    #              legend = False,
    #              ax = ax[0,2])
    
    # sns.despine()

    f.tight_layout()


#%% 


df_mu = df_dot.copy()

df_mu['dot_diff'] = df_mu.l1_trial_dot_45-df_mu.l1_trial_dot_90

# df_mu = df_mu.groupby(['subject','trained','stim'],as_index=False).mean()

sns.displot(data = df_mu, x = 'dot_diff', hue = 'stim', col = 'trained', 
            kind = 'hist')


#%% Plot results on a logistic function

df_plot = df_prob_k.groupby(['subject','trained','stim','trial'])['stim_prob'].mean().reset_index()

df_plot = df_plot.groupby(['subject','trained','stim'])['stim_prob'].mean().reset_index()
df_plot['prob_left'] = 1 - df_plot['stim_prob']

x = np.linspace(-8,8,5000)
y = logistic(x,1)

df_plot['x_pos'] = df_plot['prob_left'].apply(lambda p: x[np.argmin(np.abs(y-p))])

cond_colors = sns.color_palette('colorblind',2)

ind = df_plot.subject=='SF170620B'


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


    f,a = plt.subplots(1,2, figsize=(1,1))

    p0 = (
            so.Plot(df_plot[(df_plot.trained=='Naive') & ind], x='x_pos', y='prob_left', marker='stim')
            .add(so.Dot(color=cond_colors[0], edgecolor='black', pointsize=2), legend=False)
            .limit(y=[-0.025,1.025])
            .scale(x=so.Continuous().tick(every=4, between=(-8,8)),
                   y=so.Continuous().tick(every=0.2, between=(0,1)),
                   marker={45 : 'o', 90 : 'v'})
            .on(a[0])
            .plot()
            
        )

    a[0].plot(x,y,'-k', zorder=-1)
    a[0].set_box_aspect(1)
    sns.despine(ax=a[0], trim=True)
    a[0].set_xticks([])
    a[0].set_yticks([])
    a[0].set_ylabel('')
    a[0].set_xlabel('')

    p1 = (
            so.Plot(df_plot[(df_plot.trained=='Proficient') & ind], x='x_pos', y='prob_left', marker='stim')
            .add(so.Dot(color=cond_colors[1], edgecolor='black', pointsize=2), legend=False)
            .limit(y=[-0.025,1.025])
            .scale(x=so.Continuous().tick(every=4, between=(-8,8)),
                    y=so.Continuous().tick(every=0.2, between=(0,1)),
                   marker={45 : 'o', 90 : 'v'})
            .on(a[1])
            .plot()
        )

    a[1].plot(x,y,'-k', zorder=-1)
    a[1].set_box_aspect(1)
    sns.despine(ax=a[1], trim=True)
    a[1].set_xticks([])
    a[1].set_yticks([])
    a[1].set_ylabel('')
    a[1].set_xlabel('')


    f.tight_layout()
    f.savefig(join(fig_save,'prob_left_logistic.svg'), format='svg')
