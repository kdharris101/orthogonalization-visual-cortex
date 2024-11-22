# %%
"""
Trial convexity and facial behavior

@author: Samuel Failor
"""

from os.path import join
import glob
from tkinter import W
from turtle import left
from types import NoneType
import numpy as np
from scipy.stats import ttest_rel,ttest_ind,ttest_1samp, mannwhitneyu, levene, ranksums, gmean, pearsonr, zscore, wilcoxon
from scipy.stats import binned_statistic, variation
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal.windows import gaussian
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style

font_dirs = ['C:/Windows/Fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# results_dir = r'H:/OneDrive for Business/Results'
results_dir = r'C:/Users/Samuel/OneDrive - University College London/Results'
# results_dir = '/mnt/c/Users/Samuel/OneDrive - University College London/Results'

# subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
#             'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
#             'SF180613']
# subjects = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B', 'SF171107',
#             'SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613']
# expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
#               '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
#               '2018-06-28', '2018-12-12']

# expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]

# trained = np.array([False, True, False, True, False, True, False, True, False,True])

subjects_file = np.array(['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613'])
subjects = np.array(['SF170620B', 'SF170620B', 'SF170905B', 'SF171107','SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613'])
expt_dates = np.array(['2017-07-04', '2017-12-21', '2017-11-26', '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12'])

expt_nums = np.array([5, 7, 1, 7, 3, 1, 3, 1, 5])

trained = np.array([False, True, True, False, True, False, True, False, True])


# subjects_file = subjects_file[trained]
# subjects = subjects[trained]
# expt_dates = expt_dates[trained]
# expt_nums = expt_nums[trained]
# trained = trained[trained]

# subjects_file = ['M170620B_SF', 'SF170620B',
#             'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
#             'SF180613']
# subjects = ['SF170620B', 'SF170620B', 'SF171107',
#             'SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613']
# expt_dates = ['2017-07-04', '2017-12-21', '2017-11-17', '2018-04-04', 
#               '2018-06-13', '2018-09-21','2018-06-28', '2018-12-12']

# expt_nums = [5, 7, 7, 3, 1, 3, 1, 5]

# trained = [False, True, False, True, False, True, False, True]

file_paths = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), r'_'.join([subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), 'orientation tuning_norm by pref ori_[2020]*'])))[-1] 
                                            for i in range(len(subjects_file))]
file_paths_pupil = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                    str(expt_nums[i]), r'_'.join([subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), 'orientation tuning_pupil and face resps_[2020]*'])))[-1] 
                                            for i in range(len(subjects_file))]

all_trials = False

if all_trials:

    file_paths_convexity = [glob.glob(join(results_dir, subjects_file[trained][i], expt_dates[trained][i],
                        str(expt_nums[trained][i]), r'_'.join([subjects_file[trained][i],
                        'trial_convexity_all_trials.npy'])))[-1] for i in range(len(subjects_file[trained]))]
    
else:
    file_paths_convexity = [glob.glob(join(results_dir, subjects_file[trained][i], expt_dates[trained][i],
                        str(expt_nums[trained][i]), r'_'.join([subjects_file[trained][i],
                        'trial_convexity.npy'])))[-1] for i in range(len(subjects_file[trained]))]
    
    
save_fig_dir = r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%% 

df_raster = pd.DataFrame()
df_tc = pd.DataFrame()

whisker_trace = np.zeros(len(subjects),dtype='object')

smooth_flag = False
kernal_sigma = 0.06 # In seconds
causal_flag = False

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    expt = np.load(file_paths[i], allow_pickle = True)[()]
    exptp = np.load(file_paths_pupil[i], allow_pickle = True)[()]
    if trained[i]:
        file_ind = np.where(np.unique(subjects)==subjects[i])[0][0]
        tc = np.load(file_paths_convexity[file_ind], allow_pickle=True)[:,0]
    
    if all_trials:
        test_ind = np.ones(len(expt['stim_ori'])).astype(bool)
    else:
        test_ind = expt['test_ind']
            
    stim_ori = expt['stim_ori'][test_ind]
    blank_ind = np.isinf(stim_ori)
    stim_ori = stim_ori[~blank_ind]

    stim_dir = expt['stim_dir'][test_ind][~blank_ind]

    stim_times = expt['stim_times'][0][test_ind][~blank_ind]

    frame_times = exptp['frame_times']

    max_mot = [exptp['facemap']['motion'][m].max() for m in [1,2]]
    whisk_ind = np.argmin(max_mot)+1

    motion_whiskers = exptp['facemap']['motion'][whisk_ind]
    pupil_area = exptp['facemap']['pupil'][0]['area_smooth']
    pupil_com = exptp['facemap']['pupil'][0]['com_smooth']

    if len(motion_whiskers) > len(frame_times):
        motion_whiskers = motion_whiskers[:-(len(motion_whiskers)-len(frame_times))]

    if len(pupil_area) > len(frame_times):
        pupil_area = pupil_area[:-(len(pupil_area)-len(frame_times))]
        
    if len(pupil_com) > len(frame_times):
        pupil_com = pupil_com[:-(len(pupil_com)-len(frame_times))]

    whisker_trace[i] = motion_whiskers

    good_ind = (frame_times > stim_times[0]-2) & (frame_times < stim_times[-1] + 4)
    frame_times = frame_times[good_ind]
    motion_whiskers = motion_whiskers[good_ind]
    pupil_area = pupil_area[good_ind]
    pupil_com = pupil_com[good_ind,:]
    
    # n second pre-baseline subtracted pupil
    # n_seconds = 5
    # fd = np.diff(frame_times).mean()
    # k = np.ones(int(2*np.ceil(n_seconds/fd))+1)
    # k[int(np.ceil((len(k)/2))):] = 0
    # k = k/k.sum()
    # pupil_baseline = np.convolve(k,pupil_area,mode='same')
    # pupil_area = pupil_area - pupil_baseline
    
    com_x = pupil_com[:,0]
    com_y = pupil_com[:,1]
    
    com_x_diff = np.diff(com_x,append=0)
    com_y_diff = np.diff(com_y,append=0)
    
    com_speed = np.sqrt(com_x_diff**2 + com_y_diff**2)
    
    if smooth_flag:
        k = gaussian(1001,kernal_sigma/fd,sym=True)
        k = k[k>0.0001]
        
        if causal_flag:
            k[np.argmax(k)+1:] = 0
            
        k = k/k.sum()
        motion_whiskers = np.convolve(k,motion_whiskers,mode = 'same')

    # Interpolate aligned to stimulus onsets
    time_range = [-1, 3]
    peri_times = np.linspace(time_range[0],time_range[1],np.diff(time_range)[0]*30+1)
    peri_times = np.repeat(peri_times[None,:],len(stim_ori),axis = 0)
    peri_trial_times = peri_times + stim_times[:,None]
    whiskers_interp = interp1d(frame_times,motion_whiskers, 'linear', fill_value = 'nan')
    whiskers_raster = whiskers_interp(peri_trial_times)
    whisker_raster_binerized = whiskers_raster>=whiskers_raster.std()*2
    pupil_interp = interp1d(frame_times,pupil_area,'linear', fill_value = 'nan')
    pupil_raster = pupil_interp(peri_trial_times)
    
    com_interp = interp1d(frame_times,com_speed,'linear',fill_value = 'nan')
    com_raster = com_interp(peri_trial_times)

    stim_labels = np.repeat(stim_ori[:,None].astype(int), whiskers_raster.shape[1], axis = 1).flatten()
    trial_num = np.repeat(np.arange(len(stim_ori))[:,None], whiskers_raster.shape[1], axis = 1).flatten()
    
    if i == 0:
        trial_num_max = 0
    elif i == 1:
        trial_num_max = df_raster.trial_num_uni.max()+1 
        trial_num_max_tc = 0
    else:
        trial_num_max = df_raster.trial_num_uni.max()+1
        trial_num_max_tc = df_tc.trial_num_uni.max()+1

    df_raster = pd.concat([df_raster, pd.DataFrame({'stim_ori' : stim_labels,
                                                    'whiskers' : whiskers_raster.flatten(),
                                                    'whiskers_z' : (whiskers_raster.flatten()-motion_whiskers.mean())/motion_whiskers.std(),
                                                    'whiskers_binerized' : whisker_raster_binerized.flatten(),
                                                    'pupil' : pupil_raster.flatten(),
                                                    'pupil_z' : (pupil_raster.flatten()-pupil_area.mean())/pupil_area.std(),
                                                    'com_speed' : com_raster.flatten(),
                                                    'com_speed_z' : (com_raster.flatten()-com_speed.mean())/com_speed.std(),
                                                    'times' : peri_times.flatten(),
                                                    'trained' : np.repeat(trained[i],whiskers_raster.size),
                                                    'subject' : np.repeat(subjects[i], whiskers_raster.size),
                                                    'trial_num' : trial_num.flatten(),
                                                    'trial_num_uni' : trial_num + trial_num_max})], ignore_index = True)

    if trained[i]:
        tc = np.repeat(tc[:,None],whiskers_raster.shape[1],axis=1).flatten()
        df_tc = pd.concat([df_tc, pd.DataFrame({'stim_ori' : stim_labels,
                                                'whiskers' : whiskers_raster.flatten(),
                                                'whiskers_z' : (whiskers_raster.flatten()-motion_whiskers.mean())/motion_whiskers.std(),
                                                'whiskers_binerized' : whisker_raster_binerized.flatten(),
                                                'pupil' : pupil_raster.flatten(),
                                                'pupil_z' : (pupil_raster.flatten()-pupil_area.mean())/pupil_area.std(),
                                                'com_speed' : com_raster.flatten(),
                                                'com_speed_z' : (com_raster.flatten()-com_speed.mean())/com_speed.std(),
                                                'times' : peri_times.flatten(),
                                                'tc' : tc.flatten(),
                                                'subject' : np.repeat(subjects[i], whiskers_raster.size),
                                                'trial_num' : trial_num.flatten(),
                                                'trial_num_uni' : trial_num + trial_num_max_tc})], ignore_index = True)

def stim_type(x):
        if (x == 45) | (x == 90):
            return '45 and 90'
        elif x == 68:
            return '68'
        else:
            return 'non-task'
        
        
bins = [-1,0,2,3]
labels = ['pre','post','late']

# bins = np.linspace(-1,3,21)
# labels = np.arange(20)

df_raster['stim_type'] = df_raster.stim_ori.map(stim_type)
df_tc['stim_type'] = df_tc.stim_ori.map(stim_type)

df_raster['trial_epoch'] = pd.cut(df_raster.times,bins,labels=labels, include_lowest= True, right = True)
df_tc['trial_epoch'] = pd.cut(df_tc.times,bins,labels=labels, include_lowest= True, right = True)


# Baseline subtract to isolate change from stimulus onset

def bs_wrapper(behavior):
    return lambda x: x[behavior].sub(x.loc[x['trial_epoch'].eq('pre'), behavior].mean())

for b in ['whiskers_z','pupil_z','com_speed_z','whiskers','pupil','com_speed']:
    df_tc[b+'_bs'] = df_tc.groupby(['trial_num_uni'], observed = True, as_index = False).apply(bs_wrapper(b)).droplevel(0)
    df_raster[b+'_bs'] = df_raster.groupby(['trial_num_uni'], observed = True, as_index = False).apply(bs_wrapper(b)).droplevel(0)


#%%

df_plot = df_raster.groupby(['subject','trained', 'stim_ori','times']).mean().reset_index()

sns.relplot(df_plot, x = 'times', y = 'whiskers_z_bs', hue = 'trained', col = 'stim_ori', kind = 'line', errorbar = ('se',2),
            palette = 'colorblind', col_wrap = 4)

sns.relplot(df_plot, x = 'times', y = 'pupil_z_bs', hue = 'trained', col = 'stim_ori', kind = 'line', errorbar = ('se',2),
            palette = 'colorblind', col_wrap = 4)

sns.relplot(df_plot, x = 'times', y = 'com_speed_bs', hue = 'trained', col = 'stim_ori', kind = 'line', errorbar = ('se',2),
            palette = 'colorblind', col_wrap = 4)

#%% Plot difference in pupil and whisking using seaborn objects


df_plot = df_raster.groupby(['subject','trained', 'stim_ori','times']).mean().reset_index()
df_plot['stim_ori'] = df_plot.stim_ori.apply(lambda x: str(x) + r'$\degree$')

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

fig_size = (4.5,2)

wh = (
        so.Plot(df_plot, x = 'times', y = 'whiskers_z_bs', color = 'trained')
        .layout(size = fig_size, engine = 'tight')
        .facet('stim_ori',wrap=4)
        .add(so.Line(), so.Agg(), legend = False)
        .add(so.Band(),so.Est(errorbar=('se',1)), legend = False)
        .limit(x = (-1.1,3.1), y = (-0.4,0.5))
        .share(x=False,y=False)
        .scale(color='colorblind',
               x = so.Continuous().tick(at=np.linspace(-1,3,5)), y = so.Continuous().tick(at=np.linspace(-0.4,0.5,5)))
        .theme(style)
        .label(y = r'$\Delta$stand. whisking', x = 'Time rel. to stim. onset (s)')
        .share(x=False,y=False)
        .plot()
    )

for i,a in enumerate(wh._figure.axes):
    # a.set_box_aspect(1)
    # a.plot([0,2],[-0.3,-0.3],'k', linewidth=0.5)
    if i == 0:
        sns.despine(ax=a,bottom=True,trim=True)
        a.set_xticks([])
    elif (i > 0) & (i <=3):
        sns.despine(ax=a,left = True, bottom = True,trim=True)
        a.set_xticks([])
        a.set_yticks([])
    elif i == 4:
        sns.despine(ax = a, trim = True)
    elif i > 4:
        sns.despine(ax=a,left=True,trim=True)
        a.set_yticks([])
        
wh._figure.tight_layout()

wh.save(join(save_fig_dir,'whisking_by_stim.svg'),format='svg')

pa = (
        so.Plot(df_plot, x = 'times', y = 'pupil_z_bs', color = 'trained')
        .layout(size = fig_size, engine = 'tight')
        .facet('stim_ori',wrap=4)
        .add(so.Line(), so.Agg(), legend = False)
        .add(so.Band(),so.Est(errorbar=('se',1)), legend = False)
        .limit(x = (-1.1,3.1), y = (-0.65,0.45))
        .scale(color='colorblind',
               x = so.Continuous().tick(at=np.linspace(-1,3,5)), y = so.Continuous().tick(at=np.linspace(-0.6,0.4,6)))
        .share(x=False,y=False)
        .theme(style)
        .label(y = r'$\Delta$stand. pupil area', x = 'Time rel. to stim. onset (s)')
        .plot()
    )

for i,a in enumerate(pa._figure.axes):
    # a.set_box_aspect(1)
    if i == 0:
        sns.despine(ax=a,bottom=True,trim=True)
        a.set_xticks([])
    elif (i > 0) & (i <=3):
        sns.despine(ax=a,left = True, bottom = True,trim=True)
        a.set_xticks([])
        a.set_yticks([])
    elif i == 4:
        sns.despine(ax = a, trim = True)
    elif i > 4:
        sns.despine(ax=a,left=True,trim=True)
        a.set_yticks([])
   
pa._figure.tight_layout()

pa.save(join(save_fig_dir,'pupil_area_by_stim.svg'),format='svg')


#%% Average in post-stim period for each stimulus and condition

df_plot = df_raster.groupby(['trial_num_uni','trial_epoch'], observed=True).agg({'subject' : 'first',
                                                                                 'stim_ori' : 'first',
                                                                                 'whiskers_z_bs' : 'mean',
                                                                                 'pupil_z_bs' : 'mean',
                                                                                 'trained' : 'first',
                                                                                 'trial_epoch' : 'first'}).reset_index(drop=True)

df_plot = df_plot[df_plot.trial_epoch=='post'].groupby(['subject','stim_ori','trained']).mean().reset_index()
df_plot['stim_ori'] = df_plot.stim_ori.apply(lambda x: str(x) + r'$\degree$')

df_plot['trained'] = df_plot.trained.map({False : 'Naive',
                                          True : 'Proficient'})

p_vals_pa = []
p_vals_wh = []

for s in df_plot.stim_ori.unique():
    
    p_vals_pa.append(ttest_ind(df_plot[(df_plot.stim_ori==s) & (df_plot.trained=='Naive')].pupil_z_bs,
                            df_plot[(df_plot.stim_ori==s) & (df_plot.trained=='Proficient')].pupil_z_bs))
    
    p_vals_wh.append(ttest_ind(df_plot[(df_plot.stim_ori==s) & (df_plot.trained=='Naive')].whiskers_z_bs,
                            df_plot[(df_plot.stim_ori==s) & (df_plot.trained=='Proficient')].whiskers_z_bs))


import statsmodels.api as sm
from statsmodels.formula.api import ols

#perform two-way ANOVA
model = ols('whiskers_z_bs ~ C(trained) + C(stim_ori) + C(trained):C(stim_ori)', data=df_plot).fit()
sm.stats.anova_lm(model, typ=2)


cond_map = {False : 'Naive',
            True : 'Proficient'}

df_plot['trained'] = df_plot.trained.map(cond_map)

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

fig_size = (4.5,1.5)

wh = (
      so.Plot(df_plot, x = 'stim_ori', y = 'whiskers_z_bs', color = 'trained')
      .add(so.Dot(pointsize=2),so.Dodge(),so.Jitter(0.2), legend = False)
      .add(so.Dash(), so.Agg(), so.Dodge(), legend = False)
      .add(so.Range(), so.Est(errorbar=("se", 1)), so.Dodge(), legend = False)
      .theme(style)
      .layout(size=fig_size)
      .label(y = r'$\Delta$stand. whisking', x = 'Stimulus orientation')
      .limit(y = (-0.25,0.55))
      .scale(color = 'colorblind',
             y = so.Continuous().tick(at=np.linspace(-0.2,0.5,4)))
      .plot()
     )

sns.despine(ax=wh._figure.axes[0], trim = True)

wh.save(join(save_fig_dir,'whisking_post_stim_by_stim_ori.svg'),format='svg')


pa = (
      so.Plot(df_plot, x = 'stim_ori', y = 'pupil_z_bs', color = 'trained')
      .add(so.Dot(pointsize=2),so.Dodge(),so.Jitter(0.2), legend = False)
      .add(so.Dash(), so.Agg(), so.Dodge(), legend = False)
      .add(so.Range(), so.Est(errorbar=("se", 1)), so.Dodge(), legend = False)
      .theme(style)
      .layout(size=fig_size)
      .label(y = r'$\Delta$stand. pupil area', x = 'Stimulus orientation')
      .scale(color = 'colorblind',
             y = so.Continuous().tick(at=np.linspace(-0.55,0.25,5)))
      .limit(y = (-0.6,0.3))
      .plot()
     )


sns.despine(ax=pa._figure.axes[0], trim = True)

pa.save(join(save_fig_dir,'pupil_post_stim_by_stim_ori.svg'),format='svg')


#%%

df_plot = df_tc.copy()

g = sns.relplot(df_plot, x = 'times', y = 'whiskers_z_bs', hue = 'stim_type', kind = 'line', errorbar = ('se',2),
            palette = 'colorblind', col = 'subject', hue_order = ['45 and 90', '68','non-task'])

for i,a in enumerate(g.axes.flatten()):
    a.set_title(f'Mouse {i}')


df_plot = df_plot.groupby(['subject','times','stim_type'],observed=True).mean().reset_index()

# df_plot =df_plot.groupby(['subject','times','stim_type']).mean().reset_index()

sns.relplot(df_plot, x = 'times', y = 'whiskers_z_bs', hue = 'stim_type', kind = 'line', errorbar = ('se',2),
            palette = 'colorblind', hue_order = ['45 and 90', '68','non-task'])

#%% 

df_plot = df_raster.groupby(['subject','trial_num_uni','trained', 'stim_type','trial_epoch'], observed = True).mean().reset_index()
df_plot = df_plot.groupby(['subject','trained','stim_type','trial_epoch']).mean().reset_index()

sns.catplot(df_plot[df_plot.trial_epoch=='post'], x = 'stim_type', y = 'whiskers_z_bs', hue = 'trained', kind = 'bar')

df_plot = df_tc.groupby(['subject','stim_type','trial_epoch','trial_num_uni'], observed = True).mean().reset_index()
df_plot = df_plot.groupby(['subject','stim_type','trial_epoch']).mean().reset_index()

sns.barplot(df_plot, x = 'trial_epoch', y = 'whiskers_z', hue = 'stim_type', errorbar = ('se',2),
            order = ['post','late'])

#%% 


# df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
#                                                               'stim_ori':'first',
#                                                               'whiskers_z_bs':'mean',
#                                                               'pupil_z_bs':'mean',
#                                                               'com_speed_z_bs':'mean',
#                                                               'subject':'first',
#                                                               'stim_type':'first'}).reset_index()

df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
                                                              'stim_ori':'first',
                                                              'whiskers_z_bs':'mean',
                                                              'pupil_z_bs':'mean',
                                                              'com_speed_z_bs':'mean',
                                                              'subject':'first',
                                                              'stim_type':'first'}).reset_index()

# sns.displot(df_plot[df_plot.trial_epoch=='post'], x = 'whiskers_z_bs', col = 'subject', row = 'stim_type', kind = 'hist',
#             common_norm = False, stat = 'probability')

# bins = [-np.inf,-1,0,1,2,3,4]
# bins = [-np.inf,-0.5,0,0.5,1,1.5,2,np.inf]

bins = np.linspace(-3,3,21)


# df_plot['whisk_bin'] = pd.cut(df_plot.whiskers_z_bs,bins=bins, labels = ['<-1', '-1 to 0', '0 to 1', '1 to 2', '2 to 3', '>3'],include_lowest=True, right = True)
df_plot['whisk_bin'] = pd.cut(df_plot.whiskers_z_bs,bins=bins, labels = np.arange(len(bins)-1),include_lowest=True, right = True)
df_plot['pupil_bin'] = pd.cut(df_plot.pupil_z_bs,bins=bins, labels = np.arange(len(bins)-1),include_lowest=True, right = True)


sns.relplot(df_plot[df_plot.trial_epoch=='post'], x = 'whisk_bin', y = 'tc', hue = 'stim_type', kind = 'line', col = 'subject',
            hue_order = ['45 and 90', '68', 'non-task'], errorbar = ('se',2))

sns.relplot(df_plot[df_plot.trial_epoch=='post'], x = 'pupil_bin', y = 'tc', hue = 'stim_type', kind = 'line', col = 'subject',
            hue_order = ['45 and 90', '68', 'non-task'], errorbar = ('se',2))

# sns.relplot(df_plot[df_plot.trial_epoch=='post'], x = 'whiskers_z_bs', y = 'tc', col = 'subject',
#             row = 'stim_type')

# sns.relplot(df_plot[df_plot.trial_epoch=='post'], x = 'whiskers_z', y = 'tc', col = 'subject',
#             row = 'stim_type')

#%% Plot relationship between behavior and tc with interpolation


# df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
#                                                               'stim_ori':'first',
#                                                               'whiskers_z_bs':'mean',
#                                                               'pupil_z_bs':'mean',
#                                                               'com_speed_z_bs':'mean',
#                                                               'subject':'first',
#                                                               'stim_type':'first'}).reset_index()

df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
                                                              'stim_ori':'first',
                                                              'whiskers_z_bs':'mean',
                                                              'pupil_z_bs':'mean',
                                                              'com_speed_z_bs':'mean',
                                                              'subject':'first',
                                                              'stim_type':'first'}).reset_index()

df_plot = df_plot[df_plot.trial_epoch=='post']

f,a = plt.subplots(1,5, figsize=(25,5))

for c,s in enumerate(df_plot.subject.unique()):    
    ind = df_plot.subject==s
    
    sind = np.argsort(df_plot[ind].whiskers_z_bs.to_numpy())
    
    interp = UnivariateSpline(df_plot[ind].whiskers_z_bs.to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind],k=2)
    
    x = np.linspace(df_plot[ind].whiskers_z_bs.min(),df_plot[ind].whiskers_z_bs.max(),1000)
    
    sns.scatterplot(df_plot[ind], x = 'whiskers_z_bs', y = 'tc', ax = a[c])
    a[c].plot(x,interp(x))
    a[c].set_box_aspect(1)


#%% Plot relationship between behavior and tc with interpolation

measure = 'whiskers_z_bs'

df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
                                                              'stim_ori':'first',
                                                               measure : 'mean',
                                                              'subject':'first',
                                                              'stim_type':'first'}).reset_index()

df_plot = df_plot[df_plot.trial_epoch=='post']

stim_colors = sns.color_palette('bright',3)

f,a = plt.subplots(1,5, figsize=(25,5), sharex=True, sharey=True)

for c,s in enumerate(df_plot.subject.unique()):
    for r,o in enumerate(df_plot.stim_type.unique()[::-1]):
    
        ind = (df_plot.subject==s) & (df_plot.stim_type==o)
        
        sind = np.argsort(df_plot.loc[ind,measure].to_numpy())
        
        interp = UnivariateSpline(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind],k=1)
        
        x = np.linspace(df_plot.loc[ind,measure].min(),df_plot.loc[ind,measure].max(),1000)
        
        sns.scatterplot(df_plot[ind], x = measure, y = 'tc', ax = a[c], color = stim_colors[r])
        a[c].plot(x,interp(x), color = stim_colors[r], label = o)
        a[c].set_box_aspect(1)
    
    handles, labels = a[c].get_legend_handles_labels()
    a[c].legend(handles = handles, labels=labels)
    
#%% Plot relationship between behavior and tc with interpolation

measure = 'whiskers_z_bs'

df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
                                                              'stim_ori':'first',
                                                               measure : 'mean',
                                                              'subject':'first',
                                                              'stim_type':'first'}).reset_index()

df_plot = df_plot[df_plot.trial_epoch=='post']

stim_colors = sns.color_palette('bright',3)

f,a = plt.subplots(1,1, figsize=(5,5), sharex=True, sharey=True)

for r,o in enumerate(df_plot.stim_type.unique()[::-1]):
    
    ind = df_plot.stim_type==o
    
    sind = np.argsort(df_plot.loc[ind,measure].to_numpy())
    
    interp = UnivariateSpline(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind],k=1)
    
    x = np.linspace(df_plot.loc[ind,measure].min(),df_plot.loc[ind,measure].max(),1000)
    
    sns.scatterplot(df_plot[ind], x = measure, y = 'tc', ax = a, color = stim_colors[r], style='subject', legend = False, s = 20)
    a.plot(x,interp(x), color = stim_colors[r])
    
#%% Plot relationship between behavior and tc with interpolation - plot all points, and subject specific fits, then average fit for each stim

measure = 'pupil_z'

diff_flag = False

if diff_flag:
    measure = measure + '_bs'

df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
                                                              'stim_ori':'first',
                                                               measure : 'mean',
                                                              'subject':'first',
                                                              'stim_type':'first'}).reset_index()

trial_epoch = 'post'

# Can't look at baseline subtracted baseline!
if trial_epoch == 'pre':
    diff_flag == False

df_plot = df_plot[df_plot.trial_epoch==trial_epoch]

stim_colors = sns.color_palette('bright',3)

f,a = plt.subplots(1,1, figsize=(5,5), sharex=True, sharey=True)

for c,s in enumerate(df_plot.subject.unique()):
    for r,o in enumerate(df_plot.stim_type.unique()[::-1]):
    
        ind = (df_plot.subject==s) & (df_plot.stim_type==o)
        sind = np.argsort(df_plot.loc[ind,measure].to_numpy())
        interp = UnivariateSpline(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind],k=1)
        x = np.linspace(df_plot[measure].min(),df_plot[measure].max(),1000)
        a.plot(x,interp(x), color = stim_colors[r], linestyle='-', alpha = 0.2)
        
for r,o in enumerate(df_plot.stim_type.unique()[::-1]):
    ind = df_plot.stim_type==o
    sind = np.argsort(df_plot.loc[ind,measure].to_numpy())
    interp = UnivariateSpline(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind],k=1)
    x = np.linspace(df_plot[measure].min(),df_plot[measure].max(),1000)
    a.plot(x,interp(x), color = stim_colors[r], linestyle = '-')
        
sns.scatterplot(df_plot, x = measure, y = 'tc', ax = a, hue = 'stim_type', palette = stim_colors, legend = False,
                hue_order = ['45 and 90', '68','non-task'], alpha = 0.2, ec = None)
a.set_box_aspect(1)
if diff_flag:
    a.set_xlabel(f'{trial_epoch}-stimulus whisking - pre-stimulus whisking')
else:
    a.set_xlabel(f'{trial_epoch}-stimulus whisking')
a.set_ylabel('Trial convexity')
sns.despine(ax=a, trim = True)    

#%% Pre-stim fig for publication

import seaborn.objects as so
from seaborn import axes_style
from scipy.stats import linregress

measure = 'whiskers_z'
trial_epoch = 'pre' 


df_plot = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
                                                              'stim_ori':'first',
                                                               measure : 'mean',
                                                              'subject':'first',
                                                              'stim_type':'first'}).reset_index()

df_plot = df_plot[df_plot.trial_epoch==trial_epoch]
df_plot['tc'] = df_plot.tc-1

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
         'lines.markersize':1,
         'lines.linewidth':0.5,
         'legend.fontsize':5,
         'legend.title_fontsize':5,
         'axes.spines.left': True,
         'axes.spines.bottom': True,
         'axes.spines.right': False,
         'axes.spines.top': False,
         'font.sans-serif' : 'Helvetica',
         'font.family' : 'sans-serif'}


# stim_colors = np.array(['#1b9e77', '#d95f02','#7570b3'])[::-1]

sub_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

if measure == 'whiskers_z':
    measure_xlabel = 'whisking'
    # xlim = (np.floor(df_plot.whiskers_z.min()),np.ceil(df_plot.whiskers_z.max()))
    xlim = (-1,2.2)
elif measure == 'pupil_z':
    measure_xlabel = 'pupil area'
    xlim = (np.floor(df_plot.pupil_z.min()),np.ceil(df_plot.pupil_z.max())) 


xlabel = trial_epoch.capitalize() + f'-stimulus {measure_xlabel} (standardized)'

fig_size = (5,1.5)

p0 = (
        so.Plot(df_plot, x = measure, y = 'tc', color = 'subject')
        .facet('stim_type', order = ['45 and 90','68','non-task'])
        .share(x=True,y=False)
        .layout(size = fig_size, engine = 'tight')
        .add(so.Dots(pointsize = 2,alpha = 1, fill = False, stroke = 0.5), legend = False)
        .scale(color=sub_colors,
               x = so.Continuous().tick(at=np.linspace(int(xlim[0]),int(xlim[1]),int(np.diff(xlim))+1)),
               y = so.Continuous().tick(at=np.linspace(-0.5,2,6)))
        .limit(y = (-0.55,1.5), x = xlim)
        .theme(style)
        .label(y = 'Trial convexity', x = xlabel)
        .plot()
    )

for c,s in enumerate(df_plot.subject.unique()):
    for r,o in enumerate(df_plot.stim_type.unique()[::-1]):    
        ind = (df_plot.subject==s) & (df_plot.stim_type==o)
        sind = np.argsort(df_plot.loc[ind,measure].to_numpy())
        # interp = UnivariateSpline(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind],k=1)
        result = linregress(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind])
        # x = np.linspace(df_plot[measure].min(),df_plot[measure].max(),1000)
        x0 = -1
        x1 = df_plot.loc[ind,measure].max() + 1
        x = np.linspace(x0,x1,1000)

        p0._figure.axes[r].plot(x,x*result.slope + result.intercept, color = sub_colors[c], linestyle='-', alpha = 1,
                                linewidth = 0.5)


for r,o in enumerate(df_plot.stim_type.unique()[::-1]):
    ind = df_plot.stim_type==o
    sind = np.argsort(df_plot.loc[ind,measure].to_numpy())
    # interp = UnivariateSpline(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind],k=1)
    # coefs[r,:] = interp.get_coeffs()
    result = linregress(df_plot.loc[ind,measure].to_numpy()[sind],df_plot[ind].tc.to_numpy()[sind])
    # x = np.linspace(df_plot[measure].min(),df_plot[measure].max(),1000)
    x = np.linspace(-1,3,1000)
    p0._figure.axes[r].plot(x,x*result.slope + result.intercept, color = 'k', linestyle = '-', linewidth = 1)


for i,a in enumerate(p0._figure.axes):
    a.set_box_aspect(1)
    if i > 0:
        sns.despine(ax=a, left = True, trim=True)
        a.set_yticks([])
    else:
        sns.despine(ax=a,trim=True)

# sns.despine(ax = p0._figure.axes[0])

p0.save(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\trial_convexity_vs_' + trial_epoch + '_stim_' + measure +'.svg',
        format = 'svg')



#%% Mixed model looking at tc and behavior + stim
import statsmodels.api as sm
import statsmodels.formula.api as smf

measure = 'whiskers_z'

df_mm = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc':'first',
                                                              'stim_ori':'first',
                                                               measure : 'mean',
                                                              'subject':'first',
                                                              'stim_type':'first'}).reset_index()

df_mm = df_mm[df_mm.trial_epoch=='post']


md = smf.mixedlm(f'tc ~ {measure} + C(stim_type)', df_mm, groups=df_mm['subject'], re_formula = f'~C(stim_type)')
mdf = md.fit(method='powell')
print(mdf.summary())

md = smf.mixedlm(f"tc ~ {measure}*C(stim_type)", df_mm, groups=df_mm['subject'], re_formula = f'{measure}')
mdf = md.fit(method='powell')
print(mdf.summary())

# Fit models to each stimulus condition seperately

md = smf.mixedlm(f'tc ~ {measure}', df_mm[df_mm.stim_type=='45 and 90'], groups = df_mm[df_mm.stim_type=='45 and 90'].subject,
                 re_formula = f'{measure}')
mdf = md.fit(method='powell')
print(mdf.summary())

md = smf.mixedlm(f'tc ~ {measure}', df_mm[df_mm.stim_type=='68'], groups = df_mm[df_mm.stim_type=='68'].subject,
                 re_formula = f'{measure}')
mdf = md.fit(method='powell')
print(mdf.summary())

md = smf.mixedlm(f'tc ~ {measure}', df_mm[df_mm.stim_type=='non-task'], groups = df_mm[df_mm.stim_type=='non-task'].subject,
                 re_formula = f'{measure}')
mdf = md.fit(method='powell')
print(mdf.summary())


#%% Mixed model looking at behavior and relationship to stim
import statsmodels.api as sm
import statsmodels.formula.api as smf


df_mm = df_raster.groupby(['trial_num_uni','trial_epoch']).agg({'trained':'first',
                                                                'stim_ori':'first',
                                                                'whiskers_z_bs' : 'mean',
                                                                'pupil_z_bs' : 'mean',
                                                                'subject':'first',
                                                                'stim_type':'first'}).reset_index()

df_mm = df_mm[df_mm.trial_epoch=='post']


md = smf.mixedlm(f'whiskers_z_bs ~ C(trained) * C(stim_ori)', df_mm, groups=df_mm['subject'], re_formula = '~C(trained)*C(stim_ori)')
mdf = md.fit(method='powell')
print(mdf.summary())




#%% Model by subject - ridge 

from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import explained_variance_score

explained_variance_r = np.zeros((4,len(df_tc.subject.unique())))

coefs = np.zeros((4,len(df_tc.subject.unique())),dtype = 'object')

alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

for i,s in enumerate(df_tc.subject.unique()):
    print(f'{s}')

    ind = df_tc.subject == s

    ohe = OneHotEncoder(sparse = False)
    stim_ori = df_tc[ind].groupby('trial_num').agg({'stim_ori' : 'first'}).stim_ori.to_numpy()[:,None]
    x_stim_type = ohe.fit_transform(stim_ori)
    # Remove column so you can fit intercept
    x_stim_type = x_stim_type[:,1:]

    y_tc = df_tc[ind].groupby(['trial_num']).agg({'tc' : 'first'}).tc.to_numpy()

    # n_folds = len(y_tc)
    n_folds = 5

    skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 1)
    # s = KFold(n_splits = n_folds, shuffle = True, random_state = 1)

    lm = RidgeCV(alphas = alphas, fit_intercept = True)

    # Stim only model
    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        lm.fit(x_stim_type[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_stim_type[test_idx,:])

    explained_variance_r[0,i] = explained_variance_score(y_tc,y_hat)
    print(f'Stim only model : {explained_variance_r[0,i]}')
    
    lm.fit(x_stim_type,y_tc)
    coefs[0,i] = lm.coef_

    # Behaviour only
    whisk_scaler = StandardScaler()
    com_scaler = StandardScaler()
    pupil_scaler = StandardScaler()
    
    whisk = pd.pivot(df_tc[ind],columns='times',values = 'whiskers_z_bs', index = 'trial_num').to_numpy()
    com = pd.pivot(df_tc[ind],columns='times',values = 'com_speed_z_bs', index = 'trial_num').to_numpy()
    pupil = pd.pivot(df_tc[ind],columns='times',values = 'pupil_z_bs', index = 'trial_num').to_numpy()
    
    lm = RidgeCV(alphas = alphas, fit_intercept = True)
    
    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        # x_whisk = np.zeros_like(whisk)
        # x_com = np.zeros_like(com)
        # x_pupil = np.zeros_like(pupil)
        
        # x_whisk[train_idx,:] = whisk_scaler.fit_transform(whisk[train_idx,:])
        # x_whisk[test_idx,:] = whisk_scaler.transform(whisk[test_idx,:])
        # x_com[train_idx,:] = com_scaler.fit_transform(com[train_idx,:])
        # x_com[test_idx,:] = com_scaler.transform(com[test_idx,:])
        # x_pupil[train_idx,:] = pupil_scaler.fit_transform(pupil[train_idx,:])
        # x_pupil[test_idx,:] = pupil_scaler.transform(pupil[test_idx,:])
        
        # x_behavior = np.hstack([x_whisk, x_com, x_pupil])
        # x_behavior = np.hstack([x_pupil])
        x_behavior = np.hstack([whisk,com,pupil])
        lm.fit(x_behavior[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_behavior[test_idx,:])
        
    explained_variance_r[1,i] = explained_variance_score(y_tc,y_hat)
    print(f'Behavior only model : {explained_variance_r[1,i]}')
    
    lm.fit(x_behavior,y_tc)
    coefs[1,i] = lm.coef_

    # Full model
    lm = RidgeCV(alphas = alphas, fit_intercept = True)

    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        
        # x_whisk = np.zeros_like(whisk)
        # x_com = np.zeros_like(com)
        # x_pupil = np.zeros_like(pupil)
        
        # x_whisk[train_idx,:] = whisk_scaler.fit_transform(whisk[train_idx,:])
        # x_whisk[test_idx,:] = whisk_scaler.transform(whisk[test_idx,:])
        # x_com[train_idx,:] = com_scaler.fit_transform(com[train_idx,:])
        # x_com[test_idx,:] = com_scaler.transform(com[test_idx,:])
        # x_pupil[train_idx,:] = pupil_scaler.fit_transform(pupil[train_idx,:])
        # x_pupil[test_idx,:] = pupil_scaler.transform(pupil[test_idx,:])
        
        # x_full = np.hstack([x_stim_type, x_whisk, x_com, x_pupil])
        # x_full = np.hstack([x_stim_type, whisk])
        x_full = np.hstack([x_stim_type, whisk, com, pupil])

        lm.fit(x_full[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_full[test_idx,:])
        
    explained_variance_r[2,i] = explained_variance_score(y_tc,y_hat)
    print(f'Full model : {explained_variance_r[2,i]}')
    
    lm.fit(x_full,y_tc)
    coefs[2,i] = lm.coef_
    
    # Full model with interactions
    lm = RidgeCV(alphas = alphas, fit_intercept = True)

    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        
        # x_whisk = np.zeros_like(whisk)
        # x_com = np.zeros_like(com)
        # x_pupil = np.zeros_like(pupil)
        
        # x_whisk[train_idx,:] = whisk_scaler.fit_transform(whisk[train_idx,:])
        # x_whisk[test_idx,:] = whisk_scaler.transform(whisk[test_idx,:])
        # x_com[train_idx,:] = com_scaler.fit_transform(com[train_idx,:])
        # x_com[test_idx,:] = com_scaler.transform(com[test_idx,:])
        # x_pupil[train_idx,:] = pupil_scaler.fit_transform(pupil[train_idx,:])
        # x_pupil[test_idx,:] = pupil_scaler.transform(pupil[test_idx,:])
        
        x_whisk, x_com, x_pupil = whisk, com, pupil
        
        x_stim_whisk = np.hstack([x[:,None]*x_whisk for x in x_stim_type.T])
        x_stim_com = np.hstack([x[:,None]*x_com for x in x_stim_type.T])
        x_stim_pupil = np.hstack([x[:,None]*x_pupil for x in x_stim_type.T])
    
        x_full = np.hstack([x_stim_type, x_whisk, x_com, x_pupil, x_stim_whisk, x_stim_com, x_stim_pupil,
                            x_whisk*x_com, x_whisk*x_pupil, x_com*x_pupil])
        # x_full = np.hstack([x_stim_type, whisk, x_stim_whisk])
        lm.fit(x_full[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_full[test_idx,:])
    explained_variance_r[3,i] = explained_variance_score(y_tc,y_hat)
    print(f'Full model with interactions: {explained_variance_r[3,i]}')
    lm.fit(x_full,y_tc)
    coefs[3,i] = lm.coef_
    
#%% Model by subject - ridge, but with binned face motion

from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import explained_variance_score

explained_variance_rb = np.zeros((4,len(df_tc.subject.unique())))

coefs = np.zeros((4,len(df_tc.subject.unique())),dtype = 'object')

alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

df_model = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc' : 'first',
                                                               'pupil' : 'mean',
                                                               'whiskers' : 'mean',
                                                               'com_speed' : 'mean',
                                                               'subject' : 'first',
                                                               'stim_ori' : 'first'}).reset_index()

for i,s in enumerate(df_model.subject.unique()):
    print(f'{s}')

    ind = df_model.subject == s

    ohe = OneHotEncoder(sparse = False)
    stim_ori = df_model[ind].groupby('trial_num_uni').agg({'stim_ori' : 'first'}).stim_ori.to_numpy()[:,None]
    x_stim_type = ohe.fit_transform(stim_ori)
    # Remove column so you can fit intercept
    x_stim_type = x_stim_type[:,1:]

    y_tc = df_model[ind].groupby(['trial_num_uni']).agg({'tc' : 'first'}).tc.to_numpy()

    # n_folds = len(y_tc)
    n_folds = 5

    skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 1)
    # s = KFold(n_splits = n_folds, shuffle = True, random_state = 1)

    lm = RidgeCV(alphas = alphas, fit_intercept = True)

    # Stim only model
    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        lm.fit(x_stim_type[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_stim_type[test_idx,:])

    explained_variance_rb[0,i] = explained_variance_score(y_tc,y_hat)
    print(f'Stim only model : {explained_variance_rb[0,i]}')
    
    lm.fit(x_stim_type,y_tc)
    coefs[0,i] = lm.coef_

    # Behaviour only
    whisk_scaler = StandardScaler()
    com_scaler = StandardScaler()
    pupil_scaler = StandardScaler()
    
    whisk = pd.pivot(df_model[ind],columns='trial_epoch',values = 'whiskers', index = 'trial_num_uni').to_numpy()
    # whisk = whisk[:,1][:,None]
    com = pd.pivot(df_model[ind],columns='trial_epoch',values = 'com_speed', index = 'trial_num_uni').to_numpy()
    # com = com[:,1][:,None]
    pupil = pd.pivot(df_model[ind],columns='trial_epoch',values = 'pupil', index = 'trial_num_uni').to_numpy()
    # pupil = pupil[:,1][:,None]
    
    lm = RidgeCV(alphas = alphas, fit_intercept = True)
    
    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        x_whisk = np.zeros_like(whisk)
        x_com = np.zeros_like(com)
        x_pupil = np.zeros_like(pupil)
        
        # x_whisk[train_idx,:] = whisk_scaler.fit_transform(whisk[train_idx,:])
        # x_whisk[test_idx,:] = whisk_scaler.transform(whisk[test_idx,:])
        # x_com[train_idx,:] = com_scaler.fit_transform(com[train_idx,:])
        # x_com[test_idx,:] = com_scaler.transform(com[test_idx,:])
        # x_pupil[train_idx,:] = pupil_scaler.fit_transform(pupil[train_idx,:])
        # x_pupil[test_idx,:] = pupil_scaler.transform(pupil[test_idx,:])
        
        # x_behavior = np.hstack([x_whisk, x_com, x_pupil])
        # x_behavior = np.hstack([x_whisk, x_pupil])
        # x_behavior = np.hstack([x_whisk])
        x_behavior = np.hstack([whisk, com, pupil])
        lm.fit(x_behavior[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_behavior[test_idx,:])
        
    explained_variance_rb[1,i] = explained_variance_score(y_tc,y_hat)
    print(f'Behavior only model : {explained_variance_rb[1,i]}')
    
    lm.fit(x_behavior,y_tc)
    coefs[1,i] = lm.coef_

    # Full model
    lm = RidgeCV(alphas = alphas, fit_intercept = True)

    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        
        # x_whisk = np.zeros_like(whisk)
        # x_com = np.zeros_like(com)
        # x_pupil = np.zeros_like(pupil)
        
        # x_whisk[train_idx,:] = whisk_scaler.fit_transform(whisk[train_idx,:])
        # x_whisk[test_idx,:] = whisk_scaler.transform(whisk[test_idx,:])
        # x_com[train_idx,:] = com_scaler.fit_transform(com[train_idx,:])
        # x_com[test_idx,:] = com_scaler.transform(com[test_idx,:])
        # x_pupil[train_idx,:] = pupil_scaler.fit_transform(pupil[train_idx,:])
        # x_pupil[test_idx,:] = pupil_scaler.transform(pupil[test_idx,:])
        
        # x_full = np.hstack([x_stim_type, x_whisk, x_com, x_pupil])
        # x_full = np.hstack([x_stim_type, x_whisk, x_pupil])
        # x_full = np.hstack([x_stim_type, x_whisk])
        x_full = np.hstack([x_stim_type,whisk,com,pupil])

        lm.fit(x_full[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_full[test_idx,:])
        
    explained_variance_rb[2,i] = explained_variance_score(y_tc,y_hat)
    print(f'Full model : {explained_variance_rb[2,i]}')
    
    lm.fit(x_full,y_tc)
    coefs[2,i] = lm.coef_
    
    # Full model with interactions
    lm = RidgeCV(alphas = alphas, fit_intercept = True)

    y_hat = np.zeros(len(y_tc)) 
    for train_idx,test_idx in skf.split(y_tc,stim_ori):
        
        # x_whisk = np.zeros_like(whisk)
        # x_com = np.zeros_like(com)
        # x_pupil = np.zeros_like(pupil)
        
        # x_whisk[train_idx,:] = whisk_scaler.fit_transform(whisk[train_idx,:])
        # x_whisk[test_idx,:] = whisk_scaler.transform(whisk[test_idx,:])
        # x_com[train_idx,:] = com_scaler.fit_transform(com[train_idx,:])
        # x_com[test_idx,:] = com_scaler.transform(com[test_idx,:])
        # x_pupil[train_idx,:] = pupil_scaler.fit_transform(pupil[train_idx,:])
        # x_pupil[test_idx,:] = pupil_scaler.transform(pupil[test_idx,:])
        
        x_whisk, x_pupil, x_com = whisk, pupil, com
        
        x_stim_whisk = np.hstack([x[:,None]*x_whisk for x in x_stim_type.T])
        x_stim_com = np.hstack([x[:,None]*x_com for x in x_stim_type.T])
        x_stim_pupil = np.hstack([x[:,None]*x_pupil for x in x_stim_type.T])
    
        x_full = np.hstack([x_stim_type, x_whisk, x_com, x_pupil, x_stim_whisk, x_stim_com, x_stim_pupil])
        # x_full = np.hstack([x_stim_type, x_whisk, x_pupil, x_stim_whisk,x_stim_pupil])
        # x_full = np.hstack([x_stim_type, x_whisk, x_stim_whisk])
        lm.fit(x_full[train_idx,:], y_tc[train_idx])
        y_hat[test_idx] = lm.predict(x_full[test_idx,:])
    explained_variance_rb[3,i] = explained_variance_score(y_tc,y_hat)
    print(f'Full model with interactions: {explained_variance_rb[3,i]}')
    lm.fit(x_full,y_tc)
    coefs[3,i] = lm.coef_
    


#%%

f,a = plt.subplots(1,1)
sns.scatterplot(x = explained_variance_r[1,:],y = explained_variance_r[0,:], ax = a, color = 'grey', ec = 'k')
a.set_xlim([-0.125,0.5])
a.set_ylim([-0.125,0.5])
a.plot([-0.1,0.5],[-0.1,0.5],'--k')
a.set_box_aspect(1)
sns.despine(ax = a, trim = True)
a.set_ylabel('Explained variance - Stimlus model')
a.set_xlabel('Explained variance - Behavior model')

f,a = plt.subplots(1,1)
sns.scatterplot(x = explained_variance_r[2,:],y = explained_variance_r[0,:], ax = a, color = 'grey', ec = 'k')
a.set_xlim([-0.125,0.5])
a.set_ylim([-0.125,0.5])
a.plot([-0.1,0.5],[-0.1,0.5],'--k')
a.set_box_aspect(1)
sns.despine(ax = a, trim = True)
a.set_xlabel('Explained variance - Stimlus and behavior model')
a.set_ylabel('Explained variance - Stimulus model')

# sns.barplot(explained_variance_r.T, errorbar = ('se',1))

#%% Results for binned facial behaviors

f,a = plt.subplots(1,1)
sns.scatterplot(x = explained_variance_rb[1,:],y = explained_variance_rb[0,:], ax = a, color = 'grey', ec = 'k')
a.set_xlim([-0.01,0.51])
a.set_ylim([-0.01,0.51])
a.plot([-0,0.5],[-0,0.5],'--k')
a.set_box_aspect(1)
sns.despine(ax = a, trim = True)
a.set_ylabel('Explained variance - Stimlus model')
a.set_xlabel('Explained variance - Behavior model')

f,a = plt.subplots(1,1)
sns.scatterplot(x = explained_variance_rb[2,:],y = explained_variance_rb[0,:], ax = a, color = 'grey', ec = 'k')
a.set_xlim([-0.01,0.51])
a.set_ylim([-0.01,0.51])
a.plot([-0,0.5],[-0,0.5],'--k')
a.set_box_aspect(1)
sns.despine(ax = a, trim = True)
a.set_xlabel('Explained variance - Stimlus and behavior model')
a.set_ylabel('Explained variance - Stimulus model')

# sns.barplot(explained_variance_r.T, errorbar = ('se',1))

#%% Compare trial convexity of 45 and 90 to model performance

mu_tc = df_tc.groupby(['trial_num_uni']).agg({  'tc' : 'first',
                                                'subject' : 'first',
                                                'stim_type' : 'first'}).reset_index()

mu_tc = mu_tc.groupby(['subject','stim_type']).agg({'tc':'mean'}).reset_index()

f,a = plt.subplots(1,1)
sns.scatterplot(x = mu_tc[mu_tc.stim_type=='45 and 90'].tc.to_numpy()-mu_tc[mu_tc.stim_type=='68'].tc.to_numpy(),
                y = explained_variance_rb[0,:]-explained_variance_rb[1,:], ax = a)
a.set_xlabel('Difference in convexity between motor-associated stimuli and 68')
a.set_ylabel('Difference in explained variance of between stimulus and behavior model')
a.set_box_aspect(1)
sns.despine(ax=a)

mu_whisk = df_tc.groupby(['subject','stim_type','trial_num_uni','trial_epoch'],observed=True).agg({ 'whiskers_z_bs' : 'mean',
                                                                                                    'tc':'first'}).reset_index()

mu_whisk = mu_whisk.groupby(['subject','stim_type','trial_epoch']).agg({'tc':'mean',
                                                                        'whiskers_z_bs' : 'mean'}).reset_index()
# mu_whisk['stim_ori'] = mu_whisk.stim_ori.astype('str')
f,a = plt.subplots(1,1)
sns.scatterplot(mu_whisk[mu_whisk.trial_epoch=='post'], x = 'whiskers_z_bs', y = 'tc', hue = 'stim_type', style = 'subject', 
                legend = True)
a.set_xlabel('Whisking')
a.set_ylabel('Trial convexity')
a.set_box_aspect(1)
sns.despine(ax=a)


mu_tc = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc' : 'first',
                                                            'subject' : 'first',
                                                            'stim_type' : 'first',
                                                            'whiskers_z_bs' : 'mean'}).reset_index()


sns.relplot(mu_tc[mu_tc.trial_epoch=='post'], x = 'whiskers_z_bs', y = 'tc', hue = 'stim_type', col = 'subject', col_wrap = 3,
            hue_order = ['45 and 90', '68', 'non-task'])


mu_tc = df_tc.groupby(['trial_num_uni','trial_epoch']).agg({'tc' : 'first',
                                                            'subject' : 'first',
                                                            'stim_type' : 'first',
                                                            'whiskers_z_bs' : 'mean'}).reset_index()

mu_tc = mu_tc.groupby(['subject','stim_type','trial_epoch']).mean().reset_index()

mu_tc = mu_tc[mu_tc.trial_epoch=='post']

sns.scatterplot(x = mu_tc[mu_tc.stim_type=='45 and 90'].whiskers_z_bs.to_numpy() - mu_tc[mu_tc.stim_type=='68'].whiskers_z_bs.to_numpy(),
                y = mu_tc[mu_tc.stim_type=='45 and 90'].tc.to_numpy() - mu_tc[mu_tc.stim_type=='68'].tc.to_numpy())



#%%

df_plot = df_tc[(df_tc.times > 0) & (df_tc.whiskers_z_bs <1.5) & (df_tc.whiskers_z_bs > -1.5)].copy()

df_plot = df_plot.groupby('trial_num_uni').agg({'subject' : 'first',
                                                'stim_type' : 'first',
                                                'tc' : 'first',
                                                'whiskers_z_bs' : 'mean'}).reset_index()

g = sns.JointGrid(df_plot, x = 'whiskers_z_bs', y = 'tc', hue = 'stim_type', hue_order = ['45 and 90', '68', 'non-task'])
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.histplot, kde=False, common_norm = False, element='step', fill = False)


#%%

# df_plot = df_tc[(df_tc.times > 0) & (df_tc.whiskers_z_bs <1.5) & (df_tc.whiskers_z_bs > -1.5)].copy()

df_plot = df_tc.copy()

df_plot = df_plot.groupby(['trial_num_uni','trial_epoch']).agg({'subject' : 'first',
                                                                'stim_type' : 'first',
                                                                'tc' : 'first',
                                                                'whiskers_z_bs' : 'mean'}).reset_index()

df_plot = df_plot[df_plot.trial_epoch=='post']

f,a = plt.subplots(1,5, figsize = (25,15), sharex = True, sharey = True)

for c,s in enumerate(df_plot.subject.unique()):
      
    ind = (df_plot.subject==s)
    sns.regplot(df_plot[ind], x = 'whiskers_z_bs', y = 'tc', ax = a[c])
    a[c].set_title(pearsonr(df_plot[ind].whiskers_z_bs, df_plot[ind].tc))
    a[c].set_box_aspect(1)


sns.catplot(df_plot, x = 'stim_type', y = 'tc', col = 'subject')
sns.catplot(df_plot, x = 'stim_type', y = 'whiskers_z_bs', col = 'subject', kind = 'bar', errorbar = ('se',2))

#%%

# df_plot = df_tc[(df_tc.times > 0) & (df_tc.whiskers_z_bs <1.5) & (df_tc.whiskers_z_bs > -1.5)].copy()

df_plot = df_tc.copy()

df_plot = df_plot.groupby('trial_num_uni').agg({'subject' : 'first',
                                                'stim_type' : 'first',
                                                'tc' : 'first',
                                                'whiskers_z_bs' : 'mean'}).reset_index()

f,a = plt.subplots(3,5, figsize = (25,15), sharex = True, sharey = True)

for c,s in enumerate(df_plot.subject.unique()):
    for r,o in enumerate(df_plot.stim_type.unique()):
        
        ind = (df_plot.subject==s) & (df_plot.stim_type==o)
        sns.regplot(df_plot[ind], x = 'whiskers_z_bs', y = 'tc', ax = a[r,c])
        a[r,c].set_title(pearsonr(df_plot[ind].whiskers_z_bs, df_plot[ind].tc))
        a[r,c].set_box_aspect(1)


sns.catplot(df_plot, x = 'stim_type', y = 'tc', col = 'subject')


#%%

com_speed_z = pd.pivot(df_raster, index = 'trial_num_uni', values = 'com_speed_z', columns = 'times')


sns.heatmap(com_speed_z, rasterized = True)

#%% Find trials where the mouse's eyes had large movements
th = 1

df_raster['bad_trial'] = False

for t in df_raster.trial_num_uni.unique():
    m = df_raster[df_raster.trial_num_uni==t].com_speed.max()
    if m >= th:
        df_raster.loc[df_raster.trial_num_uni==t,'bad_trial'] = True
        
com_speed = pd.pivot(df_raster[df_raster.bad_trial==False], index = 'trial_num_uni', values = 'com_speed', columns = 'times')
sns.heatmap(com_speed, rasterized = True)

#%%


order = ['pre','early','mid','late']
hue_order = ['45 and 90', '68', 'non-task']

df_plot = df_raster.groupby(['subject','times'], as_index =False).mean()
f,a = plt.subplots(1,1)
sns.lineplot(data = df_plot, x = 'times', y ='whiskers_z', ax = a, errorbar = ('se',1))
a.vlines(bins[1:4], ymin = min(a.get_ylim()), ymax = max(a.get_ylim()))
for l,x in zip(order,[sum(l)/2 for l in zip(bins[0::],bins[1::])]):
    a.text(x,0.14,l, horizontalalignment = 'center')
sns.despine(ax = a)
a.set_xlabel('Time relative to stimulus onset (s)')
a.set_ylabel('Whisking (z-scored)')

df_plot = df_raster[df_raster.bad_trial==False].groupby(['subject','times','stim_type'], as_index =False).mean()
f,a = plt.subplots(1,1)
sns.lineplot(data = df_plot, x = 'times', y ='whiskers', hue = 'stim_type', hue_order = hue_order, ax = a, errorbar = ('se',2))
a.legend_.set_visible(False)
# a.vlines(bins[1:4], ymin = min(a.get_ylim()), ymax = max(a.get_ylim()), color = 'k')
# for l,x in zip(order,[sum(l)/2 for l in zip(bins[0::],bins[1::])]):
#     a.text(x,0.45,l, horizontalalignment = 'center')
sns.despine(ax = a)
a.set_xlabel('Time relative to stimulus onset (s)')
a.set_ylabel('Whisking')

df_plot = df_raster[df_raster.bad_trial==False].groupby(['subject','times','stim_type'], as_index =False).mean()
f,a = plt.subplots(1,1)
sns.lineplot(data = df_plot, x = 'times', y ='whiskers_z', hue = 'stim_type', hue_order = hue_order, ax = a, errorbar = ('se',2))
a.legend_.set_visible(False)
a.vlines(bins[1:4], ymin = min(a.get_ylim()), ymax = max(a.get_ylim()), color = 'k')
for l,x in zip(order,[sum(l)/2 for l in zip(bins[0::],bins[1::])]):
    a.text(x,0.45,l, horizontalalignment = 'center')
sns.despine(ax = a)
a.set_xlabel('Time relative to stimulus onset (s)')
a.set_ylabel('Whisking (z-scored)')

df_plot = df_raster.groupby(['subject','times','stim_type'], as_index =False).mean()
f,a = plt.subplots(1,1)
sns.lineplot(data = df_plot, x = 'times', y ='com_speed_z', hue = 'stim_type', hue_order = hue_order, ax = a, errorbar = ('se',2))
a.legend_.set_visible(False)
sns.despine(ax = a)
a.set_xlabel('Time relative to stimulus onset (s)')
a.set_ylabel('Eye speed')

sns.relplot(data = df_raster, x = 'times', y ='com_speed', hue = 'stim_type', kind = 'line', hue_order = hue_order, errorbar = ('se',2),
            col = 'subject', estimator = None, units = 'trial_num', alpha = 0.3)


# df_plot = df_raster.groupby(['subject','times','stim_type'], as_index =False).mean()
# f,a = plt.subplots(1,1)
# sns.lineplot(data = df_plot, x = 'times', y ='pupil_z', hue = 'stim_type', hue_order = hue_order, ax = a, errorbar = ('se',2))
# a.legend_.set_visible(False)
# sns.despine(ax = a)
# a.set_xlabel('Time relative to stimulus onset (s)')
# a.set_ylabel('Pupil (zscored)')


df_plot = df_raster[df_raster.bad_trial==False].groupby(['subject','times','stim_type'], as_index =False).mean()
f,a = plt.subplots(1,1)
sns.lineplot(data = df_plot, x = 'times', y ='whisk_binerize', hue = 'stim_type', hue_order = hue_order, ax = a, errorbar = ('se',1))
a.legend_.set_visible(False)
a.vlines(bins[1:4], ymin = min(a.get_ylim()), ymax = max(a.get_ylim()), color = 'k')
sns.despine(ax = a)
a.set_xlabel('Time relative to stimulus onset (s)')

df_plot = df_raster.copy()
sns.relplot(data = df_plot, x = 'times', y ='whiskers', col = 'subject', hue = 'stim_type', 
            hue_order = hue_order, kind = 'line', errorbar = ('se',1))

corr_type = 'pearson'

df_plot = df_raster[df_raster.bad_trial==False].groupby(['subject','times', 'stim_type'])[['tc','whiskers_z']].corr(corr_type).iloc[0::2,-1].reset_index()
f,a = plt.subplots(1,1)
sns.lineplot(df_plot, x = 'times', y = 'whiskers_z', hue = 'stim_type', errorbar = ('se',2), hue_order = hue_order, ax = a)
sns.despine(ax = a)
a.set_xlabel('Time relative to stimulus onset (s)')
a.set_ylabel('Correlation of whisking with trial convexity')
a.legend_.set_title('Stimulus type')
a.legend_.set_frame_on(False)

p_values = [ttest_ind(df_plot[np.logical_and(df_plot.stim_type=='45 and 90', df_plot.times == t)].whiskers_z,
                      df_plot[np.logical_and(df_plot.stim_type=='68', df_plot.times == t)].whiskers_z)[1] for t in df_plot.times.unique()]

df_plot = df_raster.groupby(['subject','times', 'stim_type'])[['tc','pupil_baseline']].corr(corr_type).iloc[0::2,-1].reset_index()
f,a = plt.subplots(1,1)
sns.lineplot(df_plot, x = 'times', y = 'pupil_baseline', hue = 'stim_type', errorbar = ('se',1), hue_order = hue_order, ax = a)
sns.despine(ax = a)
a.set_xlabel('Time relative to stimulus onset (s)')
a.set_ylabel('Correlation of pupil baseline with trial convexity')
a.legend_.set_title('Stimulus type')
a.legend_.set_frame_on(False)

df_plot = df_raster.groupby(['subject','trial_epoch','stim_type','trial_num'])[['tc','whiskers']].max().reset_index()
df_plot = df_plot.groupby(['subject','trial_epoch', 'stim_type'])[['tc','whiskers']].corr().iloc[0::2,-1].reset_index()
f,a = plt.subplots(1,1)
sns.barplot(data = df_plot, x = 'trial_epoch', y = 'whiskers', hue = 'stim_type', 
            order = order, errorbar = ('se',1), hue_order = hue_order, ax = a)
sns.stripplot(data = df_plot, x = 'trial_epoch', y = 'whiskers', dodge = True, hue = 'stim_type', palette='dark:k',
            order = order, legend = False, hue_order = hue_order, ax = a)
a.set_xlabel('Trial period')
a.set_xticklabels([f'Pre-stimulus ({bins[0]} - {bins[1]} s)', f'Early ({bins[1]} to {bins[2]} s)', f'Mid ({bins[2]} to {bins[3]} s)', f'Late ({bins[3]} to {bins[4]} s)'])
a.set_ylabel('Correlation of whisking with trial convexity')
sns.despine(ax = a)
a.legend_.set_title('Stimulus type')
a.legend_._set_loc(3)
a.legend_.set_frame_on(False)


#%%

# Bin pupil area and look at average TC in different trial periods

whisk_bins = [-1,0,1,2,3,np.inf]
# whisk_labels = [np.mean(x) for x in zip(whisk_bins[0::],whisk_bins[1::])]
whisk_labels = np.arange(len(whisk_bins)-1)

df_plot = df_raster.copy()

df_plot = df_plot.groupby(['subject','stim_type','trial_epoch','trial_num'],observed=True).agg({'whiskers_z' : 'max',
                                                                                                'tc' : 'first'}).reset_index()
df_plot['whisk_bin'] = pd.cut(df_plot.whiskers_z,bins = whisk_bins, labels = whisk_labels,right = True, include_lowest = True)

df_plot = df_plot.groupby(['subject','stim_type','trial_epoch','whisk_bin']).mean().reset_index()


g=sns.relplot(data = df_plot, x = 'whisk_bin', y = 'tc', hue = 'stim_type', col = 'trial_epoch', kind = 'line', 
            errorbar = ('se',2))
g.set_axis_labels('Whisking','Trial convexity')

#%% Plot relationship between whisking and tc for each mouse by stim type

ori_colors = sns.color_palette('bright',3)

f,a = plt.subplots(1,4, sharex = True, sharey = True, figsize = (8,2))

for s in df_raster.subject.unique():
    for si, st in enumerate(df_raster.stim_type.unique()[::-1]):
        for ti,te in enumerate(df_raster.trial_epoch.unique()):
            ind = (df_raster.subject == s) & (df_raster.stim_type==st) & (df_raster.trial_epoch==te)
            df_plot = df_raster[ind].groupby('trial_num')[['whiskers_z','tc']].mean()
            sns.regplot(data = df_plot, x = 'whiskers_z', y = 'tc', color = ori_colors[si], ax = a[ti], scatter_kws = {'s' : 2})

#%% mixed effects model
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_stat = df_raster[df_raster.bad_trial==False].copy()
df_stat = df_stat[df_stat.trial_epoch=='mid'].groupby(['trial_num_uni']).agg({'stim_type':'first',
                                                                                    'tc':'first',
                                                                                    'whiskers_z':'mean',
                                                                                    'subject':'first'}).reset_index()

md = smf.mixedlm("tc ~ stim_type*whiskers_z", df_stat, groups=df_stat["subject"],re_formula="~whiskers_z")
mdf = md.fit()
print(mdf.summary())

df_stat = df_raster[df_raster.bad_trial==False].copy()
df_stat = df_stat.groupby(['trial_num_uni','trial_epoch']).agg({'stim_type':'first',
                                                                  'tc':'first',
                                                                  'whiskers_z':'mean',
                                                                  'subject':'first'}).reset_index()

md = smf.mixedlm("tc ~ C(stim_type)*whiskers_z*C(trial_epoch)", df_stat, groups=df_stat["subject"])
mdf = md.fit()

print(mdf.summary())

#%%

g = sns.relplot(df_raster, x = 'times', y = 'whiskers', col = 'stim_type', row = 'subject', kind = 'line')



df_plot = df_raster.groupby(['trial_num','stim_type','trial_epoch'])[['whiskers','tc']].agg({'whiskers' : 'mean',
                                                                                           'tc' : 'first'}).reset_index()
f,a = plt.subplots(1,3, sharex = True, sharey = True)
for i,s in enumerate(df_plot.stim_type.unique()):
    sns.regplot(df_plot[np.logical_and(df_plot.stim_type==s,df_plot.trial_epoch=='mid')], x = 'whiskers', y = 'tc',
                ax = a[i])
    # a[i].set_xlim([-0.5,4.5])
    # a[i].set_ylim([0.7,3])
    

df_plot = df_raster.groupby(['trial_num','stim_type','trial_epoch'])[['whiskers','tc']].agg({'whiskers' : 'mean',
                                                                                           'tc' : 'first'}).reset_index()
f,a = plt.subplots(1,3, sharex = True, sharey = True)  
ori_colors = sns.color_palette('colorblind', 3)
for i,s in enumerate(df_plot.stim_type.unique()):
    sns.regplot(df_plot[np.logical_and(df_plot.stim_type==s,df_plot.trial_epoch=='late')], x = 'whiskers', y = 'tc', order = 1,
                color = ori_colors[i])


for t in df_raster[df_raster.trial_epoch=='mid'].times.unique():

    f,a = plt.subplots(1,3, sharex = True, sharey = True, figsize = (6,2))
    ori_colors = sns.color_palette('colorblind', 3)
    for i,s in enumerate(df_raster.stim_type.unique()[::-1]):
        sns.regplot(df_raster[(df_raster.stim_type==s) & (df_raster.times==t)], x = 'whiskers', y = 'tc', order = 3,
                    color = ori_colors[i], ax = a[i], scatter_kws = {'s' : 2})
        if i == 1:
            a[i].set_title(t)

        a[i].set_box_aspect(1)
        # a[i].set_ylim([-0.5,3])

#%%

df_plot = df_raster.copy()
df_plot = df_plot.groupby(['subject','stim_type','trial_epoch','trial_num'],observed=True).agg({'whiskers_z' : 'max',
                                                                                                'tc' : 'first'}).reset_index()

df_plot = df_plot.groupby(['subject','stim_type','trial_epoch']).mean()

sns.relplot(data = df_plot, x = 'whiskers_z', y = 'tc', hue = 'stim_type', col = 'trial_epoch')





# %%

import statsmodels.api as sm
import statsmodels.formula.api as smf

md = smf.mixedlm("tc ~ whiskers*C(times)", df_raster[df_raster.stim_type == '45 and 90'], groups=df_raster[df_raster.stim_type == '45 and 90'].subject)
mdf = md.fit()
print(mdf.summary())

md = smf.mixedlm("tc ~ whiskers*C(times)*C(stim_type)", df_raster, groups=df_raster['subject'])
mdf = md.fit()
print(mdf.summary())
# %%
