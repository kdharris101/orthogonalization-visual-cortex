#%%
"""

Plot drifting grating pixel maps

@author: Samuel Failor
"""
from os.path import join
import glob
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# results_dir = r'H:/OneDrive for Business/Results'
results_dir = r'C:/Users/Samuel/OneDrive - University College London/Results'
# results_dir = r'C:/Users/samue/OneDrive - University College London/Results'
# results_dir = '/mnt/c/Users/Samuel/OneDrive - University College London/Results'

# subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
#             'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
#             'SF180613']

# subjects = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B','SF171107', 
#             'SF171107','SF180515', 'SF180515', 'SF180613', 'SF180613']

# expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
#               '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
#               '2018-06-28', '2018-12-12','2018-08-06']

# expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]
# expt_nums_ret = [6, 8, 9, 2, 8, 4, 2, 4, 2, 6]

# trained = [False, True, False, True, False, True, False, True, False, True]

# condition = ['Naïve', 'Proficient', 'Naïve', 'Proficient', 'Naïve', 
#              'Proficient', 'Naïve', 'Proficient', 'Naïve', 'Proficient',
#              'Naïve']

# subjects_file = ['SF190319']
# subjects = ['SF190319']
# expt_dates = ['2019-05-21']
# expt_nums = [4]
# expt_nums_ret = [5]
# trained = [False]
# condition = ['Naïve']

subjects_file = ['SF008']
subjects = ['SF008']
expt_dates = ['2023-08-31']
expt_nums = [3]
expt_nums_ret = [4]
trained = [True]
condition = ['Passive']


# Load most up-to-date saved results
file_paths = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                   str(expt_nums[i]), 'drift_pixel_maps*'))[-1] 
            for i in range(len(subjects))]

file_paths_ret = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
            str(expt_nums_ret[i]), r'_'.join([subjects_file[i], expt_dates[i],
            str(expt_nums_ret[i]), 'peak_pixel_retinotopy*'])))[-1] 
                    for i in range(len(subjects))]

# file_paths_cells = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
#                     str(expt_nums[i]), r'_'.join([subjects_file[i], expt_dates[i],
#                     str(expt_nums[i]), 'orientation tuning_norm by pref ori_[2020]*'])))[-1] 
#                                             for i in range(len(subjects))]

#%%
def plane_sm(plane, med_bin = 10, sigma = 60, mode = 'reflect'):
    
    plane_sm = np.zeros_like(plane)
    
    for i,p in enumerate(plane):
        plane_sm[i,...] = ndimage.median_filter(p,med_bin)
        plane_sm[i,...] = ndimage.gaussian_filter(plane_sm[i,...],sigma,
                                                  mode = mode)
        
    return plane_sm

#%% 

'''For each subject and plane, load pixel df/f averages for each stimulus condition, 
flatten and add to a dataframe for plotting response distributions by 
stimulus condition'''

# planes = [2,3,4]

pm_df = pd.DataFrame(columns = ['Stimulus', 'Pixels', 'Task_dist','Subject', 
                                'Trained'])

for i,f in enumerate(file_paths):
    
    # print('Loading cell info ' + file_paths_cells[i])
    cells = np.load(file_paths_cells[i], allow_pickle = True)[()]
    
    # Add expt retinotopy
    print('Loading pixel retinotopy ' + file_paths_ret[i])
    plane_ret = np.load(file_paths_ret[i])
    
    # Average across all planes
    plane_ret = plane_ret.mean(2)
    # Reshape to 2d array for elv and azi
    plane_ret = plane_ret.reshape(2,512,512)
  
    # Smooth with function    
    plane_ret_sm = plane_sm(plane_ret)
    
    task_ret = plane_ret_sm[0,...] + (plane_ret_sm[1,...] + 80) * 1j
    task_ret = np.abs(task_ret)
    
    # Use gradient maps of elevation to find border of V1
    sx_elv = ndimage.sobel(plane_ret_sm[0,...], axis = 0, mode = 'reflect')
    sy_elv = ndimage.sobel(plane_ret_sm[0,...], axis = 1, mode = 'reflect')
    ang_elv = np.arctan2(sx_elv,sy_elv)
    
    sx_azi = ndimage.sobel(plane_ret_sm[1,...], axis = 0, mode = 'reflect')
    sy_azi = ndimage.sobel(plane_ret_sm[1,...], axis = 1, mode = 'reflect')
    ang_azi = np.arctan2(sx_azi,sy_azi)

    # Threshold, i.e. sign map > 0 for V1
    # ang_elv_th = ang_elv > 0
    
    ang_diff = np.sin(ang_elv - ang_azi)
    
    ang_th = np.zeros_like(ang_elv)

    ang_th[ang_diff > 0.31] = 1
    
    # Find largest positive region
    ang_th,_ = ndimage.label(ang_th)
    ang_th = (ang_th == (np.bincount(ang_th.flat)[1:].argmax() + 1))
    
    for p in range(1,cells['ops'][0]['nplanes']):
    # for p in planes:
        print('Loading pixel grating responses - plane ' + str(p))
        pm = np.load(f)[...,p]
               
        plane_ind = cells['cell_plane'] == p
        
        non_cell = np.ones_like(pm[...,0]).astype(bool)
        
        for c in cells['stat']:
            non_cell[c['ypix'],c['xpix']] = False
    
        # breakpoint()
        
        # Only include pixels in V1 and exclude cell rois
        pm = pm[np.logical_and(ang_th,non_cell),...]
        task_ret_plane = task_ret[np.logical_and(ang_th,non_cell),...]
       
        stim_label = np.repeat(np.insert((np.arange(0,360,22.5)),0,-1)[None,:],pm.shape[0],axis=0)
        # stim_label = np.ceil(stim_label)
        sub_label = np.repeat(subjects[i],pm.size)
        cond_label = np.repeat(condition[i],pm.size)
        
        pm_df = pd.DataFrame.append(pm_df,
                                    pd.DataFrame({'Stimulus' : np.mod(stim_label,180).flatten(),
                                                  'Pixels' : pm.flatten(),
                                                  'Task_dist' : np.repeat(
                                                      task_ret_plane[:,None]
                                                      ,17,axis=1).flatten(),
                                                  'Subject' : sub_label,
                                                  'Trained' : cond_label}))
        
#%% Bin by task_dist

pm_df['Task_dist_bin'] = pd.cut(pm_df.Task_dist, np.arange(0,40,5),
                                labels = np.arange(5,40,5))
    
#%% Do stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp

df_stats = pd.DataFrame.copy(pm_df)

df_stats = df_stats.groupby(['Subject','Trained','Stimulus']).mean().reset_index()
df_stats['Stimulus'] = df_stats['Stimulus'].astype(int).astype(str)

model = ols('Pixels ~ Trained * Stimulus', df_stats).fit()

modelRM = AnovaRM(df_stats, 'Pixels', 'Subject', within=['Trained','Stimulus'], aggregate_func='mean')
result = modelRM.fit()

# Multiple comparisons
p = np.zeros(len(np.unique(df_stats.Stimulus)))
stim = np.copy(p)
for i,s in enumerate(np.unique(df_stats.Stimulus)):
    p[i] = ttest_1samp((df_stats[np.logical_and(df_stats.Stimulus == s,
                                               df_stats.Trained == 'Proficient')].Pixels
                        .to_numpy() - 
                       df_stats[np.logical_and(df_stats.Stimulus == s,
                                               df_stats.Trained == 'Naïve')].Pixels
                       .to_numpy()),
                       0)[1]
    stim[i] = s

ind = np.argsort(stim)
stim = stim[ind]
p = p[ind]

p_corr = multipletests(p,0.05,'hs')[1]

#%%

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

df_plot = pd.DataFrame.copy(pm_df)

df_plot = df_plot.groupby(['Subject','Trained','Stimulus','Task_dist_bin'],
                          as_index = False).mean().reset_index()

# df_plot = df_plot[df_plot.Stimulus != 179]

#%%

colors = sns.color_palette('rainbow_r',6)

df_plot_t = pd.DataFrame.copy(df_plot[df_plot.Task_dist_bin<35])
df_plot_t.Task_dist_bin = df_plot_t.Task_dist_bin.astype(int)

xlabels = [str(x) + r'$\degree$' for x in np.ceil(np.arange(0,180,22.5)).astype(int)]
leg_labels = ['0 - 5', '5 - 10', '10 - 15', '15 - 20', '20 - 25', '25 - 30']

df_plot_t = df_plot_t[df_plot_t.Stimulus != 179]

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5, 
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":5,
                                        "ytick.major.size":5,
                                        "patch.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):
    
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    
    f = plt.figure(figsize=(2,1.55))
    ax = f.add_subplot(111)
    sns.lineplot(data = df_plot_t, x = 'Stimulus', y = 'Pixels', 
                      hue = 'Task_dist_bin', style = 'Trained',
                      errorbar = 'se', palette = colors, 
                      style_order = ['Proficient','Naïve'],
                      legend = True, ax = ax)
    
    # fig = sns.relplot(data = df_plot, x = 'Stimulus', y = 'Pixels', 
    #                   style = 'Trained', hue = 'Task_dist_bin', kind = 'line',
    #                   ci = 68, palette = 'rainbow',
    #                   legend = True)
    # plt.ylabel(r'Pixel $\frac{\delta f}{f}$')
    ax.set_ylabel('Neuropil df/f')
    ax.set_xlabel('Stimulus orientation')
    ax.set_xticks(np.ceil(np.arange(0,180,22.5)))
    # plt.xticks(np.unique(df_plot.Stimulus),np.insert(np.ceil(np.arange(0,180,22.5))
    #                                            .astype(int).astype(str),8,'Blank'))
    ax.set_xticklabels(xlabels)
    ax.get_xticklabels()[2].set_weight("bold")
    ax.get_xticklabels()[3].set_weight("bold")
    ax.get_xticklabels()[4].set_weight("bold")
       
    h,l = ax.get_legend_handles_labels()
    
    h_1 = h[1:7]
    h_2 = h[8:]
    h_2.reverse()
    l_1 = leg_labels
    l_2 = l[8:]
    l_2.reverse()


    ax.legend_.remove()
    leg1 = f.legend(h_1,l_1, ncol=2, 
                      title = 'Ret. dist. from task stim. (vis. deg)', 
                      frameon = False,
                      bbox_to_anchor = (0.55,0.9), 
                      loc = 'center', handletextpad=0.2, labelspacing=0.2, 
                      borderpad = 0)
    leg2 = f.legend(h_2,l_2, title = '', frameon = False,
                    bbox_to_anchor = (0.85,0.88), loc = 'center',
                    handletextpad=0.2, labelspacing=0.2, borderpad = 0)
    # f.add_artist(leg1)
    sns.despine(trim = True)
    f.tight_layout()
    f.savefig(join(results_dir,'Figures','Draft','pixel_responses_by_task_dist.svg'),
              format = 'svg', dpi = 600)
      
    # Plot difference
    # df_naive = df_plot[df_plot.Trained == 'Naïve']
    # df_trained = df_plot[df_plot.Trained == 'Proficient']
    
    # df_diff = pd.DataFrame(columns = ['Pixels', 'Stimulus'])
    
    # df_diff['Pixels'] = df_trained.Pixels.to_numpy() - df_naive.Pixels.to_numpy()
    # df_diff['Stimulus'] = df_trained.Stimulus.to_numpy()
    
    # d_fig = sns.relplot(data=df_diff,x = 'Stimulus', y='Pixels', kind = 'line',
    #                    ci = 68)
    # plt.ylabel('Pixel df/f')
    # plt.xlabel('Stimulus orientation (deg)')
    # # plt.xticks(np.ceil(np.arange(0,180,22.5)))
    # plt.xticks(np.unique(df_plot.Stimulus),np.insert(np.ceil(np.arange(0,180,22.5))
    #                                            .astype(int).astype(str),8,'Blank'))
    
    # colors = sns.hls_palette(8)
    # colors.append((0,0,0))
    
    # colors = sns.xkcd_palette(['grey'])
    
    # df_diff['Stimulus'] = np.ceil(df_diff['Stimulus']).astype(int).astype(str)
    # df_diff.loc[df_diff.Stimulus == '179','Stimulus'] = 'Blank'
    
    # plt.figure(figsize = (2.2,1.5))
    # d_fig = sns.stripplot(data = df_diff, x = 'Stimulus', y = 'Pixels',
    #             palette = colors, edgecolor = 'k', zorder = 1,
    #             size = 3, linewidth = 0.5)
    # plt.ylabel(r'$\Delta$Pixel df/f')
    # plt.xlabel('Stimulus orientation (deg)')
    # d_fig.get_xticklabels()[2].set_weight("bold")
    # d_fig.get_xticklabels()[3].set_weight("bold")
    # d_fig.get_xticklabels()[4].set_weight("bold")
    
    # mean_width = 1
    
    # for tick, text in zip(d_fig.get_xticks(), d_fig.get_xticklabels()):
    #     sample_name = text.get_text()  # "X" or "Y"
    
    #     # calculate the median value for all replicates of either X or Y
    #     mean_val = df_diff[df_diff['Stimulus']==sample_name].Pixels.mean()
    
    #     # plot horizontal lines across the column, centered on the tick
    #     # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2], 
    #     #               [mean_val, mean_val], lw=4, color='k')
    #     n_points = len(df_diff)/len(np.unique(df_diff['Stimulus']))
    #     ci_val = mean_confidence_interval(
    #         df_diff[df_diff['Stimulus']==sample_name].Pixels, confidence = 0.68)
    #     # pdif_fig.plot([mean_val, mean_val], 
    #     #               [tick-mean_width/2, tick+mean_width/2],
    #     #               lw=4, color='k', linestyle='--')
    #     # pdif_fig.plot([ci_val1, ci_val2], 
    #     #               [tick, tick],
    #     #               lw=4, color='k')
    #     d_fig.plot([tick-mean_width/2, tick+mean_width/2],
    #                   [mean_val, mean_val],
    #                   color='k', linestyle='--', zorder = 2)
    #     # pdif_fig.plot([tick, tick],
    #     #               [ci_val1, ci_val2],
    #     #               lw=4, color='k')
    #     d_fig.errorbar(tick,mean_val, ci_val, ecolor = 'k',
    #                       capsize = 3,capthick=0.5, zorder = 2)
        
    # d_fig.plot([min(d_fig.get_xlim()),max(d_fig.get_xlim())], [0,0],'--k',
    #            zorder = 1)
    
    # sns.despine()
    
    # # for t,tick in enumerate(d_fig.get_xticks()):
    # #     if p_corr[t] < 0.05:
    # #         d_fig.text(tick,0.001,'*', fontsize = 14, horizontalalignment = 'center')
    
    # plt.tight_layout()

#%% Plot retinotopic maps

from mpl_toolkits.axes_grid1 import make_axes_locatable

ret_sigma = 20
med_bin = 10

ret_map = np.load(file_paths_ret[0])

# ret_map = ret_map.mean(2)
ret_map = ret_map[...,2]
# Reshape to 2d array for elv and azi
ret_map = ret_map.reshape(2,512,512)
for i,r in enumerate(ret_map):
    ret_map[i,...] = np.flip(r,1)
    
# Smooth averaged map
# ret_map_sm = np.append(
# ndimage.gaussian_filter(ndimage.median_filter(ret_map[0,...],10), 
#                         ret_sigma, mode = 'reflect')[None,...],
# ndimage.gaussian_filter(ndimage.median_filter(ret_map[1,...],10), 
#                         ret_sigma, mode = 'reflect')[None,...],
# axis = 0)

# ret_map_sm = plane_sm(ret_map, med_bin = med_bin, sigma = ret_sigma)
ret_map_sm = ret_map

# Distance from task stim
task_ret = ret_map_sm[0,...] + (ret_map_sm[1,...] + 80) * 1j
task_ret = np.abs(task_ret)

# Use gradient maps of elevation to find border of V1
sx_elv = ndimage.sobel(ret_map_sm[0,...], axis = 0)
sy_elv = ndimage.sobel(ret_map_sm[0,...], axis = 1)
ang_elv = np.arctan2(sy_elv,sx_elv)

sx_azi = ndimage.sobel(ret_map_sm[1,...], axis = 0)
sy_azi = ndimage.sobel(ret_map_sm[1,...], axis = 1)
ang_azi = np.arctan2(sy_azi,sx_azi)


vmin_elv = np.percentile(ret_map_sm[0,...].flatten(), 1)
vmax_elv = np.percentile(ret_map_sm[0,...].flatten(), 99)

vmin_azi = np.percentile(ret_map_sm[1,...].flatten(), 5)
vmax_azi = np.percentile(ret_map_sm[1,...].flatten(), 95)

xv,yv = np.meshgrid(np.arange(512),np.arange(512))

# cmap = 'gist_rainbow'
cmap = 'turbo'
# cmap = 'jet'

# f,axes = plt.subplots(1,2,figsize = (20,8))
f,axes = plt.subplots(1,2,figsize = (20,8))

im_elv = sns.heatmap(ret_map_sm[0,...], vmin = vmin_elv, vmax = vmax_elv,
            square = True, cbar = True, rasterized = True, cmap = cmap,
            ax = axes[1])
axes[1].tick_params(left = False, bottom = False)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title('Elevation')

im_azi = sns.heatmap(ret_map_sm[1,...], vmin = vmin_azi, vmax = vmax_azi,
            square = True, cbar = True, rasterized = True, cmap = cmap,
            ax = axes[0])
axes[0].tick_params(left = False, bottom = False)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title('Azimuth')


#%% Plot retinotopic maps and V1 mask

from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
from skimage.morphology import local_minima

ret_sigma = 60

ret_map = np.load(file_paths_ret[0])

ret_map = ret_map.mean(2)
# Reshape to 2d array for elv and azi
ret_map = ret_map.reshape(2,512,512)
for i,r in enumerate(ret_map):
    ret_map[i,...] = np.flip(r,1)
    
# Smooth averaged map
# ret_map_sm = np.append(
# ndimage.gaussian_filter(ndimage.median_filter(ret_map[0,...],10), 
#                         ret_sigma, mode = 'reflect')[None,...],
# ndimage.gaussian_filter(ndimage.median_filter(ret_map[1,...],10), 
#                         ret_sigma, mode = 'reflect')[None,...],
# axis = 0)

ret_map_sm = plane_sm(ret_map, sigma = ret_sigma)

# Distance from task stim
task_ret = ret_map_sm[0,...] + (ret_map_sm[1,...] + 80) * 1j
task_ret = np.abs(task_ret)

# Use gradient maps of elevation to find border of V1
sx_elv = ndimage.sobel(ret_map_sm[0,...], axis = 0)
sy_elv = ndimage.sobel(ret_map_sm[0,...], axis = 1)
ang_elv = np.arctan2(sy_elv,sx_elv)

sx_azi = ndimage.sobel(ret_map_sm[1,...], axis = 0)
sy_azi = ndimage.sobel(ret_map_sm[1,...], axis = 1)
ang_azi = np.arctan2(sy_azi,sx_azi)

# Threshold, i.e. sign map > 0 for V1
# ang_diff = ang_elv
ang_diff = np.sin(ang_elv - ang_azi)


ang_th = np.zeros_like(ang_elv)
# ang_th[ang_diff > scale*np.std(ang_diff)] = 1
# ang_th[ang_diff < -scale*np.std(ang_diff)] = -1


ang_th[ang_diff > 0.31] = 1

ang_th,_ = ndimage.label(ang_th)
ang_th = (ang_th == (np.bincount(ang_th.flat)[1:].argmax() + 1))


vmin_elv = np.percentile(ret_map_sm[0,...].flatten(), 1)
vmax_elv = np.percentile(ret_map_sm[0,...].flatten(), 99)

vmin_azi = np.percentile(ret_map_sm[1,...].flatten(), 1)
vmax_azi = np.percentile(ret_map_sm[1,...].flatten(), 99)

xv,yv = np.meshgrid(np.arange(512),np.arange(512))

task_ret_bin = np.digitize(task_ret,np.arange(0,30,5))
b_contours = task_ret_bin
colors = sns.color_palette('rainbow_r',6)

# cmap = 'gist_rainbow'
cmap = 'turbo'
# cmap = 'jet'

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5, 
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":5,
                                        "ytick.major.size":5,
                                        "patch.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    # f,axes = plt.subplots(1,2,figsize = (20,8))
    # f,axes = plt.subplots(1,3,figsize = (3,1))
    
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'
    
    f = plt.figure(figsize=(20,8))
    axes = [f.add_subplot(1,3,i+1) for i in range(3)]
    
    im_elv = sns.heatmap(ret_map_sm[0,...], vmin = vmin_elv, vmax = vmax_elv,
                square = True, cbar = False, rasterized = True, cmap = cmap,
                ax = axes[1])
    axes[1].tick_params(left = False, bottom = False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('Elevation')
    # axes[1].contour(xv,yv,ang_th,colors='w')
    
    im_azi = sns.heatmap(ret_map_sm[1,...], vmin = vmin_azi, vmax = vmax_azi,
                square = True, cbar = False, rasterized = True, cmap = cmap,
                ax = axes[0])
    axes[0].tick_params(left = False, bottom = False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title('Azimuth')
    # axes[0].contour(xv,yv,ang_th,colors='w')
    
    # im_ang = sns.heatmap(ang_diff, center = 0, square = True, cbar = False, rasterized = True, 
    #             cmap = 'bwr', ax = axes[2], vmin = -1, vmax = 1)
    # axes[2].tick_params(left = False, bottom = False)
    # axes[2].set_xticks([])
    # axes[2].set_yticks([])
    # axes[2].set_title('Angle of elevation gradient')
    
    
    # sns.heatmap(ang_th, square = True, cbar = False, rasterized = True, 
    #             cmap = 'bwr', ax = axes[3])
    # axes[3].tick_params(left = False, bottom = False)
    # axes[3].set_xticks([])
    # axes[3].set_yticks([])
    # axes[3].set_title('V1 mask')
    
    im_task = sns.heatmap(task_ret, square = True, cbar = False, 
                          rasterized = True, cmap = 'Greys_r', axes = axes[2])
    axes[2].tick_params(left = False, bottom = False)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title('Distance from task stimulus')
    # axes[2].contour(xv,yv,ang_th,colors='w')
    # axes[2].contour(xv,yv,b_contours,cmap = colors)

    c_width = 0.5
    
    c_levels = [5,10,15,20,25]
    
    for i,c in enumerate(c_levels):
    
        task_contours = measure.find_contours(task_ret,c)
        
        for contour in task_contours:
            axes[2].plot(contour[:,1],contour[:,0],linestyle='dashed',color = colors[i+1], 
                         linewidth=c_width, dashes=(3,6))
            
    V1_contours =  measure.find_contours(ang_th,0)
        
    for contour in V1_contours:
            axes[0].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)
            axes[1].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)
            axes[2].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)


    # Plot center of stim
    
    minima = np.where(local_minima(task_ret))
    
    for m in range(2):
        axes[2].plot(minima[1][m], minima[0][m],color=colors[0], markersize = 1,
                     marker = '.')

    l = axes[2].legend(labels=[r'0',r'5',r'10',
                           r'15',r'20',r'25'],
                   fancybox = False, frameon = False, ncol = 1,
                   handletextpad=0.2, labelspacing=0.2, borderpad = 0,
                   handlelength = 1, columnspacing = 0.75)
    
    for line in l.get_lines():
        line.set_linestyle('-')    
    
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("bottom", size="3%", pad=0.5)
    plt.colorbar(im_azi.get_children()[0], cax=cax, orientation='horizontal',
                  label = 'Visual degrees')
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("bottom", size="3%", pad=0.5)
    plt.colorbar(im_elv.get_children()[0], cax=cax, orientation='horizontal',
                  label = 'Visual degrees')
    
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("bottom", size="3%", pad=0.5)
    plt.colorbar(im_task.get_children()[0], cax=cax, orientation='horizontal',
                  label = 'Visual degrees')
    cax.axis('off')
    
    # divider = make_axes_locatable(axes[3])
    # cax = divider.append_axes("bottom", size="3%", pad=0.5)
    # cax.axis('off')
    
    f.tight_layout()
    f.savefig(r'C:\Users\Samuel\OneDrive - University College London\Results\Figures\Draft\ret_maps.svg',
        format = 'svg', dpi = 600)

#%% Plot retinotopic distance from task stimulus

from mpl_toolkits.axes_grid1 import make_axes_locatable


ret_sigma = 60

ret_map = np.load(file_paths_ret[1])

ret_map = ret_map.mean(2)
# Reshape to 2d array for elv and azi
ret_map = ret_map.reshape(2,512,512)
for i,r in enumerate(ret_map):
    ret_map[i,...] = np.flip(r,1)
    
# Smooth averaged map
ret_map_sm = np.append(
ndimage.gaussian_filter(ndimage.median_filter(ret_map[0,...],10), 
                        ret_sigma, mode = 'reflect')[None,...],
ndimage.gaussian_filter(ndimage.median_filter(ret_map[1,...],10), 
                        ret_sigma, mode = 'reflect')[None,...],
axis = 0)

task_ret = ret_map_sm[0,...] + (ret_map_sm[1,...] + 80) * 1j
task_ret = np.abs(task_ret)

cmap = 'Greys_r'

f,axes = plt.subplots(1,1,figsize = (8,8))


im_task = sns.heatmap(task_ret, 
            square = True, cbar = False, rasterized = True, cmap = cmap,
            ax = axes)
axes.tick_params(left = False, bottom = False)
axes.set_xticks([])
axes.set_yticks([])
axes.set_title('Distance from task stimulus')

divider = make_axes_locatable(axes)
cax = divider.append_axes("bottom", size="3%", pad=0.5)
plt.colorbar(im_task.get_children()[0], cax=cax, orientation='horizontal',
             label = 'Visual degrees')

f.tight_layout()

#%% Plot contours

from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
from skimage.morphology import local_minima

colors = sns.color_palette('rainbow_r',6)
cmap = 'icefire'

expt_ind = [0,1]

map_sigma = 60

ret_map_0 = np.load(file_paths_ret[expt_ind[0]])
ret_map_1 = np.load(file_paths_ret[expt_ind[0]])
    
task_pos = [0,-80]

b_contours = np.zeros(2,dtype='object')
task_ret = np.zeros(2,dtype='object')

for i,r in enumerate([ret_map_0,ret_map_1]):

    r = r.mean(2)
    # Reshape to 2d array for elv and azi
    r = r.reshape(2,512,512)
    
    r[0,...] = ndimage.gaussian_filter(ndimage.median_filter(
                        np.flip(r[0,...],1),10), map_sigma, mode = 'reflect')
    r[1,...] = ndimage.gaussian_filter(ndimage.median_filter(
                        np.flip(r[1,...],1),10), map_sigma, mode = 'reflect')
    
    task_ret[i] = (r[0,...] - task_pos[0]) + (r[1,...] - task_pos[1]) * 1j
    task_ret[i] = np.abs(task_ret[i])

    # task_ret_bin = np.digitize(task_ret,np.arange(0,30,5))

    # b_contours[i] = task_ret_bin

    # for i,b in enumerate(np.unique(task_ret_bin)):
    #     b_ind = task_ret_bin == b
    #     b_contours[i] = np.where(np.diff(b_ind,prepend=b_ind[:,0][:,None], axis = 1) + 
    #                   np.diff(b_ind,prepend=b_ind[0,:][None,:],axis = 0) != 0)


# Find V1 outline for expts

map_sigma = 60

ret_map_0 = np.load(file_paths_ret[expt_ind[0]])
ret_map_1 = np.load(file_paths_ret[expt_ind[0]])

V1_outlines = np.zeros(2,dtype='object')

for i,r in enumerate([ret_map_0,ret_map_1]):

    r = r.mean(2)
    # Reshape to 2d array for elv and azi
    r = r.reshape(2,512,512)
    
    r[0,...] = np.flip(r[0,...],1)
    r[1,...] = np.flip(r[1,...],1)
    
    r = plane_sm(r)
    
    # r[0,...] = ndimage.gaussian_filter(
    #             ndimage.median_filter(np.flip(r[0,...],1),10),map_sigma)
    # r[1,...] = ndimage.gaussian_filter(
    #             ndimage.median_filter(np.flip(r[1,...],1),10),map_sigma)
    
    # Use gradient maps of elevation to find border of V1
    sx_elv = ndimage.sobel(r[0,...], axis = 0, mode = 'reflect')
    sy_elv = ndimage.sobel(r[0,...], axis = 1, mode = 'reflect')
    ang_elv = np.arctan2(sy_elv,sx_elv)
    
    sx_azi = ndimage.sobel(r[1,...], axis = 0, mode = 'reflect')
    sy_azi = ndimage.sobel(r[1,...], axis = 1, mode = 'reflect')
    ang_azi = np.arctan2(sy_azi,sx_azi)
    
    # Threshold, i.e. sign map > 0 for V1
    # ang_diff = ang_elv
    ang_diff = np.sin(ang_elv - ang_azi)
    
   
    ang_th = np.zeros_like(ang_elv)
    # ang_th[ang_diff > scale*np.std(ang_diff)] = 1
    # ang_th[ang_diff < -scale*np.std(ang_diff)] = -1
    
    
    ang_th[ang_diff > 0.31] = 1
    
    ang_th,_ = ndimage.label(ang_th)
    ang_th = (ang_th == (np.bincount(ang_th.flat)[1:].argmax() + 1))
    
    # V1_outlines[i] = np.where((np.diff(ang_th,prepend=ang_th[:,0][:,None],axis = 1) + 
    #               np.diff(ang_th,prepend=ang_th[0,:][None,:],axis = 0)) != 0)
    V1_outlines[i] = ang_th
    

from matplotlib import ticker

stimuli = np.insert(np.ceil(np.arange(0,360,22.5)%180),0,-1).astype(int)

plane = 3
i_stim = [23,45,68,90]
stim_inds = np.isin(stimuli,i_stim)
# stim_inds = [7,8,1] # 135, 158, 0

pm1 = np.load(file_paths[expt_ind[0]])[...,stim_inds,plane].reshape(512,512,
                                            int(np.sum(stim_inds)/2),2).mean(3)
pm2 = np.load(file_paths[expt_ind[0]])[...,stim_inds,plane].reshape(512,512,
                                            int(np.sum(stim_inds)/2),2).mean(3)

# f,axes = plt.subplots(nrows = 2, ncols = len(i_stim), figsize = (20,12))
f,axes = plt.subplots(nrows = 1, ncols = len(i_stim), figsize = (20,12))


vmin = np.percentile(np.append(pm1,pm2).flatten(),0.3)
vmax = np.percentile(np.append(pm1,pm2).flatten(),99.7)

xv,yv = np.meshgrid(np.arange(512),np.arange(512))

from matplotlib.colors import TwoSlopeNorm

norm = TwoSlopeNorm(vcenter = 0, vmin = vmin, vmax = vmax)




for i,s in enumerate(i_stim):
    
    # im = sns.heatmap(np.flip(pm1[...,i],1), #center = 0, vmin = vmin, vmax = vmax, 
    #             ax = axes[0,i], square = True, cbar = False, rasterized = True,
    #             cmap = cmap, norm = norm)
    # axes[0,i].tick_params(left = False, bottom = False)
    # axes[0,i].set_xticks([])
    # axes[0,i].set_yticks([])
    # axes[0,i].set_title(str(s) + r'$\degree$', pad = 2)

    im = sns.heatmap(np.flip(pm1[...,i],1), #center = 0, vmin = vmin, vmax = vmax, 
                ax = axes[i], square = True, cbar = False, rasterized = True,
                cmap = cmap, norm = norm)
    axes[i].tick_params(left = False, bottom = False)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(str(s) + r'$\degree$', pad = 2)
   
    # sns.heatmap(np.flip(pm2[...,i],1), #center = 0, vmin = vmin, vmax = vmax, 
    #             ax = axes[1,i], square = True, cbar = False, rasterized = True,
    #             cmap = cmap, norm = norm)
    # axes[1,i].tick_params(left = False, bottom = False)
    # axes[1,i].set_xticks([])
    # axes[1,i].set_yticks([])
    # axes[1,i].plot(V1_outlines[1][1],V1_outlines[1][0],'.g', ms = 0.5)
    # axes[1,i].contour(xv,yv,V1_outlines[1],colors='w', linewidths = 0.25)
    
    # for b in b_contours:
    #     s_ind = np.lexsort(b)
    #     axes[1,i].plot(b[1][s_ind],b[0][s_ind],'.w',ms = 1)
    # axes[1,i].plot(b_contours[1][1],b_contours[1][0],'.w', ms = 0.5)
    # axes[1,i].contour(xv,yv,b_contours[1], cmap = colors, linewidths = 0.25)
    
    if i == 0:
        # axes[0,i].set_ylabel('Naïve')
        # axes[1,i].set_ylabel('Proficient')
        axes[i].set_ylabel('Passive')
        
        
    c_width = 0.5
    
    c_levels = [5,10,15,20,25]

    task_contours = np.zeros(2,dtype='object')
    V1_contours = np.copy(task_contours)

    for ic,c in enumerate(c_levels):

        task_contours[0] = measure.find_contours(task_ret[0],c)
        # task_contours[1] = measure.find_contours(task_ret[1],c)
        
        for contour in task_contours[0]:
            # axes[0,i].plot(contour[:,1],contour[:,0],linestyle='dashed',color = colors[ic+1], 
            #              linewidth=c_width, dashes=(3,6))
            axes[i].plot(contour[:,1],contour[:,0],linestyle='dashed',color = colors[ic+1], 
                         linewidth=c_width, dashes=(3,6))
            
        # for contour in task_contours[1]:
        #     axes[1,i].plot(contour[:,1],contour[:,0],linestyle='dashed',color = colors[ic+1], 
        #                  linewidth=c_width, dashes=(3,6))
        
    V1_contours[0] =  measure.find_contours(V1_outlines[0],0)
    # V1_contours[1] =  measure.find_contours(V1_outlines[1],0)
    
    for contour in V1_contours[0]:
            # axes[0,i].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)
            axes[i].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)
            
    # for contour in V1_contours[1]:
    #         axes[1,i].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)
            
    # Plot center of stim
    
    minima = np.where(local_minima(task_ret[0]))
    
    for m in range(2):
        # axes[0,i].plot(minima[1][m], minima[0][m],color=colors[0], markersize = 1,
        #              marker = '.')
        axes[i].plot(minima[1][m], minima[0][m],color=colors[0], markersize = 1,
                     marker = '.')
        
    minima = np.where(local_minima(task_ret[1]))
    
    # for m in range(2):
    #     axes[1,i].plot(minima[1][m], minima[0][m],color=colors[0], markersize = 1,
    #                  marker = '.')

f.tight_layout()
    

mappable = im.get_children()[0]
cb = plt.colorbar(mappable, ax = axes, orientation = 'vertical', shrink = 0.35,
             label = 'df/f', pad = 0.02)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()

# f.savefig(r'C:\Users\Samuel\OneDrive - University College London\Results\Figures\Draft\drift_maps_23_45_68_90.svg',
#         format = 'svg', dpi = 600)

#%% Plot retinotopic maps and V1 mask

ret_sigma = 20
map_sigma = 50

ret_map = np.load(file_paths_ret[0])

ret_map = ret_map.mean(2)
# Reshape to 2d array for elv and azi
ret_map = ret_map.reshape(2,512,512)

# Smooth averaged map
ret_map_sm = np.append(
ndimage.gaussian_filter(ndimage.median_filter(ret_map[0,...],10), 
                        ret_sigma, mode = 'reflect')[None,...],
ndimage.gaussian_filter(ndimage.median_filter(ret_map[1,...],10), 
                        ret_sigma, mode = 'reflect')[None,...],
axis = 0)

# Use gradient maps of elevation to find border of V1
sx_elv = ndimage.sobel(ndimage.gaussian_filter(ret_map[0,...],map_sigma), 
                       axis = 0, mode = 'reflect')
sy_elv = ndimage.sobel(ndimage.gaussian_filter(ret_map[0,...],map_sigma), 
                       axis = 1, mode = 'reflect')
ang_elv = np.arctan2(sy_elv,sx_elv)

sx_azi = ndimage.sobel(ndimage.gaussian_filter(ret_map[1,...],map_sigma), 
                       axis = 0, mode = 'reflect')
sy_azi = ndimage.sobel(ndimage.gaussian_filter(ret_map[1,...],map_sigma), 
                       axis = 1, mode = 'reflect')
ang_azi = np.arctan2(sy_azi,sx_azi)

# Threshold, i.e. sign map > 0 for V1
ang_diff = ang_elv
# ang_diff = np.sin(ang_elv - ang_azi)

scale = 0.5

ang_th = np.zeros_like(ang_elv)
# ang_th[ang_diff > scale*np.std(ang_diff)] = 1
# ang_th[ang_diff < -scale*np.std(ang_diff)] = -1


ang_th[ang_diff > 0] = 1
ang_th[ang_diff <=0] = -1


vmin_elv = np.percentile(ret_map_sm[0,...].flatten(), 5)
vmax_elv = np.percentile(ret_map_sm[0,...].flatten(), 95)

vmin_azi = np.percentile(ret_map_sm[1,...].flatten(), 5)
vmax_azi = np.percentile(ret_map_sm[1,...].flatten(), 95)

f,axes = plt.subplots(1,4,figsize = (20,8))

sns.heatmap(ret_map_sm[0,...], vmin = vmin_elv, vmax = vmax_elv,
            square = True, cbar = False, rasterized = True, cmap = 'hsv',
            ax = axes[1])
axes[1].tick_params(left = False, bottom = False)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title('Elevation')

sns.heatmap(ret_map_sm[1,...], vmin = vmin_azi, vmax = vmax_azi,
            square = True, cbar = False, rasterized = True, cmap = 'hsv',
            ax = axes[0])
axes[0].tick_params(left = False, bottom = False)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title('Azimuth')

sns.heatmap(ang_diff, center = 0, square = True, cbar = False, rasterized = True, 
            cmap = 'RdBu_r', ax = axes[2])
axes[2].tick_params(left = False, bottom = False)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].set_title('Angle of elevation gradient')

sns.heatmap(ang_th, center = 0, square = True, cbar = False, rasterized = True, 
            cmap = 'RdBu_r', ax = axes[3])
axes[3].tick_params(left = False, bottom = False)
axes[3].set_xticks([])
axes[3].set_yticks([])
axes[3].set_title('V1 mask')
  

f.tight_layout()