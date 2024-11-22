#%%
"""
Created on Thu Oct  1 13:35:14 2020

Generate single cell tuning plots and trial population analysis figures

@author: Samuel Failor
"""

from os.path import join
import glob
import numpy as np
from scipy import interpolate
from scipy.stats import kurtosis, skew, ttest_ind, ttest_rel
import scipy.optimize as so
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import cycler
import seaborn as sns
import seaborn.objects as sob
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import sklearn.model_selection as skms
from sklearn.metrics.pairwise import cosine_similarity
# import oripy as op

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# Add windows fonts
font_dirs = [r'C:\Windows\Fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

computer = 'work'

if computer == 'work':
    results_dir = r'C:\Users\samue\OneDrive - University College London\Results'
    fig_save_dir = r'C:\Users\samue\OneDrive - University College London\Orthogonalization project\Figures\Draft'
else:
    results_dir = 'C:/Users/Samuel/OneDrive - University College London/Results'
    fig_save_dir = r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft'


# subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
#             'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
#             'SF180613','SF190319','FR187']
# subjects = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B', 'SF171107',
#             'SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613','SF190319', 'FR187']
# expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
#               '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
#               '2018-06-28', '2018-12-12', '2019-05-21','2021-07-02']

# expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5, 4,7]

# trained = ['Naive', 'Proficient', 'Naive', 'Proficient', 'Naive', 'Proficient', 'Naive', 'Proficient', 'Naive',
#            'Proficient', 'Passive', 'Passive']


subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF',
                 'M171107_SF', 'SF171107', 'SF180515', 'SF180515', 'SF180613',
                 'SF180613']
subjects = ['SF170620B', 'SF170620B', 'SF170905B', 'SF170905B', 'SF171107',
            'SF171107', 'SF180515', 'SF180515', 'SF180613', 'SF180613']
expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26',
              '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21',
              '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]

trained = ['Naive', 'Proficient', 'Naive', 'Proficient', 'Naive', 'Proficient', 'Naive', 'Proficient', 'Naive',
           'Proficient']

# subjects_file = ['SF190613']
# subjects = ['SF190613']
# expt_dates = ['2019-08-05']
# expt_nums = [1]
# trained = ['Proficient']

# subjects_file = ['SF180816','SF180816']
# subjects = ['SF180816','SF180816']
# expt_dates = ['2018-09-17','2018-09-25']
# expt_nums = [1,4]
# trained = [False,True]

file_paths = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), r'_'.join([subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), 'orientation tuning_norm by pref ori_*'])))[0]
                                            for i in range(len(subjects_file))]

# %% Load data

single_cell_metrics = ['cell_plane','pref_ori_all', 'pref_ori_train', 'r','th',
                        'pref_ori_test','v_x', 'v_y','r_all','th_all','v_x_all',
                        'v_y_all', 'v_x_test','v_y_test','th_test','r_test',
                        'r_dir', 'th_dir', 'v_x_dir', 'v_y_dir',
                        'r_dir_all', 'th_dir_all', 'v_x_dir_all', 'v_y_dir_all',
                        'ROI_task_ret', 'V1_ROIs']

# single_cell_metrics = ['cell_plane','pref_ori_all', 'pref_ori_train', 'r','th',
#                         'pref_ori_test','v_x', 'v_y','r_all','th_all','v_x_all',
#                         'v_y_all', 'v_x_test','v_y_test','th_test','r_test',
#                         'r_dir', 'th_dir', 'v_x_dir', 'v_y_dir',
#                         'r_dir_all', 'th_dir_all', 'v_x_dir_all', 'v_y_dir_all']

ori_metrics = ['mean_ori_train','mean_ori_test', 'mean_ori_all']


# df_scm = pd.DataFrame(columns = single_cell_metrics
#                       + ['subject'] + ['trained'])

ori_dict = {o : np.zeros((9,0)) for o in ori_metrics}
ori_dict['subject'] = None
ori_dict['trained'] = None

for i,f in enumerate(file_paths):

    print('Loading ' + f)
    expt = np.load(f, allow_pickle = True)[()]

    cell_skew = skew(expt['spks'], axis = 1)

    # g_cells = expt['V1_ROIs'] == 1
    g_cells = np.logical_and(expt['cell_plane'] > 0, expt['V1_ROIs'] == 1)
    # g_cells = expt['cell_plane'] > 0
    # g_cells = np.logical_and(expt['cell_plane'] > 0, cell_skew > 3)
    # g_cells = np.ones(len(expt['cell_plane'])).astype(bool)

    print(g_cells.sum())

    stim_ori = np.repeat(expt['stim_ori'].reshape(-1,1),
                     expt['trial_resps'][:,g_cells].shape[1], axis = 1)
    stim_dir = np.repeat(expt['stim_dir'].reshape(-1,1),
                     expt['trial_resps'][:,g_cells].shape[1], axis = 1)
    trial_num = np.repeat(
        np.arange(expt['trial_resps'][:,g_cells].shape[0]).reshape(-1,1),
                           expt['trial_resps'][:,g_cells].shape[1], axis = 1)
    train_ind = np.isin(trial_num,expt['train_ind'])
    if i == 0:
        trials_dict = {'trial_resps_raw' : expt['trial_resps_raw'][:,g_cells].flatten(),
                       'trial_resps' : expt['trial_resps'][:,g_cells].flatten(),
                       'stim_ori' : stim_ori.flatten(),
                       'stim_dir' : stim_dir.flatten(),
                       'subject' : np.repeat(subjects[i],
                                          expt['trial_resps'][:,g_cells].size),
                       'trained' : np.repeat(trained[i],
                                          expt['trial_resps'][:,g_cells].size),
                       'cell' : np.repeat(np.arange(g_cells.sum()).reshape(1,-1),
                                          expt['trial_resps'].shape[0],axis=0).flatten(),
                       'task_ret' : np.repeat(expt['ROI_task_ret'][g_cells].reshape(1,-1),
                                               stim_ori.shape[0],axis=0).flatten(),
                       'V1_ROIs': np.repeat(expt['V1_ROIs'][g_cells].reshape(1,-1), stim_ori.shape[0], axis=0).flatten(),
                       'trial_num' : trial_num.flatten(),
                       'r' : np.repeat(expt['r'][g_cells].reshape(1,-1),
                                       stim_ori.shape[0],axis = 0).flatten(),
                       'pref' : np.repeat(expt['th'][g_cells].reshape(1,-1),
                                          stim_ori.shape[0],axis = 0).flatten(),
                       'train_ind' : train_ind.flatten()}
    else:
        cell_nums = np.arange(trials_dict['cell'].max()+1,
                              trials_dict['cell'].max()+g_cells.sum()+1)
        cell_nums = np.repeat(cell_nums.reshape(1,-1),
                              expt['trial_resps'].shape[0], axis = 0)
        # print(str(cell_nums.min()))
        trials_dict['trial_resps_raw'] = np.concatenate(
            (trials_dict['trial_resps_raw'], expt['trial_resps_raw'][:,g_cells].flatten()))
        trials_dict['trial_resps'] = np.concatenate(
            (trials_dict['trial_resps'], expt['trial_resps'][:,g_cells].flatten()))
        trials_dict['subject'] = np.concatenate((trials_dict['subject'],
                                 np.repeat(subjects[i],
                                 expt['trial_resps'][:,g_cells].size)))
        trials_dict['trained'] = np.concatenate((trials_dict['trained'],
                                 np.repeat(trained[i],
                                 expt['trial_resps'][:,g_cells].size)))
        trials_dict['stim_ori'] = np.concatenate((trials_dict['stim_ori'],stim_ori.flatten()))
        trials_dict['stim_dir'] = np.concatenate((trials_dict['stim_dir'],stim_dir.flatten()))
        trials_dict['cell'] = np.concatenate((trials_dict['cell'],cell_nums.flatten()))
        trials_dict['task_ret'] = np.concatenate((trials_dict['task_ret'],np.repeat(
            expt['ROI_task_ret'][g_cells].reshape(1,-1),stim_ori.shape[0],axis=0).flatten()))
        trials_dict['V1_ROIs'] = np.concatenate((trials_dict['V1_ROIs'],np.repeat(
            expt['V1_ROIs'][g_cells].reshape(1,-1),stim_ori.shape[0],axis=0).flatten()))
        trials_dict['trial_num'] = np.concatenate((trials_dict['trial_num'],
                                                   trial_num.flatten()))
        trials_dict['r'] = np.concatenate((trials_dict['r'],np.repeat(
            expt['r'][g_cells].reshape(1,-1),stim_ori.shape[0],axis=0).flatten()))
        trials_dict['pref'] = np.concatenate((trials_dict['pref'],np.repeat(
            expt['th'][g_cells].reshape(1,-1),stim_ori.shape[0],axis=0).flatten()))
        trials_dict['train_ind'] = np.concatenate((trials_dict['train_ind'],
                                                   train_ind.flatten()))

    # Use circular measures of direction preference

    expt['stim_dir_train'] = expt['stim_dir'][expt['train_ind']]

    nb = expt['stim_ori_train'] != np.inf

    x = np.sum(np.cos(expt['stim_dir_train'][nb,None]*np.pi/180.)*expt['trial_resps_train'][nb,:],0) \
        / np.sum(expt['trial_resps_train'][nb,:],0)

    y = np.sum(np.sin(expt['stim_dir_train'][nb,None]*np.pi/180.)*expt['trial_resps_train'][nb,:],0) \
        / np.sum(expt['trial_resps_train'][nb,:],0)

    expt['r_dir'] = np.sqrt(x**2 + y**2)
    expt['th_dir'] = np.arctan2(y, x)*180/np.pi-11.25
    expt['v_x_dir'] = x
    expt['v_y_dir'] = y

    nb = expt['stim_ori'] != np.inf

    x = np.sum(np.cos(expt['stim_dir'][nb,None]*np.pi/180.)*expt['trial_resps'][nb,:],0) \
        / np.sum(expt['trial_resps'][nb,:],0)

    y = np.sum(np.sin(expt['stim_dir'][nb,None]*np.pi/180.)*expt['trial_resps'][nb,:],0) \
        / np.sum(expt['trial_resps'][nb,:],0)

    expt['r_dir_all'] = np.sqrt(x**2 + y**2)
    expt['th_dir_all'] = np.arctan2(y, x)*180/np.pi-11.25
    expt['v_x_dir_all'] = x
    expt['v_y_dir_all'] = y


    # Save single cell metrics and append to dataframe
    single_cell_values = {s : expt[s][g_cells,None] for s in single_cell_metrics}

    for s in single_cell_metrics:
        if type(single_cell_values[s]) == np.ndarray:
            single_cell_values[s] = single_cell_values[s].reshape(-1,)
        else:
            single_cell_values[s] = np.array(single_cell_values[s]).reshape(-1,)

    single_cell_values['subject'] = np.repeat(subjects[i],
                                    len(expt['cell_plane'][g_cells]))
    single_cell_values['trained'] = np.repeat(trained[i],
                                    len(expt['cell_plane'][g_cells]))

    # This already only includes good cells
    # single_cell_values['rel_cont_model'] = expt['rel_cont_model']

    if i == 0:
        df_scm = pd.DataFrame(single_cell_values)
    else:
        df_scm = pd.concat([df_scm,pd.DataFrame(single_cell_values)],ignore_index=True)
    # df_scm = df_scm.append(pd.DataFrame(single_cell_values),
    #                        ignore_index = True)

    uni_stim = np.unique(expt['stim_dir'])
    uni_stim = uni_stim[:-1]

    z_stim = expt['z_stim'][:,g_cells]

    z_stim_ori = np.concatenate([z_stim[uni_stim % 180 == o,:].mean(0)[None,:]
                  for o in np.unique(uni_stim%180)], axis = 0)

    for o in ori_metrics:
        ori_dict[o] = np.concatenate([ori_dict[o], expt[o][:,g_cells]], axis = 1)

    if i == 0:
        ori_dict['subject'] = np.repeat(subjects[i], sum(g_cells))
        ori_dict['trained'] = np.repeat(trained[i], sum(g_cells))
        ori_dict['z_stim_ori'] = z_stim_ori
    else:
        ori_dict['subject'] = np.concatenate([ori_dict['subject'],
                                          np.repeat(subjects[i],
                                             sum(g_cells))])
        ori_dict['trained'] = np.concatenate([ori_dict['trained'],
                                          np.repeat(trained[i],
                                             sum(g_cells))])
        ori_dict['z_stim_ori'] = np.concatenate([ori_dict['z_stim_ori'],
                                                 z_stim_ori], axis = 1)

# %% Bin cells by th and r

df_scm['pref_bin'] = pd.cut(df_scm.th, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_scm['r_bin'] = pd.cut(df_scm.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

df_scm['pref_bin_all'] = pd.cut(df_scm.th_all, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_scm['r_bin_all'] = pd.cut(df_scm.r_all, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

df_scm['pref_dir_bin'] = pd.cut(df_scm.th, np.linspace(-11.25,360-11.25,17),
                        labels = np.ceil(np.arange(0,360,22.5)))
df_scm['r_dir_bin'] = pd.cut(df_scm.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

df_scm['pref_dir_bin_all'] = pd.cut(df_scm.th_all, np.linspace(-11.25,360-11.25,17),
                        labels = np.ceil(np.arange(0,360,22.5)))
df_scm['r_dir_bin_all'] = pd.cut(df_scm.r_all, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

df_scm['task_ret_bin'] = pd.cut(df_scm.ROI_task_ret, np.linspace(0,60,5))

# trials_dict['task_ret_bin'] = np.digitize(trials_dict['task_ret'],np.arange(0,40,5))

trained_cp = sns.color_palette('colorblind')[0:2]


#%% Some plotting functions

def sig_label(ax,p,h1,h2,label,fontsize=7,color='k',):


    # Get range of y
    r = np.diff(ax.get_ylim())

    h1 = r*h1
    h2 = r*h2


    y = max([ax.get_children()[p[0]].get_offsets()[:,1].max(),
                        ax.get_children()[p[1]].get_offsets()[:,1].max()])

    y1 = y + h1
    y2 = y1 + h2

    ax.plot([p[0],p[0],p[1],p[1]],[y1,y2,y2,y1], color = 'k')

    ax.text((sum(p))*.5, y2, label, ha='center', va='bottom', color = color,
            fontsize = fontsize)


    return ax

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

#%% Bullseye plots

def bull_plot(cell_x, cell_y, s0, s1, s_mult=10, s_0=1, label=None, c0=[0,1,0], c1=[1,0,1],
              contrast=1, brightness=0, rasterized=True, edgecolor=None, linewidth=0.025):
    ''' bulls-eye viewer for cell activity'''

    # s = np.abs(s0-s1)
    s = np.max(np.array([s0,s1]),0)
    # s = gmean(np.array([s0,s1])+0.01,axis = 0)
    # s = np.ones_like(s0)
    # o = np.argsort(np.abs(s))
    o = np.argsort(np.abs(s0-s1))[::-1]
    # o = np.arange(len(s))
    sz = np.abs(s[o])*s_mult + s_0
    s0_c = np.repeat(np.array(c0)[None,:],len(s0),axis = 0)
    s1_c = np.repeat(np.array(c1)[None,:],len(s1),axis = 0)

    colors = np.ones((len(s0),3))
    colors -= s0[o][:,None] * s1_c + s1[o][:,None] * s0_c
    # colors = np.zeros((len(s0),3))
    # colors += ((s0[o][:,None] * s1_c + s1[o][:,None] * s0_c))
    colors = np.concatenate([colors,s[:,None]],axis = 1)
    # Adjust contrast and brightness
    colors = (colors * contrast) + brightness
    # Ensure values fall between 0 and 1
    colors[colors<0] = 0
    colors[colors>1] = 1

    # breakpoint()
    plt.scatter(x = cell_x[o],y = cell_y[o], s=sz, c=colors,
                label = label, rasterized = rasterized, edgecolor = edgecolor, linewidth = linewidth)
    plt.gca().set_facecolor('w')
    plt.axis([-1,1,-1,1])
    plt.rc('path',simplify=False)



def bull_plot_simple(cell_x, cell_y, s0, s1, c, s_mult=10, s_0=0.1, label=None,
              contrast=1, brightness=0, rasterized=True, edgecolor='black', linewidth=0.025):
    ''' bulls-eye viewer for cell activity'''

    # s = np.abs(s0-s1)
    s = np.max(np.array([s0,s1]),0)
    # s = gmean(np.array([s0,s1])+0.01,axis = 0)
    # s = np.ones_like(s0)
    # o = np.argsort(np.abs(s))
    o = np.argsort(np.abs(s0-s1))[::-1]
    # o = np.arange(len(s))
    sz = np.abs(s[o])*s_mult + s_0
    c = [c[i] for i in o]

    # breakpoint()
    plt.scatter(x=cell_x[o], y=cell_y[o], s=sz, c=c,
                label=label, rasterized=rasterized, edgecolor=edgecolor, linewidth=linewidth)
    plt.gca().set_facecolor('w')
    plt.axis([-1,1,-1,1])
    plt.rc('path',simplify=False)



def bull_plot_mdl(cell_x,cell_y,s0,s1,mdl_cont,s_mult=2,s_0 = 0.1,label = None,
                  rasterized = True,edgecolor = 'black',linewidth = 0.025,
                  vmin = 0,vmax = 1):
    ''' bulls-eye viewer for cell activity'''

    # s = np.abs(s0-s1)
    s = np.max(np.array([s0,s1]),0)
    # s = gmean(np.array([s0,s1])+0.01,axis = 0)
    # s = np.ones_like(s0)
    # o = np.argsort(np.abs(s))
    o = np.argsort(mdl_cont)
    # o = np.arange(len(s))
    sz = np.abs(s[o])*s_mult + s_0

    # breakpoint()
    sns.scatterplot(x = cell_x[o],y = cell_y[o], s=sz, hue = mdl_cont[o],
                    label = label, rasterized = rasterized, edgecolor = edgecolor,
                    linewidth = linewidth, legend = False,  hue_norm = (vmin,vmax),
                    palette = 'magma_r')
    plt.gca().set_facecolor('w')
    plt.axis([-1,1,-1,1])
    plt.rc('path',simplify=False)


#%% Bullseye plots

trained_cond = df_scm['trained'].to_numpy()

cell_x = df_scm['v_x'].to_numpy()
cell_y = df_scm['v_y'].to_numpy()

tuning_curves = ori_dict['mean_ori_test'][:-1,:]

trained_cp = sns.color_palette('colorblind')[0:2]

# c_45 = sns.color_palette('colorblind')[2]
# c_90 = sns.color_palette('colorblind')[6]

# c_45 = (0.1,0.5,0.2)
# c_90 = (0.9,0.5,0.8)

# c_45 = (1,0,0)
# c_90 = (0,1,1)

c_45 = (0,1,0)
c_90 = (1,0,1)

# cp_45 = sns.light_palette(c_45,100,as_cmap = True)
# cp_90 = sns.light_palette(c_90,100,as_cmap = True)

cp = sns.diverging_palette(318, 163, n=100, as_cmap = True)

trained_cells = np.where(trained_cond=='Proficient')[0]
naive_cells = np.where(trained_cond=='Naive')[0]

# trained_blocks = np.random.choice(trained_cells,
#                             (np.floor(1000).astype(int),10),
#                             replace = False)
# naive_blocks = np.random.choice(naive_cells,
#                             (np.floor(1000).astype(int),10),
#                             replace = False)

trained_blocks = np.random.choice(trained_cells, 16000, replace = False)
naive_blocks = np.random.choice(naive_cells, 16000, replace = False)

vmin = 0
vmax = 1
s_mult = 0.5
s_min = 0.1


color_fun = np.concatenate((np.ones((3,1)),-np.array(c_45)[:,None],
                            -np.array(c_90)[:,None]),axis=1)

n_pts = 1000
x_c, y_c = np.meshgrid(np.linspace(0,1,n_pts),np.linspace(0,1,n_pts))

x_y = np.concatenate((x_c.reshape(1,-1),y_c.reshape(1,-1)),axis=0)
x_y = np.concatenate((np.ones((1,x_y.shape[1])), x_y), axis = 0)

cm_legend = np.flip(np.dot(color_fun,x_y).T.reshape(n_pts,n_pts,3),axis=0)
# plt.figure('color legend', figsize=(10,10))
# plt.imshow(cm_legend)
# plt.gca().get_xaxis().set_ticks([])
# plt.gca().axes.get_yaxis().set_ticks([])


# colors = sns.color_palette('Spectral', n_colors = 1001)
# colors = sns.diverging_palette(120,300, center='dark', n=1001)
colors = sns.diverging_palette(250, 30, l=65, n=1001, center="dark")

from matplotlib.colors import LinearSegmentedColormap

# Define the colors and their positions in the colormap
cm_colors = [(0.0, 'blue'), (0.5, 'magenta'), (1.0, 'red')]

# Create the colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", cm_colors)

n_colors = 1001  # You can adjust the number of colors as needed
colors = [custom_cmap(i / (n_colors - 1)) for i in range(n_colors)]

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1,
                          rc = {'lines.linewidth':0.5}):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['figure.dpi'] = 1000

    fontsize = 5

    f = plt.figure(figsize=(3.75,3))
    # ax_prof0 = f.add_subplot(1,2,2, aspect = 'equal')
    gs = gridspec.GridSpec(1,2, figure = f)
    gs.update(wspace=-0.03)

    ax_naive0 = f.add_subplot(gs[0],aspect = 'equal')
    ax_naive0.set_facecolor('None')

    cells = naive_blocks

    resps_45 = tuning_curves[2,cells]
    resps_90 = tuning_curves[4,cells]

    resps_diff = resps_45-resps_90
    resps_diff = np.clip(resps_diff, -1, 1)
    resps_diff += 1
    resps_diff /= 2
    resps_diff *= 1000
    resps_diff = np.round(resps_diff)

    c = [colors[i] for i in resps_diff.astype(int)]

    # bull_plot_simple(cell_x[cells], cell_y[cells], tuning_curves[2,cells], tuning_curves[4,cells],
    #              s_mult = s_mult, c = c)
    bull_plot(cell_x[cells], cell_y[cells], tuning_curves[2,cells], tuning_curves[4,cells], s_mult=s_mult, s_0=s_min,
              c0=c_45, c1=c_90)
    # plt.grid()

    ax_naive0.set_xlim([-1,1])
    ax_naive0.set_ylim([-1,1])

    th=np.arange(0,1.0001,.0001)*2*np.pi

    # draw circles
    for rd in [0.16, 0.32, 0.48, 0.64]:
        ax_naive0.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)


    # for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

    #     p = [np.cos(a)*0.64, np.sin(a)*0.64]

    #     if int(np.cos(a)) == 0:
    #         x = [p[0], p[0]]
    #         y = [p[1]-0.08, p[1]+0.08]
    #         ax_naive0.plot(x,y,'k')
    #     else:
    #         y = [p[1], p[1]]
    #         x = [p[0]-0.08, p[0]+0.08]
    #         ax_naive0.plot(x,y,'k')


    # ang = np.deg2rad(315)
    # ax_naive0.text(np.cos(ang)*0.75, np.sin(ang)*0.75, 'Selectivity',rotation = 45,
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # for r in [0.16, 0.32, 0.48, 0.64]:
    #     ax_naive0.text(np.cos(ang)*(r+0.02), np.sin(ang)*(r+0.02), r, rotation = 45,
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

    #     if i == 0:
    #         label = 'Mean ori. pref.\n' + str(int(np.rad2deg(a)/2)) + r'$\degree$'
    #     else:
    #         label = str(int(np.rad2deg(a)/2)) + r'$\degree$'

    #     if i == 0:
    #         ha = 'center'
    #         va = 'center'
    #     elif i == 1:
    #         ha = 'center'
    #         va = 'top'
    #     elif i == 2:
    #         ha = 'center'
    #         va = 'center'
    #     elif i == 3:
    #         ha = 'center'
    #         va = 'bottom'

    #     # va = 'center'
    #     # ha = 'center'

    #     ax_naive0.text(np.cos(a)*0.92, np.sin(a)*0.92, label,
    #               horizontalalignment=ha, verticalalignment = va,
    #               fontsize = fontsize)


    plt.title('Naïve', color = trained_cp[0], fontsize = 6, pad = 1)
    plt.tick_params(axis='both', left=False, top=False, right=False,
                        bottom=False, labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)




    ax_prof0 = f.add_subplot(gs[1],aspect = 'equal')
    ax_prof0.set_facecolor('None')

    cells = trained_blocks

    resps_45 = tuning_curves[2,cells]
    resps_90 = tuning_curves[4,cells]

    resps_diff = resps_45-resps_90
    resps_diff = np.clip(resps_diff, -1, 1)
    resps_diff += 1
    resps_diff /= 2
    resps_diff *= 1000
    resps_diff = np.round(resps_diff)

    c = [colors[i] for i in resps_diff.astype(int)]

    # bull_plot_simple(cell_x[cells], cell_y[cells], tuning_curves[2,cells], tuning_curves[4,cells],
    #              s_mult = s_mult, c = c)
    # plt.grid()

    bull_plot(cell_x[cells], cell_y[cells], tuning_curves[2,cells], tuning_curves[4,cells],
                  s_mult=s_mult, s_0=s_min, c0=c_45, c1=c_90)
    # plt.grid()

    ax_prof0.set_xlim([-1,1])
    ax_prof0.set_ylim([-1,1])

    # draw circles

    for rd in [0.16, 0.32, 0.48, 0.64]:
        plt.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)


    # for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

    #     p = [np.cos(a)*0.64, np.sin(a)*0.64]

    #     if int(np.cos(a)) == 0:
    #         x = [p[0], p[0]]
    #         y = [p[1]-0.08, p[1]+0.08]
    #         ax_prof0.plot(x,y,'k')
    #     else:
    #         y = [p[1], p[1]]
    #         x = [p[0]-0.08, p[0]+0.08]
    #         ax_prof0.plot(x,y,'k')


    # Plot vector
    # ang = np.deg2rad(45)
    # ax_naive0.plot([0,0.8],[0,0],'k',linewidth = 0.5)
    # ax_naive0.quiver(0,0, np.cos(ang)*0.8, np.sin(ang)*0.8,
    #                 color ='black', angles='xy', scale_units='xy', scale=1)

    # ang = np.deg2rad(17)
    # ax_naive0.text(np.cos(ang)*0.5,np.sin(ang)*0.5, r'$\theta_{pref.}$', fontsize = fontsize,
    # horizontalalignment = 'center', verticalalignment = 'center')

    ang = np.deg2rad(315)
    ax_naive0.text(np.cos(ang)*0.75, np.sin(ang)*0.75, 'Selectivity',rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')


    for r in [0.16, 0.32, 0.48, 0.64]:
        ax_naive0.text(np.cos(ang)*(r+0.01), np.sin(ang)*(r+0.01), r, rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')



    h_dist = 0.85
    v_dist = 0.85

    for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

        # if i == 0:
        #     label = 'Mean ori. pref.\n' + str(int(np.rad2deg(a)/2)) + r'$\degree$'
        # else:
        #     label = str(int(np.rad2deg(a)/2)) + r'$\degree$'

        label = str(int(np.rad2deg(a)/2)) + r'$\degree$'
        if i == 0:
            ha = 'left'
            va = 'center'
            d = h_dist
        elif i == 1:
            ha = 'center'
            va = 'bottom'
            d = v_dist
        elif i == 2:
            ha = 'right'
            va = 'center'
            d = h_dist
        elif i == 3:
            ha = 'center'
            va = 'top'
            d = v_dist
        # va = 'center'
        # ha = 'center'

        ax_naive0.text(np.cos(a)*d, np.sin(a)*d, label,
                  horizontalalignment=ha, verticalalignment = va,
                  fontsize = fontsize)

    # plt.scatter([0.65,0.65],[-0.81,-0.65],s = [s_max,s_min], c = 'k')
    # plt.text(0.7,-0.814, 'Max resp.',
    #           horizontalalignment='left', verticalalignment='center')
    # plt.text(0.7,-0.654, 'Min resp.',
    #           horizontalalignment='left', verticalalignment='center')
    # plt.text(0.8,1, '45', color = c_45)
    # plt.text(0.8,0.89, '90', color = c_90)

    plt.title('Proficient', color = trained_cp[1], fontsize = 6, pad = 1)
    plt.tick_params(axis='both', left=False, top=False, right=False,
                    bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)

    for s in ax_prof0.spines.keys():
        ax_prof0.spines[s].set_visible(False)

    for s in ax_naive0.spines.keys():
        ax_naive0.spines[s].set_visible(False)


    f.tight_layout()

    # savefile = join(fig_save_dir,'bullseye_plots.svg')
    plt.savefig(savefile, format = 'svg')


#%% Bullseye plots - model contribution

trained_cond = df_scm['trained'].to_numpy()

cell_x = df_scm['v_x'].to_numpy()
cell_y = df_scm['v_y'].to_numpy()

tuning_curves = ori_dict['mean_ori_test'][:-1,:]
mdl_cont = df_scm.rel_cont_model.to_numpy()

trained_cells = np.where(trained_cond=='Proficient')[0]
naive_cells = np.where(trained_cond=='Naive')[0]

# trained_blocks = np.random.choice(trained_cells,
#                             (np.floor(1000).astype(int),10),
#                             replace = False)
# naive_blocks = np.random.choice(naive_cells,
#                             (np.floor(1000).astype(int),10),
#                             replace = False)

trained_blocks = np.random.choice(trained_cells, 16000, replace = False)
naive_blocks = np.random.choice(naive_cells, 16000, replace = False)

vmin = 0
vmax = 1
s_mult = 1

smin = 0.001


color_fun = np.concatenate((np.ones((3,1)),-np.array(c_45)[:,None],
                            -np.array(c_90)[:,None]),axis=1)

n_pts = 1000
x_c, y_c = np.meshgrid(np.linspace(0,1,n_pts),np.linspace(0,1,n_pts))

x_y = np.concatenate((x_c.reshape(1,-1),y_c.reshape(1,-1)),axis=0)
x_y = np.concatenate((np.ones((1,x_y.shape[1])), x_y), axis = 0)

cm_legend = np.flip(np.dot(color_fun,x_y).T.reshape(n_pts,n_pts,3),axis=0)
# plt.figure('color legend', figsize=(10,10))
# plt.imshow(cm_legend)
# plt.gca().get_xaxis().set_ticks([])
# plt.gca().axes.get_yaxis().set_ticks([])

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1,
                          rc = {'lines.linewidth':0.5}):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['figure.dpi'] = 1000

    fontsize = 5

    f = plt.figure(figsize=(3.75,3))
    # ax_prof0 = f.add_subplot(1,2,2, aspect = 'equal')
    gs = gridspec.GridSpec(1,2, figure = f)
    gs.update(wspace=-0.03)

    ax_naive0 = f.add_subplot(gs[0],aspect = 'equal')
    ax_naive0.set_facecolor('None')

    c = naive_blocks
    bull_plot_mdl(cell_x[c], cell_y[c], tuning_curves[2,c], tuning_curves[4,c],
                  mdl_cont[c], s_mult = s_mult, s_0 = smin,
                  vmin = vmin, vmax = vmax)
    # plt.grid()

    ax_naive0.set_xlim([-1,1])
    ax_naive0.set_ylim([-1,1])

    th=np.arange(0,1.0001,.0001)*2*np.pi

    # draw circles
    for rd in [0.16, 0.32, 0.48, 0.64]:
        ax_naive0.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)

    # for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

    #     p = [np.cos(a)*0.64, np.sin(a)*0.64]

    #     if int(np.cos(a)) == 0:
    #         x = [p[0], p[0]]
    #         y = [p[1]-0.08, p[1]+0.08]
    #         ax_naive0.plot(x,y,'k')
    #     else:
    #         y = [p[1], p[1]]
    #         x = [p[0]-0.08, p[0]+0.08]
    #         ax_naive0.plot(x,y,'k')

    # ang = np.deg2rad(315)
    # ax_naive0.text(np.cos(ang)*0.75, np.sin(ang)*0.75, 'Selectivity',rotation = 45,
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # for r in [0.16, 0.32, 0.48, 0.64]:
    #     ax_naive0.text(np.cos(ang)*(r+0.02), np.sin(ang)*(r+0.02), r, rotation = 45,
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

    #     if i == 0:
    #         label = 'Mean ori. pref.\n' + str(int(np.rad2deg(a)/2)) + r'$\degree$'
    #     else:
    #         label = str(int(np.rad2deg(a)/2)) + r'$\degree$'

    #     if i == 0:
    #         ha = 'center'
    #         va = 'center'
    #     elif i == 1:
    #         ha = 'center'
    #         va = 'top'
    #     elif i == 2:
    #         ha = 'center'
    #         va = 'center'
    #     elif i == 3:
    #         ha = 'center'
    #         va = 'bottom'

    #     # va = 'center'
    #     # ha = 'center'

    #     ax_naive0.text(np.cos(a)*0.92, np.sin(a)*0.92, label,
    #               horizontalalignment=ha, verticalalignment = va,
    #               fontsize = fontsize)

    plt.title('Naïve', color = trained_cp[0], fontsize = 6, pad = 1)
    plt.tick_params(axis='both', left=False, top=False, right=False,
                        bottom=False, labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)

    ax_prof0 = f.add_subplot(gs[1],aspect = 'equal')
    ax_prof0.set_facecolor('None')
    c = trained_blocks
    bull_plot_mdl(cell_x[c], cell_y[c], tuning_curves[2,c], tuning_curves[4,c],
                  mdl_cont[c], s_mult = s_mult, s_0 = smin,
                  vmin = vmin, vmax = vmax)
    # plt.grid()

    ax_prof0.set_xlim([-1,1])
    ax_prof0.set_ylim([-1,1])

    # draw circles

    for rd in [0.16, 0.32, 0.48, 0.64]:
        plt.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)

    # for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

    #     p = [np.cos(a)*0.64, np.sin(a)*0.64]

    #     if int(np.cos(a)) == 0:
    #         x = [p[0], p[0]]
    #         y = [p[1]-0.08, p[1]+0.08]
    #         ax_prof0.plot(x,y,'k')
    #     else:
    #         y = [p[1], p[1]]
    #         x = [p[0]-0.08, p[0]+0.08]
    #         ax_prof0.plot(x,y,'k')


    # Plot vector
    # ang = np.deg2rad(45)
    # ax_naive0.plot([0,0.8],[0,0],'k',linewidth = 0.5)
    # ax_naive0.quiver(0,0, np.cos(ang)*0.8, np.sin(ang)*0.8,
    #                 color ='black', angles='xy', scale_units='xy', scale=1)

    # ang = np.deg2rad(17)
    # ax_naive0.text(np.cos(ang)*0.5,np.sin(ang)*0.5, r'$\theta_{pref.}$', fontsize = fontsize,
    # horizontalalignment = 'center', verticalalignment = 'center')

    ang = np.deg2rad(315)
    ax_naive0.text(np.cos(ang)*0.75, np.sin(ang)*0.75, 'Selectivity',rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')

    for r in [0.16, 0.32, 0.48, 0.64]:
        ax_naive0.text(np.cos(ang)*(r+0.01), np.sin(ang)*(r+0.01), r, rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')

    h_dist = 0.85
    v_dist = 0.85

    for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

        # if i == 0:
        #     label = 'Mean ori. pref.\n' + str(int(np.rad2deg(a)/2)) + r'$\degree$'
        # else:
        #     label = str(int(np.rad2deg(a)/2)) + r'$\degree$'

        label = str(int(np.rad2deg(a)/2)) + r'$\degree$'
        if i == 0:
            ha = 'left'
            va = 'center'
            d = h_dist
        elif i == 1:
            ha = 'center'
            va = 'bottom'
            d = v_dist
        elif i == 2:
            ha = 'right'
            va = 'center'
            d = h_dist
        elif i == 3:
            ha = 'center'
            va = 'top'
            d = v_dist
        # va = 'center'
        # ha = 'center'

        ax_naive0.text(np.cos(a)*d, np.sin(a)*d, label,
                  horizontalalignment=ha, verticalalignment = va,
                  fontsize = fontsize)

    # plt.scatter([0.65,0.65],[-0.81,-0.65],s = [s_max,s_min], c = 'k')
    # plt.text(0.7,-0.814, 'Max resp.',
    #           horizontalalignment='left', verticalalignment='center')
    # plt.text(0.7,-0.654, 'Min resp.',
    #           horizontalalignment='left', verticalalignment='center')
    # plt.text(0.8,1, '45', color = c_45)
    # plt.text(0.8,0.89, '90', color = c_90)

    plt.title('Proficient', color = trained_cp[1], fontsize = 6, pad = 1)
    plt.tick_params(axis='both', left=False, top=False, right=False,
                    bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)

    for s in ax_prof0.spines.keys():
        ax_prof0.spines[s].set_visible(False)

    for s in ax_naive0.spines.keys():
        ax_naive0.spines[s].set_visible(False)


    # f.tight_layout()

    # savefile = join(fig_save_dir,'bullseye_plots.svg')
    # plt.savefig(savefile, format = 'svg')

#%% Create two heatmaps showing naive and proficient neurons, sorted by orientation preference and selectivity.
# Pref and selectivity from training set, tuning curves from test set

df_plot = df_scm[['pref_bin','r','trained']].copy()

tuning_curves = np.hsplit(ori_dict['mean_ori_test'][:-1,:], ori_dict['mean_ori_test'][:-1,:].shape[1])

df_plot['tuning_curves'] = tuning_curves

trained_cond = df_plot['trained'].to_numpy()

trained_cells = np.where(trained_cond=='Proficient')[0]
naive_cells = np.where(trained_cond=='Naive')[0]

trained_blocks = np.random.choice(trained_cells, 1000, replace = False)
naive_blocks = np.random.choice(naive_cells, 1000, replace = False)

df_plot = df_plot.loc[np.hstack([trained_blocks,naive_blocks])]

df_plot = df_plot.sort_values(['pref_bin','r','trained'])


f,a = plt.subplots(1,2)

sns.heatmap(np.hstack(df_plot[df_plot.trained=='Naive'].tuning_curves.to_numpy()).T, ax = a[0],
            vmin = 0, vmax = 1.5, cmap = 'gray', cbar = False)
sns.heatmap(np.hstack(df_plot[df_plot.trained=='Proficient'].tuning_curves.to_numpy()).T, ax = a[1],
            vmin = 0, vmax = 1.5, cmap = 'gray', cbar = False)



#%% Orthogonalization cartoon

c_45 = (0,1,0)
c_90 = (1,0,1)

train_cond = sns.color_palette('colorblind')[:2]

width = 0.015
headwidth = 4
headlength = 5
headaxislength = 4

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,
                                        "axes.titlesize":5,
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
                                        "legend.title_fontsize":5,
                                        'mathtext.fontset' : 'stix',
                                        'mathtext.rm'      : 'serif',
                                        'font.family'      : 'serif',
                                        'font.serif'       : "Times", # or "Times"
                                        }):


    # mpl.style.use('classic')

    f,ax = plt.subplots(2,1,sharex=True, sharey = True, figsize = (1.6,2.2))
    # Axes
    ax[0].quiver(0,-0.00625, np.cos(np.pi/2), np.sin(np.pi/2),
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)
    ax[0].quiver(-0.00625,0, np.cos(0), np.sin(0),
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)

    # Axis dim labels
    ax[0].text(np.cos(np.pi/2), np.sin(np.pi/2), r'$f_1$', fontsize = 5, horizontalalignment='center',
    verticalalignment = 'bottom')
    ax[0].text(np.cos(0), np.sin(0), r'$f_2$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'center')

    # 45 and 90
    scale = 0.9
    ax[0].quiver(0,0, np.cos(np.deg2rad(65))*scale, np.sin(np.deg2rad(65))*scale,
                    color =c_90, angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength)
    ax[0].quiver(0,0, np.cos(np.deg2rad(55))*scale, np.sin(np.deg2rad(55))*scale,
                    color =c_45, angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength)

    # Draw arc between vectors
    x = np.linspace(np.deg2rad(65-0.5), np.deg2rad(55+0.5), 100)
    ax[0].plot(np.cos(x)*0.7,np.sin(x)*0.7,'-k', linewidth = 0.5, zorder = -1)
    ax[0].text(np.cos(np.deg2rad(58.5))*0.725, np.sin(np.deg2rad(58.5))*0.725, r'$\theta$', fontsize = 5,
    horizontalalignment = 'center')

    # Other dims
    ax[0].quiver(0,0, np.cos(np.pi/3.8)*scale, np.sin(np.pi/3.8)*scale,
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)
    ax[0].quiver(0,0, np.cos(np.pi/5)*scale, np.sin(np.pi/5)*scale,
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)
    ax[0].quiver(0,0, np.cos(np.pi/6)*scale, np.sin(np.pi/6)*scale,
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)

    # Labels
    # Axis dim labels
    ax[0].text(np.cos(np.pi/3.8)*scale, np.sin(np.pi/3.8)*scale, r'$f_N$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'bottom')
    ax[0].text(np.cos(np.pi/5)*scale, np.sin(np.pi/5)*scale, r'$f_4$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'bottom')
    ax[0].text(np.cos(np.pi/6.5)*scale, np.sin(np.pi/6.5)*scale, r'$f_3$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'bottom')
    ax[0].text(np.cos(np.deg2rad(41.68))*0.8, np.sin(np.deg2rad(41.68))*0.8, r'$\cdots$', rotation = 130.5, horizontalalignment = 'center',
    verticalalignment = 'center')

    ax[0].set_ylim([-0.031,1])
    ax[0].set_xlim([-0.031,1])
    sns.despine(left=True,bottom=True)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_aspect('equal')

    # ax[0].set_title('Naïve', color = train_cond[0], fontsize = 6)

    # Proficient

    ax[1].quiver(0,-0.00625, np.cos(np.pi/2), np.sin(np.pi/2),
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)
    ax[1].quiver(-0.00625,0, np.cos(0), np.sin(0),
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)

    # Axis dim labels
    ax[1].text(np.cos(np.pi/2), np.sin(np.pi/2), r'$f_1$', fontsize = 5, horizontalalignment='center',
    verticalalignment = 'bottom')
    ax[1].text(np.cos(0), np.sin(0), r'$f_2$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'center')

    # 45 and 90
    ax[1].quiver(0,0, np.cos(np.deg2rad(75))*scale, np.sin(np.deg2rad(75))*scale,
                    color =c_90, angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength)
    ax[1].quiver(0,0, np.cos(np.deg2rad(15))*scale, np.sin(np.deg2rad(15))*scale,
                    color =c_45, angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength)

    # Draw arc between vectors
    x = np.linspace(np.deg2rad(75-0.5), np.deg2rad(15+0.5), 100)
    ax[1].plot(np.cos(x)*0.4,np.sin(x)*0.4,'-k', linewidth = 0.5, zorder = -1)
    ax[1].text(np.cos(np.deg2rad(59))*0.425, np.sin(np.deg2rad(59))*0.425, r'$\theta$', fontsize = 5,
    horizontalalignment = 'center')

    # Other dims
    ax[1].quiver(0,0, np.cos(np.pi/3.8)*scale, np.sin(np.pi/3.8)*scale,
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)
    ax[1].quiver(0,0, np.cos(np.pi/5)*scale, np.sin(np.pi/5)*scale,
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)
    ax[1].quiver(0,0, np.cos(np.pi/6)*scale, np.sin(np.pi/6)*scale,
                    color ='black', angles='xy', scale_units='xy', scale=1, width = width,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                    zorder = 100)

    # Labels
    # Axis dim labels
    ax[1].text(np.cos(np.pi/3.8)*scale, np.sin(np.pi/3.8)*scale, r'$f_N$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'bottom')
    ax[1].text(np.cos(np.pi/5)*scale, np.sin(np.pi/5)*scale, r'$f_4$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'bottom')
    ax[1].text(np.cos(np.pi/6.5)*scale, np.sin(np.pi/6.5)*scale, r'$f_3$', fontsize = 5, horizontalalignment='left',
    verticalalignment = 'bottom')
    ax[1].text(np.cos(np.deg2rad(41.68))*0.8, np.sin(np.deg2rad(41.68))*0.8, r'$\cdots$', rotation = 130.5, horizontalalignment = 'center',
    verticalalignment = 'center')

    ax[1].set_ylim([-0.031,1])
    ax[1].set_xlim([-0.031,1])
    sns.despine(left=True,bottom=True)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_aspect('equal')


    # ax[1].set_title('Proficient', color = train_cond[1], fontsize = 6, horizontalalignment = 'center')


    f.tight_layout()
    # f.savefig(join(fig_save_dir,'orthog_cartoon.svg'), format = 'svg')

#%% Plot mean response by std response by condition

df_trials = pd.DataFrame(trials_dict)

df_trials = df_trials[~df_trials.train_ind]
df_trials = df_trials[df_trials.stim_dir != np.inf]
df_trials = df_trials[df_trials.trained != 'Passive']

df_trials['pref_bin'] = pd.cut(df_trials.pref, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_trials['r_bin'] = pd.cut(df_trials.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                      labels = np.arange(5))


df_trial_stats = pd.DataFrame(columns = ['mu_45','sigma_45','mu_90','sigma_90',
                                         'trained','subject','cell'])

df_trial_stats['mu_45'] = df_trials[df_trials.stim_ori==45].groupby(['cell'],as_index=False)['trial_resps'].mean().trial_resps
df_trial_stats['sigma_45'] = df_trials[df_trials.stim_ori==45].groupby(['cell'],as_index=False)['trial_resps'].std().trial_resps

df_trial_stats['var_45'] = df_trials[df_trials.stim_ori==45].groupby(['cell'],as_index=False)['trial_resps'].var().trial_resps

df_trial_stats['mu_90'] = df_trials[df_trials.stim_ori==90].groupby(['cell'],as_index=False)['trial_resps'].mean().trial_resps
df_trial_stats['sigma_90'] = df_trials[df_trials.stim_ori==90].groupby(['cell'],as_index=False)['trial_resps'].std().trial_resps

df_trial_stats['var_90'] = df_trials[df_trials.stim_ori==90].groupby(['cell'],as_index=False)['trial_resps'].var().trial_resps

df_trial_stats['pooled_sigma'] = np.sqrt((df_trial_stats.var_45 + df_trial_stats.var_90)/2)

df_trial_stats['mu_diff'] = df_trial_stats.mu_45 - df_trial_stats.mu_90
df_trial_stats['abs_mu_diff'] = df_trial_stats.mu_diff.abs()

df_trial_stats['trained'] = df_trials[df_trials.stim_ori==90].groupby(['cell'],as_index=False)['trained'].agg(pd.Series.mode).trained
df_trial_stats['subject'] = df_trials[df_trials.stim_ori==90].groupby(['cell'],as_index=False)['subject'].agg(pd.Series.mode).subject
df_trial_stats['r_bin'] = df_trials[df_trials.stim_ori==90].groupby(['cell'],as_index=False)['r_bin'].agg(pd.Series.mode).r_bin
df_trial_stats['pref_bin'] = df_trials[df_trials.stim_ori==90].groupby(['cell'],as_index=False)['pref_bin'].agg(pd.Series.mode).pref_bin.astype(int).astype(str)

df_trial_stats['weak_tuning'] = df_trial_stats.r_bin < 2
df_trial_stats['strong_tuning'] = df_trial_stats.r_bin >= 3


f,a = plt.subplots(2,2, sharex = True, sharey = True)
sns.scatterplot(data = df_trial_stats[df_trial_stats.weak_tuning], x = 'mu_45', y = 'sigma_45',
                ax = a[0,0], s = 1, hue = 'trained', legend = False, palette = 'colorblind')
sns.scatterplot(data = df_trial_stats[df_trial_stats.strong_tuning], x = 'mu_45', y = 'sigma_45',
                ax = a[0,1], s = 1, hue = 'trained', legend = False, palette = 'colorblind')
sns.scatterplot(data = df_trial_stats[df_trial_stats.weak_tuning], x = 'mu_90', y = 'sigma_90',
                ax = a[1,0], s = 1, hue = 'trained', legend = False, palette = 'colorblind')
sns.scatterplot(data = df_trial_stats[df_trial_stats.strong_tuning], x = 'mu_90', y = 'sigma_90',
                ax = a[1,1], s = 1, hue = 'trained', legend = False, palette = 'colorblind')


sns.displot(data = df_trial_stats, x = 'abs_mu_diff', y = 'pooled_sigma',
            row = 'trained',col = 'pref_bin',
            kind = 'hist', col_order = np.ceil(np.arange(0,180,22.5)).astype(int).astype(str),
            common_norm = False)

a = sns.relplot(data = df_trial_stats, x = 'abs_mu_diff', y = 'pooled_sigma',
            hue = 'trained', palette = 'colorblind', col = 'pref_bin',
            kind = 'scatter', size = 1, col_order = np.ceil(np.arange(0,180,22.5)).astype(int).astype(str),
            style = 'r_bin')

for a in a.axes.flatten():
    # a.plot([1,1],[0,5], '--k')
    # a.plot([0,2],[1,1], '--k')
    # a.plot([0,2],[0,0], '--k')
    a.grid()


#%% By cell type

df_trials = pd.DataFrame(trials_dict)

df_trials = df_trials[~df_trials.train_ind]
df_trials = df_trials[df_trials.stim_dir != np.inf]
df_trials = df_trials[df_trials.trained != 'Passive']


df_trials['pref_bin'] = pd.cut(df_trials.pref, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_trials['r_bin'] = pd.cut(df_trials.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                      labels = np.arange(5))

# bins = np.hstack([np.linspace(0,0.8,20),1])
# df_trials['r_bin'] = pd.cut(df_trials.r, bins,
#                      labels = np.arange(len(bins)-1))


df_trial_stats = pd.DataFrame(columns = ['mu_45','var_45','mu_90','var_90',
                                         'trained','subject', 'pooled_sigma',
                                         'sigma_45','sigma_90'])

df_trial_stats[['subject','trained','r_bin','pref_bin','mu_45']] = df_trials[df_trials.stim_ori==45].groupby(
    ['cell','subject','trained','r_bin','pref_bin'],as_index=False, observed = True)['trial_resps'].mean().groupby(
        ['subject','trained','r_bin','pref_bin'],as_index=False).mean().drop('cell',axis=1)


var_45 = df_trials[df_trials.stim_ori==45].groupby(
    ['cell','subject','trained','r_bin','pref_bin'],as_index=False, observed = True)['trial_resps'].var()
var_45 = var_45.rename(columns = {'trial_resps' : 'var_45'})


df_trial_stats['var_45'] = var_45.groupby(['subject','trained','r_bin','pref_bin'],as_index=False).mean().var_45

df_trial_stats['sigma_45'] = df_trials[df_trials.stim_ori==45].groupby(
    ['cell','subject','trained','r_bin','pref_bin'],as_index=False, observed = True)['trial_resps'].std().groupby(
        ['subject','trained','r_bin','pref_bin'],as_index=False).mean().trial_resps

df_trial_stats['mu_90'] = df_trials[df_trials.stim_ori==90].groupby(
    ['cell','subject','trained','r_bin','pref_bin'],as_index=False, observed = True)['trial_resps'].mean().groupby(
        ['subject','trained','r_bin','pref_bin'],as_index=False).mean().trial_resps

df_trial_stats['sigma_90'] = df_trials[df_trials.stim_ori==90].groupby(
    ['cell','subject','trained','r_bin','pref_bin'],as_index=False, observed = True)['trial_resps'].std().groupby(
        ['subject','trained','r_bin','pref_bin'],as_index=False).mean().trial_resps

var_90 = df_trials[df_trials.stim_ori==90].groupby(
    ['cell','subject','trained','r_bin','pref_bin'],as_index=False, observed = True)['trial_resps'].var()
var_90 = var_90.rename(columns = {'trial_resps' : 'var_90'})

df_trial_stats['var_90'] = var_90.groupby(['subject','trained','r_bin','pref_bin'],as_index=False).mean().var_90

var_all = pd.concat([var_45,var_90.var_90],axis=1)

var_all['pooled_sigma'] = np.sqrt((var_all.var_45 + var_all.var_90)/2)


df_trial_stats['pooled_sigma'] = var_all.groupby(['subject','trained','r_bin','pref_bin'],as_index=False).mean().pooled_sigma

df_trial_stats['mu_diff'] = df_trial_stats['mu_45'] - df_trial_stats['mu_90']
df_trial_stats['abs_mu_diff'] = df_trial_stats.mu_diff.abs()

ori_colors = sns.color_palette('hsv',8)

# sns.relplot(data = df_trial_stats, x = 'mu_45', y = 'sigma_45', row = 'trained',
#             col = 'subject', hue = 'pref_bin', palette = 'hsv', style = 'r_bin')
# sns.relplot(data = df_trial_stats, x = 'mu_90', y = 'sigma_90', row = 'trained',
#             col = 'subject', hue = 'pref_bin', palette = 'hsv', style = 'r_bin')

# sns.relplot(data = df_trial_stats, x = 'mu_45', y = 'sigma_45', hue = 'trained',
#             col = 'subject', style = 'r_bin', palette = 'colorblind')
# sns.relplot(data = df_trial_stats, x = 'mu_90', y = 'sigma_90', hue = 'trained',
#             col = 'subject', style = 'r_bin', palette = 'colorblind')

# sns.relplot(data = df_trial_stats, x = 'mu_45', y = 'var_45', style = 'trained',
#             col = 'subject', palette = 'hsv', hue = 'pref_bin', legend = True)
# sns.relplot(data = df_trial_stats, x = 'mu_90', y = 'var_90', style = 'trained',
#             col = 'subject', palette = 'hsv', hue = 'pref_bin', legend = False)

f,a = plt.subplots(1,2, sharex=True, sharey=True)

sns.scatterplot(data = df_trial_stats, x = 'mu_45', y = 'sigma_45', style = 'trained',
            palette = 'hsv', hue = 'pref_bin', legend = False, ax = a[0])
sns.scatterplot(data = df_trial_stats, x = 'mu_90', y = 'sigma_90', style = 'trained',
            palette = 'hsv', hue = 'pref_bin', legend = False, ax = a[1])

for a in a:
    a.set_aspect(1)

f,a = plt.subplots(2,2)
sns.scatterplot(x = df_trial_stats[df_trial_stats.trained=='Naive'].sigma_45.to_numpy(),
                y = df_trial_stats[df_trial_stats.trained=='Proficient'].sigma_45.to_numpy(),
                hue = df_trial_stats[df_trial_stats.trained=='Naive'].pref_bin.to_numpy().astype(int).astype(str),
                style = df_trial_stats[df_trial_stats.trained=='Proficient'].r_bin.to_numpy(),
                ax = a[0,1], palette = ori_colors, legend = False)

sns.scatterplot(x = df_trial_stats[df_trial_stats.trained=='Naive'].mu_45.to_numpy(),
                y = df_trial_stats[df_trial_stats.trained=='Proficient'].mu_45.to_numpy(),
                hue = df_trial_stats[df_trial_stats.trained=='Naive'].pref_bin.to_numpy().astype(int).astype(str),
                style = df_trial_stats[df_trial_stats.trained=='Proficient'].r_bin.to_numpy(),
                ax = a[0,0], palette = ori_colors, legend = False)

a[0,0].set_xlabel('Naive mean')
a[0,0].set_ylabel('Trained mean')

a[0,1].set_xlabel('Naive std')
a[0,1].set_ylabel('Trained std')

sns.scatterplot(x = df_trial_stats[df_trial_stats.trained=='Naive'].sigma_90.to_numpy(),
                y = df_trial_stats[df_trial_stats.trained=='Proficient'].sigma_90.to_numpy(),
                hue = df_trial_stats[df_trial_stats.trained=='Naive'].pref_bin.to_numpy().astype(int).astype(str),
                style = df_trial_stats[df_trial_stats.trained=='Proficient'].r_bin.to_numpy(),
                ax = a[1,1], palette = ori_colors, legend = False)

sns.scatterplot(x = df_trial_stats[df_trial_stats.trained=='Naive'].mu_90.to_numpy(),
                y = df_trial_stats[df_trial_stats.trained=='Proficient'].mu_90.to_numpy(),
                hue = df_trial_stats[df_trial_stats.trained=='Naive'].pref_bin.to_numpy().astype(int).astype(str),
                style = df_trial_stats[df_trial_stats.trained=='Proficient'].r_bin.to_numpy(),
                ax = a[1,0], palette = ori_colors, legend = False)

a[1,0].set_xlabel('Naive mean')
a[1,0].set_ylabel('Trained mean')

a[1,1].set_xlabel('Naive std')
a[1,1].set_ylabel('Trained std')


for a in a.flatten():
    a.set_xlim([0,1])
    a.set_ylim([0,1])
    a.plot([0,1],[0,1],'--k')
    a.set_aspect(1)


a = sns.relplot(data = df_trial_stats,
                x = 'mu_45',
                y = 'mu_90',
                hue = 'trained',
                style = 'r_bin',
                palette = 'colorblind',
                col = 'pref_bin',
                col_wrap = 4)

for a in a.axes:
    a.plot([0,1],[0,1],'--k')


bins = np.linspace(0,1,11)

f,a = plt.subplots(2,4, sharex = True, sharey = True)

for i,p in enumerate(df_trial_stats.pref_bin.unique()):

    sns.histplot(x = np.abs(df_trial_stats[df_trial_stats.pref_bin==p].mu_45.to_numpy()
                 - df_trial_stats[df_trial_stats.pref_bin==p].mu_90.to_numpy()),
                hue = df_trial_stats[df_trial_stats.pref_bin==p].trained,
                palette = 'colorblind',
                ax = a.flatten()[i], legend = False, bins = bins)
    a.flatten()[i].set_title(p)


sns.relplot(data = df_trial_stats,
                x = 'sigma_45',
                y = 'sigma_90',
                hue = 'trained',
                style = 'r_bin',
                palette = 'colorblind',
                col = 'pref_bin',
                col_wrap = 4,
                legend = False)

bins = np.linspace(0,1,11)

f,a = plt.subplots(2,4, sharex = True, sharey = True)


for i,p in enumerate(df_trial_stats.pref_bin.unique()):

    sns.histplot(x = df_trial_stats[df_trial_stats.pref_bin==p].pooled_sigma,
                hue = df_trial_stats[df_trial_stats.pref_bin==p].trained,
                palette = 'colorblind',
                ax = a.flatten()[i], legend = False, bins = bins)



f,a = plt.subplots(1,1)
sns.scatterplot(x = df_trial_stats[df_trial_stats.trained=='Naive'].pooled_sigma.to_numpy(),
                y = df_trial_stats[df_trial_stats.trained=='Proficient'].pooled_sigma.to_numpy(),
                hue = df_trial_stats[df_trial_stats.trained=='Naive'].pref_bin.to_numpy().astype(int).astype(str),
                style = df_trial_stats[df_trial_stats.trained=='Proficient'].r_bin.to_numpy(),
                ax = a, palette = ori_colors, legend = False)

a.set_xlim([0,0.8])
a.set_ylim([0,0.8])
a.plot([0,0.8],[0,0.8],'--k')


f,a = plt.subplots(1,1)
sns.scatterplot(x = df_trial_stats[df_trial_stats.trained=='Naive'].abs_mu_diff.to_numpy(),
                y = df_trial_stats[df_trial_stats.trained=='Proficient'].abs_mu_diff.to_numpy(),
                style = df_trial_stats[df_trial_stats.trained=='Naive'].pref_bin.to_numpy().astype(int).astype(str),
                #style = df_trial_stats[df_trial_stats.trained=='Proficient'].r_bin.to_numpy(),
                ax = a,
                #palette = ori_colors,
                legend = True,
                hue = df_trial_stats[df_trial_stats.trained=='Proficient'].pooled_sigma.to_numpy()-df_trial_stats[df_trial_stats.trained=='Naive'].pooled_sigma.to_numpy())

a.set_xlim([0,1])
a.set_ylim([0,1])
a.plot([0,1],[0,1],'--k')


diff_std = df_trial_stats[df_trial_stats.trained=='Proficient'].pooled_sigma.to_numpy() - df_trial_stats[df_trial_stats.trained=='Naive'].pooled_sigma.to_numpy()
diff_mu = df_trial_stats[df_trial_stats.trained=='Proficient'].abs_mu_diff.to_numpy() - df_trial_stats[df_trial_stats.trained=='Naive'].abs_mu_diff.to_numpy()

f,a = plt.subplots(1,1)
sns.scatterplot(x = diff_mu, y = diff_std,
                hue = df_trial_stats[df_trial_stats.trained=='Naive'].pref_bin.to_numpy().astype(int).astype(str),
                style = df_trial_stats[df_trial_stats.trained=='Proficient'].r_bin.to_numpy(),
                ax = a,
                palette = ori_colors)

a.plot([0,0],[-0.5,0.5], '--k')
a.plot([-0.5,0.5],[0,0], '--k')
a.plot([-0.5,0.5,],[-0.5,0.5], '--k')
a.set_xlabel('Change in mean difference')
a.set_ylabel('Change in std')

#%% Plot orientation preferences - Max response

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

g_df = pd.DataFrame.copy(df_scm)

g_df = g_df.astype({'pref_ori_all' : int})
# g_df = g_df.astype({'pref_ori_all' : str})
# g_df = g_df.astype({'trained' : str})

g_df = g_df.groupby(['subject', 'trained'])\
    ['pref_ori_all'].value_counts(normalize = True)\
        .rename('Proportion_of_cells').reset_index()

model = ols('Proportion_of_cells ~ C(subject) + C(trained)*C(pref_ori_all)', data = g_df).fit()

table = sm.stats.anova_lm(model,typ=2)
print(table)

# Repeated measures anova
# model_rm = pg.rm_anova(g_df, 'Proportion_of_cells', ['trained','pref_ori_all'], 'subject')

# Do pairwise tests
p_val_pref = np.zeros(len(np.unique(g_df.pref_ori_all)))
mu = np.zeros((len(np.unique(g_df.pref_ori_all))))
sem = np.copy(mu)

for i,p in enumerate(np.unique(g_df.pref_ori_all)):
    ind = g_df.pref_ori_all == p
    cond = g_df[ind].trained
    prop = g_df[ind].Proportion_of_cells.to_numpy()
    p_val_pref[i] = stats.ttest_rel(prop[cond=='Naive'], prop[cond=='Proficient'])[1]
    diff = prop[cond=='Proficient']-prop[cond=='Naive']
    mu[i] = diff.mean()
    sem[i] = diff.std()/np.sqrt(5)

# # p_val_pref = multipletests(p_val_pref,method = 'hs')[1]

# p_val_label = np.zeros(len(np.unique(g_df.pref_ori_all)), dtype = 'object')

# for i,p in enumerate(p_val_pref):
#     if p < 0.0001:
#         p_val_label[i] = '****'
#     elif p<0.001:
#         p_val_label[i] = '***'
#     elif p<0.01:
#         p_val_label[i] = '**'
#     elif p<0.05:
#         p_val_label[i] = '*'
#     else:
#         p_val_label[i] = ''

#%%

cp = sns.color_palette('Paired',20)
cp = [cp[i] for i in [0,1]]


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,
                                        "axes.titlesize":5,
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


    # mpl.rcParams['font.serif'] = "Times New Roman"
    # mpl.rcParams["font.family"] = "serif"

    # mpl.rcParams['font.sans-serif'] = "Helvetica Neue"
    # mpl.rcParams["font.family"] = "sans-serif"

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    fig = plt.figure(figsize=(1.8,1.5))
    ax = fig.add_subplot(111)
    ori_fig = sns.barplot(x = 'pref_ori_all', y = 'Proportion_of_cells',
                          data = g_df, hue = 'trained', hue_order = ['Naive','Proficient','Passive'],
                          linewidth = 0.5, edgecolor = 'k',
                          palette = 'colorblind', errorbar=('se',1),
                          ax = ax)
    ori_fig.legend_.remove()
    ori_fig.set_xlabel('Modal orientation preference',labelpad=1)
    ori_fig.set_ylabel('Proportion of cells',labelpad=1)
    ax.set_ylim([0,0.3])

    # Manually add SEM
    # cc = 0
    # for c in np.unique(g_df.trained):
    #     for si,s in enumerate(np.unique(g_df.pref_ori_all)):
    #         ind = np.logical_and(g_df.pref_ori_all == s, g_df.trained == c)
    #         sem = g_df.loc[ind,'Proportion_of_cells'].std()/np.sqrt(ind.sum())
    #         mean = g_df.loc[ind,'Proportion_of_cells'].mean()
    #         ax.errorbar(ax.patches[cc].get_x()+ax.patches[cc].get_width()/2,mean,sem,
    #                     color = 'black')
    #         cc+=1

    xlabels = [s+r'$\degree$' for s in np.unique(g_df.pref_ori_all.to_numpy()).astype(str)]

    ori_fig.set_xticklabels(xlabels, rotation = 20)
    # ori_fig.set_xticklabels(xlabels)


    plt.setp(ori_fig.axes.get_xticklabels()[2:5], fontweight = 'bold')
    # plt.setp(ori_fig.axes.get_xticklabels()[0], fontweight = 'bold')
    # plt.setp(ori_fig.axes.get_xticklabels()[4], fontweight = 'bold')



    # plt.setp(ori_fig.axes.get_xticklabels()[10:13], fontweight = 'bold')

    # for x in ori_fig.axes.get_xticks():
    #     plt.text(x,0.175, p_val_label[x], horizontalalignment = 'center',
    #              fontsize = 7)

    sns.despine(trim = True)
    plt.tight_layout()

    savefile = join(fig_save_dir,'ori_pref_modal_histogram_2.svg')
    fig.savefig(savefile, format = 'svg')


    fig = plt.figure(figsize=(1.8,1.5))
    ax = fig.add_subplot(111)

    # sns.stripplot(data=g_df,x = 'pref_ori_all', y = 'Proportion_of_cells', hue = 'trained',
    #               dodge = 0.4, edgecolor = 'k', linewidth = 0.5, ax = ax, palette = 'colorblind',
    #               s = 2, zorder = 2, jitter = True)

    sns.stripplot(data=g_df,x = 'pref_ori_all', y = 'Proportion_of_cells', hue = 'trained',
                  dodge = 0.4, edgecolor = 'k', linewidth = 0.5, ax = ax, palette = 'colorblind',
                  s = 2, zorder = 2, jitter = True, hue_order = ['Naive','Proficient','Passive'])

    plt.setp(ax.collections,zorder=2)

    # Add connecting lines
    # for c in range(1,16,2):
    #     for p0,p1 in zip(ax.collections[c-1].get_offsets(),ax.collections[c].get_offsets()):
    #         ax.plot([p0[0], p1[0]],[p0[1],p1[1]], color = [0.5,0.5,0.5], zorder = 0)

    # v_val=0.5
    # h_val=3.0
    # verts = list(zip([-h_val,h_val,h_val,-h_val],[-v_val,-v_val,v_val,v_val]))

    black = sns.dark_palette("black",n_colors = 1,reverse=True)
    # sns.pointplot(data=g_df, x = 'pref_ori_all', y = 'Proportion_of_cells', hue = 'trained',
    #               dodge = 0.4, join = False, ax = ax, legend = False, palette = black,
    #               errwidth = 0.5, capsize = 0.2, color = 'k', errorbar = (1,'se'),
    #               linestyles = 'dashed', markers = '_', scale = 1.5)

    plt.setp(ax.lines[-48:],zorder=100)
    for c in ax.collections[-2:]:
        plt.setp(c,zorder=100)

    ax.legend_.remove()

    ax.set_ylabel('Proportion of cells',labelpad=1)
    ax.set_xlabel('Modal orientation preference',labelpad=1)
    ax.set_xticklabels(xlabels, rotation = 20)
    # for x in ax.get_xticks():
    #     plt.text(x,0.175, p_val_label[x], horizontalalignment = 'center',
    #              fontsize = 7)

    plt.setp(ax.get_xticklabels()[2:5], fontweight = 'bold')
    plt.setp(ax.get_xticklabels()[10:13], fontweight = 'bold')

    sns.despine(trim = True)

    fig.tight_layout()

    savefile = join(fig_save_dir,'ori_pref_modal_paired_plot_2.svg')
    fig.savefig(savefile, format = 'svg')



#%% Plot orientation preferences - Max response, close retinotopy

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg


# g_df = pd.DataFrame.copy(df_scm[df_scm.task_ret_bin <= 10])
# g_df = pd.DataFrame.copy(df_scm[df_scm.ROI_task_ret <= 30])
g_df = df_scm.copy()

g_df = g_df.astype({'pref_ori_all' : int})
# g_df = g_df.astype({'pref_ori_all' : str})

g_df = g_df.groupby(['trained', 'subject'])\
    ['pref_ori_all'].value_counts(normalize = True)\
        .rename('Proportion_of_cells').reset_index()

model = ols('Proportion_of_cells ~ C(subject) + C(trained)*C(pref_ori_all)', data = g_df).fit()

table = sm.stats.anova_lm(model, typ=2)

print(table)

# Repeated measures anova

model_rm = pg.rm_anova(g_df, 'Proportion_of_cells', ['trained','pref_ori_all'], 'subject')

# Do pairwise tests
p_val_pref = np.zeros(len(np.unique(g_df.pref_ori_all)))
mu = np.zeros((len(np.unique(g_df.pref_ori_all))))
sem = np.copy(mu)

# for i,p in enumerate(np.unique(g_df.pref_ori_all)):
#     ind = g_df.pref_ori_all == p
#     cond = g_df[ind].trained
#     prop = g_df[ind].Proportion_of_cells
#     p_val_pref[i] = stats.ttest_rel(prop[cond==False], prop[cond])[1]

for i,p in enumerate(np.unique(g_df.pref_ori_all)):
    ind = g_df.pref_ori_all == p
    cond = g_df[ind].trained
    prop = g_df[ind].Proportion_of_cells.to_numpy()
    p_val_pref[i] = stats.ttest_rel(prop[cond=='Naive'], prop[cond=='Proficient'])[1]
    diff = prop[cond=='Proficient']-prop[cond=='Naive']
    mu[i] = diff.mean()
    sem[i] = diff.std()/np.sqrt(5)


# p_val_pref = multipletests(p_val_pref,method = 'hs')[1]

p_val_label = np.zeros(len(np.unique(g_df.pref_ori_all)), dtype = 'object')

for i,p in enumerate(p_val_pref):
    if p < 0.0001:
        p_val_label[i] = '****'
    elif p<0.001:
        p_val_label[i] = '***'
    elif p<0.01:
        p_val_label[i] = '**'
    elif p<0.05:
        p_val_label[i] = '*'
    else:
        p_val_label[i] = 'n.s.'

cp = sns.color_palette('Paired',20)
cp = [cp[i] for i in [0,1]]

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,
                                        "axes.titlesize":5,
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


    # mpl.rcParams['font.serif'] = "Times New Roman"
    # mpl.rcParams["font.family"] = "serif"

    # mpl.rcParams['font.sans-serif'] = "Helvetica Neue"
    # mpl.rcParams["font.family"] = "sans-serif"

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(1.5,1.6))
    ax = fig.add_subplot(111)
    ori_fig = sns.barplot(x = 'pref_ori_all', y = 'Proportion_of_cells',
                          data = g_df, hue = 'trained',
                          linewidth = 0.5, edgecolor = 'k',
                          palette = 'colorblind', ci = None,
                          ax = ax)
    ori_fig.legend_.remove()
    ori_fig.set_xlabel('Preferred modal orientation',labelpad=1)
    ori_fig.set_ylabel('Proportion of cells',labelpad=1)
    ax.set_ylim([0,0.35])

    # Manually add SEM
    cc = 0
    for c in np.unique(g_df.trained):
        for si,s in enumerate(np.unique(g_df.pref_ori_all)):
            ind = np.logical_and(g_df.pref_ori_all == s, g_df.trained == c)
            sem = g_df.loc[ind,'Proportion_of_cells'].std()/np.sqrt(ind.sum())
            mean = g_df.loc[ind,'Proportion_of_cells'].mean()
            ax.errorbar(ax.patches[cc].get_x()+ax.patches[cc].get_width()/2,mean,sem,
                        color = 'black')
            cc+=1

    xlabels = [s+r'$\degree$' for s in np.unique(g_df.pref_ori_all.to_numpy()).astype(str)]

    ori_fig.set_xticklabels(xlabels, rotation = 20)
    # ori_fig.set_xticklabels(xlabels)

    plt.setp(ori_fig.axes.get_xticklabels()[2:5], fontweight = 'bold')
    plt.setp(ori_fig.axes.get_xticklabels()[10:13], fontweight = 'bold')

    for x in ori_fig.axes.get_xticks():
        plt.text(x,0.175, p_val_label[x], horizontalalignment = 'center',
                 fontsize = 7)

    plt.tight_layout()
    sns.despine(trim = True)

    savefile = join(results_dir,'Figures','Draft','ori_pref_modal_histogram_close ret.svg')
    # fig.savefig(savefile, format = 'svg')

#%% Plot orientation preferences - Max response, far retinotopy

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

g_df = pd.DataFrame.copy(df_scm[df_scm.ROI_task_ret > 20])

g_df = g_df.astype({'pref_ori_all' : int})
# g_df = g_df.astype({'pref_ori_all' : str})

g_df = g_df.groupby(['trained', 'subject'])\
    ['pref_ori_all'].value_counts(normalize = True)\
        .rename('Proportion_of_cells').reset_index()

model = ols('Proportion_of_cells ~ C(subject) + C(trained)*C(pref_ori_all)', data = g_df).fit()

table = sm.stats.anova_lm(model, typ=2)

print(table)

# Repeated measures anova
model_rm = pg.rm_anova(g_df, 'Proportion_of_cells', ['trained','pref_ori_all'], 'subject')

# Do stats
p_val_pref = np.zeros(len(np.unique(g_df.pref_ori_all)))
mu = np.zeros((len(np.unique(g_df.pref_ori_all))))
sem = np.copy(mu)

# for i,p in enumerate(np.unique(g_df.pref_ori_all)):
#     ind = g_df.pref_ori_all == p
#     cond = g_df[ind].trained
#     prop = g_df[ind].Proportion_of_cells
#     p_val_pref[i] = stats.ttest_rel(prop[cond==False], prop[cond])[1]

for i,p in enumerate(np.unique(g_df.pref_ori_all)):
    ind = g_df.pref_ori_all == p
    cond = g_df[ind].trained
    prop = g_df[ind].Proportion_of_cells.to_numpy()
    p_val_pref[i] = stats.ttest_rel(prop[cond=='Naive'], prop[cond=='Proficient'])[1]
    diff = prop[cond=='Proficient']-prop[cond=='Naive']
    mu[i] = diff.mean()
    sem[i] = diff.std()/np.sqrt(5)

# p_val_pref = multipletests(p_val_pref,method = 'hs')[1]

p_val_label = np.zeros(len(np.unique(g_df.pref_ori_all)), dtype = 'object')

for i,p in enumerate(p_val_pref):
    if p < 0.0001:
        p_val_label[i] = '****'
    elif p<0.001:
        p_val_label[i] = '***'
    elif p<0.01:
        p_val_label[i] = '**'
    elif p<0.05:
        p_val_label[i] = '*'
    else:
        p_val_label[i] = 'n.s.'

cp = sns.color_palette('Paired',20)
cp = [cp[i] for i in [0,1]]


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,
                                        "axes.titlesize":5,
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


    # mpl.rcParams['font.serif'] = "Times New Roman"
    # mpl.rcParams["font.family"] = "serif"

    # mpl.rcParams['font.sans-serif'] = "Helvetica Neue"
    # mpl.rcParams["font.family"] = "sans-serif"

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"


    fig = plt.figure(figsize=(1.5,1.6))
    ax = fig.add_subplot(111)
    ori_fig = sns.barplot(x = 'pref_ori_all', y = 'Proportion_of_cells',
                          data = g_df, hue = 'trained',
                          linewidth = 0.5, edgecolor = 'k',
                          palette = 'colorblind', ci = None,
                          ax = ax)
    ori_fig.legend_.remove()
    ori_fig.set_xlabel('Preferred modal orientation',labelpad=1)
    ori_fig.set_ylabel('Proportion of cells',labelpad=1)
    ax.set_ylim([0,0.35])

    # Manually add SEM
    cc = 0
    for c in np.unique(g_df.trained):
        for si,s in enumerate(np.unique(g_df.pref_ori_all)):
            ind = np.logical_and(g_df.pref_ori_all == s, g_df.trained == c)
            sem = g_df.loc[ind,'Proportion_of_cells'].std()/np.sqrt(ind.sum())
            mean = g_df.loc[ind,'Proportion_of_cells'].mean()
            ax.errorbar(ax.patches[cc].get_x()+ax.patches[cc].get_width()/2,mean,sem,
                        color = 'black')
            cc+=1

    xlabels = [s+r'$\degree$' for s in np.unique(g_df.pref_ori_all.to_numpy()).astype(str)]

    ori_fig.set_xticklabels(xlabels, rotation = 20)
    # ori_fig.set_xticklabels(xlabels)


    plt.setp(ori_fig.axes.get_xticklabels()[2:5], fontweight = 'bold')
    plt.setp(ori_fig.axes.get_xticklabels()[10:13], fontweight = 'bold')

    for x in ori_fig.axes.get_xticks():
        plt.text(x,0.175, p_val_label[x], horizontalalignment = 'center',
                 fontsize = 7)

    plt.tight_layout()
    sns.despine(trim = True)

    savefile = join(results_dir,'Figures','Draft','ori_pref_modal_histogram_far ret.svg')
    fig.savefig(savefile, format = 'svg')

#%% Plot orientation preferences - Max response - Relationship to task retinotopy

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols


g_df = pd.DataFrame.copy(df_scm)

g_df = g_df.astype({'pref_ori_all' : int})

g_df = g_df.groupby(['trained', 'subject', 'task_ret_bin'])\
    ['pref_ori_all'].value_counts(normalize = True)\
        .rename('Proportion of cells').reset_index()


cp = sns.color_palette('Paired',20)
cp = [cp[i] for i in [0,1]]


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



    ori_fig = sns.catplot(x = 'pref_ori_all', y = 'Proportion of cells',
                          col = 'task_ret_bin', kind = 'bar', col_wrap = 4,
                          data = g_df, hue = 'trained',
                          linewidth = 0.5, edgecolor = 'k',
                          palette = 'colorblind',
                          ci = 68)
    ori_fig._legend.remove()
    # ori_fig.set_xlabel('Preferred modal orientation (deg)',labelpad=1)
    # ori_fig.set_ylabel('Proportion of cells',labelpad=1)
    ori_fig.set_axis_labels('Preferred modal orientation (deg)',
                            'Proportion of cells')


    plt.tight_layout()
    sns.despine()

#%% Plot change in proportion of cells preferring task orientations as function of r - for each stim
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

g_df = pd.DataFrame.copy(df_scm)

g_df = g_df.astype({'pref_ori_all' : int})

g_df['weights'] = np.nan

for s in np.unique(g_df.subject):
    for c in np.unique(g_df.trained):
        ind = np.logical_and(g_df.subject == s, g_df.trained == c)
        g_df.loc[ind, 'weights'] = 1/np.sum(ind)


n_conds = len(np.unique(trained))
n_subs = len(np.unique(subjects))

prop_pref = np.zeros((n_subs,n_conds,len(np.unique(g_df.pref_ori_all)),5))

for io,p in enumerate(np.unique(g_df.pref_ori_all)):
       for isub,s in enumerate(np.unique(g_df.subject)):
        for ic,c in enumerate(np.unique(trained)):
            ind = (g_df.subject == s) & (g_df.trained == c) & (g_df.pref_ori_all==p)
            prop_pref[isub,ic,io,:] = np.histogram(g_df[ind].r_all, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), weights = g_df[ind].weights)[0]
            # prop_pref[isub,ic,io,:] = np.histogram(g_df[ind].r_all, np.linspace(0,1,6), weights = g_df[ind].weights)[0]


prop_pref_diff = np.diff(prop_pref,axis=1).squeeze()

mu = np.zeros((3,5))
sem = np.copy(mu)
p_vals = np.zeros((3,5))

for i in range(3):
    for s in range(5):
        p_vals[i,s] = ttest_rel(prop_pref[:,0,i+2,s],prop_pref[:,1,i+2,s])[1]
        mu[i,s] = prop_pref_diff[:,i+2,s].mean(0)
        sem[i,s] = prop_pref_diff[:,i+2,s].std(0)/np.sqrt(5)

pref_label = np.tile(np.unique(g_df.pref_ori_all)[None,None,:,None],(n_subs,n_conds,1,5))
sub_label = np.tile(np.unique(g_df.subject)[:,None,None,None],(1,n_conds,8,5))
r_label = np.tile(np.unique(np.arange(5))[None,None,None,:],(n_subs,n_conds,8,1))
trained_label = np.tile(np.unique(g_df.trained)[None,:,None,None],(n_subs,1,8,5))

df_prop = pd.DataFrame({'subject' : sub_label.flatten(),
                        'pref_ori' : pref_label.flatten(),
                        'r' : r_label.flatten(),
                        'trained' : trained_label.flatten(),
                        'prop_cells' : prop_pref.flatten()})

# model = ols('prop_cells ~ C(subject) + C(trained)*C(r)', data = df_prop[df_prop.pref_ori == 0]).fit()

# table = sm.stats.anova_lm(model)

# model_rm = pg.rm_anova(df_prop[df_prop.pref_ori==0], 'prop_cells', ['r','trained'], 'subject')

# def set_label(p):
#     if p < 0.0001:
#         return '****'
#     elif p<0.001:
#         return '***'
#     elif p<0.01:
#         return '**'
#     elif p<0.05:
#         return '*'
#     else:
#         return 'n.s.'

# pval_label = np.array([set_label(p) for p in p_vals.flatten()])
# pval_label = pval_label.reshape(3,2)

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

    f,ax = plt.subplots(1,3,figsize=(3.2,1.45))

    prefs = [45,68,90]
    # prefs = [0,45,90]

    for i,a in enumerate(ax.flatten()):

        df = df_prop[df_prop.pref_ori==prefs[i]]

    #     a.stairs(prop_pref[:,0,i,:].mean(0),np.linspace(0,1,6),color = cond_colors[0], linewidth = 0.5)
    #     a.stairs(prop_pref[:,1,i,:].mean(0),np.linspace(0,1,6),color = cond_colors[1], linewidth = 0.5)
        sns.barplot(data = df, x = 'r', y = 'prop_cells', hue='trained',
                    linewidth = 0.5, edgecolor = 'k',
                    palette = 'colorblind', ci = None,
                    ax = a)
        a.legend_.remove()
        a.set_xticks(np.arange(5))
        a.tick_params(axis='x', which='major', pad=1)
        a.set_xticklabels(['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64','0.64 - 1'], rotation = 30)
        a.set_title('Modal ori. pref. ' + str(prefs[i]))
        a.set_xlabel('Selectivity',labelpad=1)
        a.set_ylim([0,0.06])
        if i > 0:
            sns.despine(ax=a,trim=True, left = True)
            a.set_yticks([])
            a.set_ylabel('')
        else:
            sns.despine(ax=a,trim = True)
            a.set_ylabel('Total proportion of cells',labelpad=1)

        # Manually add SEM
        cc = 0
        for c in np.unique(df.trained):
            for si,s in enumerate(np.unique(df[df.pref_ori==prefs[i]].r)):
                ind = np.logical_and(df.r == s, df.trained == c)
                sem = df.loc[ind,'prop_cells'].std()/np.sqrt(ind.sum())
                mean = df.loc[ind,'prop_cells'].mean()
                a.errorbar(a.patches[cc].get_x()+a.patches[cc].get_width()/2,mean,sem,
                            color = 'black')
                cc+=1

        # a.text(0,0.077,pval_label[i,0],fontsize = 7,horizontalalignment='center', verticalalignment = 'center')
        # a.text(1,0.077,pval_label[i,1],fontsize = 7,horizontalalignment='center', verticalalignment = 'center')

    f.tight_layout()
    # f.savefig(join(fig_save_dir,'ori_pref_modal_histogram_by_r.svg'), format = 'svg')


#%% Plot change in proportion of cells preferring task orientations as function of r
# TASK STIM ONLY

from scipy.stats import ttest_rel

g_df = pd.DataFrame.copy(df_scm)

g_df = g_df[(g_df.pref_ori_all >= 45) & (g_df.pref_ori_all <= 90)]

g_df = g_df.astype({'pref_ori_all' : int})

g_df['weights'] = np.nan

for s,c in zip(subjects,trained):
    ind = np.logical_and(g_df.subject == s, g_df.trained == c)
    g_df.loc[ind, 'weights'] = 1/np.sum(ind)

prop_pref = np.zeros((5,2,3,5))

for io,p in enumerate([45,68,90]):
       for isub,s in enumerate(np.unique(g_df.subject)):
        for ic,c in enumerate([False,True]):
            ind = (g_df.subject == s) & (g_df.trained == c) & (g_df.pref_ori_all==p)
            prop_pref[isub,ic,io,:] = np.histogram(g_df[ind].r_all, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), weights = g_df[ind].weights)[0]
            # prop_pref[isub,ic,io,:] = np.histogram(g_df[ind].r_all, np.linspace(0,1,6),
            # weights = g_df[ind].weights)[0]

# Average proportion of cells preferring 45, 68, and 90 that have selectivity below 0.5

# prop_pref_lowr = prop_pref[...,0:2].mean(3)
prop_pref_lowr = prop_pref[...,0]


pval=[]
pval.append(ttest_rel(prop_pref_lowr[:,0,0],prop_pref_lowr[:,1,0])[1])
pval.append(ttest_rel(prop_pref_lowr[:,0,1],prop_pref_lowr[:,1,1])[1])
pval.append(ttest_rel(prop_pref_lowr[:,0,2],prop_pref_lowr[:,1,2])[1])

cond_colors = sns.color_palette('colorblind')[0:2]

oris = [r'45$\degree$', r'68$\degree$', r'90$\degree$']

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

    f,ax = plt.subplots(1,3,figsize=(3.2,1.4), sharex = True, sharey = True)

    for i,a in enumerate(ax.flatten()):
        a.stairs(prop_pref[:,0,i,:].mean(0),np.linspace(0,1,6),color = cond_colors[0], linewidth = 0.5)
        a.stairs(prop_pref[:,1,i,:].mean(0),np.linspace(0,1,6),color = cond_colors[1], linewidth = 0.5)
        # a.set_xticks([0,2,4])
        # a.set_xticklabels([0,0.5,1])
        a.set_title('Pref. modal ori. ' + oris[i])
        a.set_xlabel('Selectivity',labelpad=1)
        if i == 0:
            a.set_ylabel('Total proportion of cells',labelpad=1)
        a.set_ylim([0,0.2])
        sns.despine(ax=a,trim=True)

    f.tight_layout()

    # r_dist_pref.axes[0][1].set_yticks([])
    # r_dist_pref.axes[0][2].set_yticks([])

    # savefile = join(results_dir,'Figures','Draft','ori_pref_modal_histogram_by_r.svg')
    # f.savefig(savefile, format = 'svg')


#%% Plot orientation preferences - th

g_df = pd.DataFrame.copy(df_scm)

n_bins = 8
bin_sp = 180/n_bins
g_df['th_all'] = g_df['th_all'] + 11.25 - bin_sp/2

g_df['pref_bin'] = pd.cut(g_df['th_all'], np.linspace(-bin_sp/2,180-bin_sp/2,n_bins+1),
                        labels = np.round(np.arange(0,180,bin_sp)).astype(int))
g_df['r_bin'] = pd.cut(g_df['r_all'], np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))

g_df = g_df.groupby(['trained', 'subject', 'r_bin'])\
    ['pref_bin'].value_counts(normalize = True)\
        .rename('Proportion of cells').reset_index()

cp = sns.color_palette('Paired',20)
cp = [cp[i] for i in [0,1]]

#plt.style.use('seaborn')
sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1):

    ori_fig = sns.catplot(x = 'pref_bin', y = 'Proportion of cells',
                data = g_df, kind = 'bar', hue = 'trained',
                #col = 'r_bin', col_wrap = 3,
                linewidth = 1.25, edgecolor = 'k',
                legend = False,
                legend_out = False,
                aspect = 1.4,
                height = 6,
                palette = 'colorblind')

    ori_fig.set_axis_labels('Preferred orientation', 'Proportion of cells')
    leg = plt.legend(loc = 'upper right')
    leg.get_frame().set_linewidth(0.0)
    leg.set_title('')
    leg.texts[0].set_text('Naïve')
    leg.texts[1].set_text('Proficient')


    # for ax in ori_fig.axes.flat:
    #     plt.setp(ax.get_xticklabels()[2:5], fontweight = 'bold')
    #     plt.setp(ax.get_xticklabels()[10:13], fontweight = 'bold')


#%% Plot difference in modal vs mean pref

g_df = pd.DataFrame.copy(df_scm)

g_df['pref_bin'] = g_df['pref_bin'].astype(int)

# g_df['pref_diff'] = np.abs(g_df['pref_bin'] - g_df['pref_ori_test'])

g_df['pref_diff'] = np.abs(g_df['th'] + 11.25 - g_df['pref_ori_test'])

def correction(x):
    if x > 90:
        x = np.abs(x-180)
    return x

g_df['pref_diff'] = g_df['pref_diff'].apply(correction)

g_df = g_df.groupby(['subject', 'trained', 'pref_bin', 'r_bin']).mean().reset_index()
# g_df = g_df.groupby(['subject', 'trained', 'pref_bin']).mean().reset_index()

# ax = sns.catplot(data = g_df, x = 'pref_bin', y = 'pref_diff', hue = 'trained',
#                   kind = 'swarm', dodge = True, col = 'r_bin')
# ax = sns.catplot(data = g_df, x = 'pref_bin', y = 'pref_diff', hue = 'trained',
#                   kind = 'bar', dodge = True, palette='colorblind')
ax = sns.catplot(data = g_df, x = 'pref_bin', y = 'pref_diff', hue = 'trained',
                  kind = 'bar', col = 'r_bin', dodge = True, palette='colorblind')
#%% Plot selectivity as function of orientation preference (modal)

g_df = pd.DataFrame.copy(df_scm)

g_df = g_df.groupby(['subject', 'trained', 'pref_ori_all']).mean().reset_index()

g_df_diff = pd.DataFrame.copy(g_df[g_df.trained == True])
g_df_diff = g_df_diff.reset_index(drop=True)
g_df_diff.r_all -= g_df[g_df.trained==False].reset_index(drop=True).r_all

# cp = sns.color_palette('Paired',20)
# cp = [cp[i] for i in [0,1]]

ori_colors = sns.hls_palette(8)
cond_colors = sns.color_palette('colorblind')[0:2]

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


    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'

    g_OSI = sns.relplot(x = 'pref_ori_all', y = 'r_all', hue = 'trained',
                        data = g_df, kind = 'line',
                        palette = 'colorblind')
    g_OSI.fig.set_size_inches((4,2.5))
    g_OSI.set(xticks = np.ceil(np.arange(0,180,22.5)))
    g_OSI._legend.set_title('')
    g_OSI._legend.texts[0].set_text('Naïve')
    g_OSI._legend.texts[1].set_text('Proficient')
    g_OSI.set_axis_labels('Preferred orientation (deg)','Selectivity')
    plt.axvline(45, ls='-', color = 'black', linewidth = 1)
    plt.axvline(68, ls='--', color = 'black', linewidth = 1)
    plt.axvline(90, ls='-', color = 'black', linewidth = 1)
    plt.tight_layout()

    for ax in g_OSI.axes.flat:
        plt.setp(ax.get_xticklabels()[2:5], fontweight = 'bold')

    plt.figure(figsize=(4,2.5))
    g_OSI = sns.pointplot(x = 'pref_ori_all', y = 'r_all', hue = 'pref_ori_all',
                        data = g_df_diff, join = False,
                        legend = False, palette = ori_colors, errorbars = ('se',1),
                        capsize = 0.3, linestyle = '--',
                        markers = '_')
    g_OSI = sns.stripplot(x = 'pref_ori_all', y = 'r_all', hue = 'pref_ori_all',
                        data = g_df_diff,
                        palette = ori_colors,
                        s = 3)
    g_OSI.set(xticklabels = np.ceil(np.arange(0,180,22.5)).astype(int))
    plt.xlabel('Modal orientation preference (deg)')
    plt.ylabel('Selectivity')
    g_OSI.legend_.set_visible(False)
    plt.tight_layout()
    sns.despine()

    plt.figure(figsize=(3,1.5))
    pref_45 = g_df.pref_ori_all == 45
    pref_90 = g_df.pref_ori_all == 90
    g_df_naive = g_df[np.logical_and(g_df.trained==False,pref_45 | pref_90)]
    g_df_trained = g_df[np.logical_and(g_df.trained,pref_45 | pref_90)]
    # g_df_naive = g_df[g_df.trained == False]
    # g_df_trained = g_df[g_df.trained]
    g_OSI = sns.scatterplot(x = g_df_naive.r_all.to_numpy(),
                            y = g_df_trained.r_all.to_numpy(),
                            hue = g_df_trained.pref_ori_all,
                            palette = ori_colors[slice(2,5,2)],
                            # palette = ori_colors,
                            s = 6, legend = False,
                            linewidth = 0.5, edgecolor = 'black')
    plt.xlabel('Naïve selectivity', color = cond_colors[0])
    plt.ylabel('Proficient selectivity', color = cond_colors[1])
    # plt.title('Selectivity')
    g_OSI.set_aspect('equal', 'box')
    plt.xlim([0.185,0.5])
    plt.ylim([0.185,0.5])
    plt.plot([0.2,0.5],[0.2,0.5],'--k')
    plt.xticks(np.arange(0.2,0.6,0.1))
    plt.yticks(np.arange(0.2,0.6,0.1))
    plt.tight_layout()
    sns.despine(trim=True)


#%% Plot selectivity as function of orientation preference (mean)

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

g_df = pd.DataFrame.copy(df_scm)

g_df['pref_bin'] = pd.cut(g_df['th'], np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)).astype(int))

g_df = g_df.groupby(['subject', 'trained', 'pref_bin']).mean().reset_index()

g_df_diff = pd.DataFrame()
g_df_diff['r'] = g_df[g_df.trained=='Proficient'].r.to_numpy() - g_df[g_df.trained=='Naive'].r.to_numpy()
g_df_diff['subject'] = g_df[g_df.trained=='Proficient'].subject.reset_index(drop=True)
g_df_diff['pref_bin'] = g_df[g_df.trained=='Proficient'].pref_bin.reset_index(drop=True)


cp = sns.color_palette('Paired',20)
cp = [cp[i] for i in [0,1]]

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
                                        "lines.markersize":1,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    f,ax = plt.subplots(figsize=(2.5,1.5))

    sns.stripplot(data = g_df_diff,x='pref_bin', y = 'r',
                          color = 'black', ax = ax, s=3)
    ax.set_xlabel(r'Orientation preference (mean)')
    ax.set_ylabel(r'$\Delta$Selectivity')

    mean_width = 0.25

    x = g_df_diff.pref_bin.to_numpy()
    y = g_df_diff.r.to_numpy()

    ticks = ax.get_xticks()
    # labels = ax.get_xticklabels()
    # labels = np.array([l.get_text() for l in labels])
    labels = ['0','23','45','68','90','113','135','158']

    for i in range(len(labels)):

    # for t, l in zip(ticks, labels):

        t = ticks[i]
        l = labels[i]

        # sample_name = text.get_text()  # "X" or "Y"
        # print(sample_name)

        # calculate the median value for all replicates of either X or Y
        mean_val = y[x.astype(str)==l].mean()
        std_val = y[x.astype(str)==l].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x)/len(np.unique(x))
        ci_val = mean_confidence_interval(
            y[x.astype(str)==l], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        ax.plot([t-mean_width/2, t+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        ax.errorbar(t,mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)

    xlabels = np.ceil(np.arange(0,180,22.5)).astype(int).astype(str)
    deg = np.repeat([r'$\degree$'],8)

    xlabels = np.char.add(xlabels,deg)

    ax.set_xticklabels(xlabels)

    ax.set_yticks(np.linspace(0,0.12,3))

    ax.set_ylim([-0.005,0.12])

    sns.despine(trim=True, ax = ax)

    f.tight_layout()


#%%

g_df = pd.DataFrame.copy(df_scm)


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1):

    g_OSI = sns.relplot(x = 'th', y = 'r', hue = 'trained',
                        data = g_df, height = 10, aspect = 1.25,
                        palette = 'colorblind', s = 2, target = 'median')
    g_OSI.set(xticks = np.ceil(np.arange(0,180,22.5)))
    g_OSI._legend.set_title('')
    g_OSI._legend.texts[0].set_text('Naïve')
    g_OSI._legend.texts[1].set_text('Proficient')
    g_OSI.set_axis_labels('Preferred orientation','OSI')
    plt.axvline(45, ls='-', color = 'black', linewidth = 1)
    plt.axvline(68, ls='--', color = 'black', linewidth = 1)
    plt.axvline(90, ls='-', color = 'black', linewidth = 1)

    for ax in g_OSI.axes.flat:
        plt.setp(ax.get_xticklabels()[2:5], fontweight = 'bold')



#%% mean response by stim and condition

df = pd.DataFrame(trials_dict)

# Remove blank
df = df[df.stim_ori != np.inf]
# Use test set (r and th are from training set)
df = df[df.train_ind == False]

# df = df.drop(columns = ['r','pref'])


mu_groups = ['subject', 'trained', 'stim_ori', 'cell']

df_mu = df.groupby(mu_groups, observed = True).mean().reset_index()
df_mu = df_mu.groupby(mu_groups[:-1], observed = True).mean().reset_index()


#%% Find mean, sigma, and CV for ecah stimulus and cell class


df_trials = pd.DataFrame(trials_dict)

df_trials = df_trials[~df_trials.train_ind]
df_trials = df_trials[df_trials.stim_dir != np.inf]

groups = ['trial_resps','stim_ori', 'r', 'pref']

df_mu = df_trials.groupby(['cell','subject','trained','stim_dir'],
                            observed = True, as_index = False)[groups].mean().reset_index(drop=True)
df_mu['stim_ori'] = df_mu['stim_dir'] % 180
df_mu = df_mu.groupby(['cell','subject','trained','stim_ori'],
                                                observed = True, as_index = False).mean().reset_index(drop = True)

df_mu['pref_ori'] = pd.cut(df_mu.pref, np.linspace(-11.25,180-11.25,9),
                                                        labels = np.round(np.arange(0,180,22.5)).astype(int))

df_mu['r_bin'] = pd.cut(df_mu.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                                                        labels = np.arange(5))

df_sigma = df_trials.groupby(['cell','subject','trained','stim_dir'],
                                observed = True,as_index = False)[groups].agg({'trial_resps':'std',
                                                                               'stim_ori' : 'first',
                                                                               'r': 'first',
                                                                               'pref' : 'first'})
df_sigma = df_sigma.groupby(['cell','subject','trained','stim_ori'],
                                        observed = True, as_index = False)[groups].mean().reset_index(drop = True)

df_sigma['pref_ori'] = pd.cut(df_sigma.pref, np.linspace(-11.25,180-11.25,9),
                                                        labels = np.round(np.arange(0,180,22.5)).astype(int))

df_sigma['r_bin'] = pd.cut(df_sigma.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                                                        labels = np.arange(5))

df_cv = pd.DataFrame.copy(df_sigma)
df_cv['trial_resps'] = df_cv.trial_resps/df_mu.trial_resps



#%% For each class of cell, plot mean, sigma for both conditions
# Each point is average of all mice, with error bars across mice
# Each point is has a different shape, each color is a different stimulus
# 40 different classes... maybe group by weak and strongly tuned? 16 total instead

mu_resps = np.zeros(len(subjects), dtype = 'object')
std_resps = np.zeros(len(subjects), dtype = 'object')
cov_resps = np.zeros(len(subjects), dtype = 'object')
r_bin = np.zeros(len(subjects), dtype = 'object')
pref_bin = np.zeros(len(subjects), dtype = 'object')
ori_label = np.zeros(len(subjects), dtype = 'object')
sub_label = np.zeros(len(subjects),dtype='object')
cond_label = np.zeros(len(subjects),dtype='object')
cell_label = np.zeros(len(subjects),dtype='object')

for i,(sub,t) in enumerate(zip(subjects,trained)):

    ind = np.logical_and(trials_dict['subject'] == sub,
                         trials_dict['trained'] == t)

    cells = trials_dict['cell'][ind]
    resps = trials_dict['trial_resps'][ind]
    resps = resps.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
    stim = trials_dict['stim_dir'][ind].reshape(-1,len(np.unique(cells)))[:,0]
    cells = cells.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
    train_ind = trials_dict['train_ind'][ind]
    train_ind = train_ind.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)[:,0]
    test_ind = np.logical_not(train_ind)
    # test_ind = np.ones(len(train_ind)).astype(bool)

    resps = resps[test_ind,:]
    stim = stim[test_ind]
    nb_ind = stim != np.inf

    resps = resps[nb_ind,:]
    stim = stim[nb_ind]

    lb = preprocessing.LabelBinarizer()
    uni_stim = np.unique(stim)
    lb.fit(uni_stim)
    dm = lb.transform(stim)

    # # Average responses to each stimulus
    lr = LinearRegression(fit_intercept = False)
    lr.fit(dm, resps)
    residuals = resps - lr.predict(dm)

    mu_resps[i] = np.array([resps[stim % 180 ==s,:].mean(0) for s in np.unique(stim%180)])
    std_resps[i] = np.array([np.sqrt((residuals[stim % 180 ==s,:]**2).mean(0))
         for s in np.unique(stim%180)])
    cov_resps[i] = std_resps[i]/mu_resps[i]

    # r_bin[i] = df_scm[np.logical_and(df_scm.trained == t,
    #                               df_scm.subject==sub)].r_bin.to_numpy()
    r_bin[i] = df_scm[np.logical_and(df_scm.trained == t,
                                  df_scm.subject==sub)].r_bin.to_numpy()
    r_bin[i] = np.repeat(r_bin[i].reshape(1,-1),len(np.unique(stim%180)),axis=0)
    pref_bin[i] = df_scm[np.logical_and(df_scm.trained == t,
                                        df_scm.subject == sub)].pref_bin.to_numpy()
    # pref_bin[i] = df_scm[np.logical_and(df_scm.trained == t,
    #                                     df_scm.subject == sub)].pref_ori_all.to_numpy()
    pref_bin[i] = np.repeat(pref_bin[i].reshape(1,-1),len(np.unique(stim%180)),axis=0)

    ori_label[i] = np.repeat(np.unique(stim%180).reshape(-1,1),
                             mu_resps[i].shape[1],axis=1)

    sub_label[i] = np.tile(sub,mu_resps[i].shape)
    cond_label[i] = np.tile(t,mu_resps[i].shape)

df_resps = pd.DataFrame({'ori' : np.concatenate(ori_label,axis=1).flatten(),
                         'mu' : np.concatenate(mu_resps,axis=1).flatten(),
                         'cv' : np.concatenate(cov_resps,axis=1).flatten(),
                         'sigma' : np.concatenate(std_resps,axis=1).flatten(),
                         'pref' : np.concatenate(pref_bin,axis=1).flatten(),
                         'r_bin' : np.concatenate(r_bin,axis=1).flatten(),
                         'subject' : np.concatenate(sub_label,axis=1).flatten(),
                         'condition' : np.concatenate(cond_label,axis=1).flatten()
                         })


#%%

df_resps_cv = pd.DataFrame.copy(df_resps)
df_resps_cv = df_resps_cv.dropna()

df_expts = df_resps.groupby(['ori','pref','r_bin','subject','condition']).mean().reset_index()
df_expts_cv = df_resps_cv.groupby(['ori','pref','r_bin','subject','condition']).mean().reset_index()

df_sem = df_expts.groupby(['ori','pref','r_bin','condition']).std().reset_index()
df_sem['mu'] = df_sem['mu']/np.sqrt(len(np.unique(len(df_expts.subject))))
df_sem['sigma'] = df_sem['sigma']/np.sqrt(len(np.unique(len(df_expts.subject))))

df_sem_cv = df_expts_cv.groupby(['ori','pref','r_bin','condition']).std().reset_index()
df_sem_cv['mu'] = df_sem_cv['mu']/np.sqrt(len(np.unique(len(df_expts.subject))))
df_sem_cv['sigma'] = df_sem_cv['sigma']/np.sqrt(len(np.unique(len(df_expts.subject))))

df_mean = df_expts.groupby(['ori','pref','r_bin','condition']).mean().reset_index()
df_mean_cv = df_expts_cv.groupby(['ori','pref','r_bin','condition']).mean().reset_index()
# Plot changes for 45, 68, and 90

stimuli = [45,68,90]

pref_colors = sns.hls_palette(len(np.unique(df_expts.pref)))
r_markers = ['X','P','s','v','D']

cond_colors = sns.color_palette('colorblind')[0:2]

fig_mu, axes_mu = plt.subplots(1,3, sharex = True, sharey=True, figsize=(12,4))
fig_sigma, axes_sigma = plt.subplots(1,3, sharex = True, sharey=True,
                                     figsize=(12,4))

fig_cv, axes_cv = plt.subplots(1,3, sharex = True, sharey=True,
                                     figsize=(12,4))

for i,s in enumerate(stimuli):
    df_stim = df_mean[df_mean.ori == s]
    df_stim_sem = df_sem[df_sem.ori == s]

    df_stim_cv = df_mean_cv[df_mean_cv.ori == s]
    df_stim_sem_cv = df_sem_cv[df_sem_cv.ori == s]

    # Loop through and plot each class, pref color, r_bin style
    for ip,p in enumerate(np.unique(df_stim.pref)):
        for ir,r in enumerate(np.unique(df_stim.r_bin)):
            df_class = df_stim[np.logical_and(df_stim.r_bin==r,
                                              df_stim.pref==p)]
            df_class_sem = df_stim_sem[np.logical_and(df_stim_sem.r_bin==r,
                                              df_stim_sem.pref==p)]

            df_class_cv = df_stim_cv[np.logical_and(df_stim_cv.r_bin==r,
                                              df_stim_cv.pref==p)]
            df_class_sem_cv = df_stim_sem_cv[np.logical_and(df_stim_sem_cv.r_bin==r,
                                              df_stim_sem_cv.pref==p)]

            axes_mu[i].plot(df_class[df_class.condition==False].mu,
                         df_class[df_class.condition==True].mu, c = pref_colors[ip],
                         marker = r_markers[ir])
            axes_mu[i].set_xlim([0,1])
            axes_mu[i].set_ylim([0,1])
            axes_mu[i].plot([0, 1],[0,1],'k--')
            axes_mu[i].errorbar(df_class[df_class.condition==False].mu,
                          df_class[df_class.condition==True].mu,
                          xerr = df_class_sem[df_class_sem.condition==False].mu,
                          yerr = df_class_sem[df_class_sem.condition==True].mu,
                          color = pref_colors[ip])
            axes_mu[i].set_xlabel('Naïve',color = cond_colors[0])
            sns.despine(ax = axes_mu[i])
            axes_mu[i].set_aspect('equal', 'box')
            axes_mu[i].set_title(str(s)+r'$\degree$')

            axes_sigma[i].plot(df_class[df_class.condition==False].sigma,
                         df_class[df_class.condition==True].sigma, c = pref_colors[ip],
                         marker = r_markers[ir])
            axes_sigma[i].set_xlim([0,1])
            axes_sigma[i].set_ylim([0,1])
            axes_sigma[i].plot([0, 1],[0,1],'k--')
            axes_sigma[i].errorbar(df_class[df_class.condition==False].sigma,
                          df_class[df_class.condition==True].sigma,
                          xerr = df_class_sem[df_class_sem.condition==False].sigma,
                          yerr = df_class_sem[df_class_sem.condition==True].sigma,
                          color = pref_colors[ip])
            axes_sigma[i].set_xlabel('Naïve',color = cond_colors[0])
            sns.despine(ax = axes_sigma[i])
            axes_sigma[i].set_aspect('equal', 'box')
            axes_sigma[i].set_title(str(s)+r'$\degree$')

            axes_cv[i].plot(df_class_cv[df_class_cv.condition==False].cv,
                         df_class_cv[df_class_cv.condition==True].cv, c = pref_colors[ip],
                         marker = r_markers[ir])
            axes_cv[i].set_xlim([0.3,2.5])
            axes_cv[i].set_ylim([0.3,2.5])
            axes_cv[i].plot([0.3, 2.5],[0.3, 2.5],'k--')
            axes_cv[i].errorbar(df_class_cv[df_class_cv.condition==False].cv,
                          df_class_cv[df_class_cv.condition==True].cv,
                          xerr = df_class_sem_cv[df_class_sem_cv.condition==False].cv,
                          yerr = df_class_sem_cv[df_class_sem_cv.condition==True].cv,
                          color = pref_colors[ip])
            axes_cv[i].set_xlabel('Naïve',color = cond_colors[0])
            sns.despine(ax = axes_cv[i])
            axes_cv[i].set_aspect('equal', 'box')
            axes_cv[i].set_title(str(s)+r'$\degree$')

            if i == 0:
                axes_sigma[i].set_ylabel('Proficient',color = cond_colors[1])
                axes_mu[i].set_ylabel('Proficient',color = cond_colors[1])
                axes_cv[i].set_ylabel('Proficient',color = cond_colors[1])

#%% Plot change in d' with change in mean and change in std

# Find d' for 45 vs 90 for all cells

df_45 = df_resps[df_resps.ori == 45].reset_index(drop=True)
df_90 = df_resps[df_resps.ori == 90].reset_index(drop=True)

df_d = pd.DataFrame(columns=['d','r_bin','pref','subject','condition',
                             'mu_diff', 'rmse'])


df_d['mu_diff'] = np.abs(df_45.mu - df_90.mu)
df_d['rmse'] = np.sqrt((df_45.sigma**2+df_90.sigma**2)/2)

df_d['d'] =df_d.mu_diff/df_d.rmse

df_d['r_bin'] = df_45.r_bin
df_d['pref'] = df_45.pref
df_d['subject'] = df_45.subject
df_d['condition'] = df_45.condition


df_d_all = pd.DataFrame.copy(df_d)
df_d = df_d.groupby(['r_bin','pref','subject','condition']).mean().reset_index()

# Get difference in d', mu_diff, and rmse with training

df_d_diff = (df_d[df_d.condition == True].reset_index(drop=True)
             .drop(['subject','condition','r_bin','pref'],axis=1) -
             df_d[df_d.condition == False].reset_index(drop=True)
             .drop(['subject','condition','r_bin','pref'],axis=1))


df_d_diff['subject'] = df_d[df_d.condition==True].subject.to_numpy()
df_d_diff['pref'] = df_d[df_d.condition==True].pref.to_numpy()
df_d_diff['r_bin'] = df_d[df_d.condition==True].r_bin.to_numpy()


df_sem = df_d_diff.groupby(['r_bin','pref']).std().reset_index()
df_sem['mu_diff'] = df_sem['mu_diff']/np.sqrt(len(np.unique(len(df_d.subject))))
df_sem['rmse'] = df_sem['rmse']/np.sqrt(len(np.unique(len(df_d.subject))))
df_sem['d'] = df_sem['d']/np.sqrt(len(np.unique(len(df_d.subject))))

df_mean = df_d_diff.groupby(['r_bin','pref']).mean().reset_index()

pref_colors = sns.hls_palette(len(np.unique(df_resps.pref)))
r_markers = ['X','P','s','v','D']

cond_colors = sns.color_palette('colorblind')[0:2]

fig, axes = plt.subplots(1,2, sharex = True, sharey=True, figsize=(12,4))


# Loop through and plot each class, pref color, r_bin style
for ip,p in enumerate(np.unique(df_mean.pref)):
    for ir,r in enumerate(np.unique(df_mean.r_bin)):

        # df_class = df_d_diff[np.logical_and(df_d_diff.r_bin==r,
        #                                     df_d_diff.pref==p)]

        df_class = df_mean[np.logical_and(df_mean.r_bin==r,
                                          df_mean.pref==p)]

        df_class_sem = df_sem[np.logical_and(df_sem.r_bin==r,
                                          df_sem.pref==p)]
        if ir == 0:
            axes[0].scatter(df_class.mu_diff, df_class.d, color = pref_colors[ip],
                     marker = r_markers[ir], label = str(int(p)))
        else:
            axes[0].scatter(df_class.mu_diff, df_class.d, color = pref_colors[ip],
                     marker = r_markers[ir])
        axes_mu[i].set_xlim([0,1])
        axes_mu[i].set_ylim([0,1])
        axes_mu[i].plot([0, 1],[0,1],'k--')
        axes[0].errorbar(df_class.mu_diff, df_class.d,
                      xerr = df_class_sem.mu_diff,
                      yerr = df_class_sem.d,
                      color = pref_colors[ip])
        axes[0].set_xlabel(r'$\Delta \| Resp 45 - Resp 90 \|$')
        sns.despine(ax = axes[0])
        # axes[0].set_aspect('equal', 'box')
        axes[0].legend(markerscale = 1, frameon = False)

        if ir == 0:
            axes[1].scatter(df_class.rmse, df_class.d, color = pref_colors[ip],
                     marker = r_markers[ir], label = str(int(p)))
        else:
            axes[1].scatter(df_class.rmse, df_class.d, color = pref_colors[ip],
                     marker = r_markers[ir])
        axes_sigma[i].set_xlim([0,1])
        axes_sigma[i].set_ylim([0,1])
        axes_sigma[i].plot([0, 1],[0,1],'k--')
        axes[1].errorbar(df_class.rmse,
                      df_class.d,
                      xerr = df_class_sem.rmse, yerr = df_class_sem.d,
                      color = pref_colors[ip])
        axes[1].set_xlabel(r'$\Delta$Average RMSE')

        axes[0].set_ylabel(r'$\Delta \|d\|^\prime$')

        sns.despine(ax = axes[1])
        # axes[1].set_aspect('equal', 'box')
        axes[0].legend(markerscale = 1, frameon = False)




df_pp = pd.DataFrame.copy(df_d_all)

# df_pp = df_pp[(df_pp.pref == 45) | (df_pp.pref == 90) | (df_pp.pref == 113)]
df_pp = df_pp[(df_pp.pref == 45) | (df_pp.pref == 90)]
df_pp = df_pp.groupby(['subject','pref','condition']).mean().reset_index()
# df_pp = df_pp[df_pp.r_bin > 3].drop(['subject','pref','r_bin'], axis = 1)
df_pp = df_pp.drop(['subject','pref','r_bin'], axis = 1)

# sns.pairplot(df_pp, hue = 'condition')

# fig,axes = plt.subplots(1,2)

# sns.scatterplot(data = df_d_all, x = 'mu_diff', y = 'd', hue = 'condition', ax = axes[0])
# sns.scatterplot(data = df_d_all, x = 'rmse', y = 'd', hue = 'condition', ax = axes[1])


pg = sns.PairGrid(df_pp,hue = 'condition')
pg.map_diag(sns.kdeplot)
pg.map_upper(sns.regplot)
pg.map_lower(sns.regplot)

#%% Plot change in d' with condition

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


# Find d' for 45 vs 90 for all cells

df_45 = df_resps[df_resps.ori == 45].reset_index(drop=True)
df_90 = df_resps[df_resps.ori == 90].reset_index(drop=True)

df_d = pd.DataFrame(columns=['d','r_bin','pref','subject','condition',
                             'mu_diff', 'rmse'])

df_d['mu_diff'] = df_45.mu - df_90.mu
df_d['rmse'] = np.sqrt((df_45.sigma**2+df_90.sigma**2)/2)

df_d['d'] = df_d.mu_diff/df_d.rmse

df_d['r_bin'] = df_45.r_bin
df_d['pref'] = df_45.pref.astype(int)
df_d['subject'] = df_45.subject
df_d['condition'] = df_45.condition

df_d_all = pd.DataFrame.copy(df_d)
df_d = df_d.groupby(['pref','subject','condition']).mean().reset_index()

pref_colors = sns.hls_palette(len(np.unique(df_resps.pref)))
r_markers = ['X','P','s','v','D']

cond_colors = sns.color_palette('colorblind')[0:2]

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":6,"axes.titlesize":6,
                                        "axes.labelsize":6,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":6,
                                        "ytick.major.size":6,
                                        "patch.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    fig = plt.figure(figsize=(3.75,1.5))
    axes = fig.add_subplot()

    d_range = [-3,3]

    # Loop through and plot each class, pref color, r_bin style
    for ip,p in enumerate(np.unique(df_d.pref)):
        # for ir,r in enumerate(np.unique(df_d.r_bin)):

            # df_class = df_d[np.logical_and(df_d.pref==p,df_d.r_bin==r)]
            df_class = df_d[df_d.pref==p]

            # if ir == 0:
            axes.scatter(df_class[df_class.condition==False].d,
                             df_class[df_class.condition==True].d,
                             color = pref_colors[ip],
                             # marker = r_markers[ir],
                             label = str(int(p)),
                             s = 2)
            # else:
            #     axes.scatter(df_class[df_class.condition==False].d,
            #                   df_class[df_class.condition==True].d,
            #                   color = pref_colors[ip],
            #                   marker = r_markers[ir])
            axes.set_xlim(d_range)
            axes.set_ylim(d_range)
            axes.plot(d_range,d_range,'k--')

            axes.set_xlabel('Naive', color = cond_colors[0])
            axes.set_ylabel('Proficient', color = cond_colors[1])

            sns.despine(ax = axes)
            axes.set_aspect('equal', 'box')
            legend = axes.legend(markerscale = 1, frameon = False, loc = 'upper right',
                                 bbox_to_anchor=(1.5, 1))
            legend.set_title('Ori. pref. (deg)')
            legend._legend_box.align = "left"

    fig.tight_layout()
    df_d_diff = pd.DataFrame(columns=['d','pref'])
    df_d_diff['d'] = df_d[df_d.condition == True].d.to_numpy() - df_d[df_d.condition == False].d.to_numpy()
    df_d_diff['pref'] = df_d[df_d.condition==True].pref.to_numpy()

    plt.figure(figsize=(2.75,1.5))
    d_fig = sns.swarmplot(data = df_d_diff, x = 'pref', y = 'd', s = 2, zorder = 1,
                          palette = 'hls')
    plt.xlabel('Preferred modal orientation (deg)')
    plt.ylabel(r'$\Delta$d$^\prime$')
    sns.despine()
    plt.tight_layout()

    mean_width = 0.5

    for tick, text in zip(d_fig.get_xticks(), d_fig.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # calculate the median value for all replicates of either X or Y
        mean_val = df_d_diff[df_d_diff['pref'].astype(str)==sample_name].d.mean()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(df_d_diff)/len(np.unique(df_d_diff['pref']))
        ci_val = mean_confidence_interval(
            df_d_diff[df_d_diff['pref'].astype(str)==sample_name].d, confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        d_fig.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        d_fig.errorbar(tick,mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)
#%% Plot some individual cell tuning curves as polar plots with modal and mean vector

n_cells = 16

# r_ind = np.where(np.logical_and(df_scm['r_bin'] == 1, (df_scm['trained'] ==True)
#                                 & (df_scm['pref_bin']==45)))[0]

r_ind = np.where((df_scm['r_bin'] == 1) & (df_scm['trained'] == 'Naive')
                                & (df_scm['subject']=='SF180515') & (df_scm['pref_bin']==45))[0]


# r_ind = np.where((df_scm['r_bin'].to_numpy() == 1) & (df_scm['trained'].to_numpy()==True) &
#                  (df_scm['pref_bin'].to_numpy()==45) &
#                  (ori_dict['z_stim_ori'].max(0) > 0.4))[0]

rand_cell = np.random.choice(r_ind,n_cells)

fig, ax = plt.subplots(int(n_cells/4),4, sharex = True, sharey = True, figsize = (8,8))

for i,c in enumerate(rand_cell):

    tc = ori_dict['mean_ori_test'][:-1,c]
    x_r = df_scm['v_x'][c]
    y_r = df_scm['v_y'][c]
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    pref_rad = ori_rad[np.argmax(tc)]
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])
    x_rad,y_rad = np.cos(ori_rad), np.sin(ori_rad)

    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori_rad)])

    ax.flatten()[i].plot(coord[:,0],coord[:,1],'k')
    ax.flatten()[i].arrow(0,0,x_r,y_r, width = 0.015, length_includes_head = True,
              color = 'g')
    ax.flatten()[i].arrow(0,0,np.cos(pref_rad)*tc.max(),np.sin(pref_rad)*tc.max(),
              width = 0.015,
              length_includes_head = True, color = 'k')
    ax.flatten()[i].plot([-1.2,1.2],[0,0],'--k')
    ax.flatten()[i].plot([0,0],[-1.2,1.2],'--k')
    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax.flatten()[i].plot(x_cir,y_cir,'k')
    ax.flatten()[i].axis('off')
    ax.flatten()[i].text(1.3, 0, r'0$\degree$', verticalalignment='center')
    ax.flatten()[i].text(0, 1.3, r'45$\degree$', horizontalalignment='center')
    ax.flatten()[i].text(-1.5, 0, r'90$\degree$', verticalalignment='center')
    ax.flatten()[i].text(0, -1.4, r'135$\degree$', horizontalalignment='center')
    ax.flatten()[i].set_title(str(c))


#%% For all cells with orientation preference 45, show individual examples of
# naive and trained with polar plots

r_bins = np.unique(df_scm.r_bin)

rand_cells = np.array([[np.random.choice(np.where(np.logical_and(
    df_scm.r_bin == r, (df_scm.pref_bin == 45) & (df_scm.trained=='Naive')))[0],1)[0],
    np.random.choice(np.where(np.logical_and(df_scm.r_bin==r,
                    (df_scm.pref_bin == 45) & (df_scm.trained=='Proficient')))[0],1)[0]]
    for r in r_bins])

# rand_cells = np.array([[13445,38636], [20819, 17477], [32952,10251],
                        # [12845, 29864], [32904,17178]])


fig, ax = plt.subplots(2,5, figsize = (12,6))

for i in range(len(rand_cells)):

    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2

    tc = ori_dict['mean_ori_test'][:-1,rand_cells[i,0]]
    x_r = df_scm['v_x'][rand_cells[i,0]]
    y_r = df_scm['v_y'][rand_cells[i,0]]
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    pref_rad = ori_rad[np.argmax(tc)]
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])
    x_rad,y_rad = np.cos(ori_rad), np.sin(ori_rad)

    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori_rad)])

    ax[0,i].plot(coord[:,0],coord[:,1],'b')
    ax[0,i].arrow(0,0,x_r,y_r, width = 0.015, length_includes_head = True,
              color = 'g', head_width = 0.015*6)
    ax[0,i].arrow(0,0,np.cos(pref_rad)*tc.max(),np.sin(pref_rad)*tc.max(),
              width = 0.015,
              length_includes_head = True, color = 'r', head_width = 0.015*6)
    ax[0,i].plot([-1.2,1.2],[0,0],'--k')
    ax[0,i].plot([0,0],[-1.2,1.2],'--k')
    # ax[0,i].set_title(str(rand_cells[i,0]))

    ax[0,i].plot(x_cir,y_cir,'k')
    ax[0,i].axis('off')
    ax[0,i].text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax[0,i].text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax[0,i].text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax[0,i].text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax[0,i].axis('equal')
    if i == 2:
        ax[0,i].set_title('Naïve', color = trained_cp[0])

    tc = ori_dict['mean_ori_test'][:-1,rand_cells[i,1]]
    x_r = df_scm['v_x'][rand_cells[i,1]]
    y_r = df_scm['v_y'][rand_cells[i,1]]
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    pref_rad = ori_rad[np.argmax(tc)]
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])
    x_rad,y_rad = np.cos(ori_rad), np.sin(ori_rad)

    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori_rad)])

    ax[1,i].plot(coord[:,0],coord[:,1],'b')
    ax[1,i].arrow(0,0,x_r,y_r, width = 0.015, length_includes_head = True,
              color = 'g', head_width = 0.015*6)
    ax[1,i].arrow(0,0,np.cos(pref_rad)*tc.max(),np.sin(pref_rad)*tc.max(),
              width = 0.015,
              length_includes_head = True, color = 'r', head_width = 0.015*6)
    ax[1,i].plot([-1.2,1.2],[0,0],'--k')
    ax[1,i].plot([0,0],[-1.2,1.2],'--k')
    # ax[1,i].set_title(str(rand_cells[i,1]))

    ax[1,i].plot(x_cir,y_cir,'k')
    ax[1,i].axis('off')
    ax[1,i].text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax[1,i].text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax[1,i].text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax[1,i].text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax[1,i].axis('equal')
    if i == 2:
        ax[1,i].set_title('Proficient', color = trained_cp[1])


#%% Plot polar tuning curves

# mpl.font_manager.get_font(r'C:\Users\Samuel\Appdata\Local\Microsoft\Windows\Fonts\Helvetica-Light-587EBE5A59211.ttf')


r_bins = np.unique(df_scm.r_bin)

n_cells = 40

i_ori = 45
i_cond = 'Naive'
i_color = 2
i_subject = 'SF180515'
i_r = 0

# ind = (df_scm.r_bin == i_r) & (df_scm.pref_bin==i_ori) & (df_scm.trained==i_cond) & (df_scm.subject==i_subject)

ind = (df_scm.r_bin == i_r) & (df_scm.pref_ori_train==i_ori) & (df_scm.trained==i_cond) & (df_scm.subject==i_subject)


rand_cells = np.random.choice(np.where(ind)[0],n_cells,
                              replace= False)

ori_cp = sns.hls_palette(8)

# rand_cells = np.array([22518])

# rand_cells = np.array([17131, 37580, 29906, 26827, 29906])

# rand_cells = np.array([19663, 21786, 6727, 1436, 33857, 21618])

circle_rad = 1.4

for c in rand_cells:

    plt.figure()
    ax = plt.subplot(111, aspect = 'equal')

    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*circle_rad, np.sin(ori_rad)*circle_rad

    tc = ori_dict['mean_ori_test'][:-1,c]
    x_r = df_scm['v_x'][c]
    y_r = df_scm['v_y'][c]
    ori_rad = np.arange(0,2*np.pi,2*np.pi/8)
    pref_rad = ori_rad[np.argmax(tc)]
    tc = np.append(tc,tc[0])
    ori_rad = np.append(ori_rad,ori_rad[0])
    x_rad,y_rad = np.cos(ori_rad), np.sin(ori_rad)

    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori_rad)])

    ax.plot(coord[:,0],coord[:,1],color = ori_cp[i_color], zorder = 2)
    ax.arrow(0,0,x_r,y_r, width = 0.015, length_includes_head = True,
              color = 'k', head_width = 0.015*6, zorder = 3)
    # ax.arrow(0,0,np.cos(pref_rad)*tc.max(),np.sin(pref_rad)*tc.max(),
    #           width = 0.015,
    #           length_includes_head = True, color = 'r', head_width = 0.015*6)
    ax.plot(np.cos(pref_rad)*tc.max(),np.sin(pref_rad)*tc.max(), 'ro', zorder = 3)
    ax.plot([-circle_rad,circle_rad],[0,0],'--k',zorder = 1)
    ax.plot([0,0],[-circle_rad,circle_rad],'--k',zorder = 1)

    ax.plot(x_cir,y_cir,'k', zorder = 1)
    ax.axis('off')
    ax.text(circle_rad+0.14, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, circle_rad+0.15, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.text(-circle_rad - 0.18, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, -circle_rad - 0.15, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.axis('equal')
    ax.set_title(str(c))
    plt.show()

    # plt.waitforbuttonpress(timeout = 20)
    # plt.close('all')


#%% Plot simulated non-linear surpression of task stimuli

from scipy.stats import vonmises,norm

colors = sns.color_palette('colorblind')

nl_exp = 2

sigma = np.deg2rad(60)

x = np.linspace(-np.pi,np.pi, 9)

y = norm(0,sigma).pdf(x)
y_scaled = np.copy(y)
# y_scale = y.max()
y_scale = 0.9
y_scaled /= y_scale
# x = np.linspace(0,2*np.pi,9)

ori_pref = 45
x = x + np.deg2rad(2*ori_pref)


coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(y_scaled,x)])


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":6,"axes.titlesize":6,
                                        "axes.labelsize":6,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":6,
                                        "ytick.major.size":6,
                                        "patch.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(1,1)
    ax[0].plot(coord[:,0],coord[:,1], zorder = 2, color = colors[0])

    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax[0].plot(x_cir,y_cir,'k', zorder = 1)
    ax[0].axis('off')
    ax[0].text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center',fontsize=5)
    ax[0].text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center',fontsize=5)
    ax[0].text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center',fontsize=5)
    ax[0].text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center',fontsize=5)
    ax[0].plot([-1.2,1.2],[0,0],'--k',zorder = 1)
    ax[0].plot([0,0],[-1.2,1.2],'--k',zorder = 1)
    ax[0].axis('equal')

    y_transform = np.copy(y)
    y_transform[x==np.deg2rad(90)] **= nl_exp
    y_transform[x==np.deg2rad(180)] **= nl_exp

    y_transform /= y_scale

    coord_transform = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(y_transform,x)])

    ax[1].plot(coord[:,0],coord[:,1], zorder = 2, color = colors[0])
    ax[1].plot(coord_transform[:,0],coord_transform[:,1], zorder = 2, color = colors[1])

    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax[1].plot(x_cir,y_cir,'k', zorder = 1)
    ax[1].axis('off')
    ax[1].text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center',fontsize=5)
    ax[1].text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center',fontsize=5)
    ax[1].text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center',fontsize=5)
    ax[1].text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center',fontsize=5)
    ax[1].plot([-1.2,1.2],[0,0],'--k',zorder = 1)
    ax[1].plot([0,0],[-1.2,1.2],'--k',zorder = 1)
    ax[1].axis('equal')

    # fig.tight_layout()


    fig_nl, ax_nl = plt.subplots(1,1)
    fig_nl.set_size_inches(1.5,1.5)
    x = np.linspace(0,1,100)
    ax_nl.plot(x,x**nl_exp, color = colors[1])
    ax_nl.plot(x,x**1.2, color = colors[0])
    ax_nl.set_xlabel('Input current', labelpad = 1)
    ax_nl.set_ylabel('Firing rate', labelpad = 1)
    ax_nl.set_xlim([0,1])
    ax_nl.set_ylim([0,1])
    ax_nl.set_xticks([0,0.5,1])
    ax_nl.set_yticks([0,0.5,1])
    ax_nl.axis('equal')
    sns.despine(ax = ax_nl)
    fig_nl.tight_layout()


#%% Plot simulated non-linear surpression of task stimuli - linear plots

from scipy.stats import vonmises,norm

colors = sns.color_palette('colorblind')

nl_exp = 3

sigma = np.deg2rad(50)

x = np.linspace(-np.pi,np.pi, 9)

y = norm(0,sigma).pdf(x)
y = np.roll(y,-2)
y_scale = 1.143
y /= y_scale
y_max = y.max()
x = np.linspace(0,180,9)


# coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(y_scaled,x)])


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

    fig,ax = plt.subplots(1,2, sharex = True, sharey = True)
    fig.set_size_inches(2,1)
    ax[0].plot(x,y/y_max, zorder = 2, color = colors[0])
    # ax[0].set_xticks(np.linspace(0,180,9))
    # ax[0].set_xticklabels(np.ceil(np.linspace(0,180,9)).astype(int))
    ax[0].set_xticks([45,90])
    ax[0].set_xticklabels([r'45$\degree$', r'90$\degree$'])
    sns.despine(ax=ax[0],left=True)
    ax[0].set_yticks([])
    # ax[0].set_xlabel(r'Stimulus orientation ($\degree$)')

    y_transform = np.copy(y)
    y_transform[x==45] **= nl_exp
    y_transform[x==90] **= nl_exp

    y_transform /= y_max

    ax[1].plot(x,y/y_max, zorder = 2, color = colors[0])
    ax[1].plot(x,y_transform, zorder = 2, color = colors[1])
    for i,o in enumerate(x):
        if (o == 45) | (o == 90):
            ax[1].plot(o,y_transform[i], '.', color = 'red', markersize = 1)
        else:
            ax[1].plot(o,y_transform[i], '.', color = 'black', markersize = 1)
    # ax[1].set_xticks(np.linspace(0,180,9))
    # ax[1].set_xticklabels(np.ceil(np.linspace(0,180,9)).astype(int))
    sns.despine(ax=ax[1],left=True)
    ax[1].set_yticks([])
    # ax[1].set_xlabel(r'Stimulus orientation ($\degree$)')

    fig.tight_layout()


    fig_nl, ax_nl = plt.subplots(1,1)
    fig_nl.set_size_inches(1.5,1.5)
    x = np.linspace(0,1,100)
    ax_nl.plot(x,x**nl_exp, color = 'red')
    ax_nl.plot(x,x**1.2, color = 'black')
    ax_nl.set_xlabel('Input current', labelpad = 1)
    ax_nl.set_ylabel('Firing rate', labelpad = 1)
    ax_nl.set_xlim([0,1])
    ax_nl.set_ylim([0,1])
    ax_nl.set_xticks([0,0.5,1])
    ax_nl.set_yticks([0,0.5,1])
    ax_nl.axis('equal')
    sns.despine(ax = ax_nl)
    fig_nl.tight_layout()


#%% Generate tuning curves for orientation preferences and selectivities

from scipy.stats import norm


sigmas = np.deg2rad(np.linspace(20,70,5))
sigmas = sigmas[[0,2,4]]

x = np.linspace(-np.pi,np.pi, 9)

t_curves = np.zeros((len(x),len(x),len(sigmas)))

for si,s in enumerate(sigmas):
    t_curves[:,:,si] = norm(0,s).pdf(x)
    for i in range(9):
        t_curves[i,:,si] = np.roll(t_curves[i,:,si],i-4)

# t_curves = np.flip(t_curves,0)
# t_curves = np.flip(t_curves,2)
t_curves /= t_curves.max()

# FI curves and response

naive_fi = np.repeat(np.linspace(t_curves.min(),t_curves.max(),9)[None,:],9,axis=0)

prof_fi = np.copy(naive_fi)

prof_fi[[2,4],:] **= 2

prof_resp = np.copy(t_curves)
prof_resp[[2,4],:] **= 2

cmap = 'viridis'

f1,ax1 = plt.subplots(2,4, figsize = (8,4))
# f1.delaxes(ax1[1,3])

for i in range(3):

    ax1[0,i].imshow(t_curves[...,i]/t_curves[...,i].max(),vmin=0, vmax = 1, cmap = cmap)
    ax1[0,i].set_xticks([])
    ax1[0,i].set_yticks([])

    ax1[0,i].set_ylabel('Stimulus orientation')
    ax1[0,i].set_xlabel('Preferred orientation')
    ax1[1,i].imshow(t_curves[...,i]/t_curves[...,i].max(),vmin=0, vmax = 1, cmap = cmap)
    ax1[1,i].set_xticks([])
    ax1[1,i].set_yticks([])
    ax1[1,i].set_ylabel('Stimulus orientation')
    ax1[1,i].set_xlabel('Preferred orientation')

ax1[0,3].imshow(naive_fi,vmin=t_curves.min(), vmax = t_curves.max(), cmap = cmap)
ax1[0,3].set_xticks([])
ax1[0,3].set_yticks([])
ax1[0,3].set_ylabel('Stimulus orientation')
ax1[0,3].set_xlabel('Input')
ax1[1,3].plot(t_curves[:,2,0],'k--')
ax1[1,3].plot(t_curves[:,2,2],'k-')
ax1[1,3].set_xticks([])
ax1[1,3].set_yticks([])
ax1[1,3].set_xlabel('Stimulus orientation')
ax1[1,3].set_ylabel('Response')

sns.despine(ax=ax1[1,3])


f1.tight_layout()

f2,ax2 = plt.subplots(2,4,figsize = (8,4))
# f2.delaxes(ax2[1,3])

for i in range(3):

    ax2[0,i].imshow(t_curves[...,i]/t_curves[...,i].max(),vmin=0, vmax = 1, cmap = cmap)
    ax2[0,i].set_xticks([])
    ax2[0,i].set_yticks([])
    ax2[0,i].set_ylabel('Stimulus orientation')
    ax2[0,i].set_xlabel('Preferred orientation')
    ax2[1,i].imshow(prof_resp[...,i]/prof_resp[...,i].max(),vmin=0, vmax = 1, cmap = cmap)
    ax2[1,i].set_xticks([])
    ax2[1,i].set_yticks([])
    ax2[1,i].set_ylabel('Stimulus orientation')
    ax2[1,i].set_xlabel('Preferred orientation')

ax2[0,3].imshow(prof_fi,vmin=t_curves.min(), vmax = t_curves.max(), cmap = cmap)
ax2[0,3].set_xticks([])
ax2[0,3].set_yticks([])
ax2[0,3].set_ylabel('Stimulus orientation')
ax2[0,3].set_xlabel('Input')
ax2[1,3].plot(prof_resp[:,2,0],'k--')
ax2[1,3].plot(prof_resp[:,2,2],'k-')
ax2[1,3].set_xticks([])
ax2[1,3].set_yticks([])
ax2[1,3].set_xlabel('Stimulus orientation')
ax2[1,3].set_ylabel('Response')

sns.despine(ax=ax2[1,3])

f2.tight_layout()


#%% Plot lots of FI curves with different levels of modulation

import husl

# colors = sns.color_palette('colorblind')

n_mc = 8

mc = np.linspace(1.4,4,n_mc)
mc = np.insert(mc,0,1.2)

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

    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    fig_nl, ax_nl = plt.subplots(1,1, figsize = (1.185,1.185))
    # fig_nl.set_size_inches(1.5,1.5)

    x = np.linspace(0,1,100)
    # ax_nl.plot(x,x**1.2, color = colors[0])

    # pro_colors = sns.dark_palette(colors[1],n_mc,reverse=False)
    # pro_colors = sns.diverging_palette(husl.rgb_to_husl(*colors[0])[0],
    #                                    husl.rgb_to_husl(*colors[1])[0], n = n_mc+1)

    pro_colors = sns.dark_palette('red', n_mc+1)

    for i,m in enumerate(mc):

        ax_nl.plot(x,x**m, color = pro_colors[i])

    # ax_nl.set_xlabel('Input current', labelpad = 1)
    # ax_nl.set_ylabel('Firing rate', labelpad = 1)
    ax_nl.set_xlim([-0.1,1.3])
    ax_nl.set_ylim([-0.1,1.3])
    ax_nl.set_xticks([0,0.5,1])
    ax_nl.set_yticks([0,0.5,1])
    ax_nl.axis('equal')
    sns.despine(ax = ax_nl, trim = True)
    fig_nl.tight_layout()
    # savefile = join(results_dir, 'Figures', 'Draft', 'nonlinear_trans_brain_state.svg')
    fig_nl.savefig(savefile, format = 'svg')

#%% Fit tuning curves then find tanget at each stimulus orientation

uni_pref = np.unique(df_scm.pref_bin.to_numpy())

stim_slope = np.zeros((len(df_scm),len(uni_pref)))

r_cond = np.zeros(len(df_scm))

for i,c in enumerate(ori_dict['mean_ori_test'][:-1,:].T):

    print('Cell ' + str(i) + ' of ' + str(len(df_scm)))

    # c /= c.max()
    r_cond_stim = np.zeros(len(uni_pref))

    for si,s in enumerate(uni_pref):
        c_shift = np.roll(c,3-si)
        stim_shift = np.roll(uni_pref,3-si)
        stim_shift -= s
        stim_shift[stim_shift>90] -= 180
        stim_shift[stim_shift<=-90] += 180
        stim_shift = np.insert(stim_shift,0,-90)
        c_shift = np.insert(c_shift,0,c_shift[-1])

        tc_fit,r_cond_stim[si] = np.polyfit(stim_shift,c_shift,10)
        tc = np.poly1d(tc_fit)
        # tc = interpolate.interp1d(stim_shift,c_shift,kind='cubic')
        x = np.linspace(stim_shift[0],stim_shift[-1],10000)
        y = tc(x)
        i0 = np.argmin(np.abs(x))
        x1 = x[i0:i0+2]
        y1 = y[i0:i0+2]
        stim_slope[i,si] = np.abs(np.diff(y1)/np.diff(x1))

    r_cond = r_cond_stim.mean()

#%% Find "slope" by just taking point at stim and another point closest to the
# cells preferred stim

uni_pref = np.unique(df_scm.pref_bin.to_numpy())

stim_slope = np.zeros((len(df_scm),len(uni_pref)))

pref_type = 'mean'

for i,c in enumerate(ori_dict['mean_ori_test'][:-1,:].T):

    if pref_type == 'mean':
        ori_pref = df_scm.loc[i,'pref_bin']
    elif pref_type == 'modal':
        ori_pref = df_scm.loc[i,'pref_ori_train']

    print('Cell ' + str(i))

    for si,s in enumerate(uni_pref):
        c_shift = np.roll(c,3-si)
        stim_shift = np.roll(uni_pref,3-si)
        stim_shift = stim_shift - s
        stim_shift[stim_shift>90] -= 180
        stim_shift[stim_shift<=-90] += 180
        stim_shift = np.insert(stim_shift,0,-90)
        c_shift = np.insert(c_shift,0,c_shift[-1])
        ori_pref_shift =  ori_pref - s
        if ori_pref_shift > 90:
            ori_pref_shift -= 180
        elif ori_pref_shift <=-90:
            ori_pref_shift += 180

        ind = np.where(stim_shift == ori_pref_shift)[0]
        if ind[0] < 4:
            # stim_slope[i,si] = np.abs((c_shift[3]-c_shift[4])/22.5)
            stim_slope[i,si] = (c_shift[4]-c_shift[3])/22.5

        elif ind[0] > 4:
            # stim_slope[i,si] = np.abs((c_shift[5]-c_shift[4])/22.5)
            stim_slope[i,si] = (c_shift[5]-c_shift[4])/22.5

        else:
            # stim_slope[i,si] = ((c_shift[3]-c_shift[4])/22.5
            #                     - (c_shift[5]-c_shift[4])/22.5)
            # stim_slope[i,si] = np.abs(stim_slope[i,si])
            # stim_slope[i,si] = np.abs((c_shift[3] - c_shift[5]))/45
            stim_slope[i,si] = (c_shift[5] - c_shift[3])/45


#%% Find "slope" by just taking point at stim and another point closest to the
# cells preferred stim

uni_pref = np.unique(df_scm.pref_bin.to_numpy())

stim_slope = np.zeros((len(df_scm),len(uni_pref)))

for i in range(ori_dict['mean_ori_test'].shape[1]):
    tc = ori_dict['mean_ori_test'][:-1,i]

    # Wrap
    first = tc[0]
    last = tc[-1]
    tc_long = np.insert(tc,0,last)
    tc_long = np.append(tc_long,first)

    for si,s in enumerate(uni_pref):
        stim_slope[i,si] = (tc_long[si] - tc_long[si+2])/45


#%% Interpolate using zero-padding in frequency domain

uni_pref = np.unique(df_scm.pref_bin.to_numpy())

stim_slope = np.zeros((len(df_scm),len(uni_pref)))

n_samples = 40000

for i in range(ori_dict['mean_ori_test'].shape[1]):
    print('Cell ' + str(i))
    tc = np.copy(ori_dict['mean_ori_test'][:-1,i])
    f = np.fft.fft(tc)
    # Zero-padding for interpolation
    f = np.concatenate((f[:4],np.zeros(n_samples-len(tc)),f[4:]))
    y = np.fft.ifft(f)
    y = y / len(tc) * len(y)
    x = np.linspace(0,180,len(f))

    for si,s in enumerate(uni_pref):
        i0 = np.argmin(np.abs(x-s))
        x1 = x[i0:i0+2]
        y1 = y[i0:i0+2]
        stim_slope[i,si] = np.abs(np.diff(y1)/np.diff(x1))

#%%

group_by = ['subject', 'trained', 'pref_bin']

df_45 = pd.DataFrame.copy(df_scm[['pref_bin','trained','subject']])
df_90 = pd.DataFrame.copy(df_scm[['pref_bin','trained','subject']])
df_68 = pd.DataFrame.copy(df_scm[['pref_bin','trained','subject']])
df_0 = pd.DataFrame.copy(df_scm[['pref_bin','trained','subject']])
df_135 = pd.DataFrame.copy(df_scm[['pref_bin','trained','subject']])

df_45['pref_bin'] = [22.5 if s==23 else 67.5 if s==68 else 112.5 if s==113 else 157.5 if s==158 else s for s in df_45.pref_bin.to_numpy()]
df_90['pref_bin'] = [22.5 if s==23 else 67.5 if s==68 else 112.5 if s==113 else 157.5 if s==158 else s for s in df_45.pref_bin.to_numpy()]
df_68['pref_bin'] = [22.5 if s==23 else 67.5 if s==68 else 112.5 if s==113 else 157.5 if s==158 else s for s in df_45.pref_bin.to_numpy()]
df_0['pref_bin'] = [22.5 if s==23 else 67.5 if s==68 else 112.5 if s==113 else 157.5 if s==158 else s for s in df_45.pref_bin.to_numpy()]
df_135['pref_bin'] = [22.5 if s==23 else 67.5 if s==68 else 112.5 if s==113 else 157.5 if s==158 else s for s in df_45.pref_bin.to_numpy()]

df_45['slope'] = stim_slope[:,2]
df_45['pref_bin'] = df_45['pref_bin'].astype(float)
df_45['pref_bin'] = df_45['pref_bin'] - 45
df_45.loc[df_45.pref_bin < -90, 'pref_bin'] = df_45.loc[df_45.pref_bin < -90, 'pref_bin'] + 180
df_45.loc[df_45.pref_bin > 90, 'pref_bin'] = df_45.loc[df_45.pref_bin > 90, 'pref_bin'] - 180

df_45 = df_45.groupby(group_by).mean().reset_index()
wrap = pd.DataFrame.copy(df_45[df_45.pref_bin == 90])
wrap['pref_bin'] = wrap['pref_bin'].apply(lambda x : -x)
df_45 = pd.concat([df_45, wrap], ignore_index=True)

df_90['slope'] = stim_slope[:,4]
df_90['pref_bin'] = df_90['pref_bin'].astype(float)
df_90['pref_bin'] = df_90['pref_bin'] - 90
df_90.loc[df_90.pref_bin < -90, 'pref_bin'] = df_90.loc[df_90.pref_bin < -90, 'pref_bin'] + 180
df_90.loc[df_90.pref_bin > 90, 'pref_bin'] = df_90.loc[df_90.pref_bin > 90, 'pref_bin'] - 180

df_90 = df_90.groupby(group_by).mean().reset_index()
wrap = pd.DataFrame.copy(df_90[df_90.pref_bin == -90])
wrap['pref_bin'] = wrap['pref_bin'].apply(lambda x : -x)
df_90 = pd.concat([df_90, wrap], ignore_index=True)

df_slope = pd.concat([df_45, df_90], ignore_index=True)
df_slope['slope_pos'] = '45/90'

df_68['slope'] = stim_slope[:,3]
df_68['pref_bin'] = df_68['pref_bin'].astype(float)
df_68['pref_bin'] = df_68['pref_bin'] - 67.5
df_68.loc[df_68.pref_bin < -90, 'pref_bin'] = df_68.loc[df_68.pref_bin < -90, 'pref_bin'] + 180
df_68.loc[df_68.pref_bin > 90, 'pref_bin'] = df_68.loc[df_68.pref_bin > 90, 'pref_bin'] - 180

df_68 = df_68.groupby(group_by).mean().reset_index()
wrap = pd.DataFrame.copy(df_68[df_68.pref_bin == 90])
wrap['pref_bin'] = wrap['pref_bin'].apply(lambda x : -x)
df_68 = pd.concat([df_68, wrap], ignore_index=True)
df_68['slope_pos'] = '68'


# if pref_type == 'modal':
#     df_0 = df_0.drop('pref_bin', axis = 1)
#     df_0 = df_0.rename(columns={'pref_ori_train' : 'pref_bin'})

#     df_135 = df_135.drop('pref_bin', axis = 1)
#     df_135 = df_135.rename(columns={'pref_ori_train' : 'pref_bin'})

df_0['slope'] = stim_slope[:,0]
df_0['pref_bin'] = df_0['pref_bin'].astype(float)
df_0.loc[df_0.pref_bin > 90, 'pref_bin'] = df_0.loc[df_0.pref_bin > 90, 'pref_bin'] - 180

df_0 = df_0.groupby(group_by).mean().reset_index()
wrap = pd.DataFrame.copy(df_0[df_0.pref_bin == 90])
wrap['pref_bin'] = wrap['pref_bin'].apply(lambda x : -x)
df_0 = pd.concat([df_0, wrap], ignore_index=True)

df_135['slope'] = stim_slope[:,6]
df_135['pref_bin'] = df_135['pref_bin'].astype(float)
df_135['pref_bin'] = df_135['pref_bin'] - 135
df_135.loc[df_135.pref_bin < -90, 'pref_bin'] = df_135.loc[df_135.pref_bin < -90, 'pref_bin'] + 180
df_135.loc[df_135.pref_bin > 90, 'pref_bin'] = df_135.loc[df_135.pref_bin > 90, 'pref_bin'] - 180

df_135 = df_135.groupby(group_by).mean().reset_index()
wrap = pd.DataFrame.copy(df_135[df_135.pref_bin == -90])
wrap['pref_bin'] = wrap['pref_bin'].apply(lambda x : -x)
df_135 = pd.concat([df_135, wrap], ignore_index=True)

df_orth_slope = pd.concat([df_0, df_135], ignore_index=True)
df_orth_slope['slope_pos'] = '0/135'

# Make pref_bins consistent

# df_slope.loc[df_slope.pref_bin==-22,'pref_bin'] = -23
# df_slope.loc[df_slope.pref_bin==-67,'pref_bin'] = -68
# df_68.loc[df_68.pref_bin==-22.5,'pref_bin'] = -23
# df_68.loc[df_68.pref_bin==22.5,'pref_bin'] = 23
# df_orth_slope.loc[df_orth_slope.pref_bin==-22,'pref_bin'] = -23
# df_orth_slope.loc[df_orth_slope.pref_bin==-67,'pref_bin'] = -68


# df_slope = df_slope[np.abs(df_slope.pref_bin) <= 90]
# wrap = df_slope.loc[df_slope.pref_bin == 112.5]
# wrap['pref_bin'] = wrap['pref_bin'].apply(lambda x: -x)
# df_slope = df_slope.append(wrap)

#%% Stats on tc slopes
from scipy.stats import ttest_rel,ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Mixed effects model

df_stats = pd.concat([df_slope, df_68, df_orth_slope], ignore_index=True)
df_stats['pref_bin'] = np.abs(df_stats.pref_bin)


# 45/90 vs 68

ind = (df_stats.slope_pos=='45/90') | (df_stats.slope_pos=='68') & (df_stats.pref_bin==23)

md = smf.mixedlm("slope ~ C(trained) * C(slope_pos)", df_stats[ind], groups=df_stats[ind]['subject'], re_formula="~C(trained)")

md_f = md.fit()

print(md_f.summary())

mds = smf.mixedlm("slope ~ C(trained) * C(slope_pos)", df_stats[ind], groups=df_stats[ind]['subject'], re_formula="~C(trained) * C(slope_pos)")

mds_f = mds.fit()

print(mds_f.summary())

ll_simple = md_f.llf
ll_complex = mds_f.llf

# Compute the likelihood ratio test statistic
lr_stat = -2 * (ll_simple - ll_complex)

# The degrees of freedom is the difference in the number of parameters between the models
df_diff = (mds_f.df_modelwc - md_f.df_modelwc)

# The p-value is the right-tail probability of the chi-squared distribution
p_value = stats.chi2.sf(lr_stat, df_diff)

# Output the results
print(f'Likelihood Ratio Test Statistic: {lr_stat}')
print(f'P-value: {p_value}')
print(f'Degrees of Freedom Difference: {df_diff}')


#%%

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


colors = sns.color_palette('colorblind')[slice(-1,-4,-1)]


# f,ax = plt.subplots(1,1)

df_stats_task = pd.DataFrame.copy(df_slope)
df_stats_task['pref_bin'] = np.abs(df_stats_task.pref_bin) # abs for average cells with adjacent pref
df_stats_task['slope'] = np.abs(df_stats_task.slope) # Since we have avearged by pref_diff already, we can take abs
df_stats_task = df_stats_task.groupby(['subject','trained','pref_bin'], as_index=False)['slope'].mean().reset_index()

# ind_n = np.logical_and(df_stats_task.trained == False, df_stats_task.pref_bin == 23)
# ind_t = np.logical_and(df_stats_task.trained, df_stats_task.pref_bin == 23)

# p_task = ttest_rel(df_stats_task[ind_n].slope,df_stats_task[ind_t].slope)

# sns.scatterplot(ax = ax, x = df_stats_task[ind_n].slope.to_numpy(), y = df_stats_task[ind_t].slope.to_numpy(),
# color = colors[0])

df_stats_68 = pd.DataFrame.copy(df_68)
df_stats_68['pref_bin'] = np.abs(df_stats_68.pref_bin)
df_stats_68['slope'] = np.abs(df_stats_68.slope)
df_stats_68 = df_stats_68.groupby(['subject','trained','pref_bin'], as_index=False)['slope'].mean().reset_index()

# ind_n = np.logical_and(df_stats_68.trained == False, df_stats_68.pref_bin == 23)
# ind_t = np.logical_and(df_stats_68.trained, df_stats_68.pref_bin == 23)

# p_68 = ttest_rel(df_stats_68[ind_n].slope,df_stats_68[ind_t].slope)

# sns.scatterplot(ax = ax, x = df_stats_68[ind_n].slope.to_numpy(), y = df_stats_68[ind_t].slope.to_numpy(),
# color = colors[1])

df_stats_orth = pd.DataFrame.copy(df_orth_slope)
df_stats_orth['pref_bin'] = np.abs(df_stats_orth.pref_bin)
df_stats_orth['slope'] = np.abs(df_stats_orth.slope)
df_stats_orth = df_stats_orth.groupby(['subject','trained','pref_bin'], as_index=False)['slope'].mean().reset_index()

# ind_n = np.logical_and(df_stats_orth.trained == False, df_stats_orth.pref_bin == 23)
# ind_t = np.logical_and(df_stats_orth.trained, df_stats_orth.pref_bin == 23)

# p_ortho = ttest_rel(df_stats_orth[ind_n].slope,df_stats_orth[ind_t].slope)

# sns.scatterplot(ax = ax, x = df_stats_orth[ind_n].slope.to_numpy(), y = df_stats_orth[ind_t].slope.to_numpy(),
# color = colors[2])

# ax.set_xlim([0.01,0.025])
# ax.set_ylim([0.01,0.025])
# ax.plot([0.01,0.025],[0.01,0.025],'--k')


diff_task = (df_stats_task[np.logical_and(df_stats_task.trained=='Proficient', df_stats_task.pref_bin==22.5)].slope.to_numpy() -
             df_stats_task[np.logical_and(df_stats_task.trained=='Naive', df_stats_task.pref_bin==22.5)].slope.to_numpy())

diff_68 = (df_stats_68[np.logical_and(df_stats_68.trained=='Proficient', df_stats_68.pref_bin==22.5)].slope.to_numpy() -
             df_stats_68[np.logical_and(df_stats_68.trained=='Naive', df_stats_68.pref_bin==22.5)].slope.to_numpy())

diff_orth = (df_stats_orth[np.logical_and(df_stats_orth.trained=='Proficient', df_stats_orth.pref_bin==22.5)].slope.to_numpy() -
             df_stats_orth[np.logical_and(df_stats_orth.trained=='Naive', df_stats_orth.pref_bin==22.5)].slope.to_numpy())


stim_label = np.concatenate([np.repeat('45 and 90',5), np.repeat('68',5), np.repeat('135 and 0',5)])

diff = np.concatenate([diff_task,diff_68,diff_orth])

df_diff = pd.DataFrame({'stim' : stim_label,
                        'slope' : diff})

# Calculate p-values

pvals = [ttest_rel(df_diff[df_diff.stim=='45 and 90'].slope.to_numpy(), df_diff[df_diff.stim=='68'].slope.to_numpy())[1]]
pvals.append(ttest_rel(df_diff[df_diff.stim=='45 and 90'].slope.to_numpy(), df_diff[df_diff.stim=='135 and 0'].slope.to_numpy())[1])
pvals.append(ttest_rel(df_diff[df_diff.stim=='68'].slope.to_numpy(), df_diff[df_diff.stim=='135 and 0'].slope.to_numpy())[1])


mu = []
mu.append(df_diff[df_diff.stim=='45 and 90'].slope.to_numpy().mean())
mu.append(df_diff[df_diff.stim=='68'].slope.to_numpy().mean())
mu.append(df_diff[df_diff.stim=='135 and 0'].slope.to_numpy().mean())

sem = []
sem.append(df_diff[df_diff.stim=='45 and 90'].slope.to_numpy().std()/np.sqrt(5))
sem.append(df_diff[df_diff.stim=='68'].slope.to_numpy().std()/np.sqrt(5))
sem.append(df_diff[df_diff.stim=='135 and 0'].slope.to_numpy().std()/np.sqrt(5))



# pvals = multipletests(pvals,method='hs')[1]

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
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):


    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    f,ax = plt.subplots(1,1,figsize = (1,1))

    # sns.swarmplot(ax = ax, data = df_diff, x = 'stim', y = 'slope', hue = 'stim', palette = colors,
    #               edgecolor = 'k', linewidth = 0.5, zorder = 1, s = 3)
    sns.swarmplot(ax = ax, data = df_diff, x = 'stim', y = 'slope', facecolor='k',
                edgecolor = 'k', linewidth = 0.5, zorder = 1, s = 3)

    # ax.legend_.remove()

    ax.set_ylim([-0.01,0.012])
    ax.set_yticks(np.linspace(-0.01,0.01,6))

    # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.4

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # calculate the median value for all replicates of either X or Y
        mean_val = diff[stim_label==sample_name].mean()
        std_val = diff[stim_label==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        n_points = len(stim_label)/len(np.unique(stim_label))
        ci_val = mean_confidence_interval(
            diff[stim_label==sample_name], confidence = 0.68)
        ax.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--', zorder = 2)
        ax.errorbar(tick,mean_val, ci_val, ecolor = 'k',
                          capsize = 3, capthick=0.5, zorder = 2)

    sns.despine(trim=True)
    ax.set_xticklabels([r'45$\degree$ and 90$\degree$', r'68$\degree$', r'135$\degree$ and 0$\degree$'])
    ax.set_xlabel('Stimulus orientation')
    ax.set_ylabel(r'$\Delta$tuning curve slope (resp./$\degree$)')

    ax = sig_label(ax,(0,1),0.1,0.05,'*',)
    ax = sig_label(ax,(0,2),0.25,0.05,'*',)
    ax = sig_label(ax,(1,2),0.1,0.05,'ns')

    f.savefig(join(fig_save_dir,'tc_slope_change.svg'), format = 'svg')

#%%

x_labels = np.append(-np.ceil(np.arange(90,0,-22.5)),
                      np.append(0,np.ceil(np.arange(22.5,90+22.5,22.5))))
x_ticks = np.copy(x_labels)

x_labels = [str(int(x))+r'$\degree$' for x in x_labels]


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
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):


    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    # df_plot = pd.DataFrame.copy(df_slope[np.logical_and(df_slope.r_bin > 1, df_slope.r_bin<4)])
    df_plot = pd.DataFrame.copy(df_slope)
    df_plot['slope'] = df_plot.slope.abs()
    # df_plot['pref_bin'] = np.abs(df_plot['pref_bin'])

    plt.figure(figsize=(1.4,1.4))
    fig_task = sns.lineplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
                 palette = 'colorblind', markers = ['o','o'], style = 'trained',
                 style_order = ['Proficient','Naive'], markeredgecolor=None, markersize = 2,
                 errorbar = ('se',1), err_style = 'band')
    # fig_task = sns.relplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
    #              ci = 68, palette = 'colorblind', markers = ['o','o'], style = 'trained',
    #              style_order = [True,False], kind = 'line', markeredgecolor=None,
    #              err_style = 'band')
    fig_task.set_xlabel(r'Preference relative to 45$\degree$ and 90$\degree$')
    fig_task.set_ylabel(r'Tuning curve slope (resp./$\degree$)')
    fig_task.set_title(r'At 45$\degree$ and 90$\degree$ ori.')
    # plt.legend(frameon=False)
    fig_task.legend_.get_texts()[0].set_text('Naïve')
    fig_task.legend_.get_texts()[1].set_text('Proficient')
    fig_task.legend_.set_title('')
    fig_task.legend_.set_frame_on(False)
    # fig_task.set_xticks(x_ticks[0::2])
    # fig_task.set_xticklabels(x_labels[0::2])
    fig_task.set_xticks(x_ticks)
    fig_task.set_xticklabels(x_labels, rotation=30)
    # fig_task.fig.set_size_inches(8,6)
    fig_task.set_ylim([-0.002, 0.018])
    fig_task.set_yticks(np.linspace(0,0.018,4))
    fig_task.vlines([-23,23], *fig_task.get_ylim(), colors='black', linestyles='dashed')

    sns.despine(trim = True)
    plt.tight_layout()

    plt.savefig(join(fig_save_dir,'tc_slope_45 and 90.svg'), format = 'svg')


    # df_plot = pd.DataFrame.copy(df_45)
    # # plt.figure()
    # fig_45 = sns.relplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
    #              ci = 95, palette = 'colorblind', markers = True, style = 'trained',
    #              style_order = [True,False], kind = 'line')
    # sns.despine()
    # plt.xlabel(r'Preference relative to 45$\degree$')
    # plt.ylabel(r'Tuning curve slope (response/$\degree$)')
    # # plt.legend(frameon=False)
    # fig_45._legend.get_texts()[0].set_text('Naïve')
    # fig_45._legend.get_texts()[1].set_text('Proficient')
    # fig_45._legend.set_title('')
    # # fig_task.set_xlim([-90,90])
    # plt.xticks(x_labels)
    # # plt.ylim([0.004, 0.020])

    # df_plot = pd.DataFrame.copy(df_90)
    # # plt.figure()
    # fig_90 = sns.relplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
    #              ci = 95, palette = 'colorblind', markers = True, style = 'trained',
    #              style_order = [True,False], kind = 'line')
    # sns.despine()
    # plt.xlabel(r'Preference relative to 90$\degree$')
    # plt.ylabel(r'Tuning curve slope (response/$\degree$)')
    # # plt.legend(frameon=False)
    # fig_90._legend.get_texts()[0].set_text('Naïve')
    # fig_90._legend.get_texts()[1].set_text('Proficient')
    # fig_90._legend.set_title('')
    # # fig_task.set_xlim([-90,90])
    # plt.xticks(x_labels)
    # # plt.ylim([0.004, 0.020])

    # df_plot = pd.DataFrame.copy(df_68[np.logical_and(df_68.r_bin > 1, df_68.r_bin<4)])
    df_plot = pd.DataFrame.copy(df_68)
    df_plot['slope'] = df_plot.slope.abs()
    # df_plot['pref_bin'] = np.abs(df_plot['pref_bin'])

    # plt.figure()
    # fig_68 = sns.relplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
    #              ci = 68, palette = 'colorblind', markers = ['o','o'],
    #              style = 'trained', style_order = [True,False], kind = 'line',
    #              markeredgecolor=None, err_style = 'band')
    plt.figure(figsize=(1.4,1.4))
    fig_68 = sns.lineplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
                 palette = 'colorblind', markers = ['o','o'], style = 'trained',
                 style_order = ['Proficient','Naive'], markeredgecolor=None, markersize = 2,
                 errorbar = ('se',1), err_style = 'band')
    fig_68.set_xlabel('Preference relative to 68$\degree$')
    fig_68.set_ylabel(r'Tuning curve slope (resp./$\degree$)')
    fig_68.set_title(r'At 68$\degree$ ori.')

    # plt.legend(frameon=False)
    fig_68.legend_.get_texts()[0].set_text('Naïve')
    fig_68.legend_.get_texts()[1].set_text('Proficient')
    fig_68.legend_.set_title('')
    fig_68.legend_.set_frame_on(False)
    # plt.xticks(x_ticks[0::2])
    fig_68.set_xticks(x_ticks)
    # fig_68.set_xticklabels(x_labels[0::2])
    fig_68.set_xticklabels(x_labels, rotation=30)
    # fig_68.fig.set_size_inches(8,6)
    fig_68.set_ylim([-0.002, 0.018])
    fig_68.set_yticks(np.linspace(0,0.018,4))
    fig_68.vlines([-23,23], *fig_68.get_ylim(), colors='black', linestyles='dashed')

    sns.despine(trim = True)
    plt.tight_layout()

    plt.savefig(join(fig_save_dir,'tc_slope_68.svg'), format = 'svg')

    # df_plot = pd.DataFrame.copy(df_orth_slope[
    #            np.logical_and(df_orth_slope.r_bin > 1, df_orth_slope.r_bin<4)])
    df_plot = pd.DataFrame.copy(df_orth_slope)
    df_plot['slope'] = df_plot.slope.abs()
    # df_plot['pref_bin'] = np.abs(df_plot['pref_bin'])
    # plt.figure()
    # fig_antitask = sns.relplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
    #               ci = 68, palette = 'colorblind', markers = ['o','p'], style = 'trained',
    #               style_order = [True,False], kind = 'line', markeredgecolor = None)
    plt.figure(figsize=(1.4,1.4))
    fig_antitask = sns.lineplot(data = df_plot, x = 'pref_bin', y = 'slope', hue = 'trained',
                 palette = 'colorblind', markers = ['o','o'], style = 'trained',
                 style_order = ['Proficient','Naive'], markeredgecolor=None, markersize = 2,
                 errorbar = ('se',1), err_style = 'band')
    sns.despine(trim = True)
    fig_antitask.set_xlabel(r'Preference relative to 135$\degree$ and 0$\degree$')
    fig_antitask.set_ylabel(r'Tuning curve slope at (resp./$\degree$)')
    fig_antitask.set_title(r'At 135$\degree$ and 0$\degree$ ori.')

    fig_antitask.legend_.get_texts()[0].set_text('Naïve')
    fig_antitask.legend_.get_texts()[1].set_text('Proficient')
    fig_antitask.legend_.set_title('')
    fig_antitask.legend_.set_frame_on(False)
    # fig_antitask.set_xticks(x_ticks[0::2])
    # fig_antitask.set_xticklabels(x_labels[0::2])
    fig_antitask.set_xticks(x_ticks)
    fig_antitask.set_xticklabels(x_labels, rotation=30)
    # fig_68.fig.set_size_inches(8,6)
    fig_antitask.set_ylim([-0.002, 0.018])
    fig_antitask.set_yticks(np.linspace(0,0.018,4))
    fig_antitask.vlines([-23,23], *fig_antitask.get_ylim(), colors='black', linestyles='dashed')
    sns.despine(trim = True)
    plt.tight_layout()

    plt.savefig(join(fig_save_dir, 'tc_slope_135_and_0.svg'), format = 'svg')

#%% cosine similarity function

def similarity(resps, stim, ori_only = False, shuffle = False):

    # Remove blank trials
    stim_ind = stim != np.inf
    stim = stim[stim_ind]

    if ori_only:
        stim = stim % 180

    resps = resps[stim_ind]

    # Split trials into two blocks

    ind_all = np.arange(len(stim))

    if shuffle:
        ind_0,ind_1,_,_ = skms.train_test_split(ind_all,ind_all,test_size = 0.5,
                                            shuffle = True, stratify = stim)

    else:
        ind_0 = np.concatenate([np.where(stim == s)[0][::2]
                                for s in np.unique(stim)])
        ind_1 = np.delete(ind_all,ind_0)


    stim_0, stim_1 = stim[ind_0], stim[ind_1]

    resps_0, resps_1 = resps[ind_0,:], resps[ind_1,:]

    # breakpoint()
    mu_0 = np.zeros((len(np.unique(stim_0)),resps_0.shape[1]))
    mu_1 = np.copy(mu_0)

    for si,s in enumerate(np.unique(stim_0)):
        mu_0[si,:] = np.mean(resps_0[stim_0==s,:],0)
        mu_1[si,:] = np.mean(resps_1[stim_1==s,:],0)

    ang_mu_mat = cosine_similarity(mu_0,mu_1)

    # ang_mu_mat = np.empty((len(np.unique(stim_0)), len(np.unique(stim_0))))

    # breakpoint()
    # for s0 in range(len(mu_0)):
    #     for s1 in range(len(mu_1)):
            # unit_0 = mu_0[s0,:] / np.linalg.norm(mu_0[s0,:])
            # unit_1 = mu_1[s1,:] / np.linalg.norm(mu_1[s1,:])
            # ang_mu_mat[s0,s1] = np.dot(unit_0, unit_1)

    return ang_mu_mat

#%% Generate cosine similarity measures for all experiments - subset
# Average over directions
n_repeats = 2000
n_cells = [5, 10, 25, 50, 100, 200, 400, 800]
shuffle = False

ang_mu_mats = np.zeros((n_repeats, len(n_cells), len(subjects), 8, 8))


for i in range(len(subjects)):
    ind = np.logical_and(trials_dict['subject'] == subjects[i],
                         trials_dict['trained'] == trained[i])

    cells = trials_dict['cell'][ind]
    resps = trials_dict['trial_resps'][ind]
    # resps = trials_dict['trial_resps_raw'][ind]
    # resps = resps/np.percentile(resps,95,axis=0)
    resps = resps.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
    stim = trials_dict['stim_dir'][ind].reshape(-1,len(np.unique(cells)))
    stim = stim[:,0]
    n_cells_expt = resps.shape[1]

    for ni,n in enumerate(n_cells):
        for r in range(n_repeats):
            print('Repeat ' + str(r+1) + ' with ' + str(n) + ' cells')
            if n_cells != -1:
                cell_ind = np.random.choice(np.arange(n_cells_expt), n, replace = False)
                ang_mu_mats[r,ni,i,...] = similarity(resps[:,cell_ind], stim, ori_only=True, shuffle=shuffle)
            else:
                ang_mu_mats[r,ni,i,...] = similarity(resps, stim, ori_only=True, shuffle=shuffle)

naive_mat = ang_mu_mats[:,:,0::2,...].mean(0)
trained_mat = ang_mu_mats[:,:,1::2,...].mean(0)


#%% Plot


f,a = plt.subplots(2,3, figsize=(10,6))
# cbar_ax = f.add_axes([.91, .3, .03, .4])
f1,cbar_ax = plt.subplots(1,1, figsize=(1,4))
# for i in range(len(n_cells)):
for i,n in enumerate([0,2,3,4,6,7]):
    diff = np.squeeze(trained_mat[n,...] - naive_mat[n,...])
    diff = diff.mean(0)
    sns.heatmap(diff, cmap = 'magma_r', ax = a.flatten()[i], vmax=0, vmin=-0.15, cbar_ax=cbar_ax, cbar_kws={'shrink' : 0.2, 'label' : 'Change in cosine similarity'}, square=True)
    a.flatten()[i].set_title(n_cells[n])
    a.flatten()[i].set_xticks(np.arange(8)+0.5)
    a.flatten()[i].set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int), rotation=0)
    a.flatten()[i].set_yticks(np.arange(8)+0.5)
    a.flatten()[i].set_yticklabels(np.ceil(np.arange(0,180,22.5)).astype(int), rotation=0)

f.tight_layout()
f1.tight_layout()

# f.savefig(r'C:\Users\samue\OneDrive\Desktop\cs_simulated.svg', format='svg')

diff = trained_mat-naive_mat

task_ori_diff = diff[:,:,[2,4],[4,2]].mean(1).mean(1)

f,a = plt.subplots(1,1)
a.plot(n_cells,task_ori_diff, '-k')
a.scatter(n_cells,task_ori_diff, s=10, color='black')
a.set_ylabel('Change in cosine similarity between 45 and 90')
a.set_xlabel('Number of neurons')
a.set_ylim([-0.14,-0.06])
sns.despine(ax=a, trim=True)
a.vlines(100, ymin=-0.135, ymax=-0.065, colors='black', linestyles='dashed')


#%% Generate cosine similarity measures for all experiments - orientations

ang_mu_mats = np.zeros((len(subjects), 8, 8))

for i in range(len(subjects)):
    ind = np.logical_and(trials_dict['subject'] == subjects[i],
                         trials_dict['trained'] == trained[i])
    # ind = np.logical_and(ind,trials_dict['V1_ROIs'] == 1)
    # ind = np.logical_and(ind,trials_dict['task_ret'] <= 20)
    cells = trials_dict['cell'][ind]
    resps = trials_dict['trial_resps'][ind]

    resps = resps.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
    stim = trials_dict['stim_ori'][ind].reshape(-1,len(np.unique(cells)))[:,0]

    ang_mu_mats[i,...] = similarity(resps, stim, ori_only=True)

naive_mat = ang_mu_mats[0::2,...]
trained_mat = ang_mu_mats[1::2,...]



#%% ANOVA on change

import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations, permutations

diff_mat = trained_mat - naive_mat

y_ori, x_ori = np.meshgrid(np.round(np.arange(0,180,22.5)),
                           np.round(np.arange(0,180,22.5)))

df_cs = pd.DataFrame({'stim0': np.tile(x_ori[None,:,:], (5,1,1)).ravel(),
                      'stim1': np.tile(y_ori[None,:,:], (5,1,1)).ravel(),
                      'cs': diff_mat.ravel(),
                      'subject': np.tile(np.arange(5)[:,None,None], (1,8,8)).ravel()})

def create_pair(row):
        return tuple(sorted([row['stim0'], row['stim1']]))

df_cs['pair'] = df_cs.apply(create_pair, axis=1)

df_mu = df_cs.groupby(['subject', 'pair'])['cs'].mean().reset_index()



model = ols('cs ~ C(pair)', data=df_mu).fit()

table = sm.stats.anova_lm(model, typ=2)

pw = model.t_test_pairwise("C(pair)")
results = pw.result_frame

index_series = results.index.to_series()

# Split the index into two parts and create new columns
results['First_Pair'] = index_series.str.split('-').str[0]
results['Second_Pair'] = index_series.str.split('-').str[1]

def extract_values(pair_str):
    x, y = pair_str.strip('()').split(',')
    return float(x), float(y)

results[['First_x', 'First_y']] = results['First_Pair'].apply(lambda x: extract_values(x)).tolist()
results[['Second_x', 'Second_y']] = results['Second_Pair'].apply(lambda x: extract_values(x)).tolist()

def should_swap(row):

    swap = False
    if row['First_x'] > row['Second_x']:
        swap = True
    elif row['First_y'] > row['Second_y']:
        swap = True
    
    return swap

# Perform the swap based on the condition
def swap_if_needed(row):
    if should_swap(row):
        # Swapping values
        row['First_Pair'], row['Second_Pair'] = row['Second_Pair'], row['First_Pair']
        row['First_x'], row['Second_x'] = row['Second_x'], row['First_x']
        row['First_y'], row['Second_y'] = row['Second_y'], row['First_y']
    return row

# Apply the function to each row
results = results.apply(swap_if_needed, axis=1)

results.sort_values(by=['First_x', 'First_y', 'Second_x', 'Second_y'], inplace=True)


# Reset the index to the default integer index
results.reset_index(drop=True, inplace=True)

results['reject-hs'] = results["reject-hs"].astype(int)

results_hm = pd.pivot(results, index='First_Pair', columns='Second_Pair', values='reject-hs')

results_hm = results_hm.reindex(index = results['First_Pair'].unique(), columns = results['Second_Pair'].unique())

unique_combos = list(combinations(df_mu['pair'].unique(),2))


differences = []

for pair in unique_combos:
    value1 = df_mu[df_mu['pair'] == pair[0]]['cs'].mean()
    value2 = df_mu[df_mu['pair'] == pair[1]]['cs'].mean()
    diff = value1 - value2
    differences.append(diff)

# Convert to DataFrame for better visualization
difference_df = pd.DataFrame(differences, columns=['Difference'])

difference_df['First_Pair'] = [c[0] for c in unique_combos]
difference_df['Second_Pair'] = [c[1] for c in unique_combos]

difference_df['First_Pair'] = difference_df['First_Pair'].astype(str)
difference_df['Second_Pair'] = difference_df['Second_Pair'].astype(str)

# results = results.set_index(['First_Pair', 'Second_Pair'])

# difference_df = difference_df.set_index(['First_Pair', 'Second_Pair'])

results['Difference'] = difference_df['Difference']

results_hm_diff = pd.pivot(results, index='First_Pair', columns='Second_Pair', values='Difference')
results_hm_diff = results_hm_diff.reindex(index = results['First_Pair'].unique(), columns = results['Second_Pair'].unique())


f,a = plt.subplots(1,2)

sns.heatmap(results_hm_diff, center=0, ax=a[0], xticklabels=1, yticklabels=1)

sns.heatmap(results_hm, ax=a[1], xticklabels=1, yticklabels=1)

#%% CS by stim type 

from itertools import product
from matplotlib.patches import Rectangle


diff_mat = trained_mat - naive_mat

y_ori, x_ori = np.meshgrid(np.ceil(np.arange(0,180,22.5)),
                           np.ceil(np.arange(0,180,22.5)))

df_cs = pd.DataFrame({'stim0': np.tile(x_ori[None,:,:], (5,1,1)).ravel(),
                      'stim1': np.tile(y_ori[None,:,:], (5,1,1)).ravel(),
                      'cs': diff_mat.ravel(),
                      'subject': np.tile(np.arange(5)[:,None,None], (1,8,8)).ravel()})


def create_pair(row):
        return tuple(sorted([row['stim0'], row['stim1']]))

df_cs['pair'] = df_cs.apply(create_pair, axis=1)

df_mu = df_cs.groupby(['subject', 'pair'])['cs'].mean().reset_index()

task = [(45,90)]

task_distractor = list(product([68],[45,90]))
task_distractor = [tuple(sorted(p)) for p in task_distractor]

non_task = np.ceil(np.arange(0, 180, 22.5))
non_task = non_task[np.logical_and(non_task != 45, non_task != 90) & (non_task != 68)]
task_nontask = list(product([45,90], non_task))
task_nontask = [tuple(sorted(p)) for p in task_nontask]

distractor_nontask = list(product([68], non_task))
distractor_nontask = [tuple(sorted(p)) for p in distractor_nontask]

nontask_nontask = list(product(non_task, non_task))
nontask_nontask = [tuple(sorted(p)) for p in nontask_nontask]
nontask_nontask = [p for p in nontask_nontask if p[0] != p[1]]

cs_change = np.zeros((5,5))

cs_change[0,:] = df_cs.groupby(['subject']).apply(lambda x: x[x.pair.isin(task)].cs.mean()).to_numpy()
cs_change[1,:] = df_cs.groupby(['subject']).apply(lambda x: x[x.pair.isin(task_nontask)].cs.mean()).to_numpy()
cs_change[2,:] = df_cs.groupby(['subject']).apply(lambda x: x[x.pair.isin(task_distractor)].cs.mean()).to_numpy()
cs_change[3,:] = df_cs.groupby(['subject']).apply(lambda x: x[x.pair.isin(distractor_nontask)].cs.mean()).to_numpy()
cs_change[4,:] = df_cs.groupby(['subject']).apply(lambda x: x[x.pair.isin(nontask_nontask)].cs.mean()).to_numpy()


# df_change = pd.DataFrame({'Stimulus type': np.tile(np.array(['45 vs 90 (V)',
#                                                              'Visuomotor vs non-task (VN)',
#                                                              'Visuomotor vs distractor (VD)',
#                                                              'Distractor vs non-task (DN)',
#                                                              'Non-task vs non-task (N)'])[:,None], (1,5)).flatten(),
#                           'Change in similarity': cs_change.flatten(),
#                           'Subject': np.tile(np.unique(subjects)[None,:], (5,1)).flatten()})

df_change = pd.DataFrame({'Stimulus type': np.tile(np.array(['45 to 90 (V)',
                                                             'Visuomotor to non-task (VN)',
                                                             'Visuomotor to distractor (VD)',
                                                             'Distractor to non-task (DN)',
                                                             'Non-task to non-task (N)'])[:,None], (1,5)).flatten(),
                          'Change in similarity': cs_change.flatten(),
                          'Subject': np.tile(np.unique(subjects)[None,:], (5,1)).flatten()})


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,
                                        "axes.titlesize":5,
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


    f,a = plt.subplots(1,1, figsize=(2,2))

    (
        sob.Plot(df_change, x = 'Stimulus type', y = 'Change in similarity')
        .layout(engine='tight')
        # .add(so.Lines(color='black', alpha=0.5))
        .add(sob.Dot(color='lightgray', edgecolor='black', pointsize=3), legend=False)
        # .add(sob.Dot(edgecolor='black', pointsize=3), color='Stimulus type', legend=False)
        .add(sob.Dash(color='black', linestyle='dashed', linewidth=0.5), sob.Agg(), legend=False)
        .add(sob.Range(color='black', linewidth=0.5), sob.Est(errorbar=('se',1)))
        .scale(y=sob.Continuous().tick(at=np.linspace(-0.18,-0.04,3)))
        .limit(y=(-0.19,-0.04))
        .on(a)
        .plot()
    )
    
    sns.despine(ax=a, trim=True)
    a.set_xticklabels(a.get_xticklabels(), rotation=25, fontdict={'horizontalalignment': 'right'})

f.savefig(join(fig_save_dir, 'change_in_cosine_similarity_stim_type.svg'), format='svg')

# pvalues

p_val = []
stim = []

for s in df_change['Stimulus type'].unique()[1:]:
    p_val.append(ttest_rel(df_change[df_change['Stimulus type']=='45 to 90 (V)']['Change in similarity'],
                           df_change[df_change['Stimulus type']==s]['Change in similarity'])[1])
    stim.append(s)



#%% Plot relationship between population sparseness and cosine similarity

import seaborn.objects as snso
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf


ps_task = ps_mat[:,[2,4]].mean(1)

ps_control = ps_mat[:,[0,7]].mean(1)

task_naive = naive_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1)
task_trained = trained_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1)

control_naive = naive_mat[:,np.logical_and(ori_ind[0], ori_ind[7])].mean(1)
control_trained = trained_mat[:,np.logical_and(ori_ind[0], ori_ind[7])].mean(1)

cs_task = np.zeros(10)
cs_task[0::2] = task_naive
cs_task[1::2] = task_trained

cs_control = np.zeros(10)
cs_control[0::2] = control_naive
cs_control[1::2] = control_trained

df = pd.DataFrame({'PS' : ps_task.ravel(),
                   'CS' : cs_task.ravel(),
                   'Mouse' : np.array(subjects),
                   'Condition':  np.array(trained)})

# md = smf.mixedlm("CS ~ PS + C(Condition)", df, groups=df["Mouse"], re_formula='~C(Condition)')

md = smf.mixedlm("CS ~ PS + C(Condition)", df, groups=df["Mouse"])


mdf = md.fit()


r = pearsonr(ps_task[1::2], cs_task[1::2])

df = pd.DataFrame({'PS' : ps_task.ravel(),
                   'CS' : cs_task.ravel(),
                   'Mouse' : np.array(subjects),
                   'Condition':  np.array(trained),
                   'Stim' : np.repeat('task',10)})

df_control = pd.DataFrame({'PS' : ps_control.ravel(),
                           'CS' : cs_control.ravel(),
                           'Mouse' : np.array(subjects),
                           'Condition':  np.array(trained),
                           'Stim' : np.repeat('control',10)})

df_plot = pd.concat([df,df_control],ignore_index=True)


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

    f,a = plt.subplots(1,1, figsize=(1.75,1.75))
    # f,a = plt.subplots(1,1)

    a.set_box_aspect(1)
    # a[1].set_box_aspect(1)


    # p = (
    #         snso.Plot(df, x='Population sparseness', y='Cosine similarity')
    #         .layout(engine='tight')
    #         .add(snso.Dot(color='black', pointsize=2))
    #         .add(snso.Line(linestyle='--', linewidth = 0.5, color='black'), snso.PolyFit(order=1))
    #         .limit(x=(0.43, 0.52), y=(0.53, 0.66))
    #         .on(a)
    #         .plot()
    # )


    p = (
            snso.Plot(df_plot[df_plot.Stim=='task'], x='PS', y='CS', marker='Stim')
            .layout(engine='tight')
            .add(snso.Dot(pointsize=3, edgecolor='black'), legend=False, color='Condition')
            # .add(snso.Line(linestyle='--', linewidth=0.5, color='black'), snso.PolyFit(order=1), legend=False)
            .scale(x=snso.Continuous().tick(every=0.1, between=(0.2,0.6)),
                   y=snso.Continuous().tick(every=0.1, between=(0.5,0.9)))
            .limit(x=(0.2,0.53), y=(0.5,0.81))
            .label(x='Population sparseness', y='Cosine similarity')
            .on(a)
            .plot()
    )

    # p = (
    #         snso.Plot(df_control, x='PS', y='CS', group='Mouse', color='Condition')
    #         .layout(engine='tight')
    #         .add(snso.Dot(pointsize=2), legend=False)
    #         .add(snso.Line(linestyle='--', linewidth=0.5, color='black'), snso.PolyFit(order=1), legend=False)
    #         # .limit(x=(0.25, 0.55), y=(0.45, 0.85))
    #         .on(a[1])
    #         .plot()
    # )


    p = np.polyfit(df_plot[df_plot.Stim=='task'].PS, df_plot[df_plot.Stim=='task'].CS, deg=1)
    x = np.linspace(0.225,0.53,100)
    a.plot(x,x*p[0]+p[1], '--k')

    r = pearsonr(df_plot[df_plot.Stim=='task'].PS, df_plot[df_plot.Stim=='task'].CS)
    a.text(0.4,0.7,f'r = {r[0].round(3)}')


    # p = np.polyfit(df_plot[df_plot.Stim=='control'].PS, df_plot[df_plot.Stim=='control'].CS, deg=1)
    # x = np.linspace(0.2,0.425,100)
    # a.plot(x,x*p[0]+p[1], '--k')

    # r = pearsonr(df_plot[df_plot.Stim=='control'].PS, df_plot[df_plot.Stim=='control'].CS)
    # a.text(0.4,0.85,f'r = {r[0].round(3)}')



    sns.despine(ax=a, offset=3, trim=True)
    # sns.despine(ax=a[1], trim=True)



f.savefig(join(fig_save_dir,'ps_cs_correlation.svg'), format='svg')

#%%

from scipy.stats import ttest_ind
from matplotlib.patches import Rectangle

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

diff_mat = trained_mat - naive_mat
pdiff_mat = (trained_mat - naive_mat)/naive_mat

y_ori, x_ori = np.meshgrid(np.round(np.arange(0,180,22.5)),
                           np.round(np.arange(0,180,22.5)))

ori_ind = [np.logical_or(x_ori == o, y_ori == o) for o in np.unique(x_ori%180)]


y_task = np.array([diff_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1),
              diff_mat[:,np.logical_and(ori_ind[6], ori_ind[0])].mean(1)]).reshape(-1,)

x_task = np.repeat([['45 vs 90'],['135 vs 0']],len(np.unique(subjects)))

naive_task = np.array([naive_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1),
              naive_mat[:,np.logical_and(ori_ind[6], ori_ind[0])].mean(1)]).reshape(-1,)
trained_task = np.array([trained_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1),
              trained_mat[:,np.logical_and(ori_ind[6], ori_ind[0])].mean(1)]).reshape(-1,)

stim_label = x_task = np.repeat([['45 vs 90'],['135 vs 0']],len(np.unique(subjects)))

cp = sns.color_palette('colorblind')
cp = [cp[2],cp[3]]

# Calculate pvalue

pval = ttest_ind(y_task[x_task == '45 vs 90'], y_task[x_task=='135 vs 0'])


# 1 samp ttest for all pairs

pvals = []
pair = []

oris = np.round(np.arange(0,180,22.5))

for o0 in range(8):
    for o1 in range(8):
       ori_diff = diff_mat[:,o0,o1]
       pvals.append(ttest_1samp(ori_diff, 0)[1])
       pair.append([oris[o0], oris[o1]])


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


    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    # plt.figure(figsize=(11.5,10))
    # fig = sns.scatterplot(x = naive_task, y = trained_task, hue = stim_label,
    #                 style = stim_label, palette = cp, legend = False,
    #                 zorder = 2)
    # # fig.set_xlim([0.5,0.85])
    # # fig.set_ylim([0.5,0.85])
    # fig.plot([0.4,0.9],[0.4,0.9],'--k', zorder = 1)
    # plt.xlabel('Naive cosine similarity')
    # plt.ylabel('Proficient cosine similarity')
    # sns.despine(trim = True)
    
    annotations = [['','N','VN','DN','VN','N','N','N'],
                   ['N','','VN','DN','V','N','N','N'],
                   ['VN','VN','','VD','V','VN','VN','VN'],
                   ['DN','DN','VD','','VD','DN','DN','DN'],
                   ['VN','VN','V','VD','','VN','VN','VN'],
                   ['N','N','VN','DN','VN','','N','N'],
                   ['N','N','VN','DN','VN','N','','N'],
                   ['N','N','VN','DN','VN','N','N','']]
    
    boxes = [[]]

    cmap = 'mako_r'
    # cmap = 'magma_r'
    # cmap = 'RdBu_r'
    # cmap = 'Greys'
    plt.figure(figsize=(2.2,2.2))
    # fig = sns.heatmap(diff_mat.mean(0), cmap = cmap, square = True,
    #                   cbar_kws = {'label' : r'$\Delta$cosine similarity',
    #                               'fraction' : 0.046,
    #                               'pad' : 0.04,
    #                               'ticks' : [-0.02,-0.04,-0.06,-0.08,-0.10,-0.12]},
    #                   annot=annotations, fmt="")
    fig = sns.heatmap(diff_mat.mean(0), cmap = cmap, square = True,
                      cbar_kws = {'label' : r'$\Delta$cosine similarity',
                                  'fraction' : 0.046,
                                  'pad' : 0.04,
                                  'ticks' : [-0.02,-0.04,-0.06,-0.08,-0.10,-0.12]})
    # fig = sns.heatmap(diff_mat.mean(0), cmap = cmap, square = True, center = 0,
    #                   cbar_kws = {'label' : r'$\Delta$cosine similarity',
    #                               'fraction' : 0.046,
    #                               'pad' : 0.04})

    fig.set_xticks(np.arange(0.5,8.5))
    fig.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    fig.set_yticks(np.arange(0.5,8.5))
    fig.set_yticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    fig.set_xlabel(r'$\vec{P}_1$ : Stimulus orientation ($\degree$)', labelpad = 1)
    fig.set_ylabel(r'$\vec{P}_2$ : Stimulus orientation ($\degree$)', labelpad = 1)
    # fig.set_title(r'$\cos(\theta)$ : Proficient - Naïve')
    fig.tick_params(left=False, bottom=False)
    plt.setp(fig.get_xticklabels()[2:5], fontweight = 'bold')
    plt.setp(fig.get_yticklabels()[2:5], fontweight = 'bold')
    # Remove tick marks from colorbar
    # fig.figure.axes[1].tick_params(size=0)
    # plt.plot([2,2],[0,8],'w',linestyle='--')
    # plt.plot([5,5],[0,8],'w',linestyle='--')
    # plt.plot([0,8],[2,2],'w',linestyle='--')
    # plt.plot([0,8],[5,5],'w',linestyle='--')

    # plt.plot([2,2],[0,2],'w',linestyle='--', lw = 0.5)
    # plt.plot([2,2],[5,8],'w',linestyle='--', lw = 0.5)
    # plt.plot([5,5],[0,2],'w',linestyle='--', lw = 0.5)
    # plt.plot([5,5],[5,8],'w',linestyle='--', lw = 0.5)
    # plt.plot([0,2],[2,2],'w',linestyle='--', lw = 0.5)
    # plt.plot([5,8],[2,2],'w',linestyle='--', lw = 0.5)
    # plt.plot([0,2],[5,5],'w',linestyle='--', lw = 0.5)
    # plt.plot([5,8],[5,5],'w',linestyle='--', lw = 0.5)
    plt.tight_layout()

    # savefile = join(results_dir,'Figures','Draft','cosine_similarity_matrix.svg')
    # plt.savefig(savefile, format = 'svg')


    plt.figure(figsize=(1.5,1.5))
    sfig = sns.swarmplot(x = x_task, y = y_task, s = 3, edgecolor='k',
                         linewidth = 0.5, color = 'k')


    # Add significance label

    # sfig = sig_label(sfig,(0,1),0.06,0.02,'**')

    # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.2

    for tick, text in zip(sfig.get_xticks(), sfig.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # calculate the median value for all replicates of either X or Y
        mean_val = y_task[x_task==sample_name].mean()
        std_val = y_task[x_task==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x_task)/len(np.unique(x_task))
        ci_val = mean_confidence_interval(
            y_task[x_task==sample_name], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        sfig.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--')
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        sfig.errorbar(tick,mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5)

    plt.ylabel(r'$\Delta$cosine similarity', labelpad = 1)
    plt.xlabel('')
    sfig.set_xticks([-0.5,0,1,1.5])
    sfig.set_xlim([-0.75,1.75])
    sfig.set_xticklabels(['',r'$\vec{P}_{45} \cdotp \vec{P}_{90}$',
                              r'$\vec{P}_{135} \cdotp \vec{P}_{0}$',''])

    sfig.set_yticks(np.arange(-0.2,0.05,0.1))

    plt.tight_layout()

    sns.despine(trim = True)

    # savefile = join(results_dir,'Figures','Draft','cosine_similarity_taskstim_vs_control.svg')
    # plt.savefig(savefile, format = 'svg')


#%% Generate cosine similarity measures for all experiments - orientations - by retinotopy

ang_mu_mats = np.zeros((len(subjects), 8, 8, 6))

for r in range(0,6):

    for i in range(len(subjects)):
        ind = np.logical_and(trials_dict['subject'] == subjects[i],
                             trials_dict['trained'] == trained[i])
        ind = np.logical_and(ind,trials_dict['task_ret_bin'] == r+1)

        cells = trials_dict['cell'][ind]
        resps = trials_dict['trial_resps'][ind]
        # resps = trials_dict['trial_resps_raw'][ind]

        resps = resps.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
        # resps = resps/np.percentile(resps,99,axis=0)
        stim = trials_dict['stim_ori'][ind].reshape(-1,len(np.unique(cells)))[:,0]

        scale_factor = np.array([resps[stim==s,:].mean(0) for s in np.unique(stim)]).max(0)
        resps = resps/scale_factor

        ang_mu_mats[i,...,r] = similarity(resps,stim,kind = 'ori')

naive_mat = ang_mu_mats[0::2,...]
trained_mat = ang_mu_mats[1::2,...]

#%% Generate cosine similarity measures for all experiments - subset
# Average over directions
n_repeats = 1000
n_cells = 100

ang_mu_mats = np.zeros((n_repeats, len(subjects), 8, 8, 6))
n_cells_expt = np.zeros(len(subjects),dtype = int)

for ret in range(0,6):
    for i in range(len(subjects)):
        ind = np.logical_and(trials_dict['subject'] == subjects[i],
                             trials_dict['trained'] == trained[i])

        ind = np.logical_and(ind,trials_dict['task_ret_bin'] == ret+1)

        cells = trials_dict['cell'][ind]
        resps = trials_dict['trial_resps'][ind]
        # resps = trials_dict['trial_resps_raw'][ind]
        # resps = resps/np.percentile(resps,95,axis=0)
        resps = resps.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
        stim = trials_dict['stim_dir'][ind].reshape(-1,len(np.unique(cells)))
        stim = stim[:,0]
        n_cells_expt[i] = resps.shape[1]


        for r in range(n_repeats):
            print('Repeat ' + str(r+1))
            if n_cells != -1:
                cell_ind = np.random.choice(np.arange(n_cells_expt[i]), n_cells)
                ang_mu_mats[r,i,...,ret] = similarity(resps[:,cell_ind],stim,
                                                      kind = 'ori', shuffle = True)
            else:
                ang_mu_mats[r,i,...,ret] = similarity(resps,stim, kind = 'ori',
                                                  shuffle = True)

naive_mat = ang_mu_mats[:,0::2,...].mean(0)
trained_mat = ang_mu_mats[:,1::2,...].mean(0)

#%%

diff_mat = trained_mat - naive_mat
pdiff_mat = (trained_mat - naive_mat)/naive_mat

y_ori, x_ori = np.meshgrid(np.round(np.arange(0,180,22.5)),
                           np.round(np.arange(0,180,22.5)))

ori_ind = [np.logical_or(x_ori == o, y_ori == o) for o in np.unique(x_ori%180)]


y_task = np.array([diff_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1),
              diff_mat[:,np.logical_and(ori_ind[6], ori_ind[0])].mean(1)]).reshape(-1,)

x_task = np.repeat([['45 vs 90'],['135 vs 0']],5)

naive_task = np.array([naive_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1),
              naive_mat[:,np.logical_and(ori_ind[6], ori_ind[0])].mean(1)]).reshape(-1,)
trained_task = np.array([trained_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1),
              trained_mat[:,np.logical_and(ori_ind[6], ori_ind[0])].mean(1)]).reshape(-1,)

stim_label = x_task = np.repeat([['45 vs 90'],['135 vs 0']],5)

cp = sns.color_palette('colorblind')
cp = [cp[2],cp[3]]


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
    f,axes = plt.subplots(2,3)

    for r in range(0,6):


        fig = sns.heatmap(diff_mat[...,r].mean(0), ax = axes.flatten()[r],
                          cmap = 'inferno_r', square = True,
                          vmin = -0.15, vmax = 0,
                          cbar_kws = {'label' : r'$\Delta$cosine similarity',
                                      'fraction' : 0.046,
                                      'pad' : 0.04,
                                      'ticks' : [-0.02,-0.04,-0.06,-0.08,-0.10,-0.12]})

        fig.set_xticks(np.arange(0.5,8.5))
        fig.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
        fig.set_yticks(np.arange(0.5,8.5))
        fig.set_yticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
        fig.set_xlabel(r'$\vec{P}_1$ : Stimulus orientation ($\degree$)', labelpad = 1)
        fig.set_ylabel(r'$\vec{P}_2$ : Stimulus orientation ($\degree$)', labelpad = 1)
        # fig.set_title(r'$\cos(\theta)$ : Proficient - Naïve')
        fig.tick_params(left=False, bottom=False)
        plt.setp(fig.get_xticklabels()[2:5], fontweight = 'bold')
        plt.setp(fig.get_yticklabels()[2:5], fontweight = 'bold')
        # Remove tick mark from colorbar
        fig.figure.axes[1].tick_params(size=0)
        # plt.plot([2,2],[0,8],'w',linestyle='--')
        # plt.plot([5,5],[0,8],'w',linestyle='--')
        # plt.plot([0,8],[2,2],'w',linestyle='--')
        # plt.plot([0,8],[5,5],'w',linestyle='--')

        axes.flatten()[r].plot([2,2],[0,2],'w',linestyle='--', lw = 1)
        axes.flatten()[r].plot([2,2],[5,8],'w',linestyle='--', lw = 1)
        axes.flatten()[r].plot([5,5],[0,2],'w',linestyle='--', lw = 1)
        axes.flatten()[r].plot([5,5],[5,8],'w',linestyle='--', lw = 1)
        axes.flatten()[r].plot([0,2],[2,2],'w',linestyle='--', lw = 1)
        axes.flatten()[r].plot([5,8],[2,2],'w',linestyle='--', lw = 1)
        axes.flatten()[r].plot([0,2],[5,5],'w',linestyle='--', lw = 1)
        axes.flatten()[r].plot([5,8],[5,5],'w',linestyle='--', lw = 1)


    plt.tight_layout()
#%% Plot generalization curves for each stim

gen_curve = np.zeros((5,2,8,8))
uni_stim = np.unique(x_ori)

for i in range(8):

    ind_i = uni_stim == uni_stim[i]
    gen_curve[:,0,i,:] = np.concatenate([naive_mat[:,ind_i,:].reshape(-1,8,1),
                                         naive_mat[:,:,ind_i].reshape(-1,8,1)],axis=2).mean(2)
    gen_curve[:,1,i,:] = np.concatenate([trained_mat[:,ind_i,:].reshape(-1,8,1),
                                         trained_mat[:,:,ind_i].reshape(-1,8,1)],axis=2).mean(2)


gen_curve_diff = np.squeeze(np.diff(gen_curve,axis = 1))
#%% Plot curves

import seaborn.objects as seao

trained_label = np.tile(np.unique(trained).reshape(1,-1,1,1), (5,1,8,8))
stim_label_0 = np.tile(np.array([0,23,45,68,90,113,135,158]).reshape(1,1,-1,1),
                     (5,2,1,8))
stim_label_1 = np.tile(np.array([0,23,45,68,90,113,135,158]).reshape(1,1,1,-1),
                     (5,2,8,1))
subject_label = np.tile(np.unique(subjects).reshape(-1,1,1,1), (1,2,8,8))

df = pd.DataFrame({'trained' : trained_label.flatten(),
                   'stim0' : stim_label_0.flatten(),
                   'stim1' : stim_label_1.flatten(),
                   'subject' : subject_label.flatten(),
                   'cosine similarity' : gen_curve.flatten()})

# sns.relplot(data = df, x = 'stim0', col = 'stim1', hue = 'trained',
#             y = 'cosine similarity', kind = 'line', col_wrap = 4)

(
    seao.Plot(df, x='stim0', y='cosine similarity', color='trained')
    .facet(col='stim1', wrap=4)
    .add(seao.Lines(), seao.Agg(), legend=False)
    .add(seao.Band(), seao.Est(errorbar=('se',1)), legend=False)
    .scale(x=seao.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))))
    .label(x='Stimulus', legend='')
    .show()
)


#%% Plot diff curves

stim_label_0 = np.tile(np.array([0,23,45,68,90,113,135,158]).reshape(1,-1,1),
                     (5,1,8))
stim_label_1 = np.tile(np.array([0,23,45,68,90,113,135,158]).reshape(1,1,-1),
                     (5,8,1))

df = pd.DataFrame({'stim0' : stim_label_0.flatten(),
                   'stim1' : stim_label_1.flatten(),
                   'cosine similarity' : gen_curve_diff.flatten()})

# sns.relplot(data = df, x = 'stim0', col = 'stim1',
#             y = 'cosine similarity', kind = 'line', col_wrap = 4,
#             ci = 68)

(
    seao.Plot(df, x='stim0', y='cosine similarity')
    .facet(col='stim1', wrap=4)
    .add(seao.Lines(), seao.Agg(), legend=False)
    .add(seao.Band(), seao.Est(errorbar=('se',1)), legend=False)
    .scale(x=seao.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))))
    .label(x='Stimulus', legend='')
    .show()
)

#%% Generate cosine similarity measures for all experiments - subset
# Average over directions
n_repeats = 1000
n_cells = -1

ang_mu_mats = np.zeros((n_repeats, len(subjects), 16, 16))
n_cells_expt = np.zeros(len(subjects),dtype = int)

for i in range(len(subjects)):
    ind = np.logical_and(trials_dict['subject'] == subjects[i],
                         trials_dict['trained'] == trained[i])

    cells = trials_dict['cell'][ind]
    resps = trials_dict['trial_resps'][ind]
    resps = resps.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
    stim = trials_dict['stim_dir'][ind].reshape(-1,len(np.unique(cells)))
    stim = stim[:,0]
    n_cells_expt[i] = resps.shape[1]

    for r in range(n_repeats):
        print('Repeat ' + str(r+1))
        if n_cells != -1:
            cell_ind = np.random.choice(np.arange(n_cells_expt[i]), n_cells)
            ang_mu_mats[r,i,...] = similarity(resps[:,cell_ind],stim,
                                                  kind = 'dir')
        else:
            ang_mu_mats[r,i,...] = similarity(resps,stim, kind = 'dir')

naive_mat = ang_mu_mats[:,0::2,...].mean(0)
trained_mat = ang_mu_mats[:,1::2,:].mean(0)

#%% Generate cosine similarity measures for all experiments - average across
# directions

ang_mu_mats = np.zeros((len(subjects), 16, 16))

for i in range(len(subjects)):
    ind = np.logical_and(trials_dict['subject'] == subjects[i],
                         trials_dict['trained'] == trained[i])

    cells = trials_dict['cell'][ind]
    resps = trials_dict['trial_resps'][ind]
    resps = resps.reshape(len(np.unique(trials_dict['trial_num'][ind])),-1)
    stim = trials_dict['stim_dir'][ind].reshape(-1,len(np.unique(cells)))
    stim = stim[:,0]

    ang_mu_mats[i,...] = similarity(resps,stim, kind = 'dir')

naive_mat = ang_mu_mats[0::2,...]
trained_mat = ang_mu_mats[1::2,:]
#%%

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

diff_mat = trained_mat - naive_mat
pdiff_mat = (trained_mat - naive_mat)/naive_mat

x_ori, y_ori = np.meshgrid(np.round(np.arange(0,360,22.5)),
                           np.round(np.arange(0,360,22.5)))

ori_ind = [np.logical_or(x_ori%180 == o, y_ori%180 == o) for o in np.unique(x_ori%180)]


y_task = np.array([diff_mat[:,np.logical_and(ori_ind[2], ori_ind[4])].mean(1),
              diff_mat[:,np.logical_and(ori_ind[6], ori_ind[0])].mean(1)]).reshape(-1,)

x_task = np.repeat([['45 vs 90'],['135 vs 0']],5)


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1):


    plt.figure(figsize=(11.5,10))
    fig = sns.heatmap(diff_mat.mean(0), cmap = 'inferno_r', square = True,
                      cbar_kws = {'label' : r'$\Delta$cosine similarity'})

    fig.set_xticks(np.arange(0.5,8.5))
    fig.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    fig.set_yticks(np.arange(0.5,8.5))
    fig.set_yticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    fig.set_xlabel(r'$\vec{P}_1$ : Stimulus orientation (deg)')
    fig.set_ylabel(r'$\vec{P}_2$ : Stimulus orientation (deg)')
    # fig.set_title(r'$\cos(\theta)$ : Proficient - Naïve')
    fig.tick_params(left=False, bottom=False)
    plt.setp(fig.get_xticklabels()[2:5], fontweight = 'bold')
    plt.setp(fig.get_yticklabels()[2:5], fontweight = 'bold')
    # Remove tick marks from colorbar
    fig.figure.axes[1].tick_params(size=0)
    # plt.plot([2,2],[0,8],'w',linestyle='--')
    # plt.plot([5,5],[0,8],'w',linestyle='--')
    # plt.plot([0,8],[2,2],'w',linestyle='--')
    # plt.plot([0,8],[5,5],'w',linestyle='--')

    plt.plot([2,2],[0,2],'w',linestyle='--')
    plt.plot([2,2],[5,8],'w',linestyle='--')
    plt.plot([5,5],[0,2],'w',linestyle='--')
    plt.plot([5,5],[5,8],'w',linestyle='--')
    plt.plot([0,2],[2,2],'w',linestyle='--')
    plt.plot([5,8],[2,2],'w',linestyle='--')
    plt.plot([0,2],[5,5],'w',linestyle='--')
    plt.plot([5,8],[5,5],'w',linestyle='--')

    plt.figure(figsize=(13,9))
    sfig = sns.swarmplot(x = x_task, y = y_task, size = 12, edgecolor='k', linewidth = 2)

    # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.27

    for tick, text in zip(sfig.get_xticks(), sfig.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # calculate the median value for all replicates of either X or Y
        mean_val = y_task[x_task==sample_name].mean()
        std_val = y_task[x_task==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x_task)/len(np.unique(x_task))
        ci_val = mean_confidence_interval(
            y_task[x_task==sample_name], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        sfig.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=4, color='k', linestyle='--')
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        sfig.errorbar(tick,mean_val, ci_val, ecolor = 'k',
                          capsize = 11,capthick=2)

    sns.despine()

    plt.ylabel(r'$\Delta$cosine similarity')
    plt.xlabel('')
    sfig.set_xticklabels([r'$\vec{P}_{45} \cdotp \vec{P}_{90}$',
                              r'$\vec{P}_{135} \cdotp \vec{P}_{0}$'])

#%% Population sparseness

def pop_sparseness(fr, kind = 'Treves-Rolls'):

    # breakpoint()
    if np.logical_or(kind.lower() == 'treves-rolls',  kind.lower() == 'tr'):
        # Treves-Rolls
        top = (np.abs(fr)/fr.shape[1]).sum(1)**2
        bottom = (fr**2/fr.shape[1]).sum(1)
        s = 1 - (top/bottom)
    elif kind.lower() == 'kurtosis':
        s = kurtosis(fr,axis = 1)
    elif kind.lower() == 'active':
        sigma = fr.std(1)
        s = (fr < sigma[:,None]).sum(1)/fr.shape[1]
    return s

kind = 'tr'

ps_mat = np.zeros((len(subjects),8))

for i, (s,t) in enumerate(zip(subjects,trained)):
    expt_ind = np.logical_and(ori_dict['subject'] == s,
                              ori_dict['trained'] == t)
    # expt_ind = np.logical_and(expt_ind, df_scm['ROI_task_ret']<=25)
    # expt_ind = np.logical_and(expt_ind, df_scm['V1_ROIs']==1)

    ori_resps = ori_dict['mean_ori_test'][:-1,expt_ind]
    ps_mat[i,:] = pop_sparseness(ori_resps,kind)

    print(ori_resps.shape)

# Average for 45 and 90, compared to 68 and all other non-task stim

stim = np.ceil(np.arange(0,180,22.5))

ps_task = np.zeros((10,3))

ps_task[:,0] = ps_mat[:,np.logical_or(stim==45,stim==90)].mean(1)
ps_task[:,1] = ps_mat[:,stim==68].mean(1)
ps_task[:,2] = ps_mat[:,(stim != 45) & (stim!=68) & (stim!=90)].mean(1)

ps_diff = np.array([ps_task[i,:]-ps_task[i-1,:] for i in range(1,len(subjects),2)])


# ps_task = np.zeros((5,3))

# ps_diff = ps_mat[1::2,:] - ps_mat[0::2,:]

# ps_task[:,0] = ps_diff[:, np.logical_or(stim==45,stim==90)].mean(1)
# ps_task[:,1] = ps_diff[:, stim==68].mean(1)
# ps_task[:,2] = ps_diff[:, (stim != 45) & (stim!=68) & (stim!=90)].mean(1)



#%% Compare change in PS with exposure to stim

ps_diff_all = np.array([ps_mat[i,:] - ps_mat[i-1,:] for i in range(1,len(subjects),2)])

ps_diff_all = ps_diff_all[1:,2:5]

ps = ps_mat[1::2,:]

ps = ps[1:,2:5]

# Stim proportion by subject
ps_subjects = np.tile(np.array(['SF170905B','SF171107', 'SF180515', 'SF180613'])[:,None], (1,3))
ps_stim = np.tile([45,68,90], (4,1))

# Stimulus proportion manually calculated previously
ps_stim_prop = np.array([0.2807, 0.2675, 0.4517, 0.3190, 0.314655, 0.366379, 0.349650, 0.340326, 0.310023,
                0.253117,0.386534,0.360349])

ps_stim_counts = np.array([64,61,103,74,73,85,150,146,133,203,310,289])


df_ps_stim = pd.DataFrame({'stim' : ps_stim.flatten(),
                           'subject' : ps_subjects.flatten(),
                           'stim_prop' : ps_stim_prop.flatten(),
                           'stim_counts' : ps_stim_counts.flatten(),
                           'ps_diff' : ps_diff_all.flatten(),
                           'ps' : ps.flatten()})

ori_colors = sns.color_palette('colorblind', 8)[2:5]

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
                                        "legend.title_fontsize":5,
                                        'font.sans-serif' : 'Helvetica',
                                        'font.family' : 'sans-serif'}):


    plt.figure()
    sns.scatterplot(data = df_ps_stim, x = 'stim_prop', y = 'ps_diff', hue = 'stim')
    plt.figure()
    sns.scatterplot(data = df_ps_stim, x = 'stim_counts', y = 'ps_diff', hue = 'stim')
    plt.figure()
    sns.scatterplot(data = df_ps_stim, x = 'stim_counts', y = 'ps', hue = 'stim')


    plt.figure(figsize = (2,2))
    for i,s in enumerate(df_ps_stim.stim.unique()):
        sns.regplot(df_ps_stim[df_ps_stim.stim==s], x = 'stim_prop', y = 'ps_diff', ci = None, scatter_kws = {'ec' : 'k',
                                                                                                              'linewidth' : 0.2},
                    color = ori_colors[i])

    plt.figure(figsize = (2,2))
    for i,s in enumerate(df_ps_stim.stim.unique()):
        sns.regplot(df_ps_stim[df_ps_stim.stim==s], x = 'stim_counts', y = 'ps', ci = None, scatter_kws = {'ec' : 'k',
                                                                                                           'linewidth' : 0.2},
                    color = ori_colors[i])

    sns.despine(trim = True)
    plt.xlabel('Stimulus exposures')
    plt.ylabel('Population sparseness')

    plt.figure(figsize = (2,2))
    for i,s in enumerate(df_ps_stim.stim.unique()):
        sns.regplot(df_ps_stim[df_ps_stim.stim==s], x = 'stim_counts', y = 'ps_diff', ci = None, scatter_kws = {'ec' : 'k',
                                                                                                                'linewidth' : 0.2},
                    color = ori_colors[i])

    sns.despine(trim = True)
    plt.xlabel('Stimulus exposures')
    plt.ylabel('Change in pop. sparseness')


    plt.figure(figsize=(2,2))
    sns.regplot(df_ps_stim, x = 'stim_counts', y = 'ps_diff', ci = None, scatter_kws = {'ec' : 'k',
                                                                                        'linewidth' : 0.2},
                color = 'k')

    sns.despine(trim = True)
    plt.xlabel('Stimulus exposures')
    plt.ylabel('Change in pop. sparseness')

#%% plot distribution of responses for each experiment

f, a = plt.subplots(2,5)

for ti,t in enumerate(np.unique(trained)):
    for si,s in enumerate(np.unique(subjects)):
        expt_ind = np.logical_and(ori_dict['subject'] == s,
                                  ori_dict['trained'] == t)

        a[ti,si].hist(ori_dict['mean_ori_test'][2,expt_ind])

#%% Plot change in population sparseness

from scipy.stats import ttest_ind, ranksums, ttest_rel
from statsmodels.stats.multitest import multipletests

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

stim_label = np.repeat(np.ceil(np.arange(0,180,22.5))[None,:],
                                           len(subjects),axis = 0).astype(int)

trained_label = np.repeat(np.array(trained)[:,None], 8, axis = 1)

stim_label_diff = stim_label[0:int(len(subjects)/2),:]


# Do stats on all pairs
pvals = []
pair = []

task_stim = [45,90]
non_task = [0,23,68,113,135,158]

uni_stim = np.unique(stim_label_diff)
# for o0 in task_stim:
#     for o1 in non_task:
#         pvals.append(ttest_ind(ps_diff[stim_label_diff==o0],
#                                ps_diff[stim_label_diff==o1], equal_var = False)[1])
#         pair.append([o0,o1])

for o0 in task_stim:
    for o1 in non_task:
        pvals.append(ttest_rel(ps_diff[stim_label_diff==o0],
                               ps_diff[stim_label_diff==o1])[1])
        pair.append([o0,o1])


# pvals = multipletests(pvals,method = 'hs')


pvals_task = []
# pvals_task.append(ttest_ind(ps_diff[stim_label_diff==45], ps_diff[stim_label_diff==68], equal_var = False)[1])
# pvals_task.append(ttest_ind(ps_diff[stim_label_diff==45], ps_diff[stim_label_diff==90], equal_var = False)[1])
# pvals_task.append(ttest_ind(ps_diff[stim_label_diff==68], ps_diff[stim_label_diff==90], equal_var = False)[1])
# pvals_task.append(ranksums(ps_diff[stim_label_diff==45], ps_diff[stim_label_diff==68])[1])
# pvals_task.append(ranksums(ps_diff[stim_label_diff==45], ps_diff[stim_label_diff==90])[1])
# pvals_task.append(ranksums(ps_diff[stim_label_diff==68], ps_diff[stim_label_diff==90])[1])

pvals_task.append(ttest_ind(ps_diff[:,0], ps_diff[:,1], equal_var = True)[1])
pvals_task.append(ttest_ind(ps_diff[:,0], ps_diff[:,2], equal_var = True)[1])


#%%


stim_label = np.repeat(np.array([r'45$\degree$ and 90$\degree$',r'68$\degree$','non-task']).reshape(1,-1),
                                           len(subjects),axis = 0)

trained_label = np.repeat(np.array(trained)[:,None], 3, axis = 1)

stim_label_diff = stim_label[0:int(len(subjects)/2),:]


# ori_pal = []

# for i in range(8):
#     if i >= 2 and i <= 4:
#         ori_pal.append(sns.color_palette('colorblind')[i])
#     else:
#         ori_pal.append(sns.color_palette('colorblind')[7])



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
                                        "legend.title_fontsize":5,
                                        'font.sans-serif' : 'Helvetica',
                                        'font.family' : 'sans-serif'}):


    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'


    plt.figure(figsize=(1.6,1.6))
    psfig = sns.swarmplot(x = stim_label_diff.flatten(), y = ps_diff.flatten(),
                  s = 2, linewidth=0.5, zorder = 1, hue = stim_label_diff.flatten(),
                  edgecolor = 'k', color = 'black')
    psfig.legend_.set_visible(False)
    plt.xlabel(r'Stimulus orientation', labelpad = 2)
    plt.ylabel(r'$\Delta$population sparseness', labelpad = 2)


    psfig = sig_label(psfig,(0,1),0.04,0.02,"*",7,'k')
    psfig = sig_label(psfig,(0,2),0.12,0.02,"**",7,'k')

    # Add stars for significance, green stars for vs 45, pink for vs 90

    # x_pos = [0,1,3,5,6,7,2,4]
    # labels = ['**\n**','**\n**','*\n*','*\n*','n.s.\nn.s.','**\n*', 'n.s.','n.s.']

    # r = np.diff(psfig.get_ylim())*0.06

    # for i,p in enumerate(x_pos):
    #     m = psfig.get_children()[p].get_offsets()[:,1].max() + r
    #     psfig.text(p, m, labels[i], ha='center', va='bottom', color = 'k',
    #         fontsize = 7)


     # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.5

    x = stim_label_diff.flatten()
    y = ps_diff.flatten()

    for tick, text in zip(psfig.get_xticks(), psfig.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # calculate the median value for all replicates of either X or Y
        mean_val = y[x.astype(str)==sample_name].mean()
        std_val = y[x.astype(str)==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x)/len(np.unique(x))
        ci_val = mean_confidence_interval(
            y[x.astype(str)==sample_name], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        psfig.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        psfig.errorbar(tick,mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)


    if kind == 'kurtosis':
        psfig.set_yticks(np.arange(-0.5,4.5,1))
    else:
        psfig.set_yticks(np.linspace(-0,0.2,3))

    sns.despine(trim = True)

    # stim_labels = [s + r'$\degree$' for s in np.unique(stim_label_diff.flatten()).astype(str)]
    # psfig.set_xticklabels(stim_labels)

    plt.tight_layout()

    savefile = join(fig_save_dir,'population_sparseness_stim type_2.svg')
    plt.savefig(savefile, format = 'svg')


#%% Population sparseness by retinotopy

def pop_sparseness(fr, kind = 'Treves-Rolls'):

    # breakpoint()
    if np.logical_or(kind.lower() == 'treves-rolls',  kind.lower() == 'tr'):
        # Treves-Rolls
        top = (fr/fr.shape[1]).sum(1)**2
        bottom = (fr**2/fr.shape[1]).sum(1)
        s = 1 - (top/bottom)
    elif kind.lower() == 'kurtosis':
        s = kurtosis(fr,axis = 1)
    elif kind.lower() == 'active':
        sigma = fr.std(1)
        s = (fr < sigma[:,None]).sum(1)/fr.shape[1]
    return s

kind = 'tr'


ret_bins = np.arange(0,40,10)

ps_mat = np.zeros((len(subjects),8,len(ret_bins)-1))


for ir in range(len(ret_bins)-1):

    for i, (s,t) in enumerate(zip(subjects,trained)):
        expt_ind = np.logical_and(ori_dict['subject'] == s,
                                  ori_dict['trained'] == t)
        expt_ind = np.logical_and(expt_ind, (df_scm.ROI_task_ret > ret_bins[ir]) & (df_scm.ROI_task_ret <= ret_bins[ir+1]))
        expt_ind = np.logical_and(expt_ind,df_scm.V1_ROIs==1)

        ori_resps = ori_dict['mean_ori_all'][:-1,expt_ind]
        ps_mat[i,:,ir] = pop_sparseness(ori_resps,kind)


ps_diff = np.array([ps_mat[i,...]-ps_mat[i-1,...] for i in range(1,10,2)])

#%% Plot change in population sparseness with retinotopy


def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a, nan_policy = 'omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

stim_label = np.repeat(np.round(np.arange(0,180,22.5))[None,:],
                                           len(subjects),axis = 0).astype(int)

trained_label = np.repeat(np.array(trained)[:,None], 8, axis = 1)

# plt.figure()
# sns.boxplot(x = stim_label.flatten(), y = ps_mat.flatten(),
#                                                   hue = trained_label.flatten())
# sns.despine()
# plt.xlabel('Stimulus orientation (deg)')
# plt.ylabel('Population sparseness (kurtosis)')

stim_label_diff = stim_label[0:5,:]


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
    f,axes = plt.subplots(3,3)
    for r in range(0,7):


        sns.swarmplot(x = stim_label_diff.flatten(), y = ps_diff[...,r].flatten(),
                      s = 3, linewidth=0.5, zorder = 1, ax = axes.flatten()[r])
        sns.despine()
        axes.flatten()[r].set(xlabel=r'Stimulus orientation ($\degree$)')
        axes.flatten()[r].set(ylabel=r'$\Delta$population sparseness')
        axes.flatten()[r].set(ylim=[-1,5])

         # distance across the "X" or "Y" stipplot column to span
        mean_width = 0.5

        x = stim_label_diff.flatten()
        y = ps_diff[...,r].flatten()
        psfig = axes.flatten()[r]

        for tick, text in zip(psfig.get_xticks(), psfig.get_xticklabels()):
            sample_name = text.get_text()  # "X" or "Y"

            # calculate the mean value for all replicates of either X or Y
            mean_val = np.nanmean(y[x.astype(str)==sample_name])
            std_val = np.nanstd(y[x.astype(str)==sample_name])

            # plot horizontal lines across the column, centered on the tick
            # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
            #               [mean_val, mean_val], lw=4, color='k')
            n_points = len(x)/len(np.unique(x))
            ci_val = mean_confidence_interval(
                y[x.astype(str)==sample_name], confidence = 0.68)
            # pdif_fig.plot([mean_val, mean_val],
            #               [tick-mean_width/2, tick+mean_width/2],
            #               lw=4, color='k', linestyle='--')
            # pdif_fig.plot([ci_val1, ci_val2],
            #               [tick, tick],
            #               lw=4, color='k')
            psfig.plot([tick-mean_width/2, tick+mean_width/2],
                          [mean_val, mean_val],
                          lw=0.5, color='k', linestyle='--', zorder = 2)
            # pdif_fig.plot([tick, tick],
            #               [ci_val1, ci_val2],
            #               lw=4, color='k')
            psfig.errorbar(tick,mean_val, ci_val, ecolor = 'k',
                              capsize = 3,capthick=0.5, zorder = 2)

        plt.tight_layout()


#%% Population sparseness - all trials

def pop_sparseness(fr, kind = 'Treves-Rolls'):

    # breakpoint()
    if np.logical_or(kind.lower() == 'treves-rolls',  kind.lower() == 'tr'):
        # Treves-Rolls
        top = (np.abs(fr)/fr.shape[1]).sum(1)**2
        bottom = (fr**2/fr.shape[1]).sum(1)
        s = 1 - (top/bottom)
    elif kind.lower() == 'kurtosis':
        s = kurtosis(fr,axis = 1)
    elif kind.lower() == 'active':
        sigma = fr.std(1)
        s = (fr < sigma[:,None]).sum(1)/fr.shape[1]
    return s

kind = 'tr'


for i, (s,t) in enumerate(zip(subjects,trained)):
    expt_ind = np.logical_and(trials_dict['subject'] == s,
                              trials_dict['trained'] == t)

    trials = trials_dict['trial_resps'][expt_ind]
    n_cells = len(np.unique(trials_dict['cell'][expt_ind]))
    trials = trials.reshape(-1,n_cells)
    stimuli = trials_dict['stim_ori'][expt_ind].reshape(-1,n_cells)[:,0]
    nb_ind = stimuli != np.inf
    trials = trials[nb_ind,:]
    stimuli = stimuli[nb_ind]

    trial_sparseness = pop_sparseness(trials)


#%% Plot d' as function of orientation modal preference and condition - 45 vs 90

df = pd.DataFrame(trials_dict)

# Only include test set
df = df[~df.train_ind]

# Remove passive expts
df = df[df.trained!='Passive']

# Remove blank
df = df[df.stim_ori != np.inf]


df['pref_bin'] = pd.cut(df.pref, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)).astype(int))
df['r_bin'] = pd.cut(df.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))


groups = ['subject', 'stim_ori', 'trained', 'cell']

df_var = df.groupby(groups, observed = True).var().reset_index()

df_mu = df.groupby(groups, observed = True).mean().reset_index()


mu = df_mu.loc[df_mu.stim_ori == 45, 'trial_resps'].to_numpy()[:,None]
mu = np.append(mu,df_mu.loc[df_mu.stim_ori==90, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)
mu = np.diff(mu,axis=1)

var = df_var.loc[df_var.stim_ori == 45, 'trial_resps'].to_numpy()[:,None]
var = np.append(var,df_var.loc[df_var.stim_ori==90, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)
var = np.sqrt(var.sum(1)/2)

d_prime = np.abs(mu)/var[:,None]
df_d = pd.DataFrame({'d' : d_prime.reshape(-1,),
                     'trained' : df_scm[df_scm.trained != 'Passive'].trained.to_numpy(),
                     'pref_ori' : df_scm[df_scm.trained != 'Passive'].pref_ori_all.to_numpy(),
                     'subject' : df_scm[df_scm.trained != 'Passive'].subject.to_numpy()})

df_d_mu = df_d.groupby(['trained', 'subject', 'pref_ori']).mean().reset_index()

# sns.set()
# sns.set_style('ticks')
# with sns.plotting_context("poster", font_scale = 0.8):

fig = sns.relplot(data = df_d_mu, x = 'pref_ori', y = 'd', hue = 'trained',
                 kind = 'line', errorbar = ('se',1.96), palette = 'colorblind',
                 markers = ['o','o'], style = 'trained',
                 style_order = ['Proficient','Naive'],
                 markeredgecolor=None,
                 err_style = 'band')
plt.xlabel('Preferred orientation (deg)')
plt.ylabel(r'd$^\prime$ : 45 vs 90')
fig._legend.set_title('')
fig._legend.texts[0].set_text('Naïve')
fig._legend.texts[1].set_text('Proficient')
plt.xticks(np.unique(df.pref_bin))
fig.fig.set_size_inches((8,6))
plt.ylim([-1.5,1.5])


#%% Plot d' as function of orientation preference and condition - 45 vs 90
# Make it similar to figures in Poort, et al

df = pd.DataFrame(trials_dict)

# Only include test set
df = df[~df.train_ind]
df = df[df.trained!='Passive']

# Remove blank
df = df[df.stim_ori != np.inf]

groups = ['subject', 'stim_dir', 'trained', 'cell']

df_var = df.groupby(groups, observed = True)['trial_resps'].var().reset_index()

df_var['stim_ori'] = df_var.stim_dir % 180

df_var = df_var.groupby(['subject','stim_ori','trained','cell'],observed = True).mean().reset_index()

df_mu = df.groupby(groups, observed = True)['trial_resps'].mean().reset_index()

df_mu['stim_ori'] = df_mu.stim_dir % 180

df_mu = df_mu.groupby(['subject','stim_ori','trained','cell'],observed = True).mean().reset_index()

mu = df_mu.loc[df_mu.stim_ori == 90, 'trial_resps'].to_numpy()[:,None]
mu = np.append(mu,df_mu.loc[df_mu.stim_ori==45, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)
df_labels = df_mu.loc[df_mu.stim_ori == 45]

var = df_var.loc[df_var.stim_ori == 90, 'trial_resps'].to_numpy()[:,None]
var = np.append(var,df_var.loc[df_var.stim_ori==45, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)
std = np.sqrt(var.sum(1)/2)

# std = np.sqrt(var).sum(1)/2

d_prime = np.abs(np.diff(mu,axis=1))/std[:,None]

df_d = pd.DataFrame({'d' : d_prime.reshape(-1,),
                     'trained' : df_scm[df_scm.trained != 'Passive'].trained.to_numpy(),
                     'pref_ori' : df_scm[df_scm.trained != 'Passive'].pref_bin.to_numpy(),
                      #'pref_ori' : df_scm.pref_ori_train.to_numpy(),
                     'r' : df_scm[df_scm.trained != 'Passive'].r_bin.to_numpy(),
                     'subject' : df_scm[df_scm.trained != 'Passive'].subject.to_numpy()})

# df_d = pd.DataFrame({'d' : d_prime.reshape(-1,),
#                      'trained' : df_labels.trained.to_numpy(),
#                      'pref_ori' : df_labels.pref_bin.to_numpy(),
#                      #'pref_ori' : df_scm.pref_ori_all.to_numpy(),
#                      'r' : df_scm.r_bin_all.to_numpy(),
#                      'subject' : df_labels.subject.to_numpy()})


# task_responsive = ((df_d.pref_ori == 45) | (df_d.pref_ori == 90) |
#                     (df_d.pref_ori == 68) | (df_d.pref_ori == 23) |
#                     (df_d.pref_ori == 113))

# task_responsive = (df_d.pref_ori == 45) | (df_d.pref_ori == 90)

df_naive = pd.DataFrame.copy(df_d[df_d.trained == 'Naive'])
df_trained = pd.DataFrame.copy(df_d[df_d.trained == 'Proficient'])

# df_naive = df_d[np.logical_and(df_d.trained == False, task_responsive)]
# df_trained = df_d[np.logical_and(df_d.trained, task_responsive)]

df_naive = df_naive.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
df_trained = df_trained.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
df_trained = df_trained.reindex(df_naive.index)

# df_naive = df_d[df_d.trained == False].sort_values('d')
# df_trained = df_d[df_d.trained].sort_values('d')

# df_naive = df_naive.sort_values('d')
# df_trained = df_trained.sort_values('d')

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

    fig, ax = plt.subplots(1,2, sharex = True)

    d_min = -3
    d_max = 3

    cm1 = ax[0].imshow(df_naive.d.to_numpy()[:,None], cmap = 'RdBu_r',
               vmin = d_min, vmax = d_max, aspect = 'auto')
    # fig.colorbar(cm1,ax=ax[0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylabel('Cell classes')
    ax[0].set_title(r'Naïve - $d\prime$')

    cm2 = ax[1].imshow(df_trained.d.to_numpy()[:,None], cmap = 'RdBu_r',
               vmin = d_min, vmax = d_max, aspect = 'auto')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title(r'Proficient - $d\prime$')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(cm2, cax=cbar_ax)
    cb.set_label(r'd$^\prime$')

    df_naive = df_naive.sort_values(by=['subject','pref_ori','r'])
    df_naive.pref_ori = df_naive.pref_ori.astype(int).astype(str)
    df_trained = df_trained.sort_values(by=['subject','pref_ori','r'])
    df_trained.pref_ori = df_trained.pref_ori.astype(int).astype(str)

    # df_naive = df_naive.groupby(['subject','pref_ori']).mean().reset_index()
    # df_naive = df_naive.sort_values(by=['subject','pref_ori'])
    # df_naive.pref_ori = df_naive.pref_ori.astype(int).astype(str)
    # df_trained = df_trained.groupby(['subject','pref_ori']).mean().reset_index()
    # df_trained = df_trained.sort_values(by=['subject','pref_ori'])
    # df_trained.pref_ori = df_trained.pref_ori.astype(int).astype(str)

    ori_colors = sns.hls_palette(8)

    # df_naive.d = np.abs(df_naive.d)
    # df_trained.d = np.abs(df_trained.d)

    df_naive = pd.DataFrame.copy(df_d[df_d.trained == False])
    df_trained = pd.DataFrame.copy(df_d[df_d.trained == True])

    # df_naive = df_naive.groupby(['subject','pref_ori','r']).mean().reset_index()
    df_naive = df_naive.groupby(['subject','pref_ori']).mean().reset_index()
    # df_trained = df_trained.groupby(['subject','pref_ori','r']).mean().reset_index()
    df_trained = df_trained.groupby(['subject','pref_ori']).mean().reset_index()

    df_naive.d = np.abs(df_naive.d)
    df_trained.d = np.abs(df_trained.d)

    # f,a = plt.subplots(2,3)

    # uni_sub = np.unique(df_naive.subject)

    # for ia, s in enumerate(uni_sub):
    #     sd = sns.scatterplot(x = df_naive[df_naive.subject==s].d,
    #                          y = df_trained[df_trained.subject==s].d,
    #                     hue = df_naive[df_naive.subject==s].pref_ori, palette = ori_colors,
    #                     legend = False, s = 50, ax = a.flatten()[ia])
    #     a.flatten()[ia].plot([-2,2],[-2,2], '--k')
    #     sns.despine(ax=a.flatten()[ia])

    cond_colors = sns.color_palette('colorblind')[0:2]

    plt.figure(figsize=(1.5,1.5))
    sp = sns.scatterplot(x = df_naive.d,
                    y = df_trained.d,
                    hue = df_naive.pref_ori, palette = ori_colors,
                    legend = False, s = 6, linewidth = 0.5, edgecolor = 'black')

    plt.ylim([-0.1,2])
    plt.xlim([-0.1,2])
    plt.plot([0,2],[0,2], '--k')
    sp.set_aspect('equal','box')
    plt.xlabel(r'Naïve d')
    plt.ylabel(r'Trained d')
    sns.despine(trim = True)
    plt.tight_layout()

    df_all = pd.DataFrame.copy(df_d)
    # df_all = df_naive.append(df_trained)

    # df_all['d'] = df_all['d']**2
    # df_all = df_all.groupby(['subject','trained','pref_ori','r']).mean().reset_index()
    # df_all['d'] = np.sqrt(df_all['d'])

    df_all['d'] = df_all['d']**2
    df_all = df_all.groupby(['subject','trained','pref_ori']).mean().reset_index()
    df_all['d'] = np.sqrt(df_all['d'])

    df_diff = df_all[df_all.trained]
    df_diff['d'] = df_diff['d'].to_numpy() - df_all.loc[df_all.trained=='Naive','d'].to_numpy()

    df_diff['pref_ori'] = df_diff['pref_ori'].astype(int)

    sns.catplot(data = df_diff, x = 'pref_ori', y = 'd',
               kind = 'bar', palette = 'hls', aspect = 1.3, ci = 95)

    plt.ylim([-0.3,2])
    # sns.relplot(data = df_diff[df_diff.r >= 2], x = 'pref_ori', y = 'd', kind = 'line',
    #             n_boot = 2000)
    plt.xlabel('Preferred orientation (deg)')
    plt.ylabel(r'$\Delta$Population d$^\prime$')
    plt.tight_layout()

    df_all = pd.DataFrame.copy(df_d)
    df_all['d'] = df_all['d']**2
    df_all = df_all.groupby(['subject','trained', 'pref_ori']).mean().reset_index()
    df_all['d'] = np.sqrt(df_all['d'])

    task_ori_ind = np.logical_or(df_all.pref_ori==45,df_all.pref_ori==90)

    # ind_n = np.logical_and(task_ori_ind,df_all.trained==False)
    # ind_t = np.logical_and(task_ori_ind,df_all.trained==True)

    ind_n = df_all.trained == 'Naive'
    ind_t = df_all.trained == 'Proficient'

    sns.relplot(x = df_all[ind_n].d.to_numpy(),
                y = df_all[ind_t].d.to_numpy(), hue = df_all[ind_n].pref_ori.to_numpy().astype(str))
    plt.ylim([0,4])
    plt.xlim([0,4])
    plt.plot([0,5],[0,5],'--k')
    plt.tight_layout()

    df_all = pd.DataFrame.copy(df_d)

    d_th = 3

    df_all['high_d'] = np.abs(df_all.d) > d_th

    df_all = df_all.groupby(['subject','trained','pref_ori'])['high_d'].value_counts(normalize=True)\
        .rename('proportion_high_d').reset_index()

    df_all = df_all[df_all.high_d==False]
    df_all['proportion_high_d'] = 1 - df_all['proportion_high_d']
    df_all['pref_ori'] = df_all['pref_ori'].astype(int)

    ax_p = sns.catplot(data = df_all, x = 'pref_ori', y = 'proportion_high_d',
                hue = 'trained', kind = 'bar', palette = 'colorblind', ci = 95,
                aspect = 1.3, height = 5)
    # plt.xticks([])
    plt.xlabel('Preferred orientation (deg)')
    plt.ylabel(r'Proportion of cells with $\left|d^\prime\right| >$' + str(d_th))
    ax_p._legend.set_title('')
    ax_p._legend.texts[0].set_text('Naive')
    ax_p._legend.texts[1].set_text('Proficient')
    plt.ylim([0,0.135])
    plt.tight_layout()
    ax_p._legend.set_bbox_to_anchor([1,0.7])


    df_diff = df_all[df_all.trained]
    df_diff['proportion_high_d'] = (df_diff['proportion_high_d'].to_numpy() -
                                    df_all[df_all.trained == 'Naive'].proportion_high_d.to_numpy())


    ax_p = sns.catplot(data = df_diff, x = 'pref_ori', y = 'proportion_high_d',
                kind = 'bar', palette = 'hls', ci = 95, aspect = 1.3)
    # plt.xticks([])
    plt.xlabel('Preferred orientation (deg)')
    plt.ylabel(r'$\Delta$Proportion of cells with $\left|d^\prime\right| >$' + str(d_th))
    plt.ylim([-0.03,0.2])
    plt.tight_layout()

#%% Plot d' for 45 vs 90, crossvalidated ori pref with population d' by ori pref

# This is actually a calcuation of Cohen's d

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler

train_colors = sns.color_palette('colorblind')[0:2]
ori_colors = sns.color_palette('hls',8)

df = pd.DataFrame(trials_dict)

# Only include test set
df = df[~df.train_ind]

# Remove passive expts
df = df[df.trained != 'Passive']

# Only include 45 and 90 ori
df = df[np.logical_or(df.stim_ori == 45, df.stim_ori == 90)]

groups = ['subject', 'stim_dir', 'trained', 'cell']

# Average by stim dir
df_mu = df.groupby(groups, observed = True)['trial_resps'].mean().reset_index()

df_mu['stim_ori'] = df_mu.stim_dir % 180

# Average for both orientations
df_mu = df_mu.groupby(['subject','stim_ori','trained','cell'],observed = True).mean().reset_index()

df_var = df.groupby(groups, observed = True)['trial_resps'].var().reset_index()

df_var['stim_ori'] = df_var.stim_dir % 180

# Average for both orientations
df_var = df_var.groupby(['subject','stim_ori','trained','cell'],observed = True).mean().reset_index()

mu = df_mu.loc[df_mu.stim_ori == 90, 'trial_resps'].to_numpy()[:,None]
mu = np.append(mu,df_mu.loc[df_mu.stim_ori==45, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)
df_labels = df_mu.loc[df_mu.stim_ori == 45]

var = df_var.loc[df_var.stim_ori == 90, 'trial_resps'].to_numpy()[:,None]
var = np.append(var,df_var.loc[df_var.stim_ori==45, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)

# Pooled std (reduces down to this since number of trials is equal for both stimuli)
std = np.sqrt(var.sum(1)/2)

mu = np.diff(mu,axis=1)

d_prime = mu/std[:,None]

scaler = StandardScaler()

r_bin = pd.cut(df_scm.r,np.linspace(0,1,6),labels = np.arange(5)).to_numpy()

df_d = pd.DataFrame({'d' : d_prime.reshape(-1,),
                      'd_abs' : np.abs(d_prime.reshape(-1,)),
                     'trained' : df_scm[df_scm.trained != 'Passive'].trained.to_numpy(),
                     # 'pref_ori' : df_scm[df_scm.trained != 'Passive'].pref_bin.to_numpy(),
                     'pref_ori' : df_scm[df_scm.trained != 'Passive'].pref_ori_train.to_numpy(),
                     'th' : df_scm[df_scm.trained != 'Passive'].th.to_numpy(),
                      #'pref_ori' : df_scm.pref_ori_train.to_numpy(),
                     'r' : df_scm[df_scm.trained != 'Passive'].r_bin.to_numpy(),
                    #  'r' : r_bin,
                     'r_raw' : df_scm[df_scm.trained != 'Passive'].r.to_numpy(),
                     'subject' : df_scm[df_scm.trained != 'Passive'].subject.to_numpy(),
                     'mu' : mu.reshape(-1,),
                     'mu_abs' : np.abs(mu.reshape(-1,)),
                     'sigma' : std})

df_d['r_scaled'] = scaler.fit_transform(df_d.r_raw.to_numpy().reshape(-1,1))

df_d = df_d[np.logical_not(np.isnan(df_d.d))]


#%% Plot d vs selectivy, pref, and training


sns.scatterplot(data = df_d, x = 'r_raw', y = 'd_abs', hue = 'pref_ori',
                style = 'trained', palette = ori_colors, legend = False)

a = sns.displot(data = df_d[np.logical_or(df_d.pref_ori==45,df_d.pref_ori==90)], x = 'r_raw', y = 'd_abs',
                col = 'trained', palette = 'colorblind', legend = True,
                common_norm = False, bins = 20, alpha = 0.4)

for a in a.axes[0]:
    a.grid()

plt.figure(),sns.scatterplot(data = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)],
                x = 'r_raw', y = 'd_abs', hue = 'trained',
                palette = 'colorblind', legend = False,
                hue_order = ['Naive','Proficient'], s = 10)

plt.figure(),sns.ecdfplot(data = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)],
                x = 'd_abs', hue = 'trained',
                palette = 'colorblind', legend = True,
                hue_order = ['Naive','Proficient'])

plt.figure(),sns.ecdfplot(data = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)],
                x = 'mu', hue = 'trained',
                palette = 'colorblind', legend = True,
                hue_order = ['Naive','Proficient'])

plt.figure(),sns.ecdfplot(data = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)],
                x = 'mu_abs', hue = 'trained',
                palette = 'colorblind', legend = False,
                hue_order = ['Naive','Proficient'])

plt.figure(),sns.ecdfplot(data = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)],
                x = 'sigma', hue = 'trained',
                palette = 'colorblind', legend = False,
                hue_order = ['Naive','Proficient'])


plt.figure(), sns.scatterplot(data = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)],
                x = 'mu_abs', y = 'sigma', hue = 'trained',
                palette = 'colorblind', legend = False,
                hue_order = ['Naive','Proficient'], s = 10)

#%% Fit curve to selectivity and dprime, etc.

from scipy.interpolate import UnivariateSpline


# Selectivity vs d
x = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].r_raw.to_numpy()
y = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].d_abs.to_numpy()
trained_label = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].trained.to_numpy()

s_ind = np.argsort(x)
x = x[s_ind]
y = y[s_ind]
trained_label = trained_label[s_ind]

x_n = x[trained_label == 'Naive']
x_t = x[trained_label=='Proficient']

y_n = y[trained_label == 'Naive']
y_t = y[trained_label == 'Proficient']

spl_n = UnivariateSpline(x_n, y_n)
spl_n.set_smoothing_factor(4000)

spl_t = UnivariateSpline(x_t, y_t)
spl_t.set_smoothing_factor(4000)


f,a = plt.subplots(1,1)

sns.scatterplot(x = x_n,y = y_n,color=sns.color_palette('colorblind',2)[0], ax = a, s = 10,
                alpha = 0.5)
a.plot(x_n,spl_n(x_n),color = sns.color_palette('colorblind',2)[0])

sns.scatterplot(x = x_t,y = y_t,color=sns.color_palette('colorblind',2)[1], ax = a, s = 10,
                alpha = 0.5)
a.plot(x_t,spl_t(x_t),color = sns.color_palette('colorblind',2)[1])

a.set_xlabel('Selectivity')
a.set_ylabel('d')


# mean diff vs sigma
x = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].mu_abs.to_numpy()
y = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].sigma.to_numpy()
trained_label = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].trained.to_numpy()
d = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].d_abs.to_numpy()

s_ind = np.argsort(x)
x = x[s_ind]
y = y[s_ind]
d = d[s_ind]
trained_label = trained_label[s_ind]

x_n = x[trained_label == 'Naive']
x_t = x[trained_label=='Proficient']

y_n = y[trained_label == 'Naive']
y_t = y[trained_label == 'Proficient']

d_n = d[trained_label == 'Naive']
d_t = d[trained_label == 'Proficient']

spl_n = UnivariateSpline(x_n, y_n)
spl_n.set_smoothing_factor(4000)

spl_t = UnivariateSpline(x_t, y_t)
spl_t.set_smoothing_factor(4000)

f,a = plt.subplots(1,1)

sns.scatterplot(x = x_n,y = y_n,color=sns.color_palette('colorblind',2)[0], ax = a, size = d_n,
                alpha = 0.5, legend = False)
a.plot(x_n,spl_n(x_n),color = sns.color_palette('colorblind',2)[0])

sns.scatterplot(x = x_t,y = y_t,color=sns.color_palette('colorblind',2)[1], ax = a, size = d_t,
                alpha = 0.5, legend = False)
a.plot(x_t,spl_t(x_t),color = sns.color_palette('colorblind',2)[1])

a.set_xlabel('Abs. mean diff.')
a.set_ylabel('Pooled std')



# mean diff vs d
x = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].mu_abs.to_numpy()
y = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].d_abs.to_numpy()
trained_label = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].trained.to_numpy()
r = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].sigma.to_numpy()

s_ind = np.argsort(x)
x = x[s_ind]
y = y[s_ind]
r = r[s_ind]
trained_label = trained_label[s_ind]

x_n = x[trained_label == 'Naive']
x_t = x[trained_label=='Proficient']

y_n = y[trained_label == 'Naive']
y_t = y[trained_label == 'Proficient']

r_n = r[trained_label == 'Naive']
r_t = r[trained_label == 'Proficient']

spl_n = UnivariateSpline(x_n, y_n)
spl_n.set_smoothing_factor(3700)

spl_t = UnivariateSpline(x_t, y_t)
spl_t.set_smoothing_factor(3700)

f,a = plt.subplots(1,1)

sns.scatterplot(x = x_n,y = y_n,color=sns.color_palette('colorblind',2)[0], ax = a, size = r_n,
                alpha = 0.5, legend = False)
a.plot(x_n,spl_n(x_n),color = sns.color_palette('colorblind',2)[0])

sns.scatterplot(x = x_t,y = y_t,color=sns.color_palette('colorblind',2)[1], ax = a, size = r_t,
                alpha = 0.5, legend = False)
a.plot(x_t,spl_t(x_t),color = sns.color_palette('colorblind',2)[1])

a.set_xlabel('Abs. mean diff.')
a.set_ylabel('d')

# sigma vs d
# Only include cells with moderate difference in mean
mu_ind = np.logical_and(df_d.mu_abs > 0.6, df_d.sigma < 2)
stim_ind = np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)
ind = mu_ind & stim_ind
x = df_d[ind].sigma.to_numpy()
y = df_d[ind].d_abs.to_numpy()
trained_label = df_d[ind].trained.to_numpy()
r = df_d[ind].mu_abs.to_numpy()

s_ind = np.argsort(x)
x = x[s_ind]
y = y[s_ind]
r = r[s_ind]
trained_label = trained_label[s_ind]

x_n = x[trained_label == 'Naive']
x_t = x[trained_label=='Proficient']

y_n = y[trained_label == 'Naive']
y_t = y[trained_label == 'Proficient']

r_n = r[trained_label == 'Naive']
r_t = r[trained_label == 'Proficient']

spl_n = UnivariateSpline(x_n, y_n)
spl_n.set_smoothing_factor(150)

spl_t = UnivariateSpline(x_t, y_t)
spl_t.set_smoothing_factor(150)

f,a = plt.subplots(1,1)

sns.scatterplot(x = x_n,y = y_n,color=sns.color_palette('colorblind',2)[0], ax = a, size = r_n,
                alpha = 0.5, legend = False)
a.plot(x_n,spl_n(x_n),color = sns.color_palette('colorblind',2)[0])

sns.scatterplot(x = x_t,y = y_t,color=sns.color_palette('colorblind',2)[1], ax = a, size = r_t,
                alpha = 0.5, legend = False)
a.plot(x_t,spl_t(x_t),color = sns.color_palette('colorblind',2)[1])

a.set_xlabel('Pooled std')
a.set_ylabel('d')


#%% Flipped axes


from scipy.interpolate import UnivariateSpline


# d vs selectivity
y = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].r_raw.to_numpy()
x = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].d_abs.to_numpy()
trained_label = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].trained.to_numpy()

s_ind = np.argsort(x)
x = x[s_ind]
y = y[s_ind]
trained_label = trained_label[s_ind]

x_n = x[trained_label == 'Naive']
x_t = x[trained_label=='Proficient']

y_n = y[trained_label == 'Naive']
y_t = y[trained_label == 'Proficient']

spl_n = UnivariateSpline(x_n, y_n)
spl_n.set_smoothing_factor(90)

spl_t = UnivariateSpline(x_t, y_t)
spl_t.set_smoothing_factor(60)


f,a = plt.subplots(1,1)

sns.scatterplot(x = x_n,y = y_n,color=sns.color_palette('colorblind',2)[0], ax = a, s = 10,
                alpha = 0.5)
a.plot(x_n,spl_n(x_n),color = sns.color_palette('colorblind',2)[0])

sns.scatterplot(x = x_t,y = y_t,color=sns.color_palette('colorblind',2)[1], ax = a, s = 10,
                alpha = 0.5)
a.plot(x_t,spl_t(x_t),color = sns.color_palette('colorblind',2)[1])

a.set_ylabel('Selectivity')
a.set_xlabel('d')




# d vs mean diff
y = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].mu_abs.to_numpy()
x = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].d_abs.to_numpy()
trained_label = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].trained.to_numpy()
r = df_d[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)].sigma.to_numpy()

s_ind = np.argsort(x)
x = x[s_ind]
y = y[s_ind]
r = r[s_ind]
trained_label = trained_label[s_ind]

x_n = x[trained_label == 'Naive']
x_t = x[trained_label=='Proficient']

y_n = y[trained_label == 'Naive']
y_t = y[trained_label == 'Proficient']

r_n = r[trained_label == 'Naive']
r_t = r[trained_label == 'Proficient']

spl_n = UnivariateSpline(x_n, y_n, k = 5)
# spl_n.set_smoothing_factor(100)

spl_t = UnivariateSpline(x_t, y_t, k = 5)
# spl_t.set_smoothing_factor(100)

f,a = plt.subplots(1,1)

sns.scatterplot(x = x_n,y = y_n,color=sns.color_palette('colorblind',2)[0], ax = a, size = r_n,
                alpha = 0.5, legend = False)
a.plot(x_n,spl_n(x_n),color = sns.color_palette('colorblind',2)[0])

sns.scatterplot(x = x_t,y = y_t,color=sns.color_palette('colorblind',2)[1], ax = a, size = r_t,
                alpha = 0.5, legend = False)
a.plot(x_t,spl_t(x_t),color = sns.color_palette('colorblind',2)[1])

a.set_ylabel('Abs. mean diff.')
a.set_xlabel('d')

# sigma vs d
# Only include cells with moderate difference in mean
mu_ind = np.logical_and(df_d.mu_abs > 0.6, df_d.sigma < 2)
stim_ind = np.logical_or(df_d.pref_ori == 45, df_d.pref_ori==90)
ind = mu_ind & stim_ind
y = df_d[ind].sigma.to_numpy()
x = df_d[ind].d_abs.to_numpy()
trained_label = df_d[ind].trained.to_numpy()
r = df_d[ind].mu_abs.to_numpy()

s_ind = np.argsort(x)
x = x[s_ind]
y = y[s_ind]
r = r[s_ind]
trained_label = trained_label[s_ind]

x_n = x[trained_label == 'Naive']
x_t = x[trained_label=='Proficient']

y_n = y[trained_label == 'Naive']
y_t = y[trained_label == 'Proficient']

r_n = r[trained_label == 'Naive']
r_t = r[trained_label == 'Proficient']

spl_n = UnivariateSpline(x_n, y_n, k = 5)
# spl_n.set_smoothing_factor(50)

spl_t = UnivariateSpline(x_t, y_t, k = 5)
# spl_t.set_smoothing_factor(50)

f,a = plt.subplots(1,1)

sns.scatterplot(x = x_n,y = y_n,color=sns.color_palette('colorblind',2)[0], ax = a, size = r_n,
                alpha = 0.5, legend = False)
# a.plot(x_n,spl_n(x_n),color = sns.color_palette('colorblind',2)[0])

sns.scatterplot(x = x_t,y = y_t,color=sns.color_palette('colorblind',2)[1], ax = a, size = r_t,
                alpha = 0.5, legend = False)
# a.plot(x_t,spl_t(x_t),color = sns.color_palette('colorblind',2)[1])

a.set_ylabel('Pooled std')
a.set_xlabel('d')



#%%

df_d_45 = df_d[df_d.pref_ori == 45].reset_index(drop=True)

df_d_45['th_dist'] = df_d_45.th - 45
df_d_45['th_dist_bin'] = pd.cut(df_d_45.th_dist, np.linspace(-11.25,11.25,8),labels = np.linspace(-11.25,11.25,7))

# n_cells = [len(df_d_45[df_d_45.subject == s]) for s in np.unique(df_d_45.subject)]

# k_cells = 500

# df_d_45_prun = pd.DataFrame(columns = df_d_45.columns)

# for s in np.unique(df_d_45.subject):
#     n_cells = len(df_d_45[df_d_45.subject==s])
#     ind = df_d_45[df_d_45.subject == s].index[np.random.choice(n_cells,500, replace = False)]
#     df_d_45_prun = pd.concat([df_d_45_prun, df_d_45.iloc[ind]],axis = 0)

# Random intercepts
model = smf.mixedlm('d ~ C(trained)', df_d_45, groups='subject').fit(reml = False)
print('====Random interecepts====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# # Random intercepts and slopes, correlated
model = smf.mixedlm('d ~ C(trained)', df_d_45, groups='subject', re_formula = '1 + C(trained)').fit(reml = False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# # Random intercepts and slopes, uncorrelated
model = smf.mixedlm('d ~ C(trained)', df_d_45, groups='subject', vc_formula = {'trained' : '0 + C(trained)'}).fit(reml = False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# With selectivity
# Random intercepts
model = smf.mixedlm('d ~ C(trained) * C(r)', df_d_45, groups='subject').fit(reml=False)
print('====Random interecepts====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# Random intercepts and slopes, correlated
model = smf.mixedlm('d ~ C(trained) * C(r)', df_d_45, groups='subject', re_formula = '1 + C(trained) * C(r)').fit(reml=False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# Random intercepts and slopes, uncorrelated
model = smf.mixedlm('d ~ C(trained) * C(r)', df_d_45, groups='subject', vc_formula = {'trained' : '0 + C(trained)',
                                                                            'r' : '0 + C(r)'}).fit(reml=False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))

df_d_90 = df_d[df_d.pref_ori == 90].reset_index(drop=True)

df_d_90['th_dist'] = df_d_90.th - 90
df_d_90['th_dist_bin'] = pd.cut(df_d_90.th_dist, np.linspace(-11.25,11.25,8),labels = np.linspace(-11.25,11.25,7))

# Random intercepts
model = smf.mixedlm('d ~ C(trained)', df_d_90, groups='subject').fit(method='powell', reml = False)
print('====Random interecepts====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# Random intercepts and slopes, correlated
model = smf.mixedlm('d ~ C(trained)', df_d_90, groups='subject', re_formula = '1 + C(trained)').fit(method='powell', reml = False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# Random intercepts and slopes, uncorrelated
model = smf.mixedlm('d ~ C(trained)', df_d_90, groups='subject', vc_formula = {'trained' : '0 + C(trained)'}).fit(method='powell', reml = False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# With selectivity
model = smf.mixedlm('d ~ C(trained) * C(r)', df_d_90, groups='subject').fit(method='powell', reml = False)
print('====Random interecepts====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# Random intercepts and slopes, correlated
model = smf.mixedlm('d ~ C(trained) * C(r)', df_d_90, groups='subject', re_formula = '1 + C(trained) * C(r)').fit(method='powell', reml = False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))

# Random intercepts and slopes, uncorrelated
model = smf.mixedlm('d ~ C(trained) * C(r)', df_d_90, groups='subject', vc_formula = {'trained' : '0 + C(trained)',
                                                                                      'r' : '0 + C(r)'}).fit(method='powell', reml = False)
print('====Random intercepts and slopes, Correlated====\n')
print(model.summary())
print('AIC :' + str(model.aic))


#%% For each orientation preference, fit model for interaction of training and selectivity for dprime

prefs = np.unique(df_d.pref_ori)

pvals = np.zeros((3,len(prefs)))

for i,p in enumerate(prefs):
    model = smf.mixedlm('d ~ C(trained)',df_d[df_d.pref_ori == p],groups = 'subject').fit(method = 'powell')
    pvals[0,i] = model.pvalues[1]
    model = smf.mixedlm('mu ~ C(trained)',df_d[df_d.pref_ori == p],groups = 'subject').fit(method = 'powell')
    pvals[1,i] = model.pvalues[1]
    model = smf.mixedlm('sigma ~ C(trained)',df_d[df_d.pref_ori == p],groups = 'subject').fit(method = 'powell')
    pvals[2,i] = model.pvalues[1]


# for i in range(3):
#     pvals[i,:] = multipletests(pvals[i,:],method='hs')[1]

#%% Plot average change across all cells in d, diff in mu, sigma

ori_colors = sns.color_palette('hls',8)
cond_colors = sns.color_palette('colorblind', 2)


df_d_mean = df_d.groupby(['trained','pref_ori', 'r']).mean().reset_index()
df_d_mean['d'] = df_d_mean.d
df_d_mean['mu'] = df_d_mean.mu
df_d_sem = df_d.groupby(['trained','pref_ori', 'r']).sem().reset_index()

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

    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams["font.family"] = "sans-serif"

    x = np.abs(df_d_mean[df_d_mean.trained=='Naive'].d.to_numpy())
    y = np.abs(df_d_mean[df_d_mean.trained=='Proficient'].d.to_numpy())
    x_sem = df_d_sem[df_d_mean.trained=='Naive'].d.to_numpy()
    y_sem = df_d_sem[df_d_mean.trained=='Proficient'].d.to_numpy()
    hue = df_d_mean[df_d_mean.trained=='Proficient'].pref_ori.to_numpy().astype(int).astype(str)
    style = df_d_mean[df_d_mean.trained=='Proficient'].r.to_numpy()

    # f = plt.figure(figsize=(4.8,1.6))
    f = plt.figure()

    ax = f.add_subplot(1,3,1, aspect = 'equal')
    sns.scatterplot(x=x,y=y,hue=hue, zorder = 100, palette = ori_colors,
    ax = ax, edgecolor = 'k', linewidth = 0.5, style = style, legend = False)
    # ax.legend_.remove()
    # ax.legend(ncol = 2, loc = 4, labelspacing = 0.2, columnspacing = 1)
    # ax.legend_.set_frame_on(False)
    # ax.legend_.set_title('Ori. pref. (mean)')
    ax.errorbar(x,y,y_sem,x_sem, ls = 'None', barsabove = False, color = 'k',
                linewidth = 0.5)
    ax.set_xlim([0-0.06,5])
    ax.set_ylim([0-0.06,5])
    ax.set_yticks(np.linspace(0,5,6))
    ax.set_xticks(np.linspace(0,5,6))
    ax.plot([0, 5], [0, 5], '--k', linewidth = 0.5)
    ax.set_xlabel(r'Naïve $|d^\prime|$', color = cond_colors[0], labelpad = 2)
    ax.set_ylabel(r'Proficient $|d^\prime|$', color = cond_colors[1], labelpad = 2)
    sns.despine(trim=True)
    # ax.text(0,1.2, r'Ori. pref. 45$\degree$'+'\np < 0.01\n' + r'Ori. pref. 90$\degree$' +
    #                   '\np < 0.0001', fontsize = 5, horizontalalignment = 'left', verticalalignment = 'top')
    # f.tight_layout()

    # f.savefig(join(fig_save_dir,'d_naive_vs_proficient.svg'), format = 'svg')

    ind_n = np.logical_and(df_d_mean.trained=='Naive', (df_d_mean.pref_ori==45) | (df_d_mean.pref_ori==90))
    ind_p = np.logical_and(df_d_mean.trained=='Proficient', (df_d_mean.pref_ori==45) | (df_d_mean.pref_ori==90))

    x = np.abs(df_d_mean[ind_n].mu.to_numpy())
    y = np.abs(df_d_mean[ind_p].mu.to_numpy())
    x_sem = df_d_sem[ind_n].mu.to_numpy()
    y_sem = df_d_sem[ind_p].mu.to_numpy()
    hue = df_d_mean[ind_p].pref_ori.to_numpy().astype(int).astype(str)
    style = df_d_mean[ind_p].r.to_numpy()

    # f = plt.figure(figsize=(1.6,1.6))
    ax = f.add_subplot(1,3,2, aspect = 'equal')
    sns.scatterplot(x=x,y=y,hue=hue, zorder = 100, palette = ori_colors[slice(2,6,2)],
    ax = ax, edgecolor = 'k', linewidth = 0.5, style = style)
    ax.legend_.remove()
    ax.errorbar(x,y,y_sem,x_sem, ls = 'None', barsabove = False, color = 'k',
                linewidth = 0.5)
    ax.set_xlim([0-0.01,1])
    ax.set_ylim([0-0.01,1])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_xticks(np.linspace(0,1,5))
    ax.plot([0, 1], [0, 1], '--k', linewidth = 0.5)
    ax.set_xlabel(r'Naïve $|\mu_{45} - \mu_{90}|$', color = cond_colors[0], labelpad = 2)
    ax.set_ylabel(r'Proficient $|\mu_{45} - \mu_{90}|$', color = cond_colors[1], labelpad = 2)
    sns.despine(trim=True)
    # ax.text(0.25,0.45, r'Ori. pref. 45$\degree$'+'\np = 0.261\n' + r'Ori. pref. 90$\degree$' +
    #                   '\np < 0.05', fontsize = 5, horizontalalignment = 'left', verticalalignment = 'top')
    # f.tight_layout()

    # f.savefig(join(fig_save_dir,'mu_diff_naive_vs_proficient.svg'), format = 'svg')


    x = np.abs(df_d_mean[ind_n].sigma.to_numpy())
    y = np.abs(df_d_mean[ind_p].sigma.to_numpy())
    x_sem = df_d_sem[ind_n].sigma.to_numpy()
    y_sem = df_d_sem[ind_p].sigma.to_numpy()
    hue = df_d_mean[ind_p].pref_ori.to_numpy().astype(int).astype(str)
    style = df_d_mean[ind_p].r.to_numpy()

    # f = plt.figure(figsize=(1.6,1.6))
    ax = f.add_subplot(1,3,3, aspect = 'equal')
    sns.scatterplot(x=x,y=y,hue=hue, zorder = 100, palette = ori_colors[slice(2,6,2)],
    ax = ax, edgecolor = 'k', linewidth = 0.5, style = style)
    ax.legend_.remove()
    ax.errorbar(x,y,y_sem,x_sem, ls = 'None', barsabove = False, color = 'k',
                linewidth = 0.5)
    ax.set_xlim([0-0.01,1])
    ax.set_ylim([0-0.01,1])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_xticks(np.linspace(0,1,5))
    ax.plot([0, 1], [0, 1], '--k', linewidth = 0.5)
    ax.set_xlabel(r'Naïve $\sigma_{45,90}$', color = cond_colors[0], labelpad = 2)
    ax.set_ylabel(r'Proficient $\sigma_{45,90}$', color = cond_colors[1], labelpad = 2)
    sns.despine(trim=True)
    # ax.text(0.45,0.65, r'Ori. pref. 45$\degree$'+'\np < 0.0001\n' + r'Ori. pref. 90$\degree$' +
    #                   '\np < 0.0001', fontsize = 5, horizontalalignment = 'left', verticalalignment = 'top')
    f.tight_layout()

    # f.savefig(join(fig_save_dir,'d_mu_sigma_naive_vs_proficient.svg'), format = 'svg')

#%%

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

    # "population d'"
    df_pop_d = pd.DataFrame.copy(df_d)
    # df_pop_d.d **= 2
    df_pop_d = df_pop_d.groupby(['subject','pref_ori','trained']).mean().reset_index()
    # df_pop_d.d = np.sqrt(df_pop_d.d)

    pvals_d = []
    n_ind = (df_pop_d.pref_ori == 45) & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_ori == 45) & (df_pop_d.trained)
    pvals_d.append(ttest_rel(df_pop_d[n_ind].d, df_pop_d[p_ind].d)[1])

    diff_45 = df_pop_d[p_ind].d.to_numpy() - df_pop_d[n_ind].d.to_numpy()

    n_ind = (df_pop_d.pref_ori == 90) & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_ori == 90) & (df_pop_d.trained)
    pvals_d.append(ttest_rel(df_pop_d[n_ind].d, df_pop_d[p_ind].d)[1])

    diff_90 = df_pop_d[p_ind].d.to_numpy() - df_pop_d[n_ind].d.to_numpy()


    # n_ind = np.logical_and(df_pop_d.pref_ori != 45,df_pop_d.pref_ori != 90) & (df_pop_d.trained==False)
    # p_ind = np.logical_and(df_pop_d.pref_ori != 45,df_pop_d.pref_ori != 90) & (df_pop_d.trained)
    # pvals.append(ttest_rel(df_pop_d[n_ind].d, df_pop_d[p_ind].d)[1])

    # diff_nontask = df_pop_d[p_ind].d.to_numpy() - df_pop_d[n_ind].d.to_numpy()


    f,ax = plt.subplots(1,1, figsize = (1.62,1.62))
    sns.scatterplot(x = df_pop_d[df_pop_d.trained==False].d.to_numpy(),
                    y = df_pop_d[df_pop_d.trained].d.to_numpy(),
                    hue = df_pop_d[df_pop_d.trained].pref_ori.to_numpy().astype(int).astype(str),
                    ax = ax, legend = True, palette = ori_colors,
                    edgecolor = 'k', linewidth = 0.5, s = 6)
    ax.set_box_aspect(1)
    ax.set_ylim([-0.13,2.2])
    ax.set_xlim([-0.13,2.2])
    ax.set_xticks(np.arange(0,3))
    ax.set_yticks(np.arange(0,3))
    h,l = ax.get_legend_handles_labels()
    ax.legend_.remove()
    ax.legend(h,l, frameon=False, title='Ori. pref. (mean)', ncol = 2, loc = 4,
              labelspacing = 0.2, columnspacing = 1)
    ax.plot([0,2],[0,2],'--k')
    sns.despine(trim=True)
    ax.set_xlabel(r'Naïve $d^\prime$')
    ax.set_ylabel(r'Proficient $d^\prime$')

    f.savefig('/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/' +
              'dprime_naive vs proficient.svg', format = 'svg')

    df_d_ave = df_d[np.logical_or(df_d.pref_ori == 45,
                                  df_d.pref_ori == 90)].groupby(
                                      ['subject','pref_ori','trained']).mean().reset_index()


    pvals_mu = []
    n_ind = (df_pop_d.pref_ori == 45) & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_ori == 45) & (df_pop_d.trained)
    pvals_mu.append(ttest_rel(df_pop_d[n_ind].mu, df_pop_d[p_ind].mu)[1])

    diff_45_mu = df_pop_d[p_ind].mu.to_numpy() - df_pop_d[n_ind].mu.to_numpy()

    n_ind = (df_pop_d.pref_ori == 90) & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_ori == 90) & (df_pop_d.trained)
    pvals_mu.append(ttest_rel(df_pop_d[n_ind].mu, df_pop_d[p_ind].mu)[1])

    diff_90_mu = df_pop_d[p_ind].mu.to_numpy() - df_pop_d[n_ind].mu.to_numpy()

    # Mean - Only 45 and 90 preferring
    f,ax = plt.subplots(1,1, figsize=(1.5,1.5))
    sns.scatterplot(x = np.abs(df_d_ave[df_d_ave.trained==False].mu.to_numpy()),
                    y = np.abs(df_d_ave[df_d_ave.trained].mu.to_numpy()),
                    hue = df_d_ave[df_d_ave.trained].pref_ori.to_numpy().astype(str),
                    ax = ax, legend = False, palette = ori_colors[slice(2,6,2)],
                    edgecolor = 'k', linewidth = 0.5, s = 6)
    ax.set_box_aspect(1)
    ax.set_ylim([0.28,0.55])
    ax.set_xlim([0.28,0.55])
    ax.set_xticks(np.linspace(0.3,0.6,5))
    ax.set_yticks(np.linspace(0.3,0.6,5))
    # ax.legend_.set_frame_on(False)
    # ax.legend_.set_title('Ori. pref. (mean)')
    ax.plot([0.3,0.6],[0.3,0.6],'--k')
    sns.despine(trim=True)
    ax.set_xlabel(r'Naïve $|\Delta\mu|$')
    ax.set_ylabel(r'Proficient $|\Delta\mu|$')

    f.savefig('/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/' +
              'dprime_delta mu_naive vs proficient.svg', format = 'svg')


    pvals_sigma = []
    n_ind = (df_pop_d.pref_ori == 45) & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_ori == 45) & (df_pop_d.trained)
    pvals_sigma.append(ttest_rel(df_pop_d[n_ind].sigma, df_pop_d[p_ind].sigma)[1])

    diff_45_sigma = df_pop_d[p_ind].sigma.to_numpy() - df_pop_d[n_ind].sigma.to_numpy()

    n_ind = (df_pop_d.pref_ori == 90) & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_ori == 90) & (df_pop_d.trained)
    pvals_sigma.append(ttest_rel(df_pop_d[n_ind].sigma, df_pop_d[p_ind].sigma)[1])

    diff_90_sigma = df_pop_d[p_ind].sigma.to_numpy() - df_pop_d[n_ind].sigma.to_numpy()

    # STD - Only 45 and 90 preferring
    f,ax = plt.subplots(1,1, figsize=(1.5,1.5))
    sns.scatterplot(x = df_d_ave[df_d_ave.trained==False].sigma.to_numpy(),
                    y = df_d_ave[df_d_ave.trained].sigma.to_numpy(),
                    hue = df_d_ave[df_d_ave.trained].pref_ori.to_numpy().astype(str),
                    ax = ax, legend = False, palette = ori_colors[slice(2,6,2)],
                    edgecolor = 'k', linewidth = 0.5, s = 6)
    ax.set_box_aspect(1)
    ax.set_ylim([0.4,0.7])
    ax.set_xlim([0.4,0.7])
    ax.set_xticks(np.linspace(0.4,0.7,5))
    ax.set_yticks(np.linspace(0.4,0.7,5))
    # ax.legend_.set_frame_on(False)
    # ax.legend_.set_title('Ori. pref. (mean)')
    ax.plot([0.4,0.7],[0.4,0.7],'--k')
    sns.despine(trim=True)
    ax.set_xlabel(r'Naïve $\sigma$')
    ax.set_ylabel(r'Proficient $\sigma$')

    f.savefig('/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/' +
              'dprime_sigma_naive vs proficient.svg', format = 'svg')

    # Heatmap
    # d_min = -3
    # d_max = 3

    # df_naive = pd.DataFrame.copy(df_d[df_d.trained==False])
    # df_trained = pd.DataFrame.copy(df_d[df_d.trained])

    # df_naive = df_naive.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
    # df_trained = df_trained.groupby(['subject','pref_ori','r']).mean().reset_index().reindex(df_naive.index)

    # f,ax = plt.subplots(1,2)

    # sns.heatmap(df_naive.d.to_numpy()[:,None], vmin = d_min, vmax = d_max, ax = ax[0], cmap = 'RdBu_r')
    # sns.heatmap(df_trained.d.to_numpy()[:,None], vmin = d_min, vmax = d_max, ax = ax[1], cmap = 'RdBu_r')


#%% Plot d' for 45 vs 90, group cells by orientation pref. 45 and 90, 68, 135 vs 0

train_colors = sns.color_palette('colorblind')[0:2]
ori_colors = sns.color_palette('hls',4)

df = pd.DataFrame(trials_dict)

# Only include test set
df = df[~df.train_ind]

# Remove blank
df = df[df.stim_ori != np.inf]

groups = ['subject', 'stim_dir', 'trained', 'cell']

df_var = df.groupby(groups, observed = True)['trial_resps'].var().reset_index()

df_var['stim_ori'] = df_var.stim_dir % 180

df_var = df_var.groupby(['subject','stim_ori','trained','cell'],observed = True).mean().reset_index()

df_mu = df.groupby(groups, observed = True)['trial_resps'].mean().reset_index()

df_mu['stim_ori'] = df_mu.stim_dir % 180

df_mu = df_mu.groupby(['subject','stim_ori','trained','cell'],observed = True).mean().reset_index()

mu = df_mu.loc[df_mu.stim_ori == 90, 'trial_resps'].to_numpy()[:,None]
mu = np.append(mu,df_mu.loc[df_mu.stim_ori==45, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)
df_labels = df_mu.loc[df_mu.stim_ori == 45]

var = df_var.loc[df_var.stim_ori == 90, 'trial_resps'].to_numpy()[:,None]
var = np.append(var,df_var.loc[df_var.stim_ori==45, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)
std = np.sqrt(var.sum(1)/2)

mu = np.abs(np.diff(mu,axis=1))

d_prime = mu/std[:,None]

df_d = pd.DataFrame({'d' : d_prime.reshape(-1,),
                     'd_abs' : np.abs(d_prime.reshape(-1,)),
                     'trained' : df_scm.trained.to_numpy(),
                     'pref_ori' : df_scm.pref_bin.to_numpy(),
                    #   'pref_ori' : df_scm.pref_ori_train.to_numpy(),
                     'r' : df_scm.r_bin.to_numpy(),
                     'subject' : df_scm.subject.to_numpy(),
                     'mu' : mu.reshape(-1,),
                     'sigma' : std})

df_d['pref_type'] = 'other'

df_d.loc[np.logical_or(df_d.pref_ori == 45, df_d.pref_ori == 90),'pref_type'] = 'informative'
df_d.loc[df_d.pref_ori == 68, 'pref_type'] = 'distractor'
df_d.loc[np.logical_or(df_d.pref_ori == 135, df_d.pref_ori == 0),'pref_type'] = 'control'

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

    # "population d'"
    df_pop_d = pd.DataFrame.copy(df_d)
    # df_pop_d.d **= 2
    df_pop_d = df_pop_d.groupby(['subject','pref_type','trained']).mean().reset_index()
    # df_pop_d.d = np.sqrt(df_pop_d.d)

    pvals = []
    n_ind = (df_pop_d.pref_type == 'informative') & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_type == 'informative') & (df_pop_d.trained)
    pvals.append(ttest_rel(df_pop_d[n_ind].d, df_pop_d[p_ind].d)[1])

    diff_task = df_pop_d[p_ind].d.to_numpy() - df_pop_d[n_ind].d.to_numpy()

    n_ind = (df_pop_d.pref_type == 'distractor') & (df_pop_d.trained==False)
    p_ind = (df_pop_d.pref_type == 'distractor') & (df_pop_d.trained)
    pvals.append(ttest_rel(df_pop_d[n_ind].d, df_pop_d[p_ind].d)[1])

    diff_nontask = df_pop_d[p_ind].d.to_numpy() - df_pop_d[n_ind].d.to_numpy()


    f,ax = plt.subplots(1,1, figsize = (1.5,1.5))
    sns.scatterplot(x = df_pop_d[df_pop_d.trained==False].d.to_numpy(),
                    y = df_pop_d[df_pop_d.trained].d.to_numpy(),
                    hue = df_pop_d[df_pop_d.trained].pref_type.to_numpy(),
                    ax = ax, legend = True,
                    edgecolor = 'k', linewidth = 0.5, s = 6)
    ax.set_box_aspect(1)
    ax.set_ylim([-0.13,2.2])
    ax.set_xlim([-0.13,2.2])
    ax.set_xticks(np.arange(0,3))
    ax.set_yticks(np.arange(0,3))
    h,l = ax.get_legend_handles_labels()
    ax.legend_.remove()
    ax.legend(h,l, frameon=False, title='Ori. pref. (mean)', ncol = 2, loc = 4,
              labelspacing = 0.2, columnspacing = 1)
    ax.plot([0,2],[0,2],'--k')
    sns.despine(trim=True)
    ax.set_xlabel(r'Naïve $d^\prime$')
    ax.set_ylabel(r'Proficient $d^\prime$')

    # f.savefig('/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/' +
    #           'dprime_naive vs proficient.svg', format = 'svg')

    df_d_ave = df_d[np.logical_or(df_d.pref_ori == 45,
                                  df_d.pref_ori == 90)].groupby(
                                      ['subject','pref_ori','trained']).mean().reset_index()

    # Mean - Only 45 and 90 preferring
    f,ax = plt.subplots(1,1, figsize=(1.5,1.5))
    sns.scatterplot(x = np.abs(df_d_ave[df_d_ave.trained==False].mu.to_numpy()),
                    y = np.abs(df_d_ave[df_d_ave.trained].mu.to_numpy()),
                    hue = df_d_ave[df_d_ave.trained].pref_ori.to_numpy(),
                    ax = ax, legend = False,
                    edgecolor = 'k', linewidth = 0.5, s = 6)
    ax.set_box_aspect(1)
    ax.set_ylim([0.18,0.55])
    ax.set_xlim([0.18,0.55])
    ax.set_xticks(np.linspace(0.2,0.6,5))
    ax.set_yticks(np.linspace(0.2,0.6,5))
    # ax.legend_.set_frame_on(False)
    # ax.legend_.set_title('Ori. pref. (mean)')
    ax.plot([0.2,0.6],[0.2,0.6],'--k')
    sns.despine(trim=True)
    ax.set_xlabel(r'Naïve $|\Delta\mu|$')
    ax.set_ylabel(r'Proficient $|\Delta\mu|$')

    # f.savefig('/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/' +
    #           'dprime_delta mu_naive vs proficient.svg', format = 'svg')

    # STD - Only 45 and 90 preferring
    f,ax = plt.subplots(1,1, figsize=(1.5,1.5))
    sns.scatterplot(x = df_d_ave[df_d_ave.trained==False].sigma.to_numpy(),
                    y = df_d_ave[df_d_ave.trained].sigma.to_numpy(),
                    hue = df_d_ave[df_d_ave.trained].pref_ori.to_numpy(),
                    ax = ax, legend = False,
                    edgecolor = 'k', linewidth = 0.5, s = 6)
    ax.set_box_aspect(1)
    ax.set_ylim([0.44,0.65])
    ax.set_xlim([0.44,0.65])
    ax.set_xticks(np.linspace(0.45,0.65,5))
    ax.set_yticks(np.linspace(0.45,0.65,5))
    # ax.legend_.set_frame_on(False)
    # ax.legend_.set_title('Ori. pref. (mean)')
    ax.plot([0.45,0.65],[0.45,0.65],'--k')
    sns.despine(trim=True)
    ax.set_xlabel(r'Naïve $\sigma$')
    ax.set_ylabel(r'Proficient $\sigma$')

    # f.savefig('/mnt/c/Users/Samuel/OneDrive - University College London/Results/Figures/Draft/' +
    #           'dprime_sigma_naive vs proficient.svg', format = 'svg')

    # Heatmap
    # d_min = -3
    # d_max = 3

    # df_naive = pd.DataFrame.copy(df_d[df_d.trained==False])
    # df_trained = pd.DataFrame.copy(df_d[df_d.trained])

    # df_naive = df_naive.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
    # df_trained = df_trained.groupby(['subject','pref_ori','r']).mean().reset_index().reindex(df_naive.index)

    # f,ax = plt.subplots(1,2)

    # sns.heatmap(df_naive.d.to_numpy()[:,None], vmin = d_min, vmax = d_max, ax = ax[0], cmap = 'RdBu_r')
    # sns.heatmap(df_trained.d.to_numpy()[:,None], vmin = d_min, vmax = d_max, ax = ax[1], cmap = 'RdBu_r')



#%%
ori_colors = sns.hls_palette(8)[slice(2,5,2)]

xlabels = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
           '0.64 - 1']

dodge = 0.25

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

    f = plt.figure(figsize=(1.7,1.65))
    ax = f.add_subplot(111)

    x_jitter = 0.025*np.random.randn(*df_d.r[df_d.trained==False].shape) - dodge
    x_jitter += df_d.r[df_d.trained==False].to_numpy()

    sns.scatterplot(x = x_jitter,
                  y = df_d[df_d.trained==False].d,
                  hue = df_d[df_d.trained==False].pref_ori.astype(int).astype(str),
                  palette = ori_colors,
                  marker = 'o', s = 3, ax = ax,
                  linewidth = 0.5, edgecolor = 'black')


    x_jitter = 0.025*np.random.randn(*df_d.r[df_d.trained==True].shape) + dodge
    x_jitter += df_d.r[df_d.trained==True].to_numpy()

    sns.scatterplot(x = x_jitter,
                  y = df_d[df_d.trained==True].d,
                  hue = df_d[df_d.trained==True].pref_ori.astype(int).astype(str),
                  palette = ori_colors,
                  marker = '^', s = 3, ax = ax, linewidth = 0.5,
                  edgecolor = 'black')

    mu = df_d.groupby(['trained','r'],as_index = False).mean()
    sem = df_d.groupby(['trained','r'], as_index = False).sem()

    # Add sem bars

    mean_width = 0.4

    for r in mu.r.unique():
        x = r - dodge
        m = mu[np.logical_and(mu.trained==False, mu.r == r)].d
        s = sem[np.logical_and(sem.trained==False, sem.r == r)].d

        ax.plot([x-mean_width/2, x+mean_width/2], [m, m],
                      lw=0.5, color='k', linestyle='-', zorder = 2)
        ax.errorbar(x, m, s, ecolor = 'k',
                          capsize = 1,capthick=0.5, zorder = 2)

        x = r + dodge
        m = mu[np.logical_and(mu.trained==True, mu.r == r)].d
        s = sem[np.logical_and(sem.trained==True, sem.r == r)].d

        ax.plot([x-mean_width/2, x+mean_width/2], [m, m],
                      lw=0.5, color='k', linestyle='-', zorder = 2)
        ax.errorbar(x, m, s, ecolor = 'k',
                          capsize = 2, capthick=0.5, zorder = 2)

    ax.set_xlabel('Selectivity')
    ax.set_ylabel(r'$d^\prime$')
    ax.set_xticks(np.arange(0,5))
    ax.set_xticklabels(xlabels, rotation = 25)
    ax.set_ylim([0,5])
    ax.set_yticks(np.arange(0,6,1))
    ax.legend_.set_visible(False)
    sns.despine(trim=True)
    f.tight_layout()
    f.savefig(
        #r'C:\Users\Samuel\OneDrive - University College London\Results\Figures\Draft\selectivity_vs_dprime.svg',
        r'/home/samuel/Projects/Learning_Orthog/Figures/selectivity_vs_dprime.svg',
        format = 'svg')



#%% Plot d' as function of orientation preference and condition - 135 vs 0

df = pd.DataFrame(trials_dict)

# Remove blank
df = df[df.stim_ori != np.inf]

df['pref_bin'] = pd.cut(df.pref, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)).astype(int))
df['r_bin'] = pd.cut(df.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))


groups = ['subject', 'stim_ori', 'trained', 'cell']

df_var = df.groupby(groups, observed = True).var().reset_index()

df_mu = df.groupby(groups, observed=True).mean().reset_index()


mu = df_mu.loc[df_mu.stim_ori == 0, 'trial_resps'].to_numpy()[:,None]
mu = np.append(mu,df_mu.loc[df_mu.stim_ori == 135, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)

var = df_var.loc[df_var.stim_ori == 0, 'trial_resps'].to_numpy()[:,None]
var = np.append(var,df_var.loc[df_var.stim_ori== 135, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)

var = np.sqrt(var.sum(1)/2)

d_prime = np.diff(mu,axis=1)/var[:,None]
df_d = pd.DataFrame({'d' : d_prime.reshape(-1,),
                     'trained' : df_scm.trained.to_numpy(),
                     'pref_ori' : df_scm.pref_ori_all.to_numpy(),
                     'subject' : df_scm.subject.to_numpy()})

df_d_mu = df_d.groupby(['trained', 'subject', 'pref_ori']).mean().reset_index()

# sns.set()
# sns.set_style('ticks')
# with sns.plotting_context("poster", font_scale = 1):

fig = sns.relplot(data = df_d_mu, x = 'pref_ori', y = 'd', hue = 'trained',
                 kind = 'line', errorbar = 'se', palette = 'colorblind',
                 markers = ['o','o'], style = 'trained',
                 style_order = [True,False],
                 markeredgecolor=None,
                 err_style = 'band')
plt.xlabel('Preferred orientation (deg)')
plt.ylabel(r'd$^\prime$ : 0 vs 135')
fig._legend.set_title('')
fig._legend.texts[0].set_text('Naïve')
fig._legend.texts[1].set_text('Proficient')
plt.xticks(np.unique(df.pref_bin))
fig.fig.set_size_inches((8,6))
plt.ylim([-1.5,1.5])

#%% Plot d' as function of orientation preference and condition - 135 vs 0
# Make it similar to figures in Poort, et al

df = pd.DataFrame(trials_dict)

# Remove blank
df = df[df.stim_ori != np.inf]

df['pref_bin'] = pd.cut(df.pref, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)).astype(int))
df['r_bin'] = pd.cut(df.r, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5))


groups = ['subject', 'stim_ori', 'trained', 'cell']

df_var = df.groupby(groups, observed = True).var().reset_index()

df_mu = df.groupby(groups, observed=True).mean().reset_index()


mu = df_mu.loc[df_mu.stim_ori == 0, 'trial_resps'].to_numpy()[:,None]
mu = np.append(mu,df_mu.loc[df_mu.stim_ori == 135, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)

var = df_var.loc[df_var.stim_ori == 0, 'trial_resps'].to_numpy()[:,None]
var = np.append(var,df_var.loc[df_var.stim_ori== 135, 'trial_resps'].to_numpy()[:,None],
                                                                       axis=1)

var = np.sqrt(var.sum(1)/2)

d_prime = np.diff(mu,axis=1)/var[:,None]
df_d = pd.DataFrame({'d' : d_prime.reshape(-1,),
                     'trained' : df_scm.trained.to_numpy(),
                       'pref_ori' : df_scm.pref_bin_all.to_numpy(),
                      # 'pref_ori' : df_scm.pref_ori_all.to_numpy(),
                     'r' : df_scm.r_bin_all.to_numpy(),
                     'subject' : df_scm.subject.to_numpy()})


# task_responsive = ((df_d.pref_ori == 45) | (df_d.pref_ori == 90) |
#                     (df_d.pref_ori == 68) | (df_d.pref_ori == 23) |
#                     (df_d.pref_ori == 113))

# task_responsive = (df_d.pref_ori == 45) | (df_d.pref_ori == 90)

df_naive = pd.DataFrame.copy(df_d[df_d.trained == False])
df_trained = pd.DataFrame.copy(df_d[df_d.trained])

# df_naive = df_d[np.logical_and(df_d.trained == False, task_responsive)]
# df_trained = df_d[np.logical_and(df_d.trained, task_responsive)]

# df_naive = df_naive.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
# df_trained = df_trained.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
# df_trained = df_trained.reindex(df_naive.index)

df_naive = df_naive.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
df_trained = df_trained.groupby(['subject','pref_ori','r']).mean().reset_index().sort_values('d')
df_trained = df_trained.reindex(df_naive.index)

# df_naive = df_d[df_d.trained == False].sort_values('d')
# df_trained = df_d[df_d.trained].sort_values('d')

# df_naive = df_naive.sort_values('d')
# df_trained = df_trained.sort_values('d')

fig, ax = plt.subplots(1,2, sharex = True)

d_min = -3
d_max = 3

cm1 = ax[0].imshow(df_naive.d.to_numpy()[:,None], cmap = 'RdBu_r',
           vmin = d_min, vmax = d_max, aspect = 'auto')
# fig.colorbar(cm1,ax=ax[0])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_ylabel('Cell classes')
ax[0].set_title('Naïve')

cm2 = ax[1].imshow(df_trained.d.to_numpy()[:,None], cmap = 'RdBu_r',
           vmin = d_min, vmax = d_max, aspect = 'auto')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Proficient')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(cm2, cax=cbar_ax)
cb.set_label(r'd$^\prime$')

df_trained['trained'] = True
df_naive['trained'] = False

df_all = pd.DataFrame.copy(df_d)
# df_all = df_naive.append(df_trained)

# df_all['d'] = df_all['d']**2
# df_all = df_all.groupby(['subject','trained','pref_ori','r']).mean().reset_index()
# df_all['d'] = np.sqrt(df_all['d'])

df_all['d'] = df_all['d']**2
df_all = df_all.groupby(['subject','trained','pref_ori']).mean().reset_index()
df_all['d'] = np.sqrt(df_all['d'])

df_diff = df_all[df_all.trained]
df_diff['d'] = df_diff['d'].to_numpy() - df_all.loc[df_all.trained==False,'d'].to_numpy()

df_diff['pref_ori'] = df_diff['pref_ori'].astype(int)

sns.catplot(data = df_diff, x = 'pref_ori', y = 'd',
           kind = 'bar', palette = 'hls', aspect = 1.3)
plt.ylim([-0.3,0.9])
# sns.relplot(data = df_diff[df_diff.r >= 2], x = 'pref_ori', y = 'd', kind = 'line',
#             n_boot = 2000)
plt.xlabel('Preferred orientation (deg)')
plt.ylabel(r'$\Delta$Population d$^\prime$')
plt.tight_layout()

df_all = pd.DataFrame.copy(df_d)

d_th = 2

df_all['high_d'] = np.abs(df_all.d) > d_th

df_all = df_all.groupby(['subject','trained','pref_ori'])['high_d'].value_counts(normalize=True)\
    .rename('proportion_high_d').reset_index()

df_all = df_all[df_all.high_d==False]
df_all['proportion_high_d'] = 1 - df_all['proportion_high_d']
df_all['pref_ori'] = df_all['pref_ori'].astype(int)

ax_p = sns.catplot(data = df_all, x = 'pref_ori', y = 'proportion_high_d',
            hue = 'trained', kind = 'bar', palette = 'colorblind', ci = 68,
            aspect = 1.3, height = 5)
# plt.xticks([])
plt.xlabel('Preferred orientation (deg)')
plt.ylabel(r'Proportion of cells with $\left|d^\prime\right| >$' + str(d_th))
ax_p._legend.set_title('')
ax_p._legend.texts[0].set_text('Naive')
ax_p._legend.texts[1].set_text('Proficient')
plt.ylim([0,0.135])
plt.tight_layout()
ax_p._legend.set_bbox_to_anchor([1,0.7])



df_diff = df_all[df_all.trained]
df_diff['proportion_high_d'] = (df_diff['proportion_high_d'].to_numpy() -
                                df_all[df_all.trained == False].proportion_high_d.to_numpy())


ax_p = sns.catplot(data = df_diff, x = 'pref_ori', y = 'proportion_high_d',
            kind = 'bar', palette = 'hls', ci = 68, aspect = 1.3)
# plt.xticks([])
plt.xlabel('Preferred orientation (deg)')
plt.ylabel(r'$\Delta$Proportion of cells with $\left|d^\prime\right| >$' + str(d_th))
plt.ylim([-0.03,0.08])
plt.tight_layout()
#%% Plot heatmaps of each cells mean orientation responses, one plot for naive and one for trained


naive_mu = ori_dict['mean_ori_all'][:-1,df_scm['trained']==False]
trained_mu = ori_dict['mean_ori_all'][:-1,df_scm['trained']==True]

naive_ind = np.random.permutation(np.arange(naive_mu.shape[1]))
trained_ind = np.random.permutation(np.arange(trained_mu.shape[1]))

naive_mu = naive_mu[:,naive_ind]
trained_mu = trained_mu[:,trained_ind]

# naive_sort = np.lexsort((df_scm[df_scm.trained==False].r,df_scm[df_scm.trained==False].pref_ori_all))
# trained_sort = np.lexsort((df_scm[df_scm.trained].r,df_scm[df_scm.trained].pref_ori_all))
naive_sort = np.argsort(df_scm[df_scm.trained==False].pref_ori_all.to_numpy()[naive_ind])
trained_sort = np.argsort(df_scm[df_scm.trained].pref_ori_all.to_numpy()[trained_ind])

naive_mu = naive_mu[:,naive_sort]
trained_mu = trained_mu[:,trained_sort]

# naive_pref = df_scm[df_scm.trained == False].pref_ori_all.to_numpy()[naive_sort]
# trained_pref = df_scm[df_scm.trained].pref_ori_all.to_numpy()[trained_sort]

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1):

    fig, ax = plt.subplots(1,2, figsize=(20,9))

    sns.heatmap(naive_mu.T, ax = ax[0], cmap = 'Greys_r',
                yticklabels = False,
                xticklabels = ['0','23','45','68','90','113','135', '158'],
                rasterized = True)
    ax[0].set_title('Naïve', color = trained_cp[0])
    ax[0].tick_params(bottom = False)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 0)
    ax[0].set_xlabel('Orientation (deg)')
    ax[0].set_ylabel('Cells')
    plt.setp(ax[0].get_xticklabels()[2:5], fontweight = 'bold')


    sns.heatmap(trained_mu.T, ax = ax[1], cmap = 'Greys_r',
                yticklabels = False,
                xticklabels = ['0','23','45','68','90','113','135', '158'],
                rasterized = True)
    ax[1].set_title('Proficient', color = trained_cp[1])
    ax[1].tick_params(bottom = False)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 0)
    ax[1].set_xlabel('Orientation (deg)')
    ax[1].set_ylabel('Cells')
    plt.setp(ax[1].get_xticklabels()[2:5], fontweight = 'bold')


#%% Orientation tuning curves

r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) -1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins)-1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_expt = np.zeros((len(subjects),8,n_p,n_r))

hist_counts = []

for i,(s,t) in enumerate(zip(subjects,trained)):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    # ind = np.logical_and(ind,df_scm.ROI_task_ret<25)

    th = df_scm.loc[ind,'th']
    r = df_scm.loc[ind,'r']

    stim_resps = ori_dict['mean_ori_test'][:-1,ind]
    
    histo_count,x_edge,y_edge = np.histogram2d(th, r, bins=[pref_bins,r_bins])
    hist_counts.append(histo_count)

    for o in range(stim_resps.shape[0]):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r, weights=stim_resps[o,:],
                                                        bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])
        # np.seterr(all='ignore')
        mean_ori_expt[i,o,...] = histo_sum/histo_count


mean_ori_expt[mean_ori_expt == np.inf] = np.nan

#%% Plot

subject_label = np.tile(np.array(subjects).reshape(-1,1,1,1), (1,8,n_p,n_r))
trained_label = np.tile(np.array(trained).reshape(-1,1,1,1), (1,8,n_p,n_r))
r_label = np.tile(np.array(r_bin_centers).reshape(1,1,1,-1), (len(subjects),8,n_p,1))
pref_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,1,-1,1), (len(subjects),8,1,n_r)).astype(int).astype(str)
stim_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,-1,1,1), (len(subjects),1,n_p,n_r)).astype(int)


s_titles = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
            '0.64 - 1']

xlabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xlabels = [str(x) + r'$\degree$' for x in xlabels]

ori_cp = sns.hls_palette(8)

tc_dict = {'subject' : subject_label.flatten(),
           'trained' : trained_label.flatten(),
           'r_bin' : r_label.flatten(),
           'pref' : pref_label.flatten(),
           'stim' : stim_label.flatten(),
           'response' : mean_ori_expt.flatten()}

df_tc = pd.DataFrame(tc_dict)

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

    fig = sns.relplot(data = df_tc, x = 'stim', y = 'response', hue = 'pref',
                col = 'r_bin', row = 'trained',
                kind = 'line', errorbar = ('se',1), palette = ori_cp,
                height = 1.2, aspect = 0.8,
                legend = True)
    # fig._fig.set_size_inches(5,2.5)
    # fig.set_axis_labels(r'Stimulus ori. ($\degree$)', 'Response')
    # plt.xticks(np.arange(0,180,22.5))
    plt.xticks([0,45,67.5,90,157.5])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    # fig.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int), rotation = 45)
    fig.set_xticklabels([0,45,68,90,158])
    # fig.set_xticklabels(xlabels, rotation = 45)
    fig._legend.remove()
    # fig._legend.set_title('Ori. pref. (mean)')
    plt.ylim([0,1])
    # fig.tight_layout()
    fig.axes.flatten()[0].legend(bbox_to_anchor = (1.06,-0.12), loc = 'center',
                ncol = 2, title = r'Pref. mean ori.', frameon=False,
                handletextpad=0.2, labelspacing=0.2, borderpad = 0,
                handlelength = 1, labels = xlabels, columnspacing = 0.75)

    for i,a in enumerate(fig.axes.flatten()):
        if (i < 10) & (i != 0) & (i != 5):
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
            sns.despine(ax=a,bottom = True, left=True, trim = True)
        elif i == 0:
            a.xaxis.set_visible(False)
            sns.despine(ax=a,bottom=True, trim = True)
        elif i == 5:
            a.xaxis.set_visible(False)
            sns.despine(ax=a,bottom=True, trim = True)
        elif i > 10:
            a.yaxis.set_visible(False)
            sns.despine(ax=a,left=True, trim = True)
        else:
            sns.despine(ax = a, trim = True)

        if i >=10:
            a.set_xlabel(r'Stimulus ori. ($\degree$)', labelpad = 1)
        if (i == 0) | (i == 5) | (i==10):
            a.set_ylabel('Response', labelpad = 1)

        if i < 5:
            a.set_title(s_titles[i])
        else:
            a.set_title('')

        a.plot([45,45],[0,1],'-k')
        a.plot([67.5,67.5],[0,1],'--k')
        # a.plot([0,0],[0,1],'-k')
        a.plot([90,90],[0,1],'-k')
        a.set_facecolor((1,1,1,0))

    savefile = join(results_dir,'Figures','Draft','ori_tuning_curves.svg')
    # fig.savefig(savefile, format = 'svg')

#%% Plot - 2D

s_titles = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
            '0.64 - 1']

xylabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xylabels = [str(x) + r'$\degree$' for x in xylabels]

subject_label = np.tile(np.array(subjects).reshape(-1,1,1,1), (1,8,n_p,n_r))
trained_label = np.tile(np.array(trained).reshape(-1,1,1,1), (1,8,n_p,n_r))
r_label = np.tile(np.array(r_bin_centers).reshape(1,1,1,-1), (len(subjects),8,n_p,1))
pref_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,1,-1,1), (len(subjects),8,1,n_r)).astype(int)
stim_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,-1,1,1), (len(subjects),1,n_p,n_r)).astype(int)


s_titles = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
            '0.64 - 1']

xlabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xlabels = [str(x) + r'$\degree$' for x in xlabels]

ori_cp = sns.hls_palette(8)

tc_dict = {'subject' : subject_label.flatten(),
           'trained' : trained_label.flatten(),
           'r_bin' : r_label.flatten(),
           'pref' : pref_label.flatten(),
           'stim' : stim_label.flatten(),
           'response' : mean_ori_expt.flatten()}

df_tc = pd.DataFrame(tc_dict)


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

    f,ax = plt.subplots(3,5, figsize=(4.75,3.25))


    for i,r in enumerate(np.unique(df_tc.r_bin)):
        df_naive = df_tc[np.logical_and(df_tc.trained==False,df_tc.r_bin==r)].groupby(
            ['subject','pref','stim']).mean().reset_index()
        df_naive = df_naive.groupby(['pref','stim']).mean().reset_index()
        df_trained = df_tc[np.logical_and(df_tc.trained,df_tc.r_bin==r)].groupby(
            ['subject','pref','stim']).mean().reset_index()
        df_trained = df_trained.groupby(['pref','stim']).mean().reset_index()

        df_naive = df_naive.pivot(index = 'stim',columns='pref',values ='response').sort_values('pref',axis=1)
        df_trained = df_trained.pivot(index='stim',columns = 'pref', values = 'response').sort_values('pref',axis=1)

        sns.heatmap(df_naive,vmin = 0, vmax = 1,ax = ax[0,i], cbar = False, square = True)
        sns.heatmap(df_trained,vmin = 0, vmax = 1,ax = ax[1,i], cbar = False, square = True)
        sns.heatmap(df_trained-df_naive,ax = ax[2,i], cbar = False, cmap = 'RdBu_r', square = True)

        for ai,a in enumerate(ax[:,i]):
            if (i != 0) & (ai != 2):
                a.set_xticks([])
                a.set_yticks([])
                a.set_xlabel('')
                a.set_ylabel('')
            if i == 0:
                a.set_yticks(np.arange(8)+0.5)
                a.set_yticklabels(xylabels)
                a.set_ylabel('Stimulus orientation')
                a.set_xticks([])
                a.set_xlabel('')
            if ai == 2:
                a.set_xticks(np.arange(8)+0.5)
                a.set_xticklabels(xylabels, rotation = 45)
                a.set_xlabel('Preferred mean orientation')
            if (ai == 2) & (i > 0):
                a.set_yticks([])
                a.set_ylabel('')
            if ai == 0:
                a.set_title(s_titles[i])

    f.tight_layout()

    # savefile = join(results_dir,'Figures','Draft','ori_tuning_curves_2d.svg')
    # fig.savefig(savefile, format = 'svg')

#%% Plot - rows seperate figures (to fix distortion issues)

subject_label = np.tile(np.array(subjects).reshape(-1,1,1,1), (1,8,n_p,n_r))
trained_label = np.tile(np.array(trained).reshape(-1,1,1,1), (1,8,n_p,n_r))
r_label = np.tile(np.array(r_bin_centers).reshape(1,1,1,-1), (len(subjects),8,n_p,1))
pref_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,1,-1,1), (len(subjects),8,1,n_r)).astype(int).astype(str)
stim_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,-1,1,1), (len(subjects),1,n_p,n_r)).astype(int)

s_titles = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
            '0.64 - 1']

xlabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xlabels = [str(x) + r'$\degree$' for x in xlabels]

ori_cp = sns.hls_palette(8)

tc_dict = {'subject' : subject_label.flatten(),
           'trained' : trained_label.flatten(),
           'r_bin' : r_label.flatten(),
           'pref' : pref_label.flatten(),
           'stim' : stim_label.flatten(),
           'response' : mean_ori_expt.flatten()}

df_tc = pd.DataFrame(tc_dict)

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

    fig_n = sns.relplot(data = df_tc[df_tc.trained == False], x = 'stim', y = 'response', hue = 'pref',
                col = 'r_bin',
                kind = 'line', errorbar = 'se', palette = ori_cp,
                height = 1.2, aspect = 0.79,
                legend = True)
    plt.xticks([0,45,67.5,90,157.5])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    fig_n.set_xticklabels([0,45,68,90,158])
    fig_n._legend.remove()
    plt.ylim([0,1.2])
    fig_n.axes.flatten()[0].legend(bbox_to_anchor = (1.06,-0.12), loc = 'center',
                ncol = 2, title = r'Pref. mean ori.', frameon=False,
                handletextpad=0.2, labelspacing=0.2, borderpad = 0,
                handlelength = 1, labels = xlabels, columnspacing = 0.75)

    for i,a in enumerate(fig_n.axes.flatten()):
        if i != 0:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
            sns.despine(ax=a,bottom = True,left=True, trim = True)
        else:
            a.set_ylabel('Response', labelpad = 1)
            a.xaxis.set_visible(False)
            sns.despine(ax=a,bottom=True, trim = True)

        a.set_title(s_titles[i])

        a.plot([45,45],[0,1],'-k')
        a.plot([67.5,67.5],[0,1],'--k')
        # a.plot([0,0],[0,1],'-k')
        a.plot([90,90],[0,1],'-k')
        a.set_facecolor((1,1,1,0))

    savefile = join(results_dir,'Figures','Draft','ori_tuning_curves_naive.svg')
    fig_n.savefig(savefile, format = 'svg')

    fig_t = sns.relplot(data = df_tc[df_tc.trained == True], x = 'stim', y = 'response', hue = 'pref',
                col = 'r_bin',
                kind = 'line', errorbar = 'se', palette = ori_cp,
                height = 1.2, aspect = 0.79,
                legend = True)
    plt.xticks([0,45,67.5,90,157.5])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    fig_t.set_xticklabels([0,45,68,90,158])
    fig_t._legend.remove()
    plt.ylim([0,1.2])

    for i,a in enumerate(fig_t.axes.flatten()):

        if i == 0:
            a.set_ylabel('Response', labelpad = 1)
            sns.despine(ax=a, trim = True)
        else:
            a.yaxis.set_visible(False)
            sns.despine(ax=a, left = True, trim = True)

        a.set_title('')
        a.set_xlabel(r'Stimulus ori. ($\degree$)', labelpad = 1)

        # a.plot([45,45],[0,1],'-k')
        # a.plot([67.5,67.5],[0,1],'--k')
        a.plot([0,0],[0,1],'-k')
        a.plot([90,90],[0,1],'-k')
        a.set_facecolor((1,1,1,0))

    # savefile = join(results_dir,'Figures','Draft','ori_tuning_curves_proficient.svg')
    fig_t.savefig(savefile, format = 'svg')



#%% Model change in life-time sparseness of each cell class

# Fit piecewise linear for each cell class (ori pref and selectivity level) where datapoints are stimulus responses between naive and proficient

def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y



stim_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,:,None,None], (10,1,8,5))
pref_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,None,:,None], (10,8,1,5))
r_label = np.tile(np.arange(5)[None,None,None,:], (10,8,8,1))
cond_label = np.tile(np.array(trained)[:,None,None,None], (1,8,8,5))

df = pd.DataFrame({'Responses': mean_ori_expt.ravel(),
                   'Stimulus': stim_label.ravel(),
                   'Ori. pref.': pref_label.ravel(),
                   'Selectivity': r_label.ravel(),
                   'Condition': cond_label.ravel()})


df = df.groupby(['Condition','Stimulus','Ori. pref.', 'Selectivity'])['Responses'].mean().reset_index()

df['Stimulus'] = df['Stimulus'].astype(int).astype(str)

f,a = plt.subplots(5,8, sharex=True, sharey=True)

ia = 0

for r in df['Selectivity'].unique():
    for p in df['Ori. pref.'].unique():
        df_plot = df[(df['Ori. pref.']==p) & (df['Selectivity']==r)]
        df_plot = pd.pivot(df_plot, index='Stimulus', columns=['Condition'], values='Responses')
        
        fit_params, _ = so.curve_fit(fitfun, df_plot['Naive'].to_numpy(), df_plot['Proficient'].to_numpy(),
                                 max_nfev=1500,
                                 p0=[.5, .5, 1], bounds=((0,0,0),(1,3,3)))
        
        if ia==39:
            legend=True
        else:
            legend=False
        
        sns.scatterplot(df_plot, x='Naive', y='Proficient', hue='Stimulus', ax=a.flat[ia],
                        legend=legend, palette='hls', hue_order=['0','23','45','68','90','113','135','158'])
        
        xr = np.linspace(0,1,100)
        a.flat[ia].plot(xr, fitfun(xr, *fit_params), color = 'k', linewidth = 0.5)
        a.flat[ia].set_box_aspect(1)
        a.flat[ia].set_xlim([0,1])
        a.flat[ia].set_ylim([0,1])
        
        if ia <= 7:
            a.flat[ia].set_title(int(p))
            
        sns.despine(ax=a.flat[ia])
        
        if ia==39:
            handles, labels = a.flat[ia].get_legend_handles_labels()
            a.flat[ia].get_legend().remove()
        
        ia += 1

f.legend(handles, labels, loc='center right', fancybox=False, frameon=False)

#%% Find fit for change in lifetime sparseness for each subject

def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y


stim_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,:,None,None], (10,1,8,5))
pref_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,None,:,None], (10,8,1,5))
r_label = np.tile(np.arange(5)[None,None,None,:], (10,8,8,1))
cond_label = np.tile(np.array(trained)[:,None,None,None], (1,8,8,5))
subject_label = np.tile(np.array(subjects)[:,None,None,None], (1,8,8,5))

df = pd.DataFrame({'Responses': mean_ori_expt.ravel(),
                   'Stimulus': stim_label.ravel(),
                   'Ori. pref.': pref_label.ravel(),
                   'Selectivity': r_label.ravel(),
                   'Condition': cond_label.ravel(),
                   'Subject': subject_label.ravel()})


fit_params_all_LS = []

for s in df['Subject'].unique():
    for r in df['Selectivity'].unique():
        for p in df['Ori. pref.'].unique():
            df_fit = df[(df['Ori. pref.']==p) & (df['Selectivity']==r) & (df['Subject']==s)]
            df_fit = pd.pivot(df_fit, index='Stimulus', columns=['Condition'], values='Responses')
            
            fit_params, _ = so.curve_fit(fitfun, df_fit['Naive'].to_numpy(), df_fit['Proficient'].to_numpy(),
                                    max_nfev=1500,
                                    p0=[.5, .5, 1], bounds=((0,0,0),(1,3,3)))

            fit_params_all_LS.append(fit_params)
            
#%% Find fit for change in lifetime sparseness, pool all subject data

def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y


stim_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,:,None,None], (10,1,8,5))
pref_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,None,:,None], (10,8,1,5))
r_label = np.tile(np.arange(5)[None,None,None,:], (10,8,8,1))
cond_label = np.tile(np.array(trained)[:,None,None,None], (1,8,8,5))
subject_label = np.tile(np.array(subjects)[:,None,None,None], (1,8,8,5))

df = pd.DataFrame({'Responses': mean_ori_expt.ravel(),
                   'Stimulus': stim_label.ravel(),
                   'Ori. pref.': pref_label.ravel(),
                   'Selectivity': r_label.ravel(),
                   'Condition': cond_label.ravel(),
                   'Subject': subject_label.ravel()})

df = df.groupby(['Stimulus','Ori. pref.','Selectivity','Condition'])['Responses'].mean().reset_index()


fit_params_all = []

for r in df['Selectivity'].unique():
    for p in df['Ori. pref.'].unique():
        df_fit = df[(df['Ori. pref.']==p) & (df['Selectivity']==r)]
        df_fit = pd.pivot(df_fit, index='Stimulus', columns=['Condition'], values='Responses')
        
        fit_params, _ = so.curve_fit(fitfun, df_fit['Naive'].to_numpy(), df_fit['Proficient'].to_numpy(),
                                max_nfev=1500,
                                p0=[.5, .5, 1], bounds=((0,0,0),(1,3,3)))

        fit_params_all.append(fit_params)




#%% Transform naive tuning curves - subject specific

sparsened_resps_LS = ori_dict['mean_ori_all'][:-1,:].copy()

i = 0

for s in df['Subject'].unique():
    for r in df['Selectivity'].unique():
        for p in df['Ori. pref.'].unique():

            ind = (df_scm.subject == s) & (df_scm.trained == 'Naive') & (df_scm.pref_bin==p) & (df_scm.r_bin==r)

            stim_resps = ori_dict['mean_ori_all'][:-1,ind]
    
            sparsened_resps_LS[:,ind] = fitfun(stim_resps, *fit_params_all_LS[i])
            
            i += 1
            
sparsened_resps_LS = sparsened_resps_LS[:,df_scm.trained=='Naive']
            
#%% Transform naive tuning curves - subject pooled

sparsened_resps_LS = ori_dict['mean_ori_all'][:-1,:].copy()

i = 0

for r in df['Selectivity'].unique():
    for p in df['Ori. pref.'].unique():

            ind = (df_scm.trained == 'Naive') & (df_scm.pref_bin==p) & (df_scm.r_bin==r)

            stim_resps = ori_dict['mean_ori_all'][:-1,ind]
    
            sparsened_resps_LS[:,ind] = fitfun(stim_resps, *fit_params_all_LS[i])
            
            i += 1
                       
sparsened_resps_LS = sparsened_resps_LS[:,df_scm.trained=='Naive']

#%% Lifetime sparsened tuning curves

r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) - 1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins) - 1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_sparse = np.zeros((int(len(subjects)/2),8,n_p,n_r))

for i,s in enumerate(np.arange(0,len(subjects),2)):
    ind = np.logical_and(df_scm.subject == subjects[s], df_scm.trained == 'Naive')

    th = df_scm.th[ind]
    r = df_scm.r[ind]

    for o in range(sparsened_resps_LS[:,ind].shape[0]):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r,
                                                 weights=sparsened_resps_LS[o,ind],
                                                 bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])

        mean_ori_sparse[i,o,...] = histo_sum/histo_count


#%% Plot lifetime sprasened tuning curves

subject_label = np.tile(np.array(subjects[0::2]).reshape(-1,1,1,1), (1,8,n_p,n_r))
r_label = np.tile(np.array(r_bin_centers).reshape(1,1,1,-1), (5,8,n_p,1))
pref_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,1,-1,1), (5,8,1,n_r)).astype(int).astype(str)
stim_label = np.tile(np.arange(0,180,22.5).reshape(1,-1,1,1), (5,1,n_p,n_r))

s_titles = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
            '0.64 - 1']

tc_dict = {'subject' : subject_label.flatten(),
           'r_bin' : r_label.flatten(),
           'pref' : pref_label.flatten(),
           'stim' : stim_label.flatten(),
           'response' : mean_ori_sparse.flatten()}

df_tc_s = pd.DataFrame(tc_dict)

ori_cp = sns.hls_palette(8)

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

    fig = sns.relplot(data = df_tc_s, x = 'stim', y = 'response', hue = 'pref',
                col = 'r_bin',
                kind = 'line', errorbar = 'se', palette = ori_cp,
                height = 1.2, aspect = 0.79,
                legend = True)
    # fig._fig.set_size_inches(5,1.25)
    # fig.set_axis_labels('Stim. ori. (deg)', 'Response')
    # plt.xticks(np.arange(0,180,22.5))
    plt.xticks([0,45,67.5,90,157.5])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    # fig.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    fig.set_xticklabels([0,45,68,90,158])
    fig._legend.remove()
    # fig._legend.set_title('Ori. pref. (mean)')
    plt.ylim([0,1])
    # fig.tight_layout()
    fig.axes.flatten()[0].legend(bbox_to_anchor = (0.9,0.95), loc = 'center',
                ncol = 2, title = r'Pref. mean ori. ($\degree$)', frameon=False,
                handletextpad=0.2, labelspacing=0.2, borderpad = 0,
                columnspacing = 0.75, handlelength = 1)

    for i,a in enumerate(fig.axes.flatten()):
        if i == 0:
            sns.despine(ax=a,trim = True)
        elif i > 0:
            a.yaxis.set_visible(False)
            sns.despine(ax=a,left=True, trim = True)

        a.set_title('')

        a.plot([45,45],[0,1],'-k')
        a.plot([67.5,67.5],[0,1],'--k')
        a.plot([90,90],[0,1],'-k')
        a.set_facecolor((1,1,1,0))
        a.set_xlabel(r'Stimulus ori. ($\degree$)', labelpad=1)
        a.set_ylabel('Response', labelpad = 1)

    # savefile = join(results_dir,'Figures','Draft','ori_tuning_curves_sparsened.svg')
    # fig.savefig(savefile, format = 'svg')



#%% Look at sparsening within a single condition using leave one out approach
isub = 'SF180613'
icond = 'Naive'
ind = np.logical_and(trials_dict['subject']==isub, trials_dict['trained']==icond)

r = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].r.to_numpy()
r_bin = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].r_bin.to_numpy()
th = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].th.to_numpy()
pref_bin = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].pref_bin.to_numpy()

trials = trials_dict['trial_resps'][ind]
cells = trials_dict['cell'][ind]
uni_cells = np.unique(cells)

trials = trials.reshape(-1,len(uni_cells))

stim = trials_dict['stim_ori'][ind].reshape(-1,len(uni_cells))[:,0]

train_ind = trials_dict['train_ind'][ind].reshape(-1,len(uni_cells))[:,0]

trials = trials[np.logical_not(train_ind),:]
stim = stim[np.logical_not(train_ind)]

ind_nb = stim != np.inf

trials = trials[ind_nb,:]
stim = stim[ind_nb]

trial_nums = np.arange(len(trials))

ncells = 600
prop_cells = ncells/trials.shape[1]

cell_classes = np.concatenate((pref_bin[:,None], r_bin[:,None]),axis=1)
uni_classes = np.unique(cell_classes,axis=0)
cell_code = np.zeros(len(th))
for i_c,u in enumerate(uni_classes):
    cell_code[np.all(cell_classes == u,axis=1)] = i_c


for c in np.unique(cell_code):
    cell_ind = np.concatenate([np.random.choice(np.where(cell_code==c)[0],
                        np.round(prop_cells*(cell_code==c).sum()).astype(int),
                        replace = False)
                        for c in np.unique(cell_code)])


trials = trials[:,cell_ind]
r = r[cell_ind]
th = th[cell_ind]

r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) -1
pref_bins = np.linspace(-11.25,180-11.25,9)

mean_trial =  np.zeros((len(trial_nums),8,n_r))
mean_train = np.zeros((len(trial_nums),8,n_r))

for i in trial_nums:

    train_ind = np.delete(trial_nums,i)

    stim_train = stim[train_ind]
    trials_train = trials[train_ind,:]

    stim_ind = stim_train == stim[i]

    # Mean from training set
    histo_sum,x_edge,y_edge = np.histogram2d(th, r, weights=trials_train[stim_ind,:].mean(0),
                                                        bins=[pref_bins,r_bins])
    histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])
    mean_train[i,:] = histo_sum/histo_count

    # Mean for individual trial
    histo_sum,x_edge,y_edge = np.histogram2d(th, r, weights=trials[i,:],
                                                        bins=[pref_bins,r_bins])
    histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])
    mean_trial[i,:] = histo_sum/histo_count


#%% Plot random subset with fits

num_rows = 5

def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y


trial_nums = np.arange(len(mean_trial))

itrials = np.random.choice(trial_nums,num_rows**2,replace=False)

f,a = plt.subplots(num_rows, num_rows, sharex = True, sharey = True, figsize = (10,10))

r_labels = np.repeat(np.arange(5).reshape(1,-1),8,axis=0).astype(str)
pref_labels = np.repeat(np.arange(8).reshape(-1,1),5,axis=1).astype(str)

for i,t in enumerate(itrials):

    fit_params, _ = so.curve_fit(fitfun, mean_train[t,:].ravel(), mean_trial[t,:].ravel(),
                                 max_nfev=1500,
                                 p0=[.5, .5, 1], bounds=((0,0,0),(1,3,3)))

    sns.scatterplot(x = mean_train[t,:].ravel(), y = mean_trial[t,:].ravel(), hue = pref_labels.ravel(),
                    style = r_labels.ravel(), ax = a.flatten()[i], legend = False, s = 10,
                    edgecolor = 'k')

    xr = np.linspace(0,1,100)
    a.flatten()[i].plot(xr, fitfun(xr, *fit_params), color = 'k', linewidth = 0.5)
    a.flatten()[i].set_box_aspect(1)
    a.flatten()[i].set_xlim([0,2])
    a.flatten()[i].set_ylim([0,2])

    rmse = np.round(np.std(fitfun(mean_train[t,:].ravel(), *fit_params) - mean_trial[t,:].ravel()),3)
    a.flatten()[i].set_title(rmse)


f.tight_layout()

#%% Look at sparsening within a single condition using leave one out approach
# Look at fit quality over different numbers of cells

# nonlinear function to sparsen
def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y

isub = 'SF180613'
icond = 'Naive'
ind = np.logical_and(trials_dict['subject']==isub, trials_dict['trained']==icond)

r = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].r.to_numpy()
r_bin = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].r_bin.to_numpy()
th = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].th.to_numpy()
pref_bin = df_scm[np.logical_and(df_scm.subject == isub,df_scm.trained == icond)].pref_bin.to_numpy()

trials = trials_dict['trial_resps'][ind]
cells = trials_dict['cell'][ind]
uni_cells = np.unique(cells)

trials = trials.reshape(-1,len(uni_cells))

stim = trials_dict['stim_ori'][ind].reshape(-1,len(uni_cells))[:,0]

train_ind = trials_dict['train_ind'][ind].reshape(-1,len(uni_cells))[:,0]

trials = trials[np.logical_not(train_ind),:]
stim = stim[np.logical_not(train_ind)]

ind_nb = stim != np.inf

trials = trials[ind_nb,:]
stim = stim[ind_nb]

trial_nums = np.arange(len(trials))

ncells = [200, 400, 600, 1000, 2000, 3000]

nrepeats = 10

rmse = np.zeros((len(ncells),nrepeats))

cell_classes = np.concatenate((pref_bin[:,None], r_bin[:,None]),axis=1)
uni_classes = np.unique(cell_classes,axis=0)
cell_code = np.zeros(len(th))
for i_c,u in enumerate(uni_classes):
    cell_code[np.all(cell_classes == u,axis=1)] = i_c

for ni,n in enumerate(ncells):
    for rep in range(nrepeats):

        prop_cells = n/trials.shape[1]

        for c in np.unique(cell_code):
            cell_ind = np.concatenate([np.random.choice(np.where(cell_code==c)[0],
                                np.round(prop_cells*(cell_code==c).sum()).astype(int),
                                replace = False)
                                for c in np.unique(cell_code)])

        trials_rep = trials[:,cell_ind]
        r_rep = r[cell_ind]
        th_rep = th[cell_ind]

        r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
        # r_bins = [0., 0.25, 0.5, 0.75, 1.]
        # r_bins = np.append(np.linspace(0,0.7,6),1)
        n_r = len(r_bins) -1
        pref_bins = np.linspace(-11.25,180-11.25,9)

        mean_trial =  np.zeros((len(trial_nums),8,n_r))
        mean_train = np.zeros((len(trial_nums),8,n_r))

        pred = np.zeros((len(trial_nums),40))

        for i in trial_nums:

            train_ind = np.delete(trial_nums,i)

            stim_train = stim[train_ind]
            trials_train = trials_rep[train_ind,:]

            stim_ind = stim_train == stim[i]

            # Mean from training set
            histo_sum,x_edge,y_edge = np.histogram2d(th_rep, r_rep, weights=trials_train[stim_ind,:].mean(0),
                                                                bins=[pref_bins,r_bins])
            histo_count,x_edge,y_edge = np.histogram2d(th_rep, r_rep,
                                                            bins=[pref_bins,r_bins])
            mean_train[i,:] = histo_sum/histo_count

            # Mean for individual trial
            histo_sum,x_edge,y_edge = np.histogram2d(th_rep, r_rep, weights=trials_rep[i,:],
                                                                bins=[pref_bins,r_bins])
            histo_count,x_edge,y_edge = np.histogram2d(th_rep, r_rep,
                                                            bins=[pref_bins,r_bins])
            mean_trial[i,:] = histo_sum/histo_count

            fit_params, _ = so.curve_fit(fitfun, mean_train[i,:].ravel(), mean_trial[i,:].ravel(),
                                    max_nfev=1500,
                                    p0=[.5, .5, 1], bounds=((0,0,0),(1,4,4)))
            pred[i,:] = fitfun(mean_train[i,:].ravel(), *fit_params)

        rmse[ni,rep] = np.std(pred.ravel() - mean_trial.ravel())

#%% Fit piecewise linear functions to model sparsening

# nonlinear function to sparsen
def fitfun(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y


mu_ori_naive = mean_ori_expt[0::2,...].mean(0)
mu_ori_trained = mean_ori_expt[1::2,...].mean(0)

# Fit piecewise linear
fit_params = np.zeros(len(np.unique(df_scm.pref_bin)), dtype = 'object')


for i in range(len(np.unique(df_scm.pref_bin))):
    fit_params[i], _ = so.curve_fit(fitfun,mu_ori_naive[i,...].ravel(),
                                           mu_ori_trained[i,...].ravel(),
                                  p0=[.5,.5], bounds=((0,0),(1,1)))


#%% Create legend for icons, polar plot format

ori_cp = sns.hls_palette(n_p)

r_markers = ['X','P','s','v','*']


th=np.arange(0,1.0001,.0001)*2*np.pi

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
                                        "lines.markersize":2,
                                        'font.family' : 'sans-serif',
                                        'font.sans-serif' : 'Helvetica'}):


    f,ax = plt.subplots(1,1, figsize=(1,1))

    for rd in [0.16, 0.32, 0.48, 0.64]:
            ax.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)


    x,y = np.cos(np.linspace(0,2*np.pi,9)),np.sin(np.linspace(0,2*np.pi,9))
    x = x[:-1]
    y = y[:-1]

    x = np.repeat(x[:,None],5,axis=1)*np.array([0.08, 0.24, 0.40, 0.56, 0.72])
    y = np.repeat(y[:,None],5,axis=1)*np.array([0.08, 0.24, 0.40, 0.56, 0.72])

    for o in range(8):
        for r in range(5):
            ax.scatter(x[o,r],y[o,r], marker = r_markers[r], color = ori_cp[o],
                       linewidths = 0.5,
                        edgecolors = 'k',
                        s = 10,
                        zorder = 2)


    sns.despine(ax=ax, left=True, bottom = True)
    ax.set_xticks([])
    ax.set_yticks([])

    fontsize = 5

    # ang = np.deg2rad(292.5)
    # dis = 0.9
    # orth_ang = np.arctan(-np.sin(ang)/np.cos(ang))
    # ax.text(np.cos(ang)*dis, np.sin(ang)*dis, 'Selectivity',rotation = 90-np.rad2deg(orth_ang),
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # ax.plot([0,np.cos(ang)*dis],[0,np.sin(ang)*dis], color = 'k')

    # for r in [0.16, 0.32, 0.48, 0.64]:
    #     ax.text(np.cos(ang)*(r+0.01), np.sin(ang)*(r+0.01), r, rotation = 45,
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # ax.annotate('Selectivity',
    #             xy = (0,0),
    #             xycoords='data',
    #             xytext= (np.cos(ang)*1.5, np.sin(ang)*1.5),
    #             textcoords='data',
    #             arrowprops=dict(arrowstyle= '<-',
    #                          color='black',
    #                          lw=0.5,
    #                          ls='-'),
    #             rotation = 45,
    #             rotation_mode = 'anchor',
    #             ha = 'center',
    #             va = 'top',
    #             fontsize = fontsize
    #             )

    h_dist = 0.85
    v_dist = 0.85

    for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

        # if i == 0:
        #     label = 'Mean ori. pref.\n' + str(int(np.rad2deg(a)/2)) + r'$\degree$'
        # else:
        #     label = str(int(np.rad2deg(a)/2)) + r'$\degree$'

        label = str(int(np.rad2deg(a)/2)) + r'$\degree$'
        if i == 0:
            ha = 'left'
            va = 'center'
            d = h_dist
        elif i == 1:
            ha = 'center'
            va = 'bottom'
            d = v_dist
        elif i == 2:
            ha = 'right'
            va = 'center'
            d = h_dist
        elif i == 3:
            ha = 'center'
            va = 'top'
            d = v_dist
        # va = 'center'
        # ha = 'center'

        ax.text(np.cos(a)*d, np.sin(a)*d, label,
                horizontalalignment=ha, verticalalignment = va,
                fontsize = fontsize)

#%% Create legend for icons, grid format !!!TO DO!!!!

ori_cp = sns.hls_palette(n_p)

r_markers = ['X','P','s','v','*']


th=np.arange(0,1.0001,.0001)*2*np.pi

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
                                        "lines.markersize":2,
                                        'font.family' : 'sans-serif',
                                        'font.sans-serif' : 'Helvetica'}):


    f,ax = plt.subplots(1,1, figsize=(1,1))

    for rd in [0.16, 0.32, 0.48, 0.64]:
            ax.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)


    x,y = np.cos(np.linspace(0,2*np.pi,9)),np.sin(np.linspace(0,2*np.pi,9))
    x = x[:-1]
    y = y[:-1]

    x = np.repeat(x[:,None],5,axis=1)*np.array([0.08, 0.24, 0.40, 0.56, 0.72])
    y = np.repeat(y[:,None],5,axis=1)*np.array([0.08, 0.24, 0.40, 0.56, 0.72])

    for o in range(8):
        for r in range(5):
            ax.scatter(x[o,r],y[o,r], marker = r_markers[r], color = ori_cp[o],
                       linewidths = 0.5,
                        edgecolors = 'k',
                        s = 10,
                        zorder = 2)


    sns.despine(ax=ax, left=True, bottom = True)
    ax.set_xticks([])
    ax.set_yticks([])

    fontsize = 5

    # ang = np.deg2rad(292.5)
    # dis = 0.9
    # orth_ang = np.arctan(-np.sin(ang)/np.cos(ang))
    # ax.text(np.cos(ang)*dis, np.sin(ang)*dis, 'Selectivity',rotation = 90-np.rad2deg(orth_ang),
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # ax.plot([0,np.cos(ang)*dis],[0,np.sin(ang)*dis], color = 'k')

    # for r in [0.16, 0.32, 0.48, 0.64]:
    #     ax.text(np.cos(ang)*(r+0.01), np.sin(ang)*(r+0.01), r, rotation = 45,
    #               horizontalalignment='center', verticalalignment = 'top',
    #               fontsize = fontsize, rotation_mode = 'anchor')

    # ax.annotate('Selectivity',
    #             xy = (0,0),
    #             xycoords='data',
    #             xytext= (np.cos(ang)*1.5, np.sin(ang)*1.5),
    #             textcoords='data',
    #             arrowprops=dict(arrowstyle= '<-',
    #                          color='black',
    #                          lw=0.5,
    #                          ls='-'),
    #             rotation = 45,
    #             rotation_mode = 'anchor',
    #             ha = 'center',
    #             va = 'top',
    #             fontsize = fontsize
    #             )

    h_dist = 0.85
    v_dist = 0.85

    for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

        # if i == 0:
        #     label = 'Mean ori. pref.\n' + str(int(np.rad2deg(a)/2)) + r'$\degree$'
        # else:
        #     label = str(int(np.rad2deg(a)/2)) + r'$\degree$'

        label = str(int(np.rad2deg(a)/2)) + r'$\degree$'
        if i == 0:
            ha = 'left'
            va = 'center'
            d = h_dist
        elif i == 1:
            ha = 'center'
            va = 'bottom'
            d = v_dist
        elif i == 2:
            ha = 'right'
            va = 'center'
            d = h_dist
        elif i == 3:
            ha = 'center'
            va = 'top'
            d = v_dist
        # va = 'center'
        # ha = 'center'

        ax.text(np.cos(a)*d, np.sin(a)*d, label,
                horizontalalignment=ha, verticalalignment = va,
                fontsize = fontsize)

#%% Plot without fit

colors_cond = sns.color_palette('colorblind')[0:3]

ori_cp = sns.hls_palette(n_p)

r_markers = ['x','+','s','v','*']

mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',np.array(ori_cp))

fig, ax = plt.subplots(2,4, figsize = (14,8))

for i,o in enumerate(np.unique(df_scm.pref_bin)):

        for r in range(n_r):
            for p in range(n_p):
                ax.flatten()[i].plot(mu_ori_naive[i,p,r], mu_ori_trained[i,p,r], marker = r_markers[r],
                                     color = ori_cp[p])

        ax.flatten()[i].plot([0,1],[0,1], '--', color=[.5,.5,.5])
        ax.flatten()[i].set_xlabel('Naïve', color = colors_cond[0])
        if (i == 0) or (i == 4):
            ax.flatten()[i].set_ylabel('Proficient', color = colors_cond[1])
        ax.flatten()[i].set_title(r'Mean pref. %d$\degree$'%o)
        x0,x1 = ax.flatten()[i].get_xlim()
        y0,y1 = ax.flatten()[i].get_ylim()
        ax.flatten()[i].set_aspect((x1-x0)/(y1-y0))
        sns.despine()

#%% Plot only 45, 68, and 90

colors_cond = sns.color_palette('colorblind')[0:3]

ori_cp = sns.hls_palette(n_p)

r_markers = ['x','+','s','v','*']

mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',np.array(ori_cp))

fig, ax = plt.subplots(1,3, figsize = (14,8))

uni_pref = np.unique(df_scm.pref_bin)

for s,i in enumerate(np.arange(2,5)):

        for r in range(n_r):
            for p in range(n_p):
                ax.flatten()[s].plot(mu_ori_naive[i,p,r], mu_ori_trained[i,p,r], marker = r_markers[r],
                                     color = ori_cp[p])

        ax.flatten()[s].plot([0,1],[0,1], '--', color=[.5,.5,.5])
        ax.flatten()[s].set_xlabel('Naïve', color = colors_cond[0])
        if s==0:
            ax.flatten()[s].set_ylabel('Proficient', color = colors_cond[1])
        ax.flatten()[s].set_title(r'Mean pref. %d$\degree$'%uni_pref[i])
        x0,x1 = ax.flatten()[s].get_xlim()
        y0,y1 = ax.flatten()[s].get_ylim()
        ax.flatten()[s].set_aspect((x1-x0)/(y1-y0))
        sns.despine()
        plt.tight_layout()

#%% Plot with fit

colors_cond = sns.color_palette('colorblind')[0:3]

ori_cp = sns.hls_palette(n_p)

r_markers = ['X','P','s','v','*']

mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',np.array(ori_cp))

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
                                        "lines.markersize":2}):


    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'

    fig, ax = plt.subplots(2,4, figsize = (4.8,2.75))

    for i,o in enumerate(np.unique(df_scm.pref_bin)):

            for r in range(n_r):
                for s in range(8):
                    ax.flatten()[i].scatter(mu_ori_naive[i,s,r],
                                            mu_ori_trained[i,s,r],
                                            marker = r_markers[r],
                                            color = ori_cp[s],
                                            linewidths = 0.5,
                                            edgecolors = 'k',
                                            s = 10)

            xr = np.linspace(0,1,100)
            ax.flatten()[i].plot(xr, fitfun(xr, *fit_params[i]), color='k')
            ax.flatten()[i].plot([0,1],[0,1], '--', color='k')
            if i > 3:
                ax.flatten()[i].set_xlabel('Naïve response', color = colors_cond[0])
            if (i == 0) or (i == 4):
                ax.flatten()[i].set_ylabel('Proficient response', color = colors_cond[1])
            ax.flatten()[i].set_title(r'Stim. ori. %d$\degree$'%o)
            x0,x1 = ax.flatten()[i].get_xlim()
            y0,y1 = ax.flatten()[i].get_ylim()
            ax.flatten()[i].set_aspect((x1-x0)/(y1-y0))
            ax.flatten()[i].set_xticks([0,0.5,1])
            ax.flatten()[i].set_yticks([0,0.5,1])
            if i == 0:
                    sns.despine(ax=ax.flatten()[i],trim = True, bottom = True)
                    ax.flatten()[i].set_xticks([])
            if (i > 0) & (i < 4):
                     sns.despine(ax=ax.flatten()[i],trim = True, bottom = True, left = True)
                     ax.flatten()[i].set_xticks([])
                     ax.flatten()[i].set_yticks([])
            if i == 4:
                     sns.despine(ax=ax.flatten()[i],trim = True)
            if i > 4:
                    sns.despine(ax=ax.flatten()[i],trim=True, left = True)
                    ax.flatten()[i].set_yticks([])
    fig.tight_layout()

#%% Plot only 45, 68, and 90 with fit

colors_cond = sns.color_palette('colorblind')[0:3]

ori_cp = sns.hls_palette(n_p)

r_markers = ['X','P','s','v','*']

mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',np.array(ori_cp))

uni_pref = np.unique(df_scm.pref_bin)

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
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":2}):


    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(1,3, figsize = (2.8,1.8))

    for s,i in enumerate(np.arange(2,5)):

            for r in range(n_r):
                for p in range(n_p):
                    ax.flatten()[s].scatter(mu_ori_naive[i,p,r],
                                         mu_ori_trained[i,p,r],
                                         marker = r_markers[r],
                                         color = ori_cp[p],
                                         linewidth = 0.5,
                                         zorder = 2,
                                         edgecolor = 'k',
                                         s = 10)

            ax.flatten()[s].plot([0,1],[0,1], '--', color='k', zorder = 3)
            ax.flatten()[s].set_xlabel('Naïve response', color = colors_cond[0],
                                       labelpad = 1)
            ax.flatten()[s].set_xticks([0,0.5,1])
            ax.flatten()[s].set_yticks([0,0.5,1])
            if s==0:
                ax.flatten()[s].set_ylabel('Proficient response', color = colors_cond[1],
                                           labelpad = 1)
            ax.flatten()[s].set_title(r'Stim. ori. %d$\degree$'%uni_pref[i])
            x0,x1 = ax.flatten()[s].get_xlim()
            y0,y1 = ax.flatten()[s].get_ylim()
            ax.flatten()[s].set_aspect((x1-x0)/(y1-y0))
            if s == 0:
                sns.despine(ax=ax.flatten()[s],trim=True)
            else:
                sns.despine(ax=ax.flatten()[s],left=True,trim=True)
                ax.flatten()[s].yaxis.set_visible(False)

            plt.tight_layout()

            xr = np.linspace(0,1,100)
            ax.flatten()[s].plot(xr, fitfun(xr, *fit_params[i]), color='k', zorder = 1)

    savefile = join(save_fig_dir,'naive_to_proficient_task_stim_with_fit.svg')
    fig.savefig(savefile, format = 'svg')
#%% Fit piecewise linear functions to each subject

# nonlinear function to sparsen
def fitfun(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y

ori_naive = mean_ori_expt[0::2,...]
ori_trained = mean_ori_expt[1::2,...]

# Fit piecewise linear
fit_params = np.zeros((len(np.unique(df_scm.pref_bin)),5), dtype = 'object')

for s in range(5):
    for i in range(len(np.unique(df_scm.pref_bin))):
        fit_params[i,s], _ = so.curve_fit(fitfun,ori_naive[s,i,...].ravel(),
                                           ori_trained[s,i,...].ravel(),
                                      p0=[.5,.5], bounds=((0,0),(1,1)))

#%% Plot with fit

colors_cond = sns.color_palette('colorblind')[0:3]

ori_cp = sns.hls_palette(n_p)

r_markers = ['X','P','s','v','*']

mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',np.array(ori_cp))

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
                                        "lines.markersize":2}):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'

    for sub in range(5):
        fig, ax = plt.subplots(2,4, figsize = (4.8,2.75))

        for i,o in enumerate(np.unique(df_scm.pref_bin)):

                for r in range(n_r):
                    for s in range(8):
                        ax.flatten()[i].scatter(ori_naive[sub,i,s,r],
                                                ori_trained[sub,i,s,r],
                                                marker = r_markers[r],
                                                color = ori_cp[s],
                                                linewidths = 0.5,
                                                edgecolors = 'k',
                                                s = 10)

                xr = np.linspace(0,1,100)
                ax.flatten()[i].plot(xr, fitfun(xr, *fit_params[i,sub]), color='k')
                ax.flatten()[i].plot([0,1],[0,1], '--', color='k')
                if i > 3:
                    ax.flatten()[i].set_xlabel('Naïve response', color = colors_cond[0])
                if (i == 0) or (i == 4):
                    ax.flatten()[i].set_ylabel('Proficient response', color = colors_cond[1])
                ax.flatten()[i].set_title(r'Stim. ori. %d$\degree$'%o)
                x0,x1 = ax.flatten()[i].get_xlim()
                y0,y1 = ax.flatten()[i].get_ylim()
                ax.flatten()[i].set_aspect((x1-x0)/(y1-y0))
                ax.flatten()[i].set_yticks([0,0.5,1])
                ax.flatten()[i].set_xticks([0,0.5,1])
                if i == 0:
                    sns.despine(ax=ax.flatten()[i],trim = True, bottom = True)
                    ax.flatten()[i].set_xticks([])
                if (i > 0) & (i < 4):
                     sns.despine(ax=ax.flatten()[i],trim = True, bottom = True, left = True)
                     ax.flatten()[i].set_xticks([])
                     ax.flatten()[i].set_yticks([])
                if i == 4:
                     sns.despine(ax=ax.flatten()[i],trim = True)
                if i > 4:
                    sns.despine(ax=ax.flatten()[i],trim=True, left = True)
                    ax.flatten()[i].set_yticks([])
        fig.tight_layout()
#%% Sparsen naive data

sparsened_resps = np.zeros(int(len(subjects)/2),dtype='object')

for i,s in enumerate(np.arange(0,len(subjects),2)):
    # print(str(subjects[s]) + ' ' + str(trained[s]))
    ind = np.logical_and(df_scm.subject == subjects[s], df_scm.trained == 'Naive')

    stim_resps = ori_dict['mean_ori_all'][:-1,ind]

    sparsened_resps[i] = np.zeros_like(stim_resps)

    for o in range(len(np.unique(df_scm.pref_bin))):
        sparsened_resps[i][o,:] = fitfun(stim_resps[o,:], *fit_params[o])

#%% Sparsen using subject specific fits

sparsened_resps = np.zeros(int(len(subjects)/2),dtype='object')

for i,s in enumerate(np.arange(0,len(subjects),2)):
    # print(str(subjects[s]) + ' ' + str(trained[s]))
    ind = np.logical_and(df_scm.subject == subjects[s], df_scm.trained == 'Naive')

    stim_resps = ori_dict['mean_ori_all'][:-1,ind]

    sparsened_resps[i] = np.zeros_like(stim_resps)

    for o in range(len(np.unique(df_scm.pref_bin))):
        sparsened_resps[i][o,:] = fitfun(stim_resps[o,:], *fit_params[o,i])

#%% Sparsened tuning curves - PS

r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) - 1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins) - 1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_sparse = np.zeros((int(len(subjects)/2),8,n_p,n_r))

for i,s in enumerate(np.arange(0,len(subjects),2)):
    ind = np.logical_and(df_scm.subject == subjects[s], df_scm.trained == 'Naive')

    th = df_scm.th[ind]
    r = df_scm.r[ind]

    for o in range(sparsened_resps[i].shape[0]):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r,
                                                 weights=sparsened_resps[i][o,:],
                                                 bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])

        mean_ori_sparse[i,o,...] = histo_sum/histo_count


#%% Sparsened tuning curves - LS

r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) - 1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins) - 1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_sparse_LS = np.zeros((int(len(subjects)/2),8,n_p,n_r))

for i,s in enumerate(np.arange(0,len(subjects),2)):
    sub_naive = df_scm.loc[df_scm.trained=='Naive', 'subject']
    ind = sub_naive == subjects[s]
    ind_metrics = np.logical_and(df_scm.subject == subjects[s], df_scm.trained == 'Naive')

    th = df_scm.th[ind_metrics]
    r = df_scm.r[ind_metrics]

    for o in range(sparsened_resps_LS.shape[0]):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r,
                                                 weights=sparsened_resps_LS[o,ind],
                                                 bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])

        mean_ori_sparse_LS[i,o,...] = histo_sum/histo_count


#%% Compare similarity of each sparsening type with actual trained orientation tuning curves

ori_trained = mean_ori_expt[1::2,...]

resps = np.concatenate([mean_ori_sparse, mean_ori_sparse_LS, ori_trained], axis=0)
stim = np.tile(np.ceil(np.arange(0,180,22.5))[None,:,None,None], (15,1,8,5))
r = np.tile(np.arange(5)[None,None,None,:], (15,8,8,1))
pref = np.tile(np.ceil(np.arange(0,180,22.5))[None,None,:,None], (15,8,1,5))
subject_label = np.tile(np.unique(subjects)[:,None,None,None], (3,8,8,5))
condition_label = np.tile(np.repeat(['PS sparsened','LS sparsened','Proficient'],5)[:,None,None,None], (1,8,8,5))

df = pd.DataFrame({'Orientation': stim.ravel(),
                   'Response': resps.ravel(),
                   'Selectivity': r.ravel(),
                   'Ori. pref.': pref.ravel(),
                   'subject': subject_label.ravel(),
                   'Condition': condition_label.ravel()})


df['Ori. pref.'] = df['Ori. pref.'].astype(int).astype(str)

df.loc[df['Response']>1, 'Response'] = 1

g = sns.relplot(df, x='Orientation', y='Response', hue='Ori. pref.', col='Selectivity', row='Condition',
            palette='hls', kind='line', errorbar=('se',1), row_order=['Proficient','PS sparsened','LS sparsened'])

g.set(xticks=np.arange(0,180,22.5), xticklabels=np.ceil(np.arange(0,180,22.5)).astype(int))


pivot_df = df.pivot_table(index=['subject', 'Ori. pref.', 'Selectivity', 'Orientation'], columns='Condition', values='Response').reset_index()

pivot_df['PS_minus_Proficient'] = pivot_df['PS sparsened'] - pivot_df['Proficient']
pivot_df['LS_minus_Proficient'] = pivot_df['LS sparsened'] - pivot_df['Proficient']


def calculate_rmse(series):
    return np.sqrt(np.mean(np.square(series)))

def calculate_mae(series):
    return np.mean(np.abs(series))

rmse_df = pivot_df.groupby(['subject', 'Ori. pref.', 'Selectivity']).agg({
    'PS_minus_Proficient': calculate_rmse,
    'LS_minus_Proficient': calculate_rmse
}).reset_index()

rmse_df.rename(columns={
    'PS_minus_Proficient': 'RMSE_PS',
    'LS_minus_Proficient': 'RMSE_LS'
}, inplace=True)

mae_df = pivot_df.groupby(['subject', 'Ori. pref.', 'Selectivity']).agg({
    'PS_minus_Proficient': calculate_mae,
    'LS_minus_Proficient': calculate_mae
}).reset_index()

mae_df.rename(columns={
    'PS_minus_Proficient': 'MAE_PS',
    'LS_minus_Proficient': 'MAE_LS'
}, inplace=True)


f,a = plt.subplots(1,1)

sns.scatterplot(rmse_df, x='RMSE_PS', y='RMSE_LS', hue='Ori. pref.', style='Selectivity', palette='hls',
            hue_order=['0','23','45','68','90','113','135','158'],
            ax=a)

a.plot([0, 0.3], [0, 0.3], '--k')

a.set_box_aspect(1)

# f,a = plt.subplots(1,1)

# sns.scatterplot(mae_df, x='MAE_PS', y='MAE_LS', hue='Ori. pref.', style='Selectivity', palette='hls',
#             hue_order=['0','23','45','68','90','113','135','158'],
#             ax=a)

# a.plot([0, 0.25], [0, 0.25], '--k')

# a.set_box_aspect(1)

sns.despine(ax=a)



rmse_df_long = pd.melt(rmse_df, id_vars=['Ori. pref.', 'Selectivity', 'subject'], value_vars = ['RMSE_PS','RMSE_LS'])
rmse_df_long['Class'] = rmse_df_long['Ori. pref.'].astype(str) + '-' + rmse_df_long['Selectivity'].astype(str) + '-' + rmse_df_long['subject']

f,a = plt.subplots(1,1)

sns.catplot(rmse_df_long, hue='Condition', y='value', x='Ori. pref.', col='Selectivity', kind='bar', errorbar=('se',2))


(
    sob.Plot(rmse_df_long, x='Condition', y='value', color='subject')
    # .facet(row='Selectivity', col='Ori. pref.')
    .facet(col='Selectivity')
    # .add(sob.Dots(alpha=0.2), color='subject', legend=False)
    # .add(sob.Lines(alpha=0.2), group='Class', color='subject', legend=False)
    .add(sob.Dot(), sob.Agg(), legend=False)
    .add(sob.Lines(), sob.Agg(), legend=False)
    .show()
)



long_df = pd.melt(pivot_df, id_vars=['subject','Ori. pref.','Selectivity','Orientation'], value_vars=['PS_minus_Proficient','LS_minus_Proficient'])

long_df['value'] = long_df['value'].abs()

g = sns.relplot(long_df, x='Orientation', y='value', row='Condition', col='Selectivity', hue='Ori. pref.', palette='hls', 
            hue_order=['0','23','45','68','90','113','135','158'], kind='line', errorbar=('se',1))

g.set(xticks=np.arange(0,180,22.5), xticklabels=np.ceil(np.arange(0,180,22.5)).astype(int))
g.set(ylabel='Abs. error')


norms = df.groupby(['subject','Selectivity','Ori. pref.','Condition'], group_keys=True)['Response'].transform('mean')

df_norm = df.copy()
df_norm['Response_norm'] = df_norm['Response'] - norms

pivot_norm = df_norm.pivot_table(index=['subject', 'Ori. pref.', 'Selectivity', 'Orientation'], columns='Condition', values='Response').reset_index()

pivot_norm['PS_minus_Proficient'] = pivot_norm['PS sparsened'] - pivot_norm['Proficient']
pivot_norm['LS_minus_Proficient'] = pivot_norm['LS sparsened'] - pivot_norm['Proficient']

rmse_norm = pivot_norm.groupby(['subject', 'Ori. pref.', 'Selectivity']).agg({
    'PS_minus_Proficient': calculate_rmse,
    'LS_minus_Proficient': calculate_rmse
}).reset_index()


rmse_norm.rename(columns={
    'PS_minus_Proficient': 'RMSE_PS',
    'LS_minus_Proficient': 'RMSE_LS'
}, inplace=True)

f,a = plt.subplots(1,1)

sns.scatterplot(rmse_norm, x='RMSE_PS', y='RMSE_LS', hue='Ori. pref.', style='Selectivity', palette='hls',
            hue_order=['0','23','45','68','90','113','135','158'],
            ax=a)

a.plot([0, 0.3], [0, 0.3], '--k')

a.set_box_aspect(1)



rmse_norm_long = pd.melt(rmse_norm, id_vars=['Ori. pref.', 'Selectivity', 'subject'], value_vars = ['RMSE_PS','RMSE_LS'])
rmse_norm_long['Class'] = rmse_norm_long['Ori. pref.'].astype(str) + '-' + rmse_norm_long['Selectivity'].astype(str) + '-' + rmse_norm_long['subject']

f,a = plt.subplots(1,1)


(
    sob.Plot(rmse_norm_long, x='Condition', y='value', color='subject')
    # .facet(row='Selectivity', col='Ori. pref.')
    .facet(col='Selectivity')
    # .add(sob.Dots(alpha=0.2), color='subject', legend=False)
    # .add(sob.Lines(alpha=0.2), group='Class', color='subject', legend=False)
    .add(sob.Dot(), sob.Agg(), legend=False)
    .add(sob.Lines(), sob.Agg(), legend=False)
    .show()
)


# Correlation

pivot_df = df.pivot_table(index=['subject', 'Ori. pref.', 'Selectivity', 'Orientation'], columns='Condition', values='Response').reset_index()

# Define a function to calculate correlation between two series
def calculate_correlation(group):
    return group['Proficient'].corr(group[col])

col = 'PS sparsened'

# Group by the three category columns and apply the correlation function
df_corr = pivot_df.groupby(['subject','Ori. pref.','Selectivity']).apply(calculate_correlation).reset_index().rename({0:'PS r'},axis=1)

col = 'LS sparsened'

df_corr['LS r'] = pivot_df.groupby(['subject','Ori. pref.','Selectivity']).apply(calculate_correlation).reset_index()[0]


# f,a = plt.subplots(1,1)

# sns.scatterplot(df_corr, x='PS r', y='LS r', hue='Ori. pref.', style='Selectivity', palette='hls',
#             hue_order=['0','23','45','68','90','113','135','158'],
#             ax=a)

f = plt.figure()

(
    sob.Plot(df_corr, x='PS r', y='LS r', color='Ori. pref.')
    .facet(col='Selectivity')
    .add(sob.Dots(), legend=False)
    .on(f)
    .plot()
)


for a in f.figure.axes:
    a.plot([-0.5,1], [-0.5,1], '--k')
    a.set_box_aspect(1)

f.show()


#%% Calculate population sparseness and cosine similarity on sparsened data

def pop_sparseness(fr, kind = 'Treves-Rolls'):

    # Find cells that have 0 responses and remove

    # breakpoint()
    if np.logical_or(kind.lower() == 'treves-rolls',  kind.lower() == 'tr'):
        # Treves-Rolls
        # breakpoint()
        top = (fr/fr.shape[1]).sum(1)**2
        bottom = (fr**2/fr.shape[1]).sum(1)
        s = 1 - (top/bottom)
    elif kind.lower() == 'kurtosis':
        s = kurtosis(fr,axis = 1)
    elif kind.lower() == 'active':
        sigma = fr.std(1)
        s = (fr < sigma[:,None]).sum(1)/fr.shape[1]
    return s

kind = 'tr'

df_s = pd.DataFrame()

for i in range(5):
    ind_n = np.logical_and(df_scm.subject==np.unique(subjects)[i], df_scm.trained=='Naive')
    ind_p = np.logical_and(df_scm.subject==np.unique(subjects)[i], df_scm.trained=='Proficient')

    n_resps = ori_dict['mean_ori_test'][:-1,ind_n]
    p_resps = ori_dict['mean_ori_test'][:-1,ind_p]

    s_resps = sparsened_resps[i]

    # Calculate population sparseness for each subject, average it for 45 and 90

    ps_n = np.array([pop_sparseness(n_resps[o,:].reshape(1,-1),kind) for o in [2,4]])
    ps_p = np.array([pop_sparseness(p_resps[o,:].reshape(1,-1),kind) for o in [2,4]])
    ps_s = np.array([pop_sparseness(s_resps[o,:].reshape(1,-1),kind) for o in [2,4]])

    if i == 0:
        df_s['ps'] = [ps_n.mean(),ps_p.mean(),ps_s.mean()]
        df_s['cs'] = [cosine_similarity(n_resps[2,:].reshape(1,-1),n_resps[4,:].reshape(1,-1))[0][0],
                      cosine_similarity(p_resps[2,:].reshape(1,-1),p_resps[4,:].reshape(1,-1))[0][0],
                      cosine_similarity(s_resps[2,:].reshape(1,-1),s_resps[4,:].reshape(1,-1))[0][0]]
        df_s['condition'] = ['Naive','Proficient','Sparsened']
        df_s['subject'] = np.repeat(np.unique(subjects)[i],3)

    else:
        df_s = pd.concat([df_s,pd.DataFrame({'ps' : [ps_n.mean(),ps_p.mean(),ps_s.mean()],
                                             'cs' : [cosine_similarity(n_resps[2,:].reshape(1,-1),n_resps[4,:].reshape(1,-1))[0][0],
                                                     cosine_similarity(p_resps[2,:].reshape(1,-1),p_resps[4,:].reshape(1,-1))[0][0],
                                                     cosine_similarity(s_resps[2,:].reshape(1,-1),s_resps[4,:].reshape(1,-1))[0][0]],
                                             'condition' : ['Naive','Proficient','Sparsened'],
                                             'subject' : np.repeat(np.unique(subjects)[i],3)})],
                         ignore_index = True, axis = 0)


f,a = plt.subplots(1,2)
sns.stripplot(df_s, x = 'condition', y = 'ps', ax = a[0])
sns.stripplot(df_s, x = 'condition', y = 'cs', ax = a[1])

#%% Plot

subject_label = np.tile(np.array(subjects[0::2]).reshape(-1,1,1,1), (1,8,n_p,n_r))
r_label = np.tile(np.array(r_bin_centers).reshape(1,1,1,-1), (5,8,n_p,1))
pref_label = np.tile(np.unique(df_scm.pref_bin).reshape(1,1,-1,1), (5,8,1,n_r)).astype(int).astype(str)
stim_label = np.tile(np.arange(0,180,22.5).reshape(1,-1,1,1), (5,1,n_p,n_r))

s_titles = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
            '0.64 - 1']

tc_dict = {'subject' : subject_label.flatten(),
           'r_bin' : r_label.flatten(),
           'pref' : pref_label.flatten(),
           'stim' : stim_label.flatten(),
           'response' : mean_ori_sparse.flatten()}

df_tc_s = pd.DataFrame(tc_dict)

ori_cp = sns.hls_palette(8)

sns.set_theme()
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

    fig = sns.relplot(data = df_tc_s, x = 'stim', y = 'response', hue = 'pref',
                col = 'r_bin',
                kind = 'line', errorbar = 'se', palette = ori_cp,
                height = 1.2, aspect = 0.79,
                legend = True)
    # fig._fig.set_size_inches(5,1.25)
    # fig.set_axis_labels('Stim. ori. (deg)', 'Response')
    # plt.xticks(np.arange(0,180,22.5))
    plt.xticks([0,45,67.5,90,157.5])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    # fig.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    fig.set_xticklabels([0,45,68,90,158])
    fig._legend.remove()
    # fig._legend.set_title('Ori. pref. (mean)')
    plt.ylim([0,1])
    # fig.tight_layout()
    fig.axes.flatten()[0].legend(bbox_to_anchor = (0.9,0.95), loc = 'center',
                ncol = 2, title = r'Pref. mean ori. ($\degree$)', frameon=False,
                handletextpad=0.2, labelspacing=0.2, borderpad = 0,
                columnspacing = 0.75, handlelength = 1)

    for i,a in enumerate(fig.axes.flatten()):
        if i == 0:
            sns.despine(ax=a,trim = True)
        elif i > 0:
            a.yaxis.set_visible(False)
            sns.despine(ax=a,left=True, trim = True)

        a.set_title('')

        a.plot([45,45],[0,1],'-k')
        a.plot([67.5,67.5],[0,1],'--k')
        a.plot([90,90],[0,1],'-k')
        a.set_facecolor((1,1,1,0))
        a.set_xlabel(r'Stimulus ori. ($\degree$)', labelpad=1)
        a.set_ylabel('Response', labelpad = 1)

    savefile = join(results_dir,'Figures','Draft','ori_tuning_curves_sparsened.svg')
    fig.savefig(savefile, format = 'svg')


#%% Calculate ratio and difference of slopes for convexity

# For each mouse, find slope of highly tuned point relative to origin, and slope
# of points for non-preferring cells

mu_ori_naive = mean_ori_expt[0::2,...]
mu_ori_trained = mean_ori_expt[1::2,...]

ratio_slope = np.zeros((int(len(subjects)/2), 8))
diff_slope = np.copy(ratio_slope)

for s in range(int(len(subjects)/2)):

    for o in range(8):
        x_pref = mu_ori_naive[s,o,o,-1]
        y_pref = mu_ori_trained[s,o,o,-1]

        pref_ind = np.arange(n_p) != o
        # r_ind = np.arange(n_r) <= 2

        x_others = mu_ori_naive[s, o, pref_ind,:]
        # x_others = x_others[...,r_ind]
        y_others = mu_ori_trained[s, o, pref_ind,:]
        # y_others = y_others[...,r_ind]

        pref_slope = y_pref/x_pref
        other_slope, _, _, _ = np.linalg.lstsq(x_others.reshape(-1,1),
                                               y_others.reshape(-1,1),
                                               rcond = None)

        ratio_slope[s,o] = pref_slope/other_slope[0][0]
        diff_slope[s,o] = pref_slope - other_slope[0][0]

#%% Calculate convexity measure as ratio of p and q


ratio_slope = np.zeros((int(len(subjects)/2), 8))

for s in range(5):
    for o in range(8):

        ratio_slope[s,o] = fit_params[o,s][0]/fit_params[o,s][1]



#%% Calculate convexity measure as ratio of slope of p,q and highest point

mu_ori_naive = mean_ori_expt[0::2,...]
mu_ori_trained = mean_ori_expt[1::2,...]

ratio_slope = np.zeros((int(len(subjects)/2), 8))

for s in range(5):
    for o in range(8):

        x = mu_ori_naive[s,o,:,:].flatten()
        y = mu_ori_trained[s,o,:,:].flatten()

        dists = np.sqrt(x**2 + y**2)

        m_ind = np.argmax(dists)

        m_slope = y[m_ind]/x[m_ind]
        f_slope = fit_params[o,s][1]/fit_params[o,s][0]

        ratio_slope[s,o] = m_slope/f_slope


#%%

from scipy import stats

a = ratio_slope
n1, n2 = a.shape
# Columns mean and squared std
m = np.mean(a,axis=0)
s2 = np.square(np.std(a,axis=0, ddof=1))

# Compute the test statistic
t = (m[:,np.newaxis] - m[np.newaxis,:])/np.sqrt((s2[:,np.newaxis] + s2[np.newaxis,:])/n1)

# Compute the 2-sided p-value
df = 2*n1 - 2
p = 2 - 2*stats.t.cdf(t,df=df)


#%% Plot convexity by stim

import matplotlib.ticker as ticker
from scipy import stats
from statsmodels.stats.multitest import multipletests

tick_spacing = 0.2

xlabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xlabels = [str(x) + r'$\degree$' for x in xlabels]

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


stim_label = np.repeat(np.ceil(np.arange(0,180,22.5)).astype(int)[None,:],5,axis=0).astype(str)

x = stim_label.flatten()
# Subtract 1 so that negative values are non-convex
y = ratio_slope.flatten() - 1
# y = diff_slope.flatten()


pval = []
pair = []

# for o0 in task_stim:
#     for o1 in non_task:
#         pval.append(stats.ttest_ind(y[x==o0],y[x==o1])[1])
#         pair.append([o0,o1])

# pval = multipletests(pval,method = 'hs')[1]

# pval_task = []

# pval_task.append(stats.ttest_ind(y[x=='45'], y[x=='68'])[1])
# pval_task.append(stats.ttest_ind(y[x=='45'], y[x=='90'])[1])
# pval_task.append(stats.ttest_ind(y[x=='68'], y[x=='90'])[1])

# pval_task = multipletests(pval_task,method='hs')[1]


ori_pal = []

for i in range(8):
    print(i)
    if i >= 2 and i <= 4:
        ori_pal.append(sns.color_palette('colorblind')[i])
    else:
        ori_pal.append(sns.color_palette('colorblind')[7])



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
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":0.5}):

    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    plt.figure(figsize = (2.5,1.6))
    ax = sns.swarmplot(x = x, y = y, s = 3, edgecolor='k', linewidth = 0.5,
                       zorder = 1, hue = x, palette = ori_pal)


    ax = sig_label(ax, (2,3), 0.06, 0.02, '**')
    ax = sig_label(ax,(3,4), 0.06, 0.02, '**')

    ax.legend_.set_visible(False)

    # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y")

        # calculate the median value for all replicates of either X or Y
        mean_val = y[x==sample_name].mean()
        std_val = y[x==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x)/len(np.unique(x))
        ci_val = mean_confidence_interval(y[x==sample_name], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        ax.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        ax.errorbar(tick, mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)

    # ax.set_xticks(np.arange(0,180,22.5))
    ax.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Stimulus orientation', labelpad = 1)
    ax.set_ylabel('Convexity', labelpad = 1)
    ax.set_ylim([-0.2,0.7])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    sns.despine(trim = True)
    plt.tight_layout()

    savefile = join(results_dir,'Figures','Draft','population_convexity.svg')
    # plt.savefig(savefile, format = 'svg')


#%% Plot convexity by stim type (motor-associated, distractor, non-task)

import matplotlib.ticker as ticker
from scipy import stats
from statsmodels.stats.multitest import multipletests

tick_spacing = 0.2

xlabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xlabels = [str(x) + r'$\degree$' for x in xlabels]

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


stim_label = np.repeat(np.ceil(np.arange(0,180,22.5)).astype(int)[None,:],5,axis=0).astype(str)

x = stim_label
# Subtract 1 so that negative values are non-convex
y = ratio_slope - 1

task_ind = (x[0,:] == '45') | (x[0,:] == '90')
dist_ind = (x[0,:] == '68')
non_task = (x[0,:] != '45') & (x[0,:] != '68') & (x[0,:] != '90')

y_by_type = np.zeros((5,3))

for it,t in enumerate([task_ind, dist_ind,non_task]):
    for s in range(len(x)):
        y_by_type[s,it] = y[s,t].mean()

y = y_by_type.flatten()

x = np.repeat(np.array(['Motor-associated','Distractor','Non-task'])[None,:],5,axis=0).flatten()

pval = []
pair = []

for o0 in range(len(np.unique(x))):
    for o1 in range(o0,len(np.unique(x))):
        if np.unique(x)[o0] == np.unique(x)[o1]:
            continue
        pval.append(stats.ttest_rel(y[x==np.unique(x)[o0]],y[x==np.unique(x)[o1]])[1])
        pair.append([np.unique(x)[o0],np.unique(x)[o1]])



sns.set_theme()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":5,"axes.titlesize":5,
                                        "axes.labelsize":5,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":0.5}):

    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    plt.figure(figsize = (1.3,1.2))
    ax = sns.swarmplot(x = x, y = y, s = 2, edgecolor='k', linewidth = 0.5, facecolor = 'k',
                       zorder = 1)


    # ax = sig_label(ax, (2,3), 0.06, 0.02, '**')
    # ax = sig_label(ax,(3,4), 0.06, 0.02, '**')


    # ax.legend_.set_visible(False)

    # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y")

        # calculate the median value for all replicates of either X or Y
        mean_val = y[x==sample_name].mean()
        std_val = y[x==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x)/len(np.unique(x))
        ci_val = mean_confidence_interval(y[x==sample_name], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        ax.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        ax.errorbar(tick, mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)

    # ax.set_xticks(np.arange(0,180,22.5))
    # ax.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    # ax.set_xticklabels(xlabels)
    ax.set_xticklabels([r'45$\degree$ and 90$\degree$', r'68$\degree$', 'non-task'])
    ax.set_xlabel('Stimulus type', labelpad = 2)
    ax.set_ylabel('Convexity', labelpad = 2)
    ax.set_ylim([-0.2,0.7])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    sns.despine(trim = True)
    plt.tight_layout()

    savefile = join(fig_save_dir,r'population_convexity_by_stim type.svg')
    plt.savefig(savefile, format = 'svg')



#%% Plot convexity by stim and relation to population sparseness

import matplotlib.ticker as ticker
from scipy import stats
from statsmodels.stats.multitest import multipletests

tick_spacing = 0.2

xlabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xlabels = [str(x) + r'$\degree$' for x in xlabels]

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


stim_label = np.repeat(np.arange(0,180,22.5)[None,:],5,axis=0).astype(str)

x = stim_label.flatten()
# Subtract 1 so that negative values are non-convex
y = ratio_slope.flatten() - 1
# y = diff_slope.flatten()

task_stim = list(map(str,[45.0, 90.0]))
non_task = list(map(str,[0.0,22.5,67.5,112.5,135.0,157.5]))

pval = []
pair = []

for o0 in task_stim:
    for o1 in non_task:
        pval.append(stats.ttest_ind(y[x==o0],y[x==o1])[1])
        pair.append([o0,o1])

pval = multipletests(pval,method = 'hs')[1]

pval_task = []

pval_task.append(stats.ttest_ind(y[x=='45.0'], y[x=='67.5'])[1])
pval_task.append(stats.ttest_ind(y[x=='45.0'], y[x=='90.0'])[1])
pval_task.append(stats.ttest_ind(y[x=='67.5'], y[x=='90.0'])[1])

# pval_task = multipletests(pval_task,method='hs')[1]


ori_pal = []

for i in range(8):
    print(i)
    if i >= 2 and i <= 4:
        ori_pal.append(sns.color_palette('colorblind')[i])
    else:
        ori_pal.append(sns.color_palette('colorblind')[7])



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
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":0.5}):

    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    plt.figure(figsize = (2.5,1.5))
    ax = sns.swarmplot(x = x, y = y, s = 3, edgecolor='k', linewidth = 0.5,
                       zorder = 1, hue = x, palette = ori_pal)


    ax = sig_label(ax, (2,3), 0.06, 0.02, '**')
    ax = sig_label(ax,(3,4), 0.06, 0.02, '**')

    ax.legend_.set_visible(False)

    # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y")

        # calculate the median value for all replicates of either X or Y
        mean_val = y[x==sample_name].mean()
        std_val = y[x==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x)/len(np.unique(x))
        ci_val = mean_confidence_interval(y[x==sample_name], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        ax.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        ax.errorbar(tick, mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)

    # ax.set_xticks(np.arange(0,180,22.5))
    ax.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Stimulus orientation', labelpad = 1)
    ax.set_ylabel('Convexity', labelpad = 1)
    ax.set_ylim([-0.2,0.7])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    sns.despine(trim = True)
    plt.tight_layout()

    savefile = join(results_dir,'Figures','Draft','population_convexity.svg')
    # plt.savefig(savefile, format = 'svg')


    slope, intercept, r_value, p_value, std_err = stats.linregress(ps_diff.flatten(),y)
    x_mdl = np.linspace(ps_diff.flatten().min()-0.2,ps_diff.flatten().max()+0.2,100)

    plt.figure(figsize=(1.5,1.5))
    ax = sns.scatterplot(x=ps_diff.flatten(),y=y, edgecolor = 'k', linewidth = 0.5,
                         s = 6, hue = x, legend = False, palette = ori_pal)
    ax.plot(x_mdl,x_mdl*slope + intercept, '--k')
    ax.set_xlabel(r'$\Delta$population sparseness', labelpad = 1)
    ax.set_ylabel('Convexity', labelpad = 1)
    ax.text(0.1,1.55-1,'r = ' + str(np.round(r_value,3)) + '\n p < 0.0001',
            multialignment='center',
            horizontalalignment = 'center')
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_ylim([-0.2,0.7])
    ax.set_xlim([-1,4])
    plt.tight_layout()
    ax.set_xticks(np.arange(-0.5,4.5,1))
    sns.despine(trim = True)

    savefile = join(results_dir,'Figures','Draft','population_convexity_vs_kurtosis.svg')
    # plt.savefig(savefile, format = 'svg')

#%% Plot convexity by stim and relation to population sparseness - group nontask stim

import matplotlib.ticker as ticker
from scipy import stats
from statsmodels.stats.multitest import multipletests

tick_spacing = 0.2

xlabels = np.ceil(np.arange(0,180,22.5)).astype(int)
xlabels = [str(x) + r'$\degree$' for x in xlabels]

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


stim_label = np.repeat(np.arange(0,180,22.5)[None,:],5,axis=0).astype(str)
task_stim_ind = (stim_label == 45) | (stim_label==68) | (stim_label==90)

x = stim_label.flatten()
# Subtract 1 so that negative values are non-convex
y = ratio_slope.flatten() - 1
# y = diff_slope.flatten()

task_stim = list(map(str,[45.0, 90.0]))
non_task = list(map(str,[0.0,22.5,67.5,112.5,135.0,157.5]))

pval = []
pair = []

for o0 in task_stim:
    for o1 in non_task:
        pval.append(stats.ttest_ind(y[x==o0],y[x==o1])[1])
        pair.append([o0,o1])

pval = multipletests(pval,method = 'hs')[1]

pval_task = []

pval_task.append(stats.ttest_ind(y[x=='45.0'], y[x=='67.5'])[1])
pval_task.append(stats.ttest_ind(y[x=='45.0'], y[x=='90.0'])[1])
pval_task.append(stats.ttest_ind(y[x=='67.5'], y[x=='90.0'])[1])

# pval_task = multipletests(pval_task,method='hs')[1]


ori_pal = []

for i in range(8):
    print(i)
    if i >= 2 and i <= 4:
        ori_pal.append(sns.color_palette('colorblind')[i])
    else:
        ori_pal.append(sns.color_palette('colorblind')[7])



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
                                        "xtick.major.size":3,
                                        "ytick.major.size":3,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":0.5}):

    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    plt.figure(figsize = (2.5,1.5))
    ax = sns.swarmplot(x = x, y = y, s = 3, edgecolor='k', linewidth = 0.5,
                       zorder = 1, hue = x, palette = ori_pal)


    ax = sig_label(ax, (2,3), 0.06, 0.02, '**')
    ax = sig_label(ax,(3,4), 0.06, 0.02, '**')

    ax.legend_.set_visible(False)

    # distance across the "X" or "Y" stipplot column to span
    mean_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y")

        # calculate the median value for all replicates of either X or Y
        mean_val = y[x==sample_name].mean()
        std_val = y[x==sample_name].std()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(x)/len(np.unique(x))
        ci_val = mean_confidence_interval(y[x==sample_name], confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        ax.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      lw=0.5, color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        ax.errorbar(tick, mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)

    # ax.set_xticks(np.arange(0,180,22.5))
    ax.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Stimulus orientation', labelpad = 1)
    ax.set_ylabel('Convexity', labelpad = 1)
    ax.set_ylim([-0.2,0.7])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    sns.despine(trim = True)
    plt.tight_layout()

    savefile = join(results_dir,'Figures','Draft','population_convexity.svg')
    # plt.savefig(savefile, format = 'svg')


    slope, intercept, r_value, p_value, std_err = stats.linregress(ps_diff.flatten(),y)
    x_mdl = np.linspace(ps_diff.flatten().min()-0.2,ps_diff.flatten().max()+0.2,100)

    plt.figure(figsize=(1.5,1.5))
    ax = sns.scatterplot(x=ps_diff.flatten(),y=y, edgecolor = 'k', linewidth = 0.5,
                         s = 6, hue = x, legend = False, palette = ori_pal)
    ax.plot(x_mdl,x_mdl*slope + intercept, '--k')
    ax.set_xlabel(r'$\Delta$population sparseness', labelpad = 1)
    ax.set_ylabel('Convexity', labelpad = 1)
    ax.text(0.1,1.55-1,'r = ' + str(np.round(r_value,3)) + '\n p < 0.0001',
            multialignment='center',
            horizontalalignment = 'center')
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_ylim([-0.2,0.7])
    ax.set_xlim([-1,4])
    plt.tight_layout()
    ax.set_xticks(np.arange(-0.5,4.5,1))
    sns.despine(trim = True)

    savefile = join(results_dir,'Figures','Draft','population_convexity_vs_kurtosis.svg')
    # plt.savefig(savefile, format = 'svg')

#%% Compare TR to slope ratio

from scipy import stats

stim_label = np.repeat(np.arange(0,180,22.5)[None,:],5,axis=0)

dict_spar = {'stim' : stim_label.flatten(),
             'TR' : ps_diff.flatten(),
             'RS' : ratio_slope.flatten()}

df_spar = pd.DataFrame(dict_spar)

r,p = stats.pearsonr(df_spar.TR,df_spar.RS)

if p < 0.0001:
    p_val = 'p < 0.0001'

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1):

    plt.figure(figsize=(10,9))
    ax = sns.regplot(data = df_spar, x = 'TR', y = 'RS')
    ax.text(0.18,1.65,'r = ' + str(np.round(r,3))
            +'\n' + p_val)
    sns.despine()
    # ax.set_xlabel('Treves-Rolls')
    ax.set_xlabel('Kurtosis')
    ax.set_ylabel('Ratio of slopes')


#%% Compare slope difference to ratio

from scipy import stats

stim_label = np.repeat(np.arange(0,180,22.5)[None,:],5,axis=0)

dict_spar = {'stim' : stim_label.flatten(),
             'SD' : diff_slope.flatten(),
             'RS' : ratio_slope.flatten()}

df_spar = pd.DataFrame(dict_spar)

r,p = stats.pearsonr(df_spar.SD,df_spar.RS)

if p < 0.0001:
    p_val = 'p < 0.0001'

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1):

    plt.figure(figsize=(10,9))
    ax = sns.regplot(data = df_spar, x = 'RS', y = 'SD')
    ax.text(0.15,0.45,'r = ' + str(np.round(r,3))
            +'\n' + p_val)
    sns.despine()
    ax.set_ylabel('Difference of slopes')
    ax.set_xlabel('Slope ratio')


#%% Trial by trial sparsening


# Get average responses for naive, if you haven't already
r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) -1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins)-1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_naive = np.zeros((int(len(subjects)/2),8,n_p,n_r))

for i,(s,t) in enumerate(zip(subjects[0::2],trained[0::2])):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    th = df_scm.loc[ind,'th']
    r = df_scm.loc[ind,'r']

    stim_resps = ori_dict['mean_ori_test'][:-1,ind]

    for o in range(len(np.unique(df_scm.pref_bin))):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r, weights=stim_resps[o,:],
                                                        bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])

        mean_ori_naive[i,o,...] = histo_sum/histo_count


# For held-out trained trials, average responses by cell class

mean_ori_trials = np.zeros(int(len(subjects)/2), dtype = object)
stim_trials = np.zeros(int(len(subjects)/2), dtype = object)

for i,(s,t) in enumerate(zip(subjects[1::2],trained[1::2])):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    th = df_scm.loc[ind,'th']
    r = df_scm.loc[ind,'r']

    trial_ind = np.logical_and(trials_dict['subject']==s,trials_dict['trained']==t)
    n_trials = trials_dict['trial_num'][trial_ind].max()+1
    trial_resps = trials_dict['trial_resps'][trial_ind,].reshape(n_trials,-1)
    stim = trials_dict['stim_ori'][trial_ind].reshape(n_trials,-1)[:,0]
    train_ind = trials_dict['train_ind'][trial_ind].reshape(n_trials,-1)[:,0]

    trial_resps = trial_resps[~train_ind,:]
    stim = stim[~train_ind]
    nb_ind = stim != np.inf
    stim = stim[nb_ind]
    trial_resps = trial_resps[nb_ind,:]
    stim_trials[i] = stim

    mean_ori_trials[i] = np.zeros((trial_resps.shape[0],n_p,n_r))

    for it, t in enumerate(trial_resps):

        histo_sum,x_edge,y_edge = np.histogram2d(th,r, weights = t,
                                                 bins = [pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th,r,bins=[pref_bins,r_bins])

        mean_ori_trials[i][it,...] = histo_sum/histo_count

#%% Plot distribution of stimulus responses by cell class

all_trials = np.vstack(mean_ori_trials)
all_stim = np.concatenate(stim_trials)
all_stim = np.tile(all_stim[:,None,None], (1,) + all_trials.shape[1:])
pref_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,:,None], (all_trials.shape[0], 1, all_trials.shape[2]))
r_label = np.tile(np.arange(5)[None,None,:], all_trials.shape[:2] + (1,))

df_plot = pd.DataFrame({'Response': all_trials.ravel(),
                        'Stimulus': all_stim.ravel(),
                        'Ori. pref.': pref_label.ravel(),
                        'Selectivity': r_label.ravel()})

df_plot['High selectivity'] = df_plot.Selectivity >= 3

df_plot['Selectivity'] = df_plot['Selectivity'].astype(str)

i_stim = 45

sns.displot(df_plot[df_plot.Stimulus==i_stim], x='Response', col='Ori. pref.', col_wrap=4, hue='Selectivity', kind='kde')
sns.displot(df_plot[df_plot.Stimulus==i_stim], x='Response', col='Ori. pref.', col_wrap=4, hue='High selectivity', kind='kde')


#%% Trial by trial sparsening - use all trials by alternating between train and test sets


# Get average responses for naive, if you haven't already
r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) -1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins)-1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_naive = np.zeros((int(len(subjects)/2),8,n_p,n_r))

for i,(s,t) in enumerate(zip(subjects[0::2],trained[0::2])):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    th = df_scm.loc[ind,'th']
    r = df_scm.loc[ind,'r']

    stim_resps = ori_dict['mean_ori_test'][:-1,ind]

    for o in range(len(np.unique(df_scm.pref_bin))):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r, weights=stim_resps[o,:],
                                                        bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])

        mean_ori_naive[i,o,...] = histo_sum/histo_count


mean_ori_trials = np.zeros(int(len(subjects)/2), dtype = object)
stim_trials = np.zeros(int(len(subjects)/2), dtype = object)

for i,(s,t) in enumerate(zip(subjects[1::2],trained[1::2])):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    th_train = df_scm.loc[ind,'th']
    r_train = df_scm.loc[ind,'r']

    th_test = df_scm.loc[ind,'th_test']
    r_test= df_scm.loc[ind,'r_test']

    trial_ind = (trials_dict['subject']==s) & (trials_dict['trained']==t) & ~np.isinf(trials_dict['stim_ori'])
    n_trials = len(np.unique(trials_dict['trial_num'][trial_ind]))
    trial_resps = trials_dict['trial_resps'][trial_ind].reshape(n_trials,-1)
    stim = trials_dict['stim_ori'][trial_ind].reshape(n_trials,-1)[:,0]
    train_ind = np.where(trials_dict['train_ind'][trial_ind].reshape(n_trials,-1)[:,0])[0]
    test_ind = np.where(~trials_dict['train_ind'][trial_ind].reshape(n_trials,-1)[:,0])[0]

    stim_trials[i] = stim

    mean_ori_trials[i] = np.zeros((trial_resps.shape[0],n_p,n_r))


    # test resps

    for it in test_ind:

        histo_sum,x_edge,y_edge = np.histogram2d(th_train,r_train, weights = trial_resps[it,:],
                                                 bins = [pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th_train,r_train,bins=[pref_bins,r_bins])

        mean_ori_trials[i][it,...] = histo_sum/histo_count

    # train resps

    for it in train_ind:

        histo_sum,x_edge,y_edge = np.histogram2d(th_test,r_test, weights = trial_resps[it,:],
                                                 bins = [pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th_test,r_test,bins=[pref_bins,r_bins])

        mean_ori_trials[i][it,...] = histo_sum/histo_count


#%% Fit piecewise linear to each trial

def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y


# Fit piecewise linear
fit_params_trials = np.zeros(int(len(subjects)/2), dtype=object)


for i in range(int(len(subjects)/2)):

    fit_params_trials[i] = np.zeros((mean_ori_trials[i].shape[0],3))

    for it,t in enumerate(mean_ori_trials[i]):

        stim_ind = stim_trials[i][it] == np.unique(stim_trials[i])

        fit_params_trials[i][it,:], _ = so.curve_fit(fitfun,
                                                 mean_ori_naive[i,stim_ind,...].ravel(),
                                                 t.ravel(), max_nfev=1500,
                                  p0=[.5, .5, 1], bounds=((0,0,0),(1,3,3)))

#%% Plot some trials and the fits

i = 4
it = 40


stim_ind = stim_trials[i][it] == np.unique(stim_trials[i])

plt.plot(mean_ori_naive[i,stim_ind,...].flatten(),mean_ori_trials[i][it,...].flatten(),'.')
xr = np.linspace(0,1.3,100)
plt.plot(xr, fitfun(xr, *fit_params_trials[i][it,...]), color=[.2,.2,.2])
plt.title(str(stim_trials[i][it]))
plt.plot([0,1.5],[0,1.5], '--', color=[.5,.5,.5])
plt.xlim([0,1.5]), plt.ylim([0,1.5])

#%% For each trial measure sparseness by convexity of activity

trial_convexity = np.zeros(int(len(subjects)/2), dtype = object)

for i in range(int(len(subjects)/2)):

    trial_convexity[i] = np.zeros(len(stim_trials[i]))

    for it,t in enumerate(mean_ori_trials[i]):
        stim_ind = stim_trials[i][it] == np.unique(stim_trials[i])

        x_pref = mean_ori_naive[i,stim_ind,stim_ind,-1]
        y_pref = t[stim_ind,-1]

        top = y_pref/x_pref

        x_others = mean_ori_naive[i,stim_ind,~stim_ind,:]
        y_others = t[~stim_ind,:]

        bot,_,_,_ = np.linalg.lstsq(x_others.reshape(-1,1),
                                                y_others.reshape(-1,1),
                                                rcond = None)
        trial_convexity[i][it] = top/bot

#%% For each trial find ratio of slope of p,q and maximal point

trial_convexity = np.zeros(int(len(subjects)/2), dtype = object)

for i in range(int(len(subjects)/2)):

    trial_convexity[i] = np.zeros(len(stim_trials[i]))

    for t in range(len(stim_trials[i])):

        stim_ind = stim_trials[i][t] == np.unique(stim_trials[i])

        x = mean_ori_naive[i,stim_ind,...].flatten()
        y = mean_ori_trials[i][t,...].flatten()

        dist = np.sqrt(x**2 + y**2)

        m_ind = np.argmax(dist)

        m_slope = y[m_ind]/x[m_ind]
        f_slope = fit_params_trials[i][t,1]/fit_params_trials[i][t,0]

        trial_convexity[i][t] = m_slope/f_slope

#%% Ratio of p and q

trial_convexity = np.zeros(int(len(subjects)/2), dtype = object)

for i in range(int(len(subjects)/2)):

    trial_convexity[i] = np.zeros(len(stim_trials[i]))

    for t in range(len(stim_trials[i])):

        trial_convexity[i][t] = fit_params_trials[i][t,0]/fit_params_trials[i][t,1]

#%% Save file for use in pixel map analysis

tc_save_files = [join(results_dir, subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), r'_'.join([subjects_file[i],
                    'trial_convexity.npy'])) for i in range(1,len(subjects_file),2)]

for i,(s,t) in enumerate(zip(subjects[1::2],trained[1::2])):

    trial_ind = np.logical_and(trials_dict['subject']==s,trials_dict['trained']==t)
    n_trials = trials_dict['trial_num'][trial_ind].max()+1
    stim = trials_dict['stim_ori'][trial_ind].reshape(n_trials,-1)[:,0]
    train_ind = trials_dict['train_ind'][trial_ind].reshape(n_trials,-1)[:,0]

    test_ind = np.where(~train_ind)[0]
    stim = stim[test_ind]
    nb_ind = stim != np.inf
    test_ind = test_ind[nb_ind]

    tc = np.zeros((len(trial_convexity[i]),3))
    tc[:,0] = trial_convexity[i]
    tc[:,1] = stim_trials[i]
    tc[:,2] = test_ind

    np.save(tc_save_files[i],tc)


#%% Save file for use in pixel map analysis - both train and test trials

tc_save_files = [join(results_dir, subjects_file[i], expt_dates[i],
                    str(expt_nums[i]), r'_'.join([subjects_file[i],
                    'trial_convexity_all_trials.npy'])) for i in range(1,len(subjects_file),2)]

for i,(s,t) in enumerate(zip(subjects[1::2],trained[1::2])):

    tc = np.zeros((len(trial_convexity[i]),3))
    tc[:,0] = trial_convexity[i]
    tc[:,1] = stim_trials[i]

    np.save(tc_save_files[i],tc)

#%% Plot std of trial convexity

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

sub_label = np.concatenate([np.repeat(s,len(stim_trials[i]))
                            for i,s, in enumerate(subjects[0::2])])

tc_dict = {'stim' : np.concatenate(stim_trials).astype(int),
           'tc' : np.concatenate(trial_convexity)-1,
           'subject' : sub_label}


df_tc = pd.DataFrame(tc_dict)

df_tc = df_tc.groupby(['subject','stim']).std().reset_index()

# df_tc = df_tc.groupby(['subject','stim']).mean().reset_index()

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":6,"axes.titlesize":6,
                                        "axes.labelsize":6,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":6,
                                        "ytick.major.size":6,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":1}):

    plt.figure(figsize=(2.75,1.5))
    tc_ax = sns.violinplot(data = df_tc, x = 'stim', y = 'tc',
                s = 1)

    mean_width = 0.5

    for tick, text in zip(tc_ax.get_xticks(), tc_ax.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # # calculate the median value for all replicates of either X or Y
        # mean_val = df_tc[df_tc['stim'].astype(str)==sample_name].tc.mean()

        # # plot horizontal lines across the column, centered on the tick
        # # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        # #               [mean_val, mean_val], lw=4, color='k')
        # n_points = len(df_tc)/len(np.unique(df_tc['stim']))
        # ci_val = mean_confidence_interval(
        #     df_tc[df_tc['stim'].astype(str)==sample_name].tc, confidence = 0.68)
        # # pdif_fig.plot([mean_val, mean_val],
        # #               [tick-mean_width/2, tick+mean_width/2],
        # #               lw=4, color='k', linestyle='--')
        # # pdif_fig.plot([ci_val1, ci_val2],
        # #               [tick, tick],
        # #               lw=4, color='k')
        # tc_ax.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val],
        #               color='k', linestyle='--', zorder = 2)
        # # pdif_fig.plot([tick, tick],
        # #               [ci_val1, ci_val2],
        # #               lw=4, color='k')
        # tc_ax.errorbar(tick,mean_val, ci_val, ecolor = 'k',
        #                   capsize = 3,capthick=0.5, zorder = 2)



    sns.despine()
    plt.xlabel('Stimulus orientation (deg)')
    plt.ylabel('STD of trial convexity')
    plt.tight_layout()

    # sns.displot(data = df_tc, hue = 'stim', x = 'tc', palette = 'hls',
    #             kind = 'ecdf')


#%% Plot distribution of trial convexity

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

sub_label = np.concatenate([np.repeat(s,len(stim_trials[i]))
                            for i,s, in enumerate(subjects[0::2])])

tc_dict = {'stim' : np.concatenate(stim_trials).astype(int),
           'tc' : np.concatenate(trial_convexity)-1,
           'subject' : sub_label}

df_tc = pd.DataFrame(tc_dict)

# df_tc = df_tc.groupby(['subject','stim']).mean().reset_index()

sns.set_theme()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":6,"axes.titlesize":6,
                                        "axes.labelsize":6,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":6,
                                        "ytick.major.size":6,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":1}):

    plt.figure(figsize=(2.75,1.5))
    tc_ax = sns.violinplot(data = df_tc, x = 'stim', y = 'tc',
                s = 1)

    mean_width = 0.5

    for tick, text in zip(tc_ax.get_xticks(), tc_ax.get_xticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # # calculate the median value for all replicates of either X or Y
        # mean_val = df_tc[df_tc['stim'].astype(str)==sample_name].tc.mean()

        # # plot horizontal lines across the column, centered on the tick
        # # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        # #               [mean_val, mean_val], lw=4, color='k')
        # n_points = len(df_tc)/len(np.unique(df_tc['stim']))
        # ci_val = mean_confidence_interval(
        #     df_tc[df_tc['stim'].astype(str)==sample_name].tc, confidence = 0.68)
        # # pdif_fig.plot([mean_val, mean_val],
        # #               [tick-mean_width/2, tick+mean_width/2],
        # #               lw=4, color='k', linestyle='--')
        # # pdif_fig.plot([ci_val1, ci_val2],
        # #               [tick, tick],
        # #               lw=4, color='k')
        # tc_ax.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val],
        #               color='k', linestyle='--', zorder = 2)
        # # pdif_fig.plot([tick, tick],
        # #               [ci_val1, ci_val2],
        # #               lw=4, color='k')
        # tc_ax.errorbar(tick,mean_val, ci_val, ecolor = 'k',
        #                   capsize = 3,capthick=0.5, zorder = 2)



    sns.despine()
    plt.xlabel('Stimulus orientation (deg)')
    plt.ylabel('Trial convexity')
    plt.tight_layout()

    # sns.displot(data = df_tc, hue = 'stim', x = 'tc', palette = 'hls',
    #             kind = 'ecdf')


#%% Trial by trial sparsening - split cell population in two

seed = 10


def seed_rand_choice(x,n_choice,seed):
    np.random.seed(seed=seed)
    return np.random.choice(x,n_choice,replace = False)

# Get average responses for naive, if you haven't already
r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) -1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins)-1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_naive = np.zeros((int(len(subjects)/2),8,n_p,n_r))

task_ret_th = -1

for i,(s,t) in enumerate(zip(subjects[0::2],trained[0::2])):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    if task_ret_th > -1:
        ind = np.logical_and(ind, df_scm.ROI_task_ret <= task_ret_th)

    th = df_scm.loc[ind,'th'].to_numpy()
    r = df_scm.loc[ind,'r'].to_numpy()

    stim_resps = ori_dict['mean_ori_test'][:-1,ind]

    for o in range(len(np.unique(df_scm.pref_bin))):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r, weights=stim_resps[o,:],
                                                        bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])

        mean_ori_naive[i,o,...] = histo_sum/histo_count

# For held-out trained trials, average responses by cell class, put into two groups

mean_ori_trials = np.zeros((int(len(subjects)/2),2), dtype = object)
stim_trials = np.zeros(int(len(subjects)/2), dtype = object)

for i,(s,t) in enumerate(zip(subjects[1::2],trained[1::2])):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    roi_task_ret = df_scm.loc[ind,'ROI_task_ret'].to_numpy()

    if task_ret_th > -1:
        sub_ind = roi_task_ret<=task_ret_th
    else:
        sub_ind = np.arange(len(roi_task_ret))

    th = df_scm.loc[ind,'th'].to_numpy()[sub_ind]
    r = df_scm.loc[ind,'r'].to_numpy()[sub_ind]

    trial_ind = np.logical_and(trials_dict['subject']==s,trials_dict['trained']==t)
    n_trials = trials_dict['trial_num'][trial_ind].max()+1
    trial_resps = trials_dict['trial_resps'][trial_ind].reshape(n_trials,-1)
    stim = trials_dict['stim_ori'][trial_ind].reshape(n_trials,-1)[:,0]
    train_ind = trials_dict['train_ind'][trial_ind].reshape(n_trials,-1)[:,0]

    trial_resps = trial_resps[~train_ind,:]
    stim = stim[~train_ind]
    nb_ind = stim != np.inf
    stim = stim[nb_ind]
    trial_resps = trial_resps[nb_ind,:]
    trial_resps = trial_resps[:,sub_ind]

    th_bin = df_scm.loc[ind,'pref_bin'].to_numpy()[sub_ind]
    r_bin = df_scm.loc[ind,'r_bin'].to_numpy()[sub_ind]

    cell_classes = np.concatenate((th_bin[:,None], r_bin[:,None]),axis=1)
    uni_classes = np.unique(cell_classes,axis=0)
    cell_code = np.zeros(len(th_bin))
    for i_c,u in enumerate(uni_classes):
        cell_code[np.all(cell_classes == u,axis=1)] = i_c

    if seed == -1:
        ind_0 = np.concatenate([np.where(cell_code==c)[0][0::2] for c in np.unique(cell_code)])
        ind_1 = np.concatenate([np.where(cell_code==c)[0][1::2] for c in np.unique(cell_code)])

    else:
        ind_0 = np.concatenate([seed_rand_choice(np.where(cell_code==c)[0],
                                    np.ceil((cell_code==c).sum()/2).astype(int),
                                    seed = seed)
                                    for c in np.unique(cell_code)])
        ind_1 = np.delete(np.arange(len(cell_code)), ind_0)

    th_0,th_1 = th[ind_0],th[ind_1]
    r_0,r_1 = r[ind_0],r[ind_1]

    trial_resps_0 = trial_resps[:,ind_0]
    trial_resps_1 = trial_resps[:,ind_1]

    stim_trials[i] = stim

    mean_ori_trials[i,0] = np.zeros((trial_resps_0.shape[0],n_p,n_r))
    mean_ori_trials[i,1] = np.zeros((trial_resps_1.shape[0],n_p,n_r))

    for it, t in enumerate(trial_resps_0):

        histo_sum,x_edge,y_edge = np.histogram2d(th_0,r_0, weights = t,
                                                 bins = [pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th_0,r_0,bins=[pref_bins,r_bins])

        mean_ori_trials[i,0][it,...] = histo_sum/histo_count


    for it, t in enumerate(trial_resps_1):

        histo_sum,x_edge,y_edge = np.histogram2d(th_1,r_1, weights = t,
                                                 bins = [pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th_1,r_1,bins=[pref_bins,r_bins])

        mean_ori_trials[i,1][it,...] = histo_sum/histo_count



#% Fit piecewise linear to each trial

def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y


# Fit piecewise linear
fit_params_trials = np.zeros((int(len(subjects)/2),2), dtype=object)

for i in range(int(len(subjects)/2)):

    fit_params_trials[i,0] = np.zeros((mean_ori_trials[i,0].shape[0],3))

    for it,t in enumerate(mean_ori_trials[i,0]):

        stim_ind = stim_trials[i][it] == np.unique(stim_trials[i])

        fit_params_trials[i,0][it,:], _ = so.curve_fit(fitfun,
                                                 mean_ori_naive[i,stim_ind,...].ravel(),
                                                 t.ravel(), max_nfev=1500,
                                  p0=[.8, .5, 1], bounds=((0,0,0),(1,3,3)))

    fit_params_trials[i,1] = np.zeros((mean_ori_trials[i,1].shape[0],3))

    for it,t in enumerate(mean_ori_trials[i,1]):

        stim_ind = stim_trials[i][it] == np.unique(stim_trials[i])

        fit_params_trials[i,1][it,:], _ = so.curve_fit(fitfun,
                                                 mean_ori_naive[i,stim_ind,...].ravel(),
                                                 t.ravel(), max_nfev=1500,
                                  p0=[.8, .5, 1], bounds=((0,0,0),(1,3,3)))


#%% Plot some trials and the fits - both cell populations

import matplotlib.ticker as ticker

tick_spacing = 0.5

i = 3

ind_trials = stim_trials[i]==45
num_trials = ind_trials.sum()

it = np.random.choice(np.where(ind_trials)[0],num_trials,replace=False)

# it = 19

cond_color = sns.color_palette('colorblind')[0:2]

pref_colors = sns.hls_palette(8)
r_markers = ['X','P','s','v','*']

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
                                        "lines.markersize":4}):

    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    for iit in it:

        stim_ind = stim_trials[i][iit] == np.unique(stim_trials[i])


        fig,ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = (2.5,1.5))


        for p in range(8):
            for r in range(5):

                ax[0].plot(mean_ori_naive[i,stim_ind,p,r],
                           mean_ori_trials[i,0][iit,p,r],
                           color = pref_colors[p],
                           marker = r_markers[r],
                           markeredgecolor = 'k',
                           markeredgewidth = 0.5)

                ax[1].plot(mean_ori_naive[i,stim_ind,p,r],
                           mean_ori_trials[i,1][iit,p,r],
                           color = pref_colors[p],
                           marker = r_markers[r],
                           markeredgecolor='k',
                           markeredgewidth = 0.5)


        xr = np.linspace(0,1,100)
        ax[0].plot(xr, fitfun(xr, *fit_params_trials[i,0][iit,...]), color=[.2,.2,.2])
        ax[0].set_title(str(int(stim_trials[i][iit])))
        ax[0].plot([0,1],[0,1], '--', color=[.5,.5,.5])
        ax[0].set_xlim([-0.1,1.3]), ax[0].set_ylim([-0.1,1.3])
        ax[0].set_ylabel('Proficient response',color = cond_color[1])
        ax[0].set_xlabel('Naïve response', color = cond_color[0])
        ax[0].set(adjustable='box', aspect='equal')
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        sns.despine(ax = ax[0], trim = True)

        ax[1].plot(xr, fitfun(xr, *fit_params_trials[i,1][iit,...]), color=[.2,.2,.2])
        # plt.title(str(stim_trials[i][it]))
        ax[1].plot([0,1],[0,1], '--', color=[.5,.5,.5])
        ax[1].set_xlim([-0.1,1.3]), ax[1].set_ylim([-0.1,1.3])
        ax[1].set_xlabel('Naïve response', color = cond_color[0])
        ax[1].set(adjustable='box', aspect='equal')
        ax[1].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        sns.despine(ax = ax[1], trim = True)

        fig.tight_layout()



#%% For each trial measure sparsness by convexity of activity - two populations

trial_convexity = np.zeros(int(len(subjects)/2), dtype = object)

for i in range(int(len(subjects)/2)):

    trial_convexity[i] = np.zeros((len(stim_trials[i]),2))

    for t in range(len(mean_ori_trials[i,0])):
        stim_ind = stim_trials[i][t] == np.unique(stim_trials[i])

        x_pref = mean_ori_naive[i,stim_ind,stim_ind,-1]
        y_pref = mean_ori_trials[i,0][t,stim_ind,-1]

        top = y_pref/x_pref

        x_others = mean_ori_naive[i,stim_ind,~stim_ind,:]
        y_others = mean_ori_trials[i,0][t,~stim_ind,:]

        bot,_,_,_ = np.linalg.lstsq(x_others.reshape(-1,1),
                                                y_others.reshape(-1,1),
                                                rcond = None)
        trial_convexity[i][t,0] = top/bot


        y_pref = mean_ori_trials[i,1][t,stim_ind,-1]

        top = y_pref/x_pref

        y_others = mean_ori_trials[i,1][t,~stim_ind,:]

        bot,_,_,_ = np.linalg.lstsq(x_others.reshape(-1,1),
                                                y_others.reshape(-1,1),
                                                rcond = None)
        trial_convexity[i][t,1] = top/bot


#%% For each trial measure sparsness by convexity of activity - two populations

# trial_convexity = np.zeros(int(len(subjects)/2), dtype = object)

# for i in range(int(len(subjects)/2)):

#     trial_convexity[i] = np.zeros((len(stim_trials[i]),2))

#     for t in range(len(fit_params_trials[i,1])):

#         trial_convexity[i][t,0] = fit_params_trials[i,0][t,0]/fit_params_trials[i,0][t,1]
#         trial_convexity[i][t,1] = fit_params_trials[i,1][t,0]/fit_params_trials[i,1][t,1]


#%% For each trial measure sparsness by taking ratio of slope of p,q and max point - two populations

# trial_convexity = np.zeros(int(len(subjects)/2), dtype = object)

# for i in range(int(len(subjects)/2)):

#     trial_convexity[i] = np.zeros((len(stim_trials[i]),2))

#     for t in range(len(mean_ori_trials[i,0])):
#         stim_ind = stim_trials[i][t] == np.unique(stim_trials[i])

#         x = mean_ori_naive[i,stim_ind,...].flatten()
#         y = mean_ori_trials[i,0][t,...].flatten()

#         dist = np.sqrt(x**2 + y**2)

#         m_ind = np.argmax(dist)
#         # m_ind = np.argmax(y)

#         m_slope = y[m_ind]/x[m_ind]
#         l_slope = fit_params_trials[i,0][t,1]/fit_params_trials[i,0][t,0]

#         trial_convexity[i][t,0] = m_slope/l_slope

#         y = mean_ori_trials[i,1][t,...].flatten()

#         dist = np.sqrt(x**2 + y**2)

#         m_ind = np.argmax(dist)
#         # m_ind = np.argmax(y)

#         m_slope = y[m_ind]/x[m_ind]
#         l_slope = fit_params_trials[i,1][t,1]/fit_params_trials[i,1][t,0]

#         trial_convexity[i][t,1] = m_slope/l_slope


#%% Plot both populations

# sub_label = np.concatenate([np.repeat(s,len(stim_trials[i]))
#                             for i,s, in enumerate(subjects[0::2])])

# sub_label = np.repeat(sub_label[:,None],2,axis=1)

# stim_label = np.repeat(np.concatenate(stim_trials)[:,None],2,axis=1)

i = 3

# stim_label = np.concatenate(stim_trials[i])
stim_label = stim_trials[i]

# tc = np.concatenate(trial_convexity,axis=0)
tc = trial_convexity[i]

# tc_dict = {'stim' : stim_label.flatten().astype(int).astype(str),
#             'tc_0' : tc[:,0],
#             'tc_1' : tc[:,1],
#             'subject' : sub_label.flatten()}

tc_dict = {'stim' : stim_label.flatten().astype(int).astype(str),
            # Subtract one so negative is non-convex
            'tc_0' : tc[:,0]-1,
            'tc_1' : tc[:,1]-1}

df_tc = pd.DataFrame(tc_dict)

istim = ['45','90']

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
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'

    # fg = sns.FacetGrid(data = df_tc, col = 'stim', col_wrap = 4)
    # fg.map_dataframe(sns.regplot, x = 'tc_0', y = 'tc_1')
    # sns.lmplot(data = df_tc, x = 'tc_0', y = 'tc_1', col = 'stim',
    #                 col_wrap = 4, legend = False)
    # plt.figure(figsize=(1.5,1.5))
    f,tc = plt.subplots(1,1,figsize = (1.5,1.5))
    # tc = sns.histplot(data = df_tc, x = 'tc_0', y = 'tc_1', hue = 'stim',
    #             hue_order = ['0','23','45','68','90','113','135','158'])
    # tc = sns.scatterplot(data = df_tc, x = 'tc_0', y = 'tc_1', hue = 'stim',
    #             hue_order = ['0','23','45','68','90','113','135','158'],
    #             edgecolor = None)
    # tc = sns.scatterplot(data = df_tc[df_tc.stim == '45'], x = 'tc_0', y = 'tc_1', hue = 'stim',
    #             edgecolor = None, legend = False)


    sns.scatterplot(data = df_tc[df_tc.stim.isin(istim)], x = 'tc_0', y = 'tc_1', style='stim',
                ax = tc, linewidth = 0.5, edgecolor = 'black', zorder = 2,
                markers = {'45' : 'o', '90' : 'v'})

    tc.plot([-0.5,1.9],[-0.5,1.9],'--k',zorder = 1)

    for i,s in enumerate(istim):
        r = np.corrcoef(df_tc[df_tc.stim==s].tc_0,df_tc[df_tc.stim==s].tc_1)[0,1]
        print(r)
        tc.text(-0.3,1.75+i*0.5,'r = ' + str(np.round(r,3)), fontsize = 5)

    # tc.legend_.set_frame_on(False)
    # tc.legend_.set_title('Stim. ori. (deg)')
    # tc.legend_._legend_box.align = 'left'
    plt.xlabel('Trial convexity - Population 1')
    plt.ylabel('Trial convexity - Population 2')
    plt.tight_layout()
    tc.set(adjustable='box', aspect='equal')
    tc.set_xlim([-0.65,2.2])
    tc.set_ylim([-0.65,2.2])
    tc.set_xticks(np.arange(-0.5,2.5,0.5))
    tc.set_yticks(np.arange(-0.5,2.5,0.5))
    sns.despine(trim = True)

    # savefile = join(results_dir, 'Figures', 'Draft', 'trial_by_trial_convexity_pop_corr_sub_3_stim_45.svg')
    # plt.savefig(savefile, format = 'svg')

#%% For each subject, find mean correlation of trial convexity for each stimulus

tc_corr = np.zeros((5,8))

for i in range(5):
    tc = trial_convexity[i]
    for si,s in enumerate(np.unique(stim_trials[i])):
        tc_corr[i,si] = np.corrcoef(tc[stim_trials[i]==s,0],tc[stim_trials[i]==s,1])[0,1]


#%% Plot for each stim

from scipy.stats import ttest_1samp

p_tc = np.zeros(len(np.unique(stim_trials[0])))

for i in range(len(np.unique(stim_trials[0]))):
    p_tc[i] = ttest_1samp(tc_corr[:,i], popmean = 0)[1]


stim_labels = np.repeat(np.unique(stim_trials[0])[None,:], 5, axis = 0)

df_tc = pd.DataFrame({'stim' : stim_labels.flatten(),
                      'tc' : tc_corr.flatten()})

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":6,"axes.titlesize":6,
                                        "axes.labelsize":6,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":6,
                                        "ytick.major.size":6,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    g = sns.swarmplot(data = df_tc, x = 'stim', y = 'tc', palette = 'colorblind')
    sns.despine()


#%% Trial by trial sparsening - split cell population in two - many splits

def fitfun(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y

seed = 10

n_repeats = 2000

tc_corr = np.zeros((5,8,n_repeats))

# Get average responses for naive, if you haven't already
r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]
# r_bins = [0., 0.25, 0.5, 0.75, 1.]
# r_bins = np.append(np.linspace(0,0.7,6),1)
n_r = len(r_bins) -1
pref_bins = np.linspace(-11.25,180-11.25,9)
n_p = len(pref_bins)-1
r_bin_centers = [(a+b)/2 for a,b in zip(r_bins,r_bins[1:])]

mean_ori_naive = np.zeros((int(len(subjects)/2),8,n_p,n_r))

task_ret_th = -1

session_trial_convexities = np.zeros((n_repeats,5), dtype=object)
session_stim = np.zeros(5, dtype=object)

mean_ori_trials_all = np.zeros((len(subjects[0::2]), n_repeats), dtype=object)


for i,(s,t) in enumerate(zip(subjects[0::2],trained[0::2])):
    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    if task_ret_th > -1:
        ind = np.logical_and(ind, df_scm.ROI_task_ret <= task_ret_th)

    th = df_scm.loc[ind,'th'].to_numpy()
    r = df_scm.loc[ind,'r'].to_numpy()

    stim_resps = ori_dict['mean_ori_test'][:-1,ind]

    for o in range(len(np.unique(df_scm.pref_bin))):

        histo_sum,x_edge,y_edge = np.histogram2d(th, r, weights=stim_resps[o,:],
                                                        bins=[pref_bins,r_bins])
        histo_count,x_edge,y_edge = np.histogram2d(th, r,
                                                       bins=[pref_bins,r_bins])

        mean_ori_naive[i,o,...] = histo_sum/histo_count

# For held-out trained trials, average responses by cell class, put into two groups

np.random.seed(seed=seed)

for i,(s,t) in enumerate(zip(subjects[1::2],trained[1::2])):

    mean_ori_trials = np.zeros(2, dtype = object)

    ind = np.logical_and(df_scm.subject == s, df_scm.trained == t)

    roi_task_ret = df_scm.loc[ind,'ROI_task_ret'].to_numpy()

    if task_ret_th > -1:
        sub_ind = roi_task_ret<=task_ret_th
    else:
        sub_ind = np.arange(len(roi_task_ret))

    th = df_scm.loc[ind,'th'].to_numpy()[sub_ind]
    r = df_scm.loc[ind,'r'].to_numpy()[sub_ind]

    trial_ind = np.logical_and(trials_dict['subject']==s,trials_dict['trained']==t)
    n_trials = trials_dict['trial_num'][trial_ind].max()+1
    trial_resps = trials_dict['trial_resps'][trial_ind].reshape(n_trials,-1)
    stim = trials_dict['stim_ori'][trial_ind].reshape(n_trials,-1)[:,0]
    train_ind = trials_dict['train_ind'][trial_ind].reshape(n_trials,-1)[:,0]

    trial_resps = trial_resps[~train_ind,:]
    stim = stim[~train_ind]
    nb_ind = stim != np.inf
    stim_trials = stim[nb_ind]
    trial_resps = trial_resps[nb_ind,:]
    trial_resps = trial_resps[:,sub_ind]

    session_stim[i] = stim_trials

    th_bin = df_scm.loc[ind,'pref_bin'].to_numpy()[sub_ind]
    r_bin = df_scm.loc[ind,'r_bin'].to_numpy()[sub_ind]

    cell_classes = np.concatenate((th_bin[:,None], r_bin[:,None]),axis=1)
    uni_classes = np.unique(cell_classes,axis=0)
    cell_code = np.zeros(len(th_bin))
    for i_c,u in enumerate(uni_classes):
        cell_code[np.all(cell_classes == u,axis=1)] = i_c

    # Repeat n times

    for n in range(n_repeats):

        print('Repeat ' + str(n))

        ind_0 = np.concatenate([np.random.choice(np.where(cell_code==c)[0],
                                np.ceil((cell_code==c).sum()/2).astype(int),
                                replace = False)
                                for c in np.unique(cell_code)])
        ind_1 = np.delete(np.arange(len(cell_code)), ind_0)

        th_0,th_1 = th[ind_0],th[ind_1]
        r_0,r_1 = r[ind_0],r[ind_1]

        trial_resps_0 = trial_resps[:,ind_0]
        trial_resps_1 = trial_resps[:,ind_1]

        mean_ori_trials[0] = np.zeros((trial_resps_0.shape[0],n_p,n_r))
        mean_ori_trials[1] = np.zeros((trial_resps_1.shape[0],n_p,n_r))

        for it, tr in enumerate(trial_resps_0):

            histo_sum,x_edge,y_edge = np.histogram2d(th_0, r_0, weights = tr,
                                                     bins = [pref_bins,r_bins])
            histo_count,x_edge,y_edge = np.histogram2d(th_0, r_0, bins=[pref_bins,r_bins])

            mean_ori_trials[0][it,...] = histo_sum/histo_count


        for it, tr in enumerate(trial_resps_1):

            histo_sum,x_edge,y_edge = np.histogram2d(th_1,r_1, weights = tr,
                                                     bins = [pref_bins,r_bins])
            histo_count,x_edge,y_edge = np.histogram2d(th_1,r_1,bins=[pref_bins,r_bins])

            mean_ori_trials[1][it,...] = histo_sum/histo_count

        mean_ori_trials_all[i,n] = mean_ori_trials.copy()

        # Fit piecewise linear

        # fit_params_trials = np.zeros((2,len(mean_ori_trials[0]),3))

        # for t in range(len(mean_ori_trials[0])):

        #     stim_ind = stim_trials[t] == np.unique(stim_trials)

        #     try:
        #         fit_params_trials[0,t,:], _ = so.curve_fit(fitfun,
        #                                             mean_ori_naive[i,stim_ind,...].ravel(),
        #                                             mean_ori_trials[0][t,:].ravel(),
        #                                             max_nfev=5000,
        #                             p0=[.8, .5, 1], bounds=((0,0,0),(1,3,3)))
        #     except:
        #         fit_params_trials[0,t,:], _ = so.curve_fit(fitfun,
        #                                             mean_ori_naive[i,stim_ind,...].ravel(),
        #                                             mean_ori_trials[0][t,:].ravel(),
        #                                             max_nfev=5000,
        #                             p0=[.5, .5, 1], bounds=((0,0,0),(1,3,3)))

        #     try:
        #         fit_params_trials[1,t,:], _ = so.curve_fit(fitfun,
        #                                             mean_ori_naive[i,stim_ind,...].ravel(),
        #                                             mean_ori_trials[1][t,:].ravel(),
        #                                             max_nfev=5000,
        #                             p0=[.8, .5, 1], bounds=((0,0,0),(1,3,3)))
        #     except:
        #         fit_params_trials[1,t,:], _ = so.curve_fit(fitfun,
        #                                             mean_ori_naive[i,stim_ind,...].ravel(),
        #                                             mean_ori_trials[1][t,:].ravel(),
        #                                             max_nfev=5000,
        #                             p0=[.5, .5, 1], bounds=((0,0,0),(1,3,3)))

        # Measure trial convexity

        trial_convexity = np.zeros((len(stim_trials),2))

        # for t in range(len(mean_ori_trials[0])):
        #     stim_ind = stim_trials[t] == np.unique(stim_trials)

        #     x = mean_ori_naive[i,stim_ind,:,:].flatten()
        #     y = mean_ori_trials[0][t,:].flatten()

        #     dist = np.sqrt(x**2 + y**2)
        #     m_ind = np.argmax(dist)

        #     m_slope = y[m_ind]/x[m_ind]
        #     l_slope = fit_params_trials[0,t,1]/fit_params_trials[0,t,0]

        #     trial_convexity[t,0] = m_slope/l_slope

        #     y = mean_ori_trials[1][t,:].flatten()

        #     dist = np.sqrt(x**2 + y**2)
        #     m_ind = np.argmax(dist)

        #     m_slope = y[m_ind]/x[m_ind]
        #     l_slope = fit_params_trials[1,t,1]/fit_params_trials[1,t,0]

        #     trial_convexity[t,1] = m_slope/l_slope

        for t in range(len(mean_ori_trials[0])):
            stim_ind = stim_trials[t] == np.unique(stim_trials)

            x_pref = mean_ori_naive[i,stim_ind,stim_ind,-1]
            y_pref = mean_ori_trials[0][t,stim_ind,-1]
            # y_pref = mean_ori_trials_all[i,n][0][t,stim_ind,-1]

            top = y_pref/x_pref

            x_others = mean_ori_naive[i,stim_ind,~stim_ind,:]
            y_others = mean_ori_trials[0][t,~stim_ind,:]

            bot,_,_,_ = np.linalg.lstsq(x_others.reshape(-1,1),
                                                    y_others.reshape(-1,1),
                                                    rcond = None)
            trial_convexity[t,0] = top/bot

            y_pref = mean_ori_trials[1][t,stim_ind,-1]
            # y_pref = mean_ori_trials_all[i,n][1][t,stim_ind,-1]

            
            top = y_pref/x_pref

            y_others = mean_ori_trials[1][t,~stim_ind,:]

            bot,_,_,_ = np.linalg.lstsq(x_others.reshape(-1,1),
                                                    y_others.reshape(-1,1),
                                                    rcond = None)
            trial_convexity[t,1] = top/bot


        # Save repeats
        session_trial_convexities[n,i] = trial_convexity


        # Calculate correlation of tc for each stim

        for si,s in enumerate(np.unique(stim_trials)):
            tc_corr[i,si,n] = np.corrcoef(trial_convexity[stim_trials==s,0],
                                          trial_convexity[stim_trials==s,1])[0,1]


#%% Covariance between cell classes across trials for within stimulus conditions

cov_mats_stim = np.zeros((40,40,8,5))
corr_mats_stim = np.zeros((40,40,8,5))

for i in range(5):
    for si,s in enumerate(np.unique(session_stim[0])):
        stim_ind = session_stim[i] == s
        
        cov_mats_repeats = np.zeros((40,40,n_repeats))
        corr_mats_repeats = np.zeros((40,40,n_repeats))

        
        for r in range(n_repeats):
            
            print(f'Mouse {i}, Stimulus {s}, repeat {r}')
            
            pop_0 = np.stack(mean_ori_trials_all[i,r])[0,stim_ind].reshape(np.sum(stim_ind),-1)
            pop_0_mu = pop_0.mean(0, keepdims=True)
            pop_0 = pop_0 - pop_0_mu
            pop_1 = np.stack(mean_ori_trials_all[i,r])[1,stim_ind].reshape(np.sum(stim_ind),-1)
            pop_1_mu = pop_1.mean(0, keepdims=True)
            pop_1 = pop_1 - pop_1_mu
            
            cov_mats_repeats[...,r] = pop_0.T @ pop_1
            
            c_0 = np.diag(pop_0.T @ pop_0)
            c_1 = np.diag(pop_1.T @ pop_1)
   
            c_0_sqrt = np.sqrt(c_0)
            c_1_sqrt = np.sqrt(c_1)
            normalization_matrix = np.outer(c_0_sqrt, c_1_sqrt)

            corr_mats_repeats[...,r] = (pop_0.T @ pop_1) / normalization_matrix
                                   
        cov_mats_stim[...,si,i] = cov_mats_repeats.mean(-1)
        corr_mats_stim[...,si,i] = corr_mats_repeats.mean(-1)


#%%

f,a = plt.subplots(2,4)


for i in range(8):
    sns.heatmap(corr_mats_stim[...,i,:].mean(-1), vmin=0, vmax=1, ax=a.flat[i])


#%% Covariance between cell classes across all trials and stimulus conditions


cov_mats = np.zeros((40,40,5))
corr_mats = np.zeros((40,40,5))

for i in range(5):
   
    cov_mats_repeats = np.zeros((40,40,n_repeats))
    corr_mats_repeats = np.zeros((40,40,n_repeats))
    
    for r in range(n_repeats):
        
        print(f'Mouse {i}, Stimulus {s}, repeat {r}')
        
        pop_0 = np.stack(mean_ori_trials_all[i,r])[0,:].reshape(-1,40)
        pop_0 = pop_0 - pop_0.mean(0, keepdims=True)
        pop_1 = np.stack(mean_ori_trials_all[i,r])[1,:].reshape(-1,40)
        pop_1 = pop_1 - pop_1.mean(0, keepdims=True)
        
        cov_mats_repeats[...,r] = pop_0.T @ pop_1
        
        c_0 = np.diag(pop_0.T @ pop_0)
        c_1 = np.diag(pop_1.T @ pop_1)

        c_0_sqrt = np.sqrt(c_0)
        c_1_sqrt = np.sqrt(c_1)
        normalization_matrix = np.outer(c_0_sqrt, c_1_sqrt)

        corr_mats_repeats[...,r] = (pop_0.T @ pop_1) / normalization_matrix
        
        
    cov_mats[...,i] = cov_mats_repeats.mean(-1)
    corr_mats[...,i] = corr_mats_repeats.mean(-1)


#%% Scatter plot of cell class responses to a specific stimulus, Pop 0 vs Pop 1


i_subject = 2
i_stim = 45
i_repeat = 100

stim_ind = session_stim[i_subject] == i_stim

resps = np.stack(mean_ori_trials_all[i_subject,i_repeat])[:,stim_ind]
resps_0, resps_1 = resps[0,:], resps[1,:]
pref_label = np.tile(np.ceil(np.arange(0,180,22.5))[None,:,None], (resps_0.shape[0],1,resps_0.shape[2]))
r_label = np.tile(np.arange(5)[None,None,:], (resps_0.shape[:2]) + (1,))

df_plot = pd.DataFrame({'Population 1': resps_0.ravel(),
                        'Population 2': resps_1.ravel(),
                        'Ori. pref.': pref_label.ravel(),
                        'Selectivity': r_label.ravel()})


df_plot['Ori. pref.'] = df_plot['Ori. pref.'].astype(int).astype(str)

g = sns.relplot(df_plot, x='Population 1', y='Population 2', hue='Selectivity', kind='scatter',
            palette='colorblind', col='Ori. pref.', col_wrap=4)

# sns.displot(df_plot, x='Population 1', y='Population 2', hue='Ori. pref.', kind='kde',
#             palette='hls', col='Selectivity', col_wrap=3)


xlims = g.axes.flat[-1].get_xlim()
ylims = g.axes.flat[-1].get_ylim()

for a in g.axes.flat:
    a.plot(xlims, ylims, '--k')


#%% plot coveriance matrix

f,a = plt.subplots(1,1)
sns.heatmap(cov_mats.mean(-1), ax=a, center=0, square=True)

ori_pref_label = np.repeat(np.ceil(np.arange(0,180,22.5)),5)
r_label = np.tile(np.arange(5),8)

pref_change = np.where(np.diff(ori_pref_label))[0]+1

a.hlines(pref_change, *a.get_xlim(), color='white')
a.vlines(pref_change, *a.get_ylim(), color='white')

a.set_xticks(np.linspace(2.5,37.5,8))
a.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
a.set_yticks(np.linspace(2.5,37.5,8))
a.set_yticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
a.set_xlabel('Population 1')
a.set_ylabel('Population 2')

a.set_title('Covariance across cell types')


f,a = plt.subplots(1,1)
sns.heatmap(corr_mats.mean(-1), ax=a, center=0, square=True)

ori_pref_label = np.repeat(np.ceil(np.arange(0,180,22.5)),5)
r_label = np.tile(np.arange(5),8)

pref_change = np.where(np.diff(ori_pref_label))[0]+1

a.hlines(pref_change, *a.get_xlim(), color='white')
a.vlines(pref_change, *a.get_ylim(), color='white')

a.set_xticks(np.linspace(2.5,37.5,8))
a.set_xticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
a.set_yticks(np.linspace(2.5,37.5,8))
a.set_yticklabels(np.ceil(np.arange(0,180,22.5)).astype(int))
a.set_xlabel('Population 1')
a.set_ylabel('Population 2')

a.set_title('Correlation across cell types')


#%% Find 45 and 90 deg trials where r is representative

df_tc = pd.DataFrame()

for i in range(5):

    # For each repeat, calculate r by stim type

    stim = session_stim[i]

    sub_trial_convexities = session_trial_convexities[:,i]

    for si, s in enumerate([45,90]):
        s_ind = stim == s

        stim_tc = np.array([pearsonr(tc[s_ind,0], tc[s_ind,1])[0] for tc in sub_trial_convexities])

        ind_best = np.argsort(np.abs(stim_tc-0.75))[0]

        df_tc = pd.concat([df_tc, pd.DataFrame({'tc0' : sub_trial_convexities[ind_best][s_ind,0],
                                                'tc1' : sub_trial_convexities[ind_best][s_ind,1],
                                                'stim' : np.repeat(s, s_ind.sum()),
                                                'subject' : np.repeat(subjects[1::2][i], s_ind.sum())})])



#%% Plot example

import seaborn.objects as so

df_plot = df_tc.copy().reset_index(drop=True)
df_plot['tc0'] = df_plot.tc0 - 1
df_plot['tc1'] = df_plot.tc1 - 1

df_plot['stim'] = df_plot.stim.astype(str)


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
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):


    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'

    f,a = plt.subplots(1,1, figsize=(1,1))

    (
        so.Plot(df_plot[(df_plot.subject=='SF180613') & ((df_plot.stim=='45') | (df_plot.stim=='90'))], x='tc0', y='tc1')
        .add(so.Dot(pointsize=3, color='lightgray', edgecolor='black'), marker='stim', legend=False)
        .label(x='Population 1 - trial convexity', y='Population 2 - trial convexity')
        .scale(marker=so.Nominal(['o','v']),
               x=so.Continuous().tick(at=[0,1,2]),
               y=so.Continuous().tick(at=[0,1,2]))
        .limit(x=(-0.3,2.2), y=(-0.3,2.2))
        .on(a)
        .plot()
    )

    z_45 = np.polyfit(df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='45')].tc0,
                      df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='45')].tc1, 1)
    z_90 = np.polyfit(df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='90')].tc0,
                      df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='90')].tc1, 1)

    x = np.linspace(-0.25,2,100)

    p45 = np.poly1d(z_45)
    p90 = np.poly1d(z_90)

    a.plot(x, p45(x), linestyle='-', color='black')
    a.plot(x, p90(x), linestyle='--', color='black')

    a.set_box_aspect(1)

    r_45 = pearsonr(df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='45')].tc0,
                    df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='45')].tc1)[0]

    r_90 = pearsonr(df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='90')].tc0,
                    df_plot[(df_plot.subject=='SF180613') & (df_plot.stim=='90')].tc1)[0]


    sns.despine(ax=a, trim=True)


#%% Plot for each stim

from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

p_tc = np.zeros(len(np.unique(stim_trials)))

tc_corr_mean = tc_corr.mean(2)

for i in range(len(np.unique(stim_trials))):
    p_tc[i] = ttest_1samp(tc_corr_mean[:,i], popmean = 0)[1]

# p_tc = multipletests(p_tc,method = 'hs')[1]

stim_labels = np.repeat(np.unique(stim_trials)[None,:], 5, axis = 0)

df_tc = pd.DataFrame({'stim' : stim_labels.flatten().astype(int),
                      'tc' : tc_corr_mean.flatten(),
                      'subject' : np.repeat(np.arange(len(tc_corr_mean))[:,None], 8, axis = 1).flatten()})

def stim_type(x):
    if (x==45) | (x==90):
        return '45 and 90'
    elif x == 68:
        return '68'
    else:
        return 'non-task'

df_tc['stim_type'] = df_tc.stim.map(stim_type)

df_tc = df_tc.groupby(['subject','stim_type']).mean().reset_index()

# ori_pal = []

# for i in range(8):
#     print(i)
#     if i >= 2 and i <= 4:
#         ori_pal.append(sns.color_palette('colorblind')[i])
#     else:
#         ori_pal.append(sns.color_palette('colorblind')[7])

# x_label = [o + r'$\degree$' for o in np.ceil(np.arange(0,180,22.5)).astype(int).astype(str)]

x_label = [r'45$\degree$/90$\degree$', '68$\degree$', 'non-task']

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
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):


    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['font.family'] = 'sans-serif'

    plt.figure(figsize=(1.75,1.4))
    g = sns.swarmplot(data = df_tc, x = 'stim_type', y = 'tc', s = 3, zorder = 1,
                      linewidth = 0.5, edgecolor = 'black', color = 'black')
    g.set_ylim([0,1])
    plt.ylabel('Correlation of trial convexity')
    plt.xlabel('Stimulus orientation')
    # g.legend_.set_visible(False)

    # g.set_yticks(np.arange(-0.2,1.2,0.2))

    mean_width = 0.3

    for i,(tick, text) in enumerate(zip(g.get_xticks(), g.get_xticklabels())):
        sample_name = text.get_text()  # "X" or "Y"

        # calculate the median value for all replicates of either X or Y
        mean_val = df_tc[df_tc['stim_type'].astype(str)==sample_name].tc.mean()

        # plot horizontal lines across the column, centered on the tick
        # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
        #               [mean_val, mean_val], lw=4, color='k')
        n_points = len(df_tc)/len(np.unique(df_tc['stim']))
        ci_val = mean_confidence_interval(
            df_tc[df_tc['stim_type'].astype(str)==sample_name].tc, confidence = 0.68)
        # pdif_fig.plot([mean_val, mean_val],
        #               [tick-mean_width/2, tick+mean_width/2],
        #               lw=4, color='k', linestyle='--')
        # pdif_fig.plot([ci_val1, ci_val2],
        #               [tick, tick],
        #               lw=4, color='k')
        g.plot([tick-mean_width/2, tick+mean_width/2],
                      [mean_val, mean_val],
                      color='k', linestyle='--', zorder = 2)
        # pdif_fig.plot([tick, tick],
        #               [ci_val1, ci_val2],
        #               lw=4, color='k')
        g.errorbar(tick,mean_val, ci_val, ecolor = 'k',
                          capsize = 3,capthick=0.5, zorder = 2)
        # if p_tc[i] < 0.0001:
        #     g.text(tick,1,'****',horizontalalignment='center', fontsize = 7)
        # elif p_tc[i] < 0.001:
        #     g.text(tick,1,'***',horizontalalignment='center', fontsize = 7)
        # elif p_tc[i] < 0.01:
        #     g.text(tick,1,'**',horizontalalignment='center', fontsize = 7)
        # elif p_tc[i] < 0.05:
        #     g.text(tick,1,'*',horizontalalignment='center', fontsize = 7)
        # else:
        #     g.text(tick,1,'n.s.',horizontallignment='center',fontsize = 5)

    g.set_yticks([0,0.25,0.5,0.75,1])
    g.set_xticklabels(x_label)
    sns.despine(trim=True)
    plt.tight_layout()

    savefile = join(results_dir, 'Figures','Draft','trial_to_trial_convex_corr_by_stim type.svg')
    plt.savefig(savefile, format = 'svg')

#%% Fit gaussian tuning curves

g_pars = np.zeros((ori_dict['mean_ori_train'].shape[1],4))

for i,tc in enumerate(ori_dict['mean_ori_train'][:-1,:].T):
    print('Fitting cell ' + str(i))
    g_pars[i,:], _ = op.fit_ori(np.arange(0,180,22.5),tc)


#%% r2 of fit

ori_r2 = np.zeros(len(df_scm))

ori = np.arange(0,180,22.5)

for c in range(len(g_pars)):
    top = np.sum((ori_dict['mean_ori_test'][:-1,c] - ori_dict['mean_ori_test'][:-1,c].mean())**2)
    bot = np.sum((ori_dict['mean_ori_test'][:-1,c] - op.ori_tune(ori,*g_pars[c,:]))**2)
    ori_r2[c] = 1 - top/bot

#%% Plot some stuff

whh = np.sqrt(np.log(2)*2)*g_pars[:,-1]

whh_dict = {'whh' : whh.flatten(),
            'subject' : df_scm.subject.to_numpy().flatten(),
            'trained' : df_scm.trained.to_numpy().flatten(),
            'ori_pref' : g_pars[:,0].flatten()-11.25,
            'r2' : ori_r2.flatten()}

df_whh = pd.DataFrame(whh_dict)
df_whh['ori_pref'] = pd.cut(df_whh.ori_pref, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_whh = df_whh.groupby(['subject','trained','ori_pref']).mean().reset_index()

ax1 = sns.catplot(data = df_whh, x = 'ori_pref', y = 'whh', hue = 'trained', kind = 'box')
ax2 = sns.catplot(data = df_whh, x = 'ori_pref', y = 'r2', hue = 'trained', kind = 'box')


#%% For a random subset of cells, plot mean responses and gaussian fit

rand_cells = np.random.choice(np.arange(len(df_scm)), 10)

ori = np.arange(0,180,22.5)
x = np.linspace(0,157.5,100)

for c in rand_cells:
    tc = ori_dict['mean_ori_all'][:-1,c]
    plt.figure()
    plt.plot(ori,tc)
    plt.plot(x,op.ori_tune(x,*g_pars[c,:]))


#%% Fit two gaussians to directional resps

g_pars_dir = np.zeros((len(df_scm),5))
mu_dir = np.zeros((len(df_scm),16))

cells = np.unique(trials_dict['cell'])

for i,c in enumerate(cells):
    print('Fitting cell ' + str(c))
    cell_ind = trials_dict['cell'] == c
    resps = trials_dict['trial_resps'][cell_ind]
    stim = trials_dict['stim_dir'][cell_ind]
    train_ind = trials_dict['train_ind'][cell_ind]
    stim = stim[train_ind]
    resps = resps[train_ind]
    nb_ind = stim != np.inf
    resps = resps[nb_ind]
    stim = stim[nb_ind]
    # Return to actual directions so spacing is equal
    stim[stim%180 == 23] -= 0.5
    stim[stim%180 == 68] -= 0.5
    stim[stim%180 == 113] -= 0.5
    stim[stim%180 == 158] -= 0.5

    # Fit double gaussian, force baseline to be 0
    g_pars_dir[i,:],_ = op.fit_dir(stim, resps)

    mu_dir[i,:] = np.array([resps[stim==d].mean() for d in np.unique(stim)])

#%% r2 of fit

ori_r2 = np.zeros(len(df_scm))

for i,c in enumerate(cells):
    print('Fitting cell ' + str(c))
    cell_ind = trials_dict['cell'] == c
    resps = trials_dict['trial_resps'][cell_ind]
    stim = trials_dict['stim_dir'][cell_ind]
    test_ind = np.logical_not(trials_dict['train_ind'][cell_ind])
    stim = stim[test_ind]
    resps = resps[test_ind]
    nb_ind = stim != np.inf
    resps = resps[nb_ind]
    stim = stim[nb_ind]
    # Return to actual directions so spacing is equal
    stim[stim%180 == 23] -= 0.5
    stim[stim%180 == 68] -= 0.5
    stim[stim%180 == 113] -= 0.5
    stim[stim%180 == 158] -= 0.5

    top = np.sum((resps - resps.mean(0))**2,0)
    bot = np.sum((resps - op.dir_tune(stim,*g_pars_dir[c,:]))**2,0)
    ori_r2[c] = 1 - top/bot

#%% Plot some stuff

whh = np.sqrt(np.log(2)*2)*g_pars_dir[:,-1]

whh_dict = {'whh' : whh.flatten(),
            'subject' : df_scm.subject.to_numpy().flatten(),
            'trained' : df_scm.trained.to_numpy().flatten(),
            'ori_pref' : np.mod(g_pars_dir[:,0].flatten(),180) - 11.25,
            'r2' : ori_r2.flatten()}

df_whh = pd.DataFrame(whh_dict)
df_whh['ori_pref'] = pd.cut(df_whh.ori_pref, np.linspace(-11.25,180-11.25,9),
                        labels = np.ceil(np.arange(0,180,22.5)))
df_whh = df_whh.groupby(['subject','trained','ori_pref']).mean().reset_index()

ax1 = sns.catplot(data = df_whh, x = 'ori_pref', y = 'whh', hue = 'trained', kind = 'box')
ax2 = sns.catplot(data = df_whh, x = 'ori_pref', y = 'r2', hue = 'trained', kind = 'box')

#%% Compare fit to mean dir responses

# mu_dir = np.zeros((len(df_scm),16))

# for i,c in enumerate(cells):
#     print('Fitting cell ' + str(c))
#     cell_ind = trials_dict['cell'] == c
#     resps = trials_dict['trial_resps'][cell_ind]
#     stim = trials_dict['stim_dir'][cell_ind]
#     nb_ind = stim != np.inf
#     resps = resps[nb_ind]
#     stim = stim[nb_ind]

#     mu_dir[i,:] = np.array([resps[stim==d].mean() for d in np.unique(stim)])

#%% For a random subset of cells, plot mean responses and gaussian fit

def sd_g(dirs, Dp, Rp, Ro, sigma):
    anglesp = 180/np.pi*np.angle(np.exp(1j*(dirs-Dp)*np.pi/180))
    # breakpoint()
    f = Ro + (Rp*np.exp(-anglesp**2 / (2 * sigma**2)))
    return f

rand_cells = np.random.choice(np.arange(len(df_scm)), 20)

ori = np.arange(0,360,22.5)
x = np.linspace(0,360-22.5,100)

for c in rand_cells:
    tc = mu_dir[c,:]
    plt.figure()
    plt.plot(ori,tc)
    plt.plot(x,op.dir_tune(x,*g_pars_dir[c,:]))
    # Plot individual guassians#
    plt.plot(x,sd_g(x,g_pars_dir[c,0],g_pars_dir[c,1],g_pars_dir[c,3],g_pars_dir[c,4]))
    plt.plot(x,sd_g(x,g_pars_dir[c,0]-180,g_pars_dir[c,2],g_pars_dir[c,3],g_pars_dir[c,4]))
    # plt.plot(x,sd_g(x,g_pars_dir[c,0],g_pars_dir[c,1],g_pars_dir[c,3],g_pars_dir[c,4])+
    #             sd_g(x,g_pars_dir[c,0]-180,g_pars_dir[c,2],g_pars_dir[c,3],g_pars_dir[c,4]))


#%% NOISE CORRELATIONS and Variance

# Fit linear model for all neurons for direction
# Look at correlation of residuals by trial type for all cell pairs (this is
# just correlation coefficient matrix)

noise_expts = np.zeros(len(subjects),dtype=object)
stim_dir_expts = np.zeros(len(subjects),dtype=object)
stim_ori_expts = np.copy(stim_dir_expts)
r_expts = np.copy(stim_dir_expts)
pref_expts = np.copy(stim_dir_expts)
pref_modal_expts = np.copy(stim_dir_expts)
mu_expts = np.copy(stim_dir_expts)

for i,(s,t) in enumerate(zip(subjects,trained)):

    expt_ind = np.logical_and(trials_dict['subject']==s,trials_dict['trained']==t)
    n_trials = trials_dict['trial_num'][expt_ind].max()+1

    trial_resps = trials_dict['trial_resps'][expt_ind].reshape(n_trials,-1)
    stim = trials_dict['stim_dir'][expt_ind].reshape(n_trials,-1)[:,0]
    stim_ori = trials_dict['stim_ori'][expt_ind].reshape(n_trials,-1)[:,0]
    test_ind = np.logical_not(trials_dict['train_ind'][expt_ind].reshape(n_trials,-1)[:,0])

    trial_resps = trial_resps[test_ind,:]
    stim = stim[test_ind]
    stim_ori = stim_ori[test_ind]

    stim[stim == np.inf] = -1
    lb = preprocessing.LabelBinarizer()
    uni_stim = np.unique(stim)
    lb.fit(uni_stim)
    dm = lb.transform(stim)
    # Remove blank column so it is intercept
    dm = dm[:,1:]

    # # Average responses to each stimulus
    lr = LinearRegression()
    lr.fit(dm, trial_resps)
    noise_expts[i] = trial_resps - lr.predict(dm)
    # noise_expts[i] = trial_resps
    stim_dir_expts[i] = stim
    stim_ori_expts[i] = stim_ori
    r_expts[i] = df_scm.r_bin[np.logical_and(df_scm.subject == s, df_scm.trained == t)]
    pref_expts[i] = df_scm.pref_bin[np.logical_and(df_scm.subject == s, df_scm.trained == t)]
    pref_modal_expts[i] = df_scm.pref_ori_train[np.logical_and(df_scm.subject == s, df_scm.trained == t)]
    mu_expts[i] = np.array([trial_resps[stim_ori==o,:].mean(0) for o in np.unique(stim_ori)]).T


#%%

n_cells = -1
noise_corrs = np.empty(len(subjects),dtype=object)
sigma = np.empty(len(subjects),dtype=object)
CoV = np.empty(len(subjects),dtype=object)


for i in range(len(subjects)):

    cell_index = np.arange(noise_expts[i].shape[1])

    if n_cells != -1:
        cell_index = np.random.choice(cell_index,n_cells,replace=False)

    noise_corrs[i] = np.zeros((len(cell_index),len(cell_index),
                               len(np.unique(stim_ori_expts[i]))))
    sigma[i] = np.zeros((len(cell_index),len(np.unique(stim_ori_expts[i]))))
    CoV[i] = np.zeros((len(cell_index),len(np.unique(stim_ori_expts[i]))))

    for si,s in enumerate(np.unique(stim_ori_expts[i])):
       stim_ind = stim_ori_expts[i] == s
       print('Noise correlations for stimulus ' + str(s))

       # noise_corrs[i][...,si] = np.corrcoef(noise_expts[i][stim_ind,:].T)
       # Already subtracted mean, so you just do x.y/norm(x)*norm(y)
       noise_corrs[i][...,si] = cosine_similarity(noise_expts[i][stim_ind,:].T)
       sigma[i][:,si] = np.sqrt(np.sum(noise_expts[i][stim_ind,:]**2,0)/noise_expts[i].shape[0])
       CoV[i][:,si] = sigma[i][:,si]/mu_expts[i][:,si]

#%% Average noise correlations per stim and expt

mu_noise_corrs = np.zeros((len(subjects),len(np.unique(stim_ori_expts[0]))))

for i in range(len(subjects)):

    ind = np.triu_indices(noise_corrs[i].shape[0],1)

    for s in range(noise_corrs[i].shape[2]):
        ns_s = noise_corrs[i][...,s]
        mu_noise_corrs[i,s] = np.nanmean(ns_s[ind])


mu_nc_diff = np.array([mu_noise_corrs[i,:]-mu_noise_corrs[i-1,:] for i in range(1,10,2)])
#%% Plot

stim_label = np.repeat(np.unique(stim_ori_expts[0]).reshape(1,-1), len(subjects),axis=0)
cond_label = np.repeat(np.array(trained).reshape(-1,1),
                       len(np.unique(stim_ori_expts[0])), axis = 1)

nc_dict = {'noise_corr' : mu_noise_corrs.flatten(),
           'stim' : stim_label.flatten(),
           'trained' : cond_label.flatten()}

df_nc = pd.DataFrame(nc_dict)

sns.catplot(data = df_nc, x = 'stim', y = 'noise_corr', hue = 'trained', kind = 'box')



#%% Plot diff

stim_label = np.repeat(np.unique(stim_ori_expts[0]).reshape(1,-1), int(len(subjects)/2),axis=0)

nc_dict = {'noise_corr' : mu_nc_diff.flatten(),
           'stim' : stim_label.flatten()}

df_nc = pd.DataFrame(nc_dict)

sns.catplot(data = df_nc, x = 'stim', y = 'noise_corr', kind = 'box')

#%% Plot stdev

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

subject_label = np.concatenate([np.repeat(s,sigma[i].size) for i,s in enumerate(subjects)])
trained_label = np.concatenate([np.repeat(t,sigma[i].size) for i,t in enumerate(trained)])
sigma_all = np.concatenate(sigma,axis=0)
cov_all = np.concatenate(CoV,axis=0)
stim_label = np.tile(np.array([0,23,45,68,90,113,135,158,'Blank'],dtype=object)
                     .reshape(1,-1),(sigma_all.shape[0],1))

r_label = np.repeat(np.concatenate(r_expts,axis=0)[:,None],len(np.unique(stim_ori_expts[0])),axis=1)
pref_label = np.repeat(np.concatenate(pref_expts,axis=0)[:,None],len(np.unique(stim_ori_expts[0])),axis=1)


sigma_dict = {'subject' : subject_label,
              'trained' : trained_label,
              'stim' : stim_label.flatten(),
              'sigma' : sigma_all.flatten(),
              'CoV' : cov_all.flatten(),
              'pref' : pref_label.flatten(),
              'r' : r_label.flatten()}

df_sigma = pd.DataFrame(sigma_dict)

df_sigma = df_sigma.groupby(['subject','trained','stim']).mean().reset_index()


yticklabels = []

for pref in np.ceil(np.arange(0,180,22.5)).astype(int):
    for s in np.arange(1,6):
        yticklabels.append(str(pref) + ', s' + str(s))

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":6,"axes.titlesize":6,
                                        "axes.labelsize":6,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":6,
                                        "ytick.major.size":6,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):
    plt.figure(figsize=(2.75,2))
    # std_ax = sns.stripplot(data = df_sigma, x = 'stim', y = 'sigma', hue = 'trained',
    #             palette = 'colorblind', dodge = True)
    std_ax = sns.lineplot(data = df_sigma[df_sigma.stim != 'Blank'], x = 'stim',
                          y = 'sigma', hue = 'trained',
                palette = 'colorblind', errorbar = 'se')
    plt.ylabel(r'Response $\sigma$')
    plt.xlabel(r'Stimulus orientation (deg)')
    sns.despine()
    std_ax.legend_.set_frame_on(False)
    std_ax.legend_.set_title('')
    std_ax.legend_.texts[0].set_text('Naïve')
    std_ax.legend_.texts[1].set_text('Proficient')
    std_ax.set_xticks(np.ceil(np.arange(0,180,22.5)).astype(int))
    plt.tight_layout()

    # mean_width = 1

    # for tick, text in zip(std_ax.get_xticks(), std_ax.get_xticklabels()):
    #     sample_name = text.get_text()  # "X" or "Y"

    #     # calculate the median value for all replicates of either X or Y
    #     mean_val = df_sigma[df_sigma['stim'].astype(str)==sample_name].sigma.mean()

    #     # plot horizontal lines across the column, centered on the tick
    #     # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
    #     #               [mean_val, mean_val], lw=4, color='k')
    #     n_points = len(df_sigma)/len(np.unique(df_sigma['stim'].astype(str)))
    #     ci_val = mean_confidence_interval(
    #         df_sigma[df_sigma['stim'].astype(str)==sample_name].sigma, confidence = 0.68)
    #     # pdif_fig.plot([mean_val, mean_val],
    #     #               [tick-mean_width/2, tick+mean_width/2],
    #     #               lw=4, color='k', linestyle='--')
    #     # pdif_fig.plot([ci_val1, ci_val2],
    #     #               [tick, tick],
    #     #               lw=4, color='k')
    #     std_ax.plot([tick-mean_width/2, tick+mean_width/2],
    #                   [mean_val, mean_val],
    #                   color='k', linestyle='--', zorder = 2)
    #     # pdif_fig.plot([tick, tick],
    #     #               [ci_val1, ci_val2],
    #     #               lw=4, color='k')
    #     std_ax.errorbar(tick,mean_val, ci_val, ecolor = 'k',
    #                       capsize = 3,capthick=0.5, zorder = 2)

    df_sigma = pd.DataFrame(sigma_dict)
    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.groupby(['subject','trained','r','pref','stim']).mean().reset_index()
    df_diff = pd.DataFrame.copy(df_sigma[df_sigma.trained])
    df_diff['sigma'] = df_diff.sigma.to_numpy() - df_sigma[df_sigma.trained==False].sigma.to_numpy()

    df_diff = df_diff.drop(['trained', 'CoV'],axis=1)

    df_diff = df_diff.pivot_table(index = ['pref','r'], columns = 'stim')
    plt.figure(figsize=(2.75,2.75))
    hm = sns.heatmap(df_diff,center = 0, cbar_kws={'label': r'$\Delta$STD'})
    hm.set_xticklabels(['0','23','45','68','90','113','135','158'],
                       rotation = 'horizontal')
    hm.set_xlabel('Stimulus orientation (deg)')
    hm.set_yticks(np.arange(0,len(yticklabels))+0.5)
    hm.set_yticklabels(yticklabels)
    hm.set_ylabel('Cell type')
    plt.tight_layout()

    df_sigma = pd.DataFrame(sigma_dict)

    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    df_sigma = df_sigma.groupby(['subject','trained','stim']).mean().reset_index()

    # cov_ax = sns.catplot(data = df_sigma, x = 'stim', y = 'CoV', hue = 'trained', kind = 'box',
    #             palette = 'colorblind')

    df_sigma = pd.DataFrame(sigma_dict)
    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df_sigma = df_sigma.groupby(['subject','trained','r','pref','stim']).mean().reset_index()
    df_diff = pd.DataFrame.copy(df_sigma[df_sigma.trained])
    df_diff['CoV'] = df_diff.CoV.to_numpy() - df_sigma[df_sigma.trained==False].CoV.to_numpy()

    df_diff = df_diff.drop(['trained', 'sigma'],axis=1)
    df_diff = df_diff.pivot_table(index = ['pref','r'], columns = 'stim')
    plt.figure(figsize=(2.75,2.75))
    hm = sns.heatmap(df_diff,center = 0, cbar_kws={'label': r'$\Delta$CoV'})
    hm.set_xticklabels(['0','23','45','68','90','113','135','158'],
                       rotation = 'horizontal')
    hm.set_xlabel('Stimulus orientation (deg)')
    hm.set_yticks(np.arange(0,len(yticklabels))+0.5)
    hm.set_yticklabels(yticklabels)
    hm.set_ylabel('Cell type')
    plt.tight_layout()


    # For cells of peak selectivity, show change in COV
    df_sigma = pd.DataFrame(sigma_dict)
    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df_sigma = df_sigma.groupby(['subject','trained','r','pref','stim']).mean().reset_index()
    df_diff = pd.DataFrame.copy(df_sigma[df_sigma.trained])
    df_diff['CoV'] = df_diff.CoV.to_numpy() - df_sigma[df_sigma.trained==False].CoV.to_numpy()

    df_diff = df_diff.drop(['trained', 'sigma'],axis=1)

    df_diff = df_diff[df_diff.r == 4]
    df_diff = df_diff[df_diff.stim == df_diff.pref]

    pref_colors = sns.hls_palette(len(np.unique(df_diff.pref)))

    plt.figure(figsize=(2.75,2))
    sp = sns.stripplot(data = df_diff, x = 'pref', y = 'CoV', palette = pref_colors,
                  hue = 'pref', s = 3)
    sp.legend_.set_visible(False)
    sns.pointplot(data = df_diff, x = 'pref', y = 'CoV', join = False,
                       errorbar = 'sem', markers = '_', capsize = 0.15,
                       linestyles = '--', palette = pref_colors)
    sns.despine()
    plt.tight_layout()
    plt.ylabel(r'$\Delta$CoV')
    plt.xlabel('Preferred mean orientation (deg)')
    sp.set_xticklabels(['0','23','45','68','90','113','135','158'])

#%% Plot stdev - group by modal pref

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

subject_label = np.concatenate([np.repeat(s,sigma[i].size) for i,s in enumerate(subjects)])
trained_label = np.concatenate([np.repeat(t,sigma[i].size) for i,t in enumerate(trained)])
sigma_all = np.concatenate(sigma,axis=0)
cov_all = np.concatenate(CoV,axis=0)
stim_label = np.tile(np.array([0,23,45,68,90,113,135,158,'Blank'],dtype=object)
                     .reshape(1,-1),(sigma_all.shape[0],1))

pref_label = np.repeat(np.concatenate(pref_modal_expts,axis=0)[:,None],len(np.unique(stim_ori_expts[0])),axis=1)


sigma_dict = {'subject' : subject_label,
              'trained' : trained_label,
              'stim' : stim_label.flatten(),
              'sigma' : sigma_all.flatten(),
              'CoV' : cov_all.flatten(),
              'pref' : pref_label.flatten()}

df_sigma = pd.DataFrame(sigma_dict)

df_sigma = df_sigma.groupby(['subject','trained','stim']).mean().reset_index()


yticklabels = []

for pref in np.ceil(np.arange(0,180,22.5)).astype(int):
        yticklabels.append(str(pref))

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", rc={"font.size":6,"axes.titlesize":6,
                                        "axes.labelsize":6,
                                        "xtick.labelsize":5,
                                        "ytick.labelsize":5,
                                        "lines.linewidth":0.5,
                                        "axes.linewidth":0.5,
                                        "xtick.major.width":0.5,
                                        "ytick.major.width":0.5,
                                        "xtick.major.size":6,
                                        "ytick.major.size":6,
                                        "patch.linewidth":0.5,
                                        "lines.markersize":3,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):
    plt.figure(figsize=(2.75,2))
    # std_ax = sns.stripplot(data = df_sigma, x = 'stim', y = 'sigma', hue = 'trained',
    #             palette = 'colorblind', dodge = True)
    std_ax = sns.lineplot(data = df_sigma[df_sigma.stim != 'Blank'], x = 'stim',
                          y = 'sigma', hue = 'trained',
                palette = 'colorblind', errorbar = 'se')
    plt.ylabel(r'Response $\sigma$')
    plt.xlabel(r'Stimulus orientation (deg)')
    sns.despine()
    std_ax.legend_.set_frame_on(False)
    std_ax.legend_.set_title('')
    std_ax.legend_.texts[0].set_text('Naïve')
    std_ax.legend_.texts[1].set_text('Proficient')
    std_ax.set_xticks(np.ceil(np.arange(0,180,22.5)).astype(int))
    plt.tight_layout()

    # mean_width = 1

    # for tick, text in zip(std_ax.get_xticks(), std_ax.get_xticklabels()):
    #     sample_name = text.get_text()  # "X" or "Y"

    #     # calculate the median value for all replicates of either X or Y
    #     mean_val = df_sigma[df_sigma['stim'].astype(str)==sample_name].sigma.mean()

    #     # plot horizontal lines across the column, centered on the tick
    #     # pdif_fig.plot([tick-mean_width/2, tick+mean_width/2],
    #     #               [mean_val, mean_val], lw=4, color='k')
    #     n_points = len(df_sigma)/len(np.unique(df_sigma['stim'].astype(str)))
    #     ci_val = mean_confidence_interval(
    #         df_sigma[df_sigma['stim'].astype(str)==sample_name].sigma, confidence = 0.68)
    #     # pdif_fig.plot([mean_val, mean_val],
    #     #               [tick-mean_width/2, tick+mean_width/2],
    #     #               lw=4, color='k', linestyle='--')
    #     # pdif_fig.plot([ci_val1, ci_val2],
    #     #               [tick, tick],
    #     #               lw=4, color='k')
    #     std_ax.plot([tick-mean_width/2, tick+mean_width/2],
    #                   [mean_val, mean_val],
    #                   color='k', linestyle='--', zorder = 2)
    #     # pdif_fig.plot([tick, tick],
    #     #               [ci_val1, ci_val2],
    #     #               lw=4, color='k')
    #     std_ax.errorbar(tick,mean_val, ci_val, ecolor = 'k',
    #                       capsize = 3,capthick=0.5, zorder = 2)

    df_sigma = pd.DataFrame(sigma_dict)
    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.groupby(['subject','trained','pref','stim']).mean().reset_index()
    df_diff = pd.DataFrame.copy(df_sigma[df_sigma.trained])
    df_diff['sigma'] = df_diff.sigma.to_numpy() - df_sigma[df_sigma.trained==False].sigma.to_numpy()

    df_diff = df_diff.drop(['trained', 'CoV'],axis=1)

    df_diff = df_diff.pivot_table(index = ['pref'], columns = 'stim')
    plt.figure(figsize=(2.75,2.75))
    hm = sns.heatmap(df_diff,center = 0, cbar_kws={'label': r'$\Delta$STD'})
    hm.set_xticklabels(['0','23','45','68','90','113','135','158'],
                       rotation = 'horizontal')
    hm.set_xlabel('Stimulus orientation (deg)')
    hm.set_yticks(np.arange(0,len(yticklabels))+0.5)
    hm.set_yticklabels(yticklabels)
    hm.set_ylabel('Cell type')
    plt.tight_layout()

    df_sigma = pd.DataFrame(sigma_dict)

    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    df_sigma = df_sigma.groupby(['subject','trained','stim']).mean().reset_index()

    # cov_ax = sns.catplot(data = df_sigma, x = 'stim', y = 'CoV', hue = 'trained', kind = 'box',
    #             palette = 'colorblind')

    df_sigma = pd.DataFrame(sigma_dict)
    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df_sigma = df_sigma.groupby(['subject','trained','pref','stim']).mean().reset_index()
    df_diff = pd.DataFrame.copy(df_sigma[df_sigma.trained])
    df_diff['CoV'] = df_diff.CoV.to_numpy() - df_sigma[df_sigma.trained==False].CoV.to_numpy()

    df_diff = df_diff.drop(['trained', 'sigma'],axis=1)
    df_diff = df_diff.pivot_table(index = ['pref'], columns = 'stim')
    plt.figure(figsize=(2.75,2.75))
    hm = sns.heatmap(df_diff,center = 0, cbar_kws={'label': r'$\Delta$CoV'})
    hm.set_xticklabels(['0','23','45','68','90','113','135','158'],
                       rotation = 'horizontal')
    hm.set_xlabel('Stimulus orientation (deg)')
    hm.set_yticks(np.arange(0,len(yticklabels))+0.5)
    hm.set_yticklabels(yticklabels)
    hm.set_ylabel('Cell type')
    plt.tight_layout()


    # For cells of peak selectivity, show change in COV
    df_sigma = pd.DataFrame(sigma_dict)
    df_sigma = df_sigma[df_sigma.stim != 'Blank']
    df_sigma = df_sigma.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df_sigma = df_sigma.groupby(['subject','trained','pref','stim']).mean().reset_index()
    df_diff = pd.DataFrame.copy(df_sigma[df_sigma.trained])
    df_diff['CoV'] = df_diff.CoV.to_numpy() - df_sigma[df_sigma.trained==False].CoV.to_numpy()

    df_diff = df_diff.drop(['trained', 'sigma'],axis=1)

    # df_diff = df_diff[df_diff.r == 4]
    # df_diff = df_diff[df_diff.stim == df_diff.pref]

    # pref_colors = sns.hls_palette(len(np.unique(df_diff.pref)))

    # plt.figure(figsize=(2.75,2))
    # sp = sns.stripplot(data = df_diff, x = 'pref', y = 'CoV', palette = pref_colors,
    #               hue = 'pref', s = 3)
    # sp.legend_.set_visible(False)
    # sns.pointplot(data = df_diff, x = 'pref', y = 'CoV', join = False,
    #                    errorbar = 'sem', markers = '_', capsize = 0.15,
    #                    linestyles = '--', palette = pref_colors)
    # sns.despine()
    # plt.tight_layout()
    # plt.ylabel(r'$\Delta$CoV')
    # plt.xlabel('Preferred mean orientation (deg)')
    # sp.set_xticklabels(['0','23','45','68','90','113','135','158'])

#%% PLot sigma by subject

subject_label = np.concatenate([np.repeat(s,sigma[i].size) for i,s in enumerate(subjects)])
trained_label = np.concatenate([np.repeat(t,sigma[i].size) for i,t in enumerate(trained)])
sigma_all = np.concatenate(sigma,axis=0)
cov_all = np.concatenate(CoV,axis=0)
stim_label = np.tile(np.array([0,23,45,68,90,113,135,158,'Blank'],dtype=object)
                     .reshape(1,-1),(sigma_all.shape[0],1))

r_label = np.repeat(np.concatenate(r_expts,axis=0)[:,None],len(np.unique(stim_ori_expts[0])),axis=1)
pref_label = np.repeat(np.concatenate(pref_expts,axis=0)[:,None],len(np.unique(stim_ori_expts[0])),axis=1)




subs = np.unique(df_sigma.subject)

for s in subs:

    fig, ax = plt.subplots(1,2, figsize=(14,5))

    sigma_dict = {'subject' : subject_label,
              'trained' : trained_label,
              'stim' : stim_label.flatten(),
              'sigma' : sigma_all.flatten(),
              'CoV' : cov_all.flatten(),
              'pref' : pref_label.flatten(),
              'r' : r_label.flatten()}

    df_sigma = pd.DataFrame(sigma_dict)

    df_sigma = df_sigma.groupby(['subject','trained','stim','pref','r']).mean().reset_index()

    ind_naive = np.logical_and(df_sigma.subject == s, df_sigma.trained== False)
    ind_trained = np.logical_and(df_sigma.subject == s, df_sigma.trained)

    df_naive = df_sigma[ind_naive].drop(['CoV','trained','subject'],axis=1)
    df_trained = df_sigma[ind_trained].drop(['CoV','trained','subject'],axis=1)

    df_naive = df_naive.pivot_table(index=['pref','r'], columns = 'stim')
    df_trained = df_trained.pivot_table(index=['pref', 'r'], columns = 'stim')

    df_diff = df_trained - df_naive


    sns.heatmap(df_diff, center = 0, vmin = -0.25, vmax = 0.15, ax = ax[0])
    ax[0].set_title(r'$\Delta$sigma' + ' subject ' + s)

    sigma_dict = {'subject' : subject_label,
              'trained' : trained_label,
              'stim' : stim_label.flatten(),
              'sigma' : sigma_all.flatten(),
              'CoV' : cov_all.flatten(),
              'pref' : pref_label.flatten(),
              'r' : r_label.flatten()}

    df_sigma = pd.DataFrame(sigma_dict)

    df_sigma = df_sigma.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    df_sigma = df_sigma.groupby(['subject','trained','stim','pref','r']).mean().reset_index()

    ind_naive = np.logical_and(df_sigma.subject == s, df_sigma.trained==False)
    ind_trained = np.logical_and(df_sigma.subject == s, df_sigma.trained)

    df_naive = df_sigma[ind_naive].drop(['sigma','trained','subject'],axis=1)
    df_trained = df_sigma[ind_trained].drop(['sigma','trained','subject'],axis=1)

    df_naive = df_naive.pivot_table(index=['pref','r'], columns = 'stim')
    df_trained = df_trained.pivot_table(index=['pref', 'r'], columns = 'stim')

    df_diff = df_trained - df_naive

    sns.heatmap(df_diff, center = 0, ax = ax[1], vmin = -0.175, vmax = 0.35)
    ax[1].set_title(r'$\Delta$CoV' + ' subject ' + s)

#%% Noise correlation, but averaging by cell class (selectivity and preference)

# Average residuals for each cell class, then calculate correlation matrix

n_classes = len(np.unique(df_scm.pref_bin)) * len(np.unique(df_scm.r_bin))

noise_corrs_cl = np.zeros((len(subjects),n_classes, n_classes,
                           len(np.unique(stim_ori_expts[0]))))

sigma_cl = np.zeros((len(subjects),n_classes,len(np.unique(stim_ori_expts[0]))))
cov_cl = np.zeros((len(subjects),n_classes,len(np.unique(stim_ori_expts[0]))))

for i in range(len(subjects)):

    # Average noise correlations by cell class

    nc_classes = np.zeros((noise_expts[i].shape[0], n_classes))
    mu_classes = np.zeros((len(np.unique(stim_ori_expts[0])), n_classes))

    cell_ind = np.logical_and(df_scm.subject == subjects[i],
                              df_scm.trained == trained[i])
    r = df_scm.r_bin[cell_ind]
    pref= df_scm.pref_bin[cell_ind]
    c = 0
    for p in np.unique(df_scm['pref_bin']):
        for rb in np.unique(df_scm['r_bin']):
            nc_classes[:,c] = noise_expts[i][:,np.logical_and(r==rb,pref == p)].mean(1)
            mu_classes[:,c] = mu_expts[i][np.logical_and(r==rb,pref==p),:].mean(0)
            c += 1

    for si,s in enumerate(np.unique(stim_ori_expts[i])):
       stim_ind = stim_ori_expts[i] == s
       print('Noise correlations for stimulus ' + str(s))

       # noise_corrs[i][...,si] = np.corrcoef(noise_expts[i][stim_ind,:].T)
       # Already subtracted mean, so you just do x.y/norm(x)*norm(y)
       noise_corrs_cl[i,...,si] = cosine_similarity(nc_classes[stim_ind,:].T)
       sigma_cl[i,:,si] = np.sqrt((nc_classes[stim_ind,:]**2).mean(0))
       cov_cl[i,:,si] = sigma_cl[i,:,si]/mu_classes[si,:]

noise_corrs_cl_diff = np.array([noise_corrs_cl[i,...] - noise_corrs_cl[i-1,...]
                                for i in range(1,10,2)])
sigma_cl_diff = np.array([sigma_cl[i,...] - sigma_cl[i-1,...]
                                for i in range(1,10,2)])
cov_cl_diff = np.array([cov_cl[i,...] - cov_cl[i-1,...]
                                for i in range(1,10,2)])

cell_classes = []
for p in np.unique(df_scm['pref_bin']):
        for rb in np.unique(df_scm['r_bin']):
            cell_classes.append(str(int(p)) + '_' + str(rb))

#%% Plot noise correlations for averaged cell classes

mu_noise_corrs = np.squeeze(noise_corrs_cl_diff.mean(0))

fig, ax = plt.subplots(2,5)

for s in range(mu_noise_corrs.shape[2]):

    im = sns.heatmap(mu_noise_corrs[...,s], cmap = 'RdBu_r',
                   center = 0, ax = ax.flatten()[s], cbar = False,
                   vmin = -0.6, vmax = 0.4,
                   square = True)
    ax.flatten()[s].set_xticks(np.arange(0,40,2))
    plt.xticks(rotation='vertical')
    ax.flatten()[s].set_xticklabels(cell_classes[::2], fontsize=6)
    # ax.flatten()[s].axis('equal')
    ax.flatten()[s].set_yticks(np.arange(0,40,2))
    # plt.xticks(rotation='vertical')
    ax.flatten()[s].set_yticklabels(cell_classes[::2], fontsize=6)
    plt.yticks(rotation='horizontal')

    ax.flatten()[s].set_title([0,23,45,68,90,113,135,158,'Blank'][s])
    # plt.colorbar(im, ax = ax)

fig.delaxes(ax[1][4])

# Plot individual mice
vmin = np.percentile(noise_corrs_cl_diff.flatten(),1)
vmax = np.percentile(noise_corrs_cl_diff.flatten(),99)

for sub in range(int(len(subjects)/2)):

    fig, ax = plt.subplots(2,5, figsize = (12,6))

    for s in range(noise_corrs_cl_diff.shape[3]):

        im = sns.heatmap(noise_corrs_cl_diff[sub,...,s], cmap = 'RdBu_r',
                       center = 0, ax = ax.flatten()[s], cbar = False,
                       vmin = vmin, vmax = vmax,
                       square = True)
        ax.flatten()[s].set_xticks(np.arange(0,40,2))
        plt.xticks(rotation='vertical')
        ax.flatten()[s].set_xticklabels(cell_classes[::2], fontsize=6)
        # ax.flatten()[s].axis('equal')
        ax.flatten()[s].set_yticks(np.arange(0,40,2))
        # plt.xticks(rotation='vertical')
        ax.flatten()[s].set_yticklabels(cell_classes[::2], fontsize=6)
        plt.yticks(rotation='horizontal')

        ax.flatten()[s].set_title([0,23,45,68,90,113,135,158,'Blank'][s])
        # plt.colorbar(im, ax = ax)

    fig.delaxes(ax[1][4])
    fig.tight_layout()

#%% As a sanity check, look at correlation of cell class for individual mouse, condition, and stim

expt = 9
stim = 45

stim_ind = stim_ori_expts[expt] == stim

nc_classes = np.zeros((stim_ind.sum(), n_classes))

cell_ind = np.logical_and(df_scm.subject == subjects[expt],
                          df_scm.trained == trained[expt])
r = df_scm.r_bin[cell_ind]
pref= df_scm.pref_bin[cell_ind]

c = 0
for p in np.unique(df_scm['pref_bin']):
    for rb in np.unique(df_scm['r_bin']):
        ind = np.ix_(stim_ind,np.logical_and(r==rb,pref == p))
        nc_classes[:,c] = noise_expts[expt][ind].mean(1)
        c += 1




# fig, ax = plt.subplots()
# ax.scatter(nc_classes[:,34],nc_classes[:,9])
# # ax.set_aspect(1./ax.get_data_ratio())
# plt.axis('square')

fig, ax = plt.subplots()
ax.scatter(nc_classes[:,19],nc_classes[:,17])
# ax.set_aspect(1./ax.get_data_ratio())
plt.axis('square')

#%% Plot stdev and cov for averaged cell classes


mu_diff_sigma = np.squeeze(sigma_cl_diff.mean(0))
mu_diff_cov = np.squeeze(cov_cl_diff.mean(0))

fig, ax = plt.subplots(1,2)

s_im = sns.heatmap(mu_diff_sigma, center = 0, ax = ax[0])
ax.flatten()[0].set_xticklabels([0,23,45,68,90,113,135,158,'Blank'])
# plt.xticks(rotation='vertical')
ax.flatten()[0].set_yticks(np.arange(0,40,2))
ax.flatten()[0].set_yticklabels(cell_classes[::2], fontsize=6)
# plt.yticks(rotation='horizontal')


cov_im = sns.heatmap(mu_diff_cov, center = 0, ax = ax[1])
ax.flatten()[1].set_xticklabels([0,23,45,68,90,113,135,158,'Blank'])
# plt.xticks(rotation='vertical')
ax.flatten()[1].set_yticks(np.arange(0,40,2))
ax.flatten()[1].set_yticklabels(cell_classes[::2], fontsize=6)
# plt.yticks(rotation='horizontal')


# Plot all subjects

vmin_s = np.percentile(sigma_cl_diff.flatten(),1)
vmax_s = np.percentile(sigma_cl_diff.flatten(),99)

vmin_cov = np.percentile(cov_cl_diff.flatten(),1)
vmax_cov = np.percentile(cov_cl_diff.flatten(),99)

for s in range(int(len(subjects)/2)):

    fig, ax = plt.subplots(1,2, figsize = (12,5))

    s_im = sns.heatmap(sigma_cl_diff[s,...], center = 0, ax = ax[0],
                       vmin = vmin_s, vmax = vmax_s)
    ax.flatten()[0].set_xticklabels([0,23,45,68,90,113,135,158,'Blank'])
    # plt.xticks(rotation='vertical')
    ax.flatten()[0].set_yticks(np.arange(0,40,2))
    ax.flatten()[0].set_yticklabels(cell_classes[::2], fontsize=6)
    # plt.yticks(rotation='horizontal')


    cov_im = sns.heatmap(cov_cl_diff[s,...], center = 0, ax = ax[1],
                         vmin = vmin_cov, vmax = vmax_cov)
    ax.flatten()[1].set_xticklabels([0,23,45,68,90,113,135,158,'Blank'])
    # plt.xticks(rotation='vertical')
    ax.flatten()[1].set_yticks(np.arange(0,40,2))
    ax.flatten()[1].set_yticklabels(cell_classes[::2], fontsize=6)
    # plt.yticks(rotation='horizontal')

#%% Instead of averaging residuals, average pairwise correlations for classes

from itertools import combinations, product

def pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            #Don't output pairs containing duplicated elements
            if pair[0] != pair[1]:
                yield pair

class_nc = np.zeros((len(subjects),40,40,9))

for i,(s,t) in enumerate(zip(subjects,trained)):
    cell_ind = np.logical_and(df_scm.subject == s,
                              df_scm.trained == t)

    r_bin = df_scm.r_bin[cell_ind].to_numpy()
    pref_ori = df_scm.pref_bin[cell_ind].to_numpy()

    uni_r = np.unique(r_bin)
    uni_pref = np.unique(pref_ori)

    class_ind = []
    # Generate indices by each class
    for p in uni_pref:
        for r in uni_r:
            class_ind.append(np.where(np.logical_and(pref_ori==p,
                                            r_bin == r))[0])

    for s in range(9):
        for ic0,c0 in enumerate(class_ind):
            for ic1,c1 in enumerate(class_ind):
                print('Expt ' + str(i) + ' stim ' + str(s) +
                      ' cell classes ' + str(ic0) + ' and ' + str(ic1))

                # Create indices for all unique pairs (exclude same cell)
                inds = pairs(c0,c1)

                nc_pairs = np.array([noise_corrs[i][ind[0],ind[1],s]

                                           for ind in inds])
                class_nc[i,ic0,ic1,s] = nc_pairs.mean()

class_nc_diff = np.array([class_nc[i,...] - class_nc[i-1,...]
                                for i in range(1,10,2)])


#%%

cell_classes = []
for p in np.unique(df_scm['pref_bin']):
        for rb in np.unique(df_scm['r_bin']):
            cell_classes.append(str(int(p)) + '_' + str(rb))

mu_noise_corrs = np.squeeze(class_nc_diff.mean(0))

fig, ax = plt.subplots(2,5)

for s in range(mu_noise_corrs.shape[2]):

    im = sns.heatmap(mu_noise_corrs[...,s], cmap = 'RdBu_r',
                   center = 0, ax = ax.flatten()[s], cbar = False,
                    vmin = -0.25, vmax = 0.03,
                   square = True)
    ax.flatten()[s].set_xticks(np.arange(0,40,2))
    plt.xticks(rotation='vertical')
    ax.flatten()[s].set_xticklabels(cell_classes[::2], fontsize=6)
    # ax.flatten()[s].axis('equal')
    ax.flatten()[s].set_yticks(np.arange(0,40,2))
    # plt.xticks(rotation='vertical')
    ax.flatten()[s].set_yticklabels(cell_classes[::2], fontsize=6)
    plt.yticks(rotation='horizontal')

    ax.flatten()[s].set_title([0,23,45,68,90,113,135,158,'Blank'][s])
    # plt.colorbar(im, ax = ax)

fig.delaxes(ax[1][4])

# Plot for individual mice

vmin = np.percentile(class_nc_diff.flatten(),1)
vmax = np.percentile(class_nc_diff.flatten(),99)

for sub in range(int(len(subjects)/2)):

    fig, ax = plt.subplots(2,5, figsize = (12,6))

    for s in range(class_nc_diff.shape[3]):

        im = sns.heatmap(class_nc_diff[sub,...,s], cmap = 'RdBu_r',
                       center = 0, ax = ax.flatten()[s], cbar = False,
                        vmin = vmin, vmax = vmax,
                       square = True)
        ax.flatten()[s].set_xticks(np.arange(0,40,2))
        plt.xticks(rotation='vertical')
        ax.flatten()[s].set_xticklabels(cell_classes[::2], fontsize=6)
        # ax.flatten()[s].axis('equal')
        ax.flatten()[s].set_yticks(np.arange(0,40,2))
        # plt.xticks(rotation='vertical')
        ax.flatten()[s].set_yticklabels(cell_classes[::2], fontsize=6)
        plt.yticks(rotation='horizontal')

        ax.flatten()[s].set_title([0,23,45,68,90,113,135,158,'Blank'][s])
        # plt.colorbar(im, ax = ax)

    fig.delaxes(ax[1][4])
    fig.tight_layout()