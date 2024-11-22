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
                    'stim_decoding_results_45 and 90 only_all cells*'))[-1]
                    for i in range(len(subjects))]


fig_save = r'C:\Users\Samuel/OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%%

df_predictions = pd.DataFrame()
df_weights = pd.DataFrame()
df_ave_resps = pd.DataFrame()
df_trials = pd.DataFrame()

for i in range(len(subjects)):

    print('Loading ' + subjects[i])
    decode_all = np.load(file_paths_all[i],allow_pickle = True)[()]

    n_cells = len(decode_all['OSI'])

    stim = decode_all['stim']
    trials_z = decode_all['trials_z_test']
    trials = decode_all['trials_test']

    ave_45_z = trials_z[stim==45,...].mean(0)
    ave_90_z = trials_z[stim==90,...].mean(0)

    ave_45 = trials[stim==45,...].mean(0)
    ave_90 = trials[stim==90,...].mean(0)

    # Recalculate d-prime by average mean and std across directions with same orientation
    stim_dir_train = decode_all['stim_dir']

    mu = np.array([trials[stim_dir_train==s,:].mean(0)
                         for s in np.unique(stim_dir_train)])
    # mu_diff = np.diff(mu,axis=0).flatten()
    stim_ori = np.unique(stim_dir_train) % 180
    mu = np.array([mu[stim_ori==s,...].mean(0) for s in np.unique(stim_ori)])

    mu_max = np.argmax(mu,axis=0)
    mu_min = np.argmin(mu,axis=0)

    mu_diff = np.diff(mu,axis = 0)

    var = np.array([trials[stim_dir_train==s,:].var(0)
                         for s in np.unique(stim_dir_train)])

    var = np.array([var[stim_ori==s,...].mean(0) for s in np.unique(stim_ori)])

    sigma = np.sqrt(var.sum(axis=0)/2)

    # sigma = np.sqrt(np.array([x_test[stim_ori_test==s,:].var(0)
    #                      for s in np.unique(stim_ori_test)]).mean(0))

    d_prime = mu_diff/sigma

    # cv = np.sqrt(var)/mu
    # cv = cv.mean(0)
    mu_pref = mu[mu_max,np.arange(len(mu_max))]
    sigma_pref = np.sqrt(var[mu_max,np.arange(len(mu_max))])
    cv_pref = sigma_pref/mu_pref
    # cv = cv.mean(0)

    mu_nonpref = mu[mu_min,np.arange(len(mu_min))]
    sigma_nonpref = np.sqrt(var[mu_min,np.arange(len(mu_min))])
    cv_nonpref = sigma_nonpref/mu_nonpref


    prop_45_resp = ave_45_z*decode_all['model_weights'].flatten()
    prop_45_resp /= prop_45_resp.sum()
    prop_90_resp = ave_90_z*decode_all['model_weights'].flatten()
    prop_90_resp /= prop_90_resp.sum()

    ave_prop_resp = (prop_45_resp + prop_90_resp)/2

    cont_45_resp = ave_45_z*decode_all['model_weights'].flatten()*-1
    rel_cont_45_resp = cont_45_resp/cont_45_resp.max()

    cont_90_resp = ave_90_z*decode_all['model_weights'].flatten()
    rel_cont_90_resp = cont_90_resp/cont_90_resp.max()

    ave_cont_resp = (cont_45_resp + cont_90_resp)/2
    ave_rel_cont_resp = (rel_cont_45_resp + rel_cont_90_resp)/2

    resp_index = (ave_45 - ave_90) / (ave_45 + ave_90)

    ave_resps_train = np.array([decode_all['trials_train'][decode_all['stim_train']==s,:].mean(0)
                       for s in [45,90]])
    ave_resps_test = np.array([decode_all['trials_test'][decode_all['stim']==s,:].mean(0)
                       for s in [45,90]])

    s_ind = np.argsort(ave_cont_resp)[::-1]
    # s_ind = np.arange(len(cv))

    cell_index = np.arange(len(resp_index))

    df_weights = pd.concat([
        df_weights,pd.DataFrame({'subject' : np.repeat(subject_unique[i],n_cells),
                                 'trained' : np.repeat(trained[i],n_cells),
                                 'selectivity' : decode_all['OSI'][s_ind],
                                 'ori_pref' : decode_all['mean_pref_ori'][s_ind],
                                 'model_weight' : decode_all['model_weights'].flatten()[s_ind],
                                 'abs_weight' : np.abs(decode_all['model_weights'].flatten()[s_ind]),
                                 'num_cells' : np.repeat(n_cells,n_cells),
                                 'cv_pref' : cv_pref.flatten()[s_ind],
                                 'sigma_pref' : sigma_pref.flatten()[s_ind],
                                 'mu_pref' : mu_pref.flatten()[s_ind],
                                 'cv_nonpref' : cv_nonpref.flatten()[s_ind],
                                 'sigma_nonpref' : sigma_nonpref.flatten()[s_ind],
                                 'mu_nonpref' : mu_nonpref.flatten()[s_ind],
                                 'd_prime' : d_prime.flatten()[s_ind],
                                 'ave_45_resp' : ave_45[s_ind],
                                 'ave_90_resp' : ave_90[s_ind],
                                 'cont_45_resp' : cont_45_resp[s_ind],
                                 'cont_90_resp' : cont_90_resp[s_ind],
                                 'rel_cont_45_resp' : rel_cont_45_resp[s_ind],
                                 'rel_cont_90_resp' : rel_cont_90_resp[s_ind],
                                 'ave_cont_resp' : ave_cont_resp[s_ind],
                                 'ave_rel_cont_resp' : ave_rel_cont_resp[s_ind],
                                 'prop_resp_45' : prop_45_resp[s_ind],
                                 'prop_resp_90' : prop_90_resp[s_ind],
                                 'ave_prop_resp' : ave_prop_resp[s_ind],
                                 'resp_index' : resp_index[s_ind],
                                 'v_x' : decode_all['v_x'][s_ind],
                                 'v_y' : decode_all['v_y'][s_ind],
                                 'cell_index' : cell_index[s_ind]
                                 })])

    
    df_trials = pd.concat([df_trials, pd.DataFrame({'subject' : np.repeat(subject_unique[i], trials.size),
                                                    'trained' : np.repeat(trained[i], trials.size),
                                                    'cell_num' : np.tile(np.arange(trials.shape[1])[None,:], (trials.shape[0], 1)).ravel(),
                                                    'stim' : np.tile(stim[:,None], (1, trials.shape[1])).ravel(),
                                                    'resps' : trials.ravel()})])
    

    df_ave_resps = pd.concat([df_ave_resps,pd.DataFrame({'subject' : np.repeat(subject_unique[i], n_cells),
                                                         'trained' : np.repeat(trained[i], n_cells),
                                                         'ave_resp_45_train' : ave_resps_train[0,:].flatten()[s_ind],
                                                         'ave_resp_90_train' : ave_resps_train[1,:].flatten()[s_ind],
                                                         'ave_resp_45_test' : ave_resps_test[0,:].flatten()[s_ind],
                                                         'ave_resp_90_test' : ave_resps_test[1,:].flatten()[s_ind],
                                                         })])


    df_predictions = pd.concat([df_predictions, pd.DataFrame({'subject': np.repeat(subject_unique[i], len(stim)),
                                                              'trained': np.repeat(trained[i], len(stim)),
                                                              'stim': stim,
                                                              'pred_stim': decode_all['pred_stim']})])


df_weights['pref_bin'] = pd.cut(df_weights.ori_pref,np.linspace(-11.25,180-11.25,9),
                                labels = np.ceil(np.arange(0,180,22.5))).astype('category')
df_weights['r_bin'] = pd.cut(df_weights.selectivity,np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]),
                     labels = np.arange(5)).astype('category')

df_weights = df_weights.reset_index(drop=True)
df_ave_resps = df_ave_resps.reset_index(drop=True)

# df_weights['abs_weight'] = np.abs(df_weights.model_weight)

# df_weights = df_weights.sort_values(['subject','trained','abs_weight'], ascending = False)

df_weights.loc[df_weights.ori_pref < 0,'ori_pref'] = df_weights[df_weights.ori_pref < 0].ori_pref + 180

#%%

def pair_points(ax):

    # p0 = ax.get_children()[0].get_offsets().data
    # p1 = ax.get_children()[1].get_offsets().data

    # breakpoint()
    p = [i.get_offsets().data for i in ax.get_children() if type(i) is type(ax.get_children()[0])]

    points = []
    i = 0
    for i in range(len(p)):
        if len(p[i]) > 0:
            print(len(p[i]))
            points.append(p[i])

        if len(points) == 2:
            break

    # breakpoint()

    for i in range(len(points[0])):
        ax.plot([points[0][i,0], points[1][i,0]],[points[0][i,1], points[1][i,1]],'-k')


def bull_plot_mdl(cell_x,cell_y,s0,s1,mdl_cont,s_mult=2,s_0 = 0.1,label = None,
                  rasterized = True,edgecolor = 'black',linewidth = 0.025,
                  vmin = 0,vmax = 1):
    ''' bulls-eye viewer for cell activity'''

    # breakpoint()
    s = np.max(np.array([s0,s1]),0)
    o = np.argsort(mdl_cont)
    sz = np.abs(s[o])*s_mult + s_0

    # breakpoint()
    sns.scatterplot(x = cell_x[o],y = cell_y[o], s=sz, hue = mdl_cont[o],
                    label = label, rasterized = rasterized, edgecolor = edgecolor,
                    linewidth = linewidth, legend = False,  hue_norm = (vmin,vmax),
                    palette = 'magma_r')
    plt.gca().set_facecolor('w')
    plt.axis([-1,1,-1,1])
    plt.rc('path',simplify=False)



#%% Plot performance for decoding 45 vs 90 by training condition




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

        f,a = plt.subplots(1,1, figsize=(1,1))

        df_plot = df_predictions.copy()
        df_plot['pred_correct'] = df_plot['stim'] == df_plot['pred_stim']

        df_plot = df_plot.groupby(['subject','trained'])['pred_correct'].mean().reset_index()
        df_plot['pred_correct'] *= 100

        (
            so.Plot(df_plot, x='trained', y='pred_correct', color='trained')
            .add(so.Dots(pointsize=3), so.Jitter(y=0, seed=0), legend=False)
            .limit(y=(50,102))
            .scale(y=so.Continuous().tick(at=[50,75,100]),
                color='colorblind')
            .label(x='', y='Classification accuracy (% correct)')
            .on(a)
            .plot()
        )   

        a.plot(a.get_xlim(), [100, 100], '--k')

        sns.despine(ax=a, trim=True)
        
        f.savefig(join(fig_save,'accuracy_45_vs_90_all_cells.svg'), format='svg')

#%% Find top cell contributing to models in each naive and trained mouse, plot distribution of respones to 45 and 90

df_best_cells = df_weights.sort_values(['subject','trained','ave_rel_cont_resp'], ascending=False).groupby(['subject','trained']).first().reset_index()
df_best_cells = df_best_cells.rename({'cell_index': 'cell_num'}, axis=1)

df_best_cells_trials = df_best_cells.merge(df_trials, on=['subject','trained', 'cell_num'], how='inner')

df_best_cells_trials['group'] = df_best_cells_trials['subject'] + '_' + df_best_cells_trials['trained'].astype(str)

df_best_cells_trials = df_best_cells_trials.sort_values(['trained','ori_pref'])

df = df_best_cells_trials
df['trained'] = df['trained'].astype(str)

df['color_group'] = df['trained'] + df['stim'].astype(str)

cell_labels = np.tile(['Cell ' + str(n+1) for n in range(5)], 2)

# colors_base = sns.color_palette('colorblind')[:2]
# colors_base = [np.array(c) for c in colors_base]

# colors_naive = [c*0.8 for c in colors_base]
# colors_prof = [c*1.2 for c in colors_base]

# colors_prof = [np.clip(c,0,1) for c in colors_prof]

# colors = colors_naive+colors_prof

# colors = [colors[i] for i in [0,2,1,3]]

colors = [(0.8,0,0.8),(0,0.8,0)]
colors = colors+colors

colors_dark = [(0.6,0,0.6),(0,0.6,0)]
colors_dark = colors_dark+colors_dark

# colors = [list(c) for c in colors]

# cell_labels = [c + f' {t}' for c,t in zip(cell_labels, np.repeat([' ','  '],5))]
# group_mapping = {g: c for g,c in zip(df['group'].unique(), cell_labels)}

# df['group'] = df['group'].apply(lambda x: group_mapping[x])

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

    f,a = plt.subplots(1,1, figsize=(2,1))

    dodge_gap = 0.1
    
    color = 'color_group'

    p1 = (
        so.Plot(df, x='group', y='resps')
        .add(so.Dots(pointsize=2, stroke=0.25),
            #  so.Dodge(by=['stim']),
             so.Jitter(x=0.1, y=0),
             group='stim', color=color, legend=False)
        # .add(so.Dash(color='black', linewidth=0.5, width=0.5), so.Agg(),
        #     #  so.Dodge(by=['stim']), 
        #     #  group='stim', color=color,
        #      legend=False)
        .scale(color=so.Nominal(values=colors,
                                order=['False45.0','False90.0','True45.0','True90.0']),
               )
        .limit(y=(-0.1,1.5))
        .label(x='', y='Trial response')
        .on(a)
        .plot()
    )

    p2 = (
        so.Plot(df, x='group', y='resps')
        .add(so.Range(linewidth=0.5),
             so.Dodge(by=['stim']),
             group='stim', color=color,
             legend=False,
             zorder=-10)
        .scale(color=so.Nominal(values=colors_dark,
                                order=['False45.0','False90.0','True45.0','True90.0']),
               )
        .limit(y=(-0.1,1.5))
        .label(x='', y='Trial response')
        .on(a)
        .plot()
    )

   

    a.set_xticklabels(cell_labels, rotation=30)
    sns.despine(ax=a, trim=True)

    f.show()


    # sns.stripplot(df, x='group', y='resps', hue='stim', dodge=False,
    #               palette=colors, ax=a, s=1.5, edgecolor='black', linewidth=0.25,
    #               legend=False, jitter=False)
    # sns.violinplot(df, x='group', y='resps', hue='stim', dodge=False,
    #                palette=colors, ax=a, legend=False, split=True)
    # a.set_xticklabels(cell_labels, rotation=30)

    f.savefig(join(fig_save, 'trial_resps_cells_with_high_contribution_to_LDA.svg'), format='svg')


#%% Bullseye plot for model contribution - Selectivity and mean response

d_prime = df_weights['d_prime'].to_numpy()

ind = ~np.isnan(d_prime)

d_prime = d_prime[ind]

trained_cp = sns.color_palette('colorblind')[0:2]

trained_cond = df_weights['trained'].to_numpy()[ind]

cell_x = df_weights['v_x'].to_numpy()[ind]
cell_y = df_weights['v_y'].to_numpy()[ind]

ave_resp_45 = df_weights['ave_45_resp'].to_numpy()[ind]
ave_resp_90 = df_weights['ave_90_resp'].to_numpy()[ind]

mdl_cont = df_weights['ave_rel_cont_resp'].to_numpy()[ind]

trained_cells = np.where(trained_cond==True)[0]
naive_cells = np.where(trained_cond==False)[0]

trained_blocks = np.random.choice(trained_cells, 16000, replace = False)
naive_blocks = np.random.choice(naive_cells, 16000, replace = False)

vmin = 0
vmax = 1
s_mult = 2

smin = 0+0.1
smax = vmax*s_mult+0.1

n_pts = 1000
x_c, y_c = np.meshgrid(np.linspace(0,1,n_pts),np.linspace(0,1,n_pts))

x_y = np.concatenate((x_c.reshape(1,-1),y_c.reshape(1,-1)),axis=0)
x_y = np.concatenate((np.ones((1,x_y.shape[1])), x_y), axis = 0)


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1,
                          rc = {'lines.linewidth':0.5}):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['figure.dpi'] = 1000

    fontsize = 5

    f = plt.figure(figsize=(3.75,3))
    gs = gridspec.GridSpec(1,2, figure = f)
    gs.update(wspace=-0.03)

    ax_naive0 = f.add_subplot(gs[0],aspect = 'equal')
    ax_naive0.set_facecolor('None')

    c = naive_blocks
    bull_plot_mdl(cell_x[c], cell_y[c], ave_resp_45[c], ave_resp_90[c],
                  mdl_cont[c], s_mult = s_mult, s_0 = smin,
                  vmin = vmin, vmax = vmax)
    # plt.grid()

    ax_naive0.set_xlim([-1,1])
    ax_naive0.set_ylim([-1,1])

    th=np.arange(0,1.0001,.0001)*2*np.pi

    # draw circles
    for rd in [0.16, 0.32, 0.48, 0.64, 0.8]:
        ax_naive0.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)



    plt.title('Naïve', color = trained_cp[0], fontsize = 6, pad = 1)
    plt.tick_params(axis='both', left=False, top=False, right=False,
                        bottom=False, labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)

    ax_prof0 = f.add_subplot(gs[1],aspect = 'equal')
    ax_prof0.set_facecolor('None')
    c = trained_blocks
    bull_plot_mdl(cell_x[c], cell_y[c], ave_resp_45[c], ave_resp_90[c],
                  mdl_cont[c], s_mult = s_mult, s_0 = smin,
                  vmin = vmin, vmax = vmax)

    ax_prof0.set_xlim([-1,1])
    ax_prof0.set_ylim([-1,1])

    # draw circles

    for rd in [0.16, 0.32, 0.48, 0.64, 0.8]:
        plt.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)


    ang = np.deg2rad(315)
    ax_naive0.text(np.cos(ang)*0.95, np.sin(ang)*0.95, 'Selectivity',rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')

    for r in [0.16, 0.32, 0.48, 0.64, 0.8]:
        ax_naive0.text(np.cos(ang)*(r+0.01), np.sin(ang)*(r+0.01), r, rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')

    h_dist = 0.85
    v_dist = 0.85

    for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

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


        ax_naive0.text(np.cos(a)*d, np.sin(a)*d, label,
                  horizontalalignment=ha, verticalalignment = va,
                  fontsize = fontsize)


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


#%% Bullseye plot for model contribution - CV and d'

d_prime = df_weights['d_prime'].to_numpy()

ind = ~np.isnan(d_prime)

d_prime = np.abs(d_prime[ind])

trained_cp = sns.color_palette('colorblind')[0:2]

# trained_cond = df_weights['trained'].to_numpy()[ind]

cv = df_weights['cv'][ind].to_numpy()

# flip the cv distribution
cv_min = 0
cv_max = cv.max()
cv_flip = cv_max-cv

lines = np.linspace(cv_min,np.round(cv_max),6)[1:].astype(int)
line_labels = np.linspace(np.round(cv_max),cv_min,6)[1:].astype(int)

# cell_x = np.cos(np.deg2rad(df_weights['ori_pref'][ind].to_numpy())*2) * df_weights['cv'][ind].to_numpy()
# cell_y = np.sin(np.deg2rad(df_weights['ori_pref'][ind].to_numpy())*2) * df_weights['cv'][ind].to_numpy()

cell_x = np.cos(np.deg2rad(df_weights['ori_pref'][ind].to_numpy())*2) * cv_flip
cell_y = np.sin(np.deg2rad(df_weights['ori_pref'][ind].to_numpy())*2) * cv_flip

mdl_cont = df_weights['ave_rel_cont_resp'].to_numpy()

# trained_cells = np.where(trained_cond==True)[0]
# naive_cells = np.where(trained_cond==False)[0]

# trained_blocks = np.random.choice(trained_cells, 16000, replace = False)
# naive_blocks = np.random.choice(naive_cells, 16000, replace = False)

vmin = 0
vmax = 1
s_mult = 0.2

smin = 0+0.1
smax = vmax*s_mult+0.1


sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1,
                          rc = {'lines.linewidth':0.5}):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['figure.dpi'] = 1000

    fontsize = 5

    f = plt.figure(figsize=(3.75,3))
    gs = gridspec.GridSpec(1,2, figure = f)
    gs.update(wspace=-0.03)

    ax_naive0 = f.add_subplot(gs[0],aspect = 'equal')
    ax_naive0.set_facecolor('None')

    c = naive_blocks
    bull_plot_mdl(cell_x[c], cell_y[c], d_prime[c], d_prime[c],
                  mdl_cont[c], s_mult = s_mult, s_0 = smin,
                  vmin = vmin, vmax = vmax)
    # plt.grid()

    ax_naive0.set_xlim([-6,6])
    ax_naive0.set_ylim([-6,6])

    th=np.arange(0,1.0001,.0001)*2*np.pi

    # draw circles
    for rd in lines:
        ax_naive0.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)



    plt.title('Naïve', color = trained_cp[0], fontsize = 6, pad = 1)
    plt.tick_params(axis='both', left=False, top=False, right=False,
                        bottom=False, labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)

    ax_prof0 = f.add_subplot(gs[1],aspect = 'equal')
    ax_prof0.set_facecolor('None')
    c = trained_blocks
    bull_plot_mdl(cell_x[c], cell_y[c], d_prime[c], d_prime[c],
                  mdl_cont[c], s_mult = s_mult, s_0 = smin,
                  vmin = vmin, vmax = vmax)

    ax_prof0.set_xlim([-6,6])
    ax_prof0.set_ylim([-6,6])

    # draw circles
    for rd in lines:
        plt.plot(np.cos(th)*rd, np.sin(th)*rd,color='k', linewidth = 0.5)


    ang = np.deg2rad(315)
    ax_naive0.text(np.cos(ang)*5.75, np.sin(ang)*5.75, 'CoV',rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')

    for r,l in zip(lines,line_labels):
        ax_naive0.text(np.cos(ang)*(r+0.01), np.sin(ang)*(r+0.01), l, rotation = 45,
                  horizontalalignment='center', verticalalignment = 'top',
                  fontsize = fontsize, rotation_mode = 'anchor')

    h_dist = 5.25
    v_dist = 5.25

    for i,a in enumerate(np.deg2rad(np.linspace(0,270,4))):

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


        ax_naive0.text(np.cos(a)*d, np.sin(a)*d, label,
                  horizontalalignment=ha, verticalalignment = va,
                  fontsize = fontsize)


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



#%% Get top contributing cells and look at orthgonalization

# plt.ioff()

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
                                        "lines.markersize":0.5,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    n_cells = [10,100,200,400,800,-1]
    # n_cells = [10000]


    f,ax = plt.subplots(2,3,figsize = (1.75,1.5))

    for i,n in enumerate(n_cells):

        df_resps = df_ave_resps.copy()

        # print(df_resps.groupby(['subject','trained']).apply(
        #     lambda x: len(x.ave_resp_45_train.to_numpy()[:n])))

        df_cs = pd.DataFrame()

        df_cs['train_45_dot_test_90'] = df_resps.groupby(['subject','trained']).apply(
            lambda x: cosine_similarity(x.ave_resp_45_train.to_numpy()[:n].reshape(1,-1),
                                        x.ave_resp_90_test.to_numpy()[:n].reshape(1,-1)))

        df_cs['test_45_dot_train_90'] = df_resps.groupby(['subject','trained']).apply(
            lambda x: cosine_similarity(x.ave_resp_45_test.to_numpy()[:n].reshape(1,-1),
                                        x.ave_resp_90_train.to_numpy()[:n].reshape(1,-1)))

        df_cs['cs'] = df_cs[['train_45_dot_test_90','test_45_dot_train_90']].mean(axis=1)

        df_cs = df_cs.reset_index()

        p = ttest_rel(df_cs[df_cs.trained==False].cs, df_cs[df_cs.trained==True].cs,
                      alternative = 'greater')
        print(p)


        sns.stripplot(data = df_cs, y = 'cs', hue = 'trained',
                      ax = ax.flatten()[i], palette = 'colorblind', jitter = True,
                      edgecolor = 'black', s = 2, dodge = True,
                      linewidth = 0.5)
        ax.flatten()[i].legend_.set_visible(False)
        if n != -1:
            ax.flatten()[i].set_title(str(n) + ' neurons')
        else:
            ax.flatten()[i].set_title('all neurons')
        ax.flatten()[i].set_ylim([0,0.9])
        pair_points(ax.flatten()[i])

        xlims = ax.flatten()[i].get_xlim()

        ax.flatten()[i].set_xlim([xlims[0]-0.15,xlims[1]+0.15])
        # ax.flatten()[i].set_xlim(xlims)

        # ax.flatten()[i].set_xlabel('')
        # ax.flatten()[i].set_xticklabels(['Naive','Proficient'])
        ax.flatten()[i].get_xaxis().set_visible(False)
        sns.despine(ax=ax.flatten()[i], bottom = True, trim = True)
        if np.logical_or(i == 0,i==3):
            ax.flatten()[i].set_yticks(np.linspace(0,0.9,3))
            ax.flatten()[i].set_ylabel('Cosine similarity')
            sns.despine(ax=ax.flatten()[i], bottom = True, trim = True)
        else:
            ax.flatten()[i].set_ylabel('')
            ax.flatten()[i].set_yticks([])
            sns.despine(ax=ax.flatten()[i], bottom = True, left = True,
                        trim = True)

    f.tight_layout()
    plt.savefig(join(fig_save,
                     'stim decoding_45 and 90_cosine similarity as function of number of model preferred cells.svg'),
                format = 'svg')


# plt.ion()

#%% Population sparseness

def pop_sparseness(fr, kind = 'Treves-Rolls'):

    # breakpoint()
    if np.logical_or(kind.lower() == 'treves-rolls',  kind.lower() == 'tr'):
        # Treves-Rolls
        top = (np.abs(fr)/len(fr)).sum()**2
        bottom = (fr**2/len(fr)).sum()
        s = 1 - (top/bottom)
    elif kind.lower() == 'kurtosis':
        s = kurtosis(fr, fisher = False)
    elif kind.lower() == 'active':
        sigma = fr.std(1)
        s = (fr < sigma[:,None]).sum(1)/fr.shape[1]
    return s

kind = 'tr'

n_cells = [10,100,200,400,800,-1]

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
                                        "lines.markersize":0.5,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"

    f,ax = plt.subplots(2,3,figsize = (1.75,1.5))

    for i,n in enumerate(n_cells):

        df_resps = df_ave_resps.copy()

        df_ps = pd.DataFrame()


        df_ps['ps_45'] = df_resps.groupby(['subject','trained']).apply(
            lambda x: pop_sparseness(x.ave_resp_45_test[:n].to_numpy(),kind))
        df_ps['ps_90'] = df_resps.groupby(['subject','trained']).apply(
            lambda x: pop_sparseness(x.ave_resp_90_test[:n].to_numpy(),kind))

        if n == -1:
            df_ps['n_cells'] = df_resps.groupby(['subject','trained']).apply(
                lambda x: len(x.ave_resp_45_test.to_numpy()))

        df_ps['ps'] = df_ps[['ps_45','ps_90']].mean(axis=1)

        df_ps = df_ps.reset_index()

        p = ttest_rel(df_ps[df_ps.trained==False].ps, df_ps[df_ps.trained==True].ps)
        print(p)

        sns.stripplot(data = df_ps, y = 'ps', hue = 'trained',
                      ax = ax.flatten()[i], palette = 'colorblind', jitter = True,
                      edgecolor = 'black', s = 2, dodge = True,
                      linewidth = 0.5)
        ax.flatten()[i].legend_.set_visible(False)
        if n != -1:
            ax.flatten()[i].set_title(str(n) + ' neurons')
        else:
            ax.flatten()[i].set_title('all neurons')
        ax.flatten()[i].set_ylim([0.1,0.8])
        pair_points(ax.flatten()[i])

        xlims = ax.flatten()[i].get_xlim()

        ax.flatten()[i].set_xlim([xlims[0]-0.15,xlims[1]+0.15])
        # ax.flatten()[i].set_xlim(xlims)

        # ax.flatten()[i].set_xlabel('')
        # ax.flatten()[i].set_xticklabels(['Naive','Proficient'])
        ax.flatten()[i].get_xaxis().set_visible(False)
        if np.logical_or(i == 0,i==3):
            ax.flatten()[i].set_yticks(np.linspace(0.1,0.8,3))
            ax.flatten()[i].set_ylabel('Sparseness')
            sns.despine(ax=ax.flatten()[i], bottom = True, trim = True)
        else:
            ax.flatten()[i].set_ylabel('')
            ax.flatten()[i].set_yticks([])
            sns.despine(ax=ax.flatten()[i], bottom = True, left = True,
                        trim = True)

    f.tight_layout()
    plt.savefig(join(fig_save,
                     'stim decoding_45 and 90_population sparseness as function of number of model preferred cells.svg'),
                format = 'svg')



#%% Weighted average of orientation preference

# Weighted circular average of orientation preference for neg weighted and pos weighted cells'''

# Convert ori pref to x,y and scale by model contribution

df_plot = df_weights.copy()

df_plot['neg_weight'] = df_plot['model_weight'] < 0

df_plot['ori_pref_rad'] = df_plot.ori_pref.apply(lambda x: np.deg2rad(x*2))

df_plot['weighted_x_pref'] = df_plot['ori_pref_rad'].apply(np.cos) * df_plot['ave_rel_cont_resp']
df_plot['weighted_y_pref'] = df_plot['ori_pref_rad'].apply(np.sin) * df_plot['ave_rel_cont_resp']
df_plot['weighted_pref_complex'] = df_plot.weighted_x_pref + df_plot.weighted_y_pref * 1j

df_plot = df_plot.groupby(['subject','trained','neg_weight']).apply(lambda x: x.weighted_pref_complex.sum()/x.ave_rel_cont_resp.sum()).reset_index()
df_plot = df_plot.rename(columns = {0 : 'weighted_pref_complex'})



df_plot['weighted_mean_pref'] = df_plot.weighted_pref_complex.apply(np.angle)
df_plot['weighted_mean_x'] = df_plot['weighted_pref_complex'].apply(np.real)
df_plot['weighted_mean_y'] = df_plot['weighted_pref_complex'].apply(np.imag)
df_plot['weighted_mean_pref_ori'] = df_plot.weighted_mean_pref.apply(np.rad2deg)
df_plot['weighted_var_pref'] = df_plot.weighted_pref_complex.apply(np.abs)


f,ax = plt.subplots()

sns.scatterplot(data = df_plot,
                x = df_plot['weighted_mean_x'],
                y = df_plot['weighted_mean_y'],
                hue = df_plot['neg_weight'],
                style = df_plot['trained'],
                ax = ax)

ax.set_ylim([-1,1])
ax.set_xlim([-1,1])



#%% Simplier approach, look at weighted difference of a cell's orientation preference from 45/90


df_plot = df_weights.copy()

df_plot['pref_diff'] = df_plot.ori_pref.apply(lambda x: np.min([x-45.,x-90.]))

df_plot = df_plot.groupby(['subject','trained']).apply(lambda x: (x.pref_diff * x.ave_rel_cont_resp).sum()/x.ave_rel_cont_resp.sum()).reset_index()

df_plot = df_plot.rename(columns = {0 : 'weighted_pref_diff'})

f,a = plt.subplots()

sns.stripplot(data = df_plot, x = 'trained', y = 'weighted_pref_diff',
              color = 'black', ax = a)

pair_points(a)

xlims = a.get_xlim()

a.set_xlim([xlims[0]-0.15,xlims[1]+0.15])

a.set_xticklabels(['Naive','Proficient'])

sns.despine(ax=a)

f.tight_layout()


#%% Distribution of model weights

sns.displot(data = df_weights, x = 'abs_weight', row = 'trained',
            col = 'subject', kind = 'hist', stat = 'probability',
            facet_kws = {'subplot_kws' : {'sharex' : False, 'sharey' : False}})

sns.displot(data = df_weights, x = 'abs_weight', row = 'trained',
            col = 'subject', kind = 'hist', log_scale = True,
            stat = 'probability',
            facet_kws = {'sharey' : False, 'sharex' : False})

#%%

f,ax = plt.subplots(2,5, sharex = True, sharey = True)

for i,(s,t) in enumerate(zip(subject_unique,trained)):
        ind = np.logical_and(df_weights.subject==s, df_weights.trained == t)

        sns.lineplot(data = df_weights[ind], x = 'rel_weight',
                      y = 'prop_resp_45_sum', ax = ax.flatten()[i])
        sns.lineplot(data = df_weights[ind], x = 'rel_weight',
                      y = 'prop_resp_90_sum', ax = ax.flatten()[i])
        sns.histplot(data = df_weights[ind], x = 'rel_weight',
                      ax = ax.flatten()[i],
                      stat = 'probability',
                      cumulative = True,
                      element = 'step',
                      fill = False)
        if np.logical_or(i == 0, i == 5):
            ax.flatten()[i].set_ylabel('Proportion of w*x\nProportion of cells')


        sns.despine()


#%% Weight d-prime,cv, etc. by model weight and average per experiment

# mpl.rcParams['figure.dpi'] = 100


df_wa = df_weights.copy()

df_wa = df_wa.dropna()

# df_wa['d_prime_weighted'] = np.abs(df_wa.d_prime) * df_wa.rel_weight
# df_wa['cv_weighted'] = df_wa.cv * df_wa.rel_weight
# df_wa['selectivity_weighted'] = df_wa.selectivity * df_wa.rel_weight

# Can't weight by a negative number, so set negative values to 0
df_wa['ave_rel_cont_resp'] = df_wa.ave_rel_cont_resp.map(lambda x: 0 if x < 0 else x)

df_wa['d_prime_log_weighted'] = np.log10(np.abs(df_wa.d_prime)) * df_wa.ave_rel_cont_resp
df_wa['d_prime_weighted'] = np.abs(df_wa.d_prime) * df_wa.ave_rel_cont_resp
df_wa['cv_weighted'] = df_wa.cv_pref * df_wa.ave_rel_cont_resp
df_wa['selectivity_weighted'] = df_wa.selectivity * df_wa.ave_rel_cont_resp
df_wa['resp_index_weighted'] = np.abs(df_wa.resp_index) * df_wa.ave_rel_cont_resp
df_wa['mu_weighted'] = df_wa.mu_pref * df_wa.ave_rel_cont_resp
df_wa['sigma_weighted'] = df_wa.sigma_pref * df_wa.ave_rel_cont_resp

df_wa = df_wa.groupby(['subject','trained'],as_index=False)[['d_prime_weighted','d_prime_log_weighted',
                    'cv_weighted','selectivity_weighted','ave_rel_cont_resp',
                    'resp_index_weighted','mu_weighted', 'sigma_weighted']].sum()

for v in ['d_prime_weighted','d_prime_log_weighted','cv_weighted','selectivity_weighted','resp_index_weighted',
          'mu_weighted','sigma_weighted']:
    df_wa[v] = df_wa[v]/df_wa['ave_rel_cont_resp']
    

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


    # f,a = plt.subplots(1,3, figsize = (1.9,1.7))

    # sns.scatterplot(x = df_wa[df_wa.trained==False].cv_weighted.to_numpy(),
    #                 y = df_wa[df_wa.trained==True].cv_weighted.to_numpy(),
    #                 ax = a[0], color = 'k')
    # a[0].set_xlim([0.3,0.7])
    # a[0].set_ylim([0.3,0.7])
    # a[0].plot([0.3,0.7],[0.3,0.7],'--k')
    # a[0].set_title('Weighted mean of coef. of variation')
    # a[0].set_xlabel('Naive')
    # a[0].set_ylabel('Proficient')

    # sns.scatterplot(x = df_wa[df_wa.trained==False].selectivity_weighted.to_numpy(),
    #                 y = df_wa[df_wa.trained==True].selectivity_weighted.to_numpy(),
    #                 ax = a[1], color = 'k')
    # a[1].set_xlim([0.4,0.7])
    # a[1].set_ylim([0.4,0.7])
    # a[1].plot([0.4,0.7],[0.4,0.7],'--k')
    # a[1].set_title('Weighted mean of selectivity')
    # a[1].set_xlabel('Naive')
    # # a[1].set_ylabel('Proficient')


    # sns.scatterplot(x = df_wa[df_wa.trained==False].d_prime_weighted.to_numpy(),
    #                 y = df_wa[df_wa.trained==True].d_prime_weighted.to_numpy(),
    #                 ax = a[2], color = 'k')
    # a[2].set_xlim([2,7])
    # a[2].set_ylim([2,7])
    # a[2].plot([2,7],[2,7],'--k')
    # a[2].set_title(r'Weighted mean of $d^{\prime}$')
    # a[2].set_xlabel('Naive')
    # # a[2].set_ylabel('Proficient')


    f,ax = plt.subplots(1,4,figsize = (4.5,0.9),sharey = False)
    # f,ax = plt.subplots(1,4,figsize = (6,3))

    labelpad = 2

    sns.stripplot(data = df_wa, x = 'trained', y = 'resp_index_weighted',
                  hue = 'trained', ax = ax[2], palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2,
                  linewidth = 0.5)
    ax[2].get_legend().remove()
    ax[2].set_ylabel(r'$|(f_{45}-f_{90})/(f_{45}+f_{90})|$', labelpad = labelpad)
    ax[2].set_ylim([0.6,0.9])
    ax[2].set_yticks(np.linspace(0.6,0.9,3))
    sns.stripplot(data = df_wa, x = 'trained', y = 'mu_weighted',
                  hue = 'trained', ax = ax[0], palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2, dodge = True,
                  linewidth = 0.5)
    ax[0].get_legend().remove()
    ax[0].set_ylabel('Mean response to pref.', labelpad = labelpad)
    ax[0].set_ylim([0.7,0.9])
    ax[0].set_yticks(np.linspace(0.7,0.9,3))
    # sns.stripplot(data = df_wa, x = 'trained', y = 'sigma_weighted',
    #               hue = 'trained', ax = ax[2], palette = 'colorblind', jitter = True,
    #               edgecolor = 'black', s = 2, dodge = True,
    #               linewidth = 0.5)
    # ax[2].get_legend().remove()
    # ax[2].set_ylabel('Std', labelpad = labelpad)
    # ax[2].set_ylim([0.25,0.65])
    # ax[2].set_yticks(np.linspace(0.25,0.65,3))
    sns.stripplot(data = df_wa, x = 'trained', y = 'cv_weighted',
                  hue = 'trained', ax = ax[1],  palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2,
                  linewidth = 0.5)
    ax[1].get_legend().remove()
    ax[1].set_ylabel('Coef. of var. for pref.', labelpad = labelpad)
    ax[1].set_ylim([0.4,0.8])
    ax[1].set_yticks(np.linspace(0.4,0.8,3))
    sns.stripplot(data = df_wa, x = 'trained', y = 'd_prime_log_weighted',
                  hue = 'trained', ax = ax[3],  palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2,
                  linewidth = 0.5)
    ax[3].get_legend().remove()
    ax[3].set_ylabel('d-prime', labelpad = labelpad)
    # ax[3].set_yscale('log')
    ax[3].set_ylim([0.4,1.5])
    ax[3].set_yticks(np.linspace(0.4,1.5,3))

    for a in ax:
        a.get_xaxis().set_visible(False)
        sns.despine(ax=a,trim=True, bottom = True)
        a.set_xticks([])
        a.set_xlabel('')
        xlims = a.get_xlim()

        a.set_xlim([xlims[0]-0.25,xlims[1]+0.25])
        pair_points(a)



    f.tight_layout()

    # plt.subplot_tool()

    f.savefig(join(fig_save,'stim decoding_45 and 90_weighted ave mean cv resp_index d.svg'), format = 'svg')

#%% stats

import statsmodels.api as sm
import statsmodels.formula.api as smf


df_wa = df_weights.copy()

# df_wa['d_prime_weighted'] = np.abs(df_wa.d_prime) * df_wa.rel_weight
# df_wa['cv_weighted'] = df_wa.cv * df_wa.rel_weight
# df_wa['selectivity_weighted'] = df_wa.selectivity * df_wa.rel_weight

df_wa['d_prime'] = np.log(df_wa.d_prime.abs())
df_wa['d_prime_weighted'] = df_wa.d_prime * df_wa.ave_rel_cont_resp
df_wa['cv_weighted'] = df_wa.cv_pref * df_wa.ave_rel_cont_resp
df_wa['selectivity_weighted'] = df_wa.selectivity * df_wa.ave_rel_cont_resp
df_wa['resp_index_weighted'] = np.abs(df_wa.resp_index) * df_wa.ave_rel_cont_resp
df_wa['mu_weighted'] = df_wa.mu_pref * df_wa.ave_rel_cont_resp
df_wa['sigma_weighted'] = df_wa.sigma_pref * df_wa.ave_rel_cont_resp

# df_wa = df_wa.groupby(['subject','trained'],as_index=False)[['d_prime_weighted',
#                     'cv_weighted','selectivity_weighted','ave_rel_cont_resp',
#                     'resp_index_weighted','mu_weighted', 'sigma_weighted']].sum()

# for v in ['d_prime_weighted','cv_weighted','selectivity_weighted','resp_index_weighted',
#           'mu_weighted','sigma_weighted']:
#     df_wa[v] = df_wa[v]/df_wa['ave_rel_cont_resp']

df_wa = df_wa.dropna()

md = smf.mixedlm(f'd_prime ~ C(trained)', df_wa, groups=df_wa['subject'], re_formula = '~C(trained)')
mdf = md.fit(method='powell')
print(mdf.summary())

#%% Instead of weighted, just find regular averages for d-prime, selectivity, CV

# mpl.rcParams['figure.dpi'] = 100


df_plot = df_weights.copy()
df_plot['d_prime'] = df_plot.d_prime.abs()
df_plot['resp_index'] = df_plot.resp_index.abs()

df_plot = df_plot.groupby(['subject','trained'])[['d_prime','selectivity','cv','resp_index',
                                                  'mu','sigma']].mean().reset_index()


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

    f,ax = plt.subplots(1,5,figsize = (4.5,0.8))
    # f,ax = plt.subplots(1,4,figsize = (6,3))

    labelpad = 2

    sns.stripplot(data = df_plot, x = 'trained', y = 'resp_index',
                  hue = 'trained', ax = ax[0], palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2,
                  linewidth = 0.5)
    ax[0].get_legend().remove()
    ax[0].set_ylabel('Resp. index', labelpad = labelpad)
    ax[0].set_ylim([0.265,0.875])
    ax[0].set_yticks(np.linspace(0.275,0.875,3))
    sns.stripplot(data = df_plot, x = 'trained', y = 'mu',
                  hue = 'trained', ax = ax[1], palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2, dodge = True,
                  linewidth = 0.5)
    ax[1].get_legend().remove()
    ax[1].set_ylabel('Mean', labelpad = labelpad)
    ax[1].set_ylim([0.3,0.9])
    ax[1].set_yticks(np.linspace(0.3,0.9,3))
    sns.stripplot(data = df_plot, x = 'trained', y = 'sigma',
                  hue = 'trained', ax = ax[2], palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2, dodge = True,
                  linewidth = 0.5)
    ax[2].get_legend().remove()
    ax[2].set_ylabel('Std', labelpad = labelpad)
    ax[2].set_ylim([0.25,0.65])
    ax[2].set_yticks(np.linspace(0.25,0.65,3))
    sns.stripplot(data = df_plot, x = 'trained', y = 'cv',
                  hue = 'trained', ax = ax[3],  palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2,
                  linewidth = 0.5)
    ax[3].get_legend().remove()
    ax[3].set_ylabel('Coef. of var.', labelpad = labelpad)
    ax[3].set_ylim([0.25,1.5])
    ax[3].set_yticks(np.linspace(0.25,1.5,3))
    sns.stripplot(data = df_plot, x = 'trained', y = 'd_prime',
                  hue = 'trained', ax = ax[4],  palette = 'colorblind', jitter = True,
                  edgecolor = 'black', s = 2,
                  linewidth = 0.5)
    ax[4].get_legend().remove()
    ax[4].set_ylabel('d-prime', labelpad = labelpad)
    ax[4].set_ylim([0.4,6.5])
    ax[4].set_yticks(np.linspace(0.5,6.5,4))


    for a in ax:
        a.get_xaxis().set_visible(False)
        sns.despine(ax=a,trim=True, bottom = True)
        a.set_xticks([])
        a.set_xlabel('')
        xlims = a.get_xlim()

        a.set_xlim([xlims[0]-0.25,xlims[1]+0.25])
        pair_points(a)



    f.tight_layout()


#%%

# For each experiment, find proportion of decision function as a set proportion of cells

df_prop_weights = df_weights.copy()

# one percent intervals
bins = np.linspace(0,1,101)

df_prop_weights['cell_prop_bins'] = pd.cut(df_prop_weights.prop_of_cells,bins,labels=np.linspace(0.01,1,100))
df_prop_weights['rel_weight_bins'] = pd.cut(df_prop_weights.rel_weight,bins,labels=np.linspace(0.01,1,100))

df_prop_weights = df_prop_weights.groupby(['subject','trained','cell_prop_bins'],as_index=False).sum()
# df_prop_weights = df_prop_weights.groupby(['subject','trained','rel_weight_bins'],as_index=False).sum()


for s,t in zip(subject_unique,trained):
    ind = np.logical_and(df_prop_weights.subject==s,df_prop_weights.trained==t)

    df_prop_weights.loc[ind,'prop_resp_45_sum'] = df_prop_weights[ind].prop_resp_45.cumsum()
    df_prop_weights.loc[ind,'prop_resp_90_sum'] = df_prop_weights[ind].prop_resp_90.cumsum()

df_prop_weights['prop_resp_sum_ave'] = (df_prop_weights.prop_resp_45_sum + df_prop_weights.prop_resp_90_sum)/2
df_prop_weights['prop_resp_ave'] = (df_prop_weights.prop_resp_45 + df_prop_weights.prop_resp_90)/2

# sns.lineplot(data = df_prop_weights, x = 'cell_prop_bins', y = 'prop_resp_45_sum', hue = 'trained',
#              errorbar = ('se',1))
# sns.lineplot(data = df_prop_weights, x = 'cell_prop_bins', y = 'prop_resp_45', hue = 'trained',
#              errorbar = ('se',1), palette = 'colorblind')
# sns.lineplot(data = df_prop_weights, x = 'cell_prop_bins', y = 'prop_resp_90', hue = 'trained',
#              errorbar = ('se',1), palette = 'colorblind', linestyle = '--')

sns.lineplot(data = df_prop_weights, x = 'cell_prop_bins', y = 'prop_resp_ave',
              hue = 'trained', errorbar=('se',1), palette = 'colorblind')

# sns.lineplot(data = df_prop_weights, x = 'rel_weight_bins', y = 'prop_resp_ave',
#               hue = 'trained', errorbar=('se',1), palette = 'colorblind')

#%%

n_cells = np.zeros(len(subject_unique))

for i,(s,t) in enumerate(zip(subject_unique,trained)):
    ind = np.logical_and(df_weights.subject==s,df_weights.trained==t)
    n_cells[i] = ind.sum()

prop_last = df_prop_weights[df_prop_weights.cell_prop_bins==1].prop_resp_ave.to_numpy()

plt.plot(n_cells,prop_last,'.')


#%%

n_cells = np.zeros(len(subject_unique))
max_weight = np.zeros(len(subject_unique))

for i,(s,t) in enumerate(zip(subject_unique,trained)):
    ind = np.logical_and(df_weights.subject==s,df_weights.trained==t)
    n_cells[i] = ind.sum()
    max_weight[i] = df_weights[ind].abs_weight.max()


sns.scatterplot(x=n_cells, y = max_weight, hue = trained)

#%%

sns.lineplot(data = df_weights, x = 'prop_of_cells', y = 'prop_resp_45',
             hue = 'trained', errorbar = ('se',1))

#%% Plot model weight by orientation preference, selectivity, cv, d_prime

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'd_prime', palette = 'vlag', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue d_prime.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'cv', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue cv.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'ave_cv', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue ave cv.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ori_pref', kind = 'scatter',
            hue = 'selectivity', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ori pref_hue selectivity.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'selectivity', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by selectivity_hue ori pref.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'd_prime', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by d-prime_hue ori pref.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'cv', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by cv_hue ori pref.svg'),format='svg')

sns.relplot(data = df_weights, col = 'subject', row = 'trained',
            x = 'model_weight', y = 'ave_cv', hue = 'pref_bin', kind = 'scatter',
            palette = 'hls', ec = 'k')
plt.savefig(join(fig_save,'stim decoding_45 and 90_weight by ave cv_hue ori pref.svg'),format='svg')


#%% CV vs selectivity with model contribution

i_subject = 'SF180515'

ind = df_weights.subject == i_subject

annoted_cells = [[5928,5255],[4724,4383]]
annote_label = ['a','b']

df_plot = df_weights[ind].copy()
# df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()

df_plot_n = df_plot[df_plot.trained==False]
df_plot_t = df_plot[df_plot.trained]

df_plot_n = df_plot_n.reset_index(drop=True)
df_plot_t = df_plot_t.reset_index(drop=True)

conditions = df_plot.trained.unique()[::-1]

for ic,c in enumerate(annoted_cells[0]):
        row = np.where((df_plot_n.cell_index==c))[0][0]
        idx = [i for i in df_plot_n.index if i != row]
        if ic == 0:
            idx.append(row)
        else:
            idx.insert(len(df_plot_n)-10,row)
        # breakpoint()
        df_plot_n = df_plot_n.reindex(idx)

for ic,c in enumerate(annoted_cells[1]):
        row = np.where((df_plot_t.cell_index==c))[0][0]
        idx = [i for i in df_plot_t.index if i != row]
        if ic == 0:
            idx.append(row)
        else:
            idx.insert(len(df_plot_t)-10,row)
        # breakpoint()
        df_plot_t = df_plot_t.reindex(idx)

# df_plot_n = df_plot_n.reset_index(drop=True)
# df_plot_t = df_plot_t.reset_index(drop=True)

save_flag = True

# cmap = 'Greys'
# cmap = 'mako_r'
cmap = 'viridis_r'

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
    mpl.rcParams['figure.dpi'] = 500


    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_size weight.svg'),format='svg')


    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = False,
    #             ec = 'k')

    f,ax = plt.subplots(1,2,figsize=(2.5,1.25),sharey=False,sharex=False)

    # sns.scatterplot(data = df_weights[df_weights.trained==False], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[0], palette = 'magma_r',
    #                       hue_norm = (0,0.1), rasterized=True)

    # sns.scatterplot(data = df_weights[df_weights.trained==True], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[1], palette = 'magma_r',
    #                       hue_norm=(0,0.1), rasterized=True)

    

    labelpad = 2
    
    if save_flag:
        rasterized = False
    else:
        rasterized = True

    sns.scatterplot(data = df_plot_n,
                    x = 'cv',
                    y = 'resp_index', 
                    hue = 'ave_rel_cont_resp',
                    hue_norm = (0,1),
                    ec = 'grey',
                    legend = False, ax = ax[0], palette = cmap,
                    # size = 'ave_rel_cont_resp',
                    # size_norm = (0,1),
                    # sizes = (0.5,3),
                    rasterized=rasterized)
    sns.scatterplot(data = df_plot_t,
                    x = 'cv',
                    y = 'resp_index', 
                    hue = 'ave_rel_cont_resp',
                    hue_norm = (0,1),
                    ec = 'grey',
                    legend = False, ax = ax[1], palette = cmap,
                    # size = 'ave_rel_cont_resp',
                    # size_norm = (0,1),
                    # sizes = (0.5,3),
                    rasterized=rasterized)

    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])


    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.825, 0.22, 0.008, 0.5])
    plt.colorbar(sm, cax = cbar_ax, label = 'Contribution to dec. fun.')
    cbar_ax.set_yticks([0,1])
    cbar_ax.set_yticklabels(['<= 0', '1'])

    # ax[0].set_ylabel('Selectivity')
    # ax[0].set_xlabel('Coef. of variation')
    # ax[1].set_xlabel('Coef. of variation')

    ax[0].set_xlabel('Inter-trial variability', labelpad = labelpad)
    ax[0].set_ylabel('Response index', labelpad = labelpad)

    ax[1].set_xlabel('Inter-trial variability', labelpad = labelpad)

    ax[1].set_ylabel('')
    ax[1].set_yticks([])

    ax[0].set_title('Naive', color = sns.color_palette('colorblind')[0],pad = 2)
    ax[1].set_title('Proficient', color = sns.color_palette('colorblind')[1], pad = 2)

    ax[0].set_xlim([-0.05,5])
    ax[1].set_xlim([-0.05,5])

    ax[0].set_xticks(np.linspace(0,5,6))
    ax[1].set_xticks(np.linspace(0,5,6))

    # ax[0].set_ylim([-0.01,1])
    # ax[1].set_ylim([-0.01,1])

    ax[0].set_ylim([-1.05,1.05])
    ax[1].set_ylim([-1.05,1.05])

    # ax[0].set_yticks(np.linspace(0,0.9,4))


    sns.despine(trim=True,offset=2,ax=ax[0])
    sns.despine(trim=True,offset=2,left = True, ax=ax[1])

    for a in ax:
        a.set_box_aspect(1)

    # Draw arrows at points
    colors = ['black','black']

    if 'annoted_cells' in locals():
        for ci,c in enumerate(annoted_cells[0]):
            # breakpoint()
            x = df_plot_n[df_plot_n.cell_index == c].cv.to_numpy()[0]
            y = df_plot_n[df_plot_n.cell_index == c].resp_index.to_numpy()[0]
            ax[0].annotate("", xy=(x-0.07,y-0.07), xytext=(x+0.3, y+0.3), xycoords = 'data', arrowprops=dict(arrowstyle="-|>",
                                                                                  color = colors[ci]),
                           zorder = 100, ha = 'center',
                           va = 'center')
            # ax[i].text(x,y,annote_label[ci], fontsize = 5, color = 'white')

    if 'annoted_cells' in locals():
        for ci,c in enumerate(annoted_cells[1]):
            # breakpoint()
            x = df_plot_t[df_plot_t.cell_index == c].cv.to_numpy()
            y = df_plot_t[df_plot_t.cell_index == c].resp_index.to_numpy()[0]
            ax[1].annotate("", xy=(x-0.07, y-0.07), xytext=(x+0.3, y+0.3), xycoords = 'data', arrowprops=dict(arrowstyle="-|>",
                                                                                  color = colors[ci]),
                           zorder = 100, ha = 'center',
                           va = 'center')
            # ax[i].text(x,y,annote_label[ci], fontsize = 5, color = 'white')

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = True,
    #             ec = 'k')

    if save_flag:
        plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by resp index_hue mean relative contribution to df_' + i_subject + '.svg'),format='svg')


#%% CV vs selectivity with model contribution - all but SF180515

ind = df_weights.subject != 'SF180515'

df_plot = df_weights[ind].copy()
# df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()


cond_colors = sns.color_palette('colorblind',2)

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



p = (
        so.Plot(df_plot, x = 'cv', y = 'resp_index', color = 'ave_rel_cont_resp',
        fillcolor = 'ave_rel_cont_resp')
        .share(x = False, y = False)
        .layout(size=(2.5,5))
        .facet(row = 'subject',col = 'trained')
        .add(so.Dot(pointsize = 2, edgewidth = 0.16, edgecolor = 'k'),legend = False)
        .scale(color = so.Continuous('viridis_r',norm = (0,1)))
        .theme(style)
        # .limit(x = (0,5), y = (-1,1))
        .label(x = 'Coef. of variation', y = 'Response index')
        .plot()
)

for ai,a in enumerate(p._figure.axes):

    if ai == 0:
        a.set_title('Naive',fontsize = 5, color = cond_colors[0], pad = 2)
    elif ai == 1:
        a.set_title('Proficient', fontsize = 5, color = cond_colors[1], pad = 2)
    else:
        a.set_title('')

    if ai < 6:
        bottom = True
        a.set_xticks([])
    else:
        bottom = False
        a.set_xticks(np.linspace(0,5,3))

    if (ai % 2) != 0:
        left = True
        a.set_yticks([])
    else:
        left = False
    
    a.set_xlim([-0.5,5.5])
    a.set_ylim([-1.2,1.2])
    sns.despine(ax=a, left = left, bottom = bottom, trim = True)
    

p.save(join(fig_save,'stim decoding_45 and 90_cv by resp index_hue mean relative contribution to df_all but SF180515.svg'),format='svg')

#%% CV vs selectivity with model contribution - same as above but using seaborn jointplot instead

ind = df_weights.subject == 'SF180515'

annoted_cells = [[5928,4952],[4724,4383]]
annote_label = ['a','b']

df_plot = df_weights[ind].copy()
# df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()

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
    mpl.rcParams['figure.dpi'] = 120


    cmap = 'viridis_r'

    g0 = sns.JointGrid(data = df_plot[df_plot.trained==False], x = 'cv', y = 'resp_index',
                      height = 1.25)
    g0.plot_marginals(sns.histplot, stat = 'probability', element = 'step', color = 'grey')
    g0.plot_joint(sns.scatterplot, data = df_plot[df_plot.trained==False],
                 hue = 'ave_rel_cont_resp', hue_norm = (0,1), palette = cmap,
                 legend = False)
    g0.set_axis_labels('Coef. of variation','Response index')
    g0.ax_joint.set_ylim([-1,1])
    g0.ax_joint.set_xlim([0,5])
    g0.ax_marg_y.set_ylim([-1,1])
    g0.ax_marg_x.set_xlim([0,5])
    g0.ax_joint.set_aspect(1)


    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_SF180515_Naive.svg'),format='svg')


    # g0 = pw.load_seaborngrid(g0,label='g0',figsize=(1.25,1.25))

    g1 = sns.JointGrid(data = df_plot[df_plot.trained==True], x = 'cv', y = 'resp_index',
                      height = 1.25)
    g1.plot_marginals(sns.histplot, stat = 'probability', element = 'step', color = 'grey')
    g1.plot_joint(sns.scatterplot, data = df_plot[df_plot.trained==True],
                 hue = 'ave_rel_cont_resp', hue_norm = (0,1), palette = cmap,
                 legend = False)
    g1.set_axis_labels('Coef. of variation','Response index')
    g1.ax_joint.set_ylim([-1,1])
    g1.ax_joint.set_xlim([0,5])
    g1.ax_marg_y.set_ylim([-1,1])
    g1.ax_marg_x.set_xlim([0,5])
    g1.ax_joint.set_aspect(1)

    # g1 = pw.load_seaborngrid(g1,label='g1',figsize=(1.25,1.25))

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_SF180515_Proficient.svg'),format='svg')

    # norm = plt.Normalize(0, 1)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # f,a = plt.subplots(figsize=(0.1,1.25))

    # # cbar_ax = g1.fig.add_axes([0.825, 0.22, 0.008, 0.5])
    # plt.colorbar(sm, cax = a, label = 'Contribution to dec. fun.')
    # cbar_ax.set_yticks([0,1])
    # cbar_ax.set_yticklabels(['<= 0', '1'])

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_SF180515_Proficient.svg'),format='svg')

    # (g0|g1).savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_size weight_test.svg'),format='svg')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_size weight.svg'),format='svg')


    # sns.despine()
    # g.fig.tight_layout()

    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_size weight.svg'),format='svg')


    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = False,
    #             ec = 'k')

    # f,ax = plt.subplots(1,2,figsize=(2.5,1.25),sharey=False,sharex=False)

    # sns.scatterplot(data = df_weights[df_weights.trained==False], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[0], palette = 'magma_r',
    #                       hue_norm = (0,0.1), rasterized=True)

    # sns.scatterplot(data = df_weights[df_weights.trained==True], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[1], palette = 'magma_r',
    #                       hue_norm=(0,0.1), rasterized=True)

    # cmap = 'Greys'
    # cmap = 'mako_r'
    # cmap = 'viridis_r'

    # labelpad = 2

    # sns.scatterplot(data = df_plot[df_plot.trained==False],
    #                 x = 'cv',
    #                 y = 'resp_index', hue = 'ave_rel_cont_resp',ec = 'k',
    #                 legend = False, ax = ax[0], palette = cmap,
    #                 hue_norm = (0,1),
    #                 rasterized=False)

    # sns.scatterplot(data = df_plot[df_plot.trained==True],
    #                 x = 'cv',
    #                 y = 'resp_index', hue = 'ave_rel_cont_resp',ec = 'k',
    #                 legend = False, ax = ax[1], palette = cmap,
    #                 hue_norm = (0,1),
    #                 rasterized=False)

    # norm = plt.Normalize(0, 1)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # # Draw arrows at points
    # if 'annoted_cells' in locals():
    #     for i in range(len(annoted_cells)):
    #         for ci,c in enumerate(annoted_cells[i]):
    #             # breakpoint()
    #             x = df_plot[(df_plot.trained==df_plot.trained.unique()[::-1][i]) & (df_plot.cell_index == c)].cv
    #             y = df_plot[(df_plot.trained==df_plot.trained.unique()[::-1][i]) & (df_plot.cell_index == c)].resp_index
    #             # ax[i].annotate("", xy=(x, y), xytext=(x+0.2, y+0.2), arrowprops=dict(arrowstyle="->",
    #             #                                                                      color = 'black'))
    #             # ax[i].text(x,y,annote_label[ci], fontsize = 5)


    # f.subplots_adjust(right=0.8)
    # cbar_ax = f.add_axes([0.825, 0.22, 0.008, 0.5])
    # plt.colorbar(sm, cax = cbar_ax, label = 'Contribution to dec. fun.')
    # cbar_ax.set_yticks([0,1])
    # cbar_ax.set_yticklabels(['<= 0', '1'])

    # # ax[0].set_ylabel('Selectivity')
    # # ax[0].set_xlabel('Coef. of variation')
    # # ax[1].set_xlabel('Coef. of variation')

    # ax[0].set_xlabel('Coef. of variation', labelpad = labelpad)
    # ax[0].set_ylabel('Response index', labelpad = labelpad)

    # ax[1].set_xlabel('Coef. of variation', labelpad = labelpad)

    # ax[1].set_ylabel('')
    # ax[1].set_yticks([])

    # ax[0].set_title('Naive', color = sns.color_palette('colorblind')[0],pad = 2)
    # ax[1].set_title('Proficient', color = sns.color_palette('colorblind')[1], pad = 2)

    # ax[0].set_xlim([-0.05,5])
    # ax[1].set_xlim([-0.05,5])

    # ax[0].set_xticks(np.linspace(0,5,6))
    # ax[1].set_xticks(np.linspace(0,5,6))

    # # ax[0].set_ylim([-0.01,1])
    # # ax[1].set_ylim([-0.01,1])

    # ax[0].set_ylim([-1.05,1.05])
    # ax[1].set_ylim([-1.05,1.05])

    # # ax[0].set_yticks(np.linspace(0,0.9,4))


    # sns.despine(trim=True,offset=2,ax=ax[0])
    # sns.despine(trim=True,offset=2,left = True, ax=ax[1])

    # for a in ax:
    #     a.set_box_aspect(1)

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = True,
    #             ec = 'k')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by resp index_hue weight.svg'),format='svg')



#%% Distributions of response index, cv, mean, std using seaborn objects

# ind = df_weights.subject == 'SF180515'

# df_plot = df_weights[ind].copy()
df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()

df_plot['high_cont'] = df_plot.ave_rel_cont_resp > 0.05
df_plot.loc[df_plot['high_cont'],'high_cont_label'] = 'High contributing'
df_plot.loc[~df_plot['high_cont'],'high_cont_label'] = 'Low contributing'

df_plot = df_plot.dropna(axis=0)

# ri_edges = np.histogram_bin_edges(df_plot.abs_resp_index, range = (0,1), bins = 'fd')
# cv_edges = np.histogram_bin_edges(df_plot.cv, range = (0,df_plot.cv.max()), bins = 'fd')

ri_edges = np.linspace(0,1,40)
cv_edges = np.linspace(0,5,40)
mu_edges = np.linspace(0,1.8,40)
sigma_edges = np.linspace(0,3,40)


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

fig_size = (1,1)

p0 = (
    so.Plot(df_plot, x = 'abs_resp_index', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .facet(col = 'high_cont_label')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = ri_edges, stat = 'probability', common_norm = False),
         legend = False)
    # .limit(x = (0,1))
    .scale(color = 'colorblind')
    .theme(style)
    .label(y = 'Proportion of neurons', x = r'|Response index|')
    .share(y = False, x = True)
    .plot()
    )

for ai,a in enumerate(p0._figure.axes):
    if ai == 1:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,4))

    else:
        a.set_ylim(0,0.06)
        a.set_yticks(np.linspace(0,0.06,4))
    sns.despine(ax=a,trim=True)

p0

p0.save(join(fig_save,'stim decoding_45 and 90_dist of resp index.svg'),format = 'svg')


p1 = (
    so.Plot(df_plot, x = 'cv_pref', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .facet(col = 'high_cont_label')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = cv_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,5))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,5), count=6))
    .theme(style)
    .label(y = 'Proportion of neurons', x = 'Coef. of variation')
    .share(y = False, x = True)
    .plot()
    )

for ai,a in enumerate(p1._figure.axes):
    if ai == 0:
        a.set_ylim(0,0.2)
        a.set_yticks(np.linspace(0,0.2,3))
    else:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,4))
    a.set_xlim((a.get_xlim()[0],5))
    sns.despine(ax=a,trim=True)

p1


p1.save(join(fig_save,'stim decoding_45 and 90_dist of cv.svg'),format = 'svg')




p2 = (
    so.Plot(df_plot, x = 'mu_pref', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .facet(col = 'high_cont_label')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = mu_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,1.8))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,1.8), count=3))
    .theme(style)
    .label(y = 'Proportion of neurons', x = 'Mean response')
    .share(y = False)
    .plot()
    )


for ai,a in enumerate(p2._figure.axes):
    if ai == 0:
        a.set_ylim(0,0.1)
        a.set_yticks(np.linspace(0,0.1,3))
    else:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,4))
    sns.despine(ax=a,trim=True)


p2.save(join(fig_save,'stim decoding_45 and 90_dist of mu.svg'),format = 'svg')


p3 = (
    so.Plot(df_plot, x = 'sigma_pref', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .facet(col = 'high_cont_label')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = sigma_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,3))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,6), count=5))
    .theme(style)
    .label(y = 'Proportion of neurons', x = 'STD of response')
    .share(y = False)
    .plot()
    )


for ai,a in enumerate(p3._figure.axes):
    if ai == 0:
        a.set_ylim(0,0.15)
        a.set_yticks(np.linspace(0,0.15,4))
    else:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,3), )
    sns.despine(ax=a,trim=True)


p3.save(join(fig_save,'stim decoding_45 and 90_dist of sigma.svg'),format = 'svg')




# p2 = (
#     so.Plot(df_plot, x = 'resp_index', color = 'trained')
#     .layout(size = (5,5), engine = 'tight')
#     .facet(col = 'high_cont_label', row = 'subject')
#     .add(so.Bars(edgecolor = 'black'), so.Hist(bins = np.linspace(-1,1,40), stat = 'count'),
#          legend = False)
#     .scale(color = 'colorblind')
#     .theme(style)
#     .label(y = 'Proportion of neurons', x = 'Response index')
#     .share(y = False)
#     .plot()
#     # .show()
#     # .save(join(fig_save,'test.svg'),format = 'svg')
# )



#%% Distributions of response index, cv, mean, std using seaborn objects

# ind = df_weights.subject == 'SF180515'

# df_plot = df_weights[ind].copy()
df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()

df_plot = df_plot.dropna(axis=0)

df_plot['d_prime'] = df_plot.d_prime.abs()

# ri_edges = np.histogram_bin_edges(df_plot.abs_resp_index, range = (0,1), bins = 'fd')
# cv_edges = np.histogram_bin_edges(df_plot.cv, range = (0,df_plot.cv.max()), bins = 'fd')

ri_edges = np.linspace(0,1,40)
cv_edges = np.linspace(0,5,40)
mu_edges = np.linspace(0,1.8,40)
sigma_edges = np.linspace(0,3,40)
d_edges = np.linspace(0,19,40)


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

fig_size = (1.2,1.2)


p0 = (
    so.Plot(df_plot, x = 'abs_resp_index', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = ri_edges, stat = 'probability', common_norm = False),
         legend = False)
    # .limit(x = (0,1))
    .scale(color = 'colorblind')
    .theme(style)
    .label(y = 'Prop. of neurons', x = r'|Response index|')
    # .on(sf[0])
    .plot()
    )

for ai,a in enumerate(p0._figure.axes):
    if ai == 1:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,4))

    else:
        a.set_ylim(0,0.06)
        a.set_yticks(np.linspace(0,0.06,4))
    a.set_box_aspect(1)
    sns.despine(ax=a,trim=True)

p0.save(join(fig_save,'stim decoding_45 and 90_dist of resp index_all cells.svg'),format = 'svg')



p1a = (
    so.Plot(df_plot, x = 'cv_pref', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = cv_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,5))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,5), count=6))
    .theme(style)
    .label(y = 'Prop. of neurons', x = 'Coef. of variation')
    # .on(sf[1])
    .plot()
    )

for ai,a in enumerate(p1a._figure.axes):
    
    if ai == 0:
        a.set_ylim(0,0.2)
        a.set_yticks(np.linspace(0,0.2,3))
    else:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,4))
    a.set_xlim((a.get_xlim()[0],5))
    a.set_box_aspect(1) 
    sns.despine(ax=a,trim=True)

p1a.save(join(fig_save,'stim decoding_45 and 90_dist of pref cv_all cells.svg'),format = 'svg')


p1b = (
    so.Plot(df_plot, x = 'cv_nonpref', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = cv_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,5))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,5), count=6))
    .theme(style)
    .label(y = 'Prop. of neurons', x = 'Coef. of variation')
    # .on(sf[1])
    .plot()
    )

for ai,a in enumerate(p1b._figure.axes):
    
    if ai == 0:
        a.set_ylim(0,0.2)
        a.set_yticks(np.linspace(0,0.2,3))
    else:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,4))
    a.set_xlim((a.get_xlim()[0],5))
    a.set_box_aspect(1) 
    sns.despine(ax=a,trim=True)

p1b.save(join(fig_save,'stim decoding_45 and 90_dist of nonpref cv_all cells.svg'),format = 'svg')


p2 = (
    so.Plot(df_plot, x = 'mu_nonpref', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = mu_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,1.8))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,1.8), count=3))
    .theme(style)
    .label(y = 'Prop. of neurons', x = 'Mean response')
    # .on(sf[2])
    .plot()
    )


for ai,a in enumerate(p2._figure.axes):
    if ai == 0:
        a.set_ylim(0,0.1)
        a.set_yticks(np.linspace(0,0.1,3))
    else:
        a.set_ylim(0,0.3)
        a.set_yticks(np.linspace(0,0.3,4))
    a.set_box_aspect(1)
    sns.despine(ax=a,trim=True)


p2.save(join(fig_save,'stim decoding_45 and 90_dist of nonpref mean resp_all cells.svg'),format = 'svg')


p3 = (
    so.Plot(df_plot, x = 'sigma_nonpref', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = sigma_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,3))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,6), count=3))
    .theme(style)
    .label(y = 'Prop. of neurons', x = 'STD of response')
    # .on(sf[3])
    .plot()
    )


for ai,a in enumerate(p3._figure.axes):
    a.set_xticks(np.linspace(0,3,4))
    a.set_ylim(0,0.15)
    a.set_yticks(np.linspace(0,0.15,4))
    a.set_box_aspect(1)

    sns.despine(ax=a,trim=True)

p3.save(join(fig_save,'stim decoding_45 and 90_dist of std of nonpref resp_all cells.svg'),format = 'svg')


p4 = (
    so.Plot(df_plot, x = 'd_prime', color = 'trained')
    .layout(size = (1.28,1.28), engine = 'tight')
    .add(so.Bars(edgecolor = 'black'), so.Hist(bins = d_edges, stat = 'probability', common_norm = False),
        legend = False)
    # .limit(x = (0,3))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,19), count=5))
    .theme(style)
    .label(y = 'Prop. of neurons', x = 'd-prime')
    # .on(sf[4])
    .plot()
    )

for ai,a in enumerate(p4._figure.axes):
    a.set_ylim(0,0.45)
    a.set_yticks(np.linspace(0,0.55,6))
    a.set_xticks(np.linspace(0,19,3))
    a.set_box_aspect(1)

   
    sns.despine(ax=a,trim=True)


p4.save(join(fig_save,'stim decoding_45 and 90_dist of dprime_all cells.svg'),format = 'svg')


#%% Distributions of response index, cv, mean, std using seaborn objects - HIGH MODEL CONTRIBUTING

# ind = df_weights.subject == 'SF180515'

# df_plot = df_weights[ind].copy()
df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()

df_plot['high_cont'] = df_plot.ave_rel_cont_resp > 0.05
df_plot.loc[df_plot['high_cont'],'high_cont_label'] = 'High contributing'
df_plot.loc[~df_plot['high_cont'],'high_cont_label'] = 'Low contributing'


df_plot = df_plot.dropna(axis=0)

df_plot['d_prime'] = df_plot.d_prime.abs()



fig_size = (1.3,1.3)

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

import math
def om(number):
    return math.floor(math.log(number, 10))

def dist_plot(df,x,n_bins = 40,bins = None,x_label='',y_label='% neurons',style=None,xscale = 'linear',
              size=(1.25,1.25),xlims=None,ylims=None,n_ticks_x=3,n_ticks_y=3,color='trained'):
    
    if xlims is None:
        xmin,xmax = df[x].min(),df[x].max()
        if xmin != 0:
            xmin = my_floor(xmin,om(np.abs(xmin))*-1)
        if xmax != 0:
            xmax = my_ceil(xmax,om(np.abs(xmax))*-1)

        xlims = (xmin,xmax)

    if style is None:
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
         'font.family' : 'sans-serif'}
         
    p = (so.Plot(df, x = x, color = color)
         .layout(size = size, engine = 'tight')
         .limit(x = xlims)
         .theme(style)
         .label(x = x_label, y = y_label)
        )
        
    if bins == 'auto':
        p = p.add(so.Bars(edgecolor = 'black'), so.Hist(bins = n_bins, stat = 'percent', common_norm = False),legend = False)
        
    else:
        bins = np.linspace(xlims[0],xlims[1],n_bins)
        p = p.add(so.Bars(edgecolor = 'black'), so.Hist(bins = bins, stat = 'percent', common_norm = False),legend = False)
        
    if xscale == 'linear':
        p = p.scale(color = 'colorblind')
    else:
        p = p.scale(color = 'colorblind', x = xscale)
        
    p = p.plot()

    xlims = p._figure.axes[0].get_xlim()
        
    if ylims is None:
        ymin,ymax = np.min(p._figure.axes[0].get_ylim()), np.max(p._figure.axes[0].get_ylim())
            
        if ymin != 0:
            ymin = my_floor(ymin,om(np.abs(ymin))*-1)
        if ymax != 0:
            ymax = my_ceil(ymax,om(np.abs(ymax))*-1)
            
        ylims = (ymin,ymax)

    if xscale != 'log':
        p._figure.axes[0].set_xticks(np.linspace(xlims[0],xlims[1],n_ticks_x))
    # else:
        # p._figure.axes[0].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    
    p._figure.axes[0].set_ylim(ylims)
    p._figure.axes[0].set_yticks(np.linspace(ylims[0],ylims[1],n_ticks_y))

    p._figure.axes[0].set_box_aspect(1)
    
    sns.despine(ax = p._figure.axes[0])

    return p

# All cells
# Resp index
dist_plot(df_plot,x='resp_index',x_label=r'($f_{45}-f_{90})/(f_{45}+f_{90})$',n_ticks_y = 4).save(
    join(fig_save,'stim decoding_45 and 90_dist resp index_all cells.svg'),format='svg')
# d-prime
dist_plot(df_plot,x='d_prime',x_label='d-prime', xscale = 'log', bins = 'auto', xlims = (0.001,20)).save(
    join(fig_save,'stim decoding_45 and 90_dist dprime_all cells.svg'),format='svg')
# Mean of pref
dist_plot(df_plot,x='mu_pref',x_label='Mean response',y_label = '% neurons',ylims=(0,10),xlims=(0,2)).save(
    join(fig_save,'stim decoding_45 and 90_dist mean pref_all cells.svg'),format='svg')
# std of pref
dist_plot(df_plot,x='sigma_pref',x_label='STD response',y_label = '% neurons', xlims = (0,3)).save(
    join(fig_save,'stim decoding_45 and 90_dist std pref_all cells.svg'),format='svg')
# cv of pref
dist_plot(df_plot,x='cv_pref',x_label='Coef. of variation',y_label = '% neurons', xlims = (0,6)).save(
    join(fig_save,'stim decoding_45 and 90_dist cv pref_all cells.svg'),format='svg')
# Mean of nonpref
dist_plot(df_plot,x='mu_nonpref',x_label='Mean response',y_label = '% neurons',xlims=(0,2)).save(
    join(fig_save,'stim decoding_45 and 90_dist mean nonpref_all cells.svg'),format='svg')
# std of nonpref
dist_plot(df_plot,x='sigma_nonpref',x_label='STD response',y_label = '% neurons',xlims = (0,3)).save(
    join(fig_save,'stim decoding_45 and 90_dist std nonpref_all cells.svg'),format='svg')
# cv of nonpref
dist_plot(df_plot,x='cv_nonpref',x_label='Coef. of variation',y_label = '% neurons', xlims = (0,6),
          ylims = (0,20)).save(
    join(fig_save,'stim decoding_45 and 90_dist cv nonpref_all cells.svg'),format='svg')


# Decoder cells
ind = df_plot.high_cont
# Resp index
dist_plot(df_plot[ind],x='resp_index',x_label=r'($f_{45}-f_{90})/(f_{45}+f_{90})$',n_ticks_y = 4).save(
    join(fig_save,'stim decoding_45 and 90_dist resp index_decoder cells.svg'),format='svg')
# d-prime
dist_plot(df_plot[ind],x='d_prime',x_label='d-prime',xscale = 'log', bins = 'auto').save(
    join(fig_save,'stim decoding_45 and 90_dist dprime_decoder cells.svg'),format='svg')
# Mean of pref
dist_plot(df_plot[ind],x='mu_pref',x_label='Mean response',y_label = '% neurons',ylims=(0,40),xlims=(0,2)).save(
    join(fig_save,'stim decoding_45 and 90_dist mean pref_decoder cells.svg'),format='svg')
# std of pref
dist_plot(df_plot[ind],x='sigma_pref',x_label='STD response',y_label = '% neurons', xlims = (0,3)).save(
    join(fig_save,'stim decoding_45 and 90_dist std pref_decoder cells.svg'),format='svg')
# cv of pref
dist_plot(df_plot[ind],x='cv_pref',x_label='Coef. of variation',y_label = '% neurons', xlims = (0,6)).save(
    join(fig_save,'stim decoding_45 and 90_dist cv pref_decoder cells.svg'),format='svg')
# Mean of nonpref
dist_plot(df_plot[ind],x='mu_nonpref',x_label='Mean response',y_label = '% neurons',xlims=(0,2)).save(
    join(fig_save,'stim decoding_45 and 90_dist mean nonpref_decoder cells.svg'),format='svg')
# std of pref
dist_plot(df_plot[ind],x='sigma_nonpref',x_label='STD response',y_label = '% neurons',xlims = (0,3)).save(
    join(fig_save,'stim decoding_45 and 90_dist std nonpref_decoder cells.svg'),format='svg')
# cv of pref
dist_plot(df_plot[ind],x='cv_nonpref',x_label='Coef. of variation',y_label = '% neurons',xlims = (0,6),
          ylims = (0,25)).save(
    join(fig_save,'stim decoding_45 and 90_dist cv nonpref_decoder cells.svg'),format='svg')



#%% stats on high cont cells

import statsmodels.api as sm
import statsmodels.formula.api as smf

df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()

df_plot['high_cont'] = df_plot.ave_rel_cont_resp > 0.05
df_plot.loc[df_plot['high_cont'],'high_cont_label'] = 'High contributing'
df_plot.loc[~df_plot['high_cont'],'high_cont_label'] = 'Low contributing'


df_plot = df_plot.dropna(axis=0)

df_plot['d_prime'] = df_plot.d_prime.abs()
df_plot['d_prime_log'] = np.log(df_plot.d_prime)

metric = 'mu_pref'

md = smf.mixedlm(f'{metric} ~ C(trained)', df_plot, groups=df_plot.subject, re_formula = '~C(trained)')
mdf = md.fit(method='powell')
print(mdf.summary())

md = smf.mixedlm(f'{metric} ~ C(trained)', df_plot[df_plot.high_cont], groups=df_plot[df_plot.high_cont].subject,
                 re_formula = '~C(trained)')
mdf = md.fit(method='powell')
print(mdf.summary())


#%% Distributions of response index, cv, mean, std using seaborn objects - cumulative histograms

# ind = df_weights.subject == 'SF180515'

# df_plot = df_weights[ind].copy()
df_plot = df_weights.copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)
df_plot['abs_resp_index'] = df_plot.resp_index.abs()

df_plot = df_plot.dropna(axis=0)

df_plot['d_prime'] = df_plot.d_prime.abs()

# ri_edges = np.histogram_bin_edges(df_plot.abs_resp_index, range = (0,1), bins = 'fd')
# cv_edges = np.histogram_bin_edges(df_plot.cv, range = (0,df_plot.cv.max()), bins = 'fd')

ri_edges = np.linspace(0,1,40)
cv_edges = np.linspace(0,5,40)
mu_edges = np.linspace(0,1.8,40)
sigma_edges = np.linspace(0,3,40)
d_edges = np.linspace(0,19)


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

fig_size = (1.4,1.4)

p0 = (
    so.Plot(df_plot, x = 'abs_resp_index', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Line(), so.Hist(stat = 'probability', common_norm = False, cumulative = True),
         legend = False)
    # .limit(x = (0,1))
    .scale(color = 'colorblind')
    .theme(style)
    .label(y = 'Proportion of neurons', x = r'|Response index|')
    .share(y = False, x = True)
    .plot()
    )

for ai,a in enumerate(p0._figure.axes):
    # if ai == 1:
    #     a.set_ylim(0,0.3)
    #     a.set_yticks(np.linspace(0,0.3,4))

    # else:
    #     a.set_ylim(0,0.06)
    #     a.set_yticks(np.linspace(0,0.06,4))
    sns.despine(ax=a,trim=True)

p0.save(join(fig_save,'stim decoding_45 and 90_dist of resp index_cum hist_all cells.svg'),format = 'svg')



p1 = (
    so.Plot(df_plot, x = 'cv', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Line(), so.Hist(stat = 'probability', common_norm = False, cumulative = True),
         legend = False)
    # .limit(x = (0,5))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,5), count=6))
    .theme(style)
    .label(y = 'Proportion of neurons', x = 'Coef. of variation')
    .share(y = False, x = True)
    .plot()
    )

for ai,a in enumerate(p1._figure.axes):
    # if ai == 0:
    #     a.set_ylim(0,0.2)
    #     a.set_yticks(np.linspace(0,0.2,3))
    # else:
    #     a.set_ylim(0,0.3)
    #     a.set_yticks(np.linspace(0,0.3,4))
    a.set_xlim((a.get_xlim()[0],5))
    sns.despine(ax=a,trim=True)

p1.save(join(fig_save,'stim decoding_45 and 90_dist of cv_cum hist_all cells.svg'),format = 'svg')


p2 = (
    so.Plot(df_plot, x = 'mu', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Line(), so.Hist(stat = 'probability', common_norm = False, cumulative = True),
         legend = False)
    # .limit(x = (0,1.8))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,1.8), count=3))
    .theme(style)
    .label(y = 'Proportion of neurons', x = 'Mean response')
    .share(y = False)
    .plot()
    )


for ai,a in enumerate(p2._figure.axes):
    # if ai == 0:
    #     a.set_ylim(0,0.1)
    #     a.set_yticks(np.linspace(0,0.1,3))
    # else:
    #     a.set_ylim(0,0.3)
    #     a.set_yticks(np.linspace(0,0.3,4))
    sns.despine(ax=a,trim=True)


p2.save(join(fig_save,'stim decoding_45 and 90_dist of mean resp_cum hist_all cells.svg'),format = 'svg')


p3 = (
    so.Plot(df_plot, x = 'sigma', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Line(), so.Hist(stat = 'probability', common_norm = False, cumulative = True),
         legend = False)
    # .limit(x = (0,3))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,6), count=5))
    .theme(style)
    .label(y = 'Proportion of neurons', x = 'STD of response')
    .share(y = False)
    .plot()
    )


for ai,a in enumerate(p3._figure.axes):
    # if ai == 0:
    #     a.set_ylim(0,0.15)
    #     a.set_yticks(np.linspace(0,0.15,4))
    # else:
    #     a.set_ylim(0,0.3)
    #     a.set_yticks(np.linspace(0,0.3,3))
    a.set_xticks(np.linspace(0,6,7))
    sns.despine(ax=a,trim=True)

p3.save(join(fig_save,'stim decoding_45 and 90_dist of std of resp_cum hist_all cells.svg'),format = 'svg')


p4 = (
    so.Plot(df_plot, x = 'd_prime', color = 'trained')
    .layout(size = fig_size, engine = 'tight')
    .add(so.Line(), so.Hist(stat = 'probability', common_norm = False, cumulative = True),
         legend = False)
    # .limit(x = (0,3))
    .scale(color = 'colorblind',
            x = so.Continuous().tick(between=(0,19), count=5))
    .theme(style)
    .label(y = 'Proportion of neurons', x = 'd-prime')
    .share(y = False)
    .plot()
    )

for ai,a in enumerate(p4._figure.axes):
   
    # a.set_ylim(0,0.5)
    # a.set_yticks(np.linspace(0,0.45,6))
    a.set_xticks(np.linspace(0,19,3))
   
    sns.despine(ax=a,trim=True)


p4.save(join(fig_save,'stim decoding_45 and 90_dist of dprime_cum hist_all cells.svg'),format = 'svg')


#%% d-prime vs ori pref with model contribution

ind = df_weights.subject == 'SF180515'

df_plot = df_weights[ind].copy()
df_plot.sort_values(['ave_rel_cont_resp'], ascending = True, inplace = True)

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
                                        "lines.markersize":1.5,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams['figure.dpi'] = 1200


    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_size weight.svg'),format='svg')


    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = False,
    #             ec = 'k')

    f,ax = plt.subplots(1,2,figsize=(2.5,1.25),sharey=False,sharex=False)

    # sns.scatterplot(data = df_weights[df_weights.trained==False], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[0], palette = 'magma_r',
    #                       hue_norm = (0,0.1), rasterized=True)

    # sns.scatterplot(data = df_weights[df_weights.trained==True], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[1], palette = 'magma_r',
    #                       hue_norm=(0,0.1), rasterized=True)

    sns.scatterplot(data = df_plot[df_plot.trained==False],
                    x = 'd_prime',
                    y = 'ori_pref', hue = 'ave_rel_cont_resp',ec = 'k',
                    legend = False, ax = ax[0], palette = 'magma_r',
                    # vmin = -0.3, vmax = 1,
                    hue_norm = (0,1),
                    rasterized=True)

    sns.scatterplot(data = df_plot[df_plot.trained==True],
                    x = 'd_prime',
                    y = 'ori_pref', hue = 'ave_rel_cont_resp',ec = 'k',
                    legend = False, ax = ax[1], palette = 'magma_r',
                    #vmin = -0.3, vmax = 1,
                    hue_norm = (0,1),
                    rasterized=True)

    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
    sm.set_array([])

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.825, 0.22, 0.008, 0.5])
    plt.colorbar(sm, cax = cbar_ax, label = 'Contribution to dec. fun.')

    # ax[0].set_ylabel('Selectivity')
    # ax[0].set_xlabel('Coef. of variation')
    # ax[1].set_xlabel('Coef. of variation')

    ax[0].set_xlabel(r'$d\prime$')
    ax[0].set_ylabel('Mean ori. pref.')

    ax[1].set_xlabel(r'$d\prime$')

    ax[1].set_ylabel('')
    ax[1].set_yticks([])

    ax[0].set_title('Naive')
    ax[1].set_title('Proficient')

    ax[0].set_xlim([-15,15])
    ax[1].set_xlim([-15,15])

    ax[0].set_xticks(np.linspace(-15,15,5))
    ax[1].set_xticks(np.linspace(-15,15,5))

    ax[0].set_ylim([-5,185])
    ax[1].set_ylim([-5,185])

    ax[0].set_yticks(np.ceil(np.linspace(0,180,9)).astype(int))
    ax[1].set_yticks([])


    sns.despine(trim=True,ax=ax[0], offset = 2)
    sns.despine(trim=True, left = True, ax=ax[1], offset = 2)

    for a in ax:
        a.set_box_aspect(1)

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = True,
    #             ec = 'k')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue weight.svg'),format='svg')


#%% resp index vs ori pref with model contribution

ind = df_weights.subject == 'SF180515'

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
                                        "lines.markersize":0.80,
                                        "lines.linewidth":0.5,
                                        "legend.fontsize":5,
                                        "legend.title_fontsize":5}):

    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams['figure.dpi'] = 1000


    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'pref_bin', size = 'rel_weight',
    #             x = 'cv', y = 'selectivity', palette = 'hls',
    #             legend = False, ec = 'k')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue ori pref_size weight.svg'),format='svg')


    # sns.relplot(data = df_weights, col = 'subject', row = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = False,
    #             ec = 'k')

    f,ax = plt.subplots(1,2,figsize=(2.5,1.25),sharey=False,sharex=False)

    # sns.scatterplot(data = df_weights[df_weights.trained==False], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[0], palette = 'magma_r',
    #                       hue_norm = (0,0.1), rasterized=True)

    # sns.scatterplot(data = df_weights[df_weights.trained==True], x = 'cv',
    #                       y = 'selectivity', hue = 'ave_prop_resp',ec = 'k',
    #                       legend = False, ax = ax[1], palette = 'magma_r',
    #                       hue_norm=(0,0.1), rasterized=True)

    sns.scatterplot(data = df_weights[np.logical_and(ind,df_weights.trained==False)],
                    x = 'resp_index',
                    y = 'ori_pref', hue = 'ave_rel_cont_resp',ec = 'k',
                    legend = False, ax = ax[0], palette = 'magma_r',
                    # vmin = -0.3, vmax = 1,
                    hue_norm = (0,1),
                    rasterized=True)

    sns.scatterplot(data = df_weights[np.logical_and(ind,df_weights.trained==True)],
                    x = 'resp_index',
                    y = 'ori_pref', hue = 'ave_rel_cont_resp',ec = 'k',
                    legend = False, ax = ax[1], palette = 'magma_r',
                    #vmin = -0.3, vmax = 1,
                    hue_norm = (0,1),
                    rasterized=True)

    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
    sm.set_array([])

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.825, 0.22, 0.008, 0.5])
    plt.colorbar(sm, cax = cbar_ax, label = 'Contribution to dec. fun.')

    # ax[0].set_ylabel('Selectivity')
    # ax[0].set_xlabel('Coef. of variation')
    # ax[1].set_xlabel('Coef. of variation')

    ax[0].set_xlabel('Response index')
    ax[0].set_ylabel('Mean ori. pref.')

    ax[1].set_xlabel('Response index')

    ax[1].set_ylabel('')
    ax[1].set_yticks([])

    ax[0].set_title('Naive')
    ax[1].set_title('Proficient')

    ax[0].set_xlim([-1.2,1.2])
    ax[1].set_xlim([-1.2,1.2])

    ax[0].set_xticks(np.linspace(-1.2,1.2,5))
    ax[1].set_xticks(np.linspace(-1.2,1.2,5))

    ax[0].set_ylim([0,180])
    ax[1].set_ylim([0,180])


    sns.despine(trim=True,offset=2,ax=ax[0])
    sns.despine(trim=True,offset=2,left = True, ax=ax[1])

    for a in ax:
        a.set_box_aspect(1)

    # sns.relplot(data = df_weights, col = 'trained',
    #             kind = 'scatter', hue = 'rel_weight',
    #             x = 'cv', y = 'selectivity', legend = True,
    #             ec = 'k')

    # plt.savefig(join(fig_save,'stim decoding_45 and 90_cv by selectivity_hue weight.svg'),format='svg')


#%% sparseness of weights

weight_sparseness = np.zeros(len(subjects))

c = 0
for s in df_weights.subject.unique():
    for t in df_weights.trained.unique():
        ind = np.logical_and(df_weights.subject==s,df_weights.trained==t)

        weight_sparseness[c] = kurtosis(np.abs(df_weights[ind].model_weight))
        c += 1


df_sparseness = pd.DataFrame({'sparseness' : weight_sparseness,
                              'subject' : subjects,
                              'trained' : trained})


sns.catplot(data = df_sparseness, x = 'trained', y = 'sparseness')


#%% Proportion of of weights above 1, 5, 10, 100

weight_sparseness = np.zeros(len(subjects))

df_sparseness = pd.DataFrame(columns = ['above_1','above_5','above_10',
                                        'above_100', 'subject', 'trained'])

c = 0
for s in df_weights.subject.unique():
    for t in df_weights.Trained.unique():
        ind = np.logical_and(df_weights.subject==s,df_weights.Trained==t)


        d = {'above_' + str(i) : np.sum(df_weights[ind].model_weight > i)/np.sum(ind)
             for i in [1,5,10,100]}

        d['subject'] = s
        d['trained'] = t

        df_sparseness = pd.concat([df_sparseness,pd.DataFrame(d,index = None)])





sns.catplot(data = df_sparseness, x = 'trained', y = 'sparseness')


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

