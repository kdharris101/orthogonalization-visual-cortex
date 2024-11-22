#%%
import numpy as np
import pandas as pd
from os.path import join
import glob
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.special import expit


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

computer = 'lab'

if computer == 'home':
    results_root = r'C:\Users\Samuel\OneDrive - University College London\Results'
    save_fig_dir = r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft'
elif computer == 'lab':
    results_root = r'C:\Users\samue\OneDrive - University College London\Results'
    save_fig_dir = r'C:\Users\samue\OneDrive - University College London\Orthogonalization project\Figures\Draft'


subjects_file = ['M170620B_SF', 'SF170620B', 'M170905B_SF', 'M170905B_SF', 'M171107_SF', 'SF171107', 'SF180515', 'SF180515','SF180613','SF180613']

expt_dates = ['2017-07-04', '2017-12-21', '2017-09-21', '2017-11-26', '2017-11-17', '2018-04-04', '2018-06-13', '2018-09-21', '2018-06-28', '2018-12-12']

expt_nums = [5, 7, 8, 1, 7, 3, 1, 3, 1, 5]

expt_files = [glob.glob(join(results_root, subjects_file[i], expt_dates[i], 
              str(expt_nums[i]), f'df_resps_{subjects_file[i]}_{expt_dates[i]}_{expt_nums[i]}.feather'))[-1] 
                    for i in range(len(subjects_file))]

match_subjects = ['SF170620B','SF170905B', 'SF180515', 'SF180613']


match_files = [r'\\zubjects.cortexlab.net\Subjects\M170620B_SF\2017-07-04\suite2p\matches_for_M170620B_SF_naive_SF170620B_proficient.npy',
               r'\\zubjects.cortexlab.net\Subjects\M170905B_SF\2017-09-21\suite2p\matches_for_M170905B_SF_naive_SF170905B_proficient.npy',
               r'\\zubjects.cortexlab.net\Subjects\SF180515\2018-06-13\suite2p\matches_for_SF180515_naive_SF180515_proficient.npy',
               r'\\zubjects.cortexlab.net\Subjects\SF180613\2018-06-28\suite2p\matches_for_SF180613_naive_SF180613_proficient.npy']


fig_save_dir = r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft'

#%% 

df_resps = pd.DataFrame()

for e in expt_files:

    print(f'loading {e}')

    df_expt = pd.read_feather(e)
    
    # Only keep curated cells
    df_expt = df_expt[df_expt.iscell]
        
    df_resps = pd.concat([df_resps, df_expt], ignore_index=True)
    


#%% Find matched cells

# Compress to just cell attributes
df_cells = df_resps.groupby(['subject', 'cell_num', 'cell_plane', 'condition'], observed=True, group_keys=True).first().reset_index()[['subject', 'cell_num', 'cell_plane', 'condition']]

df_cells['tracked'] = False
df_cells['cell_num_tracked'] = np.nan

for i,m in enumerate(match_files):

    print(f'Loading {m}')    
    matches = np.load(m, allow_pickle=True)[()]['matches']
    matches = matches[matches.true_match]
    
    # Remove any possible duplicates
    matches = matches.sort_values('overlap').drop_duplicates(matches.columns[[0,1]], keep='last', ignore_index=True)
    matches = matches.sort_values('overlap').drop_duplicates(matches.columns[[2,3]], keep='last', ignore_index=True)

    naive_rois = np.concatenate(matches.iloc[:,[0]].to_numpy())
    naive_planes = np.concatenate(matches.iloc[:,[1]].to_numpy())
      
    proficient_rois = np.concatenate(matches.iloc[:,[2]].to_numpy())
    proficient_planes = np.concatenate(matches.iloc[:,[3]].to_numpy())  
    
    ind_n = (df_cells.subject==match_subjects[i]) & (df_cells.condition=='naive')
    ind_p = (df_cells.subject==match_subjects[i]) & (df_cells.condition=='proficient')
    
    for c,(nr,npl,pr,ppl) in enumerate(zip(naive_rois, naive_planes, proficient_rois, proficient_planes)):
               
        ind_n_c = (df_cells.cell_num==nr) & (df_cells.cell_plane==npl) & ind_n
        ind_p_c = (df_cells.cell_num==pr) & (df_cells.cell_plane==ppl) & ind_p
        
        if ind_n_c.sum() == 1 and ind_p_c.sum() == 1:
                        
            df_cells.loc[ind_n_c, 'tracked'] = True
            df_cells.loc[ind_n_c, 'cell_num_tracked'] = c
            
            df_cells.loc[ind_p_c, 'tracked'] = True
            df_cells.loc[ind_p_c, 'cell_num_tracked'] = c
            
        else:
            print(f'Something went wrong for pair {c} for in match file {m}')
            

# Expand to fill trial response dataframe
df_resps = df_resps.merge(df_cells, on = ['subject', 'cell_num', 'cell_plane', 'condition'], how = 'left')


#%%

# Save original in case you need to restore it
df_resps_original = df_resps.copy()

# Scale responses by preferred stimulus
pref_resp = df_resps.groupby(['cell_num','cell_plane','stim_ori','condition', 'subject'], observed = True).mean().reset_index()
# pref_stim = pref_resp.loc[pref_resp.groupby(['cell_num','cell_plane','stim_ori','condition','subject'],observed=True)['trial_resps'].idxmax(),'stim_ori'].reset_index(drop=True)
pref_resp = pref_resp.groupby(['cell_num','cell_plane','condition','subject'], observed = True)['trial_resps'].max().reset_index()
pref_resp = pref_resp.rename(columns={'trial_resps':'pref_resp'})

df_resps = df_resps.merge(pref_resp, on =['cell_num','cell_plane','condition', 'subject'], how='left')

df_resps['trial_resps_raw'] = df_resps['trial_resps']

df_resps['trial_resps'] = df_resps['trial_resps']/df_resps['pref_resp']


#%% Function to calculate ori tuning

def find_tuning(df_resps, train_only = True):

    ind = ~np.isinf(df_resps.stim_ori)

    if train_only:
        ind = df_resps.train_ind & ind

    df_tuning = df_resps[ind].copy()
    df_tuning['stim_ori_rad'] = df_tuning['stim_ori'] * np.pi/90
    df_tuning['stim_dir_rad'] = df_tuning['stim_dir'] * np.pi/180
        
    df_tuning['exp(stim_ori_rad)*trial_resp'] = np.exp(df_tuning.stim_ori_rad*1j) * df_tuning.trial_resps
    df_tuning['exp(stim_dir_rad)*trial_resp'] = np.exp(df_tuning.stim_dir_rad*1j) * df_tuning.trial_resps

    df_tuning = df_tuning.groupby(['cell_num', 'cell_plane', 'subject', 'condition'], observed = True).agg({'exp(stim_ori_rad)*trial_resp' : 'sum',
                                                                                                            'exp(stim_dir_rad)*trial_resp' : 'sum',
                                                                                                            'trial_resps' : 'sum',
                                                                                                            'ROI_ret_azi' : 'first',
                                                                                                            'ROI_ret_elv' : 'first',
                                                                                                            'cell_num_tracked' : 'first',
                                                                                                            'tracked' : 'first',
                                                                                                            'V1_ROI' : 'first'})

    df_tuning['ori_tune_vec'] = df_tuning['exp(stim_ori_rad)*trial_resp']/df_tuning['trial_resps']
    df_tuning['dir_tune_vec'] = df_tuning['exp(stim_dir_rad)*trial_resp']/df_tuning['trial_resps']
    df_tuning['r_ori'] = df_tuning.ori_tune_vec.abs()
    df_tuning['r_dir'] = df_tuning.dir_tune_vec.abs()
    df_tuning['mean_ori_pref'] = np.mod(11.25+np.arctan2(np.imag(df_tuning.ori_tune_vec),np.real(df_tuning.ori_tune_vec))*90/np.pi,180)-11.25
    df_tuning['mean_dir_pref'] = np.mod(11.25+np.arctan2(np.imag(df_tuning.dir_tune_vec),np.real(df_tuning.dir_tune_vec))*180/np.pi,360)-11.25


    df_tuning = df_tuning.drop(columns = ['exp(stim_ori_rad)*trial_resp','trial_resps'])
    df_tuning = df_tuning.reset_index()

    mu_resp = df_resps[~np.isinf(df_resps.stim_ori)].groupby(['cell_num', 'cell_plane', 'subject', 'condition', 'stim_ori'], observed = True)['trial_resps'].mean().reset_index()
    df_tuning['modal_ori_pref'] = mu_resp.loc[mu_resp.groupby(['cell_num', 'cell_plane', 'subject', 'condition'], observed = True)['trial_resps'].idxmax(),'stim_ori'].reset_index(drop=True)
    mu_resp = df_resps[~np.isinf(df_resps.stim_dir)].groupby(['cell_num', 'cell_plane', 'subject', 'condition', 'stim_dir'], observed = True)['trial_resps'].mean().reset_index()
    df_tuning['modal_dir_pref'] = mu_resp.loc[mu_resp.groupby(['cell_num', 'cell_plane', 'subject', 'condition'], observed = True)['trial_resps'].idxmax(),'stim_dir'].reset_index(drop=True)

    return df_tuning


#%% Cell tuning

df_tuning = find_tuning(df_resps)

# Bin cells by mean preference and selectivity (r)
df_tuning['pref_bin_ori'] = pd.cut(df_tuning.mean_ori_pref, np.linspace(-11.25,180-11.25,9), labels=np.ceil(np.arange(0,180,22.5)))
df_tuning['r_bin_ori'] = pd.cut(df_tuning.r_ori, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels=np.arange(5))

# Tuning curves from train and test set
tuning_curves = df_resps.groupby(['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori', 'train_ind'], observed = True)['trial_resps'].mean().reset_index()
# Exclude training set and blank trials
tuning_curves_test = tuning_curves[~tuning_curves.train_ind & ~np.isinf(tuning_curves.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp'})
tuning_curves_train = tuning_curves[tuning_curves.train_ind & ~np.isinf(tuning_curves.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp_train'})

tuning_curves_all = df_resps.groupby(['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori'], observed = True)['trial_resps'].mean().reset_index()
tuning_curves_all = tuning_curves_all[~np.isinf(tuning_curves_all.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp_all_trials'})

df_tuning = df_tuning.merge(tuning_curves_train, on = ['cell_num', 'cell_plane', 'condition', 'subject'], how = 'left').drop(columns='train_ind')
df_tuning = df_tuning.merge(tuning_curves_test, on = ['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori'], how = 'left').drop(columns = 'train_ind')
df_tuning = df_tuning.merge(tuning_curves_all, on = ['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori'], how = 'left')

# df_tuning = df_tuning.astype({'pref_bin_naive' : int, 'r_bin_naive' : int, 'pref_bin_proficient' : int, 'r_bin_proficient' : int})


#%% Tracked change in modal pref, by selectivity

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

df_prefs = df_tuning[df_tuning.tracked].copy().set_index(['subject','cell_num_tracked'])

# Remove cells whose ROIs are not in V1 in both recordings
not_in_V1 = df_prefs.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_prefs = df_prefs.drop(not_in_V1).reset_index()

naive_prefs = df_prefs[df_prefs.condition=='naive'][['modal_ori_pref','r_bin_ori','cell_num_tracked','subject']]
naive_prefs = naive_prefs.rename({'modal_ori_pref' : 'modal_pref_naive',
                                  'r_bin_ori' : 'r_bin_naive'}, axis=1).reset_index(drop=True)

proficient_prefs = df_prefs[df_prefs.condition=='proficient'][['modal_ori_pref','cell_num_tracked','subject']]
proficient_prefs = proficient_prefs.rename({'modal_ori_pref' : 'modal_pref_proficient'}, axis=1).reset_index(drop=True)

df_prefs = df_prefs.merge(naive_prefs, on=['subject','cell_num_tracked'], how='left')
df_prefs = df_prefs.merge(proficient_prefs, on=['subject', 'cell_num_tracked'], how='left')

df_prefs = df_prefs.groupby(['subject','cell_num_tracked'])[['modal_pref_naive', 'modal_pref_proficient', 'r_bin_naive']].first().reset_index()

naive_props = df_prefs.groupby('r_bin_naive').modal_pref_naive.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'modal_pref_naive' : 'proportion'}, axis=1).reset_index()
proficient_props = df_prefs.groupby('r_bin_naive').modal_pref_proficient.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'modal_pref_proficient' : 'proportion'}, axis=1).reset_index()

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


    f0 = plt.figure(figsize=(4,1.25))

    n_r_bins = len(naive_props.r_bin_naive.unique())

    min_shade = 1


    for r in range(n_r_bins):

        df_change = df_prefs[df_prefs.r_bin_naive==r].drop(['subject','cell_num_tracked','r_bin_naive'], axis=1)

        df_change = df_change.groupby('modal_pref_naive').value_counts(normalize=True).reset_index()
        df_change.rename({0 : 'Prop'}, axis=1, inplace=True)
        
        df_change.loc[(df_change.modal_pref_naive==0) & (df_change.modal_pref_proficient>90), 'modal_pref_naive'] = 180
        df_change.loc[(df_change.modal_pref_proficient==0) & (df_change.modal_pref_naive>90), 'modal_pref_proficient'] = 180 

        df_0 = df_change[(df_change.modal_pref_naive==0) & (df_change.modal_pref_proficient==0)].copy()
        df_0['modal_pref_naive'] = 180
        df_0['modal_pref_proficient'] = 180
        
        df_change = pd.concat([df_change, df_0], ignore_index=True)

        x = np.concatenate([np.zeros(len(naive_props[naive_props.r_bin_naive==r])), np.ones(len(naive_props[naive_props.r_bin_naive==r]))])
        y = list(naive_props.modal_pref_naive.unique())*2

        df_plot = pd.DataFrame({'x' : x,
                                'y' : y,
                                'prop' : np.concatenate([naive_props[naive_props.r_bin_naive==r].proportion,
                                                         proficient_props[proficient_props.r_bin_naive==r].proportion])*100})
        
        df_180 = df_plot[df_plot.y==0].copy()
        df_180['y'] = 180
        
        df_plot = pd.concat([df_plot,df_180], ignore_index=True)
        
        a0 = f0.add_subplot(1, n_r_bins, r+1)

        (
            so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'y')
            .layout(engine='tight')
            .add(so.Dot(edgecolor='black', edgewidth=0.5), legend=False)
            .scale(y=so.Continuous().tick(at=np.ceil(np.linspace(0,180,9))),
                   x=so.Continuous().tick(at=[0,1]),
                   pointsize=(1,8),
                   color = 'hls')
            .label(y='Modal orientation preference',
                x='')
            .limit(x=(-0.5,1.5),
                   y=(-20,200))
            .theme({**style})
            .on(a0)
            .plot()
        )

        for i in range(len(df_change)):
            if df_change.loc[i, 'Prop'] > 0:
                # shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
                if df_change.loc[i, 'modal_pref_naive'] == df_change.loc[i, 'modal_pref_proficient']:
                    shade = 'xkcd:dark grey'
                    zorder = 0
                else:
                    shade = 'xkcd:light grey'
                    zorder = -1
                
                a0.plot([0,1],
                    [df_change.loc[i, 'modal_pref_naive'], 
                    df_change.loc[i, 'modal_pref_proficient']],
                    linewidth=df_change.loc[i, 'Prop']*3,
                    # color = np.where(shade>min_shade, min_shade, shade),
                    color = shade,
                    zorder = zorder)

        a0.set_xticklabels(['Naive','Proficient'])

        if r == 0:
            sns.despine(ax=a0, trim=True)
        else:
            sns.despine(ax=a0, trim=True, left=True)
            a0.set_yticks([])
            a0.set_ylabel('')


    # f0.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\tracked_modal_pref.svg', format='svg')


#%% Change in mean response to 45, 68, non-task

import statsmodels.api as sm
import statsmodels.formula.api as smf


# TRACKED

df_plot = df_tuning[df_tuning.tracked].copy().set_index(['subject','cell_num_tracked'])

not_in_V1 = df_plot.groupby(['subject','cell_num_tracked'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_plot = df_plot.drop(not_in_V1).reset_index()

df_n = df_plot[df_plot.condition=='naive'].sort_values(['subject','cell_num_tracked']).reset_index(drop=True)
df_p = df_plot[df_plot.condition=='proficient'].sort_values(['subject','cell_num_tracked']).reset_index(drop=True)

df_diff = df_n.copy()
df_diff['resp_diff'] = df_p.mean_resp - df_n.mean_resp

df_diff['stim_type'] = df_diff.stim_ori.apply(lambda x: '45/90' if x==45 or x==90 else '68' if x==68 else 'non-task')

df_diff = df_diff.groupby(['subject','condition','cell_num_tracked','stim_type'])['resp_diff'].mean().reset_index()


#45/90 vs 68
df_diff_45_68 = df_diff[(df_diff.stim_type == '45/90') | (df_diff.stim_type=='68')]

md = smf.mixedlm("resp_diff ~ C(stim_type)", df_diff_45_68, groups=df_diff_45_68['subject'])

mdf = md.fit()

# print(mdf.summary())

# 45/90 vs nontask
df_diff_45_nontask = df_diff[(df_diff.stim_type == '45/90') | (df_diff.stim_type=='non-task')]

md = smf.mixedlm("resp_diff ~ C(stim_type)", df_diff_45_nontask, groups=df_diff_45_nontask['subject'])

mdf = md.fit()

# print(mdf.summary())

# 68 vs non-task

df_diff_68_nontask = df_diff[(df_diff.stim_type == '68') | (df_diff.stim_type=='non-task')]

md = smf.mixedlm("resp_diff ~ C(stim_type)", df_diff_68_nontask, groups=df_diff_68_nontask['subject'])

mdf = md.fit()

# print(mdf.summary())

df_plot = df_diff.groupby(['subject','stim_type'])['resp_diff'].mean().reset_index()

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

    f0,a0 = plt.subplots(1,1, figsize=(0.75,0.75))
    
    y_ticks= [-0.25, -0.15, -0.05, 0.05]

    # limits = best_limits(y_ticks)    
    
    (
        so.Plot(df_plot, x='stim_type', y='resp_diff')
        .add(so.Dots(pointsize=3, color='black'), so.Jitter(), legend=False)
        .add(so.Range(linewidth=0.5, color='black'), so.Est(errorbar=('se',1)), legend=False)
        .add(so.Dash(linewidth=0.5, width=0.5, linestyle='dashed', color='black'), so.Agg(), legend=False)
        .scale(y = so.Continuous().tick(at=y_ticks))
        # .limit(y=limits[0])
        .on(a0)
        .label(x='Stimulus orientation', y='Change in response')
        .plot()
    )
    
    sns.despine(ax=a0, trim=True)
    
    a0_lim = print(a0.get_ylim())


# Change averaged across all neurons per subject

df_plot = df_tuning.copy()

df_plot = df_plot[(df_plot.cell_plane>0) & (df_plot.V1_ROI==1)]

df_plot = df_plot.groupby(['subject', 'condition', 'stim_ori'])['mean_resp'].mean().reset_index()

df_plot_diff = df_plot[df_plot.condition=='naive'].reset_index(drop=True)

df_plot_diff['resp_diff'] = df_plot.groupby(['subject','stim_ori'])['mean_resp'].diff().dropna().reset_index(drop=True)

df_plot_diff['stim_type'] = ['45/90' if s==45 or s==90 else '68' if s==68 else 'non-task' for s in df_plot_diff.stim_ori.to_numpy()]

df_plot_diff = df_plot_diff.groupby(['subject','condition','stim_type'])['resp_diff'].mean().reset_index()

# stats

ttest_rel(df_plot_diff[df_plot_diff.stim_type=='45/90'].resp_diff,
          df_plot_diff[df_plot_diff.stim_type=='68'].resp_diff)

ttest_rel(df_plot_diff[df_plot_diff.stim_type=='45/90'].resp_diff,
          df_plot_diff[df_plot_diff.stim_type=='non-task'].resp_diff)

ttest_rel(df_plot_diff[df_plot_diff.stim_type=='68'].resp_diff,
          df_plot_diff[df_plot_diff.stim_type=='non-task'].resp_diff)

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
    
    y_ticks= [-0.25, -0.15, -0.05, 0.05]

    # limits = best_limits(y_ticks)

    f0,a1 = plt.subplots(1,1, figsize=(0.75,0.75))
    
    (
        so.Plot(df_plot_diff, x='stim_type', y='resp_diff')
        .add(so.Dots(pointsize=3, color='black'), so.Jitter(), legend=False)
        .add(so.Range(linewidth=0.5, color='black'), so.Est(errorbar=('se',1)), legend=False)
        .add(so.Dash(linewidth=0.5, width=0.5, linestyle='dashed', color='black'), so.Agg(), legend=False)
        .scale(y = so.Continuous().tick(at=y_ticks))
        # .limit(y=limits[0])
        .on(a1)
        .label(x='Stimulus orientation', y='Change in response')
        .plot()
    )
    
    sns.despine(ax=a1, trim=True)


#%% Change in mean response to 45, 68, non-task



# Tracked cells

df_plot = df_tuning[df_tuning.tracked].copy().set_index(['subject','cell_num_tracked'])

# Remove cells whose ROIs are not in V1 in both recordings
not_in_V1 = df_plot.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_plot = df_plot.drop(not_in_V1).reset_index()

df_plot = df_plot.groupby(['subject','cell_num_tracked','stim_ori','condition'])['mean_resp'].mean().reset_index()

df_plot = df_plot[df_plot.condition=='naive'].reset_index(drop=True)

df_plot['stim_type'] = df_plot.stim_ori.apply(lambda x: '45/90' if x==45 or x==90 else '68' if x==68 else 'non-task')

df_plot_diff['resp_diff'] = df_plot_diff.groupby(['subject','cell_num_tracked','stim_type'])['mean_resp'].diff().dropna().reset_index(drop=True)

df_plot_diff = df_plot_diff.groupby(['subject','stim_type'])['resp_diff'].mean().reset_index()




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

    f0,a0 = plt.subplots(1,1, figsize=(0.75,0.75))
    
    y_ticks= [-0.25, -0.15, -0.05, 0.05]

    limits = best_limits(y_ticks)    
    
    (
        so.Plot(df_plot_diff, x='stim_type', y='resp_diff')
        .add(so.Dots(pointsize=3, color='black'), so.Jitter(), legend=False)
        .add(so.Range(linewidth=0.5, color='black'), so.Est(errorbar=('se',1)), legend=False)
        .add(so.Dash(linewidth=0.5, width=0.5, linestyle='dashed', color='black'), so.Agg(), legend=False)
        .scale(y = so.Continuous().tick(at=y_ticks))
        .limit(y=limits[0])
        .on(a0)
        .label(x='Stimulus orientation', y='Change in response')
        .plot()
    )
    
    sns.despine(ax=a0, trim=True)
    
    a0_lim = print(a0.get_ylim())
    
# Change averaged across all neurons per subject

df_plot = df_tuning.copy()

df_plot = df_plot[(df_plot.cell_plane>0) & (df_plot.V1_ROI==1)]

df_plot['stim_type'] = ['45/90' if s==45 or s==90 else '68' if s==68 else 'non-task' for s in df_plot.stim_ori.to_numpy()]

df_plot = df_plot.groupby(['subject','condition','stim_type'])['mean_resp'].mean().reset_index()

df_plot_diff = df_plot[df_plot.condition=='naive'].reset_index(drop=True)

df_plot_diff['resp_diff'] = df_plot.groupby(['subject','stim_type'])['mean_resp'].diff().dropna().reset_index(drop=True)

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
    
    y_ticks= [-0.25, -0.15, -0.05, 0.05]

    limits = best_limits(y_ticks)

    f0,a1 = plt.subplots(1,1, figsize=(0.75,0.75))
    
    (
        so.Plot(df_plot_diff, x='stim_type', y='resp_diff')
        .add(so.Dots(pointsize=3, color='black'), so.Jitter(), legend=False)
        .add(so.Range(linewidth=0.5, color='black'), so.Est(errorbar=('se',1)), legend=False)
        .add(so.Dash(linewidth=0.5, width=0.5, linestyle='dashed', color='black'), so.Agg(), legend=False)
        .scale(y = so.Continuous().tick(at=y_ticks))
        .limit(y=limits[0])
        .on(a1)
        .label(x='Stimulus orientation', y='Change in response')
        .plot()
    )
    
    sns.despine(ax=a1, trim=True)

#%% Tracked change in mean pref, by selectivity
style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False


df_prefs = df_tuning[df_tuning.tracked].copy().set_index(['subject','cell_num_tracked'])

# Remove cells whose ROIs are not in V1 in both recordings
not_in_V1 = df_prefs.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_prefs = df_prefs.drop(not_in_V1).reset_index()

naive_prefs = df_prefs[df_prefs.condition=='naive'][['pref_bin_ori','r_bin_ori','cell_num_tracked','subject']]
naive_prefs = naive_prefs.rename({'pref_bin_ori' : 'pref_bin_naive',
                                  'r_bin_ori' : 'r_bin_naive'}, axis=1).reset_index(drop=True)

proficient_prefs = df_prefs[df_prefs.condition=='proficient'][['pref_bin_ori','cell_num_tracked','subject']]
proficient_prefs = proficient_prefs.rename({'pref_bin_ori' : 'pref_bin_proficient'}, axis=1).reset_index(drop=True)

df_prefs = df_prefs.merge(naive_prefs, on=['subject','cell_num_tracked'], how='left')
df_prefs = df_prefs.merge(proficient_prefs, on=['subject', 'cell_num_tracked'], how='left')

df_prefs = df_prefs.groupby(['subject','cell_num_tracked'])[['pref_bin_naive', 'pref_bin_proficient', 'r_bin_naive']].first().reset_index()

naive_props = df_prefs.groupby('r_bin_naive').pref_bin_naive.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'pref_bin_naive' : 'proportion'}, axis=1).reset_index()
proficient_props = df_prefs.groupby('r_bin_naive').pref_bin_proficient.value_counts(normalize=True).sort_index().reset_index(level=0).rename({'pref_bin_proficient' : 'proportion'}, axis=1).reset_index()

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


    f0 = plt.figure(figsize=(4.5,1.5))

    n_r_bins = len(naive_props.r_bin_naive.unique())

    min_shade = 1


    for r in range(n_r_bins):

        df_change = df_prefs[df_prefs.r_bin_naive==r].drop(['subject','cell_num_tracked','r_bin_naive'], axis=1)

        df_change = df_change.groupby('pref_bin_naive').value_counts(normalize=True).reset_index()
        df_change.rename({0 : 'Prop'}, axis=1, inplace=True)

        x = np.concatenate([np.zeros(len(naive_props[naive_props.r_bin_naive==r])), np.ones(len(naive_props[naive_props.r_bin_naive==r]))])
        y = list(df_change.pref_bin_naive.unique())*2

        df_plot = pd.DataFrame({'x' : x,
                                'y' : y,
                                'prop' : np.concatenate([naive_props[naive_props.r_bin_naive==r].proportion,
                                                        proficient_props[proficient_props.r_bin_naive==r].proportion])*100})
        
        a0 = f0.add_subplot(1,n_r_bins,r+1)

        (
            so.Plot(df_plot, x='x', y='y', pointsize='prop', color = 'y')
            .layout(engine='tight')
            .add(so.Dot(), legend=False)
            .scale(y=so.Continuous().tick(at=np.ceil(np.linspace(0,180-22.5,8))),
                x=so.Continuous().tick(at=[0,1]),
                pointsize=(2,10),
                color = 'hls')
            .label(y='Mean orientation preference',
                x='')
            .limit(x=(-0.5,1.5))
            .theme({**style})
            .on(a0)
            .plot()
        )

        for i in range(len(df_change)):
            if df_change.loc[i, 'Prop'] > 0:
                shade = np.array([1,1,1])-df_change.loc[i, 'Prop']
                a0.plot([0,1],
                    [df_change.loc[i, 'pref_bin_naive'], 
                    df_change.loc[i, 'pref_bin_proficient']],
                    linewidth=df_change.loc[i, 'Prop']*3,
                    color = np.where(shade>min_shade, min_shade, shade),
                    zorder = 0)

        a0.set_xticklabels(['Naive','Proficient'])

        if r == 0:
            sns.despine(ax=a0, trim=True)
        else:
            sns.despine(ax=a0, trim=True, left=True)
            a0.set_yticks([])
            a0.set_ylabel('')


    f0.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\tracked_mean_pref.svg', format='svg')
#%% Stability of modal ori pref vs mean ori pref

df_prefs = df_tuning[df_tuning.tracked].copy().set_index(['subject','cell_num_tracked'])

# Remove cells whose ROIs are not in V1 at least one recordings
not_in_V1 = df_prefs.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_prefs = df_prefs.drop(not_in_V1).reset_index()

df_prefs = df_prefs.groupby(['subject','cell_num_tracked','condition']).agg({'modal_ori_pref' : 'first',
                                                                             'pref_bin_ori' : 'first',
                                                                             'r_bin_ori' : 'first'}).reset_index()

naive_prefs = df_prefs[df_prefs.condition=='naive'][['modal_ori_pref', 'pref_bin_ori', 'r_bin_ori','cell_num_tracked','subject']]
naive_prefs = naive_prefs.rename({'modal_ori_pref' : 'modal_pref_naive',
                                  'pref_bin_ori' : 'pref_bin_naive',
                                  'r_bin_ori' : 'r_bin_naive'}, axis=1).reset_index(drop=True)

proficient_prefs = df_prefs[df_prefs.condition=='proficient'][['modal_ori_pref','pref_bin_ori','cell_num_tracked','subject']]
proficient_prefs = proficient_prefs.rename({'modal_ori_pref' : 'modal_pref_proficient',
                                            'pref_bin_ori' : 'pref_bin_proficient'}, axis=1).reset_index(drop=True)

df_prefs = df_prefs.merge(naive_prefs, on=['subject','cell_num_tracked'], how='left')
df_prefs = df_prefs.merge(proficient_prefs, on=['subject', 'cell_num_tracked'], how='left')

df_plot = df_prefs[df_prefs.condition=='naive'].copy().reset_index().drop(columns=['condition', 'modal_ori_pref', 'pref_bin_ori', 'r_bin_ori'])

df_plot['modal_pref_naive'] = df_plot.modal_pref_naive.astype('category')
df_plot['modal_pref_proficient'] = df_plot.modal_pref_proficient.astype('category')


# For each subject and selectivity level, make a transition matrix (normalized by sum of columns) and look at probabilities cells stay preferring 45/90, 68, non-task

# pref_prob = np.zeros((len(df_plot.subject.unique()), len(df_plot.r_bin_naive.unique()), 3))

# for i,s in enumerate(df_plot.subject.unique()):
#     for ii,r in enumerate(np.sort(df_plot.r_bin_naive.unique().to_numpy())):
        
#         ind = (df_plot.r_bin_naive==r) & (df_plot.subject==s)
        
#         counts = df_plot[ind].groupby(['modal_pref_naive'])['modal_pref_proficient'].value_counts(normalize=True).to_frame().rename({'modal_pref_proficient':'prop'}, axis=1).reset_index()
        
#         check = counts.groupby(['modal_pref_naive'])['prop'].sum()
#         no_pref = check[check==0].index.astype(int)
        
#         print(no_pref)
        
#         counts[counts['modal_pref_naive'].isin(no_pref)] = np.nan
        
#         counts['pref_type_naive'] = counts.modal_pref_naive.apply(lambda x: '45/90' if x==45 or x==90 else '68' if x==68 else 'non-task')
#         counts['pref_type_proficient'] = counts.modal_pref_proficient.apply(lambda x: '45/90' if x==45 or x==90 else '68' if x==68 else 'non-task')
        
#         counts = counts[counts.modal_pref_naive == counts.modal_pref_proficient]
        
#         counts = counts.groupby(['pref_type_naive'])['prop'].mean().reset_index()
                
#         pref_prob[i,ii,:] = counts.prop.to_numpy()

# df_prop = pd.DataFrame()

# for i,r in enumerate(df_plot.r_bin_naive.unique()):
    
#     counts = df_plot[df_plot.r_bin_naive==r].groupby(['modal_pref_naive', 'subject'])['modal_pref_proficient'].value_counts(normalize=True).to_frame().rename({'modal_pref_proficient':'prop'}, axis=1).reset_index()

#     counts = counts[counts.modal_pref_naive==counts.modal_pref_proficient]
    
#     counts['r_bin'] = r
    
#     df_prop = pd.concat([df_prop, counts], ignore_index=True)


df_prop = df_plot.copy()

df_prop['stable'] = df_prop.modal_pref_naive == df_plot.modal_pref_proficient

df_prop['stable'] = df_prop.stable.astype(int)
df_prop['r_bin_naive'] = df_prop.r_bin_naive.astype(int)

df_prop['stim_type'] = df_prop['modal_pref_naive'].apply(lambda x: '45/90' if x==45 or x==90 else '68' if x==68 else 'non-task')

# df_prop = df_prop.groupby(['stim_type','r_bin','subject'])['prop'].mean().reset_index()

r_bins = [0., 0.16, 0.32, 0.48, 0.64, 1.]

tick_labels = [f'{r_bins[i]} - {r_bins[i+1]}' for i in range(len(r_bins)-1)]

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
    
    
    colors = sns.color_palette('colorblind', 8)
    sns.set_palette(colors[-3:],3)

    f,a = plt.subplots(1,1, figsize=(1.4,1.3))

    # (
    #     so.Plot(df_prop, x='r_bin_naive', y='stable', color='stim_type')
    #     .layout(engine='tight')
    #     .add(so.Lines(linewidth=0.5), so.Agg(), legend=False)
    #     .add(so.Band(), so.Est(errorbar=('se',1)), legend=False)
    #     .label(x='Selectivity', y='Proportion stable')
    #     .scale(color=["#49b", "#a6a", "#5b8"],
    #            x=so.Continuous().tick(at=[0,1,2,3,4]))
    #     .limit(y=(0,0.85))
    #     .on(a)
    #     .plot()
    # )
    
    (
        so.Plot(df_prop, x='r_bin_naive', y='stable', marker='stim_type')
        .layout(engine='tight')
        .add(so.Range(linewidth=0.5, artist_kws={'zorder' : 0}, color='black'), so.Est(errorbar=('se',1)), so.Dodge(), legend=False)
        .add(so.Dot(pointsize=3, edgecolor='black', edgewidth=0.5, color='xkcd:light grey'), so.Agg(), so.Dodge(), legend=False)
        .label(x='Selectivity', y='Proportion stable')
        .scale(
            #    color = so.Nominal(["#49b", "#a6a", "#5b8"], order=["45/90", "68", "non-task"]),
               marker=so.Nominal( order=["45/90", "68", "non-task"]),
               x=so.Continuous().tick(at=[0,1,2,3,4]))
        .limit(y=(0,0.85))
        .on(a)
        .plot()
    )


    sns.despine(ax=a, trim=True)
    
    a.set_xticklabels(tick_labels, rotation=30)
    # a.set_box_aspect(1)
    
    f.savefig(join(save_fig_dir, 'prop_stable_modal_pref.svg'), format='svg')
    
#%% Stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

df_stats = df_prop.copy()

r_bin = 4
stim0 = '68'
stim1 = 'non-task'

md = smf.mixedlm("stable ~ C(stim_type)", df_stats[(df_stats.r_bin_naive==r_bin) & ((df_stats.stim_type==stim0) | (df_stats.stim_type==stim1))], groups=df_stats[(df_stats.r_bin_naive==r_bin) & ((df_stats.stim_type==stim0) | (df_stats.stim_type==stim1))]['subject'])

mdf = md.fit()

print(mdf.summary())




#%% Plot modal mean mismatch by mean pref, selectivity, and condition

from scipy.stats import ttest_rel

df_tuning['pref_diff'] = (df_tuning['modal_ori_pref'] - df_tuning['mean_ori_pref'] + 180) % 180
df_tuning['pref_diff'] = df_tuning['pref_diff'].apply(lambda x: 180 - x if x > 90 else x)
df_tuning['bimodal'] = df_tuning.modal_ori_pref != df_tuning.pref_bin_ori.astype(float)
df_tuning['bimodal'] = df_tuning.bimodal.astype(int)

df_plot = df_tuning.copy()

df_plot['pref_bin_ori'] = df_plot['pref_bin_ori'].astype(int)


df_plot_bimodal = df_plot.groupby(['subject','condition', 'r_bin_ori', 'pref_bin_ori'])['bimodal'].apply(lambda x: x.sum()/len(x)).reset_index()

df_plot_bimodal['pref_type'] = df_plot_bimodal['pref_bin_ori'].apply(lambda x: '$45\degree$/$90\degree$' if (x==45) | (x==90) else '$68\degree$' if x==68 else 'non-task')

p_vals_bimodal = np.zeros((len(df_plot_bimodal.r_bin_ori.unique()),len(df_plot_bimodal.pref_type.unique())))


for i,r in enumerate(df_plot_bimodal.r_bin_ori.unique()):
    for ii,p in enumerate(df_plot_bimodal.pref_type.unique()):
        
        ind_n = (df_plot_bimodal.r_bin_ori==r) & (df_plot_bimodal.pref_type==p) & (df_plot_bimodal.condition=='naive')
        ind_c = (df_plot_bimodal.r_bin_ori==r) & (df_plot_bimodal.pref_type==p) & (df_plot_bimodal.condition=='proficient')

        
        p_vals_bimodal[i,ii] = ttest_rel(df_plot_bimodal[ind_n].bimodal.to_numpy(), 
                                         df_plot_bimodal[ind_c].bimodal.to_numpy())[1]


r_bins = np.array([0., 0.16, 0.32, 0.48, 0.64, 1.])
selectivity_labels = [f'{r_bins[i]} - {r_bins[i+1]}' for i in range(len(r_bins)-1)]

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style.update({"font.size":5,
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
              'font.sans-serif' : 'Arial',
              'font.family' : 'sans-serif'
              })


f = plt.figure(figsize=(5,1.5))

(
    so.Plot(df_plot_bimodal, x='pref_type', y='bimodal', color='condition')
    .layout(engine='tight')
    .theme({**style})
    .on(f)
    .facet(col='r_bin_ori')
    .add(so.Bar(edgecolor='black'), so.Agg(), so.Dodge(), legend=False)
    # .add(so.Dots(), so.Dodge(), legend=False)
    .add(so.Range(color='black'), so.Est(errorbar=('se',1)), so.Dodge(), legend=False)
    .limit(y=(0,1))
    .scale(color='colorblind',
           x=so.Nominal(order=[r'$45\degree$/$90\degree$', r'$68\degree$','non-task']))
    .label(y='Prop. with mismatch', x='Mean ori. pref.')
    .plot()
)

for i,a in enumerate(f.axes):
    sns.despine(ax=a, trim=True)
    a.set_title(selectivity_labels[i], fontsize=5)


# f.savefig(join(save_fig_dir,'mean_modal_mismatch_by_stim_pref.svg'), format='svg')





#%% Tuning curves - all cells

df_plot = df_tuning.copy()

df_plot = df_plot[df_plot.V1_ROI==1]

df_plot = df_plot.groupby(['condition', 'subject', 'stim_ori', 'r_bin_ori', 'pref_bin_ori'], observed=True).mean().reset_index()

# All cells

(
    so.Plot(df_plot, x='stim_ori', y='mean_resp', color='pref_bin_ori')
    .facet(row='condition', col = 'r_bin_ori')
    .add(so.Lines(), so.Agg(), legend=False)
    .add(so.Band(), so.Est(errorbar=('se',1)), legend=False)
    .scale(color='hls',
                x = so.Continuous().tick(every=22.5))
    .show()
)


#%% Tuning curves - tracked cells

%matplotlib qt

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style.update({"font.size":5,
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
              "legend.title_fontsize":5})


df_plot = df_tuning[df_tuning.tracked].copy().set_index(['subject','cell_num_tracked'])

# Remove cells whose ROIs are not in V1 in both recordings
not_in_V1 = df_plot.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_plot = df_plot.drop(not_in_V1).reset_index()

naive_pref = df_plot[df_plot.condition=='naive'].copy()
naive_pref = naive_pref.rename({'pref_bin_ori' : 'pref_bin_naive', 'r_bin_ori' : 'r_bin_naive'}, axis=1)

prof_pref = df_plot[df_plot.condition=='proficient'].copy()
prof_pref = prof_pref.rename({'pref_bin_ori' : 'pref_bin_proficient', 'r_bin_ori' : 'r_bin_proficient'}, axis=1)

df_plot = df_plot.merge(naive_pref[['cell_num_tracked', 'subject', 'pref_bin_naive', 'r_bin_naive']], on=['cell_num_tracked', 'subject'], how='left')
df_plot = df_plot.merge(prof_pref[['cell_num_tracked', 'subject', 'pref_bin_proficient', 'r_bin_proficient']], on=['cell_num_tracked', 'subject'], how='left')

df_plot = df_plot.groupby(['cell_num_tracked', 'subject', 'condition', 'pref_bin_naive', 'r_bin_naive', 'stim_ori'], observed=True).mean().reset_index()

# df_plot = df_plot.groupby(['subject','condition','pref_bin_naive', 'r_bin_naive', 'stim_ori'], observed=True).mean().reset_index()


f = (
        so.Plot(df_plot, x='stim_ori', y='mean_resp', color='pref_bin_naive')
        .facet(row='condition', col = 'r_bin_naive', order = {'row' : ['naive','proficient']})
        .layout(engine='tight', size=(5,4))
        .add(so.Lines(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar=('se',1)), legend=False)
        .scale(color='hls',
               x = so.Continuous().tick(at=[0,45,68,90,158]),
               y = so.Continuous().tick(every=0.2))
        .theme({**style})
        .label(x='Stimulus ori.',
               y='Response',
               title='')
        .limit(y=(-0.1,1.1))
        .share(x=False, y=False)
        .plot()
    )


s_bins = [0,0.16,0.32,0.48,0.64,1]

s_labels = [f'{s0} - {s1}' for s0,s1 in zip(s_bins[0:],s_bins[1:])]


for i,a in enumerate(f._figure.axes):
    
    a.vlines([45,90], 0, 1, colors='black', linestyles='solid', linewidths=0.5)
    a.vlines(68, 0, 1, colors='black', linestyles='dashed', linewidths=0.5)
    
    if i < 5:
        bottom=True
        a.set_xticks([])
        a.set_title(s_labels[i], fontsize=5)
    else:
        bottom=False
        a.set_title('')
    
    if i == 0 or i == 5:
        left=False
    else:
        left=True
        a.set_yticks([])
    
    sns.despine(ax=a, left=left, bottom=bottom, trim=True)
    
    a.set_box_aspect(1)
    
f.save(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\ori_tuning_curves_tracked.svg', format='svg')
    

#%% Tuning curves, split by retinotopy


df_plot = df_tuning.copy()

df_plot = df_plot[df_plot.V1_ROI==1]

df_plot['task_ret_dist'] = np.sqrt((df_plot.ROI_ret_azi + 80)**2 + df_plot.ROI_ret_elv**2)

df_plot['task_ret_dist_far'] = df_plot.task_ret_dist.apply(lambda x: 'near' if x <=10 else 'far' if x >= 30 else 'middle')

df_plot = df_plot[df_plot.condition=='proficient']

df_plot = df_plot.groupby(['subject','stim_ori','task_ret_dist_far', 'r_bin_ori', 'pref_bin_ori'])['mean_resp'].mean().reset_index()

p = (
        so.Plot(df_plot, x = 'stim_ori', y='mean_resp', color='pref_bin_ori')
        .facet(col='r_bin_ori', row='task_ret_dist_far')
        .add(so.Lines(), so.Agg(), legend=False)
        .add(so.Band(), so.Est('mean', errorbar=('se',1)), legend=False)
        .scale(color='hls')
        .show()
    )



#%% Piecewise linear fit on all neurons

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style['axes.labelsize'] = 10
style['xtick.labelsize'] = 8
style['ytick.labelsize'] = 8
style['legend.fontsize'] = 8
style['legend.title_fontsize'] = 8
style['legend.frameon'] = False

# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y


df_plot = df_tuning.copy()

df_plot = df_plot[(df_plot.V1_ROI==1) & (df_plot.cell_plane>0)]

df_plot = df_plot.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori', 'subject'])['mean_resp'].mean().reset_index()
df_plot = df_plot.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori'])['mean_resp'].mean().reset_index()

df_plot_pivot_all = pd.pivot(df_plot, index = ['r_bin_ori','pref_bin_ori'], columns = ['stim_ori','condition'], values = 'mean_resp').reset_index()


# stim_to_plot = np.ceil(np.arange(0,180,22.5))
stim_to_plot = np.array([45,68,90])

fit_params_all = np.zeros(len(stim_to_plot), dtype='object')

# f,a = plt.subplots(2,4, figsize = (12,7.5))

f,a = plt.subplots(1,len(stim_to_plot), figsize = (12,7.5))


for i,s in enumerate(stim_to_plot):
    
    fit_params_all[i], covar = curve_fit(pl, df_plot_pivot_all[s].naive, df_plot_pivot_all[s].proficient, p0=[.5,.5], bounds=((0,0),(1,1)))       
    
    (
        so.Plot(df_plot_pivot_all[s], x='naive', y='proficient', color=np.tile(np.ceil(np.arange(0,180,22.5)),5), marker=np.repeat(np.arange(5),8))
        .layout(size = (4,4),engine='tight')
        .limit(x = (-0.1,1.1),
                y = (-0.1,1.1))
        .add(so.Dot(pointsize=10, alpha = 1, edgecolor='k'), legend = False)
        .theme({**style})
        .scale(color = so.Continuous('hls').tick(at=np.ceil(np.arange(0,180,22.5))),
               marker = so.Nominal(['X','P','s','v','*']))
        .label(color = 'Naive ori. pref.',
            x = f'Naive resp. to {s.astype(int)}',
            y = f'Proficient resp. to {s.astype(int)}')
        .on(a.ravel()[i])
        .plot()
    )
    
    
for i,a in enumerate(a.ravel()):
    a.plot([0,1], [0,1], '--k')
    x = np.linspace(0,1,1000)
    a.plot(x,pl(x,*fit_params_all[i]),'-k')   
    sns.despine(ax=a)
    a.set_box_aspect(1)    
    
f.show()

#%% Population sparsening - Transform tuning curves using subject specific fits, All cells

# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y

df_ps = df_tuning.copy()

df_ps = df_ps[(df_ps.V1_ROI==1) & (df_ps.cell_plane>0)]

df_ps = df_ps.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori', 'subject'])['mean_resp'].mean().reset_index()

fit_params_PS = np.zeros((5,df_ps.stim_ori.nunique()), dtype=object)

for i, sub in enumerate(df_ps.subject.unique()):
    
    df_sub = pd.pivot(df_ps[df_ps.subject==sub], index = ['r_bin_ori','pref_bin_ori'], columns = ['condition','stim_ori'], values = 'mean_resp').reset_index()

    for si, s in enumerate(df_ps.stim_ori.unique()):

        fit_params_PS[i,si], _ = curve_fit(pl, df_sub[('naive',s)], df_sub[('proficient',s)], p0=[.5,.5], bounds=((0,0),(1,1)))       

#%% Population sparsening - Transform tuning curves using fit to average, All cells

# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y


df_ps = df_tuning.copy()

df_ps = df_ps[(df_ps.V1_ROI==1) & (df_ps.cell_plane>0)]

# df_ps = df_ps.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori', 'subject'])['mean_resp'].mean().reset_index()
df_ps = df_ps.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori'])['mean_resp'].mean().reset_index()

fit_params_PS = np.zeros(df_ps.stim_ori.nunique(), dtype=object)

df_ps = pd.pivot(df_ps, index = ['r_bin_ori','pref_bin_ori'], columns = ['condition','stim_ori'], values = 'mean_resp').reset_index()

for si, s in enumerate(df_tuning.stim_ori.unique()):

    fit_params_PS[si], _ = curve_fit(pl, df_ps[('naive',s)].to_numpy(), df_ps[('proficient',s)].to_numpy(), p0=[.5,.5], bounds=((0,0),(1,1)))
    
    
    
fit_params_PS = np.tile(fit_params_PS[None,:], (5,1))


#%% Transform naive tuning curves

df_sparse = df_tuning[df_tuning.condition=='naive'].copy()

df_sparse['mean_resp_ps'] = None

for i, sub in enumerate(df_sparse.subject.unique()):
    for si, s in enumerate(df_sparse.stim_ori.unique()):
        ind = (df_sparse.subject==sub) & (df_sparse.stim_ori==s)
        df_sparse.loc[ind, 'mean_resp_ps'] = pl(df_sparse.loc[ind, 'mean_resp_all_trials'], *fit_params_PS[i,si])


#%% Fit life-stime sprasening function

df_ls = df_tuning.copy()

df_ls = df_ls[(df_ls.V1_ROI==1) & (df_ls.cell_plane>0)]

df_ls = df_ls.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori', 'subject'])['mean_resp'].mean().reset_index()

fit_params_LS = np.zeros((5,df_ls.pref_bin_ori.nunique(),df_ls.r_bin_ori.nunique()), dtype=object)

for i, sub in enumerate(df_ls.subject.unique()):
    
    df_sub = pd.pivot(df_ls[df_ls.subject==sub], index = 'stim_ori', columns = ['pref_bin_ori','r_bin_ori','condition'], values = 'mean_resp').reset_index()

    for pi, p in enumerate(df_ls.pref_bin_ori.unique()):
        for ri, r in enumerate(df_ls.r_bin_ori.unique()):

            fit_params_LS[i,pi,ri], _ = curve_fit(pl, df_sub[(p,r,'naive')].to_numpy(), df_sub[(p,r,'proficient')].to_numpy(), p0=[.5,.5], bounds=((0,0),(1,1))) 


#%% Transform naive tuning curves

df_sparse['mean_resp_ls'] = None

for i, sub in enumerate(df_sparse.subject.unique()):
    for pi, p in enumerate(df_ls.pref_bin_ori.unique()):
        for ri, r in enumerate(df_ls.r_bin_ori.unique()):
            
            ind = (df_sparse.subject==sub) & (df_sparse.pref_bin_ori==p) & (df_sparse.r_bin_ori==r)
            df_sparse.loc[ind, 'mean_resp_ls'] = pl(df_sparse.loc[ind, 'mean_resp_all_trials'], *fit_params_LS[i,pi,ri])


#%% Plot tuning curves for PS and LS models - only one orientation preference

i_pref = 45

df_plot = df_sparse[(df_sparse.pref_bin_ori==i_pref) & (df_sparse.V1_ROI==1) & (df_sparse.cell_plane > 0)]

df_plot = df_plot.groupby(['subject','stim_ori','r_bin_ori'], observed=True)[['mean_resp_ls',
                                                                    'mean_resp_ps']].mean().reset_index()

df_prof = df_tuning[(df_tuning.condition=='proficient') & (df_tuning.pref_bin_ori==i_pref)].reset_index(drop=True)

df_prof = df_prof.rename({'mean_resp': 'mean_resp_prof'}, axis=1)

df_prof = df_prof.groupby(['subject','stim_ori','r_bin_ori'], observed=True)['mean_resp_prof'].mean().reset_index()

df_plot = df_plot.merge(df_prof, on=['subject','stim_ori','r_bin_ori'], how='left')

df_plot = pd.melt(df_plot, id_vars=['subject', 'stim_ori', 'r_bin_ori'],
                  value_vars=['mean_resp_prof', 'mean_resp_ls', 'mean_resp_ps'],
                  var_name='model', value_name='response')


s_titles = ['0 - 0.16', '0.16 - 0.32', '0.32 - 0.48', '0.48 - 0.64',
            '0.64 - 1']

cp = sns.color_palette('colorblind',5)
cp = [cp[i] for i in range(5) if i in (1,2,4)]

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

    
    fig = sns.relplot(df_plot, x='stim_ori', y='response', hue='model', col='r_bin_ori', kind='line',
                errorbar=('se',1), height=1.3, aspect = 0.75, palette=cp,
                hue_order=['mean_resp_prof','mean_resp_ps','mean_resp_ls'])
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

    for i,a in enumerate(fig.axes.flatten()):
        if i > 0:
            a.yaxis.set_visible(False)
            sns.despine(ax=a, left=True, trim=True)
        else:
            sns.despine(ax=a, trim=True)
           
        a.set_xlabel(r'Stimulus ori. ($\degree$)', labelpad = 1)
        a.set_ylabel('Response', labelpad = 1)
  
        a.set_title(s_titles[i])
   
        a.plot([45,45],[0,1],'-k')
        a.plot([67.5,67.5],[0,1],'--k')
        # a.plot([0,0],[0,1],'-k')
        a.plot([90,90],[0,1],'-k')
        a.set_facecolor((1,1,1,0))


    savefile = join(save_fig_dir,'tuning_curve_model_PS_vs_LS.svg')
    fig.savefig(savefile, format = 'svg')


#%% Compare transformations for tracked neurons

def polar_tc_plot(ori, tc, ax, sf=1, color='black'):
    
    tc *= sf
    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori)])

    ax.plot(coord[:,0],coord[:,1], color=color)
    
    ori_rad = np.linspace(0,2*np.pi,5000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax.plot(x_cir,y_cir,'k')
    ax.axis('off')
    ax.text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.plot([-1.2,1.2],[0,0],'--k',zorder = 1)
    ax.plot([0,0],[-1.2,1.2],'--k',zorder = 1)
    
    modal_pref = coord[np.argmax(tc),:]
    
    ax.scatter(modal_pref[0],modal_pref[1], s=40, color=color, edgecolor='black', zorder=100)
    
    mean_pref = (np.exp(1j*ori[:-1]) * tc[:-1]).sum()/tc[:-1].sum()
    
    ax.quiver(0, 0, np.real(mean_pref), np.imag(mean_pref), angles='xy', units='xy', scale=1, color=color,
              width=0.04, headlength=2, headaxislength=2, headwidth=3, edgecolor='k', linewidth=0.5)
    
    
    ax.set_aspect(1)
   

df_plot = df_tuning[df_tuning.tracked & (df_tuning.V1_ROI==1) & (df_tuning.cell_plane>0)].copy()

df_sparse_V1 = df_sparse[df_sparse.tracked & (df_sparse.V1_ROI==1) & (df_sparse.cell_plane>0)].copy()

df_plot = df_plot.merge(df_sparse_V1[['subject','cell_num_tracked','mean_resp_ps','mean_resp_ls','stim_ori']], on=['subject','cell_num_tracked','stim_ori'], how='left')


isubject = 'SF180613'
# icells = np.random.choice(df_plot[df_plot.subject==isubject].cell_num_tracked.unique(), 40, replace=False)
icells=[11]

for i in icells:

    tc_n = df_plot[(df_plot.subject==isubject) & (df_plot.cell_num_tracked==i) & (df_plot.condition=='naive')].mean_resp.to_numpy()
    tc_n = np.insert(tc_n, len(tc_n), tc_n[0])
    tc_p = df_plot[(df_plot.subject==isubject) & (df_plot.cell_num_tracked==i) & (df_plot.condition=='proficient')].mean_resp.to_numpy()
    tc_p = np.insert(tc_p, len(tc_p), tc_p[0])
    tc_ps = df_plot[(df_plot.subject==isubject) & (df_plot.cell_num_tracked==i) & (df_plot.condition=='naive')].mean_resp_ps.to_numpy()
    tc_ps = np.insert(tc_ps, len(tc_ps), tc_ps[0])
    tc_ls = df_plot[(df_plot.subject==isubject) & (df_plot.cell_num_tracked==i) & (df_plot.condition=='naive')].mean_resp_ls.to_numpy()
    tc_ls = np.insert(tc_ls, len(tc_ls), tc_ls[0])

    ori = np.linspace(0,2*np.pi,9)

    f,a = plt.subplots(1,3)

    polar_tc_plot(ori, tc_n,
                ax=a[0], color='blue')
    polar_tc_plot(ori, tc_p,
                ax=a[0], color='orange')
    polar_tc_plot(ori, tc_n,
                ax=a[1], color='blue')
    polar_tc_plot(ori, tc_ps,
                ax=a[1], color='orange')
    polar_tc_plot(ori, tc_n,
                ax=a[2], color='blue')
    polar_tc_plot(ori, tc_ls,
                ax=a[2], color='orange')



#%% Piecewise linear fit on tracked neurons


# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y


df_plot = df_tuning[df_tuning.tracked==True].copy().set_index(['subject','cell_num_tracked'])

not_in_V1 = df_plot.groupby(['subject','cell_num_tracked'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_plot = df_plot.drop(np.unique(not_in_V1)).reset_index()

naive_pref = df_plot[df_plot.condition=='naive'].copy()
naive_pref = naive_pref.rename({'pref_bin_ori' : 'pref_bin_naive', 'r_bin_ori' : 'r_bin_naive'}, axis=1)

prof_pref = df_plot[df_plot.condition=='proficient'].copy()
prof_pref = prof_pref.rename({'pref_bin_ori' : 'pref_bin_proficient', 'r_bin_ori' : 'r_bin_proficient'}, axis=1)

df_plot = df_plot.merge(naive_pref[['cell_num_tracked', 'subject', 'pref_bin_naive', 'r_bin_naive']], on=['cell_num_tracked', 'subject'], how='left')
df_plot = df_plot.merge(prof_pref[['cell_num_tracked', 'subject', 'pref_bin_proficient', 'r_bin_proficient']], on=['cell_num_tracked', 'subject'], how='left')

df_plot = df_plot.groupby(['r_bin_naive', 'pref_bin_naive', 'condition', 'stim_ori'], observed=True)['mean_resp'].mean().reset_index()

df_plot_pref = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].pref_bin_naive
df_plot_r = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].r_bin_naive

df_plot_pivot_tracked = pd.pivot(df_plot, index = ['r_bin_naive','pref_bin_naive'], columns = ['stim_ori','condition'], values = 'mean_resp').reset_index()


stim_to_plot = np.ceil(np.arange(0,180,22.5))
# stim_to_plot = np.array([45,68,90])

fit_params_tracked = np.zeros(len(stim_to_plot), dtype='object')


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


    # f,a = plt.subplots(2,4, figsize = (12,7.5))

    h = 1.25

    f,a = plt.subplots(1,len(stim_to_plot), figsize = (3*h,h))


    for i,s in enumerate(stim_to_plot):

        fit_params_tracked[i], covar = curve_fit(pl, df_plot_pivot_tracked[s].naive, df_plot_pivot_tracked[s].proficient, p0=[.5,.5], bounds=((0,0),(1,1)))       
        
        (
            so.Plot(df_plot_pivot_tracked[s], x='naive', y='proficient', color=np.tile(np.ceil(np.arange(0,180,22.5)),5), marker=np.repeat(np.arange(5),8))
            .layout(engine='tight')
            .limit(x = (-0.1,1.1),
                    y = (-0.1,1.1))
            .add(so.Dot(pointsize=3, alpha=1, edgecolor='k'), legend=False)
            # .theme({**style})
            .scale(color = so.Continuous('hls').tick(at=np.ceil(np.arange(0,180,22.5))),
                   marker = so.Nominal(['X','P','s','v','*']),
                   x=so.Continuous().tick(at=[0,0.5,1]),
                   y=so.Continuous().tick(at=[0,0.5,1]))
            .label(color = 'Naive ori. pref.',
                x = f'Naive resp. to {s.astype(int)}',
                y = f'Proficient resp. to {s.astype(int)}')
            .on(a.ravel()[i])
            .plot()
        )
        
        
    for i,a in enumerate(a.ravel()):
        a.plot([0,1], [0,1], '--k', dashes=(5, 10))
        x = np.linspace(0,1,1000)
        a.plot(x,pl(x,*fit_params_tracked[i]), '-k')
        a.plot(x,pl(x,*fit_params_all[i]), '--k', dashes=(3, 2))      
        sns.despine(ax=a, trim=True)
        a.set_box_aspect(1)    
        
        
    f.tight_layout()
    # f.savefig(join(save_fig_dir,'naive_vs_proficient_tracked.svg'), format='svg')


#%% Modify tracked cells naive tuning curves with convex functions

def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y


def polar_tc_plot(ori, tc, ax, sf=1, color='black'):
    
    tc *= sf
    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori)])

    ax.plot(coord[:,0],coord[:,1], color=color)
    
    ori_rad = np.linspace(0,2*np.pi,1000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax.plot(x_cir,y_cir,'k')
    ax.axis('off')
    ax.text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.plot([-1.2,1.2],[0,0],'--k',zorder = 1)
    ax.plot([0,0],[-1.2,1.2],'--k',zorder = 1)
    
    modal_pref = coord[np.argmax(tc),:]
    
    ax.scatter(modal_pref[0],modal_pref[1], s=40, color=color, edgecolor='black', zorder=100)
    
    mean_pref = (np.exp(1j*ori[:-1]) * tc[:-1]).sum()/tc[:-1].sum()
    
    ax.quiver(0, 0, np.real(mean_pref), np.imag(mean_pref), angles='xy', units='xy', scale=1, color=color,
              width=0.04, headlength=2, headaxislength=2, headwidth=3, edgecolor='k', linewidth=0.5)
    
    
    ax.set_aspect(1)

df_convex = df_tuning[df_tuning.tracked]

not_in_V1 = df_convex.groupby(['subject','cell_num_tracked'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_convex = df_convex.drop(np.unique(not_in_V1)).reset_index()

for i,s in enumerate(df_convex.stim_ori.unique()):

    df_pred = df_convex[(df_convex.stim_ori==s) & (df_convex.condition=='naive')].copy()

    df_pred['condition'] = 'predicted'

    df_pred['mean_resp'] = pl(df_pred['mean_resp'].to_numpy(), *fit_params_tracked[i])

    df_convex = pd.concat([df_convex, df_pred], ignore_index=True)

i_subject = 'SF180515'

n_cells = 20

# i_cells = np.random.choice(df_convex[(df_convex.subject==i_subject) & (df_convex.pref_bin_ori==90) & (df_convex.condition=='naive')].cell_num_tracked.unique(), 20, replace=False)

i_cells = df_convex[(df_convex.subject==i_subject) & (df_convex.pref_bin_ori==45
                                                      ) & (df_convex.condition=='naive')].cell_num_tracked.unique()

print(len(i_cells))

cond_colors = sns.color_palette('colorblind',2)

for i,c in enumerate(i_cells):
    
    cell_ind = (df_convex.subject==i_subject) & (df_convex.cell_num_tracked==c)
    
    tc_n = df_convex[cell_ind & (df_convex.condition=='naive')]['mean_resp'].to_numpy()
    tc_n = np.insert(tc_n, len(tc_n), tc_n[0])
    tc_p = df_convex[cell_ind & (df_convex.condition=='predicted')]['mean_resp'].to_numpy()
    tc_p = np.insert(tc_p, len(tc_p), tc_p[0])

    ori = np.linspace(0,2*np.pi,9)

    f,a = plt.subplots(1,1)
    
    polar_tc_plot(ori, tc_n, ax=a, color=cond_colors[0])
    polar_tc_plot(ori, tc_p, ax=a, color=cond_colors[1])
    
    a.set_title(f'Subject {i_subject} and tracked cell {c}')



#%% Calculate convexity for each subject's tracked cells

df_plot = df_tuning[df_tuning.tracked==True].copy().set_index(['subject','cell_num_tracked'])

not_in_V1 = df_plot.groupby(['subject','cell_num_tracked'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_plot = df_plot.drop(np.unique(not_in_V1)).reset_index()

naive_pref = df_plot[df_plot.condition=='naive'].copy()
naive_pref = naive_pref.rename({'pref_bin_ori' : 'pref_bin_naive', 'r_bin_ori' : 'r_bin_naive'}, axis=1)

prof_pref = df_plot[df_plot.condition=='proficient'].copy()
prof_pref = prof_pref.rename({'pref_bin_ori' : 'pref_bin_proficient', 'r_bin_ori' : 'r_bin_proficient'}, axis=1)

df_plot = df_plot.merge(naive_pref[['cell_num_tracked', 'subject', 'pref_bin_naive', 'r_bin_naive']], on=['cell_num_tracked', 'subject'], how='left')
df_plot = df_plot.merge(prof_pref[['cell_num_tracked', 'subject', 'pref_bin_proficient', 'r_bin_proficient']], on=['cell_num_tracked', 'subject'], how='left')

df_plot = df_plot.groupby(['subject', 'r_bin_naive', 'pref_bin_naive', 'condition', 'stim_ori'], observed=True)['mean_resp'].mean().reset_index()

df_plot_pref = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].pref_bin_naive
df_plot_r = df_plot[(df_plot.condition=='naive') & (df_plot.stim_ori==0)].r_bin_naive


ratio_slope = np.zeros((len(match_subjects), 8))

for i,s in enumerate(df_plot.subject.unique()):
         
    df_subject = pd.pivot(df_plot[df_plot.subject==s], index = ['r_bin_naive','pref_bin_naive'], columns = ['stim_ori','condition'], values = 'mean_resp').reset_index()
    r_bins = df_subject['r_bin_naive'].astype(int)
    ori_pref = df_subject['pref_bin_naive'].astype(int)


    for ii,o in enumerate(df_plot.stim_ori.unique()):
        pref_ind = (ori_pref==o) & (r_bins==4)
        x_pref = df_subject[o]['naive'][(ori_pref==o) & (r_bins==4)].to_numpy()
        y_pref = df_subject[o]['proficient'][(ori_pref==o) & (r_bins==4)].to_numpy()

        x_others = df_subject[o]['naive'][ori_pref!=o].to_numpy()
        y_others = df_subject[o]['proficient'][ori_pref!=o].to_numpy()


        if len(x_pref) > 0 and len(y_pref) > 0:
            pref_slope = y_pref/x_pref
            other_slope, _, _, _ = np.linalg.lstsq(x_others.reshape(-1,1),
                                                   y_others.reshape(-1,1),
                                                   rcond = None)

            ratio_slope[i,ii] = pref_slope/other_slope
        else:
            ratio_slope[i,ii] = np.nan


#%% Plot change in mean response to 45,68,90 in tracked cells

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style.update({"font.size":5,
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
              "legend.title_fontsize":5})

df_plot = df_tuning[df_tuning.tracked].copy().set_index(['subject','cell_num_tracked'])

not_in_V1 = df_plot.groupby(['subject','cell_num_tracked'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_plot = df_plot.drop(not_in_V1).reset_index()

df_n = df_plot[df_plot.condition=='naive'].sort_values(['subject','cell_num_tracked']).reset_index(drop=True)
df_p = df_plot[df_plot.condition=='proficient'].sort_values(['subject','cell_num_tracked']).reset_index(drop=True)

df_diff = df_n.copy()
df_diff['resp_diff'] = df_p.mean_resp - df_n.mean_resp

df_diff['stim_type'] = df_diff.stim_ori.apply(lambda x: '45/90' if x==45 or x==90 else '68' if x==68 else 'non-task')

df_diff = df_diff.groupby(['stim_type','subject'], observed=True)['resp_diff'].mean().reset_index()

f = (
        so.Plot(df_diff, x='stim_type', y='resp_diff')
        .layout(engine='tight', size=(1.4,1.4))
        .theme({**style})
        .add(so.Range(color='black'), so.Est('mean', errorbar=('se',1)), legend=False)
        .add(so.Dot(color='black', pointsize=2), so.Jitter(), legend=False)
        .add(so.Dash(color='black'), so.Agg('mean'), legend=False)
        .label(x='Stimulus orientation',
               y='Change in response')
        .plot()
)

# f = (
#         so.Plot(df_diff, x='stim_type', y='resp_diff')
#         .layout(engine='tight', size=(1.5,1.5))
#         .theme({**style})
#         .add(so.Range(color='black'), so.Est('mean', errorbar=('se',1)), legend=False)
#         .add(so.Bar(color='grey'), so.Agg('mean'), legend=False)
#         .add(so.Dots(color='grey', pointsize=1, alpha=0.001), so.Jitter(), legend=False)
#         .scale(color='colorblind', 
#                x=so.Nominal(order=['45/90','68','non-task']))
#         .label(x='Stimulus orientation',
#                y='Change in response')
#         .plot()
#     )


# f = (
#         so.Plot(df_diff, x='stim_ori', y='resp_diff', color='stim_type')
#         .layout(engine='tight', size=(1.5,1.5))
#         .theme({**style})
#         .add(so.Range(color='black'), so.Est('mean', errorbar=('se',1)), legend=False)
#         .add(so.Bar(), so.Agg('mean'), legend=False)
#         .add(so.Dots(pointsize=1, alpha=0.001), so.Jitter(), legend=False)
#         .scale(color='colorblind', 
#                x=so.Continuous().tick(at=np.ceil(np.arange(0,180,22.5))))
#         .label(x='Stimulus orientation',
#                y='Change in response')
#         .plot()
# )



xlim = f._figure.axes[0].get_xlim()
f._figure.axes[0].plot(xlim,[0,0],'--k', linewidth=0.5)
sns.despine(ax = f._figure.axes[0], trim = True)

f.save(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\change_in_resp_tracked.svg', format='svg')


#%% Transform naive mean respones by piecewise linear fits and look at change in modal orientation preference


style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style['axes.labelsize'] = 10
style['xtick.labelsize'] = 8
style['ytick.labelsize'] = 8
style['legend.fontsize'] = 8
style['legend.title_fontsize'] = 8
style['legend.frameon'] = False

# piecewise linear
def pl(x,p,q):
    # piecewise linear (0,0) to (p,q) to (1,1)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q/p
    y[x>p] = 1 + (x[x>p]-1)*(q-1)/(p-1)
    return y


df_fit = df_tuning.copy()

df_fit = df_fit[(df_fit.V1_ROI==1) & (df_fit.cell_plane>0)]


stim_to_fit = df_fit.stim_ori.unique()

df_trans = df_tuning[df_tuning.condition=='naive'].copy()[['subject', 'cell_num', 'cell_plane', 'stim_ori', 'mean_resp_all_trials']]

fit_type = 'together'

if fit_type=='subject':
    
    
    df_fit = df_fit.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori', 'subject'])['mean_resp'].mean().reset_index()
    
    fit_params = np.zeros((len(stim_to_fit), len(df_tuning.subject.unique())), dtype='object')

    for j,m in enumerate(df_tuning.subject.unique()):
        
        df_fit_pivot = pd.pivot(df_fit[df_fit.subject==s], index=['r_bin_ori','pref_bin_ori'], columns=['stim_ori','condition'], values='mean_resp').reset_index()

        for i,s in enumerate(stim_to_fit):

            fit_params[i,j], covar = curve_fit(pl, df_fit_pivot[s].naive.to_numpy(), df_fit_pivot[s].proficient.to_numpy(), p0=[.5,.5], bounds=((0,0),(1,1)))       


    for j,m in enumerate(df_tuning.subject.unique()):
        for i,s in enumerate(stim_to_fit):
            ind = (df_trans.stim_ori==s) & (df_trans.subject==m)
            df_trans.loc[ind, 'mean_resp_all_trials'] = pl(df_trans[ind].mean_resp_all_trials, *fit_params[i,j])
            
else:
    
    df_fit = df_fit.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori', 'subject'])['mean_resp'].mean().reset_index()
    df_fit = df_fit.groupby(['r_bin_ori', 'pref_bin_ori', 'condition', 'stim_ori'])['mean_resp'].mean().reset_index()
    
    df_fit_pivot = pd.pivot(df_fit, index=['r_bin_ori','pref_bin_ori'], columns=['stim_ori','condition'], values='mean_resp').reset_index()
    
    fit_params = np.zeros(len(stim_to_fit), dtype='object')
    
    for i,s in enumerate(stim_to_fit):

        fit_params[i], covar = curve_fit(pl, df_fit_pivot[s].naive.to_numpy(), df_fit_pivot[s].proficient.to_numpy(), p0=[.5,.5], bounds=((0,0),(1,1)))       

    for i,s in enumerate(stim_to_fit):
        ind = df_trans.stim_ori==s
        df_trans.loc[ind, 'mean_resp_all_trials'] = pl(df_trans[ind].mean_resp_all_trials, *fit_params[i])



df_trans['condition'] = 'transformed'

# Recalculate tuning pref from train tuning curve

def ori_vec(stim,r):
    top = np.sum(np.exp(2j*np.deg2rad(stim))*r)
    bottom = r.sum()
    
    return top/bottom

df_pref = df_trans.groupby(['subject','cell_num','cell_plane']).apply(lambda x: ori_vec(x.stim_ori.to_numpy(), x.mean_resp_all_trials.to_numpy())).reset_index()

df_pref['condition'] = 'transformed'
df_pref = df_pref.rename({0 : 'ori_tune_vec'}, axis=1)

df_pref['mean_ori_pref'] = np.mod(11.25+np.arctan2(np.imag(df_pref.ori_tune_vec),np.real(df_pref.ori_tune_vec))*90/np.pi,180)-11.25
df_pref['r_ori'] = np.abs(df_pref.ori_tune_vec)

df_pref['pref_bin_ori'] = pd.cut(df_pref.mean_ori_pref, np.linspace(-11.25,180-11.25,9), labels=np.ceil(np.arange(0,180,22.5)))
df_pref['r_bin_ori'] = pd.cut(df_pref.r_ori, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels=np.arange(5))

df_pref['modal_ori_pref'] = df_trans.groupby(['subject','cell_num','cell_plane']).apply(lambda x: x.stim_ori[x.mean_resp_all_trials.idxmax()]).reset_index()[0]

df_trans = df_trans.merge(df_pref, on=['subject', 'cell_num', 'cell_plane', 'condition'])

df_trans = pd.concat([df_trans, df_tuning], ignore_index=True)

df_plot = df_trans.groupby(['subject','condition'])['modal_ori_pref'].value_counts(normalize=True).to_frame().rename({'modal_ori_pref' : 'proportion'}, axis=1).reset_index()

p = (
        so.Plot(df_plot, x='modal_ori_pref', y='proportion', color='condition')
        .add(so.Bars(), so.Agg(), so.Dodge())
        .add(so.Range(), so.Est('mean', errorbar=('se',2)), so.Dodge())
        .scale(x=so.Nominal(np.ceil(np.arange(0,180,22.5))))
        .show()    
    )

df_plot = df_trans.groupby(['subject', 'condition', 'cell_num', 'cell_plane', 'stim_ori'])['mean_resp_all_trials'].mean().reset_index()

p = (
        so.Plot(df_plot, x='stim_ori', y='mean_resp_all_trials', color='condition')
        .add(so.Bars(), so.Agg(), so.Dodge())
        .show()
    )   



pref_to_plot = [45,68,90]

f,a = plt.subplots(2,len(pref_to_plot))

df_plot = df_trans.groupby(['subject','condition'])['modal_ori_pref'].value_counts(normalize=True).to_frame().rename({'modal_ori_pref' : 'proportion'}, axis=1).reset_index()


for i,s in enumerate(pref_to_plot):
    
    (
        so.Plot(df_plot[(df_plot.modal_ori_pref==s) & (df_plot.condition!='transformed')], x = 'condition', y = 'proportion', group='subject')
        .add(so.Dots(), legend=False)
        .add(so.Lines(), legend=False)
        .on(a[0,i])
        .plot()         
    )
     
    a[0,i].set_title(s)
        
    
    (
        so.Plot(df_plot[(df_plot.modal_ori_pref==s) & (df_plot.condition!='proficient')], x = 'condition', y = 'proportion', group='subject')
        .add(so.Dots(), legend=False)
        .add(so.Lines(), legend=False)
        .on(a[1,i])
        .plot()         
    )
    







#%%



style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

style['axes.labelsize'] = 10
style['xtick.labelsize'] = 8
style['ytick.labelsize'] = 8
style['legend.fontsize'] = 8
style['legend.title_fontsize'] = 8
style['legend.frameon'] = False

df_plot = df_tuning[df_tuning.tracked].copy()

not_in_V1 = df_plot.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.all(x['V1_ROI'])).index
df_plot = df_plot.drop(np.unique(not_in_V1))

df_n = df_plot[df_plot.condition=='naive'].reset_index(drop=True).sort_values(['subject','cell_num_tracked'])
df_p = df_plot[df_plot.condition=='proficient'].reset_index(drop=True).sort_values(['subject','cell_num_tracked'])

df_diff = df_n.copy()
df_diff['diff'] = df_p.mean_resp - df_n.mean_resp

df_diff = df_diff.groupby(['subject','stim_ori'])['diff'].mean().reset_index()

f,a = plt.subplots(1,1)

f = (
        so.Plot(df_diff, x='stim_ori', y='diff', color='stim_ori')
        .theme({**style})
        .add(so.Range(color='black'), so.Est('mean', errorbar=('se',1)), legend=False)
        .add(so.Dots(), so.Jitter(), legend=False)
        .add(so.Dash(color='black'), so.Agg('mean'), legend=False)
        .scale(color='colorblind', 
               x=so.Continuous().tick(at=np.ceil(np.arange(0,180,22.5))))
        .label(x='Stimulus orientation',
               y='Change in response')
        .on(a)
        .plot()
)

xlim = a.get_xlim()
a.plot(xlim,[0,0],'--k')
sns.despine(ax = a, trim = True)

f.show()


#%% Cosine similarity

def cosine_similarity(a,b):
    
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


df_mean = df_resps.groupby(['cell_num','cell_plane','subject','condition','stim_ori','train_ind'], observed=True)['trial_resps'].mean().reset_index()

df_cs = pd.DataFrame()

for s0 in df_mean.stim_ori.unique():
    for s1 in df_mean.stim_ori.unique():
        
        df_cs_stim = df_mean.groupby(['subject','condition']).apply(lambda x: cosine_similarity(x[(x.stim_ori==s0) & (x.train_ind)].trial_resps.to_numpy(),
                                                                                                x[(x.stim_ori==s1) & (~x.train_ind)].trial_resps.to_numpy())).reset_index()
        df_cs_stim['stim_0'] = s0
        df_cs_stim['stim_1'] = s1
        
        df_cs = pd.concat([df_cs, df_cs_stim], ignore_index=True)
    
# %%

df_cs_naive = df_cs[df_cs.condition=='naive'].reset_index()
df_cs_proficient = df_cs[df_cs.condition=='proficient'].reset_index()

df_cs_diff = df_cs_naive.copy()

df_cs_diff[0] = df_cs_proficient[0] - df_cs_naive[0]

cs_pivot = pd.pivot_table(df_cs_diff, index='stim_1', columns='stim_0', values=0)

sns.heatmap(cs_pivot, cmap='magma_r')



#%% Population sparseness as function of retinotopic location

%matplotlib qt

def ps(fr):
    top = (fr/fr.shape[0]).sum(0)**2
    bottom = (fr**2/fr.shape[0]).sum(0)
    s = 1 - (top/bottom)

    return s

df_plot = df_tuning.copy()

df_plot = df_plot[df_plot.V1_ROI==1]

df_plot['task_ret'] = np.sqrt((df_plot.ROI_ret_azi + 80)**2 + df_plot.ROI_ret_elv**2)
df_plot['task_ret_bin'] = pd.cut(df_plot.task_ret, [0,20,100], labels=np.arange(2))

df_plot['stim_type'] = df_plot['stim_ori'].apply(lambda x: '45/90' if (x==45) | (x==90) else '68' if x==68 else 'non-task')

df_ps = df_plot.groupby(['subject','condition','stim_type', 'task_ret_bin'])['mean_resp'].apply(lambda x: ps(x.to_numpy())).reset_index().rename({'mean_resp' : 'ps'}, axis=1)

df_diff = df_ps[df_ps.condition=='naive'].copy().reset_index(drop=True)
df_diff['ps'] = df_ps[df_ps.condition=='proficient'].sort_values(['subject','stim_type','task_ret_bin']).reset_index(drop=True).ps - df_ps[df_ps.condition=='naive'].sort_values(['subject','stim_type','task_ret_bin']).reset_index(drop=True).ps


(
    so.Plot(df_diff, x='stim_type', y='ps', color='task_ret_bin')
    .add(so.Dots(), so.Dodge(), legend=False)
    .add(so.Range(), so.Est('mean', errorbar=('se',2)), so.Dodge(), legend=False)
    .add(so.Dash(), so.Agg(), so.Dodge(), legend=False)
    .scale(color=so.Nominal(order=[0,1]))
    .show()
)


#%% Population sparseness for 45, 68, 90

df_plot = df_tuning.copy()

df_plot = df_plot[df_plot.V1_ROI==1]

df_ps = df_plot.groupby(['subject','condition','stim_ori'])['mean_resp'].apply(lambda x: ps(x.to_numpy())).reset_index().rename({'mean_resp' : 'ps'}, axis=1)

df_diff = df_ps[df_ps.condition=='naive'].copy().reset_index(drop=True)
df_diff['ps'] = df_ps[df_ps.condition=='proficient'].sort_values(['subject','stim_ori']).reset_index(drop=True).ps - df_ps[df_ps.condition=='naive'].sort_values(['subject','stim_ori']).reset_index(drop=True).ps

df_diff = df_diff[df_diff.stim_ori.isin([45,68,90])].reset_index(drop=True)





# %% Plot changes in tuning curves with training for tracked cells

def polar_tc_plot(ori, tc, ax, sf=1, color='black'):
    
    tc *= sf
    coord = np.array([[r*np.cos(i), r*np.sin(i)] for r,i in zip(tc,ori)])

    ax.plot(coord[:,0],coord[:,1], color=color)
    
    ori_rad = np.linspace(0,2*np.pi,5000)
    x_cir, y_cir = np.cos(ori_rad)*1.2, np.sin(ori_rad)*1.2
    ax.plot(x_cir,y_cir,'k')
    ax.axis('off')
    ax.text(1.34, 0, r'0$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, 1.35, r'45$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.text(-1.38, 0, r'90$\degree$', verticalalignment='center',
                 horizontalalignment='center')
    ax.text(0, -1.35, r'135$\degree$', horizontalalignment='center',
                 verticalalignment='center')
    ax.plot([-1.2,1.2],[0,0],'--k',zorder = 1)
    ax.plot([0,0],[-1.2,1.2],'--k',zorder = 1)
    
    modal_pref = coord[np.argmax(tc),:]
    
    ax.scatter(modal_pref[0],modal_pref[1], s=40, color=color, edgecolor='black', zorder=100)
    
    mean_pref = (np.exp(1j*ori[:-1]) * tc[:-1]).sum()/tc[:-1].sum()
    
    ax.quiver(0, 0, np.real(mean_pref), np.imag(mean_pref), angles='xy', units='xy', scale=1, color=color,
              width=0.04, headlength=2, headaxislength=2, headwidth=3, edgecolor='k', linewidth=0.5)
    
    
    ax.set_aspect(1)
   
   
 
df_plot = df_tuning[df_tuning.tracked].copy()

not_in_V1 = df_plot.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.all(x['V1_ROI'])).index
df_plot = df_plot.drop(not_in_V1)

# i_subject = 'SF170905B'
# i_cells = [240,203,107,533,188,302,246,151,500]

# i_subject = 'SF180613'
# i_cells = [11]
# i_cells = np.random.choice(df_plot[df_plot.subject==i_subject].cell_num_tracked.to_numpy(), 50, replace=False)

i_subject = 'SF170620B'
i_cells = [139]
# i_cells = np.random.choice(df_plot[df_plot.subject==i_subject].cell_num_tracked.unique(), 50, replace=False)

# i_subject='SF180515'
# i_cells = [144, 347, 353, 409]

# i_cells = df_plot.set_index(['subject','cell_num_tracked']).groupby(['subject','cell_num_tracked'])['pref_bin_ori'].apply(lambda x: np.all(x==i_pref)).reset_index()
# i_cells = df_plot.set_index(['subject','cell_num_tracked']).groupby(['subject','cell_num_tracked'])['pref_bin_ori'].apply(lambda x: len(set(x))==1).reset_index()

# i_cells = i_cells[(i_cells.subject==i_subject) & i_cells.pref_bin_ori].cell_num_tracked.to_numpy()

# i_cells = df_plot[(df_plot.subject==i_subject) & (df_plot.condition=='proficient')].groupby(['cell_num_tracked'])[['stim_ori','mean_resp']].filter(lambda x: (x[x.stim_ori==45].mean_resp < x.mean_resp.mean()) & (x[x.stim_ori==90].mean_resp < x.mean_resp.mean())).index

# i_cells = np.random.choice(df_plot[df_plot.subject==i_subject].cell_num_tracked.unique(), 50, replace=False)

print(len(i_cells))

cond_colors = sns.color_palette('colorblind',2)

for i,c in enumerate(i_cells):
    
    cell_ind = (df_plot.subject==i_subject) & (df_plot.cell_num_tracked==c)
    
    tc_n = df_plot[cell_ind & (df_plot.condition=='naive')]['mean_resp_all_trials'].to_numpy()
    tc_n = np.insert(tc_n, len(tc_n), tc_n[0])
    tc_p = df_plot[cell_ind & (df_plot.condition=='proficient')]['mean_resp_all_trials'].to_numpy()
    tc_p = np.insert(tc_p, len(tc_p), tc_p[0])

    ori = np.linspace(0,2*np.pi,9)

    f,a = plt.subplots(1,1)
    
    polar_tc_plot(ori, tc_n, ax=a, color=cond_colors[0])
    polar_tc_plot(ori, tc_p, ax=a, color=cond_colors[1])
    
    a.set_title(f'Subject {i_subject} and tracked cell {c}')


#%% Decoding 45 vs 90 with tracked neurons - using naive fit model

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

df_tracked = df_resps[df_resps.tracked].copy()

# Remove cells whose ROIs are not in V1 at least one recordings
not_in_V1 = df_tracked.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_tracked = df_tracked.drop(not_in_V1).reset_index(drop=True)

df_decoding_naive_weights = pd.DataFrame()

pred = np.zeros((len(match_subjects), 2), dtype=object)
prob = pred.copy()
true_stim = pred.copy()

for i,s in enumerate(df_tracked.subject.unique()):
    
    scaler = StandardScaler()
    
    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    
    for ic, c in enumerate(df_tracked.condition.unique()):
        if c == 'naive':
                        
            train_ind = (df_tracked.subject==s) & df_tracked.train_ind & (df_tracked.condition==c) & ~np.isinf(df_tracked.stim_ori)
            train_ind = train_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
            
            stim_train = df_tracked[train_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
            resps_train = pd.pivot(df_tracked[train_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()
                    
            resps_train = scaler.fit_transform(resps_train)
            
            clf.fit(resps_train, stim_train)
            
            test_ind = (df_tracked.subject==s) & ~df_tracked.train_ind & (df_tracked.condition==c) & ~np.isinf(df_tracked.stim_ori)
            test_ind = test_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
            
        else:
            
            # Use all trials for proficient condition, since model is only fit to naive data
            
            test_ind = (df_tracked.subject==s) & (df_tracked.condition==c) & ~np.isinf(df_tracked.stim_ori)
            test_ind = test_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
            
        stim_test = df_tracked[test_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
        
        resps_test = pd.pivot(df_tracked[test_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()       
        resps_test = scaler.transform(resps_test)
        
        df_decoding_naive_weights = pd.concat([df_decoding_naive_weights, pd.DataFrame({'subject' : np.repeat(s, len(stim_test)),
                                                                                        'condition' : np.repeat(c, len(stim_test)),
                                                                                        'pred' : clf.predict(resps_test),
                                                                                        'stim' : stim_test})], ignore_index=True)
        
        

#%% Plot results

df_plot = df_decoding_naive_weights.copy()
df_plot['correct'] = df_plot['pred']==df_plot['stim']

df_plot = df_plot.groupby(['subject','condition'])['correct'].mean().reset_index()
df_plot['correct'] = df_plot['correct']*100


(
    so.Plot(df_plot, x='condition', y='correct')
    .add(so.Dots(), color='condition')
    .add(so.Lines(linewidth=0.5, color='k'), group='subject')
    .show()
)

#%% Decoding 45 vs 90 with tracked neurons - model fit per session

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

df_tracked = df_resps[df_resps.tracked].copy()

# Remove cells whose ROIs are not in V1 at least one recordings
not_in_V1 = df_tracked.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_tracked = df_tracked.drop(not_in_V1).reset_index(drop=True)

df_decoding_new_weights = pd.DataFrame()

pred = np.zeros((len(match_subjects), 2), dtype=object)
prob = pred.copy()
true_stim = pred.copy()

for i,s in enumerate(df_tracked.subject.unique()):
    
    scaler = StandardScaler()
    
    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    
    for ic, c in enumerate(df_tracked.condition.unique()):
      
        train_ind = (df_tracked.subject==s) & df_tracked.train_ind & (df_tracked.condition==c) & ~np.isinf(df_tracked.stim_ori)
        train_ind = train_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
        
        stim_train = df_tracked[train_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
        resps_train = pd.pivot(df_tracked[train_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()
                
        resps_train = scaler.fit_transform(resps_train)
        
        clf.fit(resps_train, stim_train)
        
        test_ind = (df_tracked.subject==s) & ~df_tracked.train_ind & (df_tracked.condition==c) & ~np.isinf(df_tracked.stim_ori)
        test_ind = test_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
            
        stim_test = df_tracked[test_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
        
        resps_test = pd.pivot(df_tracked[test_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()       
        resps_test = scaler.transform(resps_test)
        
        df_decoding_new_weights = pd.concat([df_decoding_new_weights, pd.DataFrame({'subject' : np.repeat(s, len(stim_test)),
                                                                                    'condition' : np.repeat(c, len(stim_test)),
                                                                                    'pred' : clf.predict(resps_test),
                                                                                    'stim' : stim_test})], ignore_index=True)

#%% Plot results

df_plot = df_decoding_new_weights.copy()
df_plot['correct'] = df_plot['pred']==df_plot['stim']

df_plot = df_plot.groupby(['subject','condition'])['correct'].mean().reset_index()
df_plot['correct'] = df_plot['correct']*100


(
    so.Plot(df_plot, x='condition', y='correct')
    .add(so.Dots(), color='condition')
    .add(so.Lines(linewidth=0.5, color='k'), group='subject')
    .show()
)

#%% Plot model performance by condition for both approaches


df_plot0 = df_decoding_naive_weights.copy()
df_plot0['correct'] = df_plot0['pred']==df_plot0['stim']

df_plot0 = df_plot0.groupby(['subject','condition'])['correct'].mean().reset_index()
df_plot0['correct'] = df_plot0['correct']*100

df_plot1 = df_decoding_new_weights.copy()
df_plot1['correct'] = df_plot1['pred']==df_plot1['stim']

df_plot1 = df_plot1.groupby(['subject','condition'])['correct'].mean().reset_index()
df_plot1['correct'] = df_plot1['correct']*100



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

    f,a = plt.subplots(1,2, figsize=(1.3,1.1))


    (
        so.Plot(df_plot0, x='condition', y='correct')
        .add(so.Dots(pointsize=3, stroke=0.5), so.Jitter(y=0, seed=1), color='condition', legend=False)
        .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), so.Jitter(y=0, seed=1), group='subject', legend=False)
        .on(a[0])
        .label(x='',  y='Classification accuracy (% correct)')
        .scale(color='colorblind',
               x=so.Nominal(['naive','proficient']))
        .plot()
    )

    (
        so.Plot(df_plot1, x='condition', y='correct')
        .add(so.Dots(pointsize=3, stroke=0.5), so.Jitter(y=0, seed=2), color='condition', legend=False)
        .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), so.Jitter(y=0, seed=2), group='subject', legend=False)
        .on(a[1])
        .label(x='', y='')
        .scale(color='colorblind',
               x=so.Nominal(['naive','proficient']))
        .limit(y=a[0].get_ylim())
        .plot()
    )

    for i,ax in enumerate(a):
        ax.set_xticklabels(['Naive','Proficient'])
        if i == 1:
            left=True
            ax.set_yticks([])
            ax.set_title('Session-specific fit')
        else:
            left=False
            ax.set_title('Naive fit')
        
        sns.despine(ax=ax, left=left, bottom=True, trim=True)
        ax.set_xticks([])



#%% Decoding 45 vs 90 with tracked neurons - fit model to naive, test naive and proficient. fit model to proficient, test naive and proficient.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

df_tracked = df_resps[df_resps.tracked].copy()

# Remove cells whose ROIs are not in V1 at least one recordings
not_in_V1 = df_tracked.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_tracked = df_tracked.drop(not_in_V1).reset_index(drop=True)

df_decoding = pd.DataFrame()

pred = np.zeros((len(match_subjects), 2), dtype=object)
prob = pred.copy()
true_stim = pred.copy()

for i,s in enumerate(df_tracked.subject.unique()):

    scaler = StandardScaler()

    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    
    for c0 in df_tracked.condition.unique():
        for c1 in df_tracked.condition.unique():
        
            train_ind = (df_tracked.subject==s) & df_tracked.train_ind & (df_tracked.condition==c0) & ~np.isinf(df_tracked.stim_ori)
            train_ind = train_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
            
            stim_train = df_tracked[train_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
            resps_train = pd.pivot(df_tracked[train_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()
                    
            resps_train = scaler.fit_transform(resps_train)
            
            clf.fit(resps_train, stim_train)
            
            if c0 == c1:
                test_ind = (df_tracked.subject==s) & ~df_tracked.train_ind & (df_tracked.condition==c1) & ~np.isinf(df_tracked.stim_ori)
                test_ind = test_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
            else:
                test_ind = (df_tracked.subject==s) & (df_tracked.condition==c1) & ~np.isinf(df_tracked.stim_ori)
                test_ind = test_ind & ((df_tracked.stim_ori==45) | (df_tracked.stim_ori==90))
                
            stim_test = df_tracked[test_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
            
            resps_test = pd.pivot(df_tracked[test_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()       
            resps_test = scaler.transform(resps_test)
            
            df_decoding = pd.concat([df_decoding, pd.DataFrame({'subject' : np.repeat(s, len(stim_test)),
                                                                'train_data' : np.repeat(c0, len(stim_test)),
                                                                'test_data' : np.repeat(c1, len(stim_test)),
                                                                'pred' : clf.predict(resps_test),
                                                                'stim' : stim_test})], ignore_index=True)

#%% Plot

df_plot = df_decoding.copy()

df_plot['correct'] = df_plot['pred'] == df_plot['stim']

df_plot = df_plot.groupby(['subject','train_data','test_data'])['correct'].mean().reset_index()


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

    f,a = plt.subplots(1,2, figsize=(1.3,1.1))


    (
        so.Plot(df_plot[df_plot.train_data=='naive'], x='test_data', y='correct')
        .add(so.Dots(pointsize=3, stroke=0.5), so.Jitter(y=0, seed=1), color='test_data', legend=False)
        .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), so.Jitter(y=0, seed=1), group='subject', legend=False)
        .on(a[0])
        .label(x='',  y='Classification accuracy (% correct)')
        .scale(color='colorblind',
               x=so.Nominal(['naive','proficient']))
        .limit(y=(0.5,1.05))
        .plot()
    )

    (
        so.Plot(df_plot[df_plot.train_data=='proficient'], x='test_data', y='correct')
        .add(so.Dots(pointsize=3, stroke=0.5), so.Jitter(y=0, seed=2), color='test_data', legend=False)
        .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), so.Jitter(y=0, seed=2), group='subject', legend=False)
        .on(a[1])
        .label(x='', y='')
        .scale(color='colorblind',
               x=so.Nominal(['naive','proficient']))
        .limit(y=(0.5,1.05))
        .plot()
    )

    for i,ax in enumerate(a):
        # ax.set_xticklabels(['Naive','Proficient'])
        if i == 1:
            left=True
            ax.set_yticks([])
            ax.set_title('Proficient')
        else:
            left=False
            ax.set_title('Naive')
        
        sns.despine(ax=ax, left=left, bottom=True, trim=True)
        ax.set_xticks([])


#%% Decoding all stimuli with tracked neurons - fit model to naive, test naive and proficient. fit model to proficient, test naive and proficient.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

df_tracked = df_resps[df_resps.tracked].copy()

# Remove cells whose ROIs are not in V1 at least one recordings
not_in_V1 = df_tracked.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_tracked = df_tracked.drop(not_in_V1).reset_index(drop=True)

df_decoding = pd.DataFrame()

pred = np.zeros((len(match_subjects), 2), dtype=object)
prob = pred.copy()
true_stim = pred.copy()

for i,s in enumerate(df_tracked.subject.unique()):

    scaler = StandardScaler()

    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    
    for c0 in df_tracked.condition.unique():
        for c1 in df_tracked.condition.unique():
        
            train_ind = (df_tracked.subject==s) & df_tracked.train_ind & (df_tracked.condition==c0) & ~np.isinf(df_tracked.stim_ori)
            train_ind = train_ind & (df_tracked.stim_ori != np.inf)
            
            stim_train = df_tracked[train_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
            resps_train = pd.pivot(df_tracked[train_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()
                    
            resps_train = scaler.fit_transform(resps_train)
            
            clf.fit(resps_train, stim_train)
            
            if c0 == c1:
                test_ind = (df_tracked.subject==s) & ~df_tracked.train_ind & (df_tracked.condition==c1) & ~np.isinf(df_tracked.stim_ori)
                test_ind = test_ind & (df_tracked.stim_ori != np.inf)
            else:
                test_ind = (df_tracked.subject==s) & (df_tracked.condition==c1) & ~np.isinf(df_tracked.stim_ori)
                test_ind = test_ind & (df_tracked.stim_ori != np.inf)
                
            stim_test = df_tracked[test_ind].groupby(['trial_num']).agg({'stim_ori' : 'first'}).to_numpy().reshape(-1,)
            
            resps_test = pd.pivot(df_tracked[test_ind], index='trial_num', columns=['cell_num_tracked'], values=['trial_resps']).to_numpy()       
            resps_test = scaler.transform(resps_test)
            
            df_decoding = pd.concat([df_decoding, pd.DataFrame({'subject' : np.repeat(s, len(stim_test)),
                                                                'train_data' : np.repeat(c0, len(stim_test)),
                                                                'test_data' : np.repeat(c1, len(stim_test)),
                                                                'pred' : clf.predict(resps_test),
                                                                'stim' : stim_test})], ignore_index=True)

#%% Plot - decoding all stim, strip plots

df_plot = df_decoding.copy()

df_plot['correct'] = df_plot['pred'] == df_plot['stim']

df_plot = df_plot.groupby(['subject','train_data','test_data','stim'])['correct'].mean().reset_index()


style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

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

    # f,a = plt.subplots(1,2, figsize=(1.3,1.1))
    # f = plt.figure()

    # (
    #     so.Plot(df_plot[df_plot.train_data=='naive'], x='test_data', y='correct')
    #     .facet(col='stim', wrap=4)
    #     .add(so.Dots(pointsize=3, stroke=0.5), so.Jitter(y=0, seed=1), color='test_data', legend=False)
    #     .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), so.Jitter(y=0, seed=1), group='subject', legend=False)
    #     .on(f)
    #     .label(x='',  y='Classification accuracy (% correct)')
    #     .scale(color='colorblind',
    #            x=so.Nominal(['naive','proficient']))
    #     .limit(y=(0.1,1.1))
    #     .theme({**style})
    #     .plot()
    # )

    # f.show()

    # f = plt.figure()

    # (
    #     so.Plot(df_plot[df_plot.train_data=='proficient'], x='test_data', y='correct')
    #     .facet(col='stim', wrap=4)
    #     .add(so.Dots(pointsize=3, stroke=0.5), so.Jitter(y=0, seed=2), color='test_data', legend=False)
    #     .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), so.Jitter(y=0, seed=2), group='subject', legend=False)
    #     .on(f)
    #     .label(x='', y='')
    #     .scale(color='colorblind',
    #            x=so.Nominal(['naive','proficient']))
    #     .limit(y=(0.1,1.1))
    #     .theme({**style})
    #     .plot()
    # )

    # f.show()


    f = plt.figure()

    (
        so.Plot(df_plot, x='test_data', y='correct', marker='train_data',  color='test_data')
        .facet(col='stim', wrap=4)
        .add(so.Dots(pointsize=3, stroke=0.5), so.Jitter(y=0, seed=1),legend=False)
        .on(f)
        .label(x='',  y='Classification accuracy (% correct)')
        .scale(color='colorblind',
               x=so.Nominal(['naive','proficient']))
        .limit(y=(0.1,1.05))
        .theme({**style})
        .plot()
    )

    f.show()

    # for i,ax in enumerate(a):
    #     # ax.set_xticklabels(['Naive','Proficient'])
    #     if i == 1:
    #         left=True
    #         ax.set_yticks([])
    #         ax.set_title('Proficient')
    #     else:
    #         left=False
    #         ax.set_title('Naive')
        
    #     sns.despine(ax=ax, left=left, bottom=True, trim=True)
    #     ax.set_xticks([])


#%% Plot - decoding all stim, scatter plots

df_plot = df_decoding.copy()

df_plot['correct'] = df_plot['pred'] == df_plot['stim']

df_plot = df_plot.groupby(['subject','train_data','test_data','stim'])['correct'].mean().reset_index()

df_plot = pd.pivot_table(df_plot, index=['subject','stim'], columns=['train_data','test_data'], values='correct')
df_plot.columns = [' '.join(col).strip() for col in df_plot.columns.values]

style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

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

    f,a = plt.subplots(1,2, figsize=(1.3,1.1))

    (
        so.Plot(df_plot, x='naive naive', y='naive proficient', color='stim')
        .add(so.Dots(pointsize=3, stroke=0.5), legend=False)
        .on(a[0])
        .label(x='Train: Naive, Test: Naive',  y='Train: Naive, Test: Proficient')
        .limit(x=[0.2,1.05], y=[0.2,1.05])
        .scale(color='colorblind')
        .theme({**style})
        .plot()
    )

    (
        so.Plot(df_plot, x='proficient proficient', y='proficient naive', color='stim')
        .add(so.Dots(pointsize=3, stroke=0.5), legend=False)
        .on(a[1])
        .label(x='Train: Proficient, Test: Proficient',  y='Train: Proficient, Test: Naive')
        .limit(x=[0.2,1.05], y=[0.2,1.05])
        .scale(color='colorblind')
        .theme({**style})
        .plot()
    )

    f.show()

    # for i,ax in enumerate(a):
    #     # ax.set_xticklabels(['Naive','Proficient'])
    #     if i == 1:
    #         left=True
    #         ax.set_yticks([])
    #         ax.set_title('Proficient')
    #     else:
    #         left=False
    #         ax.set_title('Naive')
        
    #     sns.despine(ax=ax, left=left, bottom=True, trim=True)
    #     ax.set_xticks([])


#%% Decoding results, all time, using polar plots

from scipy.stats import ttest_rel


df_plot = df_decoding.copy()

df_plot['correct'] = df_plot['pred'] == df_plot['stim']

df_plot = df_plot.groupby(['subject','train_data','test_data','stim'])['correct'].mean().reset_index()

df_plot['x_correct'] = df_plot.apply(lambda x: np.cos(np.deg2rad(x['stim']*2 + 5))*x['correct'] if x['test_data']=='naive' else np.cos(np.deg2rad(x['stim']*2 - 5))*x['correct'], axis=1)
df_plot['y_correct'] = df_plot.apply(lambda x: np.sin(np.deg2rad(x['stim']*2 + 5))*x['correct'] if x['test_data']=='naive' else np.sin(np.deg2rad(x['stim']*2 - 5))*x['correct'], axis=1)

df_plot['group'] = df_plot.apply(lambda x: x['subject'] + str(x['stim']), axis=1)

p_vals = np.zeros((2,8))

for ci, c in enumerate(df_plot.train_data.unique()):
    ind = df_plot.train_data == c
    for si, s in enumerate(df_plot.stim.unique()):
        ind_s = ind & (df_plot.stim==s)
        p_vals[ci,si] =  ttest_rel(df_plot[ind_s & (df_plot.test_data=='naive')].correct.to_numpy(),
                                   df_plot[ind_s & (df_plot.test_data=='proficient')].correct.to_numpy())[1]


style = axes_style("ticks")

style['axes.spines.right'] = False
style['axes.spines.top'] = False

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

    f,a = plt.subplots(1,2, figsize=(3,1.5), sharex=True, sharey=True)

    (
        so.Plot(df_plot[df_plot.train_data=='naive'], x='x_correct', y='y_correct')
        .add(so.Dots(pointsize=3, stroke=0.5), color='test_data', legend=False)
        .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), group='group', legend=False)
        .on(a[0])
        .label(x='',  y='')
        .scale(color='colorblind')
        .limit(x=(-1.05,1.05), y=(-1.05,1.05))
        .theme({**style})
        .plot()
    )


    (
        so.Plot(df_plot[df_plot.train_data=='proficient'], x='x_correct', y='y_correct')
        .add(so.Dots(pointsize=3, stroke=0.5), color='test_data', legend=False)
        .add(so.Lines(linewidth=0.5, color='k', artist_kws={'zorder' : -1}), group='group', legend=False)
        .on(a[1])
        .label(x='', y='')
        .scale(color='colorblind')
        .limit(x=(-1.1,1.1), y=(-1.1,1.1))
        .theme({**style})
        .plot()
    )

    text_offset = 0.075

    titles = ['Training set: Naive \n', 'Training set: Proficient \n']

    for a,t in zip(a, titles):
        a.set_box_aspect(1)
        a.set_title(t)

        p_levels = [1, 0.5, 0.125]
        p_labels = [f'{int(np.ceil(p*100))}%' for p in p_levels]
        p_labels[-1] = '12.5%'
                    
        for p,l in zip(p_levels, p_labels):
            if p == 1:
                linestyle = '-k'
            else:
                linestyle = '--k'
            a.plot(np.cos(np.linspace(0,2*np.pi,1000))*p, np.sin(np.linspace(0,2*np.pi,1000))*p, linestyle)
            a.text(np.cos(np.deg2rad(68.5))*p+text_offset, np.sin(np.deg2rad(68.5))*p+text_offset, l, fontsize=5, horizontalalignment='center')
            a.axis('off')


        va = ['center','bottom','bottom','bottom','center','top','top','top']
        ha = ['left','left','center','right','right','right','center','left']

        for s,v,h in zip(df_plot.stim.unique()*2, va, ha):
            a.text(np.cos(np.deg2rad(s))*(1+text_offset), np.sin(np.deg2rad(s))*(1+text_offset), str(int(s/2))+'$\degree$',
                   horizontalalignment=h, verticalalignment=v, fontsize=5)
        
    # f.show()
    f.savefig(join(save_fig_dir, 'train_by_condition_decoding_all_ori.svg'), format='svg')




#%% D-prime of tracked cells

df_tracked = df_resps[df_resps.tracked].copy()

# Remove cells whose ROIs are not in V1 at least one recordings
not_in_V1 = df_tracked.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_tracked = df_tracked.drop(not_in_V1).reset_index(drop=True)


df_variance = df_tracked.groupby(['subject','cell_num_tracked','condition','stim_ori'])['trial_resps'].var().reset_index()

df_mu = df_tracked.groupby(['subject','cell_num_tracked','condition','stim_ori'])['trial_resps'].mean().reset_index()

df_variance = df_variance[(df_variance.stim_ori==45) | (df_variance.stim_ori==90)]
df_std = df_variance.groupby(['subject','cell_num_tracked','condition'])['trial_resps'].apply(lambda x: np.sqrt(x.sum()/2)).reset_index()

df_mu = df_mu[(df_mu.stim_ori==45) | (df_mu.stim_ori==90)]
df_mu = df_mu.groupby(['subject','cell_num_tracked','condition'])['trial_resps'].diff().dropna().reset_index(drop=True)

df_d = df_std.copy()
df_d['d'] = df_mu/df_std.trial_resps

df_d['abs_d'] = df_d.d.abs()

(
    so.Plot(df_d, x='condition', y='abs_d', group='subject')
    .add(so.Dots(), so.Agg())
    .add(so.Lines(), so.Agg())
    .show()
)


f,a = plt.subplots(1,1)

df_pivot = pd.pivot(df_d, index=['subject','cell_num_tracked'], columns='condition', values='d').reset_index()

(
    so.Plot(df_pivot, x='naive', y='proficient')
    .add(so.Dots())
    .on(a)
    .plot()
)


a.plot([-20,20],[-20,20],'--k')

#%%


import statsmodels.api as sm
import statsmodels.formula.api as smf


df_tracked = df_resps[df_resps.tracked].copy()

# Remove cells whose ROIs are not in V1 at least one recordings
not_in_V1 = df_tracked.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_tracked = df_tracked.drop(not_in_V1).reset_index(drop=True)


df_variance = df_tracked.groupby(['subject','cell_num_tracked','condition','stim_ori'])['trial_resps'].var().reset_index()

df_mu = df_tracked.groupby(['subject','cell_num_tracked','condition','stim_ori'])['trial_resps'].mean().reset_index()

df_variance = df_variance[(df_variance.stim_ori==45) | (df_variance.stim_ori==90)]
df_std = df_variance.groupby(['subject','cell_num_tracked','condition'])['trial_resps'].apply(lambda x: np.sqrt(x.sum()/2)).reset_index()

df_mu = df_mu[(df_mu.stim_ori==45) | (df_mu.stim_ori==90)]
df_mu = df_mu.groupby(['subject','cell_num_tracked','condition'])['trial_resps'].diff().dropna().reset_index(drop=True)

df_d = df_std.copy()
df_d['d'] = df_mu/df_d.trial_resps

df_d['abs_d'] = df_d.d.abs()

df_d['mu_diff'] = df_mu
df_d['sigma'] = df_std['trial_resps']

md = smf.mixedlm("abs_d ~ C(condition)", df_d, groups=df_d['subject'])

mdf = md.fit()

print(mdf.summary())

#%% Decision model using tracked neurons only - subset cells with repeats


df_model = df_resps[df_resps.tracked].copy()

not_in_V1 = df_model.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_model= df_model.drop(not_in_V1).reset_index(drop=True)

def logistic(x,k):
    return expit(x*k)


df_output = pd.DataFrame()

k = 1000
n_cells = 100
n_repeats = 100

np.random.seed(0)

for r in range(n_repeats):
    for s in df_model.subject.unique():
        for c in df_model.condition.unique():
                        
            print(f'Subject {s}, condition {c}, repeat {r}')
            
            df_sub = df_model[(df_model.subject==s) & (df_model.condition==c)].copy()
            
            ind_cells = np.random.choice(df_sub.cell_num_tracked.unique(), n_cells, replace=False)
            
            df_sub = df_sub[df_sub.cell_num_tracked.isin(ind_cells)]
            
            if c == 'naive':
            
                df_weights = df_sub.groupby(['cell_num_tracked','stim_ori','train_ind'])['trial_resps'].mean().reset_index()
                
                df_weights = df_weights[((df_weights.stim_ori==45) | (df_weights.stim_ori==90)) & df_weights.train_ind]
                
                df_weights = pd.pivot(df_weights, index='cell_num_tracked', columns='stim_ori', values='trial_resps')
                
                weights = df_weights.to_numpy().T
                            
                weights /= weights.sum(axis=1, keepdims=True)
                
                
            df_trials = df_sub[((df_sub.stim_ori==45) | (df_sub.stim_ori==90)) & ~df_sub.train_ind]
            
            stim = df_trials.groupby(['trial_num'])['stim_ori'].first().to_numpy()
            
            df_trials = pd.pivot(df_trials, index='trial_num', columns='cell_num_tracked', values='trial_resps')
            
            trials = df_trials.to_numpy()

            l1 = trials.sum(1, keepdims=True)

            trials /= l1
            
            z = trials @ weights.T
                                
            df_output = pd.concat([df_output, pd.DataFrame({'trial_num' : np.arange(len(stim)),
                                                            'stim' : stim,
                                                            'z_c' : z[:,0],
                                                            'z_a' : z[:,1],
                                                            'prob_c' : logistic(np.diff(z,1),k).ravel(),
                                                            'l1' : l1.ravel(),
                                                            'subject' : np.repeat(s, len(stim)),
                                                            'condition' : np.repeat(c, len(stim)),
                                                            'repeat_num' : np.repeat(r, len(stim))})], ignore_index=True)
        

    
#%%

df_plot = df_output.groupby(['subject','trial_num','stim','condition'], observed=True)['prob_c'].mean().reset_index()


(
    so.Plot(df_plot, x='stim', y='prob_c', color='condition')
    .facet(col='subject')
    .add(so.Dots(), so.Dodge(), legend=True)
)


#%% Decision model using tracked neurons only


df_model = df_resps[df_resps.tracked].copy()

not_in_V1 = df_model.groupby(['cell_num_tracked','subject'], observed = True).filter(lambda x: ~np.any(x['V1_ROI'])).index
df_model= df_model.drop(not_in_V1).reset_index(drop=True)

def pop_sparseness(x):
    pass


df_output = pd.DataFrame()

k = 1

for s in df_model.subject.unique():
    for c in df_model.condition.unique():
                    
        print(f'Subject {s}, condition {c}')
        
        df_sub = df_model[(df_model.subject==s) & (df_model.condition==c)].copy()
        
        if c == 'naive':
        
            df_weights = df_sub.groupby(['cell_num_tracked','stim_ori','train_ind'])['trial_resps'].mean().reset_index()
            
            df_weights = df_weights[((df_weights.stim_ori==45) | (df_weights.stim_ori==90)) & df_weights.train_ind]
            
            df_weights = pd.pivot(df_weights, index='cell_num_tracked', columns='stim_ori', values='trial_resps')
            
            weights = df_weights.to_numpy().T
                        
            weights /= weights.sum(axis=1, keepdims=True)
            
            
        df_trials = df_sub[((df_sub.stim_ori==45) | (df_sub.stim_ori==90)) & ~df_sub.train_ind]
        
        stim = df_trials.groupby(['trial_num'])['stim_ori'].first().to_numpy()
        
        df_trials = pd.pivot(df_trials, index='trial_num', columns='cell_num_tracked', values='trial_resps')
        
        trials = df_trials.to_numpy()

        l1 = trials.sum(1, keepdims=True)

        trials /= l1
        
        z = trials @ weights.T
                            
        df_output = pd.concat([df_output, pd.DataFrame({'trial_num' : np.arange(len(stim)),
                                                        'stim' : stim,
                                                        'z_c' : z[:,0],
                                                        'z_a' : z[:,1],
                                                        'prob_c' : expit(np.diff(z,1)*k).ravel(),
                                                        'sup' : l1.ravel(),
                                                        'subject' : np.repeat(s, len(stim)),
                                                        'condition' : np.repeat(c, len(stim))})], ignore_index=True)
        
#%%

df_plot = df_output.groupby(['subject','trial_num','stim','condition'], observed=True)['prob_c'].mean().reset_index()

(
    so.Plot(df_plot, x='stim', y='prob_c', color='condition')
    .facet(col='subject')
    .add(so.Dots(), so.Dodge(), legend=True)
    .scale(color = so.Nominal(order=['naive','proficient']))
)

#%% Decision model using all cells

def ps(fr):
    top = (fr/fr.shape[1]).sum(1)**2
    bottom = (fr**2/fr.shape[1]).sum(1)
    s = 1 - (top/bottom)

    return s


df_model = df_resps[(df_resps.V1_ROI==1) & (df_resps.cell_plane>0)].copy()

df_output = pd.DataFrame()

k = 2500
n_cells = 1600
n_repeats = 2000

np.random.seed(0)

for s in df_model.subject.unique():
    for c in df_model.condition.unique():
        
        df_sub = df_model[(df_model.subject==s) & (df_model.condition==c)]

        df_mu = df_sub.groupby(['cell_plane', 'cell_num', 'stim_ori', 'train_ind'])['trial_resps'].mean().reset_index()
        
        df_mu = df_mu[((df_mu.stim_ori==45) | (df_mu.stim_ori==90)) & df_mu.train_ind]
        
        df_mu = pd.pivot(df_mu, index=['cell_plane', 'cell_num'], columns='stim_ori', values='trial_resps')
        
        mu = df_mu.to_numpy()
        
        df_trials = df_sub[((df_sub.stim_ori==45) | (df_sub.stim_ori==90)) & ~df_sub.train_ind]
        
        stim = df_trials.groupby(['trial_num'])['stim_ori'].first().to_numpy()
        
        trials = pd.pivot(df_trials, index='trial_num', columns=['cell_plane', 'cell_num'], values='trial_resps').to_numpy()
            
        for r in range(n_repeats):

            print(f'Subject {s}, condition {c}, repeat {r}')
                                
            # Random subset of size n_cells
            cell_ind = np.random.choice(mu.shape[0], n_cells, replace=False)
            
            weights = (mu[cell_ind,:]/mu[cell_ind,:].sum(axis=0, keepdims=True)).T
            
            # weights = mu[cell_ind,:].T
            
            test_trials = trials[:, cell_ind]

            sup = test_trials.sum(1, keepdims=True)
            
            e = test_trials @ weights.T
            
            z = e/sup
                                
            df_output = pd.concat([df_output, pd.DataFrame({'trial_num' : np.arange(len(stim)),
                                                            'stim' : stim,
                                                            'z_c' : z[:,0],
                                                            'z_a' : z[:,1],
                                                            'prob_c' : logistic(z[:,0]-z[:,1],k).ravel(), # Larger for 45 stim
                                                            'sup' : sup.ravel(),
                                                            'e_c' : e[:,0],
                                                            'e_a' : e[:,1],
                                                            'ps' : ps(test_trials),
                                                            'subject' : np.repeat(s, len(stim)),
                                                            'condition' : np.repeat(c, len(stim)),
                                                            'repeat_num' : np.repeat(r, len(stim))})], ignore_index=True)


#%%


df_plot = df_output.groupby(['subject','trial_num','stim','condition'], observed=True)[['prob_c','sup','e_c','e_a','z_c','z_a','ps']].mean().reset_index()

df_plot['z_diff'] = df_plot['z_c'] - df_plot['z_a']

df_plot['e_diff'] = df_plot['e_c'] - df_plot['e_a']

df_plot['e_sum'] = df_plot['e_c'] + df_plot['e_a']

df_plot['prob_correct'] = [p if s == 45 else 1-p if s == 90 else None for p,s in zip(df_plot.prob_c.to_numpy(), df_plot.stim.to_numpy())]

# df_plot.to_csv(r'C:\Users\Samuel\Desktop\model_output.csv')


def best_limits(x_ticks=None, y_ticks=None, gap_size=0.025):
    
    limits = []
    
    for t in [x_ticks, y_ticks]:
        
        if t is None:
            continue
    
        if type(t) is list:
            t = np.array(t)
        
        min_tick = t.min()
        
        range_t = t[-1] - t[0]
        
        limits.append((min_tick - range_t*gap_size, t[-1] + range_t*gap_size))
    
    return limits



    # (
    #     so.Plot(df_plot, x='stim', y='prob_c', color='condition')
    #     .facet(col='subject')
    #     .add(so.Dots(), so.Dodge(), legend=True)
    #     .scale(color = so.Nominal(order=['naive','proficient']))
    #     .show()
    # )
    
    
    
    
 
    
# f,a = plt.subplots(1,1)

# (
#     so.Plot(df_plot, x='sup', y='z_diff', marker='stim', color='condition')
#     .add(so.Dots(pointsize=2), legend=False)
#     .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
#         marker=so.Nominal(['o','v']))
#     .label(x='Feedforward inhibition', y='Z difference')
#     .limit(x=(250,2000), y=(-0.0008, 0.0004))
#     .on(a)
#     .plot()
# )

# sns.despine(ax=a, trim=True)

# a.set_box_aspect(1)    
    
    
    
f,a = plt.subplots(1,1)

(
    so.Plot(df_plot, x='sup', y='e_c', marker='stim', color='condition')
    .add(so.Dots(pointsize=2), legend=False)
    .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
        marker=so.Nominal(['o','v']))
    .label(x='Feedforward inhibition', y=r'$f \cdot w_{C}$')
    # .limit(x=(250,2000), y=(-0.0008, 0.0004))
    .on(a)
    .plot()
)

sns.despine(ax=a, trim=True)

a.set_box_aspect(1)    



    
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

    f,a = plt.subplots(1,1, figsize=(1.3,1.3))

     
    x_ticks = [0.5,0.6,0.7,0.8,0.9]
    y_ticks = [0, 0.5, 1, 1.5]
    
    limits = best_limits(x_ticks, y_ticks)

    (
        so.Plot(df_plot, x='ps', y='e_c', marker='stim', color='condition')
        .add(so.Dots(pointsize=1, stroke=0.25), legend=False)
        .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
               marker=so.Nominal(['o','v']),
               x=so.Continuous().tick(at=x_ticks),
               y=so.Continuous().tick(at=y_ticks))
        .label(x='Population sparseness', y=r'$f \cdot w_{C}$')
        .limit(x=limits[0], y=limits[1])
        .on(a)
        .plot()
    )

    sns.despine(ax=a, trim=True)

    a.set_box_aspect(1)
    f.tight_layout()    
    

    
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

    f,a = plt.subplots(1,1, figsize=(1.35,1.35))

     
    x_ticks = [0.5,0.6,0.7,0.8,0.9]
    y_ticks = [-0.4, -0.2, 0, 0.2, 0.4]
    
    limits = best_limits(x_ticks, y_ticks)

    (
        so.Plot(df_plot, x='ps', y='e_diff', marker='stim', color='condition')
        .add(so.Dots(pointsize=1, stroke=0.25), legend=False)
        .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
               marker=so.Nominal(['o','v']),
               x=so.Continuous().tick(at=x_ticks),
               y=so.Continuous().tick(at=y_ticks))
        .label(x='Population sparseness', y=r'$f \cdot w_{C} - f \cdot w_{A}$')
        .limit(x=limits[0], y=limits[1])
        .on(a)
        .plot()
    )

    sns.despine(ax=a, trim=True)

    a.set_box_aspect(1)
    f.tight_layout()    


    a.hlines(0,*a.get_xlim(), colors='black', linestyles='dashed')


f,a = plt.subplots(1,1)

(
    so.Plot(df_plot, x='ps', y='e_diff', marker='stim', color='condition')
    .add(so.Dots(pointsize=2), legend=False)
    .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
        marker=so.Nominal(['o','v']))
    # .label(x='Inhibition', y='Z difference')
    # .limit(x=(250,2000), y=(-0.0008, 0.0004))
    .on(a)
    .plot()
)

sns.despine(ax=a, trim=True)

a.set_box_aspect(1)


f,a = plt.subplots(1,1)

(
    so.Plot(df_plot, x='ps', y='e_sum', marker='stim', color='condition')
    .add(so.Dots(pointsize=2), legend=False)
    .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
        marker=so.Nominal(['o','v']))
    # .label(x='Inhibition', y='Z difference')
    # .limit(x=(250,2000), y=(-0.0008, 0.0004))
    .on(a)
    .plot()
)

sns.despine(ax=a, trim=True)

a.set_box_aspect(1)



f,a = plt.subplots(1,1)

(
    so.Plot(df_plot, x='e_c', y='e_a', marker='stim', color='condition')
    .add(so.Dots(pointsize=2), legend=False)
    .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
        marker=so.Nominal(['o','v']))
    # .label(x='e_c', y='Z difference')
    # .limit(x=(250,2000), y=(-0.0008, 0.0004))
    .on(a)
    .plot()
)

sns.despine(ax=a, trim=True)

a.set_box_aspect(1)


f,a = plt.subplots(1,1)

(
    so.Plot(df_plot, x='sup', y='e_diff', marker='stim', color='condition')
    .add(so.Dots(pointsize=2), legend=False)
    .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
        marker=so.Nominal(['o','v']))
    # .label(x='e_c', y='Z difference')
    # .limit(x=(250,2000), y=(-0.0008, 0.0004))
    .on(a)
    .plot()
)

sns.despine(ax=a, trim=True)

a.set_box_aspect(1)

    
    
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

    # f,a = plt.subplots(1,1, figsize=(1.3,1.3))
    f,a = plt.subplots(1,1)

    
    
    x_ticks = [0.5,0.6,0.7,0.8,0.9]
    
    limits = best_limits(x_ticks)
    

    (
        so.Plot(df_plot, x='ps', y='e_c', marker='stim', color='condition')
        .add(so.Dots(pointsize=2, stroke=0.25), legend=False)
        # .add(so.Dots(), legend=False)
        .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
               marker=so.Nominal(['o','v']),
               x=so.Continuous().tick(at=[0.5,0.6,0.7,0.8,0.9]))
        .label(x='Population sparseness', y='Feedforward excitation')
        .limit(x=limits[0])
        .on(a)
        .plot()
    )
    
    sns.despine(ax=a, trim=True)
    a.set_box_aspect(1)
    f.tight_layout()    



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

    f,a = plt.subplots(1,1, figsize=(1.3,1.4))
    # f,a = plt.subplots(1,1)

    
    
    x_ticks = [0.5,0.6,0.7,0.8,0.9]
    y_ticks = np.linspace(0,2000,5)
    
    limits = best_limits(x_ticks, [0,2000])
    

    (
        so.Plot(df_plot, x='ps', y='sup', marker='stim', color='condition')
        .add(so.Dots(pointsize=1, stroke=0.25), legend=False)
        # .add(so.Dots(), legend=False)
        .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
               marker=so.Nominal(['o','v']),
               x=so.Continuous().tick(at=x_ticks),
               y=so.Continuous().tick(at=y_ticks))
        .label(x='Population sparseness', y='Feedforward inhibition')
        .limit(x=limits[0], y=limits[1])
        .on(a)
        .plot()
    )
    
    
    sns.despine(ax=a, trim=True)
    a.set_box_aspect(1)
    a.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    f.tight_layout()
    
    f.savefig(join(fig_save_dir,'decision_model_ps_vs_inhibition.svg'), format='svg')
    
    



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

    # (
    #     so.Plot(df_plot, x='ps', y='z_diff', marker='stim', color='condition')
    #     .add(so.Dots())
    #     .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
    #         marker=so.Nominal(['o','v']))
    #     .label(x='Population sparseness', y='Z difference')
    #     .show()
    # )


    f,a = plt.subplots(1,1, figsize=(1.3,1.3))
    # f,a = plt.subplots(1,1)




    x_ticks = [0.5,0.6,0.7,0.8,0.9]
    y_ticks = [0.1,0.3,0.5,0.7,0.9]
    
    limits = best_limits(x_ticks, y_ticks)

    (
        so.Plot(df_plot, x='ps', y='prob_c', marker='stim', color='condition')
        .add(so.Dots(pointsize=1, stroke=0.25), legend=False)
        # .add(so.Dots(), legend=False)
        .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
               marker=so.Nominal(['o','v']),
               x=so.Continuous().tick(at=x_ticks),
               y=so.Continuous().tick(at=y_ticks))
        .label(x='Population sparseness', y='P(clockwise turn)')
        .limit(x=limits[0], y=limits[1])
        .on(a)
        .plot()
    )
    
     
    sns.despine(ax=a, trim=True)
    a.set_box_aspect(1)
    a.hlines(0.5, *a.get_xlim(), linestyle='dashed', colors='black')
    f.tight_layout()
    
    f.savefig(join(fig_save_dir,'decision_model_ps_vs_prob_C.svg'), format='svg')
    



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
    
    
    
    # f,a = plt.subplots(1,1, figsize=(1.3,1.3))
    f,a = plt.subplots(1,1)

    
    # x_ticks = [0, 500, 1000, 1500, 2000]
    # y_ticks = [0.1,0.3,0.5,0.7,0.9]
    
    # limits = best_limits(x_ticks, y_ticks)
    
    
    (
        so.Plot(df_plot, x='sup', y='prob_c', marker='stim', color='condition')
        # .add(so.Dots(pointsize=2, stroke=0.25), legend=False)
        .add(so.Dots(), legend=False)
        .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
               marker=so.Nominal(['o','v']),
            #    x=so.Continuous().tick(at=x_ticks),
               y=so.Continuous().tick(at=y_ticks))
        .label(x='Feedforward inhibition', y='P(clockwise turn)')
        # .limit(x=limits[0], y=limits[1])
        .on(a)
        .plot()
    )

    
    sns.despine(ax=a, trim=True)
    a.set_box_aspect(1)
    a.hlines(0.5, *a.get_xlim(), linestyle='dashed', colors='black')
    f.tight_layout()
    
    # f.savefig(join(fig_save_dir,'decision_model_inhibition_vs_prob_C.svg'), format='svg')





df_plot_prob = df_plot.groupby(['subject','condition'])['prob_correct'].mean().reset_index()

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
    
    
    
    # f,a = plt.subplots(1,1, figsize=(1,1.3))
    f,a = plt.subplots(1,1)
    
    
    xtick_labels = ['Naive','Proficient']   
    
    # y_ticks = [0.55,0.6,0.65,0.7,0.75,0.8]
    
    # limits = best_limits(y_ticks, gap_size=0.025)
    
    (
        so.Plot(df_plot_prob, x='condition', y='prob_correct')
        # .add(so.Dot(pointsize=2, stroke=0.25, alpha=1, edgecolor='black', artist_kws={'zorder' : 100}), color='condition', legend=False)
        .add(so.Dot(alpha=1, edgecolor='black', artist_kws={'zorder' : 100}), color='condition', legend=False)
        .add(so.Lines(linewidth=0.5, color='grey'), group='subject')
        # .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
        #        y=so.Continuous().tick(at=y_ticks))
        .scale(color=so.Nominal('colorblind', order=['naive','proficient']))
        # .limit(y=limits[0])
        .label(x='', y='Prob. correct choice')
        .on(a)
        .plot()
    )

    a.set_xticklabels(xtick_labels)
        
    sns.despine(ax=a, trim=True)
    f.tight_layout()
    
    # f.savefig(join(fig_save_dir,'decision_model_prob_correct_choice.svg'), format='svg')


#%% Calculate slope of tuning curves at 45/90, 68, 0/135

df_slope = pd.DataFrame()

df_tc = df_tuning[(df_tuning.V1_ROI==1) & (df_tuning.cell_plane>0)].copy()

tuning_curves = pd.pivot(df_tc, index=['subject','condition','cell_plane','cell_num'], columns='stim_ori', values='mean_resp')

cell_stats = df_tc.groupby(['subject','condition','cell_plane','cell_num'], observed=True)['pref_bin_ori'].first().reset_index()

# Find slopes at 45, 90, 68, 0, 135

slopes_at = [45, 90, 68, 0, 135]

stim_index = np.ceil(np.arange(0,180,22.5))



#%%


def find_tuning(df_resps, train_only = True):

    ind = ~np.isinf(df_resps.stim_ori)

    if train_only:
        ind = df_resps.train_ind & ind

    df_tuning = df_resps[ind].copy()
    df_tuning['stim_ori_rad'] = df_tuning['stim_ori'] * np.pi/90
    df_tuning['stim_dir_rad'] = df_tuning['stim_dir'] * np.pi/180
        
    df_tuning['exp(stim_ori_rad)*trial_resp'] = np.exp(df_tuning.stim_ori_rad*1j) * df_tuning.trial_resps
    df_tuning['exp(stim_dir_rad)*trial_resp'] = np.exp(df_tuning.stim_dir_rad*1j) * df_tuning.trial_resps

    df_tuning = df_tuning.groupby(['cell_num', 'cell_plane', 'subject', 'condition'], observed = True).agg({'exp(stim_ori_rad)*trial_resp' : 'sum',
                                                                                                            'exp(stim_dir_rad)*trial_resp' : 'sum',
                                                                                                            'trial_resps' : 'sum',
                                                                                                            'ROI_ret_azi' : 'first',
                                                                                                            'ROI_ret_elv' : 'first',
                                                                                                            'cell_num_tracked' : 'first',
                                                                                                            'tracked' : 'first',
                                                                                                            'V1_ROI' : 'first'})

    df_tuning['ori_tune_vec'] = df_tuning['exp(stim_ori_rad)*trial_resp']/df_tuning['trial_resps']
    df_tuning['dir_tune_vec'] = df_tuning['exp(stim_dir_rad)*trial_resp']/df_tuning['trial_resps']
    df_tuning['r_ori'] = df_tuning.ori_tune_vec.abs()
    df_tuning['r_dir'] = df_tuning.dir_tune_vec.abs()
    df_tuning['mean_ori_pref'] = np.mod(11.25+np.arctan2(np.imag(df_tuning.ori_tune_vec),np.real(df_tuning.ori_tune_vec))*90/np.pi,180)-11.25
    df_tuning['mean_dir_pref'] = np.mod(11.25+np.arctan2(np.imag(df_tuning.dir_tune_vec),np.real(df_tuning.dir_tune_vec))*180/np.pi,360)-11.25


    df_tuning = df_tuning.drop(columns = ['exp(stim_ori_rad)*trial_resp','trial_resps'])
    df_tuning = df_tuning.reset_index()

    mu_resp = df_resps[~np.isinf(df_resps.stim_ori)].groupby(['cell_num', 'cell_plane', 'subject', 'condition', 'stim_ori'], observed = True)['trial_resps'].mean().reset_index()
    df_tuning['modal_ori_pref'] = mu_resp.loc[mu_resp.groupby(['cell_num', 'cell_plane', 'subject', 'condition'], observed = True)['trial_resps'].idxmax(),'stim_ori'].reset_index(drop=True)
    mu_resp = df_resps[~np.isinf(df_resps.stim_dir)].groupby(['cell_num', 'cell_plane', 'subject', 'condition', 'stim_dir'], observed = True)['trial_resps'].mean().reset_index()
    df_tuning['modal_dir_pref'] = mu_resp.loc[mu_resp.groupby(['cell_num', 'cell_plane', 'subject', 'condition'], observed = True)['trial_resps'].idxmax(),'stim_dir'].reset_index(drop=True)

    return df_tuning


# Split trials into three datasets, first to assign cell class, second to fit piecewise linear, third to compare cell classes to trained

df_trials = df_resps.groupby(['subject','condition','trial_num'], observed=True).agg({'stim_ori': 'first',
                                                                                      'stim_dir': 'first'}).reset_index()

df_trials['block_num'] = np.floor(df_trials['trial_num'] / 17)
df_trials['set'] = df_trials['block_num'] % 3

df_resps_cv = df_resps.merge(df_trials[['subject','condition','trial_num','set']], on=['subject','condition','trial_num'], how='left')

df_tuning_cv = find_tuning(df_resps_cv[df_resps_cv.set==0], train_only=False)

# df_resps_cv = df_resps_cv.merge(df_tuning_cv[['subject','condition','mean_ori_pref','r_ori', 'cell_num','cell_plane']],
#                                 on=['subject','condition','cell_num','cell_plane'],
#                                 how='left')


df_tuning_cv['pref_bin_ori'] = pd.cut(df_tuning_cv.mean_ori_pref, np.linspace(-11.25,180-11.25,9), labels=np.ceil(np.arange(0,180,22.5)))
df_tuning_cv['r_bin_ori'] = pd.cut(df_tuning_cv.r_ori, np.array([0., 0.16, 0.32, 0.48, 0.64, 1.]), labels=np.arange(5))


# Tuning curves from train and test set
tuning_curves = df_resps_cv.groupby(['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori', 'set'], observed = True)['trial_resps'].mean().reset_index()

# Exclude training set and blank trials
tuning_curves_test = tuning_curves[~tuning_curves.train_ind & ~np.isinf(tuning_curves.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp'})
tuning_curves_train = tuning_curves[tuning_curves.train_ind & ~np.isinf(tuning_curves.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp_train'})

tuning_curves_all = df_resps.groupby(['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori'], observed = True)['trial_resps'].mean().reset_index()
tuning_curves_all = tuning_curves_all[~np.isinf(tuning_curves_all.stim_ori)].reset_index(drop=True).rename(columns = {'trial_resps' : 'mean_resp_all_trials'})

df_tuning = df_tuning.merge(tuning_curves_train, on = ['cell_num', 'cell_plane', 'condition', 'subject'], how = 'left').drop(columns='train_ind')
df_tuning = df_tuning.merge(tuning_curves_test, on = ['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori'], how = 'left').drop(columns = 'train_ind')
df_tuning = df_tuning.merge(tuning_curves_all, on = ['cell_num', 'cell_plane', 'condition', 'subject', 'stim_ori'], how = 'left')




# %%
