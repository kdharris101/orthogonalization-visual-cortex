#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from seaborn import axes_style
import seaborn.objects as so
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_1samp
from scipy.stats import wilcoxon

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

poort_file = r'C:\Users\Samuel\OneDrive - University College London\Poort data\DATmbsl.mat'
# poort_file = r'C:\Users\samue\OneDrive - University College London\Poort data\DATmbsl.mat'


data = loadmat(poort_file, squeeze_me=True)['DAT'][()]

#%%

n_cells = data[()][0][0].shape[0]

df_resps = pd.DataFrame({'stim' : np.concatenate([['go']*n_cells, ['nogo']*n_cells, ['go']*n_cells, ['nogo']*n_cells]),
                         'trained' : np.concatenate([np.zeros(n_cells*2), np.ones(n_cells*2)]),
                         'mouse' : np.tile(data[3],4),
                         'resps' : np.concatenate([data[0][0]-data[2][0],data[1][0]-data[2][0],data[0][1]-data[2][1],data[1][1]-data[2][1]]),
                         'cell_num' : np.tile(np.arange(n_cells),4)})



def best_limits(x_ticks=None, y_ticks=None, gap_size=0.025):
    
    limits = []
    
    for t in [x_ticks, y_ticks]:
        
        if t is None:
            continue
    
        if type(t) is list:
            t = np.array(t)
        
        min_tick = t.min()
        
        range_t = t[-1] - t[0]
        
        limits.append((min_tick - range_t*gap_size, t[-1]))
    
    return limits


#%% Orthogonalization

def cosine_similarity(a,b):
    
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

cs = np.zeros(2)


df_cs = df_resps.groupby(['trained','mouse'])[['stim','resps']].apply(lambda x: cosine_similarity(x[x.stim=='go'].resps.to_numpy(),
                                                                                x[x.stim=='nogo'].resps.to_numpy())).reset_index()


df_diff = df_cs[df_cs.trained==1][0].reset_index(drop=True) - df_cs[df_cs.trained==0][0].reset_index(drop=True)

n_cells = df_resps.groupby(['mouse'])['cell_num'].nunique()

df_cs['n_cells'] = np.tile(n_cells,2)

df_cs = df_cs.rename({0:'cs'}, axis=1)

df_cs = df_cs[df_cs.n_cells > 100]

df_cs['trained'] = [['naive','proficient'][int(i)] for i in df_cs.trained.to_numpy()]


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


    f,a = plt.subplots(1,1, figsize=(1.25,1.25))

    yticks = [0.2,0.4,0.6,0.8,1]
    limits = best_limits(yticks)

    p = (
            so.Plot(df_cs, y='cs', x='trained')
            .layout(engine='tight')
            .add(so.Dot(pointsize=2, edgecolor='black'), color='trained', legend=False)
            .add(so.Line(color='black', linewidth=0.5, artist_kws={'zorder' : -1}), group='mouse', legend=False)
            .label(x='Condition',
                   y='Cosine similarity')
            .scale(color=so.Nominal('colorblind', order=['naive','proficient']),
                   y=so.Continuous().tick(at=yticks))
            .limit(x=[-0.3,1.3],
                   y=limits[0])
            .on(a)
            .plot()
        )


    sns.despine(ax=a, trim=True)
    
    a.set_xticks([0,1])

    a.set_xticklabels(['Naive','Proficient'])

    f.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\Poort_cosine_similarity.svg', format='svg')


#%% Look at change in response by cell and mouse

df_pivot = pd.pivot(df_resps, index = ['mouse','cell_num'], columns = ['stim','trained'], values = 'resps')

f,a = plt.subplots(2,9)

for i,s in enumerate(['go','nogo']):
    
    df_plot = df_pivot[s]
    
    for im, m in enumerate(df_resps.mouse.unique()):
        ind = df_plot.index.get_level_values(0) == m
        sns.scatterplot(df_plot[ind], x = 0, y = 1, ax = a[i,im])
        a[i,im].set_xlim([-0.3,1.3])
        a[i,im].set_ylim([-0.3,1.3])
        a[i,im].plot([-0.3,1.3],[-0.3,1.3],'--k')
        a[i,im].set_box_aspect(1)



df_pivot = pd.pivot(df_resps, index = ['mouse','cell_num'], columns = ['trained','stim'], values = 'resps')

f,a = plt.subplots(2,9)

for i,s in enumerate([0,1]):
    
    df_plot = df_pivot[s]
    
    for im, m in enumerate(df_resps.mouse.unique()):
        ind = df_plot.index.get_level_values(0) == m
        sns.scatterplot(df_plot[ind], x = 'go', y = 'nogo', ax = a[i,im])
        a[i,im].set_xlim([-0.3,1.3])
        a[i,im].set_ylim([-0.3,1.3])
        a[i,im].plot([-0.3,1.3],[-0.3,1.3],'--k')
        a[i,im].set_box_aspect(1)

#%% Change in response


df_change = df_resps.sort_values(['cell_num','stim','trained']).groupby(['cell_num','stim']).diff().dropna()

tmp = df_resps.sort_values(['cell_num','stim','trained']).groupby(['cell_num','stim']).agg({'resps' : 'mean',
                                                                                            'mouse' : 'first'}).reset_index()

df_change['mouse'] = tmp.mouse.to_numpy()
df_change['cell_num'] = tmp.cell_num.to_numpy()
df_change['stim'] = tmp.stim.to_numpy()

n_cells = df_resps.groupby(['mouse'])['cell_num'].nunique()
g_mice = n_cells[n_cells>100].index

mask = df_change.mouse.isin(g_mice)

df_change = df_change[mask]

df_change = df_change.groupby(['stim','mouse'])['resps'].mean().reset_index()


p = (
        so.Plot(df_change, y='resps', x='stim', group='stim')
        .theme({**style})
        .layout(engine='tight',
                size=[1.5,1.5])
        .add(so.Dots(color='black', pointsize=2), so.Jitter())
        .add(so.Dash(color='black', width=0.5, linewidth=1), so.Agg())
        .add(so.Range(color='black', linewidth=1), so.Est(errorbar=('se',2)))
        .label(x='Stimulus',
            y='Change in df/f')
        .limit(y=[-0.05,0.03])
        .plot()
    )

# p._figure.axes[0].set_xticklabels(['Naive','Proficient'])

sns.despine(ax=p._figure.axes[0], trim=True)

p.save(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\Poort_change_resp.svg', format='svg')

#%%


df = df_resps.groupby(['cell_num','stim','trained'])['resps'].mean().reset_index()

f = sns.catplot(df, x = 'trained', y = 'resps', hue = 'stim', kind='bar', dodge=True)
f.set_xlabels('Trained (0 = no, 1 = yes)')
f.set_ylabels('Mean response')


#%% Stats for change in response

import statsmodels.api as sm
import statsmodels.formula.api as smf

df_diff_go = df_resps[(df_resps.stim=='go') & (df_resps.trained==0)].drop('trained', axis=1).reset_index(drop=True)
df_diff_go['resp_diff'] = df_resps[(df_resps.stim=='go') & (df_resps.trained==1)].resps.to_numpy() - df_resps[(df_resps.stim=='go') & (df_resps.trained==0)].resps.to_numpy()

df_diff_nogo = df_resps[(df_resps.stim=='nogo') & (df_resps.trained==0)].drop('trained', axis=1).reset_index(drop=True)
df_diff_nogo['resp_diff'] = df_resps[(df_resps.stim=='nogo') & (df_resps.trained==1)].resps.to_numpy() - df_resps[(df_resps.stim=='nogo') & (df_resps.trained==0)].resps.to_numpy()

df_diff = pd.concat([df_diff_go, df_diff_nogo], ignore_index=True)


md = smf.mixedlm("resps ~ C(trained)", df_resps[df_resps.stim=='nogo'], groups=df_resps[df_resps.stim=='nogo']['mouse'], re_formula="~C(trained)")

mdf = md.fit()

print(mdf.summary())



# %%

go_resps_naive = df_resps[(df_resps.stim=='go') & (df_resps.trained==0)].resps.to_numpy()
go_resps_trained = df_resps[(df_resps.stim=='go') & (df_resps.trained==1)].resps.to_numpy()

nogo_resps_naive = df_resps[(df_resps.stim=='nogo') & (df_resps.trained==0)].resps.to_numpy()
nogo_resps_trained = df_resps[(df_resps.stim=='nogo') & (df_resps.trained==1)].resps.to_numpy()


go_ind_naive = np.argsort(go_resps_naive)[::-1]
nogo_ind_naive = np.argsort(nogo_resps_naive)[::-1]
go_ind_trained= np.argsort(go_resps_trained)[::-1]
nogo_ind_trained = np.argsort(nogo_resps_trained)[::-1]

vmin = -0.08
vmax = 0.08

sns.set()
sns.set_style('ticks')
with sns.plotting_context("poster", font_scale = 1,
                          rc = {'lines.linewidth':0.5}):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['figure.dpi'] = 1000

    f,a = plt.subplots(2,2)
    sns.heatmap(go_resps_naive[go_ind_trained,None], ax=a[0,0], center = 0, vmin = vmin, vmax = vmax, cmap = "viridis")
    sns.heatmap(go_resps_trained[go_ind_trained,None], ax=a[0,1], center = 0, vmin = vmin, vmax = vmax, cmap = "viridis")

    sns.heatmap(nogo_resps_naive[nogo_ind_trained,None], ax=a[1,0], center=0, vmin = vmin, vmax = vmax, cmap = "viridis")
    sns.heatmap(nogo_resps_trained[nogo_ind_trained,None], ax=a[1,1], center=0, vmin = vmin, vmax = vmax, cmap = "viridis")


    for a in a.flatten():
        a.set_xticks([])
        a.set_yticks([])

go_diff = go_resps_trained - go_resps_naive
nogo_diff = nogo_resps_trained - nogo_resps_naive

go_ind = np.argsort(go_diff)[::-1]
nogo_ind = np.argsort(nogo_diff)[::-1]

cmap = 'seismic'

vmin = -0.1
vmax = 0.1

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
    
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['figure.dpi'] = 1000

    f,a=plt.subplots(1,2, figsize=(1.1,1.25))
    cbar_ax = f.add_axes([.91, .3, .03, .4])

    sns.heatmap(go_diff[go_ind,None], center=0, vmin=vmin, vmax=vmax, ax=a[0], cbar_ax=cbar_ax, cbar_kws={'label' : 'Change in df/f with training'}, cmap=cmap)
    sns.heatmap(nogo_diff[nogo_ind,None], center=0, vmin=vmin, vmax=vmax, ax=a[1], cbar_ax=cbar_ax, cbar_kws={'label' : 'Change in df/f with training'}, cmap=cmap)
    a[0].set_title('Lick cue')
    a[1].set_title('No lick cue')
    a[0].set_yticks([])
    a[0].set_xticks([])
    a[1].set_yticks([])
    a[1].set_xticks([])
    a[0].set_ylabel('Neurons')
    
    cbar_ax.set_yticks([vmin,0,vmax])
    cbar_ax.set_yticklabels([f'<{vmin}','0',f'>{vmax}'])



f,a = plt.subplots(1,2)

sns.histplot(go_diff, ax=a[0])
sns.histplot(nogo_diff, ax=a[1])

a[0].set_title('go')
a[1].set_title('nogo')


sns.displot(df_resps, x = 'resps', col = 'stim', hue = 'trained', kind='hist')

n_cells = go_resps_naive.shape[0]

df = pd.DataFrame({'resps' : np.concatenate([go_resps_naive, nogo_resps_naive, go_resps_trained, nogo_resps_trained]),
                   'stim' : np.concatenate([np.repeat('go', n_cells), np.repeat('nogo', n_cells), np.repeat('go', n_cells), np.repeat('nogo', n_cells)]),
                   'trained' : np.concatenate([np.repeat(False, n_cells*2), np.repeat(True, n_cells*2)]),
                   'cell_num' : np.tile(np.arange(n_cells),4)})

f,a = plt.subplots(1,2)
sns.lineplot(df[df.stim=='go'], x='trained', y='resps' , marker='o', units = 'cell_num', estimator=None, ax=a[0])
a[0].set_xticks([0, 1], ['Naive', 'Proficient'])
a[0].set_xlim(-0.2, 1.2)
a[0].set_xlabel('Condition')
a[0].set_ylabel('df/f')
a[0].set_ylim([-1,3])
# sns.despine(ax=a[0], trim=True)

sns.lineplot(df[df.stim=='nogo'], x='trained', y='resps' , marker='o', units = 'cell_num', estimator=None, ax=a[1])
a[1].set_xticks([0, 1], ['Naive', 'Proficient'])
a[1].set_xlim(-0.2, 1.2)
a[1].set_xlabel('Condition')
a[1].set_ylabel('df/f')
a[1].set_ylim([-1,3])
# sns.despine(ax=a[1], trim=True)


# %%

sns.catplot(df_resps, x = 'stim', y = 'resps', hue = 'trained', kind = 'bar', errorbar=('se',1))

# %% Sparseness

def pop_sparseness(fr):

    top = (np.abs(fr)/len(fr)).sum()**2
    bottom = (fr**2/len(fr)).sum()
    s = 1 - (top/bottom)
  
    return s

df_s = df_resps.groupby(['trained','stim','mouse'])['resps'].apply(lambda x: pop_sparseness(x)).reset_index()

n_cells = df_resps.groupby(['mouse'])['cell_num'].nunique()

df_s['n_cells'] = np.tile(n_cells,4)

f,a = plt.subplots(1,1)
sns.lineplot(df_s[df_s.n_cells>=100], x='trained', y='resps' , marker='o', units='mouse', estimator=None, ax=a, hue='stim', legend=False)
a.set_xticks([0, 1], ['Naive', 'Proficient'])
a.set_xlim(-0.2, 1.2)
a.set_xlabel('Condition')
a.set_ylabel('Population sparseness')
sns.despine(ax=a, trim=True)


#%%

def pop_sparseness(fr):

    top = (np.abs(fr)/len(fr)).sum()**2
    bottom = (fr**2/len(fr)).sum()
    s = 1 - (top/bottom)
  
    return s

df_s = df_resps.groupby(['trained','stim','mouse'])['resps'].apply(lambda x: pop_sparseness(x)).reset_index()

n_cells = df_resps.groupby(['mouse'])['cell_num'].nunique()

df_s['n_cells'] = np.tile(n_cells,4)

df_s = df_s[df_s.n_cells>100]

df_change = df_s.sort_values(['mouse','stim','trained']).groupby(['mouse','stim']).diff().dropna().reset_index(drop=True)

tmp = df_s.sort_values(['mouse','stim','trained']).groupby(['mouse','stim']).agg({'resps' : 'mean'}).reset_index()

df_change['mouse'] = tmp.mouse
df_change['stim'] = tmp.stim


p = (
        so.Plot(df_change, y='resps', x='stim', group='stim')
        .theme({**style})
        .layout(engine='tight',
                size=[1.5,1.5])
        .add(so.Dots(color='black', pointsize=2), so.Jitter())
        .add(so.Dash(color='black', width=0.5, linewidth=1), so.Agg())
        .add(so.Range(color='black', linewidth=1), so.Est(errorbar=('se',2)))
        .label(x='Condition',
            y='Change in pop. sparseness')
        .limit(y=[-0.2,0.45])
        .plot()
    )


sns.despine(ax=p._figure.axes[0], trim=True)


p.save(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\Poort_change_ps.svg', format='svg')


#%%

def pop_sparseness(fr):

    top = (np.abs(fr)/len(fr)).sum()**2
    bottom = (fr**2/len(fr)).sum()
    s = 1 - (top/bottom)
  
    return s

df_ps = df_resps.groupby(['trained','stim','mouse'])['resps'].apply(lambda x: pop_sparseness(x)).reset_index()

n_cells = df_resps.groupby(['mouse'])['cell_num'].nunique()

df_ps['n_cells'] = np.tile(n_cells,4)

df_ps = df_ps[df_ps.n_cells>100]

df_ps = df_ps.groupby(['mouse','trained'])['resps'].mean().reset_index()

df_ps['trained'] = [['naive','proficient'][int(i)] for i in df_ps.trained.to_numpy()]


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


    f,a = plt.subplots(1,1, figsize=(1.25,1.25))

    p = (
            so.Plot(df_ps, y='resps', x='trained')
            .theme({**style})
            .layout(engine='tight')
            .add(so.Dot(pointsize=2, edgecolor='black'), color='trained', legend=False)
            .add(so.Line(color='black', linewidth=0.5, artist_kws={'zorder' : -1}), group='mouse', legend=False)
            .label(x='Condition',
                   y='Pop. sparseness')
            .scale(color=so.Nominal('colorblind', order=['naive','proficient']))
            .limit(x=[-0.3,1.3],
                y=[0.4,1])
            .on(a)
            .plot()
        )


    sns.despine(ax=a, trim=True)
    
    a.set_xticks([0,1])

    a.set_xticklabels(['Naive','Proficient'])

    f.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\Poort_ps.svg', format='svg')
#%%

# Change in response as function of a cell's selectivity index


df_si = df_resps.copy()


df_si.loc[df_si.resps<0, 'resps'] = 0
df_si = df_si.groupby(['mouse', 'trained', 'cell_num'], group_keys=True)['resps'].apply(lambda x: (x.diff()/x.sum())).rename('SI').dropna().reset_index(level=3, drop=True).reset_index()
# df_si = df_si.groupby(['mouse', 'trained', 'cell_num'], group_keys=True)['resps'].apply(lambda x: x.diff().abs()).rename('SI').dropna().reset_index(level=3, drop=True).reset_index()

df_resps_diff = df_resps.groupby(['mouse','stim','cell_num'], group_keys=True)['resps'].apply(lambda x: x.diff()).rename('resp_diff').dropna().reset_index(level=3, drop=True).reset_index()

df_si = df_si[df_si.trained==0].drop('trained', axis=1)


df_resps_diff = df_resps_diff.merge(df_si, on=['mouse','cell_num'], how='left')

df_resps_diff['SI_bin'] = pd.cut(df_resps_diff.SI.abs(), bins=5)

df_resps_diff_mu = df_resps_diff.groupby(['mouse','stim','SI_bin'])['resp_diff'].mean().reset_index()

# %%


(
    so.Plot(df_resps_diff, x='SI', y='resp_diff', color='mouse')
    .facet(col='stim')
    .add(so.Dots())
    .scale(color='colorblind')
    .limit(y=(-0.1,0.1))
    .show()
)



sns.displot(df_resps_diff, x='SI', y='resp_diff', col='stim', kind='hist', facet_kws={'ylim': (-0.1,0.1)})

sns.catplot(df_resps_diff, x='SI_bin', y='resp_diff', col='stim', kind='point')

sns.catplot(df_resps_diff_mu, x='SI_bin', y='resp_diff', col='stim', kind='strip')




#%% 

df_naive = df_resps[df_resps.trained==0].copy()

df_naive = pd.pivot(df_naive, index=['mouse','cell_num'], columns='stim', values='resps').reset_index()

df_diff = df_resps.groupby(['mouse','stim','cell_num'], group_keys=True)['resps'].apply(lambda x: x.diff()).rename('resp_diff').dropna().reset_index(level=3, drop=True).reset_index()

df = df_naive.merge(df_diff, on=['mouse','cell_num'], how='left')


# (
#     so.Plot(df, x='go', y='nogo', color='resp_diff')
#     .facet(col='stim')
#     .add(so.Dots())
#     .limit(x=(-0.2,1.2), y=(-0.2,1.2))
#     .scale(color=so.Continuous('icefire').tick(at=(np.linspace(-0.8,0.8,11))))
#     .show()
# )


# sns.relplot(df[df.resp_diff<0], x='go', y='nogo', hue='resp_diff', col='stim', palette='magma', s=10)
# sns.relplot(df[df.resp_diff>0], x='go', y='nogo', hue='resp_diff', col='stim', palette='magma_r', s=10)

vmin=-0.2
vmax=0.2
cmap='vlag'

f,a = plt.subplots(1,2)

a[0].scatter(x=df[df.stim=='go'].go.to_numpy(), y=df[df.stim=='go'].nogo.to_numpy(), 
            c=df[df.stim=='go'].resp_diff.to_numpy(), vmin=vmin, vmax=vmax, cmap=cmap, s=10)
a[0].plot([-0.8,1.3],[-0.8,1.3],'--k')
# plt.xlim([-0.8,1.3])
# plt.ylim([-0.8,1.3])
a[0].set_xlabel('Naive go response')
a[0].set_ylabel('Naive nogo response')
a[0].set_box_aspect(1)
a[0].set_title('Change in go response')

a[1].scatter(x=df[df.stim=='nogo'].go.to_numpy(), y=df[df.stim=='nogo'].nogo.to_numpy(), 
            c=df[df.stim=='nogo'].resp_diff.to_numpy(),  vmin=vmin, vmax=vmax, cmap=cmap, s=10)
a[1].plot([-0.8,1.3],[-0.8,1.3],'--k')
# plt.xlim([-0.8,1.3])
# plt.ylim([-0.8,1.3])
a[1].set_xlabel('Naive go response')
a[1].set_ylabel('Naive nogo response')
a[1].set_box_aspect(1)
a[1].set_title('Change in nogo response')

points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

cax = f.add_axes([.92, .25, .02, .5])

f.colorbar(points, cax=cax)



f,a = plt.subplots(1,2)

a[0].scatter(x=df[df.stim=='go'].go.to_numpy(), y=df[df.stim=='go'].nogo.to_numpy(), 
            c=df[df.stim=='go'].resp_diff.to_numpy(), vmin=vmin, vmax=vmax, cmap=cmap, s=10)
a[0].plot([-0.8,1.3],[-0.8,1.3],'--k')
a[0].set_xlim([-0.3,0.3])
a[0].set_ylim([-0.3,0.3])
a[0].set_xlabel('Naive go response')
a[0].set_ylabel('Naive nogo response')
a[0].set_box_aspect(1)
a[0].set_title('Change in go response')

a[1].scatter(x=df[df.stim=='nogo'].go.to_numpy(), y=df[df.stim=='nogo'].nogo.to_numpy(), 
            c=df[df.stim=='nogo'].resp_diff.to_numpy(),  vmin=vmin, vmax=vmax, cmap=cmap, s=10)
a[1].plot([-0.8,1.3],[-0.8,1.3],'--k')
a[1].set_xlim([-0.3,0.3])
a[1].set_ylim([-0.3,0.3])
a[1].set_xlabel('Naive go response')
a[1].set_ylabel('Naive nogo response')
a[1].set_box_aspect(1)
a[1].set_title('Change in nogo response')

points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

cax = f.add_axes([.92, .25, .02, .5])

f.colorbar(points, cax=cax)

# %%
