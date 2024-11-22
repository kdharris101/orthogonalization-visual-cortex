#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import vonmises,norm





#%% Plot simulated non-linear surpression of task stimuli - PS model


colors = sns.color_palette('colorblind')

nl_exp = 1.05

x = np.linspace(-np.pi,np.pi, 9)[:8]

y = vonmises.pdf(x, 0.1, np.deg2rad(-135))
# y = vonmises.pdf(x, 10, np.deg2rad(-135))

# y = np.roll(y,-2)
# y_scale = 1.143
# y_max = y.max() + 0.1
# y /= y_max
x = np.linspace(0,180,9)[:8]
y_max = y.max()

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
    for i,o in enumerate(x):
        if (o == 45) | (o == 90):
            ax[0].plot(o,y[i]/y_max, '.', color = 'red', markersize = 1)
        else:
            ax[0].plot(o,y[i]/y_max, '.', color = 'black', markersize = 1)

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


    fig.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\low_r_ps_model.svg', format='svg')

#%% Plot simulated sharpening of tuning - LS model - weak selectivity to high selectivity


colors = sns.color_palette('colorblind')

kappa_1 = 0.1
kappa_2 = 1

x = np.linspace(-np.pi,np.pi, 9)[:8]

y_1 = vonmises.pdf(x, kappa_1, np.deg2rad(-135))
y_2 = vonmises.pdf(x, kappa_2, np.deg2rad(-135))



x = np.linspace(0,180,9)[:8]


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
    ax[0].plot(x,y_1, zorder = 2, color = colors[0])
    # ax[0].set_xticks(np.linspace(0,180,9))
    # ax[0].set_xticklabels(np.ceil(np.linspace(0,180,9)).astype(int))
    ax[0].set_xticks([45,90])
    ax[0].set_xticklabels([r'45$\degree$', r'90$\degree$'])
    sns.despine(ax=ax[0],left=True)
    ax[0].set_yticks([])
    # ax[0].set_xlabel(r'Stimulus orientation ($\degree$)')
    for i,o in enumerate(x):
        if (o == 45) | (o == 90):
            ax[0].plot(o,y_1[i], '.', color = 'red', markersize = 1)
        else:
            ax[0].plot(o,y_1[i], '.', color = 'black', markersize = 1)
   

    ax[1].plot(x,y_1, zorder = 2, color = colors[0])
    ax[1].plot(x,y_2, zorder = 2, color = colors[1])
    for i,o in enumerate(x):
        if (o == 45) | (o == 90):
            ax[1].plot(o,y_2[i], '.', color = 'red', markersize = 1)
        else:
            ax[1].plot(o,y_2[i], '.', color = 'black', markersize = 1)
    # ax[1].set_xticks(np.linspace(0,180,9))
    # ax[1].set_xticklabels(np.ceil(np.linspace(0,180,9)).astype(int))
    sns.despine(ax=ax[1],left=True)
    ax[1].set_yticks([])
    # ax[1].set_xlabel(r'Stimulus orientation ($\degree$)')
    
    
    fig.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\low_r_ls_model.svg', format='svg')


#%%


import seaborn.objects as so

kappa_1 = 0.1
kappa_2 = 10

x = np.linspace(-np.pi,np.pi, 9)[:8]

y_1 = vonmises.pdf(x, kappa_1, np.deg2rad(-135))
y_2 = vonmises.pdf(x, kappa_2, np.deg2rad(-135))



x = np.linspace(0,180,9)[:8]


# Normalize the responses to range from 0 to 1
y_1 = (y_1 - np.min(y_1)) / (np.max(y_1) - np.min(y_1))
y_2 = (y_2 - np.min(y_2)) / (np.max(y_2) - np.min(y_2))

min_firing_rate = 0.1  
max_firing_rate = 0.6

y_1 = y_1 * (max_firing_rate - min_firing_rate) + min_firing_rate

y_1 = y_2

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
    ax[0].plot(x,y_1, zorder = 2, color = colors[0])
    # ax[0].set_xticks(np.linspace(0,180,9))
    # ax[0].set_xticklabels(np.ceil(np.linspace(0,180,9)).astype(int))
    ax[0].set_xticks([45,90])
    ax[0].set_xticklabels([r'45$\degree$', r'90$\degree$'])
    sns.despine(ax=ax[0],left=True)
    ax[0].set_yticks([])
    # ax[0].set_xlabel(r'Stimulus orientation ($\degree$)')
    for i,o in enumerate(x):
        if (o == 45) | (o == 90):
            ax[0].plot(o,y_1[i], '.', color = 'red', markersize = 1)
        else:
            ax[0].plot(o,y_1[i], '.', color = 'black', markersize = 1)
   

    ax[1].plot(x,y_1, zorder = 2, color = colors[0])
    ax[1].plot(x,y_2, zorder = 2, color = colors[1])
    for i,o in enumerate(x):
        if (o == 45) | (o == 90):
            ax[1].plot(o,y_2[i], '.', color = 'red', markersize = 1)
        else:
            ax[1].plot(o,y_2[i], '.', color = 'black', markersize = 1)
    # ax[1].set_xticks(np.linspace(0,180,9))
    # ax[1].set_xticklabels(np.ceil(np.linspace(0,180,9)).astype(int))
    sns.despine(ax=ax[1],left=True)
    ax[1].set_yticks([])
    # ax[1].set_xlabel(r'Stimulus orientation ($\degree$)')



(
    so.Plot(x=y_1, y=y_2)
    .add(so.Dots())
    .limit(x=[-0.1,1.1], y=[-0.1,1.1])
)


