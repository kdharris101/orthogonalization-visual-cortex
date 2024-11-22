#%%

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import vonmises

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

x = np.deg2rad(np.arange(-180,180,45))*-1
kappa=1.25

s_index = [1, 0.8, 0.6, 0.4, 0.2]

ori_pref = 45

mean_pref = np.zeros(len(s_index))
modal_pref = np.zeros(len(s_index))

s_colors = sns.color_palette('magma',len(s_index))
pref_colors = sns.color_palette('colorblind')[2:4]

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


    f,a = plt.subplots(1,2, figsize=(3,1.11))

    for i,s in enumerate(s_index):

        tc = vonmises.pdf(x, kappa, np.deg2rad(ori_pref*2))

        tc[x==np.pi/2] *= s
        tc[x==np.pi] *= s

        tc_x = np.cos(x)*tc
        tc_y = np.sin(x)*tc

        a[0].plot(np.insert(tc_x, len(tc_x), tc_x[0]), np.insert(tc_y, len(tc_y), tc_y[0]), color=s_colors[i])
        
        vec = np.exp(x*1j) * tc
        
        vec = vec.sum() / tc.sum()
        
        mean_pref[i] = np.rad2deg(np.arctan2(np.imag(vec),np.real(vec)))/2
        modal_pref[i] = np.rad2deg(x[np.argmax(tc)])/2    
        
    a[0].set_xlim(-0.6,0.6)
    a[0].set_ylim(-0.6,0.6)
    a[0].set_xticks([])
    a[0].set_yticks([])
    x = np.cos(np.linspace(-np.pi,np.pi,1000))*0.55
    y = np.sin(np.linspace(-np.pi,np.pi,1000))*0.55
    a[0].plot(x,y,'black')
    a[0].hlines(0,-0.55,0.55, colors='black', linestyles='dashed')
    a[0].vlines(0,-0.55,0.55, colors='black', linestyles='dashed')
    sns.despine(ax=a[0], left=True, bottom=True)
    a[0].set_box_aspect(1)


    ori_prefs = np.ceil(np.arange(0,180,22.5)).astype(int)
    ori_ind = np.argmin(np.abs(ori_prefs-ori_pref))

    mean_pref[mean_pref<0] += 180
    modal_pref[modal_pref<0] += 180

    a[1].plot(s_index, mean_pref, color=pref_colors[0])
    a[1].plot(s_index, modal_pref, color=pref_colors[1])
    # a[1].hlines(45, 0.1, 1, colors='black', linestyles='dashed')
    a[1].hlines([np.round(ori_pref)+11.5, np.round(ori_pref)-11.25], 0.2, 1, colors='black', linestyles='dashed')
    a[1].set_xlim([0.2,1])
    a[1].set_xticks([])
    a[1].set_xlabel('Increasing suppression')
    a[1].set_ylabel('Preferred orientation')
    a[1].set_box_aspect(1)
    a[1].set_ylim(ori_prefs[ori_ind-1]-3,ori_prefs[ori_ind+1]+3)
    a[1].set_yticks(ori_prefs[ori_ind-1:ori_ind+2])
    a[1].set_yticklabels([str(o) + r'$\degree$' for o in ori_prefs[ori_ind-1:ori_ind+2]])
    sns.despine(ax=a[1], trim=True, bottom=True)
    a[1].invert_xaxis()
    a[1].arrow(1-0.9/4,ori_prefs[ori_ind-1]-2,-0.9/2,0, head_width=1, head_length=0.05, facecolor='black', edgecolor='black', length_includes_head=True)
    a[1].legend(['Mean ori. pref', 'Modal ori. pref.'], frameon=False, loc='upper left')
    f.tight_layout()

    # f.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\model_suppression_and_ori_pref.svg', format='svg')

# %%
