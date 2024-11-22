#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pwl(x,p,q,r):
    # piecewise linear (0,0) to (p,q) to (1,r)
    y = np.zeros_like(x)
    y[x<=p] = x[x<=p]*q
    y[x>p] = q*p + (x[x>p]-p)*(r-q*p)/(1-p)
    return y
    
    
ps = np.linspace(0.6,0.9,5)
qs = np.linspace(0.1,0.9,5)[::-1]
x = np.linspace(0,1,1000)
r = 1

cmap = sns.dark_palette('red', n_colors = len(ps))

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

    f,a = plt.subplots(1,1, figsize=(1,1))

    for i,(p,q) in enumerate(zip(ps,qs)):
        a.plot(x, pwl(x,p,q,r), color=cmap[i])

    a.set_box_aspect(1)
    a.set_xticks([0,0.5,1])
    a.set_yticks([0,0.5,1])

    sns.despine(ax=a, trim=True)


# plt.savefig(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft\convexity_cartoon_piecewise.svg', format='svg')

# %%
