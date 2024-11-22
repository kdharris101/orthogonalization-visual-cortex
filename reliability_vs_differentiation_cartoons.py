#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from os.path import join


# save_fig_dir = r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft'
save_fig_dir = r'C:\Users\Samuel\Desktop\Data Science Application Stuff\Biofidelity_Exercise'

#%% Cartoons regarding learning hypotheses


# colors = [[1,0,1], [0,1,0]]
colors = sns.color_palette('Spectral', 9)

lims = (-0.25,1.5)

n_points = 40

point_size = 20

width = 0.015
headwidth = 4
headlength = 5
headaxislength = 4

xlabel = ''
ylabel = ''

# Naive high variability

ang_diff = 15

s0_ang = 45+ang_diff/2
s1_ang = 45-ang_diff/2

s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) *1.1
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) *1.1

sigma = 0.3

np.random.seed(seed=12)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

f,a = plt.subplots(1,2, figsize = (5,2.5))

a[0].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[0].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[0].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)

# Proficient low variability

ang_diff = 30
center_ang = 60

s0_ang = center_ang+ang_diff/2
s1_ang = center_ang-ang_diff/2

s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) 
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) 

sigma = 0.1

np.random.seed(seed=12)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

a[1].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[1].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[1].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)


for a in a:

    a.set_xlim(lims)
    a.set_ylim(lims)
    a.set_box_aspect(1)

    a.spines['top'].set_color('none')
    a.spines['left'].set_position('zero')
    a.spines['left'].set_bounds((0,1.5))
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['bottom'].set_bounds((0,1.5))

    # sns.despine(ax = a, offset = 10)
    a.set_yticks([])
    a.set_xticks([])

    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)

    a.spines['top'].set_color('none')
    a.spines['left'].set_position('zero')
    a.spines['left'].set_bounds((0,1.5))
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['bottom'].set_bounds((0,1.5))


f.savefig(join(save_fig_dir,'trial_spread_precision.svg'), format = 'svg')



# Naive high variability - with vectors

ang_diff = 15
center_ang = 60

s0_ang = center_ang+ang_diff/2
s1_ang = center_ang-ang_diff/2

s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) *1.1
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) *1.1

sigma = 0.2

np.random.seed(seed=12)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

f,a = plt.subplots(1,2, figsize = (5,2.5))

a[0].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[0].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[0].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)

a[0].quiver(0,0, np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang)), 
                facecolor = colors[1], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)
a[0].quiver(0,0, np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang)), 
                facecolor = colors[0], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)

# Proficient low variability - with vectors

ang_diff = 15
center_ang = 60

s0_ang = center_ang+ang_diff/2
s1_ang = center_ang-ang_diff/2

s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) 
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) 

sigma = 0.05

np.random.seed(seed=12)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

a[1].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[1].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[1].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)

a[1].quiver(0,0, np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang)), 
                facecolor = colors[1], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)
a[1].quiver(0,0, np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang)), 
                facecolor = colors[0], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)


for a in a:

    a.set_xlim(lims)
    a.set_ylim(lims)
    a.set_box_aspect(1)

    a.spines['top'].set_color('none')
    a.spines['left'].set_position('zero')
    a.spines['left'].set_bounds((0,1.5))
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['bottom'].set_bounds((0,1.5))

    # sns.despine(ax = a, offset = 10)
    a.set_yticks([])
    a.set_xticks([])

    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)

    a.spines['top'].set_color('none')
    a.spines['left'].set_position('zero')
    a.spines['left'].set_bounds((0,1.5))
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['bottom'].set_bounds((0,1.5))


f.savefig(join(save_fig_dir,'trial_spread_precision_with_vectors.svg'), format = 'svg')


# Naive generalized

ang_diff = 15
center_ang = 60

s0_ang = center_ang+ang_diff/2
s1_ang = center_ang-ang_diff/2

amp = 0.75
s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) 
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) 

sigma = 0.1

np.random.seed(seed=10)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

f,a = plt.subplots(1,2, figsize = (5,2.5))

a[0].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[0].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[0].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)

# Proficient degeneralized

ang_diff = 75
center_ang = 60

s0_ang = center_ang+ang_diff/2
s1_ang = center_ang-ang_diff/2

amp = 0.75
s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) 
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) 

sigma = 0.1

np.random.seed(seed=5)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

a[1].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[1].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[1].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)


for a in a:
    a.set_xlim(lims)
    a.set_ylim(lims)
    a.set_box_aspect(1)

    a.spines['top'].set_color('none')
    a.spines['left'].set_position('zero')
    a.spines['left'].set_bounds((0,1.5))
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['bottom'].set_bounds((0,1.5))


    a.set_yticks([])
    a.set_xticks([])

    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)


f.savefig(join(save_fig_dir,'trial_spread_degeneralized.svg'), format = 'svg')


# Naive generalized - with vectors

ang_diff = 15
center_ang = 60

s0_ang = center_ang+ang_diff/2
s1_ang = center_ang-ang_diff/2

amp = 0.75
s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) 
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) 

sigma = 0.1

np.random.seed(seed=10)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

f,a = plt.subplots(1,2, figsize = (5,2.5))


a[0].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[0].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[0].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)

a[0].quiver(0,0, np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang)), 
                facecolor = colors[1], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)
a[0].quiver(0,0, np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang)), 
                facecolor = colors[0], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)


# Proficient degeneralized - with vectors

ang_diff = 110
center_ang = 60

s0_ang = np.min([center_ang+ang_diff/2, 85])
s1_ang = center_ang-ang_diff/2

amp = 0.75
s0_mu = np.array([np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang))]) 
s1_mu = np.array([np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang))]) 

sigma = 0.1

np.random.seed(seed=5)
s0_x,s0_y = np.random.randn(n_points)*sigma + s0_mu[0], np.random.randn(n_points)*sigma + s0_mu[1]
s1_x,s1_y = np.random.randn(n_points)*sigma + s1_mu[0], np.random.randn(n_points)*sigma + s1_mu[1]

a[1].scatter(s0_x,s0_y,s = point_size, color = colors[1], zorder = 10, alpha = 0.5, ec = None)
a[1].scatter(s1_x,s1_y,s = point_size, color = colors[0], zorder = 10, alpha = 0.5, ec = None)
a[1].plot([0,np.cos(np.deg2rad(45))],[0, np.sin(np.deg2rad(45))], 'k', linewidth = 0.75, zorder = 0)

a[1].quiver(0,0, np.cos(np.deg2rad(s0_ang)), np.sin(np.deg2rad(s0_ang)), 
                facecolor = colors[1], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)
a[1].quiver(0,0, np.cos(np.deg2rad(s1_ang)), np.sin(np.deg2rad(s1_ang)), 
                facecolor = colors[0], angles='xy', scale_units='xy', scale=1, width = width,
                headwidth = headwidth, headlength = headlength, headaxislength = headaxislength,
                ec = 'k', linewidth = 0.5, zorder = 20)


for a in a:
    a.set_xlim(lims)
    a.set_ylim(lims)
    a.set_box_aspect(1)

    a.spines['top'].set_color('none')
    a.spines['left'].set_position('zero')
    a.spines['left'].set_bounds((0,1.5))
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['bottom'].set_bounds((0,1.5))


    a.set_yticks([])
    a.set_xticks([])

    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)



f.savefig(join(save_fig_dir,'trial_spread_degeneralized_with_vectors.svg'), format = 'svg')


