# -*- coding: utf-8 -*-
"""

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

subjects_file = ['SF170620B','M170905B_SF','SF171107','SF180515','SF180613']

subjects = ['SF170620B', 'SF170905B', 'SF171107','SF180515', 'SF180613']

expt_dates = ['2017-12-21', '2017-11-26', '2018-04-04', '2018-09-21',
              '2018-12-12']

expt_nums = [7, 1, 3, 3, 5]
expt_nums_ret = [8, 2, 4, 4, 6]

# Load most up-to-date saved results
# file_paths = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
#                    str(expt_nums[i]), 'drift_pixel_maps_trial_convexity*'))[-1] 
#             for i in range(len(subjects))]

file_paths = []

file_paths = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
                  str(expt_nums[i]), 'drift_pixel_maps_trial_convexity_taskstim_*'))[-1]
                  for i in range(len(subjects))]

file_paths_ret = [glob.glob(join(results_dir, subjects_file[i], expt_dates[i], 
            str(expt_nums_ret[i]), r'_'.join([subjects_file[i], expt_dates[i],
            str(expt_nums_ret[i]), 'peak_pixel_retinotopy*'])))[-1] 
                    for i in range(len(subjects))]



#%%
def plane_sm(plane,med_bin = 10, sigma = 60, mode = 'reflect'):
    
    plane_sm = np.zeros_like(plane)
    
    for i,p in enumerate(plane):
        plane_sm[i,...] = ndimage.median_filter(p,med_bin)
        plane_sm[i,...] = ndimage.gaussian_filter(plane_sm[i,...],sigma,
                                                  mode = mode)
        
    return plane_sm

#%% Plot retinotopic maps and V1 mask

from mpl_toolkits.axes_grid1 import make_axes_locatable

ret_sigma = 60

ret_map = np.load(file_paths_ret[1])

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

ret_map_sm = plane_sm(ret_map)

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

# cmap = 'gist_rainbow'
cmap = 'turbo'
# cmap = 'jet'

# f,axes = plt.subplots(1,2,figsize = (20,8))
f,axes = plt.subplots(1,3,figsize = (20,8))

im_elv = sns.heatmap(ret_map_sm[0,...], vmin = vmin_elv, vmax = vmax_elv,
            square = True, cbar = False, rasterized = True, cmap = cmap,
            ax = axes[1])
axes[1].tick_params(left = False, bottom = False)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title('Elevation')
axes[1].contour(xv,yv,ang_th,colors='w')

im_azi = sns.heatmap(ret_map_sm[1,...], vmin = vmin_azi, vmax = vmax_azi,
            square = True, cbar = False, rasterized = True, cmap = cmap,
            ax = axes[0])
axes[0].tick_params(left = False, bottom = False)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title('Azimuth')
axes[0].contour(xv,yv,ang_th,colors='w')

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
axes[2].contour(xv,yv,ang_th,colors='w')
  
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
# cax.axis('off')

# divider = make_axes_locatable(axes[3])
# cax = divider.append_axes("bottom", size="3%", pad=0.5)
# cax.axis('off')

f.tight_layout()

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

#%% Plot contours with response map for low convexity vs high convexity

colors = sns.color_palette('rainbow_r',6)
# colors = sns.color_palette("Spectral", 6,as_cmap=True)
cond_c = sns.color_palette('colorblind')[:2]

expt_ind = 0

map_sigma = 60

r = np.load(file_paths_ret[expt_ind])
    
task_pos = [0,-80]

b_contours = np.zeros(2,dtype='object')

r = r.mean(2)
# Reshape to 2d array for elv and azi
r = r.reshape(2,512,512)

r[0,...] = ndimage.gaussian_filter(ndimage.median_filter(
                    np.flip(r[0,...],1),10), map_sigma, mode = 'reflect')
r[1,...] = ndimage.gaussian_filter(ndimage.median_filter(
                    np.flip(r[1,...],1),10), map_sigma, mode = 'reflect')

task_ret = (r[0,...] - task_pos[0]) + (r[1,...] - task_pos[1]) * 1j
task_ret = np.abs(task_ret)

task_ret_bin = np.digitize(task_ret,np.arange(0,30,5))

# b_contours[i] = np.where(np.diff(task_ret_bin,prepend=task_ret_bin[:,0][:,None], 
#                                  axis = 1) + 
#                   np.diff(task_ret_bin,prepend=task_ret_bin[0,:][None,:],
#                           axis = 0) != 0)

b_contours = task_ret_bin

# for i,b in enumerate(np.unique(task_ret_bin)):
#     b_ind = task_ret_bin == b
#     b_contours[i] = np.where(np.diff(b_ind,prepend=b_ind[:,0][:,None], axis = 1) + 
#                   np.diff(b_ind,prepend=b_ind[0,:][None,:],axis = 0) != 0)


# Find V1 outline for expts

map_sigma = 60

r = np.load(file_paths_ret[expt_ind])

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
V1_outlines = ang_th
    

from matplotlib import ticker
import matplotlib.colors as mc
from skimage import measure
from skimage.morphology import local_minima

rasterize = True

# plane = 3
# pm = np.load(file_paths[expt_ind])[...,plane]
pm = np.load(file_paths[expt_ind]).mean(3)

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

    f,axes = plt.subplots(nrows = 1, ncols = 2, figsize = (3,1.5))
    
    vmin = np.percentile(pm.flatten(),0.5)
    vmax = np.percentile(pm.flatten(),99.5)
    
    xv,yv = np.meshgrid(np.arange(512),np.arange(512))
    
    norm = mc.TwoSlopeNorm(vcenter = 0, vmin=vmin, vmax=vmax) 
        
    im = sns.heatmap(np.flip(pm[...,0],1), #center = 0, vmin = vmin, vmax = vmax, 
                ax = axes[0], square = True, cbar = False, rasterized = rasterize,
                norm = norm, cmap = 'icefire')
    axes[0].tick_params(left = False, bottom = False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title('No convexity', color = cond_c[0], pad = 2)
    
    c_width = 0.5
    
    c_levels = [5,10,15,20,25]
    
    for i,c in enumerate(c_levels):
    
        task_contours = measure.find_contours(task_ret,c)
        
        for contour in task_contours:
            axes[0].plot(contour[:,1],contour[:,0],linestyle='dashed',color = colors[i+1], 
                         linewidth=c_width, dashes=(3,6))
            
    V1_contours =  measure.find_contours(V1_outlines,0)
        
    for contour in V1_contours:
            axes[0].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)
            
    
    # Plot center of stim
    
    minima = np.where(local_minima(task_ret))
    
    for m in range(2):
        axes[0].plot(minima[1][m], minima[0][m],color=colors[0], markersize = 1,
                     marker = '.')
     
    sns.heatmap(np.flip(pm[...,1],1), #center = 0, vmin = vmin, vmax = vmax, 
                ax = axes[1], square = True, cbar = False, rasterized = rasterize,
                norm = norm, cmap = 'icefire')
    axes[1].tick_params(left = False, bottom = False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('High convexity', color = cond_c[1], pad = 2)
    
    for i,c in enumerate(c_levels):
    
        task_contours = measure.find_contours(task_ret,c)
        
        for contour in task_contours:
            axes[1].plot(contour[:,1],contour[:,0],linestyle='dashed',color = colors[i+1], 
                         linewidth=c_width,dashes=(3,6))
            # axes[1].plot(contour[:,1],contour[:,0],color = colors[i+1], linewidth=c_width)
 
    V1_contours =  measure.find_contours(V1_outlines,0)
    
    for contour in V1_contours:
        axes[1].plot(contour[:,1],contour[:,0],'-w', linewidth=c_width)
    
    # Plot center of stim
    for m in range(2):
        axes[1].plot(minima[1][m], minima[0][m],color=colors[0], markersize = 1,
                     marker = '.')
        
    # Legend
    l = axes[1].legend(labels=[r'0$\degree$',r'5$\degree$',r'10$\degree$',
                           r'15$\degree$',r'20$\degree$',r'25$\degree$'],
                   fancybox = False, frameon = False, ncol = 2,
                   title = 'Ret. dist. from task stim. (vis. deg)',
                   handletextpad=0.2, labelspacing=0.2, borderpad = 0,
                   handlelength = 1, columnspacing = 0.75)
    
    for line in l.get_lines():
        line.set_linestyle('-')          

    f.tight_layout()
    
    mappable = im.get_children()[0]
    cb = plt.colorbar(mappable, ax = axes, orientation = 'vertical', shrink = 0.8,
                 label = 'df/f', pad = 0.02)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

    f.savefig(r'C:\Users\Samuel\OneDrive - University College London\Results\Figures\Draft\gratings_pixelmap_convexity_task_stim.svg',
        format = 'svg', dpi = 600)

#%% Plot retinotopic maps and V1 mask

ret_sigma = 20
map_sigma = 50

ret_map = np.load(file_paths_ret[6])

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