#%%
import sys
# sys.path.append(r"C:\Users\samue\OneDrive - University College London\Code\Python\Behavior")
# sys.path.append(r"C:\Users\samue\OneDrive - University College London\Code\Python\Recordings")
sys.path.append(r"C:\Users\Samuel\OneDrive - University College London\Code\Python\Behavior")
sys.path.append(r"C:\Users\Samuel\OneDrive - University College London\Code\Python\Recordings")

from os.path import join
import cortex_lab_utils as clu
import choiceworldpy as cw
import signalspy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
import pandas as pd
import os
import glob
import re
from datetime import datetime
from scipy.stats import pearsonr


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'


def find_cw_sessions(mouse):
    
    print(f'Loading for {mouse}')
    
    subject_dirs = clu.find_subject_dirs(mouse)
    
    subject_dirs = list(filter(lambda x: 'OneDrive' not in x, subject_dirs))    
    
    largest_block_mats = []
    
    for subject_dir in subject_dirs:

        # Iterate over date directories
        for date_dir in os.listdir(subject_dir):
            date_dir_path = os.path.join(subject_dir, date_dir)
            
            print(f'Loading {date_dir}')

            # Check if it's a directory
            if os.path.isdir(date_dir_path):
                largest_date_block_mat = None
                largest_size = 0

                for experiment_dir in os.listdir(date_dir_path):
                    experiment_dir_path = os.path.join(date_dir_path, experiment_dir)

                    # Check if it's a directory
                    if os.path.isdir(experiment_dir_path):
                        # Use glob to find files ending with 'Block.mat' in the experiment directory
                        block_mat_file = glob.glob(os.path.join(experiment_dir_path, '*Block.mat'))

                        if block_mat_file:
                            block_mat_path = block_mat_file[0]
                            if block_mat_path in largest_block_mats:
                                continue
                            block_mat_size = os.path.getsize(block_mat_path)

                            # Check if it's the largest for this date
                            if block_mat_size > largest_size:
                                largest_size = block_mat_size
                                largest_date_block_mat = block_mat_path

                if largest_date_block_mat:
                    largest_block_mats.append(largest_date_block_mat)

    return largest_block_mats

def is_valid_date_format(date_str):
    # Define a regular expression pattern to match 'yyyy-mm-dd' format
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    
    # Check if the given string matches the date pattern
    return re.match(date_pattern, date_str)

def is_valid_expDef_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            first_line = file.readline().decode('latin-1')  # Specify 'latin-1' encoding
            return "oriChoiceWorld" in first_line and "oriChoiceWorldPassive" not in first_line
    except Exception as e:
        return False

def find_signals_sessions(mouse, min_trials=30):
    
    print(f'Loading for {mouse}')
    
    subject_dirs = clu.find_subject_dirs(mouse)
    
    largest_expt_directories = []
    
    for subject_dir in subject_dirs:
    
        for date_dir in os.listdir(subject_dir):
            date_dir_path = os.path.join(subject_dir, date_dir)
            if not os.path.isdir(date_dir_path) or not is_valid_date_format(date_dir):
                continue
            else:
                print(f'Checking expts for {date_dir}')

            largest_expt_dir = None
            largest_num_trials = 0
            
            for expt_dir in os.listdir(date_dir_path):
                expt_dir_path = os.path.join(date_dir_path, expt_dir)
                if not os.path.isdir(expt_dir_path):
                    continue

                if expt_dir_path in largest_expt_directories:
                    continue

                choice_file = os.path.join(expt_dir_path, '_misc_trials.choice.npy')

                # Check if the '_misc_trials.choice.npy' file exists in the directory
                if not os.path.exists(choice_file):
                    continue

                if not any(glob.glob(os.path.join(expt_dir_path, '*expDef.m'))):
                    continue

                # Check the '_expDef.m' file for the required conditions
                expDef_file = glob.glob(os.path.join(expt_dir_path, '*expDef.m'))[0]
                if not is_valid_expDef_file(expDef_file):
                    continue

                # Load the file and get the number of trials
                choice_data = np.load(choice_file)
                num_trials = choice_data.shape[0]

                if num_trials < min_trials:
                    continue

                if num_trials > largest_num_trials:
                    largest_num_trials = num_trials
                    largest_expt_dir = expt_dir_path
            
            if largest_expt_dir:
                largest_expt_directories.append(largest_expt_dir)

    return largest_expt_directories

#%%

cw_subjects = ['SF170620B','SF170905B','SF171107','SF180515','SF180613']
# signals_subjects = ['SF004','SF005','SF007', 'SF008', 'SF010', 'SF011']
signals_subjects=[]

df_trials = pd.DataFrame()

min_trials = 50

# Add choiceworld trials to dataframe


for m in cw_subjects:
    
    c = 0
    
    sessions = find_cw_sessions(m)
    
    for s in sessions:
    
        df_sub, _ = cw.load_choiceworld(s, min_trials=min_trials, align_to_pd=False, wheel=False)
        
        if df_sub is None:
            continue
        
        df_sub = df_sub[['feedbackType','responseMadeID','responseMadeTime',
                        'interactiveStartedTime','repeatNum', 'exptDate', 'targetOrientation']]
        
        df_sub = df_sub.rename({'feedbackType' : 'feedback',
                                'responseMadeID' : 'choice',
                                'responseMadeTime' : 'response_time',
                                'interactiveStartedTime' : 'go_time',
                                'repeatNum' : 'repeat_num',
                                'exptDate' : 'date'}, axis=1)
        
        df_sub['choice'] = df_sub.choice.apply(lambda x: -1 if x==1 else 1)
        df_sub['trial_num'] = np.arange(len(df_sub))
        df_sub['subject'] = np.repeat(m, len(df_sub))
        df_sub['session_num'] = np.repeat(c, len(df_sub))
        df_sub['task_type'] = np.repeat('original', len(df_sub))
        df_sub['left_stim'] = [90 - o[0][0] for o in df_sub['targetOrientation']]
        df_sub['right_stim'] = [90 - o[0][1] for o in df_sub['targetOrientation']]

        df_trials = pd.concat([df_trials, df_sub], ignore_index=True)
        
        c+=1

for m in signals_subjects:
    
    sessions = find_signals_sessions(m, min_trials=min_trials)
    
    for i,s in enumerate(sessions):

        choices = np.hstack(sp.load_signals_file(s,'choice'))
        feedback = np.hstack(sp.load_signals_file(s,'feedbackType'))
        resp_times = np.hstack(sp.load_signals_file(s,'response_times'))
        go_times = np.hstack(sp.load_signals_file(s,'goCue_times'))
        repeat_num = np.hstack(sp.load_signals_file(s,'repNum'))

        df_sub = pd.DataFrame({'choice' : choices,
                               'feedback' : feedback,
                               'response_time' : resp_times,
                               'go_time' : go_times,
                               'repeat_num' : repeat_num,
                               'trial_num' : np.arange(len(choices)),
                               'subject' : np.repeat(m, len(choices)),
                               'date' : np.repeat(s.split('\\')[1], len(choices)),
                               'session_num' : np.repeat(i, len(choices)),
                               'task_type' : np.repeat('simplified', len(choices))})
        
        df_trials = pd.concat([df_trials, df_sub], ignore_index=True)
        
        
df_trials['go_to_resp_time'] = df_trials['response_time'] - df_trials['go_time']


#%% Save

df_trials.to_csv(r"C:\Users\Samuel\OneDrive - University College London\Results\trials.csv")

#%%

from scipy.ndimage import convolve1d

df_plot = df_trials[df_trials.repeat_num==1].copy()
# df_plot = df_plot[df_plot.subject!='SF004']
# df_plot = df_plot[df_plot.subject!='SF170905B']

df_plot['task_type'] = df_plot.task_type.transform(lambda x: x.capitalize())
df_plot['session_num'] = df_plot.session_num+1

df_performance = df_plot.groupby(['subject','session_num', 'date'], observed=True)['feedback'].value_counts(normalize=True).to_frame()
df_performance = df_performance.rename({'feedback' : 'performance'}, axis=1).reset_index()
df_performance = df_performance[df_performance.feedback==1].reset_index(drop=True)
df_performance['task_type'] = df_plot.groupby(['subject','session_num'])['task_type'].first().reset_index().task_type
df_performance['performance'] = df_performance.performance*100

df_performance = df_performance.sort_values(['subject','session_num'])

# n = 3
# kernel = np.ones(3)/3

# def smooth(x):
#     return convolve1d(x, kernel, mode='nearest')

df_performance['performance_smoothed'] = df_performance.groupby(['subject'])['performance'].rolling(window=3).mean().reset_index(level=0, drop=True)

task_colors = sns.color_palette('colorblind',2)[::-1]

f,a = plt.subplots(1,1)

p = (
        so.Plot(df_performance, x='session_num', y='performance_smoothed')
        .add(so.Lines(alpha=0.5), color='task_type', group='subject', legend=False)
        # .add(so.Lines(), so.Agg(), color='task_type', legend=False)
        # .add(so.Band(), so.Est(errorbar=('se',1)), color='task_type')
        .label(x='Session number', y='Performance', color='Task type')
        .limit(x=(0,100), y=(40,90))
        .scale(x=so.Continuous().tick(at=[1,10,20,30,40,50,60,70,80,90,100]),
               color='colorblind')
        .on(a)
        .plot()
    )

# df_p_mu = df_performance.groupby(['task_type','session_num'], observed=True)['performance_smoothed'].mean().reset_index()
df_p_mu = df_performance.groupby(['subject', 'task_type', 'session_num'], observed=True)['performance_smoothed'].mean().reset_index()

# original_x = df_p_mu[(df_p_mu.task_type=='Original') & (df_p_mu.performance_smoothed >=70)].iloc[0]
original_x = df_p_mu[df_p_mu.task_type=='Original'].groupby(['subject']).apply(lambda x: x[(x.performance_smoothed>=70)].session_num.iloc[0] if (x.performance_smoothed>=70).max()==1 else None)
original_x.dropna(inplace=True)

# simplified_x = df_p_mu[(df_p_mu.task_type=='Simplified') & (df_p_mu.performance_smoothed >=70)].iloc[0]
simplified_x = df_p_mu[df_p_mu.task_type=='Simplified'].groupby(['subject']).apply(lambda x: x[(x.performance_smoothed>=70)].session_num.iloc[0] if (x.performance_smoothed>=70).max()==1 else None)
simplified_x.dropna(inplace=True)

# a.vlines([original_x.session_num, simplified_x.session_num], *a.get_ylim(), colors=task_colors, linestyles='dashed')
# a.hlines(70,*a.get_xlim(), colors='black', linestyles='dashed')
a.vlines(original_x.to_numpy(), *a.get_ylim(), colors=task_colors[0])
a.vlines(simplified_x.to_numpy(), *a.get_ylim(), colors=task_colors[1])
a.vlines(original_x.to_numpy().mean(), *a.get_ylim(), colors=task_colors[0], linestyles='dashed')
a.vlines(simplified_x.to_numpy().mean(), *a.get_ylim(), colors=task_colors[1], linestyles='dashed')

sns.despine(ax=a, trim=True)

p.show()

# Original only

f,a = plt.subplots(1,1)

p = (
        so.Plot(df_performance[df_performance.task_type=='Original'], x='session_num', y='performance_smoothed')
        .add(so.Lines(alpha=0.5, color=task_colors[0]), group='subject', legend=False)
        # .add(so.Lines(color=task_colors[0]), so.Agg(), legend=False)
        # .add(so.Band(), so.Est(errorbar=('se',1)), color='task_type')
        .label(x='Session number', y='Performance', color='Task type')
        .limit(x=(0,100), y=(40,90))
        .scale(x=so.Continuous().tick(at=[1,10,20,30,40,50,60,70,80,90,100]),
               color='colorblind')
        .on(a)
        .plot()
    )


# df_p_mu = df_performance.groupby(['task_type','session_num'], observed=True)['performance_smoothed'].mean().reset_index()
df_p_mu = df_performance.groupby(['subject', 'task_type', 'session_num'], observed=True)['performance_smoothed'].mean().reset_index()

# original_x = df_p_mu[(df_p_mu.task_type=='Original') & (df_p_mu.performance_smoothed >=70)].iloc[0]
original_x = df_p_mu[df_p_mu.task_type=='Original'].groupby(['subject']).apply(lambda x: x[(x.performance_smoothed>=70)].session_num.iloc[0] if (x.performance_smoothed>=70).max()==1 else None)
original_x.dropna(inplace=True)

a.vlines(original_x.to_numpy(), *a.get_ylim(), colors=task_colors[0])
a.vlines(original_x.to_numpy().mean(), *a.get_ylim(), colors=task_colors[0], linestyles='dashed')


sns.despine(ax=a, trim=True)

p.show()


# Number of trials

df_num_trials = df_plot.groupby(['subject','session_num','task_type'], observed=True)['choice'].count().reset_index()
df_num_trials = df_num_trials.rename({'choice' : 'num_trials'}, axis=1)

df_num_trials['num_trials_smoothed'] = df_num_trials.groupby(['subject'])['num_trials'].transform(lambda x: smooth(x))

f,a = plt.subplots(1,1)

p = (
        so.Plot(df_num_trials, x='session_num', y='num_trials_smoothed')
        .add(so.Lines(alpha=0.4), color='task_type', group='subject', legend=False)
        .add(so.Lines(), so.Agg(), color='task_type', legend=False)
        # .add(so.Band(), so.Est(errorbar=('se',1)), color='task_type')
        .label(x='Session number', y='Number of trials', color='Task type')
        .limit(x=(0,100))
        .scale(x=so.Continuous().tick(at=[1,10,20,30,40,50,60,70,80,90,100]),
                color='colorblind')

        .on(a)
        .plot()
    )

sns.despine(ax=a, trim=True)

p.show()


df_resp_time = df_plot.groupby(['subject','session_num','task_type'], observed=True)['go_to_resp_time'].median().reset_index()

f,a = plt.subplots(1,1)

p = (
        so.Plot(df_resp_time, x='session_num', y='go_to_resp_time')
        .add(so.Lines(alpha=0.4), group='subject', color='task_type', legend=False)
        .add(so.Lines(), so.Agg(), color='task_type', legend=False)
        .label(x='Session number', y='Median response time', color='Task type', legend=['Simplified','Original'])
        .limit(x=(0,100))
        .scale(x=so.Continuous().tick(at=[1,10,20,30,40,50,60,70,80,90,100]),
                color='colorblind')
        .on(a)
        .plot()
    )

sns.despine(ax=a, trim=True)

p.show()


#%% For choice world mice, find average performance in n number of sesssions prior to imaging session

subjects = ['SF170620B', 'SF170905B', 'SF171107', 'SF180515', 'SF180613']

imaging_dates = ['2017-12-21', '2017-11-26', '2018-04-04', '2018-09-21', '2018-12-12']


df_plot = df_trials[df_trials.repeat_num==1].copy()
# df_plot = df_trials.copy()
df_plot = df_plot[df_plot.task_type=='original']

df_plot['session_num'] = df_plot.session_num+1

df_performance = df_plot.groupby(['subject','session_num', 'date'], observed=True)['feedback'].value_counts(normalize=True).to_frame()
df_performance = df_performance.rename({'feedback' : 'performance'}, axis=1).reset_index()
df_performance = df_performance[df_performance.feedback==1].reset_index(drop=True)
df_performance['task_type'] = df_plot.groupby(['subject','session_num'])['task_type'].first().reset_index().task_type
df_performance['performance'] = df_performance.performance*100

df_performance['date'] = pd.to_datetime(df_performance.date)
df_performance = df_performance.sort_values(['subject','date'])

N = 3  # Number of sessions to average
subject_dates = {s: pd.to_datetime(d) for s,d in zip(subjects, imaging_dates)}

# Function to calculate average for N sessions before the specified date for each subject
def calculate_average(group, subject, subject_dates):
    specific_date = subject_dates[subject]
    filtered_group = group[group['date'] < specific_date][-N:]
    return pd.Series({'average': filtered_group['performance'].mean()})

# Calculate the averages
averages = {}
for subject in df_performance['subject'].unique():
    subject_group = df_performance[df_performance['subject'] == subject]
    avg = calculate_average(subject_group, subject, subject_dates)
    averages[subject] = avg['average']




#%%

df_plot = df_trials.copy()
df_plot = df_plot[df_plot.task_type=='original']

num_trials = []

for i,subject in enumerate(df_plot.subject.unique()):
    num_trials.append(len(df_plot[df_plot.date<=imaging_dates[i]].session_num.unique()))



#%% Load original task sessions

df_trials = pd.read_csv(r'C:\Users\Samuel\OneDrive - University College London\Results\trials.csv')
# df_trials = pd.read_csv(r'C:\Users\samue\OneDrive - University College London\Results\trials.csv')


df_trials = df_trials[df_trials.task_type=='original']

# Remove sessions with nontypical stim for SF171107

abnormal_sessions = df_trials[~df_trials['left_stim'].isin([45, 67.5, 90])]
abnormal_session_ids = abnormal_sessions[['subject', 'session_num']].drop_duplicates()

df_trials = pd.merge(df_trials, abnormal_session_ids, on=['subject', 'session_num'], 
                       how='left', indicator=True)
df_trials = df_trials[df_trials['_merge'] == 'left_only'].drop(columns=['_merge'])

imaging_dates = {'SF170620B': pd.to_datetime('2017-12-21'), 
                 'SF170905B': pd.to_datetime('2017-11-26'),
                 'SF171107': pd.to_datetime('2018-04-04'),
                 'SF180515': pd.to_datetime('2018-09-21'),
                 'SF180613': pd.to_datetime('2018-12-12')}

df_trials['date'] = pd.to_datetime(df_trials['date'])

df_trials = df_trials.sort_values(['subject','date'])
df_trials['session_num'] = df_trials.groupby('subject')['date'].rank(method='dense').astype(int)

df_oriworld = df_trials[df_trials.apply(lambda row: row['date'] <= imaging_dates[row['subject']], axis=1)]

df_stim_counts = df_oriworld.groupby(['subject','session_num'])['left_stim'].value_counts().reset_index(name='stim_counts')

df_stim_counts['cumulative_stim_counts'] = df_stim_counts.groupby(['subject', 'left_stim'], group_keys=False)['stim_counts'].apply(lambda x: x[::-1].cumsum()[::-1])


ps = np.load(join(r'C:\Users\Samuel\OneDrive - University College London\Results','pop_sparse_45_68_90.npy'))
# ps = np.load(join(r'C:\Users\samue\OneDrive - University College London\Results','pop_sparse_45_68_90.npy'))


df_stim_counts['r_session_num'] = df_stim_counts.groupby('subject', group_keys=False)['session_num'].transform(lambda x: x[::-1].values)

df_stim_counts = df_stim_counts.sort_values(['subject','session_num','left_stim'])

# Remove those odd trials for SF171107
# df_stim_counts = df_stim_counts[df_stim_counts.left_stim != 78.75]

df_ps = pd.DataFrame({'left_stim': np.tile(np.array([45,67.5,90])[None,:], (5,1)).flatten(),
                      'ps': ps.flatten(),
                      'subject': np.tile(df_stim_counts.subject.unique()[:,None], (1,3)).flatten()})

# df_stim_ps = df_stim_counts[df_stim_counts.r_session_num <= 35].copy()

df_stim_ps = df_stim_counts.copy()

df_stim_ps = df_stim_ps.merge(df_ps, how='left', on=['left_stim', 'subject'])

#%% Compare correlation with shuffled

# df_r = df_r.groupby(['r_session_num'], group_keys=True)['cumulative_stim_counts'].apply(lambda x: pearsonr(x.to_numpy(), ps.flatten())[0])

df_r = df_stim_ps.groupby('r_session_num')[['ps', 'cumulative_stim_counts']].apply(lambda x: pearsonr(x.ps.to_numpy(), x.cumulative_stim_counts.to_numpy())[0])

# df_r.plot()

def shift_column(group):
    shift_value = np.random.randint(len(group))
    values = group['cumulative_stim_counts'].values
    shifted_values = np.roll(values, shift_value)
    group['cumulative_stim_counts'] = shifted_values

    return group

shuf_num = 2000

shuf_corr = np.zeros((35, shuf_num))

for i in range(shuf_num):
    
    print(f"Shuffle {i}")
    
    # Shuffle 'column_to_shuffle' within each 'group_column' group
    df_stim_ps_shuf = df_stim_ps.groupby('subject', group_keys=False).apply(shift_column).reset_index(drop=True)

    df_r_shuf = df_stim_ps_shuf.groupby('r_session_num')[['ps', 'cumulative_stim_counts']].apply(lambda x: pearsonr(x.ps.to_numpy(), x.cumulative_stim_counts.to_numpy())[0])

    shuf_corr[:,i] = df_r_shuf.to_numpy()
    

f,a = plt.subplots(1,1)

a.plot(np.arange(1,36), df_r)
a.plot(np.arange(1,36), shuf_corr.mean(1))

lower_bound = np.percentile(shuf_corr, 2.5, axis=1)
upper_bound = np.percentile(shuf_corr, 97.5, axis=1)

a.fill_between(np.arange(1,36), lower_bound, upper_bound, color='gray', alpha=0.5, label='95% Confidence Interval')

#%% Plot total cumulative stimulus presentations vs PS

r_session_num = 20

df = df_stim_ps[df_stim_ps.r_session_num==r_session_num].copy()
# df = df_stim_ps[df_stim_ps.session_num==1].copy()


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


    f,a = plt.subplots(1,1, figsize=(1.5,1.5))

    (
        so.Plot(df, x ='cumulative_stim_counts', y='ps', color='left_stim')
        .layout(engine='tight')
        .add(so.Dot(edgecolor='black', pointsize=3), legend=False)
        .add(so.Lines(linewidth=0.5), so.PolyFit(order=1), legend=False)
        .scale(color=so.Nominal(order=[45,67.5,90]))
        .label(x='Stimulus presentations', y='Population sparseness')
        .limit(y=(0.1,0.25))
        .scale(color='Set1')
        .on(a)
        .plot()
    )

    # r,p = pearsonr(df.cumulative_stim_counts.to_numpy(), df.ps.to_numpy())

    # text = f"""
    #     r = {np.round(r,3)}
    #     p = {np.round(p,3)}
    #     """

    # # a.text(2000, 0.2, [f'r = {np.round(r,3)}', f'p = {np.round(p,3)}'])

    # a.text(1500, 0.18, text)

    # x = np.vstack([df.cumulative_stim_counts.to_numpy(), np.ones(len(df.cumulative_stim_counts))]).T

    # m, c = np.linalg.lstsq(x, df.ps.to_numpy(), rcond=None)[0]

    # x = np.linspace(a.get_xlim()[0]+100, a.get_xlim()[1]-100, 500)

    # a.plot(x,x*m+c, '--k')
    
    # a.set_box_aspect(1)
    
    sns.despine(ax=a)

    f.savefig(join(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft', f'n_back_{r_session_num}_stim_exposure_PS.svg'), format='svg')
    # f.savefig(join(r'C:\Users\Samuel\OneDrive - University College London\Orthogonalization project\Figures\Draft', f'all_sessions_stim_exposure_PS.svg'), format='svg')

#%%

import statsmodels.api as sm
from statsmodels.formula.api import ols

# mdl = ols('ps ~ C(left_stim) + cumulative_stim_counts + C(left_stim):cumulative_stim_counts', data=df).fit()
mdl = ols('ps ~ C(left_stim) + cumulative_stim_counts', data=df).fit()


anova_table = sm.stats.anova_lm(mdl, typ=2) 
print(anova_table)
# print(mdl.summary())




#%% For choiceworld, look at number of presentations across stimuli, including repeat trials

df_stim_count = df_trials.groupby('subject')['left_stim'].value_counts(normalize=True).reset_index(name = 'stim_count')

df_stim_count = df_stim_count.sort_values(['subject','left_stim'])

ps = np.load(join(r'C:\Users\Samuel\OneDrive - University College London\Results','pop_sparse_45_68_90.npy'))

df_stim_count = df_stim_count[df_stim_count.left_stim!=78.75]

df_stim_count['ps'] = ps.flatten()

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


    (
        so.Plot(df_stim_count, x='stim_count', y='ps', color='left_stim')
        .layout(engine='tight')
        .add(so.Dots())
        .scale(color=so.Nominal())
        .show()
    )
