# %%
from fitparse import FitFile
import pandas as pd
import numpy as np
from colored_logger import customLogger
logger = customLogger('root')
from pint import UnitRegistry
ureg = UnitRegistry()
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator
import seaborn as sns
import datetime
sns.set()
# end%%

# %%
fn = '2018_04_25_Sprints.fit'
fitfile = FitFile(fn)
# end%%

# %%
def strfdelta(seconds,fmt):
    seconds = int(seconds)

    d = {'total_seconds':seconds}
    d['total_minutes'],d['seconds'] = divmod(seconds,60)
    d['hours'],rem = divmod(seconds,3600)
    d['minutes'],d['seconds'] = divmod(rem,60)

    d['total_seconds'] = '{:02d}'.format(d['total_seconds'])
    d['total_minutes'] = '{:02d}'.format(d['total_minutes'])
    d['hours'] = '{:02d}'.format(d['hours'])
    d['minutes'] = '{:02d}'.format(d['minutes'])
    d['seconds'] = '{:02d}'.format(d['seconds'])

    return fmt.format(**d)

def td_formatter(x,pos,fmt='{total_minutes}:{seconds}'):
    return strfdelta(x,fmt)

# end%%

# %%
def get_obj_names(obj):
    names = []
    for data in obj:
        names.append(data.name)

    return names

def get_unique_names(message_iter):
    names = []

    for obj in message_iter:
        names.extend(get_obj_names(obj))

    return np.unique(names)

def df_from_messages(message_iter):
    message_iter = list(message_iter)
    cols = get_unique_names(message_iter)

    d = dict((el,[]) for el in cols)
    for obj in message_iter:
        for key in d.keys():
             d[key].append(obj.get_value(key))

    return remove_empty_cols(pd.DataFrame(d))

def empty_tup(x):
    try:
        if all([i is None for i in x]):
            return None
        else:
            return x
    except:
        return x

def remove_empty_cols(df):


    df_new = df.applymap(empty_tup)
    return df_new.dropna(axis=1,how='all')



def laps_check_overlap(df):
    got_bad = any((df['start_time'] - df['timestamp'].shift().fillna(df['start_time'].iloc[0])) != pd.to_timedelta(0))

    if got_bad:
        fn_bad_laps = 'Bad Laps Information.csv'
        laps[['start_time','timestamp']].to_csv(fn_bad_laps)
        logger.warning('We found overlapping laps.  See Laps output: in {}'.format(fn_bad_laps))
        logger.warning('Continuing anyway...')
    return got_bad


def app_lap_get(val,laps,laps_start='start_time',laps_end='timestamp'):
    idx = laps.index[(laps[laps_start]<= val) & (laps[laps_end] > val)]

    if len(idx) ==1:
        return idx[0]
    else:
        raise ValueError('Couldnt find laps')

def app_sprint_get(row,ave_thresh=9.0*ureg('mph'),max_thresh=12.0*ureg('mph')):
    ave = (row['enhanced_avg_speed'] * ureg('m/s')).to('mph')
    fastest = (row['enhanced_max_speed'] * ureg('m/s')).to('mph')
    return (ave > ave_thresh) and (fastest > max_thresh)


def get_seconds_series(df,key,seconds_key):
    return df.set_index(seconds_key)[key]

def second_axis_match(ax,ax2):
    l = ax.get_ylim()
    l2 = ax2.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax.get_yticks())
    return FixedLocator(ticks)

def plot_sprints(data_df,lap_df,x,y,sprints_only=True,ax=None,**kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    if sprints_only:
        filtered_data = data_df.loc[data_df['lap_id'].isin(lap_df.index[lap_df['sprint']])]
    else:
        filtered_data = data_df


    legend = kwargs.get('legend',True)
    filtered_data.groupby('lap_id').plot(x=x,y=y,ax=ax,**kwargs)

    if legend:
        labels = lap_df['descrip'].loc[lap_df.index.isin(filtered_data['lap_id'].unique())].values
        ax.legend(labels)
    return ax



def lap_descrip(row):
    if row['sprint']:
        return 'S{} - {:0.0f}m'.format(row['sprint_count'],row['total_distance'])
    else:
        return 'Other'

def get_lap_distances(data_df):
    start_dis = data_df[['lap_id','distance']].groupby('lap_id').min()
    return data_df.apply(lambda x: x['distance'] - start_dis.loc[x['lap_id']],axis=1)


ms_mph = (1*ureg('m/s')).to('mph')
cp = sns.color_palette()
# end%%


# %%

# Get raw dataframes
lap_df = df_from_messages(fitfile.get_messages('lap'))
data_points = df_from_messages(fitfile.get_messages('record'))

# Clean up and add values to data_points
data_points.sort_values('timestamp',inplace=True)
data_points['lap_id'] = data_points['timestamp'].map(lambda x: app_lap_get(x,laps))
data_points['lap_distance'] = get_lap_distances(data_points)
data_points['speed_mph'] = data_points['speed'] * ms_mph
data_points['shitty_speed_mph'] = data_points['distance'].diff()/data_points['timestamp'].diff().dt.total_seconds() * ms_mph

# Auto Determine sprints
laps['sprint'] = laps.apply(app_sprint_get,axis=1)
laps['sprint_count'] = laps.groupby('sprint').cumcount()
laps['descrip'] = laps.apply(lap_descrip,axis=1)

# Pandas sucks at time differences as indexes when plotting...we just want to get
# seconds
data_points['seconds'] = (data_points['timestamp'] - data_points['timestamp'].loc[0]).map(datetime.timedelta.total_seconds)
laps['starts_seconds'] = (laps['start_time'] - laps['start_time'].loc[0]).map(datetime.timedelta.total_seconds)
laps['end_seconds'] = (laps['timestamp'] - laps['start_time'].loc[0]).map(datetime.timedelta.total_seconds)

# end%%




# %% Overall Plot

fig,ax = plt.subplots()
ax2 = ax.twinx()

#First Axis Plot
get_seconds_series(data_points,'speed_mph','seconds').plot(ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Speed [MPH]',color=sns.color_palette()[0])
ax.tick_params('y', colors=sns.color_palette()[0])

#Highlight Sprints
for i,row in laps.loc[laps['sprint']].iterrows():
    ax.axvspan(row['starts_seconds'],row['end_seconds'],color=cp[2],alpha=0.5,zorder=0)

#Secondary axis
get_seconds_series(data_points,'heart_rate','seconds').plot(ax=ax2,color=cp[1])
ax2.set_ylabel('Heart Rate [BPM]',color=cp[1])
ax2.tick_params('y', colors=cp[1])
ax2.grid(False)

#Set up the xaxis
ax.xaxis.set_major_formatter(FuncFormatter(td_formatter))

#Match second Axis
ax2.yaxis.set_major_locator(second_axis_match(ax,ax2))
ax.set_title('Overall Activity with Hightlight Sprints')
# end%%

# %% Sprint Bar charts
fig,ax = plt.subplots()
laps.loc[laps['sprint']].plot.bar(x='descrip',y='total_timer_time',legend=False,color=cp[0],ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Time [s]')

for i,time in enumerate(laps['total_timer_time'].loc[laps['sprint']]):
    ax.text(i,time-2,'{:0.1f}'.format(time),rotation=90,ha='center',va='top',color='w')
# end%%

# %%

fig, axs = plt.subplots(2,sharex=True,sharey=True,figsize=(10,8))
plot_sprints(data_points,laps,'lap_distance','speed_mph',ax=axs[0],legend=True)
plot_sprints(data_points,laps,'lap_distance','shitty_speed_mph',ax=axs[1],legend=False)


axs[0].set_ylabel('Speed [MPH]')
axs[1].set_ylabel('Speed [MPH]')
axs[0].set_xlabel('Distance [m]')
axs[1].set_xlabel('Distance [m]')
axs[0].set_title('Garmin smoothing')
axs[1].set_title('My Euler')
fig.tight_layout()
fig.suptitle('Speed over Distance')
fig.subplots_adjust(top=0.88)

# end%%

# %% BPM Over Distance
fig, ax = plt.subplots(1)
plot_sprints(data_points,laps,'lap_distance','heart_rate',ax=ax,legend=True)
ax.set_ylabel('Heart Rate [BPM]')
ax.set_title('Heart Rate')
# end%%
# %% BPM Over Distance
fig, ax = plt.subplots(1)
plot_sprints(data_points,laps,'heart_rate','shitty_speed_mph',ax=ax,legend=True,style='o')
ax.set_ylim([10,18])
ax.set_ylabel('Speed [MPH]')
ax.set_title('Looking for speed and heart rate trends')
ax.set_xlabel('Heart Rate [BPM]')
# end%%

# %%
if __name__ == '__main__':
    main() 

# end%%
