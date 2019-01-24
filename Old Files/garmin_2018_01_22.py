# %% Import Base Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import sys,os
import xml.etree.ElementTree as ET
sns.set()
# end%%

# %% INPUTS
fn = 'activity_2451044875.tcx'
fol = '2018_01_22_Figs'
sprints = [3,5,7,9,11,13]
# end%%

# %%
if not os.path.exists(fol):
    os.makedirs(fol)
# end%%

# %%
tree = ET.parse(fn)
root = tree.getroot()
schema_string = '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}'

def named_node(name,schema_string=schema_string):
    return schema_string + name
# for child in root[0][0]:
#     print(child.tag,child.attrib)
# end%%


# %% XPath searching
activity = root[0][0]
time = pd.to_datetime([x.text for x in activity.findall('.//' + named_node('Time'))])
bpm = np.array([x[0].text for x in activity.findall('.//' + named_node('HeartRateBpm'))]).astype(np.float)


distance = np.array(
    [x.text for x in
    activity.findall('./' + named_node('Lap') + '/' + named_node('Track') + '//' + named_node('DistanceMeters'))]
    ).astype(np.float)
speed = np.array([x.text for x in activity.findall('.//'  + '{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Speed')]).astype(np.float)
cadence = np.array([x.text for x in activity.findall('.//'  + '{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Cadence')]).astype(np.float)

speed = speed*2.2369
lap_starts = pd.to_datetime([x.attrib['StartTime'] for x in activity.findall('.//' + named_node('Lap'))])

elapsed_time = (time - time[0]).total_seconds()
lap_starts_times = (lap_starts - lap_starts[0]).total_seconds()

# end%%

# %% Plot overall
fig, ax = plt.subplots()
ax.plot(elapsed_time,speed)
for sprint_start in sprints:
    ax.axvspan(lap_starts_times[sprint_start],lap_starts_times[sprint_start+1],color=sns.color_palette()[2],alpha=0.5,zorder=0)
# for lap_starts_time in lap_starts_times:
#     ax.axvline(lap_starts_time,color='k',ls='--')

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


# elapsed_time[-1] - 3600-5*60
# strfdelta(elapsed_time[-1],'{total_minutes}:{seconds} or {total_seconds}')
def timeTicks(x,pos):
    return strfdelta(x,'{total_minutes}:{seconds}')
ax.xaxis.set_major_formatter(FuncFormatter(timeTicks))
ax.set_ylabel('Speed [MPH]',color=sns.color_palette()[0])
ax.tick_params('y', colors=sns.color_palette()[0])
ax2 = ax.twinx()
ax2.plot(elapsed_time,bpm,color=sns.color_palette()[1])
ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax.get_yticks())-2))
ax2.set_ylabel('Heart Rate [BPM]', color=sns.color_palette()[1])
ax2.tick_params('y', colors=sns.color_palette()[1])
fig.savefig(fol + '\overall.png',bbox_inches='tight')

# for x in ax.get_ygridlines():
#     x.set_color(sns.color_palette()[0])
#     x.set_alpha(0.5)
# for x in ax2.get_ygridlines():
#     x.set_color(sns.color_palette()[1])
#     x.set_alpha(0.5)

# ax.grid(color=sns.color_palette()[0],zorder=0)
# ax2.grid(color=sns.color_palette()[1],zorder=0)


# end%%

# %% Now we isolate the actual running bits


lap_objs = activity.findall('.//' + named_node('Lap'))


dfs = []
distance_counter = []
for i,lap_obj in enumerate([lap_objs[x] for x in sprints]):
    # print(lap_obj.find(named_node('DistanceMeters')).text)
    # print(lap_obj.find(named_node('TotalTimeSeconds')).text)

    bpm = np.array(
        [x[0].text for x in lap_obj.findall('.//' + named_node('HeartRateBpm'))]
        ).astype(np.float)
    distance = np.array(
        [x.text for x in
        lap_obj.findall('./' + named_node('Track') + '//' + named_node('DistanceMeters'))]
        ).astype(np.float)
    speed = np.array(
        [x.text for x in lap_obj.findall(
        './/'  +
        '{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Speed')]
        ).astype(np.float)*2.2369
    cadence = np.array(
        [x.text for x in lap_obj.findall(
        './/'  +
        '{http://www.garmin.com/xmlschemas/ActivityExtension/v2}RunCadence')]
        ).astype(np.float)
    time = pd.to_datetime([x.text for x in lap_obj.findall('.//' + named_node('Time'))])

    distance = distance - distance[0]
    elapsed_time = (time - time[0]).total_seconds()

    # print len(elapsed_time),len(cadence),len(speed),len(distance),len(bpm)
    dfs.append(pd.DataFrame({'time':elapsed_time,'cadence':cadence,'speed':speed,
        'distance':distance,'bpm':bpm}))
    distance_counter.append('%s %d'%(lap_obj.find(named_node('DistanceMeters')).text,i))

# ax.xaxis.set_major_formatter(FuncFormatter(timeTicks))

# end%%


# %% Plot functions
def plot_dfs(df_array,x,y,legend=distance_counter,ax=None,**kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    for df,leg in zip(df_array,legend):
        df.plot(x=x,y=y,label=leg,ax=ax,**kwargs)
    return ax
# end%%

# %% Make the plots
ax = plot_dfs(dfs,'distance','speed')
ax.set_ylabel('Speed [MPH]')
ax.set_xlabel('Distance [m]')
ax.get_figure().savefig(fol + '\Speed_vs_distance.png',bbox_inches='tight')
ax = plot_dfs(dfs,'distance','bpm')
ax.set_ylabel('Heart Rate [BPM]')
ax.set_xlabel('Distance [m]')
ax.get_figure().savefig(fol + '\BPM_vs_distance.png',bbox_inches='tight')
ax = plot_dfs(dfs,'distance','cadence')
ax.set_ylabel('Cadence [Steps/min]')
ax.set_xlabel('Distance [m]')
ax.get_figure().savefig(fol + '\Cadence_vs_distance.png',bbox_inches='tight')
ax = plot_dfs(dfs,'time','distance',ls='none',marker='o')
ax.set_ylabel('Distance [m]')
ax.set_xlabel('Time [s]')
ax.legend()
ax.get_figure().savefig(fol + '\distance_vs_time.png',bbox_inches='tight')
ax = plot_dfs(dfs,'bpm','speed',ls='none',marker='o')
ax.set_ylabel('Speed [MPH]')
ax.set_xlabel('Heart Rate [BPM]')
# ax.set_ylim([10,20])
ax.legend()
ax.get_figure().savefig(fol + '\BPM_vs_Sp.png',bbox_inches='tight')
# end%%

# %% Bar Chart
times = []
for df in dfs:
    times.append(df['time'].iloc[-1])
times
ax = pd.DataFrame(times,index=distance_counter).plot(kind='bar',legend=False)
ax.set_ylabel('Time [s]')
ax.set_title('Times for various sprints')
# ax.set_ylim([200.8,202.2])
ax.get_figure().savefig(fol + '\Times_bar.png',bbox_inches='tight')
# end%%

# %% Normalized Bar Chart
normed_distance = 800.0
times = []
for df in dfs:

    times.append(df['time'].iloc[-1]/df['distance'].iloc[-1]*normed_distance)
ax = pd.DataFrame(times,index=[range(len(sprints))]).plot(kind='bar',legend=False)
ax.set_ylabel('Time [s]')
ax.set_title('Normalized %0.0fm Times for various sprints'%(normed_distance))

ax.get_figure().savefig(fol + '\Times_bar_normed.png',bbox_inches='tight')
# end%%
