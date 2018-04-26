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

# # %%
# def print_record_data(obj):
#      for record_data in obj:
#         if record_data.units:
#             print(" * %s: %s %s" % (
#                 record_data.name, record_data.value, record_data.units,
#             ))
#         else:
#             print( " * %s: %s" % (record_data.name, record_data.value))
#
# # end%%
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
            # data = obj.get(key)
            # print(empty_tup(data.value))
            # if data.units and data.value and (empty_tup(data.value) is not None):
            #     print(data.value,data.units)
            #     d[key].append(data.value * ureg(data.units))
            # else:
            #     d[key].append(data.value)

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
    return any((df['start_time'] - df['timestamp'].shift().fillna(df['start_time'].iloc[0])) != pd.to_timedelta(0))

# l_list = list(fitfile.get_messages('lap'))
# obj = l_list[2]
# obj.get_value('enhanced_avg_speed')
# data = obj.get('enhanced_avg_speed')
#
# data.value*ureg(data.units).to('mph')
laps = df_from_messages(fitfile.get_messages('lap'))
if laps_check_overlap(laps):
    fn_bad_laps = 'Bad Laps Information.csv'
    laps[['start_time','timestamp']].to_csv(fn_bad_laps)
    logger.warning('We found overlapping laps.  See Laps output: in {}'.format(fn_bad_laps))

data_points = df_from_messages(fitfile.get_messages('record'))
data_points.sort_values('timestamp',inplace=True)
# laps[['start_time','timestamp']]


# data_points['timestamp'].between(laps['start_time'].loc[0],laps['timestamp'].loc[0])
def app_lap_get(val,laps,laps_start='start_time',laps_end='timestamp'):
    idx = laps.index[(laps[laps_start]<= val) & (laps[laps_end] > val)]

    if len(idx) ==1:
        return idx[0]
    else:
        raise ValueError('Couldnt find laps')

data_points['lap_id'] = data_points['timestamp'].map(lambda x: app_lap_get(x,laps))
data_points['seconds'] = (data_points['timestamp'] - data_points['timestamp'].loc[0])
data_points.set_index('seconds',inplace=True)
data_points.index = data_points.index.total_seconds()

laps['starts_seconds'] = (laps['start_time'] - laps['start_time'].loc[0]).map(datetime.timedelta.total_seconds)
laps['end_seconds'] = (laps['timestamp'] - laps['start_time'].loc[0]).map(datetime.timedelta.total_seconds)

def app_sprint_get(row,ave_thresh=9.0*ureg('mph'),max_thresh=12.0*ureg('mph')):
    ave = (row['enhanced_avg_speed'] * ureg('m/s')).to('mph')
    fastest = (row['enhanced_max_speed'] * ureg('m/s')).to('mph')

    return (ave > ave_thresh) and (fastest > max_thresh)


laps['sprint'] = laps.apply(app_sprint_get,axis=1)

laps[['starts_seconds','end_seconds']]

# end%%
laps
data_points['timestamp']
data_points.columns.values
# %% Overall Plot
fig,ax = plt.subplots()
(data_points['speed']*(1*ureg('m/s')).to('mph')).plot(ax=ax)
ax.xaxis.set_major_formatter(FuncFormatter(td_formatter))
ax.set_xlabel('')
ax.set_ylabel('Speed [MPH]',color=sns.color_palette()[0])
ax.tick_params('y', colors=sns.color_palette()[0])
for i,row in laps.loc[laps['sprint']].iterrows():
    ax.axvspan(row['starts_seconds'],row['end_seconds'],color=sns.color_palette()[2],alpha=0.5,zorder=0)
ax2 = ax.twinx()
data_points['heart_rate'].plot(ax=ax2,color=sns.color_palette()[1])
ax2.set_ylabel('Heart Rate [BPM]',color=sns.color_palette()[1])
ax2.tick_params('y', colors=sns.color_palette()[1])

l = ax.get_ylim()
l2 = ax2.get_ylim()
f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
ticks = f(ax.get_yticks())
ax2.yaxis.set_major_locator(FixedLocator(ticks))
# end%%

# %%

# data_points.set_index('lap_id',drop=True,append=True)
# end%%



# # %% Mess with auto dict and df creation
# for idx, lap in enumerate(fitfile.get_messages('lap',as_dict=False)):
#      if idx == 2:
#          d = {}
#          for data in lap:
#              d[data.name] = data.value
#
#
# d
# # end%%
#
# # %%
# message = fitfile.messages[0]
# laps = fitfile.get_messages('lap')
#
#
#
# lap_objs = []
# lap_number = 0
# lap_record_objs = {}
# lap_record_objs[lap_number] = []
# for obj in fitfile.messages:
#     if obj.name == 'lap':
#         lap_objs.append(obj)
#         lap_number += 1
#         lap_record_objs[lap_number] = []
#     if (obj.name=='record') and (lap_number >= 0):
#         lap_record_objs[lap_number].append(obj)
#
# lap_objs[2]
# lap_record_objs[2]
# lap_objs
# for obj in fitfile.messages:
#     print(obj.name)
#
# for obj in fitfile.get_messages('unknown_233'):
#     print_record_data(obj)
#
# for index,lap in enumerate(laps):
#     if index == 2:
#         print_record_data(lap)
# # message.
# # for index, lap in enumerate(fitfile.get_messages('lap')):
# #     if index == 2:
# #         for record_data in lap:
# #             if record_data.units:
# #                 print(" * %s: %s %s" % (
# #                     record_data.name, record_data.value, record_data.units,
# #                 ))
# #             else:
# #                 print( " * %s: %s" % (record_data.name, record_data.value))
#
# # Get all data messages that are of type record
# for record in fitfile.get_messages('record'):
#
#     # Go through all the data entries in this record
#     for record_data in record:
#
#         # Print the records name and value (and units if it has any)
#         if record_data.units:
#             print(" * %s: %s %s" % (
#                 record_data.name, record_data.value, record_data.units,
#             ))
#         else:
#             print( " * %s: %s" % (record_data.name, record_data.value))
#     print
#
# # end%%
