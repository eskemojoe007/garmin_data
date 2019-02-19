# %% Imports
import sys,os
import pandas as pd
import numpy as np
import argparse

from fitparse import FitFile
import tkinter as tk
from colored_logger import customLogger
from tkinter.filedialog import askopenfilename
import traceback

from pint import UnitRegistry

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator
import seaborn as sns

import datetime

ureg = UnitRegistry()
sns.set()
ms_mph = (1*ureg('m/s')).to('mph')
cp = sns.color_palette()
# end%%


def main():
    logger.info('--------------------------sprint_metrics - STARTING SCRIPT--------------------------')

    # Parse inputs
    parser = argparse_logger(
        description='Read Garmin Fit File and extract sprints, create and save plots')

    parser.add_argument('-f','--fit_fn',metavar='Fit File PATH',
        type=is_valid_file,
        help='Needs to be the full or relative path to the Garming Fit file')
    parser.add_argument('--ave_thresh',metavar='Average Speed Threshold',
        type=float, default=9.0,
        help='Average speed of a lap must be above this value in MPH in order to be identified as a sprint. Default = 9 MPH')
    parser.add_argument('--max_thresh',metavar='Max Speed Threshold',
        type=float, default=12.0,
        help='Max speed of a lap must be above this value in MPH in order to be identified as a sprint. Default = 12 MPH')
    parser.add_argument('-o','--output_descrip',metavar='Output Descrip',type=str,
        default='%s'%datetime.date.today().strftime('%Y_%m_%d'),
        help='Output file location, defaults to YYYY_MM_DD_Figs/')
    args = parser.parse_args()

    # Get inputs
    fit_fn = get_input_file(args,'fit_fn','Fit File')
    fol = '{}_Figs/'.format(args.output_descrip)
    if not os.path.exists(fol):
        os.makedirs(fol)

    ave_thresh = args.ave_thresh
    max_thresh = args.max_thresh


    #Open the file
    fitfile = FitFile(fit_fn)
    logger.debug('Successfully read the fitfile')

    # Get raw dataframes
    laps = df_from_messages(fitfile.get_messages('lap'))
    data_points = df_from_messages(fitfile.get_messages('record'))

    # Clean up and add values to data_points
    data_points.sort_values('timestamp',inplace=True)
    data_points['lap_id'] = data_points['timestamp'].map(lambda x: app_lap_get(x,laps))
    data_points['lap_distance'] = get_lap_distances(data_points)
    data_points['speed_mph'] = data_points['speed'] * ms_mph
    data_points['shitty_speed_mph'] = data_points['distance'].diff()/data_points['timestamp'].diff().dt.total_seconds() * ms_mph

    # Auto Determine sprints
    laps['sprint'] = laps.apply(lambda x: app_sprint_get(x,ave_thresh=ave_thresh*ureg('mph'),max_thresh=max_thresh*ureg('mph')),axis=1)
    # logger.debug('Found the following sprints {} (0 indexed)'.format(', '.join(laps.index[laps['sprint']]).values))
    laps['sprint_count'] = laps.groupby('sprint').cumcount()
    laps['descrip'] = laps.apply(lap_descrip,axis=1)

    # Pandas sucks at time differences as indexes when plotting...we just want to get
    # seconds
    data_points['seconds'] = (data_points['timestamp'] - data_points['timestamp'].loc[0]).map(datetime.timedelta.total_seconds)
    laps['starts_seconds'] = (laps['start_time'] - laps['start_time'].loc[0]).map(datetime.timedelta.total_seconds)
    laps['end_seconds'] = (laps['timestamp'] - laps['start_time'].loc[0]).map(datetime.timedelta.total_seconds)



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
    fig.savefig(os.path.join(fol,'overall.png'),bbox_inches='tight')
    # end%%

    # %% Sprint Bar charts
    fig,ax = plt.subplots()
    laps.loc[laps['sprint']].plot.bar(x='descrip',y='total_timer_time',legend=False,color=cp[0],ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Time [s]')

    for i,time in enumerate(laps['total_timer_time'].loc[laps['sprint']]):
        ax.text(i,time-2,'{:0.1f}'.format(time),rotation=90,ha='center',va='top',color='w')
    fig.savefig(os.path.join(fol,'bar_times.png'),bbox_inches='tight')
    # end%%

    # %% Speeds over distance multiple methods

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

    fig.savefig(os.path.join(fol,'speeds.png'),bbox_inches='tight')
    # end%%

    # %% BPM Over Distance
    fig, ax = plt.subplots(1)
    plot_sprints(data_points,laps,'lap_distance','heart_rate',ax=ax,legend=True)
    ax.set_ylabel('Heart Rate [BPM]')
    ax.set_xlabel('Distance [M]')
    ax.set_title('Heart Rate')
    fig.savefig(os.path.join(fol,'heart_rate.png'),bbox_inches='tight')
    # end%%

    # # %% Elevation Over Distance
    # fig, ax = plt.subplots(1)
    # plot_sprints(data_points,laps,'lap_distance','altitude',ax=ax,legend=True)
    # ax.set_ylabel('Elevation [M above sea]')
    # ax.set_xlabel('Distance [M]')
    # ax.set_title('Profile of course')
    # fig.savefig(os.path.join(fol,'profile.png'),bbox_inches='tight')
    # # end%%

    # %% BPM Over Distance
    fig, ax = plt.subplots(1)
    plot_sprints(data_points,laps,'heart_rate','shitty_speed_mph',ax=ax,legend=True,style='o')
    # ax.set_ylim([10,18])
    ax.set_ylabel('Speed [MPH]')
    ax.set_title('Looking for speed and heart rate trends')
    ax.set_xlabel('Heart Rate [BPM]')
    fig.savefig(os.path.join(fol,'heart_rate_v_speed.png'),bbox_inches='tight')
    # end%%

    logger.info('--------------------------sprint_metrics - ENDING SCRIPT--------------------------')


# %%
def strfdelta(seconds,fmt):
    '''
    Variable formatter for seconds...only necessary because pandas sucks at
    plotting of time deltas.  So We use this to make a variable plot over timeself.
    '''

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
    '''
    Gets the names for all the data in a fit message
    '''
    names = []
    for data in obj:
        names.append(data.name)

    return names

def get_unique_names(message_iter):
    '''
    Gets all the unique names for the various messages within the message_iter object
    Relies on get_obj_names to get the names from each object.
    '''
    names = []

    for obj in message_iter:
        names.extend(get_obj_names(obj))

    return np.unique(names)

def df_from_messages(message_iter):
    '''
    This takes a message iter object from fitfile.get_messages and creates a
    dataframe of all the non-nan fields.  It will look for all unique fields
    even if the messages may have different fields.
    '''
    message_iter = list(message_iter)
    cols = get_unique_names(message_iter)

    d = dict((el,[]) for el in cols)
    for obj in message_iter:
        for key in d.keys():
             d[key].append(obj.get_value(key))

    return remove_empty_cols(pd.DataFrame(d))

def empty_tup(x):
    '''
    Beacause the fit files often return tuples of content, we may want to know if those
    tuples are just full of empty items.  Biking data has this, so for running,
    we want to remove them.  This just removes a tuple of nones and replaces it
    with a single none value.  otherwise it returns the original input
    '''
    try:
        if all([i is None for i in x]):
            return None
        else:
            return x
    except:
        return x

def remove_empty_cols(df):
    '''
    Just stacks up with empty_tup and dropna to remove dead columns
    '''
    df_new = df.applymap(empty_tup)
    return df_new.dropna(axis=1,how='all')



def laps_check_overlap(df):
    '''
    For the laps information...we want to make sure that our laps don't overlap.
    This outputs a warning if the laps aren't all sequential.
    '''

    got_bad = any((df['start_time'] - df['timestamp'].shift().fillna(df['start_time'].iloc[0])) != pd.to_timedelta(0))

    if got_bad:
        fn_bad_laps = 'Bad Laps Information.csv'
        laps[['start_time','timestamp']].to_csv(fn_bad_laps)
        logger.warning('We found overlapping laps.  See Laps output: in {}'.format(fn_bad_laps))
        logger.warning('Continuing anyway...')
    return got_bad


def app_lap_get(val,laps,laps_start='start_time',laps_end='timestamp'):
    '''
    Function meant to be used with pandas apply.  It looks for the lap that coorelates
    with the data point and returns the unique identifier
    '''
    idx = laps.index[(laps[laps_start]<= val) & (laps[laps_end] > val)]

    if len(idx) ==1:
        return idx[0]
    else:
        logger.warning('For this datapoint, I couldnt find an appropriate lap. This point will be ommited: {}'.format(val))
        return np.nan

def app_sprint_get(row,ave_thresh=9.0*ureg('mph'),max_thresh=12.0*ureg('mph')):
    '''
    meant to be used with pandas apply, identify which laps are sprints by Looking
    at the overall lap data for speed and comparing to the thresholds.
    '''

    ave = (row['enhanced_avg_speed'] * ureg('m/s')).to('mph')
    fastest = (row['enhanced_max_speed'] * ureg('m/s')).to('mph')
    return (ave > ave_thresh) and (fastest > max_thresh)


def get_seconds_series(df,key,seconds_key):
    '''
    Just a filler function to get seconds as the index.
    '''

    return df.set_index(seconds_key)[key]

def second_axis_match(ax,ax2):
    '''
    Returns a Fixed axis locator that forces ax2 to match major locations with
    ax.  The locator still needs to be applied.  The intervals won't be great
    '''

    l = ax.get_ylim()
    l2 = ax2.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax.get_yticks())
    return FixedLocator(ticks)

def plot_sprints(data_df,lap_df,x,y,sprints_only=True,ax=None,**kwargs):
    '''
    Generic function to plot x vs y for all the sprints.
    '''

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
    '''
    Pandas apply function to make a description out of the sprints
    '''
    if row['sprint']:
        return 'S{} - {:0.0f}m'.format(row['sprint_count'],row['total_distance'])
    else:
        return 'Other'

def get_lap_distances(data_df):
    '''
    Distance for each point is total distance, we often want lap distance...This
    removes the first value from each lap from the distance.
    '''
    start_dis = data_df[['lap_id','distance']].groupby('lap_id').min()
    return data_df.apply(lambda x: x['distance'] - start_dis.loc[x['lap_id']],axis=1)



# end%%







# %% Arg parse and file loaders
class argparse_logger(argparse.ArgumentParser):
    def _print_message(self, message, file=None):
        if file is sys.stderr:
            logger.warning('Arg Parse did something bad...see below:')
            logger.error(message)
        else:
            super()._print_message(message,file=file)
def is_valid_file(arg):
    if not os.path.exists(arg):
        # parser.error("Cannot find the file: %s" % arg)
        raise argparse.ArgumentTypeError("Specified file does not exist = {}".format(arg))
    return arg

def tk_open_file(title=None):
    # http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/tkFileDialog.html - For options
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(title=title) # show an "Open" dialog box and return the path to the selected file
    if not filename:
        logger.error('You didnt select a filename for %s'%(title))
        logger.critical('ABORTING SCRIPT')
        sys.exit(0)
    return filename

def get_input_file(parsed_args,key,real_text=None):

    if real_text is None:
        real_text = key

    if vars(parsed_args)[key] is None:
        logger.debug('You didnt enter a file in the command line for %s...opening dialog'%(key))
        return tk_open_file('Select File for %s'%real_text)
    else:
        return vars(parsed_args)[key]

def log_uncaught_exceptions(ex_cls, ex, tb):
    logger.critical(''.join(traceback.format_tb(tb)))
    logger.critical('{0}: {1}'.format(ex_cls, ex))

# end%%



# %%
if __name__ == '__main__':
    logger = customLogger('root','sprint_metrics.log',mode='a')
    sys.excepthook = log_uncaught_exceptions

    main()
# end%%
