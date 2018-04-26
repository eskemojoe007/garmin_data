# sprint_metrics
This pulls in a fit file and reads it, splits it up by sprints and output several
useful plots

## Installation
You can install all the python mumbo jumbo if you'd like or just
head over to https://github.com/eskemojoe007/garmin_data/releases
to download the executable.

### Installing with all the command line and python etc
Clone, and run pipenv install in the proper directory

It is dependent on https://github.com/eskemojoe007/colored_logger
so make sure you install that.  `pipenv install -e <path to cloned/downloaded colored_logger`

## Usage
You can use it from the command line or by double clicking, they both work.

### Double Clicking
It will open 1 file dialogs asking for:
- Fit File

If you need all the log info...check the log file in the folder where the EXE is.  Outputs are put into the EXE output folder as well by date_figs\

### Command line
You can always run `sprint_metrics.exe --help` to understand, but you can also specify all the files and sheets from the command line.  Sample command line with either exe or python is:
```
sprint_metrics.exe -f 2018_04_25_Sprints.fit --ave_thresh 3.0 --max_thresh 12.0 -o with_gang
```
