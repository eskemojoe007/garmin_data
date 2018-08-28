from cx_Freeze import Executable
from cx_Freeze import setup as cx_setup
from distutils.core import setup
import os
os.environ['TCL_LIBRARY'] = r'C:\Users\212333077\AppData\Local\Programs\Python\Python36\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\212333077\AppData\Local\Programs\Python\Python36\tcl\tk8.6'

cx_setup(name='Sprint_Metrics',
  version='0.1',
  description='Sprint_Metrics 0.1',
  options={"build_exe": {"packages":['pandas','numpy','argparse','fitparse','tkinter','colored_logger','pint','seaborn']}},
  executables=[Executable('sprint_metrics.py',targetName="Sprint_Metrics.exe",base = None)],
  requires=['pandas','numpy','argparse','fitparse','tkinter','colored_logger','pint','seaborn']
  )
