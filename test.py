import os
os.environ['MNE_BROWSER_BACKEND'] = 'qt'

import matplotlib
matplotlib.use('Qt5Agg')

import mne
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from mne.preprocessing import ICA

from scipy import interpolate
from mne.filter import notch_filter
from scipy.stats import sem
from scipy.stats import ttest_ind
from itertools import combinations
import pandas as pd
