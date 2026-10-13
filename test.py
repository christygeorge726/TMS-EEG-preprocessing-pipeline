import os
os.environ["MNE_BROWSER_BACKEND"] = "qt"

import matplotlib
matplotlib.use("Qt5Agg")

from pathlib import Path
from itertools import combinations

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.filter import notch_filter
from scipy import interpolate
from scipy.stats import sem, ttest_ind


DATA_DIR = Path("/space/Lab1/TMS-EEG/TMS-EEG-preprocessing-pipeline/data")
VHDR_PATH = DATA_DIR / "test_biphasic" / "test.vhdr"

NON_EEG = {"EMG": "emg", "VEOG": "eog", "HEOG": "eog"}
TMS_MARKERS = ["D0/D  1", "SB/A"]

if not VHDR_PATH.exists():
    raise FileNotFoundError(f"File not found: {VHDR_PATH}")

print(f"Loading: {VHDR_PATH}")
raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True)

raw.set_channel_types({ch: t for ch, t in NON_EEG.items() if ch in raw.ch_names})
raw.set_montage("colin27_1020")

raw.pick("eeg")
raw.filter(l_freq=1.0, h_freq=None)

events, event_id = mne.events_from_annotations(raw)
marker = next((m for m in TMS_MARKERS if m in event_id), None)
if marker is None:
    raise KeyError(f"No TMS marker found in {VHDR_PATH.name}. Available: {list(event_id)}")
print(f"Using marker: {marker}")
chosen_event_id = event_id[marker]

epochs = mne.Epochs(
    raw, events, event_id=chosen_event_id,
    tmin=-1.0, tmax=1.0, baseline=None, preload=True,
)

epochs.apply_baseline(baseline=(-0.01, -0.002))

print(f"Number of epochs: {len(epochs)}")