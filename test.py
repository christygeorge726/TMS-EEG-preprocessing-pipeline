import os
os.environ["MNE_BROWSER_BACKEND"] = "qt"

import matplotlib
matplotlib.use("Qt5Agg")

import re
from pathlib import Path
from itertools import combinations
from tkinter import Tk, filedialog

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.filter import notch_filter
from scipy import interpolate
from scipy.stats import sem, ttest_ind


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", path.name)]


def select_vhdr_files(initial_dir=None):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        paths = filedialog.askopenfilenames(
            parent=root,
            title="Select BrainVision .vhdr files",
            initialdir=initial_dir,
            filetypes=[("BrainVision header", "*.vhdr")],
        )
    finally:
        root.destroy()

    if not paths:
        raise RuntimeError("No files selected.")

    return sorted((Path(p) for p in paths), key=natural_key)


vhdr_paths = select_vhdr_files(initial_dir="data")
for i, p in enumerate(vhdr_paths, 1):
    print(f"Run {i}: {p.name}")


NON_EEG = {"EMG": "emg", "VEOG": "eog", "HEOG": "eog"}

all_epochs = []
epoch_counts_per_run = []

for vhdr_path in vhdr_paths:
    print(f"Loading: {vhdr_path}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

    raw.set_channel_types({ch: t for ch, t in NON_EEG.items() if ch in raw.ch_names})
    raw.set_montage("colin27_1020")

    raw.pick("eeg")
    raw.filter(l_freq=1.0, h_freq=None)

    events, event_id = mne.events_from_annotations(raw)
    chosen_event_id = event_id["D0/D  1"]

    epochs = mne.Epochs(
        raw, events, event_id=chosen_event_id,
        tmin=-1.0, tmax=1.0, baseline=None, preload=True,
    )

    epochs.apply_baseline(baseline=(-0.01, -0.002))

    all_epochs.append(epochs)
    epoch_counts_per_run.append(len(epochs))

print(f"Epochs per run: {epoch_counts_per_run}")