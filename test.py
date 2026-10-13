import os
os.environ["MNE_BROWSER_BACKEND"] = "qt"

import matplotlib
matplotlib.use("Qt5Agg")

from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

# %%
def tep_plot(ax, time, y, xlim, ylim, title=None, xticks=None, yticks=None, **kwargs):
    ax.clear()
    ax.plot(time, np.asarray(y).T, **kwargs)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Time (ms)", fontsize=16)
    ax.set_ylabel("Amplitude (µV)", fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)


def new_figure(num):
    fig = plt.figure(num)
    fig.clf()
    return fig, fig.add_subplot()


def show_now():
    plt.show(block=False)
    plt.pause(0.1)


def inspect(n_items, label, draw, fig_num):
    good, bad = [], []
    fig, ax = new_figure(fig_num)
    for i in range(n_items):
        draw(ax, i)
        fig.canvas.draw_idle()
        plt.figure(fig.number)
        button = plt.waitforbuttonpress()
        if button is None:
            raise RuntimeError(f"Figure closed during {label.lower()} inspection.")
        if button is False:
            print(f"{label} {i + 1}, Accepted")
            good.append(i)
        else:
            print(f"{label} {i + 1}, Rejected")
            bad.append(i)
    plt.close(fig)
    return np.array(good, dtype=int), np.array(bad, dtype=int)
# %%

# %% Read data
data_dir = Path("/space/Lab1/TMS-EEG/TMS-EEG-preprocessing-pipeline/data/test_biphasic")
file = "test.vhdr"

raw = mne.io.read_raw_brainvision(data_dir / file, preload=True)

srate = int(round(raw.info["sfreq"]))
data = raw.get_data(picks=raw.ch_names[:64]) * 1e6
n_times = data.shape[1]
data_dt = detrend(data, axis=1, type="linear")
del data


# %% Trials
TEP = 1

if TEP == 1:
    events, _ = mne.events_from_annotations(raw, regexp=None)
    print(f"First markers: {list(raw.annotations.description[:3])}")
    trials = events[1:, 0] - raw.first_samp
else:
    trials = np.arange(3 * srate, n_times - srate + 1, srate) - 1

print(f"Number of trials: {len(trials)}")


# %% 3D matrix: channels x time x trials
n_ch = data_dt.shape[0]
all_trials_dt = np.zeros((n_ch, 2 * srate, len(trials)))

for i, s in enumerate(trials):
    all_trials_dt[:, :, i] = data_dt[:, s - srate:s + srate]


# %% Baseline correction
time1 = np.arange(1, srate + 1) / srate
time = np.concatenate([-time1[::-1], time1]) * 1000

t1, t2 = -500, -10
idx = np.arange(len(time))
i1 = int(np.floor(np.interp(t1, time, idx) + 0.5))
i2 = int(np.floor(np.interp(t2, time, idx) + 0.5))
tind = np.arange(i1, i2 + 1)

all_trials_bc_dt = all_trials_dt - all_trials_dt[:, tind, :].mean(axis=1, keepdims=True)


# %% Average
y_dt = all_trials_bc_dt
ya_dt = y_dt.mean(axis=2)

fig, ax = new_figure(1)
tep_plot(ax, time, ya_dt, (-20, 250), (-80, 80))
show_now()

grand_ave = ya_dt.mean(axis=0)
fig, ax = new_figure(2)
tep_plot(ax, time, grand_ave, (-50, 300), (-20, 20), color="r")
show_now()


# %% Visual inspection of trials
good_tr, bad_tr = inspect(
    y_dt.shape[2], "Trial",
    lambda ax, i: tep_plot(ax, time, y_dt[:, :, i], (-100, 500), (-100, 100), title=f"Trial {i + 1}"),
    fig_num=3,
)


# %% Event file
run = file.replace(".vhdr", "")
visit = Path(str(data_dir).replace("raw_brainproducts", "raw_prepro"))

good_events = trials[good_tr] + 1
time_events = good_events / srate

visit.mkdir(parents=True, exist_ok=True)
file2 = f"{run}_dt_goodTrials_205to100ms_raw.txt"
with open(visit / file2, "w") as f:
    f.write("finalwrite1 finalwrite2 finalwrite3 finalwrite4\n")
    for e, t in zip(good_events, time_events):
        f.write(f"{e:.15g} {t:.15g} 0 1\n")
print(f"Wrote {len(good_events)} good trials to {visit / file2}")


# %% Debugged trials, all channels
y_good_tri = y_dt[:, :, good_tr]
ya_good_tri = y_good_tri.mean(axis=2)

fig, ax = new_figure(4)
tep_plot(ax, time, ya_good_tri, (-10, 250), (-80, 80), title="Debugged trials all channels")
show_now()


# %% Channel selection
y = y_good_tri
ya = y.mean(axis=2)

good_ch, bad_ch = inspect(
    y.shape[0], "Channel",
    lambda ax, i: tep_plot(
        ax, time, ya[i], (-10, 150), (-50, 50), title=f"Channel {i + 1}",
        xticks=np.arange(-10, 151, 5), yticks=np.arange(-80, 81, 10),
    ),
    fig_num=5,
)
print(f"Bad channels: {(bad_ch + 1).tolist()}")


# %% Debugged trials and channels
y_good_cha = y[good_ch, :, :]
ya_good_cha = y_good_cha.mean(axis=2)

fig, ax = new_figure(6)
tep_plot(ax, time, ya_good_cha, (-20, 150), (-30, 30),
         title="All channels with debugged trials and channels", linewidth=1)
plt.show()