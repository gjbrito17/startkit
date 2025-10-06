# %% IMPORTS

import math
import os

from pathlib import Path
import random

from braindecode.datasets.base import BaseConcatDataset, BaseDataset, EEGWindowsDataset
from braindecode.models import EEGNeX
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.preprocessing.preprocess import Preprocessor, preprocess
from braindecode.samplers import RelativePositioningSampler
from eegdash import EEGChallengeDataset
from joblib import Parallel, delayed
from sklearn.preprocessing import scale as standard_scale
import torch
from torch import optim
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader

from scripts.crop_eeg import crop_eeg
from scripts.dataset_wrapper import DatasetWrapper
from scripts.datasets_utils import custom_keep_only_recordings_with, merge_datasets_by_release, delete_Cz
from scripts.relative_positioning_dataset import RelativePositioningDataset

# %% SET CONSTANTS
SFREQ = 100

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

release_list = ["R10", "R11"] # "R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"

DICT_TASKS = {
    "contrastChangeDetection": [
        "contrastChangeB1_start",
        "contrastChangeB2_start",
        "contrastChangeB3_start",
        "contrastChangeB4_start",
    ],
    # "DespicableMe": ["video_start"],
    # "DiaryOfAWimpyKid": ["video_start"],
    # "FunwithFractals": ["video_start"],
    # "ThePresent": ["video_start"],
    # "RestingState": ["resting_start"],
    # "seqLearning8target": ["seqLearning_start"],
    # "seqLearning6target": ["seqLearning_start"],
    # "symbolSearch": ["symbolSearch_start"],
    # "surroundSupp": [
    #     "surroundSuppB1_start",
    #     "surroundSuppB2_start",
    #     "surroundSuppB3_start",
    #     "surroundSuppB4_start",
    #     "surroundSuppB5_start",
    # ],
}

# %% PREPARE DATASETS

final_datasets = []
task_list = list(DICT_TASKS.keys())
print(task_list)
for task_key in task_list:
    print(f"Loading {task_key} datasets")
    print("--------------------------------")

    task_current_dataset = merge_datasets_by_release(release_list, task_key, DATA_DIR)
    task_current_dataset = custom_keep_only_recordings_with(
        DICT_TASKS[task_key], task_current_dataset
    )
    final_datasets.append(task_current_dataset)

final_datasets = BaseConcatDataset(final_datasets)

final_datasets = BaseConcatDataset([ds for ds in final_datasets.datasets if ds.description.ntimes >= 30])

print("All individual task datasets loaded")
sub_rm = [
    "NDARWV769JM7",
    "NDARME789TD2",
    "NDARUA442ZVF",
    "NDARJP304NK1",
    "NDARTY128YLU",
    "NDARDW550GU6",
    "NDARLD243KRE",
    "NDARUJ292JXV",
    "NDARBA381JGH",
]

print(final_datasets.datasets[25].raw.duration)

# %%
import mne
raw = final_datasets.datasets[50].raw
montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
raw.set_montage(montage)

# Then try plotting again
raw.plot_sensors(ch_type="eeg", show_names=True)
positions = montage.get_positions()["ch_pos"]
positions


# %% CROP DATA

# Filter out recordings that: are too short, does not have p-factor,
# have not 129 channels and are from subjects in sub_rm list
final_datasets = BaseConcatDataset(
    [
        ds
        for ds in final_datasets.datasets
        if not ds.description.subject in sub_rm
        and ds.raw.n_times >= 4 * SFREQ
        and len(ds.raw.ch_names) == 129
        and not math.isnan(ds.description["p_factor"])
    ]
)

preprocessors = [
    Preprocessor(crop_eeg, apply_on_array=False), Preprocessor(fn=delete_Cz, apply_on_array=False)  
]
preprocess(final_datasets, preprocessors, n_jobs=1)

# %% LOW-PASS FILTER
# Only frequencies below high_cut_hz will be kept.
high_cut_hz = 30  # Frequencies above 30 Hz often contain muscle artifacts (EMG), line noise, etc.

preprocessors = [
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz, n_jobs=-1),
]
preprocess(final_datasets, preprocessors)
# %% Create 4-seconds windows with 2-seconds stride
windows_ds = create_fixed_length_windows(
    final_datasets,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)

# Preprocess the windows by applying channel-wise z-score normalization.
preprocess(windows_ds, [Preprocessor(standard_scale, channel_wise=True)])


# %% APPLY RELATIVE POSITIONING
ssl_windows_ds = RelativePositioningDataset(windows_ds.datasets)
# %%  RELATIVE POSITIONING SAMPLER
tau_pos, tau_neg = int(SFREQ * 30), int(SFREQ * 30)  # 30 seconds worth of samples
n_examples_train = 250 * len(
    ssl_windows_ds.datasets
)  # Fixed number of samples (here 250 per recording) for training
random_state = 42

ssl_sampler = RelativePositioningSampler(
    ssl_windows_ds.get_metadata(),
    tau_pos=tau_pos,  # Maximum temporal distance allowed for two positive pairs (windows considered "close" in time)
    tau_neg=tau_neg,  # Minimum temporal distance required for two negative pairs (windows considered "far" in time)
    n_examples=n_examples_train,
    same_rec_neg=False,  # Negative recordings can be from different recordings
    random_state=random_state,
)

# %%
