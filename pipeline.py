#%% IMPORTS

from pathlib import Path
import math
import os
import random
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import l1_loss
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.models import EEGNeX
from braindecode.preprocessing.preprocess import Preprocessor, preprocess
from braindecode.samplers import RelativePositioningSampler
from eegdash import EEGChallengeDataset
from sklearn.preprocessing import scale as standard_scale

from scripts.crop_eeg import crop_eeg
from scripts.dataset_wrapper import DatasetWrapper
from scripts.custom_keep_recordings_with import custom_keep_only_recordings_with
from scripts.relative_positioning_dataset import RelativePositioningDataset
#%% LOAD DATA

## Load each individual task dataset and keep only recordings that contains the start annotations.
## Those with no stop annotation will be manage by the crop_eeg function.

tasks_list = [
    "contrastChangeDetection",
    "seqLearning8target",
    "seqLearning6target",
    "symbolSearch",
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunWithFractals",
    "ThePresent",
    "RestingState",
    "surroundSupp",
]

from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

release_list = "R5"

dataset_ccd = EEGChallengeDataset(
    release=release_list,
    task="contrastChangeDetection",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_ccd = custom_keep_only_recordings_with(["contrastChangeB1_start", "contrastChangeB2_start", "contrasChangeB3_start"], dataset_ccd)

dataset_seq8 = EEGChallengeDataset(
    release=release_list,
    task="seqLearning8target",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_seq8 = custom_keep_only_recordings_with(["seqLearning_start"], dataset_seq8)

dataset_seq6 = EEGChallengeDataset(
    release=release_list,
    task="seqLearning6target",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_seq6 = custom_keep_only_recordings_with(["seqLearning_start"], dataset_seq6)


dataset_symbol = EEGChallengeDataset(
    release=release_list,
    task="symbolSearch",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_symbol = custom_keep_only_recordings_with("symbolSearch_start", dataset_symbol)

dataset_dm = EEGChallengeDataset(
    release=release_list,
    task="DespicableMe",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_dm = custom_keep_only_recordings_with(["video_start"], dataset_dm)

dataset_diary = EEGChallengeDataset(
    release=release_list,
    task="DiaryOfAWimpyKid",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_diary = custom_keep_only_recordings_with(["video_start"], dataset_diary)

dataset_fun = EEGChallengeDataset(
    release=release_list,
    task="FunwithFractals",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_fun = custom_keep_only_recordings_with(["video_start"], dataset_fun)

dataset_present = EEGChallengeDataset(
    release=release_list,
    task="ThePresent",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_present = custom_keep_only_recordings_with(["video_start"], dataset_present)

dataset_rest = EEGChallengeDataset(
    release=release_list,
    task="RestingState",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_rest = custom_keep_only_recordings_with("resting_start", dataset_rest)

dataset_surround = EEGChallengeDataset(
    release=release_list,
    task="surroundSupp",
    mini=True,
    description_fields=[
            "subject",
            "session",
            "ntimes",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    cache_dir=DATA_DIR,
)
dataset_surround = custom_keep_only_recordings_with(["surroundSuppB1_start", "surroundSuppB2_start", "surroundSuppB3_start"], dataset_surround)

print("All individual task datasets loaded")
sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
          "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
     

SFREQ = 100

#%% CONCATENATE DATA

all_datasets_list = [
    dataset_ccd,       # contrastChangeDetection
    dataset_seq8,      # seqLearning8target
    dataset_seq6,      # seqLearning6target
    dataset_symbol,    # symbolSearch
    dataset_dm,        # DespicableMe
    dataset_diary,     # DiaryOfAWimpyKid
    dataset_fun,       # FunWithFractals
    dataset_present,   # ThePresent
    dataset_rest,      # RestingState
    dataset_surround   # surroundSupp
]
all_datasets = BaseConcatDataset(all_datasets_list)

# Take only the recordings with at least 30 seconds
all_datasets = BaseConcatDataset(
    [ds for ds in all_datasets.datasets if ds.description.ntimes >= 30]
)
#print(all_datasets.description)
print(all_datasets.datasets[25].raw.duration)

#%% LOW-PASS FILTER
# Only frequencies below high_cut_hz will be kept. 
high_cut_hz = 30 # Frequencies above 30 Hz often contain muscle artifacts (EMG), line noise, etc.

preprocessors = [
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz, n_jobs=-1),
]

preprocess(all_datasets, preprocessors)
#%% CROP DATA

# Filter out recordings that: are too short, does not have p-factor,
# have not 129 channels, are from subjects in sub_rm list, and apply crop_eeg function
all_datasets = BaseConcatDataset([
    ds for ds in all_datasets.datasets
    if (
        ds.description.subject not in sub_rm
        and ds.raw.n_times >= 4 * SFREQ
        and len(ds.raw.ch_names) == 129
        and not math.isnan(ds.description["p_factor"])
        and (crop_eeg(ds.raw) or True)  # exect the crop, but does not matter the return
    )
])

# Create 4-seconds windows with 2-seconds stride
windows_ds = create_fixed_length_windows(
    all_datasets,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)

# Preprocess the windows by applying channel-wise z-score normalization.
preprocess(windows_ds, [Preprocessor(standard_scale, channel_wise=True)])

print(all_datasets.datasets[25].raw.duration)
#%% APPLY RELATIVE POSITIONING 
ssl_windows_ds = RelativePositioningDataset(windows_ds.datasets)
# %%  RELATIVE POSITIONING SAMPLER
tau_pos, tau_neg = int(SFREQ * 30), int(SFREQ * 30) # 30 seconds worth of samples
n_examples_train = 250 * len(ssl_windows_ds.datasets) # Fixed number of samples (here 250 per recording) for training
random_state = 42

ssl_sampler = RelativePositioningSampler(
    ssl_windows_ds.get_metadata(),
    tau_pos=tau_pos, # Maximum temporal distance allowed for two positive pairs (windows considered "close" in time)
    tau_neg=tau_neg, # Minimum temporal distance required for two negative pairs (windows considered "far" in time)
    n_examples=n_examples_train,
    same_rec_neg=False, #Negative recordings can be from different recordings
    random_state=random_state,
)
