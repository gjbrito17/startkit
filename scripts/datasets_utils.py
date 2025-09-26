import logging

from braindecode.datasets.base import BaseConcatDataset
from eegdash import EEGChallengeDataset
import numpy as np


def custom_keep_only_recordings_with(desc, concat_ds):
    if isinstance(desc, str):
        desc = [desc]  # lo convertimos a lista para unificar

    kept = []
    for ds in concat_ds.datasets:
        descriptions = ds.raw.annotations.description
        if np.any([d in descriptions for d in desc]):
            kept.append(ds)
        else:
            logging.warning(f"Recording {ds.raw.filenames[0]} does not contain any of {desc}")

    return BaseConcatDataset(kept)


def merge_datasets_by_release(release_list, task, data_dir):
    all_datasets = None

    for release in release_list:
        print(f"Loading {release}-{task}")
        print("--------------------------------")

        try:
            current_dataset = EEGChallengeDataset(
                release=release,
                task=task,
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
                cache_dir=data_dir,
                query={"subject": {"$not": "NDARAR935TGZ"}},
            )
        except KeyError as e:
            print(f"Error loading {release} {task} because is not available")
            print(e)
            continue

        if all_datasets is None:
            all_datasets = current_dataset
        else:
            try:
                all_datasets = BaseConcatDataset([all_datasets, current_dataset])
            except Exception as e:
                print(f"Error concatenating {release} {task} because is not available")
                print(e)

    return all_datasets
