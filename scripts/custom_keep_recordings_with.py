import numpy as np
import logging
from braindecode.datasets.base import BaseConcatDataset
from eegdash import EEGChallengeDataset


def custom_keep_only_recordings_with(desc, concat_ds):
    if isinstance(desc, str):
        desc = [desc]  # lo convertimos a lista para unificar

    kept = []
    for ds in concat_ds.datasets:
        descriptions = ds.raw.annotations.description
        if np.any([d in descriptions for d in desc]):
            kept.append(ds)
        else:
            logging.warning(
                f"Recording {ds.raw.filenames[0]} does not contain any of {desc}"
            )

    return BaseConcatDataset(kept)    

"""if __name__ == "__main__":
    release_list = "R5"
    DATA_DIR = "data"
    dataset_dm = EEGChallengeDataset(
    release=release_list,
    task="DespicableMe",
    mini=True,
    description_fields=[
        "subject", "session", "run", "task", "age", "gender", "sex", "p_factor"
    ],
    cache_dir=DATA_DIR,
    )
    for i in range(len(dataset_dm.datasets)):
        print(dataset_dm.datasets[i].raw.annotations.description)
        
        """

