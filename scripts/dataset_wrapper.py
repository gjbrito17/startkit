from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
import random

class DatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int, seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        # P-factor label:
        p_factor = self.dataset.description["p_factor"]
        p_factor = float(p_factor)

        # Additional information:
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop the signal to the desired length:
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, p_factor, (i_window_in_trial, i_start, i_stop), infos