import mne

"""from eegdash.dataset import EEGChallengeDataset"""

FILM_EVENTS = ["video_start", "video_stop"]
DICT_TASKS = {
    "contrastChangeDetection": "contrastChangeBL_start",
    "DespicableMe": FILM_EVENTS,
    "DiaryOfAWimpyKid": FILM_EVENTS,
    "FunwithFractals": FILM_EVENTS,
    "ThePresent": FILM_EVENTS,
    "RestingState": ["resting_start"],
    "seqLearning8target": ["seqLearning_start", "seqLearning_stop"],
    "seqLearning6target": ["seqLearning_start", "seqLearning_stop"],
    "symbolSearch": ["symbolSearch_start"],
    "surroundSupp": "surroundSuppBL_start",
}


def obtain_task(raw):
    name_num = str(raw.filenames[0]).split("/")[-1].split("-")

    if len(name_num) > 3:
        return name_num[2].split("_")[0], name_num[3][0]

    else:
        return name_num[2].split("_")[0], None


def check_start(start_sample: list):
    if len(start_sample) > 1:
        return start_sample[1]
    else:
        return start_sample[0]


def crop_eeg(raw):
    task, num = obtain_task(raw)

    events, event_id = mne.events_from_annotations(raw)  # Extract events and their IDs

    value = DICT_TASKS[task]

    if type(value) is list:
        if len(value) == 2:
            start_sample = check_start(
                events[events[:, 2] == event_id[DICT_TASKS[task][0]], 0]
            )  # Fin start event

            try:
                stop_sample = events[events[:, 2] == event_id[DICT_TASKS[task][1]], 0][
                    0
                ]  # Find stop event
            except KeyError:
                print(
                    f"Warning: In {str(raw.filenames[0])} task, the stop event is missing. The crop will be applied from the start event to the end of the recording."
                )
                raw.crop(tmin=start_sample / raw.info["sfreq"])
                return raw
            try:
                raw.crop(
                    tmin=start_sample / raw.info["sfreq"], tmax=stop_sample / raw.info["sfreq"]
                )
                return raw
            except ValueError:
                print(
                    f"Warning: In {str(raw.filenames[0]).split('/')[-1]} task, the crop had been already applied before."
                )
            return raw

        elif len(value) == 1:
            start_sample = check_start(
                events[events[:, 2] == event_id[DICT_TASKS[task][0]], 0]
            )  # Find start event
            try:
                raw.crop(tmin=start_sample / raw.info["sfreq"])

            except ValueError:
                print(
                    f"Warning: In {str(raw.filenames[0]).split('/')[-1]} task, the crop had been already applied before."
                )
            return raw

    else:
        id_start = DICT_TASKS[task].replace(
            "BL", "B" + num
        )  # Get the key corresponding to the 'start' event
        start_sample = events[events[:, 2] == event_id[id_start], 0][0]  # Find 'start' event
        try:
            raw.crop(tmin=start_sample / raw.info["sfreq"])

        except ValueError:
            print(
                f"Warning: In {str(raw.filenames[0]).split('/')[-1]} task, the crop had been already applied before."
            )
        return raw

    return raw


"""if __name__ == "__main__":
    DATA_DIR = "data"
    dataset_ccd = EEGChallengeDataset(task="FunWithFractals",
                                  release="R5", cache_dir=DATA_DIR,
                                  mini=True)
    for i in range(len(dataset_ccd.datasets)):
        print(i)
        raw = dataset_ccd.datasets[i].raw

        crop_eeg(raw)"""
