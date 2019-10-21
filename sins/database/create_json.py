"""
This script prepares a json file comprising information about the SINS
database. Information about the database can be found here:

https://github.com/KULeuvenADVISE/SINS_database

For each audio file a dict is stored containing the audio_path with
the following meta data:

timestamp: The global start time in seconds
    (0.0 is the start of the first recording of Node1).
timestamp_orig: timestamp according to file name.
audio_length: The length of the audio in seconds.
num_samples: The number of samples in the file.
sample_loss: denotes whether a sample loss occurred during recording.
counter_reset_samples: sample indices where counter resets occurred.
counter_reset_loss: denotes whether a reset of the counter value is missing.
scene: scene labels with local start and stop times.
scene_start_times: start times of activities within the file in seconds.
scene_stop_times: stop times of activities within the file in seconds.

The timestamps of the files appeared to not be consistent with the audio length
of the files. Therefore the counter values provided by the database were used
to refine the timestamps and compute the audio lengths. During recording
occasional sample losses occurred, which were detected by the counter values
as well.

Example usage:
python -m sins.database.create_json -db /path/to/db
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import scipy.io
from natsort import natsorted
from sins.paths import jsons_dir
from tqdm import tqdm

# global start time: timestamp of the first recording of Node1
T0 = datetime.strptime('20170130_043836_435', '%Y%m%d_%H%M%S_%f')

ROOM_NAMES = ["bathroom", "bedroom", "hall", "living", "wcroom"]
rooms = {
    "bathroom": [13],
    "bedroom": [9, 10],
    "hall": [12],
    "living": [1, 2, 3, 4, 6, 7, 8],
    "wcroom": [11]
}
node_to_room = {
    f'Node{node_id}': room for room, nodes in rooms.items()
    for node_id in nodes
}
offsets = {node: 0. for node in node_to_room}
offsets.update({
    "Node2": -0.6,
    "Node6": -0.6,
    "Node8": 0.4
})  # rough sync according to sound activity detection


def construct_database_dict(database_path):
    """
    Constructs the database dict which is going to be dumped to the json file.

    Args:
        database_path:

    Returns:

    """
    datasets = dict()  # dict of datasets (keys are Node1, Node2, ...)

    item_progress = tqdm(
        natsorted(node_to_room.keys()), desc="{0: <25s}".format('Process nodes'),
        file=sys.stdout, leave=False, ascii=True
    )

    # Collect print messages to be printed at the end.
    # Not printed directly to avoid conflict with progress bar.
    messages = list()

    # process each node individually
    for node_str in item_progress:
        node_id = int(node_str[len('Node'):])

        # load list of wav files
        wavfiles = scipy.io.loadmat(str(
            database_path / 'example_code' / 'other' /
            'WavTimestamps_Node{}.mat'.format(node_id)
        ))['WavFiles'].squeeze().tolist()
        wavfiles = [f.tolist()[0] for f in wavfiles]

        # load sample indices where counter resets occurred
        counter_reset_samples = scipy.io.loadmat(str(
            database_path / 'example_code' / 'other' /
            'Pulse_samples_Node{}.mat'.format(node_id)
        ))
        num_samples = counter_reset_samples['length_files'][..., 0]
        counter_reset_samples = counter_reset_samples['pulses'].squeeze().tolist()
        counter_reset_samples = [
            resets.squeeze() - 1 for resets in counter_reset_samples
        ]
        sample_loss = []
        counter_reset_loss = []

        # find sample losses and missing counter resets
        pulse_loss_threshold = 1000
        sample_loss_threshold = 100
        for i, reset_samples in enumerate(counter_reset_samples):
            sample_loss.append(False)
            counter_reset_loss.append(False)
            if any(
                    abs(reset_samples[1:] - reset_samples[:-1] - 16000)
                    > sample_loss_threshold
            ):
                sample_loss[-1] = True
            if any(
                    (reset_samples[1:] - reset_samples[:-1] - 16000)
                    > pulse_loss_threshold
            ):
                counter_reset_loss[-1] = True
        assert len(wavfiles) == len(num_samples) == len(counter_reset_samples)

        # Prepare dataset dict. Contains "example dicts" for each audio
        # containing the audio_path and further meta data.
        dataset = dict()
        for i, wavfile in enumerate(wavfiles):
            example_id = wavfile[:-10]
            node_id = int(example_id.split('Node')[1].split('_')[0])
            timestamp = example_id.split('_', 1)[-1]
            timestamp = (
                datetime.strptime(timestamp, '%Y%m%d_%H%M%S_%f') - T0
            ).total_seconds()
            dataset[example_id] = {
                'timestamp': timestamp,
                'audio_path': str(
                    database_path / 'audio' / node_str / 'audio' / wavfile
                ),
                'num_samples': int(num_samples[i]),
                'sample_loss': sample_loss[i],
                'counter_reset_samples': counter_reset_samples[i].tolist(),
                'counter_reset_loss': counter_reset_loss[i],
                'node_id': node_id
            }

        # refine timestamps and add audio_length using the information about
        # the counter resets
        refine_timestamps(dataset, counter_reset_samples, offsets[node_str])

        # filter the dataset based on whether audio files are available locally
        available = {str(file) for file in (
            database_path / 'audio' / node_str / 'audio'
        ).glob('*_audio.wav')}
        dataset = {
            key: example for key, example in dataset.items()
            if example['audio_path'] in available
        }

        # Collecting information about how many audio files are missing.
        # Not printed directly to avoid conflict with progress bar.
        messages.append(
            '{} files missing for Node {}.'.format(
                len(wavfiles) - len(dataset), node_id
            )
        )
        datasets[node_str] = dataset

    # print messages
    for msg in messages:
        print(msg)

    annotations = read_annotation_from_file(database_path)
    database = {
        'node_to_room': node_to_room,
        'annotations': annotations,
        'datasets': datasets
    }
    return database


def read_annotation_from_file(database_path):
    """
    Reads the annotation file.

    Args:
        database_path: local path to database where annotations can be found at

    Returns:

    """
    database_path = Path(database_path)

    # read annotations
    annotations = {}
    for room in ['living', 'bedroom', 'wcroom', 'hall', 'bathroom']:
        annotations[room] = []
        with (database_path / 'annotation' / f'{room}_labels.csv').open() as fid:
            for label, start_time, stop_time in csv.reader(fid, delimiter=';'):
                if label in ['Class', 'dont use']:
                    continue
                start_time, stop_time = [
                    datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
                    for t in [start_time, stop_time]
                ]
                label = (
                    label,
                    (start_time - T0).total_seconds(),
                    (stop_time - T0).total_seconds()
                )
                annotations[room].append(label)
    return get_apartment_annotation(annotations)


def get_apartment_annotation(scene_annotation, label_fmt="{scene}:{room}"):
    apartment_annotation_ = sorted([
        (label_fmt.format(scene=scene, room=room), start, stop)
        for room, annotation in scene_annotation.items()
        for scene, start, stop in annotation if scene != "absence"
    ], key=lambda x: x[1])

    last_scene = None
    apartment_annotation = []
    for cur_scene in apartment_annotation_:
        cur_scene = [*cur_scene]
        if last_scene is None:
            last_scene = cur_scene
        elif 3. > (cur_scene[1] - last_scene[2]) > 0.:
            cur_scene[1] = last_scene[2] = int((last_scene[2] + cur_scene[1]) / 2 * 1000)/1000
        apartment_annotation.append(cur_scene)
        if cur_scene[2] > last_scene[2]:
            last_scene = cur_scene
    return apartment_annotation


def refine_timestamps(dataset, counter_reset_idx, offset=None):
    """
    Refine timestamps and add audio_length using information from the counter
    values

    Args:
        dataset: dict of example dicts
        counter_reset_idx: list of lists with sample indices of counter resets
        offset:

    Returns:

    """
    ds = sorted(dataset.values(), key=lambda x: x['timestamp'])
    assert len({ex['node_id'] for ex in ds}) == 1, len({ex['node_id'] for ex in ds})

    # number of samples after the last counter reset in the last file
    latest_tail = 0

    # estimate sample_rate based on the number of samples between the first and
    # the last counter reset in each file
    samples, seconds = zip(*[
        (p[-1] - p[0] - 1, len(p) - 1) for i, p in enumerate(counter_reset_idx)
        if not (ds[i]['sample_loss'] or ds[i]['counter_reset_loss'])
    ])
    sr = sum(samples) / sum(seconds)

    # refine timestamps for each file
    ds[0]['timestamp_orig'] = ds[0]['timestamp']
    for i, example in enumerate(ds):

        # time in seconds before the first counter reset in the file
        if (
            i == 0
            or (
                abs(
                    example['timestamp'] - ds[i-1]['timestamp']
                    - ds[i-1]['audio_length']
                ) > 5
            )
        ):
            audio_length = counter_reset_idx[i][0] / sr
        else:
            audio_length = (
                counter_reset_idx[i][0] / (
                    counter_reset_idx[i][0] + latest_tail
                )
            )

        # time in seconds between first and last counter reset
        audio_length += (
                len(counter_reset_idx[i]) - 1 + example['counter_reset_loss']
        )

        # time in seconds after the last counter reset in the file
        latest_tail = example['num_samples'] - counter_reset_idx[i][-1]
        if (
            i == len(ds) - 1
            or (abs(ds[i+1]['timestamp'] - example['timestamp'] - audio_length) > 5)
        ):
            audio_length += latest_tail / sr
        else:
            audio_length += (
                latest_tail / (latest_tail + counter_reset_idx[i + 1][0])
            )
        example['audio_length'] = audio_length

        # get timestamp from next file and overwrite if refined timestamp
        # (current timestamp + audio length) does not deviate more than 2s.
        # Only deviates more than two seconds for Node 11 where some files are
        # missing. In that case we need to fall back to the filename timestamp.
        timestamp = example['timestamp']
        if i < len(ds) - 1:
            ds[i + 1]['timestamp_orig'] = ds[i + 1]['timestamp']
            if abs(
                timestamp + audio_length - ds[i + 1]['timestamp']
            ) < 2.1:
                ds[i + 1]['timestamp'] = (timestamp + audio_length)
    if offset is not None:
        for example in ds:
            example['timestamp'] += offset


def create_json(database_path: Path, json_path: Path):
    """
    construct database dict and dump as json

    Args:
        database_path: local path to database
        json_path: destination path of json file

    Returns:

    """
    database = construct_database_dict(database_path)
    os.makedirs(str(json_path.parent), exist_ok=True)
    with json_path.open('w') as fid:
        json.dump(database, fid, indent=4, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description='Create Json')
    if "SINS_DB_DIR" in os.environ:
        parser.add_argument(
            '--database_path', '-db', default=os.environ["SINS_DB_DIR"],
            help='Local path of SINS database.'
        )
    else:
        parser.add_argument(
            '--database_path', '-db', help='Local path of SINS database.'
        )
    parser.add_argument(
        '--json_path', '-j', default=str(jsons_dir / 'sins.json'),
        help='Local path of json file.'
    )

    args = parser.parse_args()
    create_json(
        Path(args.database_path).absolute(), Path(args.json_path).absolute()
    )


if __name__ == "__main__":
    main()
