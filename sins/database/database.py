from collections import defaultdict
from copy import deepcopy

import lazy_dataset
import numpy as np
import samplerate
import soundfile
from cached_property import cached_property
from lazy_dataset.database import JsonDatabase
from sins.database.utils import annotate, add_audio_paths
from sins.paths import jsons_dir


class SINS(JsonDatabase):
    def __init__(self, json_path=jsons_dir / 'sins.json'):
        super().__init__(json_path)

    @property
    def node_to_room(self):
        return self.data["node_to_room"]

    @cached_property
    def room_to_nodes(self):
        room_to_nodes_ = defaultdict(list)
        for node, room in self.node_to_room.items():
            room_to_nodes_[room].append(node)
        return {room: sorted(nodes) for room, nodes in room_to_nodes_.items()}

    @property
    def sessions(self):
        return deepcopy(self.data["annotations"])

    @cached_property
    def day_ranges(self):
        ranges = []
        sessions = sorted(self.sessions, key=(lambda x: x[1]))
        cur_offset = sessions[0][1]
        for i, session in enumerate(sessions):
            if session[0] == "sleeping:bedroom":
                split_time = (session[1] + session[2]) / 2
                ranges.append((cur_offset, split_time))
                cur_offset = split_time
        ranges.append((cur_offset, sessions[-1][2]))
        return ranges

    @property
    def train_ranges(self):
        return [self.day_ranges[i] for i in [0, 2, 3, 6, 7]]

    @property
    def validate_ranges(self):
        return [self.day_ranges[5]]

    @property
    def eval_ranges(self):
        return [self.day_ranges[i] for i in [1, 4]]

    def get_segments(
            self, dataset_name, max_segment_length, min_segment_length=.1,
            time_ranges=None, sessions=None, session_key="scene",
            annotations=None
    ):
        if isinstance(dataset_name, (tuple, list)):
            return [
                self.get_segments(
                    name,
                    max_segment_length=max_segment_length,
                    min_segment_length=min_segment_length,
                    time_ranges=time_ranges,
                    sessions=sessions, session_key=session_key,
                    annotations=annotations
                )
                for name in dataset_name
            ]

        if time_ranges is None:
            time_ranges = [(
                self.data["annotations"][0][1], self.data["annotations"][-1][2]
            )]
        else:
            time_ranges = sorted(time_ranges, key=lambda x: x[0])
            assert all([
                time_ranges[i][0] >= time_ranges[i - 1][1]
                for i in range(1, len(time_ranges))]
            )
        if sessions is None:
            segments = get_segments(
                time_ranges,
                max_segment_length=max_segment_length,
                min_segment_length=min_segment_length,
                segment_key_prefix=dataset_name + '_',
                dataset=self.get_examples(dataset_name)
            )
        else:
            sessions = [
                (session[0], max(session[1], start), min(session[2], stop))
                for start, stop in time_ranges
                for session in sessions
                if session[1] < stop and session[2] > start
            ]
            segments = get_session_segments(
                sessions=sessions,
                max_segment_length=max_segment_length,
                min_segment_length=min_segment_length,
                session_key=session_key,
                segment_key_prefix=dataset_name + '_',
                dataset=self.get_examples(dataset_name)
            )

        if annotations is not None:
            for key, annotation in annotations.items():
                if isinstance(annotation, dict):
                    annotation = annotation.values()
                annotate(
                    segments,
                    annotation=sorted(annotation, key=lambda x: x[1]),
                    label_key=key
                )
        return self._lazy_dataset_from_dict(segments, dataset_name)

    @staticmethod
    def _lazy_dataset_from_dict(examples, dataset_name):
        for example_id in examples.keys():
            examples[example_id] = {
                **examples[example_id],
                'example_id': example_id,
                'dataset': dataset_name,
            }
        return lazy_dataset.from_dict(examples)


def get_segments(
        time_ranges, max_segment_length=60 * 60, min_segment_length=.1,
        segment_key_prefix='', dataset=None
):
    """

    Args:
        time_ranges
        max_segment_length:
        min_segment_length:
        segment_key_prefix:
        dataset:

    Returns:

    """
    segments = {
        '{}{:.0f}_{:.0f}'.format(
            segment_key_prefix, segment_start,
            min(segment_start + max_segment_length, stop_time)
        ): {
            'timestamp': segment_start,
            'audio_length': min(max_segment_length, stop_time - segment_start)
        }
        for start_time, stop_time in time_ranges
        for segment_start in np.arange(
            start_time, stop_time, max_segment_length
        ) if (
            min(max_segment_length, stop_time - segment_start)
            >= min_segment_length
        )
    }
    if dataset is not None:
        add_audio_paths(segments, dataset)
    return segments


def get_session_segments(
        sessions: (list, tuple), max_segment_length=1e6, min_segment_length=.1,
        session_key="scene", segment_key_prefix='', dataset=None
):
    """

    Args:
        sessions:
        max_segment_length:
        min_segment_length:
        session_key:
        segment_key_prefix:
        dataset:

    Returns:

    """
    sessions = sorted(sessions, key=lambda x: x[1])
    segments = {}
    for (label, session_start, session_stop) in sessions:
        for segment_start in np.arange(
            session_start, session_stop, max_segment_length
        ):
            if session_stop - segment_start >= min_segment_length:
                key = '{}{:.0f}_{:.0f}'.format(
                    segment_key_prefix, segment_start,
                    min(segment_start + max_segment_length, session_stop)
                )
                segments[key] = {
                    'timestamp': segment_start,
                    'audio_length': min(
                        max_segment_length, session_stop - segment_start
                    )
                }
                if session_key is not None:
                    segments[key][session_key] = label
    if dataset is not None:
        add_audio_paths(segments, dataset)
    return segments


class AudioReader:
    def __init__(self, source_sample_rate=16000, target_sample_rate=16000):
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate

    def read_file(self, filepath, start_sample=0, stop_sample=None):
        if isinstance(filepath, (list, tuple)):
            start_sample = start_sample \
                if isinstance(start_sample, (list, tuple)) \
                else len(filepath) * [start_sample]
            stop_sample = stop_sample \
                if isinstance(stop_sample, (list, tuple)) \
                else len(filepath) * [stop_sample]
            return np.concatenate([
                self.read_file(filepath_, start_, stop_)
                for filepath_, start_, stop_ in zip(
                    filepath, start_sample, stop_sample
                )
            ], axis=-1)

        filepath = str(filepath)
        x, sr = soundfile.read(
            filepath, start=start_sample, stop=stop_sample, always_2d=True
        )
        assert sr == self.source_sample_rate
        if self.target_sample_rate != sr:
            x = samplerate.resample(
                x, self.target_sample_rate / sr, "sinc_fastest"
            )
        return x.T

    def __call__(self, example):
        audio_path = example["audio_path"]
        start_samples = 0
        if "audio_start_samples" in example:
            start_samples = example["audio_start_samples"]
        stop_samples = None
        if "audio_stop_samples" in example:
            stop_samples = example["audio_stop_samples"]

        audio = self.read_file(audio_path, start_samples, stop_samples)
        example["audio_data"] = audio
        return example
