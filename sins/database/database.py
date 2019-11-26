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
    """
    SINS Database class providing utility functions to read and preprocess
    the SINS data.
    """
    def __init__(self, json_path=jsons_dir / 'sins.json'):
        """

        Args:
            json_path: path to json that was created using
            `sins.database.create_json`
        """
        super().__init__(json_path)

    @property
    def node_to_room(self):
        """

        Returns: dict with nodes as keys and corresponding room as value.

        """
        return self.data["node_to_room"]

    @cached_property
    def room_to_nodes(self):
        """

        Returns: dict with rooms as keys and list of corresponding nodes
            as values.

        """
        room_to_nodes_ = defaultdict(list)
        for node, room in self.node_to_room.items():
            room_to_nodes_[room].append(node)
        return {room: sorted(nodes) for room, nodes in room_to_nodes_.items()}

    @property
    def sessions(self):
        """

        Returns: list of session tuples (<label>, <onset>, <offset>) sorted by
            onset.

        """
        return deepcopy(self.data["annotations"])

    @cached_property
    def day_ranges(self):
        """

        Returns: a list of (<onset>, <offset>) for each day.
            Day boundaries are in the middle of the sleeping session resulting
            in 8 days with the first and the last being half days.

        """
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
        """

        Returns: list of the suggested training ranges (days)

        """
        return [self.day_ranges[i] for i in [0, 2, 3, 6, 7]]

    @property
    def validate_ranges(self):
        """

        Returns: list of the suggested validation ranges (days)

        """
        return [self.day_ranges[5]]

    @property
    def eval_ranges(self):
        """

        Returns: list of the suggested evaluation ranges (days)

        """
        return [self.day_ranges[i] for i in [1, 4]]

    def get_segments(
            self, dataset_name, max_segment_length, min_segment_length=.1,
            time_ranges=None, sessions=None, session_key="scene",
            annotations=None
    ):
        """prepare dataset(s) providing time segments within certain
        time ranges with configurable segment length.

        Segments are dictionaries providing the paths the audio data together
        with additional information such as timestamp, audio_length and labels.
        The structure of a segment is
        {
            "example_id": str
            "timestamp": float,
            "audio_length": float,
            "audio_path": list of str,
            "audio_start_samples": list of int,
            "audio_stop_samples": list of int,
            "dataset": str,
            "node_id": int,
            # optional:
            <label_name>: str or list of str,
            <label_name>_start_times: str or list of str,
            <label_name>_stop_times: str or list of str,
        }

        The timestamp states the starting point of the segment in seconds,
        counted from the beginning of recording.
        Do note that the timing information provided by the file names was
        found to be inaccurate, which is why we used the reference clock signal
        to refine the timestamps. This is done when writing the database json.
        If you request segments from multiple nodes they will be in parallel,
        i.e. the n-th segment from Node1 and the n-th segment of Node2 have the
        same on- & offsets (timestamp & audio_length).
        Note that a segment does not start at the beginning of a certain audio,
        it rather starts somewhere within an audio file, then may span over a
        couple of complete audio files, and then stop within an audio file
        again. The exact position of the start and stop points is given by
        audio_start_samples and audio_stop_samples, which is given for each
        audio file in the segment.

        Args:
            dataset_name: str or list of dataset name(s) where dataset names
                are of the form Node<idx>
            max_segment_length: maximal length of segment in seconds. This will
                be the length of the returned segment unless the segment is the
                last within a certain range.
            min_segment_length: If a segment at the end of a time range is
                shorter, than it will be discarded. If you want to have fix
                length segments choose min_segment_length=max_segment_length.
            time_ranges: time ranges from which to read segments
            sessions: optional list of tuples
                (<session>, <start_time>, <stop_time>). If given each segment
                will be from a single session, i.e. no segment wil span over a
                session change.
            session_key: If not None and sessions not None session labels will
                be added to the segment dict under this key.
            annotations: None or dict of lists of tuples
                (<label>, <start_time>, <stop_time>).  If not None, for each
                key (label_name) there will be entries <label_name>
                <label_name>_start_times <label_name>_stop_times with values
                stating the labels, start_times and stop_times within the
                segment.

        Returns: (list of) lazy dataset(s) providing (parallel) segments.

        """
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
    """prepares segment dicts and adds audio paths to dicts.

    Args:
        time_ranges:
        max_segment_length:
        min_segment_length:
        segment_key_prefix: prefix in example_id of a segment.
            example_id will be <prefix><segment_onset>_<segment_offset>
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
    """prepares segment dicts from single sessions, i.e. which not span over
    session changes, and adds audio paths to dicts.

    Args:
        sessions:
        max_segment_length:
        min_segment_length:
        session_key:
        segment_key_prefix: prefix in example_id of a segment.
            example_id will be <prefix><segment_onset>_<segment_offset>
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
    """
    takes an example (or segment) and reads and concatenates the audio files.
    """
    def __init__(self, source_sample_rate=16000, target_sample_rate=16000):
        """

        Args:
            source_sample_rate: sample rate of the stored audio file
            target_sample_rate: target sample rate. If != source_sample_rate,
            the audio will be resampled.
        """
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
