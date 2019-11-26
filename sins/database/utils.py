import numpy as np


def prepare_sessions(
        sessions, room=None, include_absence=True, discard_other_rooms=False,
        discard_ambiguities=False, label_map_fn=None, merge=True
):
    """preprocesses sessions for a certain task.

    If you, e.g., perform sound activity detection each session can be mapped
    to presence, consecutive presence sessions can be merged to a single
    presence session, and in the gabs between presence absence can be added.

    Args:
        sessions: list of tuples (<session>, <session_onset>, <session_offset>)
        room: optional str. If set sessions for a single room will be prepared.
        include_absence: bool stating whether to add absence sessions in the
            gaps of sessions.
        discard_other_rooms: bool stating, if you prepare sessions for a single
            room, whether to discard sessions where the person is absent in the
            current room but present in another room and potentially making
            sounds. The is a special case for sleeping though: sleeping is
            always treated as absence in the other rooms even if
            discard_other_rooms is True.
        discard_ambiguities: bool stating whether to discard those times where
            two sessions overlap, e.g., when the person is between two rooms.
        label_map_fn: optional mapping function to be applied to the labels.
        merge: bool stating whether to merge consecutive sections

    Returns: preprocessed sessions as list of tuples

    """
    sessions = sorted(sessions, key=lambda x: x[1])
    start = sessions[0][1]
    stop = sessions[-1][2]
    if room is not None:
        if include_absence:
            if discard_other_rooms:
                if room != "bedroom":
                    sessions = discard_sessions_by_scene(sessions, "sleeping")
                sessions = fill_with_label(
                    sessions, f"absence:{room}", start, stop
                )
                sessions = filter_sessions_by_room(sessions, room)
            else:
                sessions = filter_sessions_by_room(sessions, room)
                sessions = fill_with_label(sessions, "absence", start, stop)
        else:
            sessions = filter_sessions_by_room(sessions, room)
    elif include_absence:
        sessions = fill_with_label(sessions, "absence", start, stop)
    if discard_ambiguities:
        sessions = discard_ambiguous_sections(sessions)
    if label_map_fn:
        sessions = [
            [label_map_fn(session[0]), *session[1:]] for session in sessions
        ]
    if merge:
        sessions = merge_consecutive_sections(sessions)
    return sessions


def filter_sessions_by_room(sessions, room):
    """only keep sessions within the room <room>

    Args:
        sessions: list of tuples
        room: str

    Returns: filtered sessions as list of tuples

    """
    return [
        [session[0].split(":")[0], session[1], session[2]]
        for session in sessions if session[0].endswith(room)
    ]


def discard_sessions_by_scene(sessions, scene):
    """discard certain scenes, e.g., sleeping, which would then be treated as
    absence.

    Args:
        sessions: list of tuples
        scene: str

    Returns: filtered sessions as list of tuples

    """
    return [
        session for session in sessions if not session[0].startswith(scene)
    ]


def fill_with_label(sessions, label, start=None, stop=None, min_gap=.1):
    """fill gaps with a certain label, e.g., absence.

    Args:
        sessions: list of tuples
        label: label str to be added in the gaps
        start: starting time
        stop: stopping time
        min_gap: minimum gap length that has to be met to be filled with
            label.

    Returns: sessions as list of tuples filled with additional sessions.

    """
    sessions = sorted(sessions, key=lambda x: x[1])
    cur_stop = sessions[0][1] if start is None else start
    for i in range(len(sessions)):
        cur_start = sessions[i][1]
        if (cur_start - cur_stop) > .1:
            sessions.append([
                label, cur_stop, cur_start
            ])
        if sessions[i][2] > cur_stop:
            cur_stop = sessions[i][2]
    stop = sessions[-1][2] if stop is None else stop
    if (stop - cur_stop) > min_gap:
        sessions.append([label, cur_stop, stop])
    return sorted(sessions, key=lambda x: x[1])


def discard_ambiguous_sections(sections):
    """discard those times where two sections overlap.

    Args:
        sections: list of tuples either of the form (<label>, <onset>, <offset>)
            or (<onset>, <offset>)

    Returns: adapted sections of the same form as input sections

    """
    sections = sorted(sections, key=lambda x: x[-2])
    if sections:
        last_section = sections[0]
        for cur_scene in sections[1:]:
            cur_start = cur_scene[-2]
            if cur_start < last_section[-1]:
                cur_scene[-2] = last_section[-1]
                last_section[-1] = cur_start
            if cur_scene[-1] > last_section[-1]:
                last_section = cur_scene
    return list(filter(lambda x: x[-1] > x[-2], sections))


def merge_consecutive_sections(sections, gap_tolerance=.1):
    """

    Args:
        sections: list of tuples either of the form (<label>, <onset>, <offset>)
            or (<onset>, <offset>)
        gap_tolerance: maximum gap length that will be bridged by merging the
            sections.

    Returns: merged sections of the same form as input sections

    """
    merged_sections = []
    active_sections = []
    for session in sorted(sections, key=lambda x: x[-2]):
        merged = False
        for i in range(len(active_sections), 0, -1):
            active_session = active_sections[i-1]
            if (active_session[-1] + gap_tolerance) < session[-2]:
                merged_sections.append(active_sections.pop(i-1))
            elif (
                (len(active_session) == 2 or active_session[0] == session[0])
                and (session[-2] - active_session[-1]) < gap_tolerance
            ):
                active_session[-1] = max(session[-1], active_session[-1])
                merged = True
        if not merged:
            active_sections.append([*session])
    merged_sections.extend(active_sections)
    return merged_sections


def annotate(dataset: (list, tuple, dict), annotation, label_key):
    """adds annotations to each example in dataset inplace.

    Args:
        dataset:
        annotation:
        label_key:

    >>> dataset = [{'timestamp': 0., 'audio_length': 1.}, {'timestamp': 2., 'audio_length': 5.}, {'timestamp': 6., 'audio_length': 3.}]
    >>> annotation = [('a', 2., 4.), ('b', 3., 10.)]
    >>> annotate(dataset, annotation, 'label')
    >>> from pprint import pprint
    >>> pprint(dataset)
    [{'audio_length': 1.0,
      'label': [],
      'label_start_times': [],
      'label_stop_times': [],
      'timestamp': 0.0},
     {'audio_length': 5.0,
      'label': ['a', 'b'],
      'label_start_times': [0.0, 1.0],
      'label_stop_times': [2.0, 5.0],
      'timestamp': 2.0},
     {'audio_length': 3.0,
      'label': ['b'],
      'label_start_times': [0.0],
      'label_stop_times': [3.0],
      'timestamp': 6.0}]

    """

    # get example list sorted by timestamp
    if not isinstance(dataset, (list, tuple)):
        assert isinstance(dataset, dict)
        dataset = dataset.values()
    dataset = sorted(dataset, key=lambda x: x['timestamp'])

    label_start_times, label_stop_times = np.array(list(zip(*[
        annot[1:] for annot in annotation
    ])))

    label_start_times_key = '{}_start_times'.format(label_key)
    label_stop_times_key = '{}_stop_times'.format(label_key)
    # add annotations to each example
    for example in dataset:
        for key in [
            label_key, label_start_times_key, label_stop_times_key
        ]:
            if key is not None:
                example[key] = list()
        rel_label_start_times = label_start_times - example['timestamp']
        rel_label_stop_times = label_stop_times - example['timestamp']
        for i in np.argwhere(
                (rel_label_stop_times > 0.)
                * (rel_label_start_times < example['audio_length'])
        ).flatten():
            example[label_key].append(annotation[i][0])
            example[label_start_times_key].append(
                max(float(rel_label_start_times[i]), 0.)
            )
            example[label_stop_times_key].append(
                min(float(rel_label_stop_times[i]), example['audio_length'])
            )


def add_audio_paths(segments: (list, tuple, dict), files: (list, tuple, dict)):
    """adds audio paths to segments inplace.

    Args:
        segments:
        files:

    >>> dataset = [{'timestamp': 0., 'audio_length': 1.}, {'timestamp': 2., 'audio_length': 5.}, {'timestamp': 6., 'audio_length': 3.}]
    >>> files = [\
            {'timestamp': 0., 'audio_length': 5., 'audio_path': '/a/b/c', 'node_id': 1, 'num_samples': 5*16000},\
            {'timestamp': 5., 'audio_length': 5., 'audio_path': '/a/b/d', 'node_id': 1, 'num_samples': 5*16000},\
        ]
    >>> add_audio_paths(dataset, files)
    >>> from pprint import pprint
    >>> pprint(dataset)
    [{'audio_length': 1.0,
      'audio_path': ['/a/b/c'],
      'audio_start_samples': [0],
      'audio_stop_samples': [16000],
      'node_id': 1,
      'timestamp': 0.0},
     {'audio_length': 5.0,
      'audio_path': ['/a/b/c', '/a/b/d'],
      'audio_start_samples': [32000, 0],
      'audio_stop_samples': [80000, 32000],
      'node_id': 1,
      'timestamp': 2.0},
     {'audio_length': 3.0,
      'audio_path': ['/a/b/d'],
      'audio_start_samples': [16000],
      'audio_stop_samples': [64000],
      'node_id': 1,
      'timestamp': 6.0}]
    """

    # get lists sorted by timestamp
    if not isinstance(segments, (list, tuple)):
        assert isinstance(segments, dict)
        segments = segments.values()
    segments = sorted(segments, key=lambda x: x['timestamp'])
    if not isinstance(files, (list, tuple)):
        assert isinstance(files, dict)
        files = files.values()
    files = sorted(files, key=lambda x: x['timestamp'])

    file_idx = 0
    segment_idx = 0
    while True:
        if segment_idx >= len(segments) or file_idx >= len(files):
            break
        file = files[file_idx]
        node_id = file['node_id']
        segment = segments[segment_idx]
        for key in [
            "audio_path", "audio_start_samples", "audio_stop_samples"
        ]:
            if key not in segment:
                segment[key] = list()
        if 'node_id' in segment:
            assert segment['node_id'] == node_id
        else:
            segment['node_id'] = node_id
        file_start = files[file_idx]['timestamp']
        file_stop = file_start + files[file_idx]['audio_length']
        segment_start = segments[segment_idx]['timestamp']
        segment_stop = segment_start + segments[segment_idx]['audio_length']
        if file_start > segment_stop:
            segment_idx += 1
        elif segment_start > file_stop:
            file_idx += 1
        else:
            segment['audio_path'].append(file['audio_path'])
            segment['audio_start_samples'].append(
                0 if segment_start <= file_start
                else int(
                    (segment_start - file_start)
                    / file['audio_length'] * file['num_samples']
                )
            )
            segment['audio_stop_samples'].append(
                file['num_samples']
                if segment_stop >= file_stop
                else int(
                    (segment_stop - file_start)
                    / file['audio_length'] * file['num_samples']
                )
            )
            if segment_stop < file_stop:
                segment_idx += 1
                while (
                    segment_idx < len(segments)
                    and (
                        segments[segment_idx]['timestamp']
                        < files[file_idx]['timestamp']
                    )
                ):
                    file_idx -= 1
            else:
                file_idx += 1


def read_annotation_from_dataset(dataset: (list, tuple, dict), label_key):
    """reads the annotation from dataset. Basically reverting annotate.

    Args:
        dataset:
        label_key:

    Returns: annotation

    >>> dataset = [{'timestamp': 0., 'audio_length': 1.}, {'timestamp': 2., 'audio_length': 5.}, {'timestamp': 6., 'audio_length': 3.}]
    >>> annotation = [('a', 2., 4.), ('b', 3., 10.)]
    >>> annotate(dataset, annotation, 'label')
    >>> read_annotation_from_dataset(dataset, 'label')
    [('a', 2.0, 4.0), ('b', 3.0, 9.0)]

    """
    annotation = []
    active_labels = {}
    label_start_times_key = \
        '{}_start_times'.format(label_key)
    label_stop_times_key = \
        '{}_stop_times'.format(label_key)
    if isinstance(dataset, dict):
        dataset = dataset.values()
    ds = sorted(dataset, key=lambda x: x['timestamp'])
    for i, example in enumerate(ds):
        cur_timestamp = example['timestamp']
        for label, label_start_time, label_stop_time in zip(
            example[label_key],
            example[label_start_times_key],
            example[label_stop_times_key]
        ):
            label_start_timestamp = cur_timestamp + label_start_time
            label_stop_timestamp = cur_timestamp + label_stop_time
            if label not in active_labels:
                active_labels[label] = [
                    label_start_timestamp, label_stop_timestamp
                ]
            else:
                if label_start_time != 0.:
                    raise Exception(label_start_time)
            if example['audio_length'] - label_stop_time > 1e-6:
                annotation.append(
                    (label, *active_labels.pop(label))
                )
            else:
                active_labels[label][1] = label_stop_timestamp

    for label, (start_time, stop_time) in active_labels.items():
        annotation.append(
            (label, start_time, stop_time)
        )
    return annotation
