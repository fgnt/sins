import numpy as np


def prepare_sessions(
        sessions, room=None, include_absence=True, discard_other_rooms=False,
        discard_ambiguities=False, label_map_fn=None, merge=True
):
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
    return [
        [session[0].split(":")[0], session[1], session[2]]
        for session in sessions if session[0].endswith(room)
    ]


def discard_sessions_by_scene(sessions, scene):
    return [
        session for session in sessions if not session[0].startswith(scene)
    ]


def fill_with_label(sessions, label, start=None, stop=None, min_gap=.1):
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
    """
    Adds annotations to each example in dataset.

    Args:
        dataset:
        annotation:
        label_key:

    >>> dataset = [{'timestamp': 0., 'audio_length': 1.}, {'timestamp': 2., 'audio_length': 5.}, {'timestamp': 6., 'audio_length': 3.}]
    >>> annotation = [('a', 2., 4.), ('b', 3., 10.)]
    >>> annotate(dataset, annotation, 'label')
    >>> from pprint import pprint
    >>> pprint(dataset)

    Returns:

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
    """
    Adds audio paths to segments.

    Args:
        segments:
        files:

    >>> dataset = [{'timestamp': 0., 'audio_length': 1.}, {'timestamp': 2., 'audio_length': 5.}, {'timestamp': 6., 'audio_length': 3.}]
    >>> annotation = [('a', 2., 4.), ('b', 3., 10.)]
    >>> annotate(dataset, annotation, 'label')
    >>> from pprint import pprint
    >>> pprint(dataset)

    Returns:

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
    """
    Reads the annotation from dataset. Basically reverting annotate.

    Args:
        dataset:
        label_key:

    Returns:

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
