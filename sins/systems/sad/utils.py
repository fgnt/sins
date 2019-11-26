from sins import paths
from pathlib import Path
import json
from sins.database.utils import merge_consecutive_sections


def load_sections(
        file=paths.exp_dir / 'sad' / '2019-10-07-06-57-23' / 'ensemble_sections.json',
        gap_tolerance=.5, min_length=.5, time_margin=.25
):
    """load section (e.g., a sound activity detection) from a json file.

    Args:
        file: path to json file
        gap_tolerance: maximum gap length that will be bridged by merging the
            sections.
        min_length: minimum length of sections. If length is less, section will be discarded.
        time_margin: margin added to the onset and offset of each section.

    Returns:

    """
    with Path(file).open() as f:
        sections = json.load(f)
    sections = merge_consecutive_sections(sections, gap_tolerance=gap_tolerance)
    sections = [
        (start - time_margin, stop + time_margin)
        for start, stop in sections if (stop - start) > min_length
    ]
    return sections


def get_sections_in_range(sections, time_range):
    """get sections in a certain time range (e.g., in the train_ranges)

    Args:
        sections: list of tuples (<onset>, <offset>) or (<label>, <onset>, <offset>)
        time_range: (list of) tuple(s) (<onset>, <offset>)

    Returns:

    """
    if isinstance(time_range[0], (list, tuple)):
        return [
            section for time_range_ in time_range
            for section in get_sections_in_range(sections, time_range_)
        ]
    else:
        return [
            (
                max(time_range[-2], section[-2]),
                min(time_range[-1], section[-1])
            )
            for section in sections
            if (
                section[-1] >= time_range[-2]
                and section[-2] <= time_range[-1]
            )
        ]
