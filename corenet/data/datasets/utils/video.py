#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Contains helper functions for reading from video detection datasets.

    NOTE: Annotations are stored via a @rectangles_dict of the form:
        Dict:
            key -> identity:
                Annotation list of Dicts for different timestamps:
                    timestamp (float): The timestamp representing the seconds since the
                        video began, ex. 1.2 is 1.2 seconds into the video.
                    x0 (float): Normalized pixel space coordinate of top left of
                        bounding box.
                    y0 (float): Normalized pixel space coordinate of top left of
                        bounding box.
                    x1 (float): Normalized pixel space coordinate of bottom right of
                        bounding box.
                    y1 (float): Normalized pixel space coordinate of bottom right of
                        bounding box.
                    <class_label_name> (int): Label of the class. The key to
                        this field depends on the dataset.
                    is_visible (bool): []Optional] Whether bounding box is
                        visible.
        See `tests/data/datasets/utils/video_test.py` for an example of this dictionary.
"""

import functools
from typing import Any, Collection, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from corenet.utils import logger

EPS = 1e-6


def _simultaneous(
    this_timestamp: float,
    other_timestamp: float,
    time_eps: float = EPS,
) -> bool:
    return abs(this_timestamp - other_timestamp) < time_eps


def _before(this_timestamp: float, other_timestamp: float) -> bool:
    return (
        not _simultaneous(this_timestamp, other_timestamp)
        and this_timestamp < other_timestamp
    )


def fetch_labels_from_timestamps(
    class_label_name: str,
    timestamps: List[float],
    rectangles_dict: Dict[str, List[Dict[str, Any]]],
    interpolation_cutoff_threshold_sec: Optional[float] = None,
    progressible_labels: Optional[Collection[int]] = None,
    carry_over_keys: Optional[Union[List[str], Literal[True]]] = None,
    required_keys: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Returns object labels for the specified video frame timestamps.

    The result will retain the structure of `rectangles_dict`, but just ensure that the
    timestamp values are as requested.

    If `progressible_labels` are supplied, the `"progress"` field will be included. This
    field represents the 'normalized' amount of time that the class label has existed
    temporally. See
    tests/data/datasets/utils/test_video.py:test_fetch_frame_with_progress for examples.

    This fetching function can be used for (per-frame) video classification pipelines.

    Args:
        class_label_name: The field name in `rectangles_dict` that maps to the class
            label.
        timestamps: A list of timestamps to fectch label from.
        rectangles_dict: (See docstring at top of file.)
        interpolation_cutoff_threshold_sec: Threshold under which we allow
            interpolation. In some `rectangles_dict`s, the labels (within the same
            track) are so far apart (e.g. 10 seconds) that interpolation is
            non-sensical. Thus this value prevents unrelated labels from being
            interpolated.
        progressible_labels: Set of labels for which to calculate `"progress"` for the
            resulting bounding boxes. If None, no `"progress"` field will be included.
        carry_over_keys: A list of keywords that specifies which keys should be carried
            over from the previous rectangle during interpolation. Defaults to None.
            True means to carry over all keys.
        required_keys: A list of keywords that specifies which keywords need to be
            included in a new bounding_box in addition to the @class_label_name.
            Defaults to None.

    Returns:
        Dict containing the labels, still indexable by track id.
    """

    if progressible_labels is not None and len(progressible_labels) > 1:
        raise NotImplementedError(
            "Currently only the calculation of one progressible label is supported;"
            f" got labels={progressible_labels}."
        )

    labels = []
    for timestamp in timestamps:
        labels.append(
            _fetch_frame_label(
                class_label_name,
                timestamp,
                rectangles_dict,
                interpolation_cutoff_threshold_sec,
                progressible_labels,
                carry_over_keys,
                required_keys,
            )
        )

    # Reslice @labels to be a dict of lists.
    ret = {}
    for label in labels:
        # @label is a dict with a key and dict of values.
        for k, v in label.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)

    return ret


def _make_fake_bbox(
    rectangles_dict: Dict[str, List[Dict[str, Any]]],
    timestamp: float,
    progressible_labels: Optional[Collection[int]] = None,
    required_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Creates a fake bounding box annotation.

    Args:
        rectangles_dict: dictionary of annotations.
        timestamp: timestamp for make bounding box.
        progressive_labels: whether to add progressive labels or not.
        required_keys: A list of keywords that specifies which keywords need to be
            included in a new bounding_box in addition to the class_label_name. Defaults
            to None.

    Returns:
        bounding box annotation at @timestamp that is not visible with all other values
        set to None.
    """
    keys = rectangles_dict[list(rectangles_dict.keys())[0]][0].keys()
    res = {key: None for key in keys}
    res["is_visible"] = False
    res["timestamp"] = timestamp
    if required_keys is not None:
        for key in required_keys:
            res[key] = None

    if progressible_labels is not None and len(progressible_labels) > 0:
        res["progress"] = None
    return res


def _fetch_frame_label(
    class_label_name: str,
    timestamp: float,
    rectangles_dict: Dict[str, List[Dict[str, Any]]],
    interpolation_cutoff_threshold_sec: Optional[float] = None,
    progressible_labels: Optional[Collection[int]] = None,
    carry_over_keys: Optional[Union[List[str], Literal[True]]] = None,
    required_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Returns object labels for the specified video frame timestamp.

    The result will retain the structure of `rectangles_dict`, but just ensure that the
    timestamp value is as requested.

    If `progressible_labels` are supplied, the `"progress"` field will be included. This
    field represents the 'normalized' amount of time that the class label has existed
    temporally. See
    tests/data/datasets/utils/test_video.py:test_fetch_frame_with_progress for examples.

    This fetching function can be used for (per-frame) video classification pipelines.

    Args:
        class_label_name: The field name in `rectangles_dict` that maps to the class
            label.
        timestamps: A list of timestamps to fectch label from.
        rectangles_dict: (See docstring at top of file.)
        interpolation_cutoff_threshold_sec: Threshold under which we allow
            interpolation. In some `rectangles_dict`s, the labels (within the
            same track) are so far apart (e.g. 10 seconds) that interpolation is
            non-sensical. Thus this value prevents unrelated labels from being
            interpolated.
        progressible_labels: Set of label values for which to calculate `"progress"` for
            the resulting bounding boxes. If None, no `"progress"` field will be included.
        carry_over_keys: A list of keywords that specifies which keys should be carried
            over from the previous rectangle during interpolation. Defaults to None.
            True means to carry over all keys.
        required_keys: A list of keywords that specifies which keywords need to be
            included in a new bounding_box in addition to the class_label_name. Defaults
            to None.

    Returns:
        Dict containing the labels, still indexable by track id.
    """

    ret = {}
    for track_label, track_rectangles in rectangles_dict.items():
        all_times = [a["timestamp"] for a in track_rectangles]
        if not (all_times) == sorted(all_times):
            raise RuntimeError("all_times should be sorted.")

        if _before(timestamp, all_times[0]) or _before(all_times[-1], timestamp):
            # The track doesn't exist or has ceased to exist.
            ret[track_label] = _make_fake_bbox(
                rectangles_dict, timestamp, progressible_labels, required_keys
            )
            continue

        idx = np.searchsorted(np.array(all_times), timestamp, side="right")

        # idx may be at the start/end if the timestamp is within EPS
        before_idx = max(idx - 1, 0)
        after_idx = min(idx, len(all_times) - 1)

        before_time = all_times[before_idx]
        after_time = all_times[after_idx]

        # Either box for interpolation is invisible.
        if (
            not track_rectangles[before_idx]["is_visible"]
            or not track_rectangles[after_idx]["is_visible"]
        ):
            # We make a fake annotation for invisible boxes.
            ret[track_label] = _make_fake_bbox(
                rectangles_dict, timestamp, progressible_labels, required_keys
            )
            continue

        # Boxes for interpolation are too far away.
        if (
            interpolation_cutoff_threshold_sec is not None
            and after_time - before_time > interpolation_cutoff_threshold_sec
        ):
            ret[track_label] = _make_fake_bbox(
                rectangles_dict, timestamp, progressible_labels, required_keys
            )
            continue

        # pylint: disable=unbalanced-tuple-unpacking
        x0, x1, y0, y1 = _interpolate_bounding_box(
            track_rectangles[before_idx],
            track_rectangles[after_idx],
            timestamp - before_time,
            after_time - before_time,
        )

        if carry_over_keys is None:
            new_label = {}
        elif carry_over_keys is True:
            # Copy the whole thing - bbox coords/timestamp are overridden below
            new_label = {**track_rectangles[before_idx]}
        else:
            new_label = {
                key: track_rectangles[before_idx][key]
                for key in carry_over_keys
                if key in track_rectangles[before_idx]
            }

        if required_keys is not None:
            for key in required_keys:
                new_label[key] = track_rectangles[before_idx].get(key, None)
        # New label will have updated coordinates and timestamp.
        new_label["x0"] = x0
        new_label["x1"] = x1
        new_label["y0"] = y0
        new_label["y1"] = y1
        new_label["timestamp"] = timestamp

        if progressible_labels is not None:
            progress = None
            if track_rectangles[before_idx][class_label_name] in progressible_labels:
                search_fn = functools.partial(
                    _search_for_label_edge_timestamp,
                    class_label_name,
                    track_rectangles,
                    before_idx,
                    interpolation_cutoff_threshold_sec=interpolation_cutoff_threshold_sec,
                )
                start_timestamp = search_fn(-1)
                end_timestamp = search_fn(+1)
                progress = (timestamp - start_timestamp) / (
                    end_timestamp - start_timestamp
                )
            new_label["progress"] = progress

        _assert_progress_repr(class_label_name, progressible_labels, new_label)
        ret[track_label] = new_label

    return ret


def _assert_progress_repr(
    class_label_name: str,
    progressible_labels: Optional[Collection[int]],
    new_label: Dict[str, Any],
) -> None:
    if progressible_labels is None:
        assert "progress" not in new_label
    else:
        assert "progress" in new_label
        if new_label[class_label_name] in progressible_labels:
            # We shouldn't ever return a 'progressible label' while not returning the
            # `"progress"` field.
            assert 0.0 <= new_label["progress"] <= 1.0
        else:
            # We shouldn't ever return the `"progress"` field while not returning a
            # 'progressible label'.
            assert new_label["progress"] == None


def _search_for_label_edge_timestamp(
    class_label_name: str,
    track_rectangles: List[Dict[str, Any]],
    cur_idx: int,
    step: int,
    interpolation_cutoff_threshold_sec: Optional[float] = None,
) -> float:
    """Find the timestamp of the furthest invisible annotation that has the same label
    with `class_label_name`.

    Args:
        class_label_name: The string name of the target class's annotation to search.
        track_rectangles: The annotation of an identity across time.
        cur_idx: The index of the annotation in `track_rectangles` to start searching.
        step: The step to search for the timestamp. A positive step indicates the
            timstamp should be after `cur_idx`'s; while a negative step indicates the
            timestamp should be before `cur_idx`'s.
        interpolation_cutoff_threshold_sec: The threshold of timestamp difference where
            the label value changes.

    Returns:
        The edge timestamp.
    """
    label = track_rectangles[cur_idx][class_label_name]

    def in_bounds(idx: int) -> bool:
        return 0 <= idx < len(track_rectangles)

    while True:
        cur_idx += step
        if not in_bounds(cur_idx):
            # If the original video clips were split into smaller clips before running
            # this pipeline (which occurs with some datasets), then there is a chance
            # that this annotation was split into two separate clips. However, since the
            # label is at the edge of the clip, we don't know if it was split, or if the
            # label just coinciedentally starts/ends at this edge of the video.
            logger.warning(
                "Annotation is potentially split across video clips. This "
                "makes the 'progress' calculation inherently unreliable."
            )
            break
        cur_rectangle = track_rectangles[cur_idx]
        if cur_rectangle[class_label_name] != label or not cur_rectangle["is_visible"]:
            break

    outside_idx = cur_idx  # First encountered outside label idx.
    inside_idx = cur_idx - step  # Last known 'inside' label idx.
    inside_timestamp = track_rectangles[inside_idx]["timestamp"]

    if in_bounds(outside_idx):
        # Since we know that the label at `outside_idx` is valid, but different than the
        # relevant action, we can interpolate halfway between (of course, only if the
        # labels are within the threshold).
        outside_timestamp = track_rectangles[outside_idx]["timestamp"]

        if step > 0 and (
            interpolation_cutoff_threshold_sec is None
            or (
                abs(inside_timestamp - outside_timestamp)
                < interpolation_cutoff_threshold_sec
            )
        ):
            # In this case, `outside_idx` is at the first timestamp which has a
            # differing label. Since we use the "floor" convention for computing the
            # label, `outside_timestamp` is our upper bound.
            return outside_timestamp

    return inside_timestamp


def _interpolate_bounding_box(
    before: Dict[str, Any],
    after: Dict[str, Any],
    delta: float,
    range_delta: float,
) -> Tuple[float, float, float, float]:
    """Given two adjacent bounding box annotations, return an interpolated label.

    Note that @delta and @range_delta must be positive, and @delta must be between 0 and
    `range_delta`, inclusively.

    Args:
        before: Bounding box label with lesser timestamp.
        after: Bounding box label with greater timestamp.
        delta: Time after `before` to compute label.
        range_delta: Time separating `before` and `after`.

    Returns:
        Bounding box label that is linearly interpolated between `before` and `after`.
    """
    if range_delta < 0:
        raise ValueError(
            f"@range_delta must be positive. Got range_delta={range_delta}."
        )

    if range_delta < 1e-5:
        return [before[key] for key in ["x0", "x1", "y0", "y1"]]

    if delta < 0 or delta - range_delta > EPS:
        # Some floating point arithmetic causes delta-range_delta to be a small value
        # above zero even if they're equal. Hence the epsilon.
        raise ValueError(
            f"@delta must be between 0 and @range_delta, inclusively. Got delta={delta},"
            f"range_delta={range_delta}."
        )

    ret = []
    for key in ["x0", "x1", "y0", "y1"]:
        alpha = 1.0 - delta / range_delta
        ret.append(before[key] * alpha + after[key] * (1.0 - alpha))

    return ret
