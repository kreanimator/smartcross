import pytest

from evaluation.detection import Score, iou, match, size_band, visible_fractions
from vision.dataset import labels_for


def test_iou_of_identical_separate_and_half_overlapping_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_each_truth_matches_one_detection_and_extras_are_false_positives():
    truth = (0, 0, 10, 10)
    matched, false_positives = match([(truth, 0.5), (truth, 0.9)], [truth])
    assert matched == [1]
    assert false_positives == 1


def test_poor_overlap_is_a_miss_and_a_false_positive():
    matched, false_positives = match([((8, 8, 18, 18), 0.9)], [(0, 0, 10, 10)])
    assert matched == [None]
    assert false_positives == 1


def test_no_detections_or_no_truths():
    assert match([], [(0, 0, 1, 1)]) == ([None], 0)
    assert match([((0, 0, 1, 1), 0.9)], []) == ([], 1)


def test_nearer_box_hides_the_one_behind_but_not_the_reverse():
    far, near = (0, 0, 10, 10), (0, 5, 10, 20)  # The lower box in the frame is nearer.
    assert visible_fractions([far, near]) == pytest.approx([0.5, 1.0])


def test_separate_boxes_are_fully_visible():
    assert visible_fractions([(0, 0, 10, 10), (50, 50, 60, 60)]) == [1, 1]


def test_size_bands_include_their_lower_edge():
    assert size_band((0, 0, 20, 5)) == (20, 40)
    assert size_band((0, 0, 500, 5)) == (80, 10_000)


def test_score_counts_recall_and_precision():
    score = Score()
    seen, missed = (0, 0, 30, 30), (100, 100, 150, 150)
    score.add([seen, missed], [(seen, 0.9), ((300, 300, 310, 310), 0.8)])
    assert score.recall("all") == 0.5
    assert score.recall((20, 40)) == 1
    assert score.recall((40, 80)) == 0
    assert score.precision() == 0.5


def test_score_with_nothing_seen_reports_no_value():
    assert Score().recall("all") is None
    assert Score().precision() is None


def test_label_is_normalised_centre_and_size():
    assert labels_for([(0, 0, 320, 240)]) == ["0 0.250000 0.250000 0.500000 0.500000"]


def test_labels_skip_hidden_and_tiny_vehicles():
    behind, front = (100, 100, 150, 150), (100, 100, 150, 160)
    tiny = (300, 300, 304, 330)
    assert len(labels_for([behind, front, tiny])) == 1
