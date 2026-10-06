import numpy as np

from live_camera import correct_lower_body_glove_detections, filter_person_related_detections


class FakeBoxes:
    def __init__(self, rows):
        self.data = rows


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)

    def update(self, boxes):
        self.boxes = FakeBoxes(boxes)


def test_filter_person_related_detections_keeps_only_main_person_and_its_ppe():
    class_names = {
        0: "person",
        1: "no_helmet",
        2: "helmet",
        3: "person",
        4: "bucket",
    }
    main_person = np.array([10.0, 10.0, 200.0, 250.0, 0.95, 0.0], dtype=float)
    other_person = np.array([300.0, 50.0, 420.0, 180.0, 0.90, 3.0], dtype=float)
    helmet_inside = np.array([50.0, 20.0, 90.0, 70.0, 0.80, 2.0], dtype=float)
    background_helmet = np.array([350.0, 120.0, 420.0, 190.0, 0.75, 2.0], dtype=float)
    background_bucket = np.array([300.0, 220.0, 380.0, 310.0, 0.70, 4.0], dtype=float)

    result = FakeResult([main_person, other_person, helmet_inside, background_helmet, background_bucket])

    filtered = filter_person_related_detections(result, class_names)

    kept_classes = [int(row[5].item()) for row in filtered.boxes.data]
    assert kept_classes == [0, 2]


def test_correct_lower_body_glove_detections_keeps_gloves_as_gloves():
    class_names = {0: "person", 1: "no_gloves", 2: "no_boots"}
    person = np.array([10.0, 10.0, 100.0, 200.0, 0.90, 0.0], dtype=float)
    glove_in_hand = np.array([20.0, 35.0, 60.0, 70.0, 0.80, 1.0], dtype=float)
    glove_in_legs = np.array([25.0, 150.0, 75.0, 185.0, 0.80, 1.0], dtype=float)
    result = FakeResult([person, glove_in_hand, glove_in_legs])

    corrected = correct_lower_body_glove_detections(result, class_names)

    kept_classes = [int(row[5].item()) for row in corrected.boxes.data]
    assert kept_classes == [0, 1, 1]


def test_filter_person_related_detections_keeps_ppe_boxes_without_person():
    class_names = {0: "no_gloves", 1: "bucket"}
    glove_only = np.array([40.0, 30.0, 90.0, 100.0, 0.85, 0.0], dtype=float)
    bucket = np.array([200.0, 10.0, 300.0, 80.0, 0.70, 1.0], dtype=float)
    result = FakeResult([glove_only, bucket])

    filtered = filter_person_related_detections(result, class_names)

    kept_classes = [int(row[5].item()) for row in filtered.boxes.data]
    assert kept_classes == [0]


def test_filter_person_related_detections_keeps_boots_boxes_without_person():
    class_names = {0: "no_boots", 1: "bucket"}
    boot_only = np.array([50.0, 180.0, 120.0, 220.0, 0.85, 0.0], dtype=float)
    bucket = np.array([200.0, 10.0, 300.0, 80.0, 0.70, 1.0], dtype=float)
    result = FakeResult([boot_only, bucket])

    filtered = filter_person_related_detections(result, class_names)

    kept_classes = [int(row[5].item()) for row in filtered.boxes.data]
    assert kept_classes == [0]


def test_filter_person_related_detections_keeps_goggles_boxes_without_person():
    class_names = {0: "no_goggle", 1: "bucket"}
    goggles_only = np.array([60.0, 40.0, 120.0, 90.0, 0.80, 0.0], dtype=float)
    bucket = np.array([200.0, 10.0, 300.0, 80.0, 0.70, 1.0], dtype=float)
    result = FakeResult([goggles_only, bucket])

    filtered = filter_person_related_detections(result, class_names)

    kept_classes = [int(row[5].item()) for row in filtered.boxes.data]
    assert kept_classes == [0]


def test_correct_lower_body_glove_detections_ignores_wide_hand_like_boxes():
    class_names = {0: "person", 1: "no_gloves", 2: "no_boots"}
    person = np.array([10.0, 10.0, 110.0, 220.0, 0.90, 0.0], dtype=float)
    hand_like = np.array([15.0, 150.0, 105.0, 210.0, 0.80, 1.0], dtype=float)
    result = FakeResult([person, hand_like])

    corrected = correct_lower_body_glove_detections(result, class_names)

    kept_classes = [int(row[5].item()) for row in corrected.boxes.data]
    assert kept_classes == [0, 1]


if __name__ == "__main__":
    test_filter_person_related_detections_keeps_only_main_person_and_its_ppe()
    test_correct_lower_body_glove_detections_keeps_gloves_as_gloves()
    test_filter_person_related_detections_keeps_ppe_boxes_without_person()
    test_filter_person_related_detections_keeps_boots_boxes_without_person()
    test_filter_person_related_detections_keeps_goggles_boxes_without_person()
    test_correct_lower_body_glove_detections_ignores_wide_hand_like_boxes()
    print("OK")
