import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from risk_engine import calculate_risk
from voice_alert import speak


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "detect" / "outputs" / "factory_safety-8" / "weights" / "best.pt"
DATA_CONFIG = ROOT / "construction-ppe" / "data.yaml"
WINDOW_NAME = "Construction PPE Live Monitor"
PERSON_MIN_CONFIDENCE = 0.30
DUPLICATE_BOX_IOU = 0.55
FEET_REGION_START = 0.70
ALERT_MESSAGES = {
    "Vest Missing": "Safety vest is missing. Please wear your safety vest.",
    "Helmet Missing": "Helmet is missing. Please wear your helmet.",
    "Gloves Missing": "Gloves are missing. Please wear your gloves.",
    "Boots Missing": "Safety boots are missing. Please wear your safety boots.",
    "Goggles Missing": "Safety goggles are missing. Please wear your goggles.",
}


def normalize_ppe_name(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "no_goggles": "no_goggle",
        "goggles": "goggles",
    }
    return aliases.get(normalized, normalized)


def load_class_names(config_path: Path) -> dict[int, str]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return {int(class_id): name for class_id, name in config["names"].items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live construction PPE detection")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to trained YOLO weights")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--confidence", type=float, default=0.10, help="Detection confidence threshold")
    return parser.parse_args()


def open_camera(camera_index: int):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if camera.isOpened():
        return camera
    camera = cv2.VideoCapture(camera_index)
    return camera


def filter_person_related_detections(result, class_names: dict[int, str]):
    boxes = result.boxes
    rows = getattr(boxes, "data", None)
    if boxes is None or rows is None or len(rows) == 0:
        return result
    person_ids = {
        class_id for class_id, name in class_names.items()
        if name.lower() == "person"
    }
    person_boxes = [
        (index, row) for index, row in enumerate(rows)
        if int(row[5].item()) in person_ids
    ]

    valid_ppe_classes = {
        "helmet",
        "gloves",
        "vest",
        "boots",
        "goggles",
        "no_vest",
        "no_helmet",
        "no_goggle",
        "no_gloves",
        "no_boots",
    }

    if not person_boxes:
        keep_indices = [
            index
            for index, row in enumerate(rows)
            if normalize_ppe_name(class_names.get(int(row[5].item()), "")) in valid_ppe_classes
        ]
        filtered_rows = np.asarray([rows[index] for index in keep_indices], dtype=float)
        result.update(boxes=filtered_rows)
        return result

    main_person_index, main_person = max(
        person_boxes,
        key=lambda item: (
            float(item[1][2].item()) - float(item[1][0].item())
        ) * (
            float(item[1][3].item()) - float(item[1][1].item())
        ),
    )
    px1, py1, px2, py2 = (float(value.item()) for value in main_person[:4])

    keep_indices = [main_person_index]
    for index, row in enumerate(rows):
        if index == main_person_index:
            continue

        class_id = int(row[5].item())
        class_name = class_names.get(class_id, "").lower()

        normalized_class_name = normalize_ppe_name(class_name)
        if normalized_class_name not in valid_ppe_classes:
            continue

        center_x = (float(row[0].item()) + float(row[2].item())) / 2.0
        center_y = (float(row[1].item()) + float(row[3].item())) / 2.0

        if px1 <= center_x <= px2 and py1 <= center_y <= py2:
            keep_indices.append(index)

    filtered_rows = np.asarray([rows[index] for index in keep_indices], dtype=float)
    result.update(boxes=filtered_rows)
    return result


def correct_lower_body_glove_detections(result, class_names: dict[int, str]):
    boxes = result.boxes
    rows = getattr(boxes, "data", None)
    if boxes is None or rows is None or len(rows) == 0:
        return result

    class_ids = {name.lower(): class_id for class_id, name in class_names.items()}
    person_id = class_ids.get("person")
    gloves_id = class_ids.get("no_gloves")
    boots_id = class_ids.get("no_boots")
    if person_id is None or gloves_id is None or boots_id is None:
        return result
    people = [row[:4] for row in rows if int(row[5].item()) == person_id]
    for row in rows:
        if int(row[5].item()) != gloves_id:
            continue

        center_x = float((row[0] + row[2]).item()) / 2
        center_y = float((row[1] + row[3]).item()) / 2
        box_width = float((row[2] - row[0]).item())
        box_height = float((row[3] - row[1]).item())
        containing_people = []
        for person in people:
            px1, py1, px2, py2 = (float(value.item()) for value in person)
            if px1 <= center_x <= px2 and py1 <= center_y <= py2 and py2 > py1:
                containing_people.append((py2 - py1, py1, py2, px2 - px1))

        if not containing_people:
            continue

        _, person_top, person_bottom, person_width = min(containing_people)
        person_height = person_bottom - person_top
        if person_height <= 0 or person_width <= 0:
            continue

        relative_y = (center_y - person_top) / person_height
        relative_height = box_height / person_height
        relative_width = box_width / person_width
        aspect_ratio = box_height / box_width if box_width > 0 else 0.0

        # Only relabel as boots when the detection is clearly in the lower-body region,
        # shoe-like in shape, and not a wide hand/arm box.
        if (
            relative_y >= 0.82
            and relative_height >= 0.30
            and 0.05 <= relative_width <= 0.25
            and box_height >= box_width
            and aspect_ratio >= 0.8
        ):
            row[5] = boots_id

    return result


def filter_duplicate_boxes(result, class_names: dict[int, str]):
    boxes = result.boxes
    rows = getattr(boxes, "data", None)
    if boxes is None or rows is None or len(rows) == 0:
        return result

    person_ids = {
        class_id for class_id, name in class_names.items()
        if name.lower() == "person"
    }
    candidates = []

    for index, row in enumerate(rows):
        class_id = int(row[5].item())
        confidence = float(row[4].item())
        if class_id not in person_ids or confidence >= PERSON_MIN_CONFIDENCE:
            candidates.append((index, row, class_id))

    candidates.sort(key=lambda item: float(item[1][4].item()), reverse=True)
    selected_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    keep_indices = []
    for index, row, class_id in candidates:
        x1, y1, x2, y2 = (float(value.item()) for value in row[:4])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False
        class_boxes = selected_boxes.setdefault(class_id, [])
        for selected in class_boxes:
            sx1, sy1, sx2, sy2 = selected
            intersection = max(0.0, min(x2, sx2) - max(x1, sx1)) * max(
                0.0, min(y2, sy2) - max(y1, sy1)
            )
            selected_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
            union = area + selected_area - intersection
            if union > 0 and intersection / union >= DUPLICATE_BOX_IOU:
                duplicate = True
                break
        if not duplicate:
            keep_indices.append(index)
            class_boxes.append((x1, y1, x2, y2))

    keep_indices.sort()
    filtered_rows = np.asarray([rows[index] for index in keep_indices], dtype=float)
    result.update(boxes=filtered_rows)
    return result


def main() -> None:
    args = parse_args()
    if not args.model.is_absolute():
        args.model = ROOT / args.model
    if not args.model.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    class_names = load_class_names(DATA_CONFIG)
    model = YOLO(str(args.model))
    aliases = {"none": "no_vest", "person": "person"}
    checkpoint_names = {
        class_id: aliases.get(name.lower(), name.lower())
        for class_id, name in model.names.items()
    }
    if checkpoint_names != class_names:
        raise ValueError(
            "Model classes do not match construction-ppe/data.yaml. "
            f"Model: {checkpoint_names}; YAML: {class_names}"
        )
    model.model.names = class_names

    camera = open_camera(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    print(f"Live camera started using {args.model}.")
    print("Press Q to quit. Missing-PPE warnings are shown on screen and spoken.")
    alert_active = False
    reconnect_attempts = 0

    try:
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            success, frame = camera.read()
            if not success:
                reconnect_attempts += 1
                print(f"Camera frame could not be read (attempt {reconnect_attempts}). Reconnecting...")
                camera.release()
                camera = open_camera(args.camera)
                if not camera.isOpened():
                    print("Camera reconnect failed. Waiting for camera to become available...")
                    continue
                reconnect_attempts = 0
                continue

            reconnect_attempts = 0
            try:
                result = model(frame, conf=args.confidence, verbose=False)[0]
                result = filter_person_related_detections(result, class_names)
                result = correct_lower_body_glove_detections(result, class_names)
                result = filter_duplicate_boxes(result, class_names)
                annotated_frame = result.plot()
                detected_classes = [
                    class_names[int(box.cls[0])]
                    for box in result.boxes
                ]
                _, _, violations = calculate_risk(detected_classes)
                current_violations = tuple(violations)

                if current_violations and not alert_active:
                    warning = " ".join(
                        ALERT_MESSAGES[violation] for violation in current_violations
                    )
                    speak("Warning. " + warning, siren=True)
                    alert_active = True
                elif not current_violations:
                    alert_active = False

                if violations:
                    cv2.putText(
                        annotated_frame,
                        "PPE WARNING",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

                cv2.imshow(WINDOW_NAME, annotated_frame)
            except Exception as exc:
                print(f"Frame processing error: {exc}. Continuing camera loop.")
                continue
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()