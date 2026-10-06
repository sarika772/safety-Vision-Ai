from ultralytics import YOLO
import cv2

# -----------------------------
# MODEL
# -----------------------------
MODEL_PATH = r".\runs\detect\outputs\factory_safety-8\weights\best.pt"

model = YOLO(MODEL_PATH)

# -----------------------------
# PPE CLASSES
# -----------------------------
PPE_CLASSES = {
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
}

MISSING_CLASSES = {
    "no_helmet": "Helmet",
    "no_gloves": "Gloves",
    "no_vest": "Vest",
    "no_boots": "Boots",
    "no_goggle": "Goggles",
}


# -----------------------------
# CHECK WHETHER PPE BELONGS
# TO A PERSON
# -----------------------------
def center_inside_box(ppe_box, person_box):

    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = ppe_box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return (
        px1 <= cx <= px2
        and py1 <= cy <= py2
    )


# -----------------------------
# RISK CALCULATION
# -----------------------------
def calculate_worker_risk(missing):

    score = 0

    if "Helmet" in missing:
        score += 40

    if "Gloves" in missing:
        score += 20

    if "Boots" in missing:
        score += 20

    if "Goggles" in missing:
        score += 20

    if "Vest" in missing:
        score += 20

    if score >= 60:
        status = "HIGH RISK"
    elif score >= 20:
        status = "MEDIUM RISK"
    else:
        status = "SAFE"

    return score, status


# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

print("Camera started.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame not received.")
        break

    # -----------------------------
    # YOLO TRACKING
    # -----------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.10,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "track_id": track_id
            })


    # -----------------------------
    # FIND PERSONS
    # -----------------------------
    persons = [
        d for d in detections
        if d["class"] == "person"
        and d["track_id"] is not None
    ]


    # -----------------------------
    # PROCESS EACH WORKER
    # -----------------------------
    for person in persons:

        person_id = person["track_id"]

        person_box = person["box"]

        px1, py1, px2, py2 = person_box


        missing = []

        detected_ppe = []


        # -----------------------------
        # CHECK PPE INSIDE PERSON BOX
        # -----------------------------
        for d in detections:

            if d["class"] == "person":
                continue

            if center_inside_box(
                d["box"],
                person_box
            ):

                if d["class"] in PPE_CLASSES:

                    detected_ppe.append(
                        d["class"]
                    )

                if d["class"] in MISSING_CLASSES:

                    missing.append(
                        MISSING_CLASSES[d["class"]]
                    )


        # remove duplicates
        detected_ppe = list(
            set(detected_ppe)
        )

        missing = list(
            set(missing)
        )


        # -----------------------------
        # RISK
        # -----------------------------
        risk_score, status = calculate_worker_risk(
            missing
        )


        # -----------------------------
        # PERSON COLOR
        # -----------------------------
        if status == "SAFE":
            color = (0, 255, 0)

        elif status == "MEDIUM RISK":
            color = (0, 165, 255)

        else:
            color = (0, 0, 255)


        # -----------------------------
        # DRAW PERSON BOX
        # -----------------------------
        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            color,
            3
        )


        # -----------------------------
        # WORKER ID
        # -----------------------------
        cv2.putText(
            frame,
            f"Worker {person_id}",
            (px1, max(25, py1 - 45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


        # -----------------------------
        # STATUS
        # -----------------------------
        cv2.putText(
            frame,
            status,
            (px1, max(25, py1 - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


        # -----------------------------
        # RISK SCORE
        # -----------------------------
        cv2.putText(
            frame,
            f"Risk: {risk_score}",
            (px1, py2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


        # -----------------------------
        # MISSING PPE
        # -----------------------------
        if missing:

            text = "Missing: " + ", ".join(missing)

            cv2.putText(
                frame,
                text,
                (px1, min(frame.shape[0] - 10, py2 + 50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )


    # -----------------------------
    # SHOW WINDOW
    # -----------------------------
    cv2.imshow(
        "Worker Tracking + Individual Risk",
        frame
    )


    # Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")