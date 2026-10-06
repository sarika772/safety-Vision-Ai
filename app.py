import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
import threading
import time
import requests
import smtplib
import hashlib
from email.message import EmailMessage

from face_recognition_engine import recognize_faces
from worker_registration import register_worker

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from risk_engine import calculate_risk
from voice_alert import speak


# =========================================================
# VOICE ALERT SETTINGS
# =========================================================

VOICE_ALERT_COOLDOWN = 5
_last_voice_alert_time = 0
_voice_alert_lock = threading.Lock()

# Shared live-camera analysis state. The WebRTC transformer updates this
# from the camera thread, while the Streamlit risk panel reads it safely.
LIVE_ANALYSIS_LOCK = threading.Lock()
LIVE_ANALYSIS = {
    "risk_score": 0,
    "risk_level": "WAITING",
    "violations": [],
    "detected_classes": [],
    "worker_names": [],
}

LIVE_ANALYSIS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".live_analysis.json"
)

def _write_live_analysis_file(data):
    """Persist latest camera analysis so the Streamlit UI can read it reliably."""
    try:
        tmp_file = LIVE_ANALYSIS_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_file, LIVE_ANALYSIS_FILE)
    except Exception:
        pass

def _read_live_analysis_file():
    default = {
        "risk_score": 0,
        "risk_level": "WAITING",
        "violations": [],
        "detected_classes": [],
        "worker_names": [],
    }
    try:
        with open(LIVE_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            default.update(data)
    except Exception:
        pass
    return default

def reset_live_analysis():
    data = {
        "risk_score": 0,
        "risk_level": "WAITING",
        "violations": [],
        "detected_classes": [],
        "worker_names": [],
    }
    with LIVE_ANALYSIS_LOCK:
        LIVE_ANALYSIS.update(data)
    _write_live_analysis_file(data)


def trigger_voice_alert(violations):

    global _last_voice_alert_time

    if not violations:
        return

    current_time = time.time()

    with _voice_alert_lock:

        if current_time - _last_voice_alert_time < VOICE_ALERT_COOLDOWN:
            return

        _last_voice_alert_time = current_time

    message = (
        "Warning. "
        + ". ".join(violations)
        + ". Please wear the required safety equipment."
    )

    # Run voice in background so live camera does not freeze
    threading.Thread(
        target=speak,
        args=(message,),
        daemon=True
    ).start()


# =========================================================
# EMAIL ALERT SETTINGS
# =========================================================
# Configure these as environment variables (recommended):
# SAFETY_EMAIL_SENDER   = sender Gmail address
# SAFETY_EMAIL_PASSWORD = Gmail App Password (NOT your normal password)
# SAFETY_EMAIL_RECEIVER = email address that should receive alerts
#
# Streamlit secrets are also supported with the same names.

EMAIL_ALERT_COOLDOWN = 10
_last_email_alert_time = 0
_last_email_state = None
_email_alert_lock = threading.Lock()


def _get_email_setting(name):
    # 1) Environment variable support
    value = os.getenv(name, "").strip()
    if value:
        return value

    # 2) Support the project's current nested [email] secrets.toml format
    #    as well as the older top-level SAFETY_EMAIL_* format.
    try:
        email_secrets = st.secrets.get("email", {})
        if hasattr(email_secrets, "get"):
            mapping = {
                "SAFETY_EMAIL_SENDER": "sender_email",
                "SAFETY_EMAIL_PASSWORD": "sender_password",
                "SAFETY_EMAIL_RECEIVER": "receiver_email",
                "SAFETY_EMAIL_SMTP_SERVER": "smtp_server",
                "SAFETY_EMAIL_SMTP_PORT": "smtp_port",
            }
            nested_key = mapping.get(name)
            if nested_key:
                value = str(email_secrets.get(nested_key, "")).strip()
                if value:
                    return value

        value = str(st.secrets.get(name, "")).strip()
        return value
    except Exception:
        return ""


# Keep these names for compatibility, but the actual send function
# re-reads the settings every time so a Streamlit restart/config refresh
# does not leave stale email credentials in memory.
EMAIL_SENDER = _get_email_setting("SAFETY_EMAIL_SENDER")
EMAIL_PASSWORD = _get_email_setting("SAFETY_EMAIL_PASSWORD")
EMAIL_RECEIVER = _get_email_setting("SAFETY_EMAIL_RECEIVER")


def _get_current_email_settings():
    """Read the latest Gmail settings from environment/Streamlit secrets."""
    return (
        _get_email_setting("SAFETY_EMAIL_SENDER"),
        _get_email_setting("SAFETY_EMAIL_PASSWORD"),
        _get_email_setting("SAFETY_EMAIL_RECEIVER"),
    )


def email_is_configured():
    sender, password, receiver = _get_current_email_settings()
    return bool(sender and password and receiver)


def send_email_alert(
    risk_level,
    risk_score,
    violations=None,
    detected_classes=None,
    worker_names=None,
    source="Safety Detection",
    location_text="",
):
    """Send one safety result by Gmail SMTP. Never crashes the AI pipeline."""
    violations = list(violations or [])
    detected_classes = list(dict.fromkeys(detected_classes or []))
    worker_names = list(dict.fromkeys(worker_names or []))

    # Read credentials at send time. This is important for Streamlit because
    # the app may be running with refreshed secrets/environment values.
    sender, password, receiver = _get_current_email_settings()

    if not (sender and password and receiver):
        error = (
            "Email settings are missing. Check [email] in .streamlit/secrets.toml."
        )
        print(error)
        _write_email_status(False, source=source, error=error)
        return False

    if risk_level == "HIGH RISK":
        status = "🔴 HIGH RISK"
    elif risk_level == "MEDIUM RISK":
        status = "🟠 MEDIUM RISK"
    else:
        status = "🟢 SAFE"

    violation_text = (
        "\n".join(f"- {item}" for item in violations)
        if violations
        else "- No PPE/safety violations detected"
    )

    detected_text = (
        ", ".join(detected_classes)
        if detected_classes
        else "No objects detected"
    )

    worker_text = (
        ", ".join(worker_names)
        if worker_names
        else "Unknown / Unregistered Worker"
    )

    body = f"""Safety Vision AI Alert

Source: {source}
Status: {status}
Risk Score: {risk_score} / 100

Worker:
{worker_text}

AI Predictions:
{detected_text}

PPE / Safety Violations:
{violation_text}

Location:
{location_text or "Location not available"}

This message was generated automatically by Safety Vision AI.
"""

    try:
        message = EmailMessage()
        message["Subject"] = f"Safety Vision AI - {risk_level} Alert"
        message["From"] = sender
        message["To"] = receiver
        message.set_content(body)

        # Try Gmail SSL first. If that connection is unavailable, retry
        # through Gmail STARTTLS. This keeps the alert automatic without
        # requiring the user to press a Send Email button.
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
                smtp.login(sender, password)
                smtp.send_message(message)
        except Exception as ssl_exc:
            print(f"Gmail SSL send failed, retrying STARTTLS: {ssl_exc}")
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(sender, password)
                smtp.send_message(message)

        print(f"Email alert sent: {risk_level} ({source})")
        _write_email_status(True, source=source)
        return True

    except Exception as exc:
        print(f"Email alert failed: {exc}")
        _write_email_status(False, source=source, error=str(exc))
        return False


def trigger_live_email_alert(
    risk_level,
    risk_score,
    violations,
    detected_classes,
    worker_names,
):
    """Send live-camera email on a safety-state change, with a short cooldown."""
    global _last_email_alert_time, _last_email_state

    state = (
        risk_level,
        tuple(sorted(set(violations or []))),
        tuple(sorted(set(worker_names or []))),
    )
    current_time = time.time()

    with _email_alert_lock:
        if state == _last_email_state and current_time - _last_email_alert_time < EMAIL_ALERT_COOLDOWN:
            return
        if current_time - _last_email_alert_time < EMAIL_ALERT_COOLDOWN:
            return

    # Send in the background so the live camera does not freeze.
    # Update the remembered state ONLY after a successful send; if Gmail
    # fails, the next live analysis is allowed to retry automatically.
    def _send_live_email():
        global _last_email_alert_time, _last_email_state
        ok = send_email_alert(
            risk_level=risk_level,
            risk_score=risk_score,
            violations=violations,
            detected_classes=detected_classes,
            worker_names=worker_names,
            source="Live Camera",
        )
        if ok:
            with _email_alert_lock:
                _last_email_alert_time = time.time()
                _last_email_state = state

    threading.Thread(target=_send_live_email, daemon=True).start()


def _upload_content_hash(uploaded_file):
    try:
        return hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    except Exception:
        return ""


EMAIL_STATUS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".email_alert_status.json"
)

def _write_email_status(sent, source="Safety Detection", error=""):
    try:
        payload = {
            "sent": bool(sent),
            "source": source,
            "error": str(error or ""),
            "timestamp": time.time(),
        }
        tmp = EMAIL_STATUS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, EMAIL_STATUS_FILE)
    except Exception:
        pass

def _read_email_status():
    try:
        with open(EMAIL_STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = r".\runs\detect\outputs\factory_safety-8\weights\best.pt"

model = YOLO(MODEL_PATH)


# =========================================================
# PPE CLASSES
# =========================================================

PPE_CLASSES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles"
]


# =========================================================
# DIFFERENT COLORS FOR EACH CLASS
# OpenCV uses BGR
# =========================================================

CLASS_COLORS = {

    # Person
    "person": (255, 0, 0),

    # PPE
    "helmet": (0, 255, 255),
    "gloves": (0, 255, 0),
    "vest": (255, 0, 255),
    "boots": (0, 165, 255),
    "goggles": (255, 255, 0),

    # Missing PPE
    "no_helmet": (0, 0, 255),
    "no_gloves": (0, 0, 200),
    "no_boots": (0, 100, 255),
    "no_goggle": (0, 0, 150),
    "no_vest": (0, 50, 255)
}


# =========================================================
# CLASS-SPECIFIC CONFIDENCE
# =========================================================

CLASS_CONFIDENCE = {

    # Person needs higher confidence
    "person": 0.50,

    # PPE can be smaller objects
    "helmet": 0.05,
    "gloves": 0.05,
    "vest": 0.05,
    "boots": 0.05,
    "goggles": 0.03,

    # Missing PPE
    "no_helmet": 0.10,
    "no_gloves": 0.10,
    "no_boots": 0.10,
    "no_goggle": 0.10,
    "no_vest": 0.10
}


# =========================================================
# IOU CALCULATION
# =========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union_area = (
        area1 +
        area2 -
        intersection_area
    )

    if union_area == 0:
        return 0

    return intersection_area / union_area


# =========================================================
# BOX CONTAINMENT
# Used to remove duplicate nested detections
# =========================================================

def calculate_overlap_with_smaller_box(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    smaller_area = min(
        area1,
        area2
    )

    if smaller_area == 0:
        return 0

    return intersection_area / smaller_area


# =========================================================
# REMOVE DUPLICATE DETECTIONS
# =========================================================

def remove_duplicate_detections(detections):

    final_detections = []

    class_names = list(
        set(
            d["class"]
            for d in detections
        )
    )

    for class_name in class_names:

        class_detections = [
            d
            for d in detections
            if d["class"] == class_name
        ]

        # Highest confidence first
        class_detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        selected = []

        for detection in class_detections:

            duplicate = False

            for selected_detection in selected:

                iou = calculate_iou(
                    detection["box"],
                    selected_detection["box"]
                )

                overlap_small = calculate_overlap_with_smaller_box(
                    detection["box"],
                    selected_detection["box"]
                )

                # Remove duplicate if boxes overlap heavily
                if iou > 0.40 or overlap_small > 0.70:

                    duplicate = True
                    break

            if not duplicate:

                selected.append(
                    detection
                )

        final_detections.extend(
            selected
        )

    return final_detections


# =========================================================
# REMOVE CONFLICTING PPE
# =========================================================

def remove_conflicting_detections(
    detections
):

    positive_negative_pairs = [

        ("helmet", "no_helmet"),

        ("goggles", "no_goggle"),

        ("gloves", "no_gloves"),

        ("boots", "no_boots"),

        ("vest", "no_vest")
    ]

    for positive, negative in positive_negative_pairs:

        positive_exists = any(
            d["class"] == positive
            for d in detections
        )

        if positive_exists:

            detections = [
                d
                for d in detections
                if d["class"] != negative
            ]

    return detections


# =========================================================
# PROCESS YOLO DETECTIONS
# =========================================================

def process_detections(results):

    detections = []

    for box in results[0].boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = model.names[
            class_id
        ]

        minimum_confidence = (
            CLASS_CONFIDENCE.get(
                class_name,
                0.10
            )
        )

        # Ignore weak detections
        if confidence < minimum_confidence:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )

    # -----------------------------------------------------
    # Remove duplicate boxes
    # -----------------------------------------------------

    detections = (
        remove_duplicate_detections(
            detections
        )
    )

    # -----------------------------------------------------
    # Missing PPE should only be considered
    # when at least one person is detected.
    # -----------------------------------------------------

    person_exists = any(
        d["class"] == "person"
        for d in detections
    )

    if not person_exists:

        detections = [
            d
            for d in detections
            if not d["class"].startswith("no_")
        ]

    # -----------------------------------------------------
    # Do not report missing PPE when the required body part
    # is outside the camera frame.  A cropped upper-body photo
    # cannot prove that boots/gloves are missing.
    # -----------------------------------------------------

    try:
        image_height, image_width = results[0].orig_shape[:2]
    except Exception:
        image_height, image_width = 0, 0

    if image_height and image_width:
        person_detections = [
            d for d in detections
            if d["class"] == "person"
        ]

        if person_detections:
            filtered_detections = []

            for detection in detections:
                class_name = detection["class"]

                if not class_name.startswith("no_"):
                    filtered_detections.append(detection)
                    continue

                dx1, dy1, dx2, dy2 = detection["box"]
                dcx = (dx1 + dx2) / 2
                dcy = (dy1 + dy2) / 2
                assessable = False

                for person in person_detections:
                    px1, py1, px2, py2 = person["box"]
                    person_height = max(1, py2 - py1)

                    # Missing PPE prediction must belong to a detected person.
                    if not (px1 <= dcx <= px2 and py1 <= dcy <= py2):
                        continue

                    top_visible = py1 > max(2, int(image_height * 0.02))
                    bottom_visible = py2 < image_height - max(8, int(image_height * 0.03))
                    torso_visible = person_height > max(120, int(image_height * 0.20))

                    if class_name in {"no_helmet", "no_goggle"}:
                        assessable = top_visible
                    elif class_name == "no_vest":
                        assessable = torso_visible
                    elif class_name in {"no_gloves", "no_boots"}:
                        # Hands/feet cannot be declared missing when the lower
                        # part of the person is cropped by the image boundary.
                        assessable = bottom_visible
                    else:
                        assessable = True

                    if assessable:
                        break

                if assessable:
                    filtered_detections.append(detection)

            detections = filtered_detections

    # -----------------------------------------------------
    # Remove conflicting positive/negative PPE
    # -----------------------------------------------------

    detections = (
        remove_conflicting_detections(
            detections
        )
    )

    return detections


# =========================================================
# DRAW CUSTOM COLORED DETECTIONS
# =========================================================

def draw_detections(
    image,
    detections
):

    output = image.copy()

    image_height, image_width = output.shape[:2]

    for detection in detections:

        class_name = detection[
            "class"
        ]

        confidence = detection[
            "confidence"
        ]

        x1, y1, x2, y2 = (
            detection["box"]
        )

        color = CLASS_COLORS.get(
            class_name,
            (255, 255, 255)
        )

        # -------------------------------------------------
        # Bounding Box
        # -------------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # -------------------------------------------------
        # Label
        # -------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.65

        text_thickness = 2

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness
        )

        # -------------------------------------------------
        # Keep label completely INSIDE image
        # -------------------------------------------------

        label_x = max(
            0,
            min(
                x1,
                image_width -
                text_width -
                8
            )
        )

        # Normally place label above box
        label_y = (
            y1 -
            6
        )

        # If box is near top of image,
        # place label INSIDE the box
        if label_y < text_height + 8:

            label_y = (
                y1 +
                text_height +
                8
            )

        # Keep label inside bottom boundary
        if label_y > image_height - 5:

            label_y = (
                image_height - 5
            )

        # -------------------------------------------------
        # Label background
        # -------------------------------------------------

        background_top = max(
            0,
            label_y -
            text_height -
            baseline -
            5
        )

        background_bottom = min(
            image_height,
            label_y + 5
        )

        background_right = min(
            image_width,
            label_x +
            text_width +
            8
        )

        cv2.rectangle(
            output,
            (
                label_x,
                background_top
            ),
            (
                background_right,
                background_bottom
            ),
            color,
            -1
        )

        # -------------------------------------------------
        # Label text
        # -------------------------------------------------

        cv2.putText(
            output,
            label,
            (
                label_x + 4,
                label_y
            ),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA
        )

    return output


# =========================================================
# LIVE CAMERA
# YOLO + FACE RECOGNITION
# =========================================================

class YOLOTransformer(
    VideoTransformerBase
):

    def __init__(self):

        self.frame_count = 0
        self.recognized_faces = []

    def transform(self, frame):

        # -------------------------------------------------
        # Convert camera frame
        # -------------------------------------------------

        img = frame.to_ndarray(
            format="bgr24"
        )

        # =================================================
        # YOLO PERSON + PPE DETECTION
        # =================================================

        results = model(
            img,
            conf=0.03,
            imgsz=416,
            verbose=False
        )

        # -------------------------------------------------
        # Process YOLO detections
        # -------------------------------------------------

        detections = process_detections(
            results
        )

        # =================================================
        # FACE RECOGNITION
        # Run every 5th frame to reduce lag
        # =================================================

        self.frame_count += 1

        if self.frame_count % 5 == 0:

            try:

                self.recognized_faces = recognize_faces(
                    img
                )

            except Exception as e:

                print(
                    "Face recognition error:",
                    e
                )

                self.recognized_faces = []

        recognized_faces = self.recognized_faces

        # =================================================
        # MATCH FACE WITH PERSON
        # =================================================

        for detection in detections:

            if detection["class"] != "person":
                continue

            px1, py1, px2, py2 = detection["box"]

            # Person center
            person_center_x = (
                px1 + px2
            ) // 2

            person_center_y = (
                py1 + py2
            ) // 2

            best_face = None

            best_distance = float("inf")

            for face in recognized_faces:

                fx1, fy1, fx2, fy2 = face["box"]

                # Face center
                face_center_x = (
                    fx1 + fx2
                ) // 2

                face_center_y = (
                    fy1 + fy2
                ) // 2

                # -------------------------------------------------
                # Check whether face is inside person box
                # -------------------------------------------------

                face_inside_person = (

                    px1 <= face_center_x <= px2

                    and

                    py1 <= face_center_y <= py2

                )

                if not face_inside_person:
                    continue

                # -------------------------------------------------
                # Distance between face and person center
                # -------------------------------------------------

                distance = (

                    (
                        face_center_x
                        -
                        person_center_x
                    ) ** 2

                    +

                    (
                        face_center_y
                        -
                        person_center_y
                    ) ** 2

                )

                if distance < best_distance:

                    best_distance = distance

                    best_face = face

            # =================================================
            # ASSIGN WORKER NAME
            # =================================================

            if best_face is not None:

                detection["display_name"] = (
                    best_face["name"]
                )

                detection["face_confidence"] = (
                    best_face["confidence"]
                )

            else:

                detection["display_name"] = (
                    "Unknown Worker"
                )

                detection["face_confidence"] = 0.0

        # =================================================
        # DRAW LIVE CAMERA
        # =================================================

        annotated_frame = img.copy()

        image_height, image_width = (
            annotated_frame.shape[:2]
        )

                # =================================================
        # DRAW EACH DETECTION
        # =================================================

        for detection in detections:

            class_name = detection["class"]

            confidence = detection["confidence"]

            x1, y1, x2, y2 = detection["box"]

            # =================================================
            # PERSON → WORKER NAME
            # =================================================

            if class_name == "person":

                display_name = detection.get(
                    "display_name",
                    "Unknown Worker"
                )

                face_confidence = detection.get(
                    "face_confidence",
                    0.0
                )

                worker_color = (
                    255,
                    0,
                    255
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    worker_color,
                    3
                )

                if display_name != "Unknown Worker":

                    label = (
                        f"{display_name} "
                        f"{face_confidence:.2f}"
                    )

                else:

                    label = "Unknown Worker"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # =================================================
            # PPE + VIOLATION CLASSES
            # =================================================

            else:

                if class_name == "helmet":

                  color = (0, 255, 0)       # Green

                elif class_name == "gloves":

                  color = (255, 0, 0)       # Blue

                elif class_name == "vest":

                 color = (0, 255, 255)     # Yellow

                elif class_name == "boots":

                 color = (0, 165, 255)     # Orange

                elif class_name == "goggles":

                 color = (255, 255, 0)     # Cyan

                elif class_name == "no_helmet":

                 color = (0, 0, 255)       # Red

                elif class_name == "no_gloves":

                 color = (0, 0, 255)       # yellow

                elif class_name == "no_vest":

                 color = (0, 0, 255)       # blue

                elif class_name == "no_boots":

                 color = (0, 0, 255)       # orange

                elif class_name == "no_goggles":

                  color = (0, 0, 255)       # purple

                else:

                  color = (255, 255, 255)   # White

                # -------------------------------------------------
                # PPE bounding box
                # -------------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # -------------------------------------------------
                # PPE label
                # -------------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        x1,
                        max(
                            20,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA
                )
                # =================================================
        # RISK CALCULATION
        # =================================================

        detected_classes = []

        for detection in detections:

            detected_classes.append(
                detection["class"]
            )

        risk_score, risk_level, violations = (
            calculate_risk(
                detected_classes
            )
        )

        # Publish the latest result for the live right-side analysis panel.
        worker_names = list(dict.fromkeys(
            d.get("display_name")
            for d in detections
            if d.get("class") == "person" and d.get("display_name")
        ))
        with LIVE_ANALYSIS_LOCK:
            LIVE_ANALYSIS.update({
                "risk_score": risk_score,
                "risk_level": risk_level,
                "violations": list(violations),
                "detected_classes": list(dict.fromkeys(detected_classes)),
                "worker_names": worker_names,
            })

        # =================================================
        # DISPLAY RISK
        # =================================================

        risk_text = (
            f"Risk: {risk_level} "
            f"| Score: {risk_score}"
        )

        cv2.putText(
            annotated_frame,
            risk_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255) if risk_score >= 20 else (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # =================================================
        # VOICE ALERT
        # =================================================

        if violations:

            alert_message = (
                "Warning. "
                + ", ".join(violations)
            )

            threading.Thread(
                target=speak,
                args=(alert_message,),
                daemon=True
            ).start()

        # =================================================
        # UPDATE LIVE STREAM ANALYSIS PANEL
        # WebRTC runs this transformer in a separate camera thread, so
        # the Streamlit UI must receive the latest risk data through the
        # shared LIVE_ANALYSIS state.
        # =================================================

        worker_names = []
        for detection in detections:
            name = detection.get("display_name")
            if name and name != "Unknown Worker" and name not in worker_names:
                worker_names.append(name)

        unique_classes = list(dict.fromkeys(detected_classes))

        live_data = {
            "risk_score": int(risk_score),
            "risk_level": risk_level,
            "violations": list(violations),
            "detected_classes": unique_classes,
            "worker_names": worker_names,
        }

        # Send an email whenever the live safety state changes.
        # This prevents one email per video frame while still notifying
        # the recipient for SAFE/HIGH/MEDIUM transitions and violation changes.
        trigger_live_email_alert(
            risk_level,
            risk_score,
            violations,
            unique_classes,
            worker_names,
        )

        with LIVE_ANALYSIS_LOCK:
            LIVE_ANALYSIS.update(live_data)
        _write_live_analysis_file(live_data)

        # =================================================
        # RETURN FRAME
        # =================================================

        return annotated_frame
        # =================================================
        # RETURN FRAME
        # =================================================

        return annotated_frame
        # =================================================
        # MATCH FACE WITH PERSON
        # =================================================

        for detection in detections:

            if detection["class"] != "person":
                continue

            px1, py1, px2, py2 = (
                detection["box"]
            )

            # Person center
            person_center_x = (
                px1 + px2
            ) // 2

            person_center_y = (
                py1 + py2
            ) // 2

            best_face = None

            best_distance = float(
                "inf"
            )

            for face in recognized_faces:

                fx1, fy1, fx2, fy2 = (
                    face["box"]
                )

                # Face center
                face_center_x = (
                    fx1 + fx2
                ) // 2

                face_center_y = (
                    fy1 + fy2
                ) // 2

                # -------------------------------------------------
                # Check whether face center is inside person box
                # -------------------------------------------------

                face_inside_person = (

                    px1 <= face_center_x <= px2

                    and

                    py1 <= face_center_y <= py2

                )

                if not face_inside_person:

                    continue

                # -------------------------------------------------
                # Distance between face and person center
                # -------------------------------------------------

                distance = (

                    (
                        face_center_x
                        -
                        person_center_x
                    ) ** 2

                    +

                    (
                        face_center_y
                        -
                        person_center_y
                    ) ** 2

                )

                if distance < best_distance:

                    best_distance = distance

                    best_face = face

            # =================================================
            # ASSIGN WORKER NAME
            # =================================================

            if best_face is not None:

                detection[
                    "display_name"
                ] = best_face["name"]

                detection[
                    "face_confidence"
                ] = best_face[
                    "confidence"
                ]

            else:

                detection[
                    "display_name"
                ] = "Unknown Worker"

                detection[
                    "face_confidence"
                ] = 0.0

        # =================================================
        # DRAW LIVE CAMERA
        # =================================================

        annotated_frame = img.copy()

        image_height, image_width = (
            annotated_frame.shape[:2]
        )

        # =================================================
        # DRAW EACH DETECTION
        # =================================================

        for detection in detections:

            class_name = detection[
                "class"
            ]

            confidence = detection[
                "confidence"
            ]

            x1, y1, x2, y2 = (
                detection["box"]
            )

            # =================================================
            # PERSON → WORKER NAME
            # =================================================

            if class_name == "person":

                display_name = detection.get(
                    "display_name",
                    "Unknown Worker"
                )

                face_confidence = detection.get(
                    "face_confidence",
                    0.0
                )

                # -------------------------------------------------
                # Worker box color
                # -------------------------------------------------

                worker_color = (
                    255,
                    0,
                    255
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    worker_color,
                    3
                )

                # -------------------------------------------------
                # Worker name
                # -------------------------------------------------

                if display_name != "Unknown Worker":

                    label = (
                        f"{display_name} "
                        f"{face_confidence:.2f}"
                    )

                else:

                    label = (
                        "Unknown Worker"
                    )

                font = (
                    cv2.FONT_HERSHEY_SIMPLEX
                )

                font_scale = 0.70

                text_thickness = 2

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    text_thickness
                )

                # -------------------------------------------------
                # Label position
                # -------------------------------------------------

                label_x = max(
                    0,
                    min(
                        x1,
                        image_width -
                        text_width -
                        8
                    )
                )

                label_y = (
                    y1 - 8
                )

                if label_y < text_height + 8:

                    label_y = (
                        y1 +
                        text_height +
                        8
                    )

                if label_y > image_height - 5:

                    label_y = (
                        image_height - 5
                    )

                # -------------------------------------------------
                # Label background
                # -------------------------------------------------

                background_top = max(
                    0,
                    label_y -
                    text_height -
                    baseline -
                    5
                )

                background_bottom = min(
                    image_height,
                    label_y + 5
                )

                background_right = min(
                    image_width,
                    label_x +
                    text_width +
                    8
                )

                cv2.rectangle(
                    annotated_frame,
                    (
                        label_x,
                        background_top
                    ),
                    (
                        background_right,
                        background_bottom
                    ),
                    worker_color,
                    -1
                )

                # -------------------------------------------------
                # Worker name text
                # -------------------------------------------------

                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        label_x + 4,
                        label_y
                    ),
                    font,
                    font_scale,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA
                )

            # =================================================
            # PPE / MISSING PPE
            # =================================================

            else:

                color = CLASS_COLORS.get(
                    class_name,
                    (255, 255, 255)
                )

                # -------------------------------------------------
                # Bounding box
                # -------------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                # -------------------------------------------------
                # PPE label
                # -------------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                font = (
                    cv2.FONT_HERSHEY_SIMPLEX
                )

                font_scale = 0.65

                text_thickness = 2

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    text_thickness
                )

                label_x = max(
                    0,
                    min(
                        x1,
                        image_width -
                        text_width -
                        8
                    )
                )

                label_y = (
                    y1 - 6
                )

                if label_y < text_height + 8:

                    label_y = (
                        y1 +
                        text_height +
                        8
                    )

                if label_y > image_height - 5:

                    label_y = (
                        image_height - 5
                    )

                background_top = max(
                    0,
                    label_y -
                    text_height -
                    baseline -
                    5
                )

                background_bottom = min(
                    image_height,
                    label_y + 5
                )

                background_right = min(
                    image_width,
                    label_x +
                    text_width +
                    8
                )

                cv2.rectangle(
                    annotated_frame,
                    (
                        label_x,
                        background_top
                    ),
                    (
                        background_right,
                        background_bottom
                    ),
                    color,
                    -1
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        label_x + 4,
                        label_y
                    ),
                    font,
                    font_scale,
                    (0, 0, 0),
                    text_thickness,
                    cv2.LINE_AA
                )

        # =================================================
        # RISK CALCULATION
        # =================================================

        detected_classes = [
            d["class"]
            for d in detections
        ]

        (
            risk_score,
            risk_level,
            violations
        ) = calculate_risk(
            detected_classes
        )

        # =================================================
        # RISK COLOR
        # =================================================

        if risk_level == "HIGH RISK":

            risk_color = (
                0,
                0,
                255
            )

        elif risk_level == "MEDIUM RISK":

            risk_color = (
                0,
                165,
                255
            )

        else:

            risk_color = (
                0,
                200,
                0
            )

        # =================================================
        # RISK SCORE
        # =================================================

        cv2.putText(
            annotated_frame,
            f"Risk Score: {risk_score}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # =================================================
        # RISK STATUS
        # =================================================

        cv2.putText(
            annotated_frame,
            f"Status: {risk_level}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            risk_color,
            2,
            cv2.LINE_AA
        )

        # =================================================
        # VOICE ALERT
        # =================================================

        trigger_voice_alert(
            violations
        )

        # =================================================
        # VIOLATIONS
        # =================================================

        y_position = 105

        for violation in violations:

            cv2.putText(
                annotated_frame,
                f"WARNING: {violation}",
                (
                    20,
                    y_position
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            y_position += 28

        # =================================================
        # RETURN FRAME
        # =================================================

        return annotated_frame


# =========================================================

try:
    from streamlit_geolocation import streamlit_geolocation
    GEOLOCATION_AVAILABLE = True
except Exception:
    GEOLOCATION_AVAILABLE = False

def _reverse_geocode_place(lat, lon):
    """Convert browser latitude/longitude into a readable place name; never expose raw coordinates."""
    cache = st.session_state.setdefault("location_place_cache", {})
    cache_key = (round(float(lat), 4), round(float(lon), 4))
    if cache_key in cache:
        return cache[cache_key]

    # Primary: OpenStreetMap Nominatim reverse geocoding.
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "zoom": 14,
                "addressdetails": 1,
            },
            headers={"User-Agent": "SafetyVisionAI/1.0"},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {})

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
        )
        state = address.get("state")
        country = address.get("country")
        parts = [part for part in (city, state, country) if part]
        place = ", ".join(parts) if parts else data.get("display_name")
        if place:
            cache[cache_key] = place
            return place
    except Exception:
        pass

    # Fallback: BigDataCloud's free client reverse-geocoder (no API key).
    try:
        response = requests.get(
            "https://api.bigdatacloud.net/data/reverse-geocode-client",
            params={"latitude": lat, "longitude": lon, "localityLanguage": "en"},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        city = (
            data.get("city")
            or data.get("locality")
            or data.get("principalSubdivision")
        )
        state = data.get("principalSubdivision")
        country = data.get("countryName")
        parts = [part for part in (city, state, country) if part]
        place = ", ".join(dict.fromkeys(parts)) if parts else None
        if place:
            cache[cache_key] = place
            return place
    except Exception:
        pass

    # Project-specific fallback for the Kakinada area.
    # This keeps the UI human-readable even if reverse-geocoding services are unavailable.
    if 16.70 <= lat <= 17.00 and 82.10 <= lon <= 82.40:
        return "Kakinada, Andhra Pradesh, India"

    # Never fall back to displaying latitude/longitude in the UI.
    return "Place name unavailable"

def show_location(risk_level=None, risk_score=0, violations=None, detected_classes=None, worker_names=None, source="Location Alert"):
    if not GEOLOCATION_AVAILABLE:
        st.warning("📍 Location service is not installed. Run: python -m pip install streamlit-geolocation")
        return None

    try:
        location = streamlit_geolocation()
    except Exception as exc:
        st.error(f"📍 Could not access browser location: {exc}")
        return None

    if not location:
        st.info("📍 Click Allow when the browser asks for location permission.")
        return None

    lat = location.get("latitude")
    lon = location.get("longitude")
    if lat is None or lon is None:
        st.info("📍 Waiting for location permission...")
        return None

    place = _reverse_geocode_place(lat, lon)
    st.success(f"📍 Location: {place}")
    st.markdown(
        f"[🗺️ Open {place} in Google Maps](https://www.google.com/maps/search/?api=1&query={lat},{lon})"
    )

    # Once the browser grants location, immediately send one email for the
    # current safety result. The hash prevents duplicate emails on reruns.
    if risk_level is not None and email_is_configured():
        location_key = hashlib.sha256(
            f"{source}|{round(float(lat),4)}|{round(float(lon),4)}|{risk_level}|{risk_score}".encode("utf-8")
        ).hexdigest()
        sent_key = f"location_email_sent_{source}"
        if st.session_state.get(sent_key) != location_key:
            ok = send_email_alert(
                risk_level=risk_level,
                risk_score=risk_score,
                violations=violations,
                detected_classes=detected_classes,
                worker_names=worker_names,
                source=source,
                location_text=place,
            )
            if ok:
                st.session_state[sent_key] = location_key
                st.success("📧 EMAIL ALERT SENT SUCCESSFULLY")
            else:
                status = _read_email_status()
                error_text = status.get("error") or "Unknown SMTP error"
                st.error(f"❌ EMAIL ALERT FAILED: {error_text}")
        else:
            status = _read_email_status()
            if status.get("sent") and status.get("source") == source:
                st.success("📧 EMAIL ALERT SENT SUCCESSFULLY")

    return {"latitude": lat, "longitude": lon, "place": place}

# =========================================================
# STREAMLIT PAGE / SAFETY VISION AI NAVIGATION
# =========================================================

st.set_page_config(page_title="Safety Vision AI", page_icon="🦺", layout="wide")

# =========================================================
# GLOBAL SAFETY VISION AI BACKGROUND
# Applies to HOME, REGISTRATION, DETECTION, LIVE, IMAGE and VIDEO pages
# =========================================================
st.html("""
<style>
/* =========================================================
   SAFETY VISION AI - GLOBAL INDUSTRIAL BACKGROUND
   Applied to the actual Streamlit document, not only cards.
   ========================================================= */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section[data-testid="stSidebar"] {
    background:transparent !important;
}

html, body {
    min-height:100%;
}

.stApp {
    min-height:100vh !important;
    position:relative !important;
    background-image:linear-gradient(rgba(237,245,250,.80),rgba(237,245,250,.80)),url("https://kevrongroup.com/assets/img/courses/eosh/eosh_industrial_safety_1775030171785.png") !important;
    background-size:cover !important;
    background-position:center center !important;
    background-repeat:no-repeat !important;
    background-attachment:fixed !important;
}

/* Keep the Streamlit content above the photo. */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background:transparent !important;
    position:relative !important;
    z-index:1 !important;
}

/* Keep the same factory photo visible behind all main pages. */
.stApp [data-testid="stMain"],
.stApp [data-testid="stMainBlockContainer"],
.stApp [data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* Keep Streamlit's content transparent */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: transparent !important;
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] .main {
    background: transparent !important;
}

/* Make all page text readable on dark background */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp span, .stApp li {
    color: #f8fafc;
}

/* Keep app content above the background */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"] {
    position: relative;
    z-index: 1;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0) !important;
}

/* Make the central content transparent so the global background remains visible */
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

/* Streamlit text on dark background */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp span, .stApp li {
    color: #f1f5f9;
}

/* Existing white cards become glass panels */
.home-description, .home-team, .home-guide,
.mode-card, .live-camera-card, .live-risk-card,
.partner-logo-card {
    background: rgba(248, 250, 252, 0.96) !important;
    color: #172033 !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.home-description *, .home-team *, .home-guide *,
.mode-card *, .live-camera-card *, .live-risk-card *,
.partner-logo-card * {
    color: inherit;
}

/* Inputs / uploaders */
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"],
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.96) !important;
    border-radius: 12px !important;
}

/* Buttons */
div.stButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(56,189,248,0.45) !important;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 20px rgba(14,165,233,0.25);
}

div.stButton > button:hover {
    border-color: #67e8f9 !important;
    transform: translateY(-1px);
}

/* Metrics / alerts */
[data-testid="stMetric"], [data-testid="stAlert"] {
    border-radius: 14px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071522, #0b2033) !important;
}


</style>
""")

# Visible factory-worker background. Uses CSS background so no broken <img> icon can appear.
st.markdown("""
<style>
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{
    position:relative !important;
    z-index:1 !important;
    background:transparent !important;
}
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "home"
if "registered_worker" not in st.session_state:
    st.session_state.registered_worker = ""
if "registration_camera_started" not in st.session_state:
    st.session_state.registration_camera_started = False

st.markdown("""
<style>
.page-title {font-size:2rem;font-weight:800;margin:0.4rem 0 0.8rem;}
.home-shell {width:100%;max-width:1280px;margin:0 auto;padding:8px 22px 0;box-sizing:border-box;overflow:visible;}
.home-project-title {
    display:flex !important;
    justify-content:center !important;
    align-items:baseline !important;
    flex-wrap:nowrap !important;
    gap:10px !important;
    width:100% !important;
    text-align:center;
    font-size:2.65rem;
    font-weight:950;
    line-height:1.05;
    margin:0;
    color:#073b63 !important;
    text-shadow:0 2px 8px rgba(255,255,255,.98),0 0 5px rgba(255,255,255,.95);
}
.home-realtime {
    display:inline-block !important;
    flex:0 0 auto !important;
    font-size:0.92rem;
    font-weight:750;
    color:#155e75 !important;
    margin:0 !important;
    white-space:nowrap !important;
    text-shadow:0 2px 7px rgba(255,255,255,.98),0 0 4px rgba(255,255,255,.95);
}
@media (max-width: 760px) {
    .home-project-title { flex-wrap:wrap !important; }
    .home-realtime { white-space:nowrap !important; }
}

.home-aicw {
    text-align:center;
    font-size:3.25rem;
    font-weight:950;
    color:#c026d3 !important;
    line-height:1.02;
    margin-top:8px;
    text-shadow:0 2px 8px rgba(255,255,255,.98),0 0 5px rgba(255,255,255,.95);
}
.home-subtitle {text-align:center;font-size:1.08rem;color:#64748b;margin:5px 0 13px;}
.partner-logo-row {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;width:100%;margin:0 auto 15px;}
.partner-logo-card {height:82px;border:1px solid #dbe3ef;border-radius:16px;background:rgba(255,255,255,.86);display:flex;align-items:center;justify-content:center;padding:8px 16px;box-sizing:border-box;}
.partner-logo-card img {max-width:100%;max-height:62px;width:auto;height:auto;object-fit:contain;display:block;}
.sap-logo-fallback {width:112px;height:54px;border-radius:4px;background:#0878b9;color:#fff;display:flex;align-items:center;justify-content:center;font-size:2.15rem;font-weight:950;letter-spacing:-2px;font-family:Arial,sans-serif;}
.home-info-grid {display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:18px;align-items:stretch;width:100%;margin:0 auto 13px;}
.home-description,.home-team,.home-guide {min-width:0;border:1px solid #dbe3ef;border-radius:16px;background:rgba(255,255,255,.84);padding:18px 20px;font-size:1.05rem;line-height:1.55;box-sizing:border-box;}
.home-description-title,.home-team-title {font-size:1.22rem;font-weight:900;margin-bottom:7px;color:#172554;}
.home-team div + div {margin-top:11px;}
@media (max-width: 760px) { .home-shell {padding:6px 12px 0;} .home-project-title {font-size:2rem;} .home-realtime {font-size:.82rem;margin-left:5px;white-space:normal;} .home-aicw {font-size:2.3rem;} .home-subtitle {font-size:.95rem;margin-bottom:9px;} .partner-logo-row {grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:10px;} .partner-logo-card {height:60px;padding:5px;} .partner-logo-card img {max-height:44px;} .sap-logo-fallback {width:82px;height:40px;font-size:1.6rem;} .home-info-grid {grid-template-columns:1fr;gap:8px;} .home-description,.home-team,.home-guide {font-size:.92rem;padding:12px 14px;} .home-description-title,.home-team-title {font-size:1.05rem;} }
.mode-card {padding:24px 18px;border:1px solid #dfe4ec;border-radius:20px;background:linear-gradient(145deg,#ffffff,#f6f8fb);text-align:center;min-height:190px;}
.mode-icon {font-size:3rem;margin-bottom:8px;}
.video-stream-container video {width:100% !important; max-width:620px !important; aspect-ratio:16/9 !important; object-fit:contain !important; margin:auto !important;}
.live-camera-card {padding:14px 16px;border:1px solid #e5e7eb;border-radius:18px;background:#f8fafc;}
.live-risk-card {padding:18px;border:1px solid #e5e7eb;border-radius:18px;background:#ffffff;min-height:420px;}
.live-risk-title {font-size:1.25rem;font-weight:800;margin-bottom:10px;}
.live-risk-score {font-size:2.4rem;font-weight:900;line-height:1.1;}
.live-risk-wait {color:#6b7280;font-size:1rem;}
.live-violation {padding:8px 10px;border-radius:10px;background:#fff1f2;margin:6px 0;color:#991b1b;font-weight:600;}
.live-prediction {padding:6px 10px;border-radius:9px;background:#f3f4f6;margin:4px 0;}

/* Home text readability over the photograph */
.home-shell, .home-shell * {
    text-shadow:0 2px 6px rgba(255,255,255,.92);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# READABLE TYPOGRAPHY OVER FACTORY BACKGROUND
# =========================================================
st.markdown(r"""
<style>
/* Page headings: dark navy + bright outline so they stay readable over the photo. */
.page-title {
    color:#082f49 !important;
    font-family:"Segoe UI", Arial, sans-serif !important;
    font-size:2.15rem !important;
    font-weight:900 !important;
    letter-spacing:-0.3px !important;
    line-height:1.15 !important;
    text-shadow:0 2px 0 rgba(255,255,255,.98), 0 3px 10px rgba(255,255,255,.88) !important;
    margin:0.35rem 0 0.45rem !important;
}

/* Captions/subtitles: no more white text disappearing into the background. */
.stApp [data-testid="stCaptionContainer"] {
    color:#334155 !important;
    font-family:"Segoe UI", Arial, sans-serif !important;
    font-size:1rem !important;
    font-weight:650 !important;
    line-height:1.45 !important;
    background:rgba(255,255,255,.82) !important;
    border-radius:10px !important;
    padding:5px 11px !important;
    width:fit-content !important;
    box-shadow:0 2px 8px rgba(15,23,42,.10) !important;
    text-shadow:none !important;
}

/* Streamlit form labels: dark, bold and crisp. */
.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] div,
.stApp [data-testid="stTextInput"] label,
.stApp [data-testid="stFileUploader"] label {
    color:#0f172a !important;
    font-family:"Segoe UI", Arial, sans-serif !important;
    font-weight:750 !important;
    text-shadow:0 1px 0 rgba(255,255,255,.95) !important;
}

/* Worker camera/info card. */
.live-camera-card {
    color:#0f172a !important;
    background:rgba(255,255,255,.94) !important;
    font-family:"Segoe UI", Arial, sans-serif !important;
}
.live-camera-card b,
.live-camera-card span {
    color:#0f172a !important;
    text-shadow:none !important;
}
.live-camera-card span {
    color:#475569 !important;
    font-weight:600 !important;
}

/* Camera-ready panel text. */
.stApp div[style*="border:1px dashed"] {
    color:#0f172a !important;
}
.stApp div[style*="border:1px dashed"] div {
    color:#0f172a !important;
    text-shadow:none !important;
}
.stApp div[style*="border:1px dashed"] div[style*="color:#64748b"] {
    color:#475569 !important;
}

/* Text input + uploader: readable against the light controls. */
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea,
.stApp [data-baseweb="select"] *,
.stApp [data-testid="stFileUploader"] * {
    color:#0f172a !important;
    font-family:"Segoe UI", Arial, sans-serif !important;
}
.stApp [data-baseweb="input"] input::placeholder,
.stApp [data-baseweb="textarea"] textarea::placeholder {
    color:#64748b !important;
    opacity:1 !important;
}
.stApp [data-testid="stFileUploader"] section,
.stApp [data-testid="stFileUploader"] section > div {
    color:#334155 !important;
    background:rgba(255,255,255,.94) !important;
}
.stApp [data-testid="stFileUploader"] small {
    color:#64748b !important;
}

/* Camera widget text, while keeping action buttons blue/white. */
.stApp [data-testid="stCameraInput"] label,
.stApp [data-testid="stCameraInput"] p,
.stApp [data-testid="stCameraInput"] span {
    color:#0f172a !important;
    -webkit-text-fill-color:#0f172a !important;
    text-shadow:none !important;
    font-weight:800 !important;
}

/* ===== CAMERA CAPTURE BUTTON — ALWAYS CLEAR ===== */
.stApp [data-testid="stCameraInput"] button {
    background:#0284c7 !important;
    border:2px solid #0369a1 !important;
    border-radius:10px !important;
    min-height:46px !important;
    min-width:150px !important;
    padding:8px 22px !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    font-size:16px !important;
    font-weight:900 !important;
    opacity:1 !important;
    box-shadow:0 2px 7px rgba(3,105,161,.28) !important;
}
.stApp [data-testid="stCameraInput"] button *,
.stApp [data-testid="stCameraInput"] button span {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    font-size:16px !important;
    font-weight:900 !important;
    opacity:1 !important;
}
.stApp [data-testid="stCameraInput"] button:hover {
    background:#0369a1 !important;
}

/* ===== HIGH-CONTRAST FONT FIX FOR ALL 3 MAIN PAGES ===== */
/* Strong navy text works against the bright factory photo, with a soft white edge. */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] b {
    color:#071a2f !important;
    font-family:"Segoe UI", Arial, sans-serif !important;
    text-shadow:0 1px 2px rgba(255,255,255,.96), 0 0 5px rgba(255,255,255,.78) !important;
}

/* Page 1 — Home */
.home-subtitle {
    color:#123a5a !important;
    font-weight:750 !important;
    text-shadow:0 1px 3px rgba(255,255,255,.98) !important;
}
.home-description, .home-team, .home-guide {
    color:#071a2f !important;
}
.home-description *, .home-team *, .home-guide * {
    color:#071a2f !important;
}
.home-description-title, .home-team-title {
    color:#062c4c !important;
    text-shadow:0 1px 3px rgba(255,255,255,.98) !important;
}

/* Page 2 — Worker Registration */
.stApp .page-title {
    color:#062c4c !important;
    text-shadow:0 2px 3px rgba(255,255,255,.98), 0 0 7px rgba(255,255,255,.92) !important;
}
.stApp [data-testid="stCaptionContainer"] * {
    color:#17324d !important;
}
.stApp .live-camera-card,
.stApp .live-camera-card * {
    color:#071a2f !important;
}
.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stTextInput"] label *,
.stApp [data-testid="stFileUploader"] label * {
    color:#061b31 !important;
    font-weight:800 !important;
    text-shadow:0 1px 2px rgba(255,255,255,.98) !important;
}

/* Page 3 — Detection mode + all detection screens */
.stApp .mode-card,
.stApp .mode-card h3,
.stApp .mode-card p,
.stApp .mode-card div {
    color:#071a2f !important;
    text-shadow:0 1px 2px rgba(255,255,255,.98) !important;
}
.stApp .live-risk-card,
.stApp .live-risk-card * {
    color:#071a2f !important;
}
.stApp .live-risk-title {
    color:#062c4c !important;
    text-shadow:0 1px 2px rgba(255,255,255,.98) !important;
}
.stApp .live-risk-wait {
    color:#334155 !important;
}

/* Inputs: dark text + stronger placeholder */
.stApp input, .stApp textarea {
    color:#061b31 !important;
    -webkit-text-fill-color:#061b31 !important;
    font-weight:600 !important;
}
.stApp input::placeholder, .stApp textarea::placeholder {
    color:#475569 !important;
    -webkit-text-fill-color:#475569 !important;
    opacity:1 !important;
}

/* Keep all action-button text white and crisp. */
.stApp button,
.stApp button * {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    text-shadow:none !important;
}

/* Warnings/success/info messages stay readable. */
.stApp [data-testid="stAlert"] * {
    color:#0f172a !important;
    text-shadow:none !important;
}

/* Consistent readable font across the application. */
.stApp, .stApp * {
    font-family:"Segoe UI", Arial, sans-serif;
}

/* File uploader browse button: keep it visible against the white uploader panel. */
.stApp [data-testid="stFileUploader"] button {
    background:#e0f2fe !important;
    border:1px solid #0284c7 !important;
    color:#075985 !important;
    -webkit-text-fill-color:#075985 !important;
    font-weight:800 !important;
    text-shadow:none !important;
    border-radius:10px !important;
}
.stApp [data-testid="stFileUploader"] button * {
    color:#075985 !important;
    -webkit-text-fill-color:#075985 !important;
    font-weight:800 !important;
    text-shadow:none !important;
}
.stApp [data-testid="stFileUploader"] button:hover {
    background:#bae6fd !important;
    border-color:#0369a1 !important;
}

/* Keep buttons as white text. */
div.stButton > button,
div.stButton > button * {
    color:#ffffff !important;
    text-shadow:none !important;
    font-weight:800 !important;
}


/* ===== UPLOAD CONTROLS — HIGH VISIBILITY ON ALL 3 PAGES ===== */
.stApp [data-testid="stFileUploader"] {
    border-radius:14px !important;
}
.stApp [data-testid="stFileUploader"] section {
    background:rgba(255,255,255,.97) !important;
    border:2px solid #0ea5e9 !important;
    border-radius:14px !important;
    padding:10px !important;
}
.stApp [data-testid="stFileUploader"] section > div {
    background:transparent !important;
}
.stApp [data-testid="stFileUploader"] label,
.stApp [data-testid="stFileUploader"] label * {
    color:#062c4c !important;
    -webkit-text-fill-color:#062c4c !important;
    font-weight:900 !important;
    text-shadow:none !important;
}
.stApp [data-testid="stFileUploader"] section p,
.stApp [data-testid="stFileUploader"] section span,
.stApp [data-testid="stFileUploader"] section small {
    color:#334155 !important;
    -webkit-text-fill-color:#334155 !important;
    font-weight:650 !important;
    opacity:1 !important;
}
/* Streamlit uploader button — force exactly ONE visible label. */
.stApp [data-testid="stFileUploader"] section button {
    position:relative !important;
    background:#0284c7 !important;
    border:2px solid #0369a1 !important;
    color:transparent !important;
    -webkit-text-fill-color:transparent !important;
    font-size:0 !important;
    font-weight:900 !important;
    border-radius:10px !important;
    min-height:44px !important;
    min-width:118px !important;
    padding:8px 18px !important;
    opacity:1 !important;
    box-shadow:0 2px 6px rgba(3,105,161,.25) !important;
    overflow:hidden !important;
}
.stApp [data-testid="stFileUploader"] section button * {
    color:transparent !important;
    -webkit-text-fill-color:transparent !important;
    font-size:0 !important;
    text-shadow:none !important;
    opacity:1 !important;
}
.stApp [data-testid="stFileUploader"] section button::after {
    content:"Upload" !important;
    position:absolute !important;
    inset:0 !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    font-family:"Segoe UI", Arial, sans-serif !important;
    font-size:0.95rem !important;
    font-weight:900 !important;
    line-height:1 !important;
    text-shadow:none !important;
    pointer-events:none !important;
}
.stApp [data-testid="stFileUploader"] section button:hover {
    background:#0369a1 !important;
    border-color:#075985 !important;
}

/* Uploaded filename/status */
.stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
.stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] * {
    color:#062c4c !important;
    -webkit-text-fill-color:#062c4c !important;
    font-weight:800 !important;
}
/* Camera/upload labels on registration, image and video screens */
.stApp [data-testid="stFileUploader"] + div,
.stApp [data-testid="stFileUploader"] ~ div {
    color:#062c4c !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HOME PAGE ----------------
if st.session_state.page == "home":
    # Compact landing page: everything is designed to fit in one viewport.
    st.markdown("""
    <div class="home-shell">
        <div class="home-project-title">
            🦺 Safety Vision AI
            <span class="home-realtime">(Real-Time PPE &amp; Worker Safety Detection)</span>
        </div>
        <div class="home-aicw">AICW 2.0</div>
        <div class="home-subtitle">Artificial Intelligence Careers for Women</div>
        <div class="partner-logo-row">
            <div class="partner-logo-card"><img src="https://www.codewithharry.com/logos/microsoft.png" alt="Microsoft logo"></div>
            <div class="partner-logo-card"><img src="https://cdn.theorg.com/3f6404d8-0c4c-4ed9-a24c-6d4342e5ddb3_small.jpg" alt="Edunet Foundation logo"></div>
            <div class="partner-logo-card"><img src="https://4.bp.blogspot.com/-kPdUf5LanSg/XtvVAJqCLCI/AAAAAAAAAH4/F_uiCXWOMoY-2iAb1Eikw_uEtOIhWpHuQCK4BGAYYCw/s390/apssdc_final.png" alt="Skill AP APSSDC logo"></div>
            <div class="partner-logo-card"><div class="sap-logo-fallback" aria-label="SAP logo">SAP</div></div>
        </div>
        <div class="home-info-grid">
            <div class="home-description">
                <b>Safety Vision AI</b> is an AI-powered industrial worker safety system that detects PPE, identifies safety violations, calculates risk levels, recognizes workers when possible, and provides real-time safety alerts.
            </div>
            <div class="home-team">
                <div class="home-team-title">👥 Team</div>
                <div><b>D. Jyothsna</b> — Team Lead</div>
                <div><b>E. Sarika</b></div>
                <div><b>K. Navya</b></div>
            </div>
            <div class="home-guide">
                <div class="home-team-title">🎓 Guided by</div>
                <div><b>Abdul Aziz Md</b></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        if st.button("🚀 DETECTION", type="primary", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()

# ---------------- REGISTRATION + MODE PAGES ----------------
# =========================================================
# WORKER REGISTRATION
# =========================================================

elif st.session_state.page == "register":

    st.markdown("<div class=\"page-title\">👤 Worker Registration</div>", unsafe_allow_html=True)
    st.caption("Registration is optional. You can continue to Safety Detection without registering a new worker.")

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown("<div class=\"live-camera-card\"><b>📷 Worker Photo Camera</b><br><span style=\"color:#6b7280\">Camera is OFF until you press Start.</span></div>", unsafe_allow_html=True)

        if not st.session_state.get("registration_camera_started", False):
            st.markdown("<div style=\"text-align:center;padding:62px 20px;border:1px dashed #cbd5e1;border-radius:18px;background:#f8fafc;margin-top:12px;\"><div style=\"font-size:4rem;\">📷</div><div style=\"font-size:1.15rem;font-weight:700;margin-top:8px;\">Camera is ready</div><div style=\"color:#64748b;margin-top:5px;\">Press Start to open the camera and take a worker photo.</div></div>", unsafe_allow_html=True)
            if st.button("▶️ START", type="primary", use_container_width=True, key="start_registration_camera"):
                st.session_state.registration_camera_started = True
                st.rerun()
        else:
            st.markdown("""
            <div style="background:white;padding:12px 16px;border-radius:12px;border:2px solid #0ea5e9;margin-top:12px;margin-bottom:8px;">
                <div style="font-size:20px;font-weight:800;color:#062c4c;">📸 TAKE WORKER PHOTO</div>
                <div style="font-size:14px;color:#475569;margin-top:4px;">Position the worker clearly inside the camera and capture the photo.</div>
            </div>
            """, unsafe_allow_html=True)

            worker_camera = st.camera_input("Take Photo", key="registration_camera_photo")
            if worker_camera is not None:
                st.success("✅ Photo captured successfully! Now click Register Worker.")
            else:
                st.info("📷 Camera ready — take a clear worker photo.")

    with right:
        st.markdown("<div class=\"team-box\"><div class=\"home-team-title\">📝 Register New Worker</div></div>", unsafe_allow_html=True)
        worker_name = st.text_input("Worker Name", placeholder="Enter worker name", key="reg_worker_name")
        worker_photo = st.file_uploader("Or Upload Worker Photo", type=["jpg","jpeg","png"], key="worker_registration_photo")

        registration_photo = st.session_state.get("registration_camera_photo") if st.session_state.get("registration_camera_photo") is not None else worker_photo

        if st.button("📝 Register Worker", type="primary", use_container_width=True):
            if not worker_name.strip():
                st.warning("Please enter the worker name.")
            elif registration_photo is None:
                st.warning("Please take a worker photo with the camera or upload a worker photo.")
            else:
                with st.spinner("Registering worker..."):
                    success, message = register_worker(worker_name, registration_photo)
                if success:
                    st.session_state.registered_worker = worker_name.strip()
                    st.success(message)
                else:
                    st.error(message)

        # Keep Safety Detection directly under Register Worker on the RIGHT side.
        # This prevents the button from being pushed below the camera when the camera opens.
        if st.button("➡️ SAFETY DETECTION", type="primary", use_container_width=True, key="next_safety_detection"):
            st.session_state.page = "modes"
            st.rerun()

    # Small navigation controls stay below the camera, while Safety Detection
    # remains next to the registration controls and does not require scrolling.
    nav_left, nav_right = st.columns(2)
    with nav_left:
        if st.button("← Back to Home", use_container_width=True, key="back_register_home"):
            st.session_state.registration_camera_started = False
            st.session_state.pop("registration_camera_photo", None)
            st.session_state.page = "home"
            st.rerun()
    with nav_right:
        if st.session_state.get("registration_camera_started", False):
            if st.button("⏹ Close Camera", use_container_width=True, key="stop_registration_camera"):
                st.session_state.registration_camera_started = False
                st.session_state.pop("registration_camera_photo", None)
                st.rerun()

# =========================================================
# DETECTION MODE SELECTION
# =========================================================

elif st.session_state.page == "modes":

    st.markdown("<div class=\"page-title\">🔍 Select Detection Mode</div>", unsafe_allow_html=True)
    st.caption("Choose how you want Safety Vision AI to analyse the worker.")

    a, b, c = st.columns(3)
    with a:
        st.markdown("<div class=\"mode-card\"><div class=\"mode-icon\">📷</div><h3>Live Camera</h3><p>Real-time PPE and safety risk detection.</p></div>", unsafe_allow_html=True)
        if st.button("Open Live Camera", key="mode_live", use_container_width=True):
            st.session_state.page = "live"
            st.rerun()
    with b:
        st.markdown("<div class=\"mode-card\"><div class=\"mode-icon\">🖼️</div><h3>Upload Image</h3><p>Analyse one worker image with predictions and risk.</p></div>", unsafe_allow_html=True)
        if st.button("Upload Image", key="mode_image", use_container_width=True):
            st.session_state.page = "image"
            st.rerun()
    with c:
        st.markdown("<div class=\"mode-card\"><div class=\"mode-icon\">🎬</div><h3>Upload Video</h3><p>Process recorded factory safety footage.</p></div>", unsafe_allow_html=True)
        if st.button("Upload Video", key="mode_video", use_container_width=True):
            st.session_state.page = "video"
            st.rerun()

    if st.button("← Back to Registration"):
        st.session_state.page = "register"
        st.rerun()

# =========================================================
# LIVE CAMERA
# =========================================================


# =========================================================

def ppe_part_is_visible(detections, image_shape, ppe_class):
    """Return True only when the relevant body region is visible enough to assess PPE."""
    try:
        image_height, image_width = image_shape[:2]
    except Exception:
        return True

    persons = [
        d for d in detections
        if d.get("class") == "person"
    ]

    if not persons:
        return False

    for person in persons:
        px1, py1, px2, py2 = person.get("box", (0, 0, 0, 0))
        person_height = max(1, py2 - py1)

        top_visible = py1 > max(2, int(image_height * 0.02))
        bottom_visible = py2 < image_height - max(8, int(image_height * 0.03))
        torso_visible = person_height > max(120, int(image_height * 0.20))

        if ppe_class in {"helmet", "goggles"} and top_visible:
            return True
        if ppe_class == "vest" and torso_visible:
            return True
        if ppe_class in {"gloves", "boots"} and bottom_visible:
            return True

    return False



if st.session_state.page == "live":

    # Reset only when the user newly enters the live page.
    if st.session_state.get("live_page_active") is not True:
        reset_live_analysis()
        st.session_state.live_page_active = True

    st.markdown("<div class=\"page-title\">📷 Live Camera Safety Detection</div>", unsafe_allow_html=True)
    st.caption("Camera is on the left; live AI risk analysis and location are shown on the right.")

    left, right = st.columns([1.25, 0.85], gap="large")

    with left:
        st.markdown("<div class=\"live-camera-card\"><b>📷 Live Camera</b><br><span style=\"color:#6b7280\">Real-time worker, PPE and safety detection</span></div>", unsafe_allow_html=True)
        webrtc_streamer(
            key="factory-safety-camera",
            video_transformer_factory=YOLOTransformer,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
        )

    # The fragment refreshes only the right-side analysis panel, so the camera
    # stream does not have to restart on every risk update.
    fragment = getattr(st, "fragment", None)

    def _render_live_risk_panel():
        # Read the latest frame result from disk. This is reliable even when
        # streamlit-webrtc runs the transformer on a different worker/thread.
        data = _read_live_analysis_file()

        st.markdown("<div class=\"live-risk-title\">⚠️ Live Risk Analysis</div>", unsafe_allow_html=True)

        if data["risk_level"] == "WAITING":
            st.markdown("<div class=\"live-risk-wait\">Start the camera to begin live AI prediction.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class=\"live-risk-score\">{data['risk_score']} / 100</div>", unsafe_allow_html=True)
            if data["risk_level"] == "HIGH RISK":
                st.error("🔴 HIGH RISK")
            elif data["risk_level"] == "MEDIUM RISK":
                st.warning("🟠 MEDIUM RISK")
            else:
                st.success("🟢 SAFE")

            st.markdown("**🤖 AI Predictions**")
            if data["detected_classes"]:
                st.write("Detected: " + ", ".join(data["detected_classes"]))
            else:
                st.info("No objects detected in the latest frame.")

            st.markdown("**🚨 PPE Not Worn / Safety Violations**")
            if data["violations"]:
                for violation in data["violations"]:
                    st.error(f"❌ {violation}")
            else:
                st.success("✅ Required PPE detected")

            st.markdown("**👤 Worker Recognition**")
            if data["worker_names"]:
                for name in data["worker_names"]:
                    st.write(f"• {name}")
            else:
                st.write("• Unknown Worker / no face match")

    with right:
        if fragment is not None:
            @st.fragment(run_every="1s")
            def live_risk_fragment():
                _render_live_risk_panel()
            live_risk_fragment()
        else:
            _render_live_risk_panel()

        st.markdown("### 📍 Live Location")
        st.caption("Click Allow in the browser permission popup. The location alert email is sent automatically once the location is available.")
        live_data_for_location = _read_live_analysis_file()
        show_location(
            risk_level=live_data_for_location.get("risk_level") or "SAFE",
            risk_score=live_data_for_location.get("risk_score", 0),
            violations=live_data_for_location.get("violations", []),
            detected_classes=live_data_for_location.get("detected_classes", []),
            worker_names=live_data_for_location.get("worker_names", []),
            source="Live Location Alert",
        )

    if st.button("← Back to Detection Modes", key="back_live"):
        st.session_state.live_page_active = False
        st.session_state.page = "modes"
        st.rerun()


# =========================================================

# UPLOAD IMAGE
# =========================================================

elif st.session_state.page == "image":

    st.header("🖼️ Upload Image")
    st.caption("Any image containing a person can be analyzed. Non-person images are rejected.")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="safety_upload_image",
    )

    if uploaded_image:
        file_bytes = uploaded_image.getvalue()
        image = cv2.imdecode(
            np.frombuffer(file_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        # ---------------------------------------------------------
        # PERSON GATE
        # ---------------------------------------------------------
        # A certificate/logo/object without a person is NOT analyzed.
        # Any detected person is analyzed for safety risk.
        # Registration is never required.
        # ---------------------------------------------------------
        gate_results = model(image, conf=0.15, imgsz=640, verbose=False)
        gate_detections = process_detections(gate_results)
        gate_classes = {d.get("class") for d in gate_detections}
        has_person = "person" in gate_classes

        if not has_person:
            st.warning("🚫 No person detected in this image.")
            st.info("Certificates, logos, documents, objects and other images without a person are not analyzed.")

            with st.expander("Why was this image rejected?", expanded=True):
                st.write("• YOLO did not detect a person in the uploaded image.")
                st.write("• Safety/PPE risk analysis requires a detected person.")

            if st.button("🔄 Choose Another Image", key="choose_another_image"):
                st.rerun()

        else:
            # FaceNet is identity-only. It never blocks safety analysis.
            try:
                recognized_faces = recognize_faces(image)
            except Exception:
                recognized_faces = []

            valid_workers = [
                face
                for face in recognized_faces
                if face.get("name")
                and str(face.get("name")).strip().lower()
                not in {"unknown", "unknown worker"}
            ]

            # -----------------------------------------------------
            # CLASSIFY THE IMAGE FOR DISPLAY PURPOSES
            # -----------------------------------------------------
            # Industrial PPE / registered worker -> full factory PPE detection.
            # Sports/swimming gear -> do NOT count sports helmet/goggles as
            # factory PPE. Still predict factory safety risk from the person.
            # Normal person -> full missing-PPE safety prediction.
            # -----------------------------------------------------
            industrial_ppe = {
                "vest", "gloves", "boots",
                "no_vest", "no_gloves", "no_boots"
            }
            has_industrial_context = bool(gate_classes.intersection(industrial_ppe)) or bool(valid_workers)
            has_nonindustrial_head_gear = bool(
                gate_classes.intersection({"helmet", "goggles"})
            )

            if has_industrial_context:
                detection_mode = "factory"
                results = model(image, conf=0.03, imgsz=640, verbose=False)
                detections = process_detections(results)
                detected_classes = [d["class"] for d in detections]

            elif has_nonindustrial_head_gear:
                # Sports/swimming/etc.: only PERSON is drawn on the image.
                # Their helmet/goggles are NOT accepted as factory PPE.
                detection_mode = "sports"
                person_detections = [
                    d for d in gate_detections
                    if d.get("class") == "person"
                ]
                detections = person_detections

                # Still calculate factory safety risk for the person.
                # Sports helmet/goggles do not satisfy factory PPE requirements.
                detected_classes = [
                    "person",
                    "no_helmet",
                    "no_gloves",
                    "no_goggle",
                    "no_vest",
                    "no_boots",
                ]

            else:
                # Normal person: run the FULL YOLO PPE model.
                # This allows the model to detect actual helmet/goggles/PPE
                # when they are present, and no_* classes when PPE is missing.
                detection_mode = "normal"
                results = model(image, conf=0.03, imgsz=640, verbose=False)
                detections = process_detections(results)
                detected_classes = [d["class"] for d in detections]

            unique_classes = list(dict.fromkeys(detected_classes))
            annotated_image = draw_detections(image, detections)

            # -----------------------------------------------------
            # INFER MISSING PPE AND SHOW IT ON THE IMAGE
            # -----------------------------------------------------
            # If a person is present but YOLO did not find either the
            # positive PPE class or its no_* class, treat that PPE as
            # missing for the safety prediction. This makes a normal
            # person photo visibly show NO HELMET / NO GOGGLES / etc.
            # just like the person label already shown by YOLO.
            required_pairs = [
                ("helmet", "no_helmet"),
                ("gloves", "no_gloves"),
                ("goggles", "no_goggle"),
                ("vest", "no_vest"),
                ("boots", "no_boots"),
            ]

            risk_classes = list(detected_classes)
            inferred_missing = []

            if detection_mode != "sports":
                for positive_class, missing_class in required_pairs:
                    if (
                        positive_class not in unique_classes
                        and missing_class not in unique_classes
                        and ppe_part_is_visible(
                            detections,
                            image.shape,
                            positive_class
                        )
                    ):
                        risk_classes.append(missing_class)
                        inferred_missing.append(missing_class)

            else:
                # Sports/swimming helmet and goggles are not factory PPE.
                # Keep the person detection, but risk is still calculated
                # against factory PPE requirements.
                inferred_missing = []
                for positive_class, missing_class in required_pairs:
                    if (
                        missing_class not in risk_classes
                        and ppe_part_is_visible(
                            detections,
                            image.shape,
                            positive_class
                        )
                    ):
                        risk_classes.append(missing_class)
                        inferred_missing.append(missing_class)

            risk_classes = list(dict.fromkeys(risk_classes))

            # Draw inferred missing-PPE labels as separate red labels on
            # the picture. No fake bounding boxes are created because the
            # model did not provide an exact missing-PPE coordinate.
            person_boxes = [
                d.get("box") for d in detections
                if d.get("class") == "person"
            ]
            if not person_boxes:
                person_boxes = [
                    d.get("box") for d in gate_detections
                    if d.get("class") == "person"
                ]

            # IMPORTANT: For normal/factory-person images, always show the
            # safety result ON THE PICTURE itself.  A no_* class may already
            # have been produced by YOLO, so do not rely only on
            # ``inferred_missing``.  Build the visible labels from the final
            # risk classes so the picture clearly shows NO HELMET, NO GOGGLES,
            # NO GLOVES, NO VEST and NO BOOTS when applicable.
            if person_boxes and detection_mode != "sports":
                px1, py1, px2, py2 = person_boxes[0]

                visible_missing = [
                    c for c in risk_classes
                    if c in {
                        "no_helmet",
                        "no_gloves",
                        "no_goggle",
                        "no_vest",
                        "no_boots",
                    }
                ]

                labels = []
                label_map = {
                    "no_helmet": "NO HELMET",
                    "no_gloves": "NO GLOVES",
                    "no_goggle": "NO GOGGLES",
                    "no_vest": "NO VEST",
                    "no_boots": "NO BOOTS",
                }
                for missing_class in visible_missing:
                    if label_map[missing_class] not in labels:
                        labels.append(label_map[missing_class])

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.58
                thickness = 2
                y = max(30, py1 + 28)

                for label in labels:
                    (tw, th), base = cv2.getTextSize(
                        label, font, scale, thickness
                    )

                    # Put the safety labels just to the right of the person
                    # box; if there is no room, put them inside the image at
                    # the left side of the person box.
                    x = px2 + 10
                    if x + tw + 12 > image.shape[1]:
                        x = max(8, px1 - tw - 12)

                    # Keep every label fully inside the image vertically.
                    if y + 8 > image.shape[0]:
                        break

                    cv2.rectangle(
                        annotated_image,
                        (x - 6, y - th - base - 6),
                        (x + tw + 6, y + 6),
                        (0, 0, 180),
                        -1,
                    )
                    cv2.putText(
                        annotated_image,
                        label,
                        (x, y),
                        font,
                        scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                    y += th + base + 14

            # EVERY PERSON gets a risk prediction.
            # Sports helmet/goggles are intentionally not counted as factory PPE.
            risk_score, risk_level, violations = calculate_risk(risk_classes)

            # Send one email for each newly uploaded image.
            image_hash = _upload_content_hash(uploaded_image)
            last_image_hash = st.session_state.get("last_image_email_hash")
            if image_hash and image_hash != last_image_hash:
                st.session_state.last_image_email_hash = image_hash
                email_ok = send_email_alert(
                    risk_level=risk_level,
                    risk_score=risk_score,
                    violations=violations,
                    detected_classes=risk_classes,
                    worker_names=[face.get("name") for face in valid_workers if face.get("name")],
                    source="Upload Image",
                )
                if email_ok:
                    st.success("📧 EMAIL ALERT SENT SUCCESSFULLY")
                else:
                    st.error("❌ EMAIL ALERT NOT SENT — check the terminal for the SMTP error.")

            left, right = st.columns([1.35, 1])

            with left:
                st.image(
                    annotated_image,
                    channels="BGR",
                    caption="Person / Factory Worker Detection",
                    use_container_width=True,
                )

            with right:
                st.subheader("🤖 AI Predictions")

                # Show the actual YOLO detections plus inferred missing PPE when
                # neither the positive PPE class nor its no_* class was detected.
                # This keeps normal-person photos useful for safety prediction while
                # still allowing sports/swimming gear to be treated as non-factory PPE.
                display_classes = list(dict.fromkeys(risk_classes))
                if "person" not in display_classes:
                    display_classes.insert(0, "person")
                if detection_mode == "sports":
                    st.info("Sports/non-industrial helmet or goggles are not counted as factory PPE.")

                for class_name in display_classes:
                    if class_name.startswith("no_"):
                        st.write(f"❌ {class_name}")
                    else:
                        st.write(f"✅ {class_name}")

                st.subheader("👤 Worker Recognition")
                if valid_workers:
                    for face in valid_workers:
                        st.success(
                            f"✅ {face['name']} — similarity: {face['confidence']:.2f}"
                        )
                else:
                    st.info("👤 Unknown / Unregistered Person")

                st.subheader("⚠️ Safety Risk Analysis")
                st.metric("Risk Score", risk_score)

                if risk_level == "HIGH RISK":
                    st.error(f"🔴 {risk_level}")
                elif risk_level == "MEDIUM RISK":
                    st.warning(f"🟠 {risk_level}")
                else:
                    st.success(f"🟢 {risk_level}")

                if violations:
                    st.write("### 🚨 PPE Not Worn")
                    for violation in violations:
                        st.write(f"❌ {violation}")
                else:
                    st.success("✅ No PPE violations detected")

                st.subheader("📍 Location")
                show_location(
                    risk_level=risk_level,
                    risk_score=risk_score,
                    violations=violations,
                    detected_classes=risk_classes,
                    worker_names=[face.get("name") for face in valid_workers if face.get("name")],
                    source="Upload Image - Location",
                )

    if st.button("← Back to Detection Modes", key="back_image"):
        st.session_state.page = "modes"
        st.rerun()

# UPLOAD VIDEO
# =========================================================

elif st.session_state.page == "video":

    st.header(
        "🎥 Upload Video"
    )

    uploaded_video = (
        st.file_uploader(
            "Choose a video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ]
        )
    )

    if uploaded_video:

        # =================================================
        # SAVE INPUT VIDEO
        # =================================================

        input_suffix = ".mp4"

        if uploaded_video.name.lower().endswith(
            ".avi"
        ):

            input_suffix = ".avi"

        elif uploaded_video.name.lower().endswith(
            ".mov"
        ):

            input_suffix = ".mov"

        elif uploaded_video.name.lower().endswith(
            ".mkv"
        ):

            input_suffix = ".mkv"

        input_temp = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=input_suffix
            )
        )

        input_temp.write(
            uploaded_video.getvalue()
        )

        input_temp.close()

        # =================================================
        # OPEN VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            input_temp.name
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open video."
            )

        else:

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:

                fps = 20

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            # =================================================
            # OUTPUT VIDEO
            # =================================================

            output_temp = (
                tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
            )

            output_path = (
                output_temp.name
            )

            output_temp.close()

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (
                    width,
                    height
                )
            )

            # =================================================
            # PROCESSING UI
            # =================================================

            st.write(
                "### 🤖 AI Video Processing"
            )

            progress_bar = (
                st.progress(0)
            )

            status_text = st.empty()

            # =================================================
            # VIDEO RISK VARIABLES
            # =================================================

            maximum_risk_score = 0

            highest_risk_level = (
                "SAFE"
            )

            all_violations = set()

            frame_count = 0

            # =================================================
            # PROCESS EACH FRAME
            # =================================================

            while True:

                ret, frame = (
                    cap.read()
                )

                if not ret:
                    break

                # ---------------------------------------------
                # YOLO
                # ---------------------------------------------

                results = model(
                    frame,
                    conf=0.03,
                    imgsz=640,
                    verbose=False
                )

                # ---------------------------------------------
                # Process detections
                # ---------------------------------------------

                detections = (
                    process_detections(
                        results
                    )
                )

                # ---------------------------------------------
                # Detected classes
                # ---------------------------------------------

                detected_classes = [
                    d["class"]
                    for d in detections
                ]

                # ---------------------------------------------
                # Risk calculation
                # ---------------------------------------------

                (
                    risk_score,
                    risk_level,
                    violations
                ) = calculate_risk(
                    detected_classes
                )

                # ---------------------------------------------
                # Store maximum risk
                # ---------------------------------------------

                if (
                    risk_score
                    > maximum_risk_score
                ):

                    maximum_risk_score = (
                        risk_score
                    )

                    highest_risk_level = (
                        risk_level
                    )

                # ---------------------------------------------
                # Store violations
                # ---------------------------------------------

                for violation in violations:

                    all_violations.add(
                        violation
                    )

                # ---------------------------------------------
                # Draw SAME COLORS as image
                # ---------------------------------------------

                annotated_frame = (
                    draw_detections(
                        frame,
                        detections
                    )
                )

                # =================================================
                # RISK DISPLAY ON VIDEO
                # =================================================

                if risk_level == "HIGH RISK":

                    risk_color = (
                        0,
                        0,
                        255
                    )

                elif risk_level == "MEDIUM RISK":

                    risk_color = (
                        0,
                        165,
                        255
                    )

                else:

                    risk_color = (
                        0,
                        200,
                        0
                    )

                # -------------------------------------------------
                # Risk score
                # -------------------------------------------------

                cv2.putText(
                    annotated_frame,
                    f"Risk Score: {risk_score}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # -------------------------------------------------
                # Risk level
                # -------------------------------------------------

                cv2.putText(
                    annotated_frame,
                    f"Status: {risk_level}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    risk_color,
                    2,
                    cv2.LINE_AA
                )

                # -------------------------------------------------
                # Violations
                # -------------------------------------------------

                y_position = 110

                for violation in violations:

                    cv2.putText(
                        annotated_frame,
                        f"WARNING: {violation}",
                        (
                            20,
                            y_position
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                    y_position += 30

                # ---------------------------------------------
                # Write processed frame
                # ---------------------------------------------

                writer.write(
                    annotated_frame
                )

                # ---------------------------------------------
                # Progress
                # ---------------------------------------------

                frame_count += 1

                if total_frames > 0:

                    progress = (
                        frame_count
                        / total_frames
                    )

                    progress_bar.progress(
                        min(
                            progress,
                            1.0
                        )
                    )

                    status_text.write(
                        f"Processing frame "
                        f"{frame_count} / "
                        f"{total_frames}"
                    )

            # =================================================
            # RELEASE
            # =================================================

            cap.release()

            writer.release()

            progress_bar.progress(
                1.0
            )

            status_text.write(
                "✅ Video processing completed."
            )

            # =================================================
            # FINAL RISK
            # =================================================

            st.subheader(
                "⚠️ Video Risk Prediction"
            )

            st.metric(
                "Maximum Risk Score",
                maximum_risk_score
            )

            if (
                highest_risk_level
                == "HIGH RISK"
            ):

                st.error(
                    f"🔴 {highest_risk_level}"
                )

            elif (
                highest_risk_level
                == "MEDIUM RISK"
            ):

                st.warning(
                    f"🟠 {highest_risk_level}"
                )

            else:

                st.success(
                    f"🟢 {highest_risk_level}"
                )

            # =================================================
            # VIDEO VIOLATIONS
            # =================================================

            if all_violations:

                st.write(
                    "### 🚨 Safety Violations Found"
                )

                for violation in sorted(
                    all_violations
                ):

                    st.write(
                        f"❌ {violation}"
                    )

            else:

                st.success(
                    "✅ No PPE violations detected in the video."
                )

            # Send one email for each newly uploaded video.
            video_hash = _upload_content_hash(uploaded_video)
            last_video_hash = st.session_state.get("last_video_email_hash")
            if video_hash and video_hash != last_video_hash:
                st.session_state.last_video_email_hash = video_hash
                email_ok = send_email_alert(
                    risk_level=highest_risk_level,
                    risk_score=maximum_risk_score,
                    violations=sorted(all_violations),
                    detected_classes=[],
                    worker_names=[],
                    source="Upload Video",
                )
                if email_ok:
                    st.success("📧 EMAIL ALERT SENT SUCCESSFULLY")
                else:
                    st.error("❌ EMAIL ALERT NOT SENT — check the terminal for the SMTP error.")

            # =================================================
            # SHOW PROCESSED VIDEO
            # =================================================

            st.subheader(
                "🎬 Processed Safety Video"
            )

            with open(
                output_path,
                "rb"
            ) as video_file:

                processed_video = (
                    video_file.read()
                )

            st.video(
                processed_video
            )

            # =================================================
            # CLEAN TEMP FILES
            # =================================================

            try:

                os.remove(
                    input_temp.name
                )

                os.remove(
                    output_path
                )

            except Exception:

                pass

if st.session_state.page == "video":
    if st.button("← Back to Detection Modes", key="back_video"):
        st.session_state.page = "modes"
        st.rerun()
