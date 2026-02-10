import base64
import io
import os
import uuid
from datetime import datetime

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    current_app,
    jsonify,
)
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import models, transforms

from routes.auth import require_login
from db.connection import get_db_connection

main_bp = Blueprint("main", __name__)

_MODEL = None
_FACE_CASCADE = None
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@main_bp.route("/")
def main():
    if not require_login():
        return redirect(url_for("auth.login"))
    _get_active_session_id()
    emotion = request.args.get("emotion")
    score = request.args.get("score")
    emotion_msg = request.args.get("emotion_msg")
    probs = session.pop("emotion_probs", None)
    display_name = session.get("display_name") or session.get("email")

    return render_template(
        "capture.html",
        username=display_name,
        display_name=display_name,
        email=session.get("email"),
        active_page="live",
        emotion=emotion,
        score=score,
        emotion_msg=emotion_msg,
        emotion_probs=probs,
    )


def _load_model(model_path="models/emotion_multi_train_acc_83.pt"):
    global _MODEL
    if _MODEL is None:
        checkpoint = torch.load(model_path, map_location="cpu")
        classes = checkpoint.get("classes", ["disgust"])
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _MODEL = (model, classes)
    return _MODEL


def _get_active_session_id():
    session_id = session.get("emotion_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        session["emotion_session_id"] = session_id
    return session_id


def _log_emotion_event(user_id, session_id, emotion, score):
    if not user_id or not session_id or not emotion:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO emotion_events (user_id, session_id, emotion, score)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, session_id, emotion, score),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _load_face_detector():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def detect_emotion(image_path):
    try:
        model, classes = _load_model()
    except Exception:
        return None, None, "Trained model not found. Train it first.", None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None, None, "Could not read image for emotion detection.", None

    try:
        x = _TRANSFORM(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(x)
            pred_idx = outputs.argmax(dim=1).item()
            probs = torch.softmax(outputs, dim=1)[0].tolist()
        emotion = classes[pred_idx]
        score = probs[pred_idx]
        prob_rows = [
            {"label": label, "prob": float(prob)}
            for label, prob in zip(classes, probs)
        ]
        prob_rows.sort(key=lambda x: x["prob"], reverse=True)
        return emotion, f"{score:.2f}", None, prob_rows
    except Exception:
        return None, None, "Emotion detection failed.", None


def detect_emotions_in_frame(image):
    try:
        model, classes = _load_model()
    except Exception:
        return None, "Trained model not found. Train it first."

    try:
        face_detector = _load_face_detector()
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
    except Exception:
        return None, "Face detection failed."

    detections = []
    for (x, y, w, h) in faces:
        try:
            crop = image.crop((x, y, x + w, y + h))
            x_tensor = _TRANSFORM(crop).unsqueeze(0)
            with torch.no_grad():
                outputs = model(x_tensor)
                pred_idx = outputs.argmax(dim=1).item()
                probs = torch.softmax(outputs, dim=1)[0].tolist()
            emotion = classes[pred_idx]
            score = float(probs[pred_idx])
            detections.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "emotion": emotion,
                    "score": round(score, 2),
                }
            )
        except Exception:
            continue

    return detections, None


@main_bp.route("/capture/save", methods=["POST"])
def capture_save():
    if not require_login():
        return redirect(url_for("auth.login"))
    image_data = request.form.get("image_data", "")

    if not image_data:
        return "No image data received", 400

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    capture_folder = current_app.config["CAPTURE_FOLDER"]
    user_id = session.get("user_id", "user")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_{user_id}_{timestamp}.jpg"
    path = os.path.join(capture_folder, filename)

    with open(path, "wb") as image_file:
        image_file.write(base64.b64decode(image_data))

    emotion, score, emotion_msg, probs = detect_emotion(path)
    if probs:
        session["emotion_probs"] = probs
    if emotion and score:
        _log_emotion_event(
            session.get("user_id"),
            _get_active_session_id(),
            emotion,
            float(score),
        )

    return redirect(
        url_for(
            "main.main",
            saved=1,
            emotion=emotion,
            score=score,
            emotion_msg=emotion_msg,
        )
    )


@main_bp.route("/capture/detect", methods=["POST"])
def capture_detect():
    if not require_login():
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image_data", "")

    if not image_data:
        return jsonify({"error": "No image data received"}), 400

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image"}), 400

    detections, error = detect_emotions_in_frame(image)
    if error:
        return jsonify({"error": error}), 500

    if detections:
        top = max(detections, key=lambda det: det.get("score", 0))
        _log_emotion_event(
            session.get("user_id"),
            _get_active_session_id(),
            top.get("emotion"),
            float(top.get("score", 0)),
        )

    return jsonify(
        {
            "width": image.width,
            "height": image.height,
            "detections": detections,
        }
    )


@main_bp.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("auth.login"))

    user_id = session.get("user_id")
    session_id = _get_active_session_id()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT emotion, COUNT(*) AS cnt
        FROM emotion_events
        WHERE user_id = %s AND session_id = %s
        GROUP BY emotion
        ORDER BY cnt DESC
        """,
        (user_id, session_id),
    )
    distribution = cursor.fetchall()

    cursor.execute(
        """
        SELECT AVG(score), COUNT(*), MIN(captured_at), MAX(captured_at)
        FROM emotion_events
        WHERE user_id = %s AND session_id = %s
        """,
        (user_id, session_id),
    )
    summary_row = cursor.fetchone() or (None, 0, None, None)

    cursor.execute(
        """
        SELECT captured_at, emotion, score
        FROM emotion_events
        WHERE user_id = %s AND session_id = %s
        ORDER BY captured_at ASC
        LIMIT 300
        """,
        (user_id, session_id),
    )
    timeline = cursor.fetchall()
    cursor.close()
    conn.close()

    avg_score = summary_row[0]
    total_frames = summary_row[1] or 0
    start_time = summary_row[2]
    end_time = summary_row[3]
    duration_seconds = 0
    if start_time and end_time:
        duration_seconds = int((end_time - start_time).total_seconds())

    dominant_emotion = distribution[0][0] if distribution else None
    display_name = session.get("display_name") or session.get("email")
    email = session.get("email")

    distribution_rows = [
        {"emotion": row[0], "count": int(row[1])} for row in distribution
    ]
    timeline_points = [
        {
            "time": row[0].isoformat(),
            "emotion": row[1],
            "score": float(row[2]),
        }
        for row in timeline
    ]

    return render_template(
        "dashboard.html",
        distribution=distribution_rows,
        timeline=timeline_points,
        dominant_emotion=dominant_emotion,
        avg_score=avg_score,
        duration_seconds=duration_seconds,
        frame_count=total_frames,
        display_name=display_name,
        email=email,
        active_page="dashboard",
    )


@main_bp.route("/coming-soon")
def coming_soon():
    if not require_login():
        return redirect(url_for("auth.login"))

    page_title = request.args.get("page") or "Coming Soon"
    display_name = session.get("display_name") or session.get("email")
    email = session.get("email")
    active_page = "model" if page_title.lower() == "model info" else "profile"

    return render_template(
        "coming_soon.html",
        page_title=page_title,
        display_name=display_name,
        email=email,
        active_page=active_page,
    )


@main_bp.route("/dashboard/export")
def dashboard_export():
    if not require_login():
        return redirect(url_for("auth.login"))

    export_type = request.args.get("type", "csv").lower()
    if export_type != "csv":
        return jsonify({"error": "Only CSV export is supported right now."}), 400

    user_id = session.get("user_id")
    session_id = _get_active_session_id()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT captured_at, emotion, score
        FROM emotion_events
        WHERE user_id = %s AND session_id = %s
        ORDER BY captured_at ASC
        """,
        (user_id, session_id),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    output = io.StringIO()
    output.write("captured_at,emotion,score\n")
    for captured_at, emotion, score in rows:
        output.write(f"{captured_at},{emotion},{score}\n")

    response = current_app.response_class(
        output.getvalue(),
        mimetype="text/csv",
    )
    response.headers["Content-Disposition"] = "attachment; filename=emotion_results.csv"
    return response
