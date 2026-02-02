import base64
import os
from datetime import datetime

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    current_app,
)
from PIL import Image
import torch
from torchvision import models, transforms

from routes.auth import require_login

main_bp = Blueprint("main", __name__)

_MODEL = None
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
    emotion = request.args.get("emotion")
    score = request.args.get("score")
    emotion_msg = request.args.get("emotion_msg")
    probs = session.pop("emotion_probs", None)
    display_name = session.get("display_name") or session.get("email")

    return render_template(
        "capture.html",
        username=display_name,
        emotion=emotion,
        score=score,
        emotion_msg=emotion_msg,
        emotion_probs=probs,
    )


def _load_model(model_path="models/emotion_multi.pt"):
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

    return redirect(
        url_for(
            "main.main",
            saved=1,
            emotion=emotion,
            score=score,
            emotion_msg=emotion_msg,
        )
    )
