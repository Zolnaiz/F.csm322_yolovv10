import os
import math
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLOv10

MODEL_CACHE = {}
FACE_DB = []

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _safe_model_load(model_id: str):
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id], None

    weight_name = f"{model_id}.pt"
    candidates = [Path(weight_name), Path("weights") / weight_name]

    urls = [
        f"https://github.com/THU-MIG/yolov10/releases/download/v1.1/{weight_name}",
        f"https://github.com/THU-MIG/yolov10/releases/download/v1.0/{weight_name}",
    ]

    try:
        chosen = next((p for p in candidates if p.exists()), None)
        if chosen is None:
            os.makedirs("weights", exist_ok=True)
            chosen = candidates[1]
            for u in urls:
                try:
                    urlretrieve(u, str(chosen))
                    break
                except Exception:
                    continue
        if not chosen.exists():
            return None, (
                f"Model '{model_id}' load failed: local file '{weight_name}' not found and auto-download failed. "
                "Put weights manually into project root or weights/ folder."
            )

        MODEL_CACHE[model_id] = YOLOv10(str(chosen))
        return MODEL_CACHE[model_id], None
    except Exception as e:
        return None, f"Model '{model_id}' initialize error: {e}"


def _to_embedding(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (112, 112))
    gray = cv2.equalizeHist(gray)
    return (gray.astype(np.float32) / 255.0).flatten()


def load_known_faces(folder="known_faces"):
    FACE_DB.clear()
    root = Path(folder)
    if not root.exists():
        return 0
    for person_dir in root.iterdir():
        if not person_dir.is_dir():
            continue
        embeddings = []
        for img_path in person_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = FACE_CASCADE.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5)
            if len(faces) == 0:
                continue
            x, y, w, h = sorted(faces, key=lambda z: z[2] * z[3], reverse=True)[0]
            embeddings.append(_to_embedding(img[y:y + h, x:x + w]))
        if embeddings:
            FACE_DB.append({"name": person_dir.name, "embeddings": embeddings})
    return len(FACE_DB)


def _recognize_face(face_roi, threshold=0.52):
    if not FACE_DB:
        return "Unknown"
    emb = _to_embedding(face_roi)
    best_name, best_dist = "Unknown", 1e9
    for person in FACE_DB:
        for ref in person["embeddings"]:
            d = np.linalg.norm(emb - ref)
            if d < best_dist:
                best_dist, best_name = d, person["name"]
    return best_name if best_dist <= threshold else "Unknown"


def _overlay_glasses(frame_bgr, glasses_rgba):
    if glasses_rgba is None:
        return frame_bgr
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(face_gray, 1.1, 6)
        if len(eyes) < 2:
            continue
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        centers = []
        for ex, ey, ew, eh in eyes:
            centers.append((x + ex + ew // 2, y + ey + eh // 2))
        (x1, y1), (x2, y2) = centers
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        eye_dist = int(math.hypot(x2 - x1, y2 - y1))

        g = np.array(glasses_rgba.convert("RGBA"))
        target_w = max(eye_dist * 2, 60)
        scale = target_w / g.shape[1]
        target_h = int(g.shape[0] * scale)
        g = cv2.resize(g, (target_w, target_h))

        M = cv2.getRotationMatrix2D((target_w // 2, target_h // 2), angle, 1.0)
        g = cv2.warpAffine(g, M, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x0, y0 = cx - target_w // 2, cy - target_h // 2
        x1c, y1c = max(0, x0), max(0, y0)
        x2c, y2c = min(frame_bgr.shape[1], x0 + target_w), min(frame_bgr.shape[0], y0 + target_h)
        if x1c >= x2c or y1c >= y2c:
            continue
        gx1, gy1 = x1c - x0, y1c - y0
        gx2, gy2 = gx1 + (x2c - x1c), gy1 + (y2c - y1c)
        roi = frame_bgr[y1c:y2c, x1c:x2c]
        g_crop = g[gy1:gy2, gx1:gx2]
        alpha = (g_crop[:, :, 3:4].astype(np.float32) / 255.0)
        roi[:] = (alpha * g_crop[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    return frame_bgr


def _process_frame(frame_bgr, model, conf, mirror, glasses_on, glasses_img, recog_on):
    if mirror:
        frame_bgr = cv2.flip(frame_bgr, 1)
    result = model.predict(source=frame_bgr, conf=conf, verbose=False)[0]
    out = result.plot()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    for x, y, w, h in faces:
        if recog_on:
            name = _recognize_face(frame_bgr[y:y + h, x:x + w])
            color = (0, 255, 0) if name != "Unknown" else (0, 160, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if glasses_on:
        out = _overlay_glasses(out, glasses_img)
    return out


def run_image(image, model_id, conf, mirror, glasses_on, glasses_img, recog_on):
    if image is None:
        return None, "Image оруулна уу."
    model, err = _safe_model_load(model_id)
    if err:
        return None, err
    frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = _process_frame(frame, model, conf, mirror, glasses_on, glasses_img, recog_on)
    return out[:, :, ::-1], "Done"


def run_video(video_path, model_id, conf, mirror, glasses_on, glasses_img, recog_on):
    if not video_path:
        return None, "Video оруулна уу."
    model, err = _safe_model_load(model_id)
    if err:
        return None, err

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(_process_frame(frame, model, conf, mirror, glasses_on, glasses_img, recog_on))

    cap.release()
    writer.release()
    return out_path, "Done"


def run_webcam(frame, model_id, conf, mirror, glasses_on, glasses_img, recog_on):
    if frame is None:
        return None
    model, err = _safe_model_load(model_id)
    if err:
        return frame
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out = _process_frame(bgr, model, conf, mirror, glasses_on, glasses_img, recog_on)
    return out[:, :, ::-1]


with gr.Blocks(title="YOLOv10 CV Demo") as demo:
    gr.Markdown("# YOLOv10 Computer Vision Demo")
    known = load_known_faces("known_faces")
    status = gr.Markdown(f"Known faces loaded: **{known}**")

    model_id = gr.Dropdown(["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"], value="yolov10n", label="Model")
    conf = gr.Slider(0.1, 0.9, value=0.25, step=0.01, label="Confidence")
    mirror = gr.Checkbox(label="Mirror mode", value=False)
    glasses_toggle = gr.Checkbox(label="Glasses filter", value=False)
    face_recog = gr.Checkbox(label="Face recognition", value=True)
    glasses_img = gr.Image(type="pil", label="Glasses PNG (transparent)")

    with gr.Tabs():
        with gr.Tab("Image"):
            inp = gr.Image(type="pil")
            out = gr.Image()
            msg = gr.Textbox(label="Status")
            btn = gr.Button("Run")
            btn.click(run_image, [inp, model_id, conf, mirror, glasses_toggle, glasses_img, face_recog], [out, msg])

        with gr.Tab("Video"):
            vin = gr.Video()
            vout = gr.Video()
            vmsg = gr.Textbox(label="Status")
            vbtn = gr.Button("Run")
            vbtn.click(run_video, [vin, model_id, conf, mirror, glasses_toggle, glasses_img, face_recog], [vout, vmsg])

        with gr.Tab("Webcam"):
            webcam_in = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam")
            webcam_out = gr.Image(type="numpy", label="Output")
            webcam_in.stream(run_webcam, [webcam_in, model_id, conf, mirror, glasses_toggle, glasses_img, face_recog], webcam_out)

    reload_btn = gr.Button("Reload known_faces")
    reload_btn.click(lambda: f"Known faces loaded: **{load_known_faces('known_faces')}**", outputs=status)


if __name__ == "__main__":
    demo.launch()
