import math
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLOv10

try:
    import face_recognition as fr
except Exception:
    fr = None

MODEL_CACHE = {}
FACE_DB = []
FACE_ENGINE = "opencv"

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
DEFAULT_GLASSES_PATH = Path("assets/glasses.png")


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
                    pass
        if not chosen.exists():
            return None, f"Model '{model_id}' load failed. Put {weight_name} in root or weights/."
        MODEL_CACHE[model_id] = YOLOv10(str(chosen))
        return MODEL_CACHE[model_id], None
    except Exception as e:
        return None, f"Model init error ({model_id}): {e}"


def _opencv_embedding(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (120, 120))
    gray = cv2.equalizeHist(gray)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-9)
    small = cv2.resize(gray, (32, 32)).astype(np.float32).flatten() / 255.0
    return np.concatenate([hist, small])


def _detect_primary_face(img_bgr):
    faces = FACE_CASCADE.detectMultiScale(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 1.1, 5)
    if len(faces) == 0:
        return None
    return sorted(faces, key=lambda z: z[2] * z[3], reverse=True)[0]


def load_known_faces(folder="known_faces"):
    global FACE_ENGINE
    FACE_DB.clear()
    root = Path(folder)
    if not root.exists():
        return 0, "known_faces folder not found"

    FACE_ENGINE = "face_recognition" if fr is not None else "opencv"

    for person_dir in root.iterdir():
        if not person_dir.is_dir():
            continue
        embeddings = []
        for img_path in person_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if FACE_ENGINE == "face_recognition":
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = fr.face_encodings(rgb)
                if encs:
                    embeddings.append(encs[0])
            else:
                face = _detect_primary_face(img)
                if face is None:
                    continue
                x, y, w, h = face
                embeddings.append(_opencv_embedding(img[y:y + h, x:x + w]))
        if embeddings:
            FACE_DB.append({"name": person_dir.name, "embeddings": embeddings})
    return len(FACE_DB), FACE_ENGINE


def _recognize_face(face_roi, threshold=0.47):
    if not FACE_DB:
        return "Unknown"

    emb = None
    if FACE_ENGINE == "face_recognition" and fr is not None:
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        encs = fr.face_encodings(rgb)
        if encs:
            emb = encs[0]
    if emb is None:
        emb = _opencv_embedding(face_roi)

    best_name, best_dist = "Unknown", 1e9
    for person in FACE_DB:
        for ref in person["embeddings"]:
            d = np.linalg.norm(emb - ref)
            if d < best_dist:
                best_dist, best_name = d, person["name"]
    return best_name if best_dist <= threshold else "Unknown"


def _resolve_glasses_image(glasses_img):
    if glasses_img is not None:
        return np.array(glasses_img.convert("RGBA"))
    if DEFAULT_GLASSES_PATH.exists():
        default = cv2.imread(str(DEFAULT_GLASSES_PATH), cv2.IMREAD_UNCHANGED)
        if default is not None:
            if default.shape[2] == 3:
                alpha = np.full((default.shape[0], default.shape[1], 1), 255, dtype=np.uint8)
                default = np.concatenate([default, alpha], axis=2)
            return cv2.cvtColor(default, cv2.COLOR_BGRA2RGBA)
    return None


def _overlay_glasses(frame_bgr, glasses_img):
    g_src = _resolve_glasses_image(glasses_img)
    if g_src is None:
        return frame_bgr

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(face_gray, 1.1, 6)

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
            p1 = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
            p2 = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            eye_dist = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            target_w = max(eye_dist * 2, 70)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        else:
            angle = 0.0
            target_w = max(int(w * 0.95), 70)
            cx, cy = x + w // 2, y + int(h * 0.38)

        target_h = max(int(target_w * g_src.shape[0] / g_src.shape[1]), 30)
        g = cv2.resize(g_src, (target_w, target_h), interpolation=cv2.INTER_AREA)
        M = cv2.getRotationMatrix2D((target_w // 2, target_h // 2), angle, 1.0)
        g = cv2.warpAffine(g, M, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        x0, y0 = cx - target_w // 2, cy - target_h // 2
        x1c, y1c = max(0, x0), max(0, y0)
        x2c, y2c = min(frame_bgr.shape[1], x0 + target_w), min(frame_bgr.shape[0], y0 + target_h)
        if x1c >= x2c or y1c >= y2c:
            continue

        gx1, gy1 = x1c - x0, y1c - y0
        gx2, gy2 = gx1 + (x2c - x1c), gy1 + (y2c - y1c)
        roi = frame_bgr[y1c:y2c, x1c:x2c]
        g_crop = g[gy1:gy2, gx1:gx2]
        if g_crop.shape[2] == 3:
            alpha = np.ones((g_crop.shape[0], g_crop.shape[1], 1), dtype=np.float32)
            rgb = g_crop
        else:
            alpha = g_crop[:, :, 3:4].astype(np.float32) / 255.0
            rgb = g_crop[:, :, :3]
        roi[:] = (alpha * cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) + (1 - alpha) * roi).astype(np.uint8)

    return frame_bgr


def _predict_with_resize(model, frame_bgr, conf, max_side=960):
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    infer = frame_bgr if scale == 1.0 else cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    r = model.predict(source=infer, conf=conf, verbose=False)[0]
    out = frame_bgr.copy()
    if r.boxes is None or len(r.boxes) == 0:
        return out
    names = r.names
    boxes = r.boxes.xyxy.detach().cpu().numpy()
    cls = r.boxes.cls.detach().cpu().numpy().astype(int)
    cfs = r.boxes.conf.detach().cpu().numpy()
    inv = (1.0 / scale) if scale != 0 else 1.0
    for b, ci, c in zip(boxes, cls, cfs):
        x1, y1, x2, y2 = (b * inv).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (48, 200, 48), 2)
        label = f"{names[int(ci)]} {c:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (48, 200, 48), 2)
    return out


def _process_frame(frame_bgr, model, conf, mirror, glasses_on, glasses_img, recog_on):
    if mirror:
        frame_bgr = cv2.flip(frame_bgr, 1)
    out = _predict_with_resize(model, frame_bgr, conf)

    if recog_on:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
        for x, y, w, h in faces:
            name = _recognize_face(frame_bgr[y:y + h, x:x + w])
            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, name, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if glasses_on:
        out = _overlay_glasses(out, glasses_img)
    return out


def run_image(image, model_id, conf, mirror, glasses_on, glasses_img, recog_on):
    try:
        if image is None:
            return None, "Image оруулна уу."
        model, err = _safe_model_load(model_id)
        if err:
            return None, err
        frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        out = _process_frame(frame, model, conf, mirror, glasses_on, glasses_img, recog_on)
        return out[:, :, ::-1], f"Done | Known: {len(FACE_DB)} | Engine: {FACE_ENGINE}"
    except Exception as e:
        return None, f"Image processing error: {e}"


def run_video(video_path, model_id, conf, mirror, glasses_on, glasses_img, recog_on):
    cap = None
    writer = None
    try:
        if not video_path:
            return None, "Video оруулна уу."
        model, err = _safe_model_load(model_id)
        if err:
            return None, err

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Video open error"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 240:
            fps = 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = tempfile.mktemp(suffix=".mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(_process_frame(frame, model, conf, mirror, glasses_on, glasses_img, recog_on))
        return out_path, f"Done | fps={fps:.2f} | size={w}x{h}"
    except Exception as e:
        return None, f"Video processing error: {e}"
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def run_webcam(frame, model_id, conf, mirror, glasses_on, glasses_img, recog_on):
    try:
        if frame is None:
            return None
        model, err = _safe_model_load(model_id)
        if err:
            return frame
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out = _process_frame(bgr, model, conf, mirror, glasses_on, glasses_img, recog_on)
        return out[:, :, ::-1]
    except Exception:
        return frame


def _reload_status():
    n, engine = load_known_faces("known_faces")
    return f"Known faces loaded: **{n}** (engine: `{engine}`)"


with gr.Blocks(title="YOLOv10 CV Demo") as demo:
    gr.Markdown("# YOLOv10 Computer Vision Demo")
    known, engine = load_known_faces("known_faces")
    status = gr.Markdown(f"Known faces loaded: **{known}** (engine: `{engine}`)")

    model_id = gr.Dropdown(["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"], value="yolov10n", label="Model")
    conf = gr.Slider(0.1, 0.9, value=0.25, step=0.01, label="Confidence")
    mirror = gr.Checkbox(label="Mirror mode", value=False)
    glasses_toggle = gr.Checkbox(label="Glasses filter", value=False)
    face_recog = gr.Checkbox(label="Face recognition", value=True)
    glasses_img = gr.Image(type="pil", label="Glasses PNG (optional)")

    with gr.Tabs():
        with gr.Tab("Image"):
            inp = gr.Image(type="pil")
            out = gr.Image()
            msg = gr.Textbox(label="Status")
            gr.Button("Run").click(run_image, [inp, model_id, conf, mirror, glasses_toggle, glasses_img, face_recog], [out, msg])

        with gr.Tab("Video"):
            vin = gr.Video()
            vout = gr.Video()
            vmsg = gr.Textbox(label="Status")
            gr.Button("Run").click(run_video, [vin, model_id, conf, mirror, glasses_toggle, glasses_img, face_recog], [vout, vmsg])

        with gr.Tab("Webcam"):
            webcam_in = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam")
            webcam_out = gr.Image(type="numpy", label="Output")
            webcam_in.stream(run_webcam, [webcam_in, model_id, conf, mirror, glasses_toggle, glasses_img, face_recog], webcam_out)

    gr.Button("Reload known_faces").click(_reload_status, outputs=status)


if __name__ == "__main__":
    demo.launch()
