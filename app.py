import gradio as gr
import cv2
import tempfile
import os
import numpy as np
from urllib.request import urlretrieve
from ultralytics import YOLOv10

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_MODEL_CACHE = {}
_FACE_GALLERY = []


def _apply_overlay(frame, overlay_image, opacity):
    if overlay_image is None or opacity <= 0:
        return frame
    overlay = cv2.cvtColor(np.array(overlay_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)


def _blend_overlay_roi(roi, overlay_rgb, opacity):
    if overlay_rgb.shape[-1] == 4:
        overlay_bgr = cv2.cvtColor(overlay_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = (overlay_rgb[:, :, 3].astype(np.float32) / 255.0) * opacity
        alpha = np.expand_dims(alpha, axis=-1)
        blended = (roi.astype(np.float32) * (1.0 - alpha) + overlay_bgr.astype(np.float32) * alpha)
        return blended.astype(np.uint8)

    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(roi, 1 - opacity, overlay_bgr, opacity, 0)


def _apply_face_filter(frame, overlay_image, opacity):
    if overlay_image is None or opacity <= 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return frame

    overlay_rgba = np.array(overlay_image.convert("RGBA"))
    for x, y, w, h in faces:
        resized = cv2.resize(overlay_rgba, (w, h))
        roi = frame[y:y + h, x:x + w]
        frame[y:y + h, x:x + w] = _blend_overlay_roi(roi, resized, opacity)
    return frame


def _extract_face_embedding(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (64, 64)).astype(np.float32) / 255.0
    face = cv2.equalizeHist((face * 255).astype(np.uint8)).astype(np.float32) / 255.0
    return face.flatten()


def register_face(person_name, person_image):
    if not person_name or person_image is None:
        return "⚠️ Нэр болон зураг оруулна уу.", gr.update(choices=[p["name"] for p in _FACE_GALLERY])

    image_bgr = cv2.cvtColor(np.array(person_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    embedding = _extract_face_embedding(image_bgr)
    if embedding is None:
        return "⚠️ Зурган дээр нүүр илэрсэнгүй.", gr.update(choices=[p["name"] for p in _FACE_GALLERY])

    _FACE_GALLERY[:] = [p for p in _FACE_GALLERY if p["name"] != person_name.strip()]
    _FACE_GALLERY.append({"name": person_name.strip(), "embedding": embedding})
    return f"✅ '{person_name.strip()}' амжилттай бүртгэгдлээ.", gr.update(choices=[p["name"] for p in _FACE_GALLERY])


def clear_faces():
    _FACE_GALLERY.clear()
    return "🧹 Бүртгэл цэвэрлэгдлээ.", gr.update(choices=[])


def _recognize_faces(frame_bgr, threshold=0.38):
    labeled = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    found_names = []

    for x, y, w, h in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64)).astype(np.float32) / 255.0
        face = cv2.equalizeHist((face * 255).astype(np.uint8)).astype(np.float32) / 255.0
        emb = face.flatten()

        best_name, best_dist = "Unknown", 1e9
        for person in _FACE_GALLERY:
            dist = np.linalg.norm(emb - person["embedding"])
            if dist < best_dist:
                best_dist = dist
                best_name = person["name"]
        if best_dist > threshold:
            best_name = "Unknown"

        found_names.append(best_name)
        color = (50, 220, 120) if best_name != "Unknown" else (70, 120, 255)
        cv2.rectangle(labeled, (x, y), (x + w, y + h), color, 2)
        cv2.putText(labeled, best_name, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return labeled, found_names


def realtime_face_recognition(frame):
    if frame is None:
        return None, "⚠️ Камер идэвхжүүлнэ үү."
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    labeled, names = _recognize_faces(frame_bgr)
    summary = "Илэрсэн хүн: " + (", ".join(names) if names else "байхгүй")
    return labeled[:, :, ::-1], summary


def _get_model(model_id):
    if model_id not in _MODEL_CACHE:
        fallback_weights = f"{model_id}.pt"
        if os.path.exists(fallback_weights):
            _MODEL_CACHE[model_id] = YOLOv10(fallback_weights)
            return _MODEL_CACHE[model_id]

        weights_dir = "weights"
        os.makedirs(weights_dir, exist_ok=True)
        cached_weights = os.path.join(weights_dir, fallback_weights)
        if not os.path.exists(cached_weights):
            urls = [
                f"https://github.com/THU-MIG/yolov10/releases/download/v1.1/{fallback_weights}",
                f"https://github.com/THU-MIG/yolov10/releases/download/v1.0/{fallback_weights}",
            ]
            last_error = None
            for url in urls:
                try:
                    urlretrieve(url, cached_weights)
                    break
                except Exception as e:
                    last_error = e
            if not os.path.exists(cached_weights):
                raise RuntimeError(
                    f"Failed to load model '{model_id}'. Missing local weights '{fallback_weights}' and "
                    f"automatic download failed. Last download error: {last_error}"
                ) from last_error
        _MODEL_CACHE[model_id] = YOLOv10(cached_weights)
    return _MODEL_CACHE[model_id]


def _build_summary(result, mode_name):
    if not result or not getattr(result, "boxes", None):
        return f"**{mode_name}**: Илэрсэн объект байхгүй."
    names = result.names
    classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    counts = {}
    for idx in classes:
        cls_name = names.get(idx, str(idx)) if isinstance(names, dict) else names[idx]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    txt = " | ".join([f"`{k}`: **{v}**" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))])
    return f"**{mode_name} Нэгдсэн дүн:** {txt}"


def yolov10_inference(image, model_id, image_size, conf_threshold, overlay_image=None, overlay_opacity=0.0,
                      face_filter_only=False):
    if image is None:
        return None, None, None, ""

    model = _get_model(model_id)
    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if face_filter_only:
        image_bgr = _apply_face_filter(image_bgr, overlay_image, overlay_opacity)
    else:
        image_bgr = _apply_overlay(image_bgr, overlay_image, overlay_opacity)
    results = model.predict(source=image_bgr, imgsz=image_size, conf=conf_threshold)
    annotated_image = results[0].plot()
    return annotated_image[:, :, ::-1], None, annotated_image[:, :, ::-1], _build_summary(results[0], "Зураг")


def yolov10_video_inference(video, model_id, image_size, conf_threshold, overlay_image=None, overlay_opacity=0.0,
                            face_filter_only=False, preview_stride=1):
    if video is None:
        yield None, None, None, ""
        return

    model = _get_model(model_id)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as input_tmp:
        with open(video, "rb") as g:
            input_tmp.write(g.read())
        video_path = input_tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as output_tmp:
        output_video_path = output_tmp.name
    if fps <= 0:
        fps = 25.0
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))
    if not out.isOpened():
        output_video_path = output_video_path.rsplit(".", 1)[0] + ".mp4"
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_index = 0
    total_counts = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if face_filter_only:
            frame = _apply_face_filter(frame, overlay_image, overlay_opacity)
        else:
            frame = _apply_overlay(frame, overlay_image, overlay_opacity)
        results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
        result = results[0]
        if getattr(result, "boxes", None):
            names = result.names
            classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
            for idx in classes:
                cls_name = names.get(idx, str(idx)) if isinstance(names, dict) else names[idx]
                total_counts[cls_name] = total_counts.get(cls_name, 0) + 1

        annotated_frame = result.plot()
        out.write(annotated_frame)
        summary_text = " | ".join([f"`{k}`: **{v}**" for k, v in sorted(total_counts.items(), key=lambda x: (-x[1], x[0]))])
        if frame_index % max(preview_stride, 1) == 0:
            yield None, None, annotated_frame[:, :, ::-1], f"**Видео Нэгдсэн дүн:** {summary_text or 'одоогоор хоосон'}"
        frame_index += 1

    cap.release()
    out.release()
    if os.path.exists(video_path):
        os.remove(video_path)

    summary_text = " | ".join([f"`{k}`: **{v}**" for k, v in sorted(total_counts.items(), key=lambda x: (-x[1], x[0]))])
    yield None, output_video_path, None, f"**Видео Нэгдсэн дүн:** {summary_text or 'илэрсэн объект байхгүй'}"


def app():
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="emerald",
            neutral_hue="slate",
            font=["Inter", "ui-sans-serif", "system-ui"],
        ),
        css="""
        body, .gradio-container {background: linear-gradient(140deg, #0f172a, #111827) !important; color: #e5e7eb !important;}
        .gr-button {border-radius: 12px !important; font-weight: 700 !important;}
        .gr-box, .block, .gr-panel {background: rgba(17,24,39,.78) !important; border: 1px solid rgba(148,163,184,.2) !important;}
        .title-main {text-align:center; font-size: 30px; font-weight: 800; margin: 8px 0 2px; color:#c7d2fe;}
        .title-sub {text-align:center; color:#cbd5e1; margin-bottom: 12px;}
        """,
    ) as demo:
        gr.Markdown("""<div class='title-main'>🚀 Smart Vision Control Center</div>
        <div class='title-sub'>Object detection, result aggregation, download, and real-time face recognition.</div>""")

        with gr.Tabs():
            with gr.TabItem("📦 Detection Studio"):
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(type="pil", label="Image", sources=["upload"], visible=True)
                        video = gr.Video(label="Video", sources=["upload"], visible=False)
                        input_type = gr.Radio(choices=["Image", "Video"], value="Image", label="Input Type")
                        overlay_image = gr.Image(type="pil", label="Filter Overlay Image (optional)", sources=["upload"])
                        overlay_opacity = gr.Slider(label="Filter Opacity", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
                        face_filter_only = gr.Checkbox(label="Apply filter on detected faces only", value=True)
                        model_id = gr.Dropdown(
                            label="Model",
                            choices=["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"],
                            value="yolov10m",
                        )
                        image_size = gr.Slider(label="Image Size", minimum=320, maximum=1280, step=32, value=640)
                        conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.25)
                        preview_stride = gr.Slider(label="Live Preview Frame Stride (Video)", minimum=1, maximum=10, step=1, value=2)
                        detect_btn = gr.Button(value="🚀 Detect & Aggregate")
                        status_text = gr.Markdown("✅ Ready")

                    with gr.Column():
                        output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                        output_video = gr.Video(label="Annotated Video (татаж авах боломжтой)", visible=False)
                        live_preview = gr.Image(type="numpy", label="Live Video Preview", visible=False)
                        result_summary = gr.Markdown("**Нэгдсэн дүн:** -")

                def update_visibility(input_mode):
                    return (
                        gr.update(visible=input_mode == "Image"),
                        gr.update(visible=input_mode == "Video"),
                        gr.update(visible=input_mode == "Image"),
                        gr.update(visible=input_mode == "Video"),
                        gr.update(visible=input_mode == "Video"),
                    )

                input_type.change(fn=update_visibility, inputs=[input_type], outputs=[image, video, output_image, output_video, live_preview])

                def run_video_inference_with_status(image, video, model_id, image_size, conf_threshold, input_type, overlay_image,
                                                    overlay_opacity, face_filter_only, preview_stride):
                    try:
                        if input_type == "Image":
                            if image is None:
                                yield None, None, None, "⚠️ Please upload an image first.", ""
                                return
                            out_img, out_vid, preview, summary = yolov10_inference(
                                image, model_id, image_size, conf_threshold, overlay_image, overlay_opacity, face_filter_only
                            )
                            yield out_img, out_vid, preview, "✅ Detection completed.", summary
                            return

                        if video is None:
                            yield None, None, None, "⚠️ Please upload a video first.", ""
                            return

                        for _, out_video_path, preview_frame, summary in yolov10_video_inference(
                            video, model_id, image_size, conf_threshold, overlay_image, overlay_opacity, face_filter_only,
                            preview_stride,
                        ):
                            status = "⏳ Processing video..." if out_video_path is None else "✅ Video done. You can download it."
                            yield None, out_video_path, preview_frame, status, summary
                    except Exception as e:
                        yield None, None, None, f"❌ Error: {str(e)}", ""

                detect_btn.click(
                    fn=run_video_inference_with_status,
                    inputs=[image, video, model_id, image_size, conf_threshold, input_type, overlay_image, overlay_opacity,
                            face_filter_only, preview_stride],
                    outputs=[output_image, output_video, live_preview, status_text, result_summary],
                )

            with gr.TabItem("🧑‍💻 Realtime Face ID"):
                with gr.Row():
                    with gr.Column(scale=1):
                        person_name = gr.Textbox(label="Хүний нэр")
                        person_image = gr.Image(type="pil", label="Бүртгэх зураг", sources=["upload"])
                        register_btn = gr.Button("➕ Хүн бүртгэх")
                        clear_btn = gr.Button("🧹 Бүртгэлийг цэвэрлэх")
                        register_status = gr.Markdown("Бүртгэл хоосон.")
                        known_people = gr.Dropdown(label="Бүртгэлтэй хүмүүс", choices=[], interactive=False)
                    with gr.Column(scale=2):
                        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Realtime Camera")
                        face_output = gr.Image(type="numpy", label="Face Recognition Output")
                        face_summary = gr.Markdown("Илэрсэн хүн: -")

                register_btn.click(register_face, [person_name, person_image], [register_status, known_people])
                clear_btn.click(clear_faces, outputs=[register_status, known_people])
                cam.stream(realtime_face_recognition, [cam], [face_output, face_summary])

    return demo


gradio_app = app()

if __name__ == '__main__':
    gradio_app.launch()
