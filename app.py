import gradio as gr
import cv2
import tempfile
import os
import numpy as np
from urllib.request import urlretrieve
from ultralytics import YOLOv10

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_MODEL_CACHE = {}


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
    """Detect faces and apply filter overlay only on face regions."""
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


def _get_model(model_id):
    if model_id not in _MODEL_CACHE:
        fallback_weights = f"{model_id}.pt"
        if os.path.exists(fallback_weights):
            _MODEL_CACHE[model_id] = YOLOv10(fallback_weights)
            return _MODEL_CACHE[model_id]

        # Avoid Hub safetensors code path by downloading plain .pt weights directly.
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


def yolov10_inference(image, model_id, image_size, conf_threshold, overlay_image=None, overlay_opacity=0.0,
                      face_filter_only=False):
    if image is None:
        return None, None, None

    model = _get_model(model_id)
    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if face_filter_only:
        image_bgr = _apply_face_filter(image_bgr, overlay_image, overlay_opacity)
    else:
        image_bgr = _apply_overlay(image_bgr, overlay_image, overlay_opacity)
    results = model.predict(source=image_bgr, imgsz=image_size, conf=conf_threshold)
    annotated_image = results[0].plot()
    return annotated_image[:, :, ::-1], None, annotated_image[:, :, ::-1]


def yolov10_video_inference(video, model_id, image_size, conf_threshold, overlay_image=None, overlay_opacity=0.0,
                            face_filter_only=False, preview_stride=1):
    if video is None:
        yield None, None, None
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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if face_filter_only:
            frame = _apply_face_filter(frame, overlay_image, overlay_opacity)
        else:
            frame = _apply_overlay(frame, overlay_image, overlay_opacity)
        results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        if frame_index % max(preview_stride, 1) == 0:
            yield None, None, annotated_frame[:, :, ::-1]
        frame_index += 1

    cap.release()
    out.release()
    if os.path.exists(video_path):
        os.remove(video_path)

    yield None, output_video_path, None


def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold, face_filter_only):
    annotated_image, _, _ = yolov10_inference(
        image,
        model_path,
        image_size,
        conf_threshold,
        face_filter_only=face_filter_only,
    )
    return annotated_image


def app():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
        .gr-button {border-radius: 12px !important; font-weight: 700 !important;}
        .title-main {text-align:center; font-size: 30px; font-weight: 800; margin: 8px 0 2px;}
        .title-sub {text-align:center; opacity:0.9; margin-bottom: 12px;}
        """,
    ):
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", sources=["upload"], visible=True)
                video = gr.Video(label="Video", sources=["upload"], visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                overlay_image = gr.Image(type="pil", label="Filter Overlay Image (optional)", sources=["upload"])
                overlay_opacity = gr.Slider(
                    label="Filter Opacity",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.0,
                )
                face_filter_only = gr.Checkbox(
                    label="Apply filter on detected faces only",
                    value=True,
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="yolov10m",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                preview_stride = gr.Slider(
                    label="Live Preview Frame Stride (Video)",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=2,
                )
                yolov10_infer = gr.Button(value="🚀 Detect Objects")
                status_text = gr.Markdown("✅ Ready. Upload an image/video and click **Detect Objects**.")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                result_summary = gr.Markdown("")
                output_video = gr.Video(label="Annotated Video", visible=False)
                live_preview = gr.Image(type="numpy", label="Live Video Preview", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            live_preview = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video, live_preview

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video, live_preview],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type, overlay_image, overlay_opacity,
                          face_filter_only, preview_stride):
            try:
                if input_type == "Image":
                    if image is None:
                        return None, None, None, "⚠️ Please upload an image first.", ""
                    annotated, _, preview = yolov10_inference(
                        image,
                        model_id,
                        image_size,
                        conf_threshold,
                        overlay_image,
                        overlay_opacity,
                        face_filter_only,
                    )
                    if annotated is None:
                        return None, None, None, "⚠️ No output generated. Try another image.", ""
                    summary = f"**Model:** `{model_id}` | **Input:** Image | **Conf:** `{conf_threshold}`"
                    return annotated, None, preview, "✅ Detection completed successfully.", summary
                if video is None:
                    return None, None, None, "⚠️ Please upload a video first.", ""
                return yolov10_video_inference(
                    video,
                    model_id,
                    image_size,
                    conf_threshold,
                    overlay_image,
                    overlay_opacity,
                    face_filter_only,
                    preview_stride,
                )
            except Exception as e:
                return None, None, None, f"❌ Error: {str(e)}", "Try disabling face-only filter or use another file."

        def run_video_inference_with_status(image, video, model_id, image_size, conf_threshold, input_type, overlay_image,
                                            overlay_opacity, face_filter_only, preview_stride):
            if input_type == "Image":
                yield run_inference(
                    image, video, model_id, image_size, conf_threshold, input_type, overlay_image,
                    overlay_opacity, face_filter_only, preview_stride,
                )
                return
            if video is None:
                yield None, None, None, "⚠️ Please upload a video first.", ""
                return
            try:
                for _, out_video, preview in yolov10_video_inference(
                    video,
                    model_id,
                    image_size,
                    conf_threshold,
                    overlay_image,
                    overlay_opacity,
                    face_filter_only,
                    preview_stride,
                ):
                    status = "⏳ Processing video..." if out_video is None else "✅ Video detection completed."
                    summary = f"**Model:** `{model_id}` | **Input:** Video | **Conf:** `{conf_threshold}`"
                    yield None, out_video, preview, status, summary
            except Exception as e:
                yield None, None, None, f"❌ Error: {str(e)}", "Try MP4/WebM file and reduce image size."


        yolov10_infer.click(
            fn=run_video_inference_with_status,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type, overlay_image, overlay_opacity,
                    face_filter_only, preview_stride],
            outputs=[output_image, output_video, live_preview, status_text, result_summary],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s",
                    640,
                    0.25,
                    True,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s",
                    640,
                    0.25,
                    True,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
                face_filter_only,
            ],
            outputs=[output_image],
            cache_examples=False,
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <div class='title-main'>🚲 YOLOv10 Smart Object Detection Studio</div>
    <div class='title-sub'>Upload an image/video and detect objects with smoother UX.</div>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()
