import gradio as gr
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLOv10

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_MODEL_CACHE = {}


def _apply_overlay(frame, overlay_image, opacity):
    if overlay_image is None or opacity <= 0:
        return frame
    overlay = cv2.cvtColor(np.array(overlay_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)


def _apply_face_filter(frame, overlay_image, opacity):
    """Detect faces and apply filter overlay only on face regions."""
    if overlay_image is None or opacity <= 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return frame

    overlay_rgb = np.array(overlay_image.convert("RGB"))
    for x, y, w, h in faces:
        resized = cv2.resize(overlay_rgb, (w, h))
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        roi = frame[y:y + h, x:x + w]
        frame[y:y + h, x:x + w] = cv2.addWeighted(roi, 1 - opacity, resized_bgr, opacity, 0)
    return frame


def _get_model(model_id):
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = YOLOv10.from_pretrained(f"jameslahm/{model_id}")
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
                            face_filter_only=False):
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
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

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
        yield None, None, annotated_frame[:, :, ::-1]

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
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", sources=["upload", "webcam"], visible=True)
                video = gr.Video(label="Video / Webcam Recording", sources=["upload", "webcam"], visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                overlay_image = gr.Image(type="pil", label="Filter Overlay (Canvas/Image)", sources=["upload", "clipboard", "webcam"])
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
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
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
                          face_filter_only):
            if input_type == "Image":
                result = yolov10_inference(
                    image,
                    model_id,
                    image_size,
                    conf_threshold,
                    overlay_image,
                    overlay_opacity,
                    face_filter_only,
                )
                return result
            else:
                return yolov10_video_inference(
                    video,
                    model_id,
                    image_size,
                    conf_threshold,
                    overlay_image,
                    overlay_opacity,
                    face_filter_only,
                )


        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type, overlay_image, overlay_opacity,
                    face_filter_only],
            outputs=[output_image, output_video, live_preview],
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
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
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
