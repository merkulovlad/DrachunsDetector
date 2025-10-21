import collections
import queue
import threading
import time
from typing import List, Optional, Sequence, Union

import cv2

from stream_platform.backend.data.preprocess import preprocess_frame
from stream_platform.backend.src.infer import run_inference
from stream_platform.backend.utils.logger import get_logger

# Number of seconds to skip inference after a no-violence prediction.
SKIP_SECONDS_AFTER_CLEAR = 3.0
T_FRAMES = 6
URL = "rtsp://localhost:8554/mystream5"
STRIDE = 1
DEFAULT_POSITIVE_LABEL = "violence"

logger = get_logger("camera_loop")
DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

def _normalize_class_names(
    class_names: Optional[Union[Sequence[str], dict]]
) -> Optional[List[str]]:
    if class_names is None:
        return None
    if isinstance(class_names, dict):
        try:
            return [class_names[i] for i in sorted(class_names)]  # type: ignore[index]
        except Exception:
            return list(class_names.values())
    if isinstance(class_names, (list, tuple)):
        return list(class_names)
    return None


def _resolve_positive_index(
    names: Optional[Sequence[str]], positive_label: str
) -> int:
    if not names:
        return 0
    positive_label_lower = positive_label.lower()
    lowered = [name.lower() for name in names]
    if positive_label_lower in lowered:
        return lowered.index(positive_label_lower)
    logger.warning(
        "positive_label_not_found",
        positive_label=positive_label,
        available=names,
    )
    return 0


def capture_loop(
    model,
    url: str = URL,
    T: int = T_FRAMES,
    logger=logger,
    *,
    class_names: Optional[Union[Sequence[str], dict]] = None,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
    detection_threshold: float = 0.5,
    idle_threshold: float = 0.2,
    skip_seconds_after_clear: float = SKIP_SECONDS_AFTER_CLEAR,
):
    class_names_list = _normalize_class_names(class_names)
    positive_idx = _resolve_positive_index(class_names_list, positive_label)
    positive_label_display = (
        class_names_list[positive_idx] if class_names_list else f"class {positive_idx}"
    )

    if class_names_list:
        logger.info(
            "class_index_mapping",
            mapping={idx: name for idx, name in enumerate(class_names_list)},
        )

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    skip_frames_after_clear = max(int(round(fps * skip_seconds_after_clear)), 1)

    clip_queue: "queue.Queue[List]" = queue.Queue(maxsize=T_FRAMES)
    result_lock = threading.Lock()
    result_state = {"data": None}
    stop_event = threading.Event()

    def inference_worker():
        while not stop_event.is_set():
            try:
                clip = clip_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if clip is None:
                    break
                if hasattr(model, "clean_activation_buffers"):
                    model.clean_activation_buffers()
                probs = run_inference(model, clip)
                prob_vec = probs.squeeze(0).detach()
                if prob_vec.device.type != "cpu":
                    prob_vec = prob_vec.cpu()
                prob_list = [float(p) for p in prob_vec.tolist()]
                top_idx = int(prob_vec.argmax().item())
                positive_prob = (
                    prob_list[0]
                    if positive_idx >= len(prob_list)
                    else prob_list[positive_idx]
                )
                top_prob = prob_list[top_idx]
                with result_lock:
                    result_state["data"] = {
                        "prob_list": prob_list,
                        "positive_prob": float(positive_prob),
                        "top_idx": top_idx,
                        "top_prob": float(top_prob),
                        "timestamp": time.time(),
                    }
            finally:
                clip_queue.task_done()

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    logger.info(
        "Starting camera loop from url",
        url=url,
        positive_label=positive_label_display,
        detection_threshold=detection_threshold,
        idle_threshold=idle_threshold,
        skip_after_clear_seconds=skip_seconds_after_clear,
        estimated_fps=fps,
    )

    if not cap.isOpened():
        logger.error("Failed to open video capture", url=url)
        return

    buf_frames = collections.deque(maxlen=T)
    frame_idx = 0
    skip_left = 0
    status_text = "waiting..."
    status_color = (255, 255, 255)
    last_result_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.warning("Stream ended or disconnected, retrying...")
            cap.release()
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(model, "clean_activation_buffers"):
                model.clean_activation_buffers()
            continue

        ts = time.time()
        buf_frames.append(preprocess_frame(frame).clone())
        frame_idx += 1

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        cv2.putText(
            frame,
            ts_str,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if skip_left > 0:
            skip_left -= 1
        elif (
            len(buf_frames) == T
            and (frame_idx % STRIDE == 0)
            and not clip_queue.full()
        ):
            clip_queue.put([f.clone() for f in buf_frames])

        with result_lock:
            latest = result_state["data"]

        if latest and latest["timestamp"] > last_result_ts:
            last_result_ts = latest["timestamp"]
            prob_list = latest["prob_list"]
            positive_prob = latest["positive_prob"]
            top_idx = latest["top_idx"]
            top_prob = latest["top_prob"]
            top_label = (
                class_names_list[top_idx]
                if class_names_list and top_idx < len(class_names_list)
                else f"class {top_idx}"
            )

            status_text = f"{top_label}: {top_prob:.2f}"
            status_color = (0, 255, 0)

            logger.info(
                "clip_probs",
                probs=prob_list,
                positive_idx=positive_idx,
                positive_label=positive_label_display,
                positive_prob=positive_prob,
                top_label=top_label,
                top_prob=top_prob,
            )

            if positive_prob >= detection_threshold:
                status_text = f"{positive_label_display.upper()} {positive_prob:.2f}"
                status_color = (0, 0, 255)
            elif positive_prob <= idle_threshold:
                skip_left = skip_frames_after_clear

        cv2.putText(
            frame,
            status_text,
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )
        if skip_left > 0:
            cv2.putText(
                frame,
                f"skipping: {skip_left / fps:.1f}s",
                (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Stream", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    stop_event.set()
    try:
        clip_queue.put_nowait(None)
    except queue.Full:
        cleared = False
        while not cleared:
            try:
                clip_queue.get_nowait()
                clip_queue.task_done()
            except queue.Empty:
                cleared = True
        clip_queue.put_nowait(None)
    worker.join(timeout=1.0)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Stopping camera loop", url=url)
