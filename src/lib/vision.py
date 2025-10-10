import time, json
from pathlib import Path
from .file_manager import images_raw_dir, images_annotated_dir, vision_results_dir
from .message_queue import message_queue_handler

# Try to import OpenCV; if unavailable (e.g., Python 3.13 wheels), run in stub mode
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

class CameraCapture:
    def __init__(self, device_index=0, width=1280, height=720):
        if not _HAS_CV2:
            self.cap = None
            return
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def snapshot(self, out_path: Path) -> Path:
        if not _HAS_CV2 or self.cap is None:
            raise RuntimeError("OpenCV non disponible: vision en mode inactif")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera capture failed")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        return out_path

class Detector:
    def __init__(self):
        # TODO: charger RF-DETR léger ou autre modèle
        pass

    def infer(self, image_path: Path):
        # TODO: renvoyer [{"label": "laptop", "score": 0.93, "box": [x1,y1,x2,y2]}, ...]
        return []

def vision_loop(socketio, period_seconds=120):
    if not _HAS_CV2:
        print("[VISION] OpenCV indisponible: la boucle vision est desactivee (no-op)")
        return
    cam = CameraCapture()
    det = Detector()
    vision_results_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ts = int(time.time())
        img_path = images_raw_dir / f"frame_{ts}.jpg"
        try:
            cam.snapshot(img_path)
            detections = det.infer(img_path)
            result = {"ts": ts, "image_path": str(img_path), "detections": detections}
            (vision_results_dir / f"{ts}.json").write_text(json.dumps(result, ensure_ascii=False))
            message_queue_handler.publish("Vision_topic", {"image_path": str(img_path), "detections": json.dumps(detections, ensure_ascii=False)})
            socketio.emit("vision_detection", result)
        except Exception as e:
            print(f"[VISION] error: {e}")
        time.sleep(period_seconds)