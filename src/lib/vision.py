import time, json, sys, traceback
from pathlib import Path
from .file_manager import images_raw_dir, images_annotated_dir, vision_results_dir
from .message_queue import message_queue_handler

# Try to import OpenCV; if unavailable (e.g., Python 3.13 wheels), run in stub mode
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
    print(f"[VISION] OpenCV {cv2.__version__} chargé avec succès")
except Exception as e:
    cv2 = None  # type: ignore
    _HAS_CV2 = False
    print(f"[VISION] Erreur lors de l'import d'OpenCV: {type(e).__name__}: {e}")
    print(f"[VISION] Python version: {sys.version}")
    traceback.print_exc()

# Optional ML deps (RF-DETR and Qwen VLM)
try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    _HAS_PIL = False

# Optional RF-DETR from pip (third-party package)
try:
    import rfdetr  # type: ignore
    _HAS_RFDETR = True
except Exception:
    rfdetr = None  # type: ignore
    _HAS_RFDETR = False

try:
    import torch  # type: ignore
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )  # type: ignore
    _HAS_TRANSFORMERS = True
except Exception:
    AutoImageProcessor = None  # type: ignore
    AutoModelForObjectDetection = None  # type: ignore
    AutoProcessor = None  # type: ignore
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore
    torch = None  # type: ignore
    _HAS_TRANSFORMERS = False

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
        self._ready = False
        self._processor = None
        self._model = None
        self._backend = None

        # Prefer RF-DETR pip package if available
        if _HAS_RFDETR and _HAS_PIL:
            try:
                # Heuristics: try common entry points
                rf_model = None
                if hasattr(rfdetr, "load_model"):
                    rf_model = rfdetr.load_model()
                elif hasattr(rfdetr, "RFDetr") and hasattr(getattr(rfdetr, "RFDetr"), "from_pretrained"):
                    rf_model = getattr(rfdetr, "RFDetr").from_pretrained()
                else:
                    # As a last resort, keep module reference; we'll try function-level predict later
                    rf_model = rfdetr

                self._rf_model = rf_model
                self._backend = "rfdetr"
                self._ready = True
                print("[VISION] RF-DETR (pip) backend enabled")
                return
            except Exception as e:
                print(f"[VISION] RF-DETR (pip) unavailable: {e}")
                self._backend = None
                self._ready = False

        # Fallback to a public DETR checkpoint via transformers
        if _HAS_TRANSFORMERS and _HAS_PIL:
            try:
                model_id = "facebook/detr-resnet-50"
                self._processor = AutoImageProcessor.from_pretrained(model_id)
                self._model = AutoModelForObjectDetection.from_pretrained(model_id)
                self._model.eval()
                self._backend = "hf_detr"
                self._ready = True
                print("[VISION] DETR (HF) backend enabled")
            except Exception as e:
                print(f"[VISION] DETR (HF) unavailable: {e}")
                self._backend = None
                self._ready = False

    def infer(self, image_path: Path):
        if not self._ready:
            return []

        # RF-DETR pip backend
        if self._backend == "rfdetr":
            try:
                # Try common prediction signatures
                if hasattr(self._rf_model, "predict"):
                    preds = self._rf_model.predict(str(image_path))
                elif hasattr(rfdetr, "predict"):
                    preds = rfdetr.predict(str(image_path))
                elif hasattr(self._rf_model, "infer"):
                    preds = self._rf_model.infer(str(image_path))
                else:
                    raise RuntimeError("No predict/infer entrypoint found in rfdetr package")

                dets = []
                # Normalize a few likely shapes
                # Case: list of dicts with keys label/score/box
                if isinstance(preds, list) and preds and isinstance(preds[0], dict):
                    for p in preds:
                        box = p.get("box") or p.get("bbox") or p.get("boxes")
                        label = p.get("label") or p.get("class") or p.get("category")
                        score = p.get("score") or p.get("confidence") or p.get("prob")
                        if box is None or label is None or score is None:
                            continue
                        if isinstance(box, dict):
                            # possibly x1,y1,x2,y2
                            b = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
                            if None not in b:
                                box = b
                        dets.append({
                            "label": str(label),
                            "score": float(score),
                            "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        })
                    return dets

                # Unknown format
                print("[VISION] RF-DETR predict returned unknown format; ignoring")
                return []
            except Exception as e:
                print(f"[VISION] RF-DETR (pip) infer error: {e}")
                return []

        # HF DETR backend
        if self._backend == "hf_detr":
            try:
                img = Image.open(image_path).convert("RGB")
                inputs = self._processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._model(**inputs)
                target_sizes = torch.tensor([img.size[::-1]])
                results = self._processor.post_process_object_detection(
                    outputs, threshold=0.5, target_sizes=target_sizes
                )[0]
                dets = []
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                    dets.append({
                        "label": self._model.config.id2label.get(int(label), str(int(label))),
                        "score": float(score),
                        "box": [x1, y1, x2, y2],
                    })
                return dets
            except Exception as e:
                print(f"[VISION] DETR (HF) infer error: {e}")
                return []

        return []


class SceneDescriber:
    def __init__(self):
        self._ready = False
        self._processor = None
        self._model = None
        if _HAS_TRANSFORMERS and _HAS_PIL:
            try:
                model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
                self._processor = AutoProcessor.from_pretrained(model_id)
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
                self._model.eval()
                self._ready = True
                print("[VISION] Qwen VLM loaded")
            except Exception as e:
                print(f"[VISION] Qwen VLM unavailable: {e}")
                self._ready = False

    def describe(self, image_path: Path, detections):
        if not self._ready:
            return ""
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[VISION] image open error: {e}")
            return ""

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Tu es une étudiante et tu dois analyser visuellement l’image et décrire la scène en une seule phrase claire et précise. Ta description doit refléter une compréhension contextuelle complète : identifie le lieu exact (ex. salle de classe, amphithéâtre, bibliothèque, couloir, extérieur…), les types de personnes présentes (étudiants, professeur, intervenant, public, etc.) et leur activité principale. L'objectif est que tu comprennes le contexte dans lequel tu es placée."},
                    ],
                }
            ]
            text_prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = self._processor(
                text=[text_prompt],
                images=[img],
                return_tensors="pt"
            ).to(self._model.device)

            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    return_dict_in_generate=True
                )

            # Decode only the newly generated tokens (exclude the prompt/input)
            input_len = inputs["input_ids"].shape[1]
            seq = out.sequences[0]
            gen_ids = seq[input_len:]
            text = self._processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            return (text or "").strip()

        except Exception as e:
            print(f"[VISION] describe (Qwen) error: {e}")
            return ""


def vision_loop(socketio, period_seconds=60, stop_event=None):
    if not _HAS_CV2:
        print("[VISION] OpenCV indisponible: la boucle vision est desactivee (no-op)")
        return
    cam = CameraCapture()
    det = Detector()
    vlm = SceneDescriber()
    vision_results_dir.mkdir(parents=True, exist_ok=True)
    while True:
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            print("[VISION] stop signal received, exiting vision loop")
            break
        ts = int(time.time())
        img_path = images_raw_dir / f"frame_{ts}.jpg"
        try:
            cam.snapshot(img_path)
            detections = det.infer(img_path)
            caption = vlm.describe(img_path, detections)
            image_url = f"/vision/image/{img_path.name}"
            result = {"ts": ts, "image_path": str(img_path), "image_url": image_url, "detections": detections, "caption": caption}
            (vision_results_dir / f"{ts}.json").write_text(json.dumps(result, ensure_ascii=False))
            message_queue_handler.publish("Vision_topic", {"image_path": str(img_path), "detections": json.dumps(detections, ensure_ascii=False)})
            socketio.emit("vision_detection", result)
        except Exception as e:
            print(f"[VISION] error: {e}")
        # Allow responsive stopping during wait
        if stop_event is not None:
            # emulate Event.wait without blocking when not provided
            if stop_event.wait(period_seconds):
                print("[VISION] stop signal received during wait, exiting vision loop")
                break
        else:
            time.sleep(period_seconds)