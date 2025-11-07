import time, json, sys, traceback
import os
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
                # Try different initialization methods for RF-DETR
                rf_model = None
                
                # Method 1: RFDETRBase class
                if hasattr(rfdetr, "RFDETRBase"):
                    rf_model = rfdetr.RFDETRBase()
                    print("[VISION] RF-DETR initialized with RFDETRBase()")
                # Method 2: load_model function
                elif hasattr(rfdetr, "load_model"):
                    rf_model = rfdetr.load_model()
                    print("[VISION] RF-DETR initialized with load_model()")
                # Method 3: RFDetr class with from_pretrained
                elif hasattr(rfdetr, "RFDetr") and hasattr(getattr(rfdetr, "RFDetr"), "from_pretrained"):
                    rf_model = getattr(rfdetr, "RFDetr").from_pretrained()
                    print("[VISION] RF-DETR initialized with RFDetr.from_pretrained()")
                # Method 4: Direct module reference (fallback)
                else:
                    rf_model = rfdetr
                    print("[VISION] RF-DETR using module reference (fallback)")

                self._rf_model = rf_model
                self._backend = "rfdetr"
                
                # Optimiser RF-DETR pour l'inférence si disponible
                try:
                    if hasattr(self._rf_model, "optimize_for_inference"):
                        self._rf_model.optimize_for_inference()
                        print("[VISION] RF-DETR optimized for inference")
                    elif hasattr(self._rf_model, "model") and hasattr(self._rf_model.model, "optimize_for_inference"):
                        self._rf_model.model.optimize_for_inference()
                        print("[VISION] RF-DETR model optimized for inference")
                except Exception as e:
                    print(f"[VISION] Could not optimize RF-DETR (non-critical): {e}")
                
                self._ready = True
                print("[VISION] RF-DETR (pip) backend enabled")
                return
            except Exception as e:
                print(f"[VISION] RF-DETR (pip) unavailable: {e}")
                import traceback
                traceback.print_exc()
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
                # Load image as PIL Image for rfdetr
                img = Image.open(image_path).convert("RGB")
                
                # Optimiser la taille de l'image pour RF-DETR (accélère la détection)
                # RF-DETR fonctionne bien avec des images plus petites
                max_size = 800
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Try common prediction signatures
                preds = None
                if hasattr(self._rf_model, "predict"):
                    # Try with PIL Image first, then fallback to path
                    try:
                        preds = self._rf_model.predict(img, threshold=0.3)
                    except:
                        preds = self._rf_model.predict(str(image_path), threshold=0.3)
                elif hasattr(rfdetr, "predict"):
                    try:
                        preds = rfdetr.predict(img, threshold=0.3)
                    except:
                        preds = rfdetr.predict(str(image_path), threshold=0.3)
                elif hasattr(self._rf_model, "infer"):
                    try:
                        preds = self._rf_model.infer(img)
                    except:
                        preds = self._rf_model.infer(str(image_path))
                else:
                    raise RuntimeError("No predict/infer entrypoint found in rfdetr package")

                if preds is None:
                    print("[VISION] RF-DETR predict returned None")
                    return []

                dets = []
                
                # Case 1: RF-DETR object format with attributes (class_id, confidence, bbox)
                if hasattr(preds, "class_id") and hasattr(preds, "confidence"):
                    try:
                        import numpy as np
                        # Get class names if available
                        class_names = None
                        if hasattr(rfdetr, "util") and hasattr(rfdetr.util, "coco_classes"):
                            class_names = rfdetr.util.coco_classes.COCO_CLASSES
                        elif hasattr(rfdetr, "COCO_CLASSES"):
                            class_names = rfdetr.COCO_CLASSES
                        
                        # Extract arrays (could be numpy arrays or tensors)
                        class_ids = preds.class_id
                        confidences = preds.confidence
                        
                        # Get bboxes (could be bbox, boxes, or xyxy)
                        bboxes = None
                        if hasattr(preds, "bbox"):
                            bboxes = preds.bbox
                        elif hasattr(preds, "boxes"):
                            bboxes = preds.boxes
                        elif hasattr(preds, "xyxy"):
                            bboxes = preds.xyxy
                        
                        # Convert to numpy if needed
                        if torch is not None and hasattr(class_ids, "cpu"):
                            class_ids = class_ids.cpu().numpy()
                            confidences = confidences.cpu().numpy()
                            if bboxes is not None:
                                bboxes = bboxes.cpu().numpy()
                        elif hasattr(class_ids, "numpy"):
                            class_ids = class_ids.numpy()
                            confidences = confidences.numpy()
                            if bboxes is not None:
                                bboxes = bboxes.numpy()
                        
                        # Ensure we have arrays
                        if not isinstance(class_ids, np.ndarray):
                            class_ids = np.array(class_ids)
                        if not isinstance(confidences, np.ndarray):
                            confidences = np.array(confidences)
                        if bboxes is not None and not isinstance(bboxes, np.ndarray):
                            bboxes = np.array(bboxes)
                        
                        # Process each detection
                        num_detections = len(class_ids) if hasattr(class_ids, "__len__") else 0
                        for i in range(num_detections):
                            class_id = int(class_ids[i])
                            confidence = float(confidences[i])
                            
                            # Get label
                            if class_names and class_id < len(class_names):
                                label = class_names[class_id]
                            else:
                                label = f"class_{class_id}"
                            
                            # Get bbox
                            if bboxes is not None:
                                if len(bboxes.shape) == 2:
                                    box = bboxes[i]
                                else:
                                    box = bboxes
                                # Handle different bbox formats: [x1, y1, x2, y2] or [x, y, w, h]
                                if len(box) >= 4:
                                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                                    # If format is [x, y, w, h], convert to [x1, y1, x2, y2]
                                    if x2 < x1 or y2 < y1:
                                        x2 = x1 + x2
                                        y2 = y1 + y2
                                else:
                                    continue
                            else:
                                continue
                            
                            dets.append({
                                "label": label,
                                "score": confidence,
                                "box": [x1, y1, x2, y2],
                            })
                        
                        if dets:
                            print(f"[VISION] RF-DETR detected {len(dets)} objects")
                        return dets
                    except Exception as e:
                        print(f"[VISION] RF-DETR attribute parsing error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Case 2: list of dicts with keys label/score/box
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

                # Unknown format - print debug info
                print(f"[VISION] RF-DETR predict returned unknown format: {type(preds)}")
                print(f"[VISION] Preds attributes: {dir(preds) if hasattr(preds, '__dict__') else 'N/A'}")
                return []
            except Exception as e:
                print(f"[VISION] RF-DETR (pip) infer error: {e}")
                import traceback
                traceback.print_exc()
                return []

        # HF DETR backend
        if self._backend == "hf_detr":
            try:
                img = Image.open(image_path).convert("RGB")
                inputs = self._processor(images=img, return_tensors="pt")
                # Utiliser inference_mode() pour de meilleures performances
                with torch.inference_mode():
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
                base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
                
                print(f"[VISION] Loading processor from {base_model_id}...")
                self._processor = AutoProcessor.from_pretrained(base_model_id)
                print(f"[VISION] Processor loaded successfully")
                
                print(f"[VISION] Loading model from {base_model_id}...")
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    base_model_id,
                    device_map="auto",
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._model.eval()
                
                # Optimisations PyTorch pour l'inférence
                if torch is not None and torch.cuda.is_available():
                    # Activer cudnn benchmark pour accélérer les convolutions
                    torch.backends.cudnn.benchmark = True
                    print("[VISION] CuDNN benchmark enabled")
                    
                    # Compiler le modèle avec torch.compile si disponible (PyTorch 2.0+)
                    try:
                        if hasattr(torch, "compile"):
                            print("[VISION] Compiling model with torch.compile...")
                            self._model = torch.compile(self._model, mode="reduce-overhead")
                            print("[VISION] Model compiled successfully")
                    except Exception as e:
                        print(f"[VISION] torch.compile not available or failed (non-critical): {e}")
                
                self._ready = True
                print("[VISION] Qwen VLM loaded and ready")
            except Exception as e:
                print(f"[VISION] Qwen VLM unavailable: {e}")
                import traceback
                traceback.print_exc()
                self._ready = False

    def describe(self, image_path: Path, detections):
        if not self._ready:
            print("[VISION] SceneDescriber not ready, skipping description")
            return ""
        
        print(f"[VISION] Starting description for {image_path.name}")
        
        # Étape 1: Charger l'image et optimiser sa taille si nécessaire
        try:
            img = Image.open(image_path).convert("RGB")
            original_size = img.size
            
            # Réduire la taille de l'image si elle est trop grande (accélère le traitement)
            # Qwen2.5-VL accepte jusqu'à ~1344px, mais 768px est optimal pour vitesse/qualité
            max_size = 768
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"[VISION] Image resized from {original_size} to {img.size} for faster processing")
            else:
                print(f"[VISION] Image loaded: {img.size}")
        except Exception as e:
            print(f"[VISION] Image open error: {e}")
            traceback.print_exc()
            return ""

        # Étape 2: Construire le prompt avec les détections
        detection_info = ""
        if detections and len(detections) > 0:
            print(f"[VISION] Processing {len(detections)} detections")
            object_counts = {}
            for det in detections:
                label = det.get("label", "unknown")
                score = det.get("score", 0.0)
                if score >= 0.5:
                    object_counts[label] = object_counts.get(label, 0) + 1
            
            if object_counts:
                objects_list = []
                for obj_type, count in object_counts.items():
                    if count == 1:
                        objects_list.append(f"1 {obj_type}")
                    else:
                        objects_list.append(f"{count} {obj_type}s")
                detection_info = f"Objets détectés : {', '.join(objects_list)}. "
                print(f"[VISION] Detection info: {detection_info}")
        
        # Construire le prompt simple
        prompt_text = f"{detection_info}Décris cette scène en une phrase : identifie le lieu (salle de classe, amphithéâtre, bibliothèque, couloir, extérieur) et l'activité principale."
        print(f"[VISION] Prompt: {prompt_text[:100]}...")
        
        # Étape 3: Préparer les messages
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            print("[VISION] Messages prepared")
        except Exception as e:
            print(f"[VISION] Error preparing messages: {e}")
            traceback.print_exc()
            return ""

        # Étape 4: Appliquer le template de chat
        try:
            text_prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"[VISION] Chat template applied, prompt length: {len(text_prompt)}")
        except Exception as e:
            print(f"[VISION] Error applying chat template: {e}")
            traceback.print_exc()
            return ""

        # Étape 5: Traiter l'image et le texte
        try:
            inputs = self._processor(
                text=[text_prompt],
                images=[img],
                return_tensors="pt"
            )
            print(f"[VISION] Inputs processed, device: {self._model.device}")
            
            # Déplacer vers le device du modèle
            if torch is not None:
                inputs = {k: v.to(self._model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            print("[VISION] Inputs moved to device")
        except Exception as e:
            print(f"[VISION] Error processing inputs: {e}")
            traceback.print_exc()
            return ""

        # Étape 6: Générer la réponse
        try:
            print("[VISION] Starting generation...")
            # Utiliser inference_mode() au lieu de no_grad() pour de meilleures performances
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=100,  # Augmenté pour éviter que le caption soit coupé
                    do_sample=False,
                    use_cache=True,  # Utiliser le cache pour accélérer la génération
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )
            print(f"[VISION] Generation complete, output shape: {out.shape}")
        except Exception as e:
            print(f"[VISION] Error during generation: {e}")
            traceback.print_exc()
            return ""

        # Étape 7: Décoder la réponse
        try:
            input_len = inputs["input_ids"].shape[1]
            print(f"[VISION] Input length: {input_len}, Output length: {out.shape[1]}")
            
            gen_ids = out[0][input_len:]
            print(f"[VISION] Generated token IDs: {gen_ids.tolist()[:10]}...")
            
            text = self._processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = text.strip()
            print(f"[VISION] Decoded text: '{text}'")
            
            if not text:
                print("[VISION] WARNING: Generated text is empty!")
            
            return text
        except Exception as e:
            print(f"[VISION] Error decoding output: {e}")
            traceback.print_exc()
            return ""


def vision_loop(socketio, period_seconds=60, stop_event=None):
    if not _HAS_CV2:
        print("[VISION] OpenCV indisponible: la boucle vision est desactivee (no-op)")
        return
    
    print("[VISION] Initializing vision components...")
    cam = CameraCapture()
    det = Detector()
    vlm = SceneDescriber()
    vision_results_dir.mkdir(parents=True, exist_ok=True)
    print("[VISION] Vision components initialized")
    
    while True:
        # Vérifier le signal d'arrêt
        if stop_event is not None and stop_event.is_set():
            print("[VISION] Stop signal received, exiting vision loop")
            break
        
        # Étape 1: Capturer une image
        ts = int(time.time())
        img_path = images_raw_dir / f"frame_{ts}.jpg"
        print(f"[VISION] === Starting vision cycle at {ts} ===")
        
        try:
            # Étape 1: Capturer l'image
            print(f"[VISION] Step 1: Capturing image to {img_path.name}")
            cam.snapshot(img_path)
            print(f"[VISION] Image captured successfully")
            
            # Démarrer le chronomètre après la capture de l'image
            start_time = time.time()
            
            # Étape 2: Détecter les objets
            print(f"[VISION] Step 2: Running object detection...")
            detections = det.infer(img_path)
            print(f"[VISION] Detection complete: {len(detections)} objects found")
            
            # Étape 3: Générer la description et attendre qu'elle soit prête
            print(f"[VISION] Step 3: Generating scene description...")
            caption = vlm.describe(img_path, detections)
            
            # Vérifier que la caption a été générée
            if not caption or caption.strip() == "":
                print(f"[VISION] WARNING: Caption is empty, skipping save and display")
                print(f"[VISION] === Vision cycle incomplete (no caption) ===\n")
                continue
            
            print(f"[VISION] Description generated: '{caption[:50]}...'")
            
            # Calculer le temps de traitement
            processing_time = time.time() - start_time
            print(f"[VISION] Processing time: {processing_time:.2f} seconds")
            
            # Étape 4: Préparer le résultat (seulement si caption est prête)
            image_url = f"/vision/image/{img_path.name}"
            result = {
                "ts": ts,
                "image_path": str(img_path),
                "image_url": image_url,
                "detections": detections,
                "caption": caption,
                "processing_time": round(processing_time, 2)
            }
            
            # Étape 5: Sauvegarder le résultat
            result_file = vision_results_dir / f"{ts}.json"
            result_file.write_text(json.dumps(result, ensure_ascii=False))
            print(f"[VISION] Result saved to {result_file.name}")
            
            # Étape 6: Publier les messages et afficher
            message_queue_handler.publish("Vision_topic", {
                "image_path": str(img_path),
                "detections": json.dumps(detections, ensure_ascii=False)
            })
            socketio.emit("vision_detection", result)
            print(f"[VISION] Messages published and displayed")
            
            print(f"[VISION] === Vision cycle complete ===\n")
            
        except Exception as e:
            print(f"[VISION] ERROR in vision cycle: {e}")
            traceback.print_exc()
        
        # Attendre avant la prochaine itération
        print(f"[VISION] Waiting {period_seconds} seconds before next cycle...")
        if stop_event is not None:
            if stop_event.wait(period_seconds):
                print("[VISION] Stop signal received during wait, exiting vision loop")
                break
        else:
            time.sleep(period_seconds)