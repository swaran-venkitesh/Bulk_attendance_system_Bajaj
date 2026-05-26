# # # # """
# # # # SCRFD Face Detector - Optimized for Jetson Nano
# # # # Handles 30-40 faces with batch processing
# # # # Supports dual camera threading
# # # # """

# # # # import cv2
# # # # import numpy as np
# # # # import onnxruntime as ort
# # # # import threading
# # # # from queue import Queue

# # # # class FaceDetector:
# # # #     def __init__(self, model_path='models/scrfd_10g_bnkps.onnx', conf_threshold=0.5):
# # # #         """
# # # #         SCRFD Face Detector optimized for Jetson Nano
# # # #         Args:
# # # #             model_path: Path to SCRFD ONNX model
# # # #             conf_threshold: Detection confidence threshold
# # # #         """
# # # #         print(f"[DETECTOR] Loading SCRFD Face Detector for Jetson Nano...")
        
# # # #         try:
# # # #             # Use CUDA execution provider for Jetson Nano GPU acceleration
# # # #             providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# # # #             self.session = ort.InferenceSession(model_path, providers=providers)
            
# # # #             self.conf_threshold = conf_threshold
# # # #             self.input_size = (640, 640)  # SCRFD input size
# # # #             self.nms_threshold = 0.4
            
# # # #             print(f"[DETECTOR] SCRFD loaded successfully")
# # # #             print(f"[DETECTOR] Providers: {self.session.get_providers()}")
# # # #             print(f"[DETECTOR] Input size: {self.input_size}")
            
# # # #         except Exception as e:
# # # #             print(f"[ERROR] Failed to load SCRFD model: {e}")
# # # #             print("[INFO] Falling back to YOLOv11n...")
# # # #             self._fallback_to_yolo()
    
# # # #     def _fallback_to_yolo(self):
# # # #         """Fallback to YOLOv11n if SCRFD fails"""
# # # #         try:
# # # #             from ultralytics import YOLO
# # # #             self.session = YOLO('models/yolo11n-face.pt')
# # # #             self.model_type = 'yolo'
# # # #             print("[DETECTOR] YOLOv11n loaded as fallback")
# # # #         except Exception as e:
# # # #             raise Exception(f"Both SCRFD and YOLO failed: {e}")
    
# # # #     def preprocess_scrfd(self, image):
# # # #         """Preprocess image for SCRFD model"""
# # # #         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #         img_resized = cv2.resize(img, self.input_size)
        
# # # #         # Normalize
# # # #         img_normalized = (img_resized.astype(np.float32) - 127.5) / 128.0
# # # #         img_transposed = np.transpose(img_normalized, (2, 0, 1))
# # # #         img_batch = np.expand_dims(img_transposed, axis=0)
        
# # # #         return img_batch
    
# # # #     def postprocess_scrfd(self, outputs, orig_shape):
# # # #         """Post-process SCRFD outputs"""
# # # #         h, w = orig_shape[:2]
# # # #         scale_h = h / self.input_size[0]
# # # #         scale_w = w / self.input_size[1]
        
# # # #         faces = []
        
# # # #         # SCRFD outputs: boxes, scores, landmarks
# # # #         boxes = outputs[0][0]  # [N, 4]
# # # #         scores = outputs[1][0]  # [N, 1]
        
# # # #         for box, score in zip(boxes, scores):
# # # #             if score[0] < self.conf_threshold:
# # # #                 continue
            
# # # #             # Scale boxes back to original image
# # # #             x1 = int(box[0] * scale_w)
# # # #             y1 = int(box[1] * scale_h)
# # # #             x2 = int(box[2] * scale_w)
# # # #             y2 = int(box[3] * scale_h)
            
# # # #             # Clamp to image boundaries
# # # #             x1, y1 = max(0, x1), max(0, y1)
# # # #             x2, y2 = min(w, x2), min(h, y2)
            
# # # #             faces.append((x1, y1, x2, y2, float(score[0])))
        
# # # #         return faces
    
# # # #     def detect_faces(self, frame):
# # # #         """
# # # #         Detect all faces in frame
# # # #         Returns: List of (x1, y1, x2, y2, confidence)
# # # #         """
# # # #         if hasattr(self, 'model_type') and self.model_type == 'yolo':
# # # #             return self._detect_yolo(frame)
        
# # # #         # SCRFD detection
# # # #         input_blob = self.preprocess_scrfd(frame)
# # # #         input_name = self.session.get_inputs()[0].name
# # # #         outputs = self.session.run(None, {input_name: input_blob})
        
# # # #         faces = self.postprocess_scrfd(outputs, frame.shape)
        
# # # #         # Apply NMS to remove overlapping boxes
# # # #         faces = self._apply_nms(faces)
        
# # # #         return faces
    
# # # #     def _detect_yolo(self, frame):
# # # #         """YOLO detection fallback"""
# # # #         results = self.session(frame, conf=self.conf_threshold, verbose=False)
        
# # # #         faces = []
# # # #         if len(results) > 0 and results[0].boxes is not None:
# # # #             boxes = results[0].boxes
# # # #             for box in boxes:
# # # #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
# # # #                 conf = float(box.conf[0].cpu().numpy())
# # # #                 faces.append((x1, y1, x2, y2, conf))
        
# # # #         return faces
    
# # # #     def _apply_nms(self, faces):
# # # #         """Apply Non-Maximum Suppression"""
# # # #         if len(faces) == 0:
# # # #             return []
        
# # # #         boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in faces])
# # # #         scores = np.array([conf for _, _, _, _, conf in faces])
        
# # # #         # Calculate areas
# # # #         x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
# # # #         areas = (x2 - x1) * (y2 - y1)
        
# # # #         # Sort by confidence
# # # #         order = scores.argsort()[::-1]
        
# # # #         keep = []
# # # #         while order.size > 0:
# # # #             i = order[0]
# # # #             keep.append(i)
            
# # # #             # Calculate IoU with remaining boxes
# # # #             xx1 = np.maximum(x1[i], x1[order[1:]])
# # # #             yy1 = np.maximum(y1[i], y1[order[1:]])
# # # #             xx2 = np.minimum(x2[i], x2[order[1:]])
# # # #             yy2 = np.minimum(y2[i], y2[order[1:]])
            
# # # #             w = np.maximum(0.0, xx2 - xx1)
# # # #             h = np.maximum(0.0, yy2 - yy1)
# # # #             inter = w * h
            
# # # #             iou = inter / (areas[i] + areas[order[1:]] - inter)
            
# # # #             # Keep boxes with IoU < threshold
# # # #             inds = np.where(iou <= self.nms_threshold)[0]
# # # #             order = order[inds + 1]
        
# # # #         return [faces[i] for i in keep]
    
# # # #     def detect_batch(self, frames):
# # # #         """
# # # #         Batch detection for multiple frames (dual camera support)
# # # #         Args:
# # # #             frames: List of frames
# # # #         Returns: List of face detections for each frame
# # # #         """
# # # #         results = []
# # # #         for frame in frames:
# # # #             faces = self.detect_faces(frame)
# # # #             results.append(faces)
# # # #         return results
    
# # # #     def extract_face_crops(self, frame, faces, padding=0.15, min_size=60):
# # # #         """
# # # #         Extract face crops with MINIMAL padding (optimized for side profiles)
        
# # # #         Args:
# # # #             frame: Input frame
# # # #             faces: List of (x1, y1, x2, y2, conf)
# # # #             padding: Padding ratio (15% for side profiles)
# # # #             min_size: Minimum face size in pixels
        
# # # #         Returns: List of face crop dictionaries
# # # #         """
# # # #         crops = []
# # # #         h, w = frame.shape[:2]
        
# # # #         for (x1, y1, x2, y2, conf) in faces:
# # # #             face_w = x2 - x1
# # # #             face_h = y2 - y1
            
# # # #             # Skip tiny faces
# # # #             if face_w < min_size or face_h < min_size:
# # # #                 continue
            
# # # #             # Calculate padding (REDUCED to 15% for better side profile capture)
# # # #             pad_w = int(face_w * padding)
# # # #             pad_h = int(face_h * padding)
            
# # # #             # Apply padding with boundary check
# # # #             x1_crop = max(0, x1 - pad_w)
# # # #             y1_crop = max(0, y1 - pad_h)
# # # #             x2_crop = min(w, x2 + pad_w)
# # # #             y2_crop = min(h, y2 + pad_h)
            
# # # #             # Extract crop
# # # #             crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            
# # # #             if crop.size == 0:
# # # #                 continue
            
# # # #             # Resize small crops
# # # #             crop_h, crop_w = crop.shape[:2]
# # # #             if crop_h < 112 or crop_w < 112:
# # # #                 scale = max(112/crop_h, 112/crop_w)
# # # #                 new_w = int(crop_w * scale)
# # # #                 new_h = int(crop_h * scale)
# # # #                 crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
# # # #             # Calculate face quality score (for filtering)
# # # #             quality_score = self._calculate_quality(crop)
            
# # # #             crops.append({
# # # #                 'image': crop,
# # # #                 'bbox': (x1, y1, x2, y2),
# # # #                 'conf': conf,
# # # #                 'quality': quality_score,
# # # #                 'size': (face_w, face_h)
# # # #             })
        
# # # #         return crops
    
# # # #     def _calculate_quality(self, face_crop):
# # # #         """
# # # #         Calculate face quality score (0-1)
# # # #         Factors: sharpness, brightness, size
# # # #         """
# # # #         if face_crop.size == 0:
# # # #             return 0.0
        
# # # #         # Convert to grayscale
# # # #         gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
# # # #         # Sharpness (Laplacian variance)
# # # #         sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
# # # #         sharpness_score = min(sharpness / 100.0, 1.0)
        
# # # #         # Brightness
# # # #         brightness = np.mean(gray)
# # # #         brightness_score = 1.0 - abs(brightness - 128) / 128.0
        
# # # #         # Size score
# # # #         h, w = face_crop.shape[:2]
# # # #         size_score = min((h * w) / (112 * 112), 1.0)
        
# # # #         # Weighted average
# # # #         quality = (0.5 * sharpness_score + 0.3 * brightness_score + 0.2 * size_score)
        
# # # #         return quality
    
# # # #     def draw_detections(self, frame, faces, color=(0, 255, 0), thickness=2):
# # # #         """Draw bounding boxes on frame"""
# # # #         display_frame = frame.copy()
        
# # # #         for (x1, y1, x2, y2, conf) in faces:
# # # #             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
# # # #             label = f"{conf:.2f}"
# # # #             cv2.putText(display_frame, label, (x1, y1-10),
# # # #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
# # # #         return display_frame


# # # # # ============================================================================
# # # # # DOWNLOAD SCRFD MODEL HELPER
# # # # # ============================================================================

# # # # def download_scrfd_model():
# # # #     """
# # # #     Download SCRFD model - UPDATED WITH WORKING LINKS
# # # #     """
# # # #     import urllib.request
# # # #     import os
    
# # # #     os.makedirs("models", exist_ok=True)
    
# # # #     # Try multiple sources
# # # #     models_to_try = [
# # # #         {
# # # #             'name': 'SCRFD 10G',
# # # #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_10g_bnkps.onnx',
# # # #             'filename': 'scrfd_10g_bnkps.onnx',
# # # #             'size': '~16MB'
# # # #         },
# # # #         {
# # # #             'name': 'SCRFD 2.5G (Lighter)',
# # # #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_2.5g_bnkps.onnx',
# # # #             'filename': 'scrfd_2.5g_bnkps.onnx',
# # # #             'size': '~3MB'
# # # #         }
# # # #     ]
    
# # # #     print("\n" + "="*70)
# # # #     print("SCRFD MODEL DOWNLOAD")
# # # #     print("="*70)
# # # #     print("\nAvailable models:")
# # # #     for idx, model in enumerate(models_to_try, 1):
# # # #         print(f"{idx}. {model['name']} ({model['size']})")
    
# # # #     choice = input("\nSelect model (1-2, default 1): ").strip() or "1"
    
# # # #     try:
# # # #         model_idx = int(choice) - 1
# # # #         selected_model = models_to_try[model_idx]
# # # #     except:
# # # #         selected_model = models_to_try[0]
    
# # # #     model_path = f"models/{selected_model['filename']}"
    
# # # #     if os.path.exists(model_path):
# # # #         print(f"\n[INFO] Model already exists at {model_path}")
# # # #         return model_path
    
# # # #     print(f"\n[INFO] Downloading {selected_model['name']}...")
# # # #     print(f"[INFO] URL: {selected_model['url']}")
# # # #     print("[INFO] This may take a few minutes...")
    
# # # #     try:
# # # #         def download_progress(block_num, block_size, total_size):
# # # #             downloaded = block_num * block_size
# # # #             percent = min(downloaded * 100 / total_size, 100)
# # # #             print(f"\r[DOWNLOAD] Progress: {percent:.1f}%", end='', flush=True)
        
# # # #         urllib.request.urlretrieve(
# # # #             selected_model['url'], 
# # # #             model_path,
# # # #             reporthook=download_progress
# # # #         )
        
# # # #         print(f"\n[SUCCESS] Model downloaded to {model_path}")
# # # #         return model_path
        
# # # #     except Exception as e:
# # # #         print(f"\n[ERROR] Download failed: {e}")
# # # #         print("\n[ALTERNATIVE] Manual download instructions:")
# # # #         print("="*70)
# # # #         print("Option 1: Use Google Drive (Fastest)")
# # # #         print("  1. Download from: https://drive.google.com/drive/folders/1-OXrAg-VaOxQL5v_K2y8g6VRz4VNKB3D")
# # # #         print("  2. Extract and copy .onnx file to models/ folder")
# # # #         print()
# # # #         print("Option 2: Use Hugging Face")
# # # #         print("  1. Visit: https://huggingface.co/SCRFD/SCRFD-10GF/tree/main")
# # # #         print("  2. Download scrfd_10g_bnkps.onnx")
# # # #         print("  3. Place in models/ folder")
# # # #         print()
# # # #         print("Option 3: Use wget (Linux/Mac)")
# # # #         print("  wget https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_10g_bnkps.onnx -P models/")
# # # #         print("="*70)
        
# # # #         return None

# # # # if __name__ == "__main__":
# # # #     # Test detector
# # # #     print("Testing Face Detector...")
    
# # # #     # Download model if needed
# # # #     download_scrfd_model()
    
# # # #     # Initialize detector
# # # #     detector = FaceDetector()
    
# # # #     # Test with webcam
# # # #     cap = cv2.VideoCapture(0)
    
# # # #     print("\nPress 'q' to quit")
    
# # # #     while True:
# # # #         ret, frame = cap.read()
# # # #         if not ret:
# # # #             break
        
# # # #         # Detect faces
# # # #         faces = detector.detect_faces(frame)
        
# # # #         # Draw results
# # # #         annotated = detector.draw_detections(frame, faces)
        
# # # #         # Show FPS and face count
# # # #         cv2.putText(annotated, f"Faces: {len(faces)}", (10, 30),
# # # #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
# # # #         cv2.imshow('Face Detection Test', annotated)
        
# # # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # # #             break
    
# # # #     cap.release()
# # # #     cv2.destroyAllWindows()

# # # """
# # # SCRFD Face Detector - Optimized for Jetson Nano
# # # FIXED: Postprocessing output parsing
# # # Handles 30-40 faces with batch processing
# # # """

# # # import cv2
# # # import numpy as np
# # # import onnxruntime as ort
# # # import threading
# # # from queue import Queue


# # # class FaceDetector:
# # #     def __init__(self, model_path='models/scrfd_10g_bnkps.onnx', conf_threshold=0.5):
# # #         """
# # #         SCRFD Face Detector optimized for Jetson Nano
# # #         Args:
# # #             model_path: Path to SCRFD ONNX model
# # #             conf_threshold: Detection confidence threshold
# # #         """
# # #         print(f"[DETECTOR] Loading SCRFD Face Detector for Jetson Nano...")
        
# # #         try:
# # #             # Use CUDA execution provider for Jetson Nano GPU acceleration
# # #             providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# # #             self.session = ort.InferenceSession(model_path, providers=providers)
            
# # #             self.conf_threshold = conf_threshold
# # #             self.input_size = (640, 640)  # SCRFD input size
# # #             self.nms_threshold = 0.4
# # #             self.model_type = 'scrfd'
            
# # #             # Get output names
# # #             self.output_names = [output.name for output in self.session.get_outputs()]
            
# # #             print(f"[DETECTOR] SCRFD loaded successfully")
# # #             print(f"[DETECTOR] Providers: {self.session.get_providers()}")
# # #             print(f"[DETECTOR] Input size: {self.input_size}")
# # #             print(f"[DETECTOR] Output layers: {len(self.output_names)}")
            
# # #         except Exception as e:
# # #             print(f"[ERROR] Failed to load SCRFD model: {e}")
# # #             print("[INFO] Falling back to YOLOv11n...")
# # #             self._fallback_to_yolo()
    
# # #     def _fallback_to_yolo(self):
# # #         """Fallback to YOLOv11n if SCRFD fails"""
# # #         try:
# # #             from ultralytics import YOLO
# # #             self.session = YOLO('models/yolo11n-face.pt')
# # #             self.model_type = 'yolo'
# # #             print("[DETECTOR] YOLOv11n loaded as fallback")
# # #         except Exception as e:
# # #             raise Exception(f"Both SCRFD and YOLO failed: {e}")
    
# # #     def preprocess_scrfd(self, image):
# # #         """Preprocess image for SCRFD model"""
# # #         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # #         img_resized = cv2.resize(img, self.input_size)
        
# # #         # Normalize
# # #         img_normalized = (img_resized.astype(np.float32) - 127.5) / 128.0
# # #         img_transposed = np.transpose(img_normalized, (2, 0, 1))
# # #         img_batch = np.expand_dims(img_transposed, axis=0)
        
# # #         return img_batch
    
# # #     def postprocess_scrfd(self, outputs, orig_shape):
# # #         """
# # #         Post-process SCRFD outputs - FIXED VERSION
        
# # #         SCRFD outputs multiple scales (stride 8, 16, 32)
# # #         Each scale has: bbox_pred, cls_pred, kps_pred
        
# # #         Output format (9 tensors total):
# # #         - outputs[0:3]: bbox predictions (3 scales)
# # #         - outputs[3:6]: score predictions (3 scales)  
# # #         - outputs[6:9]: keypoint predictions (3 scales)
# # #         """
# # #         h, w = orig_shape[:2]
# # #         scale_h = h / self.input_size[0]
# # #         scale_w = w / self.input_size[1]
        
# # #         faces = []
        
# # #         # SCRFD has 3 detection scales (stride 8, 16, 32)
# # #         num_scales = 3
        
# # #         try:
# # #             # Process each scale
# # #             for idx in range(num_scales):
# # #                 # Get predictions for this scale
# # #                 if len(outputs) >= (num_scales * 2):
# # #                     bbox_pred = outputs[idx]  # Shape: [1, H, W, 4]
# # #                     score_pred = outputs[idx + num_scales]  # Shape: [1, H, W, 1]
                    
# # #                     # Flatten spatial dimensions
# # #                     bbox_pred = bbox_pred.reshape(-1, 4)  # [N, 4]
# # #                     score_pred = score_pred.reshape(-1, 1)  # [N, 1]
                    
# # #                     # Filter by confidence
# # #                     mask = score_pred[:, 0] >= self.conf_threshold
                    
# # #                     if not np.any(mask):
# # #                         continue
                    
# # #                     filtered_boxes = bbox_pred[mask]
# # #                     filtered_scores = score_pred[mask]
                    
# # #                     # Scale boxes to original image size
# # #                     for box, score in zip(filtered_boxes, filtered_scores):
# # #                         x1 = int(box[0] * scale_w)
# # #                         y1 = int(box[1] * scale_h)
# # #                         x2 = int(box[2] * scale_w)
# # #                         y2 = int(box[3] * scale_h)
                        
# # #                         # Clamp to image boundaries
# # #                         x1, y1 = max(0, x1), max(0, y1)
# # #                         x2, y2 = min(w, x2), min(h, y2)
                        
# # #                         # Validate box
# # #                         if x2 > x1 and y2 > y1:
# # #                             faces.append((x1, y1, x2, y2, float(score[0])))
            
# # #         except Exception as e:
# # #             print(f"[DEBUG] SCRFD postprocess error: {e}")
# # #             # Return empty list if parsing fails
# # #             return []
        
# # #         return faces
    
# # #     def detect_faces(self, frame):
# # #         """
# # #         Detect all faces in frame
# # #         Returns: List of (x1, y1, x2, y2, confidence)
# # #         """
# # #         if hasattr(self, 'model_type') and self.model_type == 'yolo':
# # #             return self._detect_yolo(frame)
        
# # #         # SCRFD detection
# # #         input_blob = self.preprocess_scrfd(frame)
# # #         input_name = self.session.get_inputs()[0].name
        
# # #         try:
# # #             outputs = self.session.run(None, {input_name: input_blob})
# # #             faces = self.postprocess_scrfd(outputs, frame.shape)
            
# # #             # Apply NMS to remove overlapping boxes
# # #             faces = self._apply_nms(faces)
            
# # #             return faces
            
# # #         except Exception as e:
# # #             print(f"[ERROR] SCRFD detection failed: {e}")
# # #             return []
    
# # #     def _detect_yolo(self, frame):
# # #         """YOLO detection fallback"""
# # #         results = self.session(frame, conf=self.conf_threshold, verbose=False)
        
# # #         faces = []
# # #         if len(results) > 0 and results[0].boxes is not None:
# # #             boxes = results[0].boxes
# # #             for box in boxes:
# # #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
# # #                 conf = float(box.conf[0].cpu().numpy())
# # #                 faces.append((x1, y1, x2, y2, conf))
        
# # #         return faces
    
# # #     def _apply_nms(self, faces):
# # #         """Apply Non-Maximum Suppression"""
# # #         if len(faces) == 0:
# # #             return []
        
# # #         boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in faces])
# # #         scores = np.array([conf for _, _, _, _, conf in faces])
        
# # #         # Calculate areas
# # #         x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
# # #         areas = (x2 - x1) * (y2 - y1)
        
# # #         # Sort by confidence
# # #         order = scores.argsort()[::-1]
        
# # #         keep = []
# # #         while order.size > 0:
# # #             i = order[0]
# # #             keep.append(i)
            
# # #             # Calculate IoU with remaining boxes
# # #             xx1 = np.maximum(x1[i], x1[order[1:]])
# # #             yy1 = np.maximum(y1[i], y1[order[1:]])
# # #             xx2 = np.minimum(x2[i], x2[order[1:]])
# # #             yy2 = np.minimum(y2[i], y2[order[1:]])
            
# # #             w = np.maximum(0.0, xx2 - xx1)
# # #             h = np.maximum(0.0, yy2 - yy1)
# # #             inter = w * h
            
# # #             iou = inter / (areas[i] + areas[order[1:]] - inter)
            
# # #             # Keep boxes with IoU < threshold
# # #             inds = np.where(iou <= self.nms_threshold)[0]
# # #             order = order[inds + 1]
        
# # #         return [faces[i] for i in keep]
    
# # #     def detect_batch(self, frames):
# # #         """
# # #         Batch detection for multiple frames (dual camera support)
# # #         Args:
# # #             frames: List of frames
# # #         Returns: List of face detections for each frame
# # #         """
# # #         results = []
# # #         for frame in frames:
# # #             faces = self.detect_faces(frame)
# # #             results.append(faces)
# # #         return results
    
# # #     def extract_face_crops(self, frame, faces, padding=0.15, min_size=60):
# # #         """
# # #         Extract face crops with MINIMAL padding (optimized for side profiles)
        
# # #         Args:
# # #             frame: Input frame
# # #             faces: List of (x1, y1, x2, y2, conf)
# # #             padding: Padding ratio (15% for side profiles)
# # #             min_size: Minimum face size in pixels
        
# # #         Returns: List of face crop dictionaries
# # #         """
# # #         crops = []
# # #         h, w = frame.shape[:2]
        
# # #         for (x1, y1, x2, y2, conf) in faces:
# # #             face_w = x2 - x1
# # #             face_h = y2 - y1
            
# # #             # Skip tiny faces
# # #             if face_w < min_size or face_h < min_size:
# # #                 continue
            
# # #             # Calculate padding (REDUCED to 15% for better side profile capture)
# # #             pad_w = int(face_w * padding)
# # #             pad_h = int(face_h * padding)
            
# # #             # Apply padding with boundary check
# # #             x1_crop = max(0, x1 - pad_w)
# # #             y1_crop = max(0, y1 - pad_h)
# # #             x2_crop = min(w, x2 + pad_w)
# # #             y2_crop = min(h, y2 + pad_h)
            
# # #             # Extract crop
# # #             crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            
# # #             if crop.size == 0:
# # #                 continue
            
# # #             # Resize small crops
# # #             crop_h, crop_w = crop.shape[:2]
# # #             if crop_h < 112 or crop_w < 112:
# # #                 scale = max(112/crop_h, 112/crop_w)
# # #                 new_w = int(crop_w * scale)
# # #                 new_h = int(crop_h * scale)
# # #                 crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
# # #             # Calculate face quality score (for filtering)
# # #             quality_score = self._calculate_quality(crop)
            
# # #             crops.append({
# # #                 'image': crop,
# # #                 'bbox': (x1, y1, x2, y2),
# # #                 'conf': conf,
# # #                 'quality': quality_score,
# # #                 'size': (face_w, face_h)
# # #             })
        
# # #         return crops
    
# # #     def _calculate_quality(self, face_crop):
# # #         """
# # #         Calculate face quality score (0-1)
# # #         Factors: sharpness, brightness, size
# # #         """
# # #         if face_crop.size == 0:
# # #             return 0.0
        
# # #         # Convert to grayscale
# # #         gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
# # #         # Sharpness (Laplacian variance)
# # #         sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
# # #         sharpness_score = min(sharpness / 100.0, 1.0)
        
# # #         # Brightness
# # #         brightness = np.mean(gray)
# # #         brightness_score = 1.0 - abs(brightness - 128) / 128.0
        
# # #         # Size score
# # #         h, w = face_crop.shape[:2]
# # #         size_score = min((h * w) / (112 * 112), 1.0)
        
# # #         # Weighted average
# # #         quality = (0.5 * sharpness_score + 0.3 * brightness_score + 0.2 * size_score)
        
# # #         return quality
    
# # #     def draw_detections(self, frame, faces, color=(0, 255, 0), thickness=2):
# # #         """Draw bounding boxes on frame"""
# # #         display_frame = frame.copy()
        
# # #         for (x1, y1, x2, y2, conf) in faces:
# # #             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
# # #             label = f"{conf:.2f}"
# # #             cv2.putText(display_frame, label, (x1, y1-10),
# # #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
# # #         return display_frame


# # # # ============================================================================
# # # # DOWNLOAD SCRFD MODEL HELPER
# # # # ============================================================================

# # # def download_scrfd_model():
# # #     """
# # #     Download SCRFD model - VERIFIED WORKING LINKS (Oct 2025)
# # #     """
# # #     import urllib.request
# # #     import os
    
# # #     os.makedirs("models", exist_ok=True)
    
# # #     models_to_try = [
# # #         {
# # #             'name': 'SCRFD 10G (Best Accuracy)',
# # #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_10g_bnkps.onnx',
# # #             'filename': 'scrfd_10g_bnkps.onnx',
# # #             'size': '~16MB'
# # #         },
# # #         {
# # #             'name': 'SCRFD 2.5G (Balanced)',
# # #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_2.5g_bnkps.onnx',
# # #             'filename': 'scrfd_2.5g_bnkps.onnx',
# # #             'size': '~3MB'
# # #         },
# # #         {
# # #             'name': 'SCRFD 500M (Lightweight)',
# # #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_500m_bnkps.onnx',
# # #             'filename': 'scrfd_500m_bnkps.onnx',
# # #             'size': '~2MB'
# # #         }
# # #     ]
    
# # #     print("\n" + "="*70)
# # #     print("SCRFD MODEL DOWNLOAD")
# # #     print("="*70)
# # #     print("\nAvailable models:")
# # #     for idx, model in enumerate(models_to_try, 1):
# # #         status = "✓" if os.path.exists(f"models/{model['filename']}") else " "
# # #         print(f"{status} {idx}. {model['name']} ({model['size']})")
    
# # #     choice = input("\nSelect model (1-3, default 1): ").strip() or "1"
    
# # #     try:
# # #         model_idx = int(choice) - 1
# # #         selected_model = models_to_try[model_idx]
# # #     except:
# # #         selected_model = models_to_try[0]
    
# # #     model_path = f"models/{selected_model['filename']}"
    
# # #     if os.path.exists(model_path):
# # #         print(f"\n[INFO] Model already exists at {model_path}")
# # #         return model_path
    
# # #     print(f"\n[INFO] Downloading {selected_model['name']}...")
# # #     print(f"[INFO] URL: {selected_model['url']}")
# # #     print("[INFO] This may take a few minutes...")
    
# # #     try:
# # #         def download_progress(block_num, block_size, total_size):
# # #             downloaded = block_num * block_size
# # #             percent = min(downloaded * 100 / total_size, 100)
# # #             print(f"\r[DOWNLOAD] Progress: {percent:.1f}%", end='', flush=True)
        
# # #         urllib.request.urlretrieve(
# # #             selected_model['url'], 
# # #             model_path,
# # #             reporthook=download_progress
# # #         )
        
# # #         print(f"\n[SUCCESS] Model downloaded to {model_path}")
# # #         return model_path
        
# # #     except Exception as e:
# # #         print(f"\n[ERROR] Download failed: {e}")
# # #         print("\n[ALTERNATIVE] Manual download instructions:")
# # #         print("="*70)
# # #         print("Option 1: Direct Download (GitHub)")
# # #         print("  Visit: https://github.com/nttstar/insightface-resources/releases/tag/v0.7")
# # #         print(f"  Download: {selected_model['filename']}")
# # #         print("  Place in: models/ folder")
# # #         print()
# # #         print("Option 2: Use wget (Linux/Mac)")
# # #         print(f"  wget {selected_model['url']} -P models/")
# # #         print()
# # #         print("Option 3: Use curl")
# # #         print(f"  curl -L {selected_model['url']} -o models/{selected_model['filename']}")
# # #         print("="*70)
        
# # #         return None


# # # if __name__ == "__main__":
# # #     # Test detector
# # #     print("Testing Face Detector...")
    
# # #     # Download model if needed
# # #     download_scrfd_model()
    
# # #     # Initialize detector
# # #     detector = FaceDetector()
    
# # #     # Test with webcam
# # #     cap = cv2.VideoCapture(0)
    
# # #     if not cap.isOpened():
# # #         print("[ERROR] Cannot open webcam")
# # #         exit()
    
# # #     print("\nPress 'q' to quit")
    
# # #     fps_start = 0
# # #     fps_counter = 0
# # #     fps = 0
    
# # #     while True:
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break
        
# # #         # Detect faces
# # #         faces = detector.detect_faces(frame)
        
# # #         # Draw results
# # #         annotated = detector.draw_detections(frame, faces)
        
# # #         # Calculate FPS
# # #         fps_counter += 1
# # #         if fps_counter >= 30:
# # #             fps = 30.0 / (cv2.getTickCount() / cv2.getTickFrequency() - fps_start) if fps_start > 0 else 0
# # #             fps_start = cv2.getTickCount() / cv2.getTickFrequency()
# # #             fps_counter = 0
        
# # #         # Show FPS and face count
# # #         cv2.putText(annotated, f"Faces: {len(faces)} | FPS: {fps:.1f}", (10, 30),
# # #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
# # #         cv2.imshow('Face Detection Test', annotated)
        
# # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # #             break
    
# # #     cap.release()
# # #     cv2.destroyAllWindows()

# # """
# # SCRFD Face Detector - Optimized for Jetson Nano
# # FIXED: Proper SCRFD output parsing with dimension handling
# # Handles 30-40 faces with batch processing
# # """

# # import cv2
# # import numpy as np
# # import onnxruntime as ort
# # import threading
# # from queue import Queue


# # class FaceDetector:
# #     def __init__(self, model_path='models/scrfd_10g_bnkps.onnx', conf_threshold=0.5):
# #         """
# #         SCRFD Face Detector optimized for Jetson Nano
# #         Args:
# #             model_path: Path to SCRFD ONNX model
# #             conf_threshold: Detection confidence threshold
# #         """
# #         print(f"[DETECTOR] Loading SCRFD Face Detector for Jetson Nano...")
        
# #         try:
# #             # Use CUDA execution provider for Jetson Nano GPU acceleration
# #             providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# #             self.session = ort.InferenceSession(model_path, providers=providers)
            
# #             self.conf_threshold = conf_threshold
# #             self.input_size = (640, 640)  # SCRFD input size
# #             self.nms_threshold = 0.4
# #             self.model_type = 'scrfd'
            
# #             # Get output names and shapes
# #             self.output_names = [output.name for output in self.session.get_outputs()]
            
# #             print(f"[DETECTOR] SCRFD loaded successfully")
# #             print(f"[DETECTOR] Providers: {self.session.get_providers()}")
# #             print(f"[DETECTOR] Input size: {self.input_size}")
# #             print(f"[DETECTOR] Output layers: {len(self.output_names)}")
            
# #         except Exception as e:
# #             print(f"[ERROR] Failed to load SCRFD model: {e}")
# #             print("[INFO] Falling back to YOLOv11n...")
# #             self._fallback_to_yolo()
    
# #     def _fallback_to_yolo(self):
# #         """Fallback to YOLOv11n if SCRFD fails"""
# #         try:
# #             from ultralytics import YOLO
# #             self.session = YOLO('models/yolo11n-face.pt')
# #             self.model_type = 'yolo'
# #             print("[DETECTOR] YOLOv11n loaded as fallback")
# #         except Exception as e:
# #             raise Exception(f"Both SCRFD and YOLO failed: {e}")
    
# #     def preprocess_scrfd(self, image):
# #         """Preprocess image for SCRFD model"""
# #         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #         img_resized = cv2.resize(img, self.input_size)
        
# #         # Normalize
# #         img_normalized = (img_resized.astype(np.float32) - 127.5) / 128.0
# #         img_transposed = np.transpose(img_normalized, (2, 0, 1))
# #         img_batch = np.expand_dims(img_transposed, axis=0)
        
# #         return img_batch
    
# #     def _generate_anchors(self, stride, feature_map_size):
# #         """Generate anchor centers for a given stride"""
# #         fm_h, fm_w = feature_map_size
# #         anchors = []
        
# #         for i in range(fm_h):
# #             for j in range(fm_w):
# #                 cx = (j + 0.5) * stride
# #                 cy = (i + 0.5) * stride
# #                 anchors.append([cx, cy, stride])
        
# #         return np.array(anchors)
    
# #     def postprocess_scrfd(self, outputs, orig_shape):
# #         """
# #         Post-process SCRFD outputs - COMPLETELY REWRITTEN
        
# #         SCRFD 10G outputs 9 tensors for 3 FPN levels (stride 8, 16, 32):
# #         - outputs[0:3]: score_8, score_16, score_32  (classification)
# #         - outputs[3:6]: bbox_8, bbox_16, bbox_32      (regression)
# #         - outputs[6:9]: kps_8, kps_16, kps_32         (keypoints)
# #         """
# #         h, w = orig_shape[:2]
# #         scale_h = h / self.input_size[0]
# #         scale_w = w / self.input_size[1]
        
# #         faces = []
        
# #         # SCRFD uses 3 FPN levels with different strides
# #         strides = [8, 16, 32]
        
# #         try:
# #             # Process each FPN level
# #             for idx, stride in enumerate(strides):
# #                 # Calculate feature map size
# #                 fm_h = self.input_size[0] // stride
# #                 fm_w = self.input_size[1] // stride
                
# #                 # Get outputs for this level
# #                 score_pred = outputs[idx]      # Shape: [1, fm_h, fm_w, 1] or [1, fm_h*fm_w, 1]
# #                 bbox_pred = outputs[idx + 3]   # Shape: [1, fm_h, fm_w, 4] or [1, fm_h*fm_w, 4]
                
# #                 # Reshape to [N, C] format
# #                 if len(score_pred.shape) == 4:
# #                     # Format: [1, H, W, C]
# #                     score_pred = score_pred.reshape(-1, score_pred.shape[-1])
# #                     bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1])
# #                 elif len(score_pred.shape) == 3:
# #                     # Format: [1, N, C]
# #                     score_pred = score_pred.reshape(-1, score_pred.shape[-1])
# #                     bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1])
                
# #                 # Generate anchors for this level
# #                 anchors = self._generate_anchors(stride, (fm_h, fm_w))
                
# #                 # Apply confidence threshold
# #                 scores = score_pred[:, 0]
# #                 mask = scores >= self.conf_threshold
                
# #                 if not np.any(mask):
# #                     continue
                
# #                 # Filter predictions
# #                 filtered_scores = scores[mask]
# #                 filtered_boxes = bbox_pred[mask]
# #                 filtered_anchors = anchors[mask]
                
# #                 # Decode bounding boxes (distance format)
# #                 # SCRFD predicts: [left, top, right, bottom] distances from anchor
# #                 for score, box, anchor in zip(filtered_scores, filtered_boxes, filtered_anchors):
# #                     cx, cy, s = anchor
                    
# #                     # Decode box (distance to LTRB format)
# #                     x1 = (cx - box[0]) * scale_w
# #                     y1 = (cy - box[1]) * scale_h
# #                     x2 = (cx + box[2]) * scale_w
# #                     y2 = (cy + box[3]) * scale_h
                    
# #                     # Clamp to image boundaries
# #                     x1 = max(0, min(int(x1), w))
# #                     y1 = max(0, min(int(y1), h))
# #                     x2 = max(0, min(int(x2), w))
# #                     y2 = max(0, min(int(y2), h))
                    
# #                     # Validate box
# #                     if x2 > x1 and y2 > y1:
# #                         faces.append((x1, y1, x2, y2, float(score)))
            
# #         except Exception as e:
# #             print(f"[DEBUG] SCRFD postprocess error: {e}")
# #             import traceback
# #             print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
# #             # Print output shapes for debugging
# #             print(f"[DEBUG] Number of outputs: {len(outputs)}")
# #             for i, out in enumerate(outputs):
# #                 print(f"[DEBUG] Output {i} shape: {out.shape}")
            
# #             return []
        
# #         return faces
    
# #     def detect_faces(self, frame):
# #         """
# #         Detect all faces in frame
# #         Returns: List of (x1, y1, x2, y2, confidence)
# #         """
# #         if hasattr(self, 'model_type') and self.model_type == 'yolo':
# #             return self._detect_yolo(frame)
        
# #         # SCRFD detection
# #         input_blob = self.preprocess_scrfd(frame)
# #         input_name = self.session.get_inputs()[0].name
        
# #         try:
# #             outputs = self.session.run(None, {input_name: input_blob})
# #             faces = self.postprocess_scrfd(outputs, frame.shape)
            
# #             # Apply NMS to remove overlapping boxes
# #             faces = self._apply_nms(faces)
            
# #             return faces
            
# #         except Exception as e:
# #             print(f"[ERROR] SCRFD detection failed: {e}")
# #             return []
    
# #     def _detect_yolo(self, frame):
# #         """YOLO detection fallback"""
# #         results = self.session(frame, conf=self.conf_threshold, verbose=False)
        
# #         faces = []
# #         if len(results) > 0 and results[0].boxes is not None:
# #             boxes = results[0].boxes
# #             for box in boxes:
# #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
# #                 conf = float(box.conf[0].cpu().numpy())
# #                 faces.append((x1, y1, x2, y2, conf))
        
# #         return faces
    
# #     def _apply_nms(self, faces):
# #         """Apply Non-Maximum Suppression"""
# #         if len(faces) == 0:
# #             return []
        
# #         boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in faces])
# #         scores = np.array([conf for _, _, _, _, conf in faces])
        
# #         # Calculate areas
# #         x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
# #         areas = (x2 - x1) * (y2 - y1)
        
# #         # Sort by confidence
# #         order = scores.argsort()[::-1]
        
# #         keep = []
# #         while order.size > 0:
# #             i = order[0]
# #             keep.append(i)
            
# #             # Calculate IoU with remaining boxes
# #             xx1 = np.maximum(x1[i], x1[order[1:]])
# #             yy1 = np.maximum(y1[i], y1[order[1:]])
# #             xx2 = np.minimum(x2[i], x2[order[1:]])
# #             yy2 = np.minimum(y2[i], y2[order[1:]])
            
# #             w = np.maximum(0.0, xx2 - xx1)
# #             h = np.maximum(0.0, yy2 - yy1)
# #             inter = w * h
            
# #             iou = inter / (areas[i] + areas[order[1:]] - inter)
            
# #             # Keep boxes with IoU < threshold
# #             inds = np.where(iou <= self.nms_threshold)[0]
# #             order = order[inds + 1]
        
# #         return [faces[i] for i in keep]
    
# #     def detect_batch(self, frames):
# #         """
# #         Batch detection for multiple frames (dual camera support)
# #         Args:
# #             frames: List of frames
# #         Returns: List of face detections for each frame
# #         """
# #         results = []
# #         for frame in frames:
# #             faces = self.detect_faces(frame)
# #             results.append(faces)
# #         return results
    
# #     def extract_face_crops(self, frame, faces, padding=0.15, min_size=60):
# #         """
# #         Extract face crops with MINIMAL padding (optimized for side profiles)
        
# #         Args:
# #             frame: Input frame
# #             faces: List of (x1, y1, x2, y2, conf)
# #             padding: Padding ratio (15% for side profiles)
# #             min_size: Minimum face size in pixels
        
# #         Returns: List of face crop dictionaries
# #         """
# #         crops = []
# #         h, w = frame.shape[:2]
        
# #         for (x1, y1, x2, y2, conf) in faces:
# #             face_w = x2 - x1
# #             face_h = y2 - y1
            
# #             # Skip tiny faces
# #             if face_w < min_size or face_h < min_size:
# #                 continue
            
# #             # Calculate padding (REDUCED to 15% for better side profile capture)
# #             pad_w = int(face_w * padding)
# #             pad_h = int(face_h * padding)
            
# #             # Apply padding with boundary check
# #             x1_crop = max(0, x1 - pad_w)
# #             y1_crop = max(0, y1 - pad_h)
# #             x2_crop = min(w, x2 + pad_w)
# #             y2_crop = min(h, y2 + pad_h)
            
# #             # Extract crop
# #             crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            
# #             if crop.size == 0:
# #                 continue
            
# #             # Resize small crops
# #             crop_h, crop_w = crop.shape[:2]
# #             if crop_h < 112 or crop_w < 112:
# #                 scale = max(112/crop_h, 112/crop_w)
# #                 new_w = int(crop_w * scale)
# #                 new_h = int(crop_h * scale)
# #                 crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
# #             # Calculate face quality score (for filtering)
# #             quality_score = self._calculate_quality(crop)
            
# #             crops.append({
# #                 'image': crop,
# #                 'bbox': (x1, y1, x2, y2),
# #                 'conf': conf,
# #                 'quality': quality_score,
# #                 'size': (face_w, face_h)
# #             })
        
# #         return crops
    
# #     def _calculate_quality(self, face_crop):
# #         """
# #         Calculate face quality score (0-1)
# #         Factors: sharpness, brightness, size
# #         """
# #         if face_crop.size == 0:
# #             return 0.0
        
# #         # Convert to grayscale
# #         gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
# #         # Sharpness (Laplacian variance)
# #         sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
# #         sharpness_score = min(sharpness / 100.0, 1.0)
        
# #         # Brightness
# #         brightness = np.mean(gray)
# #         brightness_score = 1.0 - abs(brightness - 128) / 128.0
        
# #         # Size score
# #         h, w = face_crop.shape[:2]
# #         size_score = min((h * w) / (112 * 112), 1.0)
        
# #         # Weighted average
# #         quality = (0.5 * sharpness_score + 0.3 * brightness_score + 0.2 * size_score)
        
# #         return quality
    
# #     def draw_detections(self, frame, faces, color=(0, 255, 0), thickness=2):
# #         """Draw bounding boxes on frame"""
# #         display_frame = frame.copy()
        
# #         for (x1, y1, x2, y2, conf) in faces:
# #             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
# #             label = f"{conf:.2f}"
# #             cv2.putText(display_frame, label, (x1, y1-10),
# #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
# #         return display_frame


# # # ============================================================================
# # # DOWNLOAD SCRFD MODEL HELPER
# # # ============================================================================

# # def download_scrfd_model():
# #     """
# #     Download SCRFD model - VERIFIED WORKING LINKS (Oct 2025)
# #     """
# #     import urllib.request
# #     import os
    
# #     os.makedirs("models", exist_ok=True)
    
# #     models_to_try = [
# #         {
# #             'name': 'SCRFD 10G (Best Accuracy)',
# #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_10g_bnkps.onnx',
# #             'filename': 'scrfd_10g_bnkps.onnx',
# #             'size': '~16MB'
# #         },
# #         {
# #             'name': 'SCRFD 2.5G (Balanced)',
# #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_2.5g_bnkps.onnx',
# #             'filename': 'scrfd_2.5g_bnkps.onnx',
# #             'size': '~3MB'
# #         },
# #         {
# #             'name': 'SCRFD 500M (Lightweight)',
# #             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_500m_bnkps.onnx',
# #             'filename': 'scrfd_500m_bnkps.onnx',
# #             'size': '~2MB'
# #         }
# #     ]
    
# #     print("\n" + "="*70)
# #     print("SCRFD MODEL DOWNLOAD")
# #     print("="*70)
# #     print("\nAvailable models:")
# #     for idx, model in enumerate(models_to_try, 1):
# #         status = "✓" if os.path.exists(f"models/{model['filename']}") else " "
# #         print(f"{status} {idx}. {model['name']} ({model['size']})")
    
# #     choice = input("\nSelect model (1-3, default 1): ").strip() or "1"
    
# #     try:
# #         model_idx = int(choice) - 1
# #         selected_model = models_to_try[model_idx]
# #     except:
# #         selected_model = models_to_try[0]
    
# #     model_path = f"models/{selected_model['filename']}"
    
# #     if os.path.exists(model_path):
# #         print(f"\n[INFO] Model already exists at {model_path}")
# #         return model_path
    
# #     print(f"\n[INFO] Downloading {selected_model['name']}...")
# #     print(f"[INFO] URL: {selected_model['url']}")
# #     print("[INFO] This may take a few minutes...")
    
# #     try:
# #         def download_progress(block_num, block_size, total_size):
# #             downloaded = block_num * block_size
# #             percent = min(downloaded * 100 / total_size, 100)
# #             print(f"\r[DOWNLOAD] Progress: {percent:.1f}%", end='', flush=True)
        
# #         urllib.request.urlretrieve(
# #             selected_model['url'], 
# #             model_path,
# #             reporthook=download_progress
# #         )
        
# #         print(f"\n[SUCCESS] Model downloaded to {model_path}")
# #         return model_path
        
# #     except Exception as e:
# #         print(f"\n[ERROR] Download failed: {e}")
# #         print("\n[ALTERNATIVE] Manual download instructions:")
# #         print("="*70)
# #         print("Option 1: Direct Download (GitHub)")
# #         print("  Visit: https://github.com/nttstar/insightface-resources/releases/tag/v0.7")
# #         print(f"  Download: {selected_model['filename']}")
# #         print("  Place in: models/ folder")
# #         print()
# #         print("Option 2: Use wget (Linux/Mac)")
# #         print(f"  wget {selected_model['url']} -P models/")
# #         print()
# #         print("Option 3: Use curl")
# #         print(f"  curl -L {selected_model['url']} -o models/{selected_model['filename']}")
# #         print("="*70)
        
# #         return None


# # if __name__ == "__main__":
# #     # Test detector
# #     print("Testing Face Detector...")
    
# #     # Download model if needed
# #     download_scrfd_model()
    
# #     # Initialize detector
# #     detector = FaceDetector()
    
# #     # Test with webcam
# #     cap = cv2.VideoCapture(0)
    
# #     if not cap.isOpened():
# #         print("[ERROR] Cannot open webcam")
# #         exit()
    
# #     print("\nPress 'q' to quit")
    
# #     fps_start = 0
# #     fps_counter = 0
# #     fps = 0
    
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
        
# #         # Detect faces
# #         faces = detector.detect_faces(frame)
        
# #         # Draw results
# #         annotated = detector.draw_detections(frame, faces)
        
# #         # Calculate FPS
# #         fps_counter += 1
# #         if fps_counter >= 30:
# #             fps = 30.0 / (cv2.getTickCount() / cv2.getTickFrequency() - fps_start) if fps_start > 0 else 0
# #             fps_start = cv2.getTickCount() / cv2.getFrequency()
# #             fps_counter = 0
        
# #         # Show FPS and face count
# #         cv2.putText(annotated, f"Faces: {len(faces)} | FPS: {fps:.1f}", (10, 30),
# #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
# #         cv2.imshow('Face Detection Test', annotated)
        
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
    
# #     cap.release()
# #     cv2.destroyAllWindows()

# """
# SCRFD Face Detector - FIXED for Jetson Nano
# CRITICAL FIX: Proper handling of already-flattened ONNX outputs
# Handles 30-40 faces with batch processing
# """

# import cv2
# import numpy as np
# import onnxruntime as ort
# import threading
# from queue import Queue


# class FaceDetector:
#     def __init__(self, model_path='models/scrfd_10g_bnkps.onnx', conf_threshold=0.5):
#         """
#         SCRFD Face Detector optimized for Jetson Nano
#         Args:
#             model_path: Path to SCRFD ONNX model
#             conf_threshold: Detection confidence threshold
#         """
#         print(f"[DETECTOR] Loading SCRFD Face Detector for Jetson Nano...")
        
#         try:
#             # Use CUDA execution provider for Jetson Nano GPU acceleration
#             providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#             self.session = ort.InferenceSession(model_path, providers=providers)
            
#             self.conf_threshold = conf_threshold
#             self.input_size = (640, 640)  # SCRFD input size
#             self.nms_threshold = 0.4
#             self.model_type = 'scrfd'
            
#             # Get output names and shapes
#             self.output_names = [output.name for output in self.session.get_outputs()]
            
#             print(f"[DETECTOR] SCRFD loaded successfully")
#             print(f"[DETECTOR] Providers: {self.session.get_providers()}")
#             print(f"[DETECTOR] Input size: {self.input_size}")
#             print(f"[DETECTOR] Output layers: {len(self.output_names)}")
            
#         except Exception as e:
#             print(f"[ERROR] Failed to load SCRFD model: {e}")
#             print("[INFO] Falling back to YOLOv11n...")
#             self._fallback_to_yolo()
    
#     def _fallback_to_yolo(self):
#         """Fallback to YOLOv11n if SCRFD fails"""
#         try:
#             from ultralytics import YOLO
#             self.session = YOLO('models/yolo11n-face.pt')
#             self.model_type = 'yolo'
#             print("[DETECTOR] YOLOv11n loaded as fallback")
#         except Exception as e:
#             raise Exception(f"Both SCRFD and YOLO failed: {e}")
    
#     def preprocess_scrfd(self, image):
#         """Preprocess image for SCRFD model"""
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img_resized = cv2.resize(img, self.input_size)
        
#         # Normalize
#         img_normalized = (img_resized.astype(np.float32) - 127.5) / 128.0
#         img_transposed = np.transpose(img_normalized, (2, 0, 1))
#         img_batch = np.expand_dims(img_transposed, axis=0)
        
#         return img_batch
    
#     def postprocess_scrfd(self, outputs, orig_shape):
#         """
#         Post-process SCRFD outputs - COMPLETELY REWRITTEN AND FIXED
        
#         CRITICAL FIX: SCRFD ONNX outputs are ALREADY FLATTENED to [N, C] format
#         - outputs[0:3]: score_8, score_16, score_32  -> [12800, 1], [3200, 1], [800, 1]
#         - outputs[3:6]: bbox_8, bbox_16, bbox_32      -> [12800, 4], [3200, 4], [800, 4]
#         - outputs[6:9]: kps_8, kps_16, kps_32         -> [12800, 10], [3200, 10], [800, 10]
        
#         DO NOT reshape - outputs are already in correct format!
#         """
#         h, w = orig_shape[:2]
#         scale_h = h / self.input_size[0]
#         scale_w = w / self.input_size[1]
        
#         faces = []
        
#         # SCRFD uses 3 FPN levels with different strides
#         strides = [8, 16, 32]
        
#         try:
#             # Process each FPN level
#             for idx, stride in enumerate(strides):
#                 # Calculate feature map size
#                 fm_h = self.input_size[0] // stride
#                 fm_w = self.input_size[1] // stride
#                 num_anchors = fm_h * fm_w
                
#                 # Get outputs for this level (ALREADY FLATTENED!)
#                 score_pred = outputs[idx]      # Shape: [N, 1] where N = fm_h * fm_w
#                 bbox_pred = outputs[idx + 3]   # Shape: [N, 4]
                
#                 # Verify shape matches expected anchor count
#                 if score_pred.shape[0] != num_anchors:
#                     print(f"[WARNING] Level {idx}: Expected {num_anchors} anchors, got {score_pred.shape[0]}")
#                     continue
                
#                 # Generate anchors for this level
#                 anchors = []
#                 for i in range(fm_h):
#                     for j in range(fm_w):
#                         cx = (j + 0.5) * stride
#                         cy = (i + 0.5) * stride
#                         anchors.append([cx, cy, stride])
                
#                 anchors = np.array(anchors)  # Shape: [num_anchors, 3]
                
#                 # Extract scores (flatten if needed)
#                 scores = score_pred.flatten()
                
#                 # Apply confidence threshold
#                 mask = scores >= self.conf_threshold
                
#                 if not np.any(mask):
#                     continue
                
#                 # Filter predictions
#                 filtered_scores = scores[mask]
#                 filtered_boxes = bbox_pred[mask]
#                 filtered_anchors = anchors[mask]
                
#                 # Decode bounding boxes (distance format)
#                 # SCRFD predicts: [left, top, right, bottom] distances from anchor
#                 for score, box, anchor in zip(filtered_scores, filtered_boxes, filtered_anchors):
#                     cx, cy, s = anchor
                    
#                     # Decode box (distance to LTRB format)
#                     x1 = (cx - box[0]) * scale_w
#                     y1 = (cy - box[1]) * scale_h
#                     x2 = (cx + box[2]) * scale_w
#                     y2 = (cy + box[3]) * scale_h
                    
#                     # Clamp to image boundaries
#                     x1 = max(0, min(int(x1), w))
#                     y1 = max(0, min(int(y1), h))
#                     x2 = max(0, min(int(x2), w))
#                     y2 = max(0, min(int(y2), h))
                    
#                     # Validate box
#                     if x2 > x1 and y2 > y1:
#                         faces.append((x1, y1, x2, y2, float(score)))
            
#         except Exception as e:
#             print(f"[DEBUG] SCRFD postprocess error: {e}")
#             import traceback
#             print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
#             # Print output shapes for debugging
#             print(f"[DEBUG] Number of outputs: {len(outputs)}")
#             for i, out in enumerate(outputs):
#                 print(f"[DEBUG] Output {i} shape: {out.shape}")
            
#             return []
        
#         return faces
    
#     def detect_faces(self, frame):
#         """
#         Detect all faces in frame
#         Returns: List of (x1, y1, x2, y2, confidence)
#         """
#         if hasattr(self, 'model_type') and self.model_type == 'yolo':
#             return self._detect_yolo(frame)
        
#         # SCRFD detection
#         input_blob = self.preprocess_scrfd(frame)
#         input_name = self.session.get_inputs()[0].name
        
#         try:
#             outputs = self.session.run(None, {input_name: input_blob})
#             faces = self.postprocess_scrfd(outputs, frame.shape)
            
#             # Apply NMS to remove overlapping boxes
#             faces = self._apply_nms(faces)
            
#             return faces
            
#         except Exception as e:
#             print(f"[ERROR] SCRFD detection failed: {e}")
#             return []
    
#     def _detect_yolo(self, frame):
#         """YOLO detection fallback"""
#         results = self.session(frame, conf=self.conf_threshold, verbose=False)
        
#         faces = []
#         if len(results) > 0 and results[0].boxes is not None:
#             boxes = results[0].boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                 conf = float(box.conf[0].cpu().numpy())
#                 faces.append((x1, y1, x2, y2, conf))
        
#         return faces
    
#     def _apply_nms(self, faces):
#         """Apply Non-Maximum Suppression"""
#         if len(faces) == 0:
#             return []
        
#         boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in faces])
#         scores = np.array([conf for _, _, _, _, conf in faces])
        
#         # Calculate areas
#         x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#         areas = (x2 - x1) * (y2 - y1)
        
#         # Sort by confidence
#         order = scores.argsort()[::-1]
        
#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
            
#             # Calculate IoU with remaining boxes
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])
            
#             w = np.maximum(0.0, xx2 - xx1)
#             h = np.maximum(0.0, yy2 - yy1)
#             inter = w * h
            
#             iou = inter / (areas[i] + areas[order[1:]] - inter)
            
#             # Keep boxes with IoU < threshold
#             inds = np.where(iou <= self.nms_threshold)[0]
#             order = order[inds + 1]
        
#         return [faces[i] for i in keep]
    
#     def detect_batch(self, frames):
#         """
#         Batch detection for multiple frames (dual camera support)
#         Args:
#             frames: List of frames
#         Returns: List of face detections for each frame
#         """
#         results = []
#         for frame in frames:
#             faces = self.detect_faces(frame)
#             results.append(faces)
#         return results
    
#     def extract_face_crops(self, frame, faces, padding=0.15, min_size=60):
#         """
#         Extract face crops with MINIMAL padding (optimized for side profiles)
        
#         Args:
#             frame: Input frame
#             faces: List of (x1, y1, x2, y2, conf)
#             padding: Padding ratio (15% for side profiles)
#             min_size: Minimum face size in pixels
        
#         Returns: List of face crop dictionaries
#         """
#         crops = []
#         h, w = frame.shape[:2]
        
#         for (x1, y1, x2, y2, conf) in faces:
#             face_w = x2 - x1
#             face_h = y2 - y1
            
#             # Skip tiny faces
#             if face_w < min_size or face_h < min_size:
#                 continue
            
#             # Calculate padding (REDUCED to 15% for better side profile capture)
#             pad_w = int(face_w * padding)
#             pad_h = int(face_h * padding)
            
#             # Apply padding with boundary check
#             x1_crop = max(0, x1 - pad_w)
#             y1_crop = max(0, y1 - pad_h)
#             x2_crop = min(w, x2 + pad_w)
#             y2_crop = min(h, y2 + pad_h)
            
#             # Extract crop
#             crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            
#             if crop.size == 0:
#                 continue
            
#             # Resize small crops
#             crop_h, crop_w = crop.shape[:2]
#             if crop_h < 112 or crop_w < 112:
#                 scale = max(112/crop_h, 112/crop_w)
#                 new_w = int(crop_w * scale)
#                 new_h = int(crop_h * scale)
#                 crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
#             # Calculate face quality score (for filtering)
#             quality_score = self._calculate_quality(crop)
            
#             crops.append({
#                 'image': crop,
#                 'bbox': (x1, y1, x2, y2),
#                 'conf': conf,
#                 'quality': quality_score,
#                 'size': (face_w, face_h)
#             })
        
#         return crops
    
#     def _calculate_quality(self, face_crop):
#         """
#         Calculate face quality score (0-1)
#         Factors: sharpness, brightness, size
#         """
#         if face_crop.size == 0:
#             return 0.0
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
#         # Sharpness (Laplacian variance)
#         sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
#         sharpness_score = min(sharpness / 100.0, 1.0)
        
#         # Brightness
#         brightness = np.mean(gray)
#         brightness_score = 1.0 - abs(brightness - 128) / 128.0
        
#         # Size score
#         h, w = face_crop.shape[:2]
#         size_score = min((h * w) / (112 * 112), 1.0)
        
#         # Weighted average
#         quality = (0.5 * sharpness_score + 0.3 * brightness_score + 0.2 * size_score)
        
#         return quality
    
#     def draw_detections(self, frame, faces, color=(0, 255, 0), thickness=2):
#         """Draw bounding boxes on frame"""
#         display_frame = frame.copy()
        
#         for (x1, y1, x2, y2, conf) in faces:
#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
#             label = f"{conf:.2f}"
#             cv2.putText(display_frame, label, (x1, y1-10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
#         return display_frame


# # ============================================================================
# # DOWNLOAD SCRFD MODEL HELPER
# # ============================================================================

# def download_scrfd_model():
#     """
#     Download SCRFD model - VERIFIED WORKING LINKS (Oct 2025)
#     """
#     import urllib.request
#     import os
    
#     os.makedirs("models", exist_ok=True)
    
#     models_to_try = [
#         {
#             'name': 'SCRFD 10G (Best Accuracy)',
#             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_10g_bnkps.onnx',
#             'filename': 'scrfd_10g_bnkps.onnx',
#             'size': '~16MB'
#         },
#         {
#             'name': 'SCRFD 2.5G (Balanced)',
#             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_2.5g_bnkps.onnx',
#             'filename': 'scrfd_2.5g_bnkps.onnx',
#             'size': '~3MB'
#         },
#         {
#             'name': 'SCRFD 500M (Lightweight)',
#             'url': 'https://github.com/nttstar/insightface-resources/releases/download/v0.7/scrfd_500m_bnkps.onnx',
#             'filename': 'scrfd_500m_bnkps.onnx',
#             'size': '~2MB'
#         }
#     ]
    
#     print("\n" + "="*70)
#     print("SCRFD MODEL DOWNLOAD")
#     print("="*70)
#     print("\nAvailable models:")
#     for idx, model in enumerate(models_to_try, 1):
#         status = "✓" if os.path.exists(f"models/{model['filename']}") else " "
#         print(f"{status} {idx}. {model['name']} ({model['size']})")
    
#     choice = input("\nSelect model (1-3, default 1): ").strip() or "1"
    
#     try:
#         model_idx = int(choice) - 1
#         selected_model = models_to_try[model_idx]
#     except:
#         selected_model = models_to_try[0]
    
#     model_path = f"models/{selected_model['filename']}"
    
#     if os.path.exists(model_path):
#         print(f"\n[INFO] Model already exists at {model_path}")
#         return model_path
    
#     print(f"\n[INFO] Downloading {selected_model['name']}...")
#     print(f"[INFO] URL: {selected_model['url']}")
#     print("[INFO] This may take a few minutes...")
    
#     try:
#         def download_progress(block_num, block_size, total_size):
#             downloaded = block_num * block_size
#             percent = min(downloaded * 100 / total_size, 100)
#             print(f"\r[DOWNLOAD] Progress: {percent:.1f}%", end='', flush=True)
        
#         urllib.request.urlretrieve(
#             selected_model['url'], 
#             model_path,
#             reporthook=download_progress
#         )
        
#         print(f"\n[SUCCESS] Model downloaded to {model_path}")
#         return model_path
        
#     except Exception as e:
#         print(f"\n[ERROR] Download failed: {e}")
#         print("\n[ALTERNATIVE] Manual download instructions:")
#         print("="*70)
#         print("Option 1: Direct Download (GitHub)")
#         print("  Visit: https://github.com/nttstar/insightface-resources/releases/tag/v0.7")
#         print(f"  Download: {selected_model['filename']}")
#         print("  Place in: models/ folder")
#         print()
#         print("Option 2: Use wget (Linux/Mac)")
#         print(f"  wget {selected_model['url']} -P models/")
#         print()
#         print("Option 3: Use curl")
#         print(f"  curl -L {selected_model['url']} -o models/{selected_model['filename']}")
#         print("="*70)
        
#         return None


# if __name__ == "__main__":
#     # Test detector
#     print("Testing Face Detector...")
    
#     # Download model if needed
#     download_scrfd_model()
    
#     # Initialize detector
#     detector = FaceDetector()
    
#     # Test with webcam
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("[ERROR] Cannot open webcam")
#         exit()
    
#     print("\nPress 'q' to quit")
    
#     fps_start = 0
#     fps_counter = 0
#     fps = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Detect faces
#         faces = detector.detect_faces(frame)
        
#         # Draw results
#         annotated = detector.draw_detections(frame, faces)
        
#         # Calculate FPS
#         fps_counter += 1
#         if fps_counter >= 30:
#             fps = 30.0 / (cv2.getTickCount() / cv2.getTickFrequency() - fps_start) if fps_start > 0 else 0
#             fps_start = cv2.getTickCount() / cv2.getTickFrequency()
#             fps_counter = 0
        
#         # Show FPS and face count
#         cv2.putText(annotated, f"Faces: {len(faces)} | FPS: {fps:.1f}", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         cv2.imshow('Face Detection Test', annotated)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# face_detector.py
# -----------------------------------------------------------------------------
# YOLOv11n-Face detector (Ultralytics) with robust post-processing:
#  - strict in-frame crop bounds
#  - minimal, configurable padding
#  - quality score (sharpness + brightness + size)
#  - optional NMS (YOLO already does NMS internally, but we keep a light pass)
#  - drawing helpers for visualization
#
# Expected model file:
#   models/yolo11n-face.pt
# You can get one from the Ultralytics community face models or train your own.
# -----------------------------------------------------------------------------

import os
import cv2
import math
import numpy as np
from typing import List, Tuple, Dict, Any

# Ultralytics YOLO (pip install ultralytics)
from ultralytics import YOLO


class FaceDetector:
    """
    YOLOv11n-Face detector with robust extraction utilities.
    """

    def __init__(
        self,
        model_path: str = "models/yolo11n-face.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        warmup: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            model_path: Path to YOLOv11n face model weights.
            conf_threshold: Detection confidence threshold.
            iou_threshold: NMS IoU threshold (YOLO internal).
            max_det: Max detections per image (YOLO internal).
            warmup: Run one dummy inference to load weights to RAM/GPU.
            verbose: Print loader messages.
        """
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.verbose = verbose

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"[FaceDetector] YOLO model not found: {self.model_path}\n"
                f"Place your face weights at: {self.model_path}"
            )

        if self.verbose:
            print("=" * 80)
            print("FaceDetector - YOLOv11n-Face")
            print("=" * 80)
            print(f"[Model] {self.model_path}")
            print(f"[Thresh] conf={self.conf_threshold}  iou={self.iou_threshold}")

        # Load model
        self.model = YOLO(self.model_path)

        # Warmup to avoid first-frame latency spikes
        if warmup:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False
            )

        if self.verbose:
            print("[FaceDetector] YOLOv11n-Face loaded ✓")
            print("=" * 80)

    # -------------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------------
    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run YOLO detection on a BGR frame.

        Returns:
            List of (x1, y1, x2, y2, conf) in absolute pixel coords.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # YOLO expects BGR/uint8 — passing as-is is fine.
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False
        )

        faces = []
        if not results:
            return faces

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return faces

        # Extract boxes
        # (YOLO already applies NMS internally; coords are xyxy in image scale)
        for box in r0.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            # Ensure non-degenerate
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append((x1, y1, x2, y2, conf))

        return faces

    # -------------------------------------------------------------------------
    # Optional extra NMS (safety net). YOLO does NMS, but if downstream
    # augmentations or TTA are used, you might call this to re-NMS combined boxes.
    # -------------------------------------------------------------------------
    @staticmethod
    def nms_boxes(
        boxes: List[Tuple[int, int, int, int, float]],
        iou_thresh: float = 0.45
    ) -> List[Tuple[int, int, int, int, float]]:
        if not boxes:
            return []
        boxes_np = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _) in boxes], dtype=np.float32)
        scores = np.array([conf for (_, _, _, _, conf) in boxes], dtype=np.float32)

        x1, y1, x2, y2 = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [boxes[idx] for idx in keep]

    # -------------------------------------------------------------------------
    # Face quality metrics
    # -------------------------------------------------------------------------
    @staticmethod
    def _quality_score(face_bgr: np.ndarray) -> float:
        """
        Compute a simple quality score in [0,1] from:
          - sharpness via Laplacian variance
          - brightness closeness to 128
          - relative size (vs 112x112)
        """
        if face_bgr is None or face_bgr.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # Sharpness
        sharp_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = min(sharp_var / 100.0, 1.0)  # simple squashing

        # Brightness (favor mid-tones)
        mean_b = float(np.mean(gray))
        bright_score = 1.0 - abs(mean_b - 128.0) / 128.0
        bright_score = max(0.0, min(1.0, bright_score))

        # Size score
        h, w = face_bgr.shape[:2]
        size_score = min((h * w) / float(112 * 112), 1.0)

        # Weighted sum
        return 0.5 * sharp_score + 0.3 * bright_score + 0.2 * size_score

    # -------------------------------------------------------------------------
    # Crop extraction
    # -------------------------------------------------------------------------
    def extract_face_crops(
        self,
        frame_bgr: np.ndarray,
        faces: List[Tuple[int, int, int, int, float]],
        padding: float = 0.20,
        min_size: int = 56,
        clamp_to_frame: bool = True,
        sort_by_quality: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract cropped face regions with minimal padding and strong bounds.

        Args:
            frame_bgr: Input BGR image.
            faces: List of (x1, y1, x2, y2, conf) from detect_faces().
            padding: Relative padding (per dimension) around the box.
            min_size: Minimum crop width/height to keep.
            clamp_to_frame: Ensure crops are fully inside the frame.
            sort_by_quality: If True, return crops sorted descending by quality.

        Returns:
            List of dicts:
              {
                'image': cropped_face_bgr,
                'bbox': (x1, y1, x2, y2),   # after padding and clamping
                'conf': detection_confidence,
                'quality': float in [0,1],
                'size': (w, h)
              }
        """
        if frame_bgr is None or frame_bgr.size == 0 or not faces:
            return []

        H, W = frame_bgr.shape[:2]
        crops = []

        for (x1, y1, x2, y2, conf) in faces:
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            # Skip tiny detections early
            if w < min_size or h < min_size:
                continue

            pad_w = int(w * padding)
            pad_h = int(h * padding)

            cx1 = x1 - pad_w
            cy1 = y1 - pad_h
            cx2 = x2 + pad_w
            cy2 = y2 + pad_h

            if clamp_to_frame:
                cx1 = max(0, cx1)
                cy1 = max(0, cy1)
                cx2 = min(W, cx2)
                cy2 = min(H, cy2)

            # Validate crop region
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            if ch < min_size or cw < min_size:
                # Upscale tiny valid crops to at least 112x112 for recognizer stability
                scale = max(112.0 / ch, 112.0 / cw)
                new_w = max(112, int(round(cw * scale)))
                new_h = max(112, int(round(ch * scale)))
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            q = self._quality_score(crop)

            crops.append({
                "image": crop,
                "bbox": (int(cx1), int(cy1), int(cx2), int(cy2)),
                "conf": float(conf),
                "quality": float(q),
                "size": (int(cx2 - cx1), int(cy2 - cy1))
            })

        if sort_by_quality:
            crops.sort(key=lambda d: d["quality"], reverse=True)

        return crops

    # -------------------------------------------------------------------------
    # Visualization helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def draw_boxes(
        frame_bgr: np.ndarray,
        faces: List[Tuple[int, int, int, int, float]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_conf: bool = True
    ) -> np.ndarray:
        """
        Draw raw YOLO boxes on a copy of the frame.
        """
        vis = frame_bgr.copy()
        for (x1, y1, x2, y2, conf) in faces:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            if show_conf:
                label = f"{conf:.2f}"
                cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return vis

    @staticmethod
    def draw_crops_grid(crops: List[dict], cols: int = 6, cell: int = 112) -> np.ndarray:
        """
        Utility to visualize extracted face crops in a grid (debugging/QA).
        """
        if not crops:
            return np.zeros((cell, cell, 3), dtype=np.uint8)

        rows = math.ceil(len(crops) / cols)
        canvas = np.full((rows * cell, cols * cell, 3), 32, dtype=np.uint8)

        for idx, c in enumerate(crops):
            r, col = divmod(idx, cols)
            thumb = cv2.resize(c["image"], (cell, cell))
            y1, y2 = r * cell, (r + 1) * cell
            x1, x2 = col * cell, (col + 1) * cell
            canvas[y1:y2, x1:x2] = thumb

        return canvas
