# # 

# """
# AdaFace Recognition System - PyTorch Version
# FIXED: Handles state_dict key mismatches correctly
# Works on Laptop (CUDA/CPU) and Jetson Nano
# """

# import cv2
# import numpy as np
# import pickle
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms

# class FaceRecognizer:
#     def __init__(self, database_path='database/embeddings_adaface.pkl'):
#         """
#         AdaFace Recognition System (PyTorch)
#         Works on: Laptop (CUDA/CPU) and Jetson Nano (CUDA)
#         """
#         print("[RECOGNIZER] Loading AdaFace Recognition System...")
#         print("[RECOGNIZER] Platform: PyTorch (CUDA/CPU compatible)")
        
#         self.database_path = database_path
#         self.embeddings_db = {}
        
#         # Auto-detect device
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#             print(f"[RECOGNIZER] Using GPU: {torch.cuda.get_device_name(0)}")
#         else:
#             self.device = torch.device('cpu')
#             print("[RECOGNIZER] Using CPU")
        
#         # Load AdaFace model
#         self._load_adaface_model()
        
#         # Transform pipeline
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((112, 112)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#         # Load database
#         self.load_database()
        
#         print(f"[RECOGNIZER] System ready with {len(self.embeddings_db)} identities")
    
#     def _load_adaface_model(self):
#         """Load AdaFace PyTorch model with FIXED state_dict handling"""
#         try:
#             # Check if model files exist
#             weight_path = 'models/adaface_ir101_webface12m.ckpt'
            
#             if not os.path.exists(weight_path):
#                 print(f"[ERROR] AdaFace weights not found at: {weight_path}")
#                 print("[INFO] Trying alternative paths...")
                
#                 # Try alternative paths
#                 alt_paths = [
#                     'models/adaface_ir101_webface12m.ckpt',
#                     'models/adaface_ir50_webface4m.ckpt',
#                     'models/adaface_ir18_vgg2.ckpt'
#                 ]
                
#                 for alt_path in alt_paths:
#                     if os.path.exists(alt_path):
#                         weight_path = alt_path
#                         print(f"[INFO] Found model at: {weight_path}")
#                         break
#                 else:
#                     raise FileNotFoundError("No AdaFace model found")
            
#             print(f"[RECOGNIZER] Loading AdaFace from: {weight_path}")
            
#             # Import AdaFace architecture
#             import sys
#             adaface_lib_path = 'adaface_lib'
#             if os.path.exists(adaface_lib_path):
#                 sys.path.insert(0, adaface_lib_path)
            
#             from net import build_model
            
#             # Build model (ir_50 architecture)
#             self.model = build_model('ir_50')
            
#             # Load weights
#             print("[RECOGNIZER] Loading checkpoint...")
#             checkpoint = torch.load(weight_path, map_location=self.device)
            
#             # Handle different checkpoint formats
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#                 print("[DEBUG] Checkpoint contains 'state_dict' key")
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#                 print("[DEBUG] Checkpoint contains 'model' key")
#             else:
#                 state_dict = checkpoint
#                 print("[DEBUG] Checkpoint is direct state_dict")
            
#             # CRITICAL FIX: Remove ALL prefixes (model., module., etc.)
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 # Remove 'model.' prefix
#                 name = k.replace('model.', '')
#                 # Remove 'module.' prefix (for DataParallel models)
#                 name = name.replace('module.', '')
#                 new_state_dict[name] = v
            
#             print(f"[DEBUG] Original keys sample: {list(state_dict.keys())[:3]}")
#             print(f"[DEBUG] Cleaned keys sample: {list(new_state_dict.keys())[:3]}")
#             print(f"[DEBUG] Model expects keys sample: {list(self.model.state_dict().keys())[:3]}")
            
#             # Load with strict=False to see what's missing/extra
#             missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
#             if missing_keys:
#                 print(f"[WARNING] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
#             if unexpected_keys:
#                 print(f"[WARNING] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            
#             # Move to device and set eval mode
#             self.model.to(self.device)
#             self.model.eval()
            
#             self.model_loaded = True
#             print("[RECOGNIZER] ✓ AdaFace loaded successfully")
            
#         except Exception as e:
#             print(f"[ERROR] AdaFace loading failed: {e}")
#             print(f"[ERROR] Error type: {type(e).__name__}")
#             import traceback
#             print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            
#             print("[RECOGNIZER] Switching to DeepFace fallback...")
#             self._load_deepface_fallback()
    
#     def _load_deepface_fallback(self):
#         """Fallback to DeepFace ArcFace"""
#         try:
#             from deepface import DeepFace
#             self.deepface = DeepFace
#             self.model_loaded = False
#             print("[RECOGNIZER] Using DeepFace ArcFace as fallback")
            
#             # Pre-load model
#             dummy = np.zeros((112, 112, 3), dtype=np.uint8)
#             _ = DeepFace.represent(dummy, model_name="ArcFace", enforce_detection=False)
#             print("[RECOGNIZER] ✓ ArcFace loaded successfully")
            
#         except Exception as e:
#             raise Exception(f"All recognition models failed: {e}")
    
#     def get_embedding(self, face_image):
#         """
#         Extract 512-dim embedding from face image
#         Args:
#             face_image: BGR face crop
#         Returns:
#             512-dim numpy array or None
#         """
#         try:
#             if self.model_loaded:
#                 # AdaFace PyTorch inference
#                 face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#                 face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
                
#                 with torch.no_grad():
#                     embedding = self.model(face_tensor)
                
#                 embedding = embedding.cpu().numpy()[0]
                
#                 # L2 normalize
#                 embedding = embedding / np.linalg.norm(embedding)
                
#                 return embedding
#             else:
#                 # DeepFace fallback
#                 face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#                 result = self.deepface.represent(
#                     face_rgb,
#                     model_name="ArcFace",
#                     enforce_detection=False,
#                     detector_backend='skip'
#                 )
                
#                 if len(result) > 0:
#                     embedding = np.array(result[0]["embedding"])
#                     embedding = embedding / np.linalg.norm(embedding)
#                     return embedding
                
#                 return None
                
#         except Exception as e:
#             print(f"[DEBUG] Embedding extraction error: {e}")
#             return None
    
#     def preprocess_face(self, face_image):
#         """Enhanced preprocessing"""
#         if len(face_image.shape) == 2:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
#         elif face_image.shape[2] == 4:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
#         else:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
#         h, w = face_image.shape[:2]
#         if h < 112 or w < 112:
#             scale = max(112/h, 112/w) * 1.1
#             new_h, new_w = int(h * scale), int(w * scale)
#             face_image = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
#         img_yuv = cv2.cvtColor(face_image, cv2.COLOR_RGB2YUV)
#         img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#         face_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
#         face_image = cv2.GaussianBlur(face_image, (3, 3), 0)
#         face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
#         return face_image
    
#     def generate_synthetic_poses(self, face_image):
#         """Generate 7 synthetic poses from single photo"""
#         augmented = []
        
#         face_image = self.preprocess_face(face_image)
#         h, w = face_image.shape[:2]
        
#         # 1. Original
#         augmented.append(face_image.copy())
        
#         # 2. Horizontal flip
#         augmented.append(cv2.flip(face_image, 1))
        
#         # 3-6. Rotations
#         for angle in [-10, -5, 5, 10]:
#             M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
#             rotated = cv2.warpAffine(face_image, M, (w, h), 
#                                     borderMode=cv2.BORDER_REPLICATE)
#             augmented.append(rotated)
        
#         # 7. Brightness
#         bright = cv2.convertScaleAbs(face_image, alpha=1.15, beta=0)
#         augmented.append(bright)
        
#         return augmented
    
#     def add_person(self, name, face_image, use_augmentation=True):
#         """Add person with synthetic augmentation"""
#         if use_augmentation:
#             samples = self.generate_synthetic_poses(face_image)
#             samples_desc = f"1 photo → {len(samples)} synthetic poses"
#         else:
#             samples = [self.preprocess_face(face_image)]
#             samples_desc = "1 sample (no augmentation)"
        
#         successful = 0
        
#         for sample in samples:
#             embedding = self.get_embedding(sample)
            
#             if embedding is not None:
#                 if name not in self.embeddings_db:
#                     self.embeddings_db[name] = []
                
#                 self.embeddings_db[name].append(embedding)
#                 successful += 1
        
#         if successful > 0:
#             self.save_database()
#             print(f"[RECOGNIZER] ✓ Added {name} ({samples_desc} → {successful} embeddings)")
#             return True
#         else:
#             print(f"[RECOGNIZER] ✗ Failed to add {name}")
#             return False
    
#     def cosine_similarity(self, emb1, emb2):
#         """Calculate cosine similarity"""
#         if emb1 is None or emb2 is None:
#             return 0.0
#         return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
#     def recognize(self, face_image, threshold=0.42):
#         """Recognize face using AdaFace"""
#         query_embedding = self.get_embedding(face_image)
        
#         if query_embedding is None:
#             return "Unknown", 0.0
        
#         best_match = None
#         best_score = threshold
        
#         for name, stored_embeddings in self.embeddings_db.items():
#             scores = []
            
#             for stored_emb in stored_embeddings:
#                 similarity = self.cosine_similarity(query_embedding, stored_emb)
#                 scores.append(similarity)
            
#             if scores:
#                 max_score = max(scores)
                
#                 if max_score > best_score:
#                     best_score = max_score
#                     best_match = name
        
#         if best_match:
#             return best_match, best_score
#         else:
#             return "Unknown", 0.0
    
#     def load_database(self):
#         """Load embeddings database"""
#         if os.path.exists(self.database_path):
#             with open(self.database_path, 'rb') as f:
#                 self.embeddings_db = pickle.load(f)
#             print(f"[RECOGNIZER] Loaded database: {len(self.embeddings_db)} identities")
#         else:
#             print("[RECOGNIZER] Starting fresh database")
    
#     def save_database(self):
#         """Save embeddings database"""
#         os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
#         with open(self.database_path, 'wb') as f:
#             pickle.dump(self.embeddings_db, f)
    
#     def get_all_registered_persons(self):
#         """Get list of registered persons"""
#         return list(self.embeddings_db.keys())


# # ============================================================================
# # MODEL DOWNLOAD HELPER
# # ============================================================================

# def download_adaface_models():
#     """
#     Download AdaFace models - VERIFIED WORKING LINKS
#     """
#     import urllib.request
#     import os
    
#     os.makedirs("models", exist_ok=True)
    
#     models = [
#         {
#             'name': 'AdaFace IR-50 (MS1MV2) - RECOMMENDED',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_ms1mv2.ckpt',
#             'filename': 'adaface_ir50_ms1mv2.ckpt',
#             'size': '~166MB'
#         },
#         {
#             'name': 'AdaFace IR-50 (WebFace4M)',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.ckpt',
#             'filename': 'adaface_ir50_webface4m.ckpt',
#             'size': '~166MB'
#         }
#     ]
    
#     print("\n" + "="*70)
#     print("AdaFace MODEL DOWNLOAD")
#     print("="*70)
#     print("\nAvailable models:")
#     for idx, model in enumerate(models, 1):
#         status = "✓" if os.path.exists(f"models/{model['filename']}") else " "
#         print(f"{status} {idx}. {model['name']} ({model['size']})")
    
#     choice = input("\nSelect model (1-2, default 1): ").strip() or "1"
    
#     try:
#         model_idx = int(choice) - 1
#         selected_model = models[model_idx]
#     except:
#         selected_model = models[0]
    
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
#             mb_downloaded = downloaded / (1024 * 1024)
#             mb_total = total_size / (1024 * 1024)
#             print(f"\r[DOWNLOAD] {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", 
#                   end='', flush=True)
        
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
#         print("Visit: https://github.com/mk-minchul/AdaFace/releases/tag/v1.0")
#         print(f"Download: {selected_model['filename']}")
#         print(f"Place in: models/ folder")
#         print("="*70)
#         return None


# if __name__ == "__main__":
#     print("AdaFace Recognition System Test")
#     print("="*70)
    
#     # Download models
#     download_adaface_models()
    
#     # Test recognition
#     recognizer = FaceRecognizer()
    
#     print("\n✓ System initialized successfully")
#     print(f"✓ Model loaded: {'AdaFace (PyTorch)' if recognizer.model_loaded else 'DeepFace (Fallback)'}")
#     print(f"✓ Device: {recognizer.device}")
#     print(f"✓ Database: {len(recognizer.embeddings_db)} identities")
# -------------------------------------
# ---------------------------------
# """
# AdaFace Recognition System - PyTorch Version
# FIXED: Architecture auto-detection + proper state_dict loading
# Works on Laptop (CUDA/CPU) and Jetson Nano
# """

# import cv2
# import numpy as np
# import pickle
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms


# class FaceRecognizer:
#     def __init__(self, database_path='database/embeddings_adaface.pkl'):
#         """
#         AdaFace Recognition System (PyTorch)
#         Works on: Laptop (CUDA/CPU) and Jetson Nano (CUDA)
#         """
#         print("[RECOGNIZER] Loading AdaFace Recognition System...")
#         print("[RECOGNIZER] Platform: PyTorch (CUDA/CPU compatible)")
        
#         self.database_path = database_path
#         self.embeddings_db = {}
        
#         # Auto-detect device
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#             print(f"[RECOGNIZER] Using GPU: {torch.cuda.get_device_name(0)}")
#         else:
#             self.device = torch.device('cpu')
#             print("[RECOGNIZER] Using CPU")
        
#         # Load AdaFace model
#         self._load_adaface_model()
        
#         # Transform pipeline
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((112, 112)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#         # Load database
#         self.load_database()
        
#         print(f"[RECOGNIZER] System ready with {len(self.embeddings_db)} identities")
    
#     def _detect_architecture_from_checkpoint(self, checkpoint_path):
#         """
#         Auto-detect model architecture from checkpoint
#         Returns: 'ir_50' or 'ir_101'
#         """
#         try:
#             checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
            
#             # Check a specific layer to determine architecture
#             # IR-50 has body.7.res_layer.1.weight with shape [256, 128, 3, 3]
#             # IR-101 has body.7.res_layer.1.weight with shape [128, 128, 3, 3]
            
#             for key in state_dict.keys():
#                 if 'body.7.res_layer.1.weight' in key or 'body.7.res_layer.1.weight' == key.replace('model.', '').replace('module.', ''):
#                     # Get the actual key (might have prefixes)
#                     actual_key = key
#                     shape = state_dict[actual_key].shape
                    
#                     print(f"[DEBUG] Detected layer shape: {shape}")
                    
#                     if shape[0] == 256:
#                         print("[DEBUG] Architecture detected: IR-50")
#                         return 'ir_50'
#                     elif shape[0] == 128:
#                         print("[DEBUG] Architecture detected: IR-101")
#                         return 'ir_101'
            
#             # Fallback: check file name
#             if 'ir50' in checkpoint_path.lower():
#                 print("[DEBUG] Architecture detected from filename: IR-50")
#                 return 'ir_50'
#             elif 'ir101' in checkpoint_path.lower():
#                 print("[DEBUG] Architecture detected from filename: IR-101")
#                 return 'ir_101'
            
#             # Default to IR-50
#             print("[DEBUG] Could not detect architecture, defaulting to IR-50")
#             return 'ir_50'
            
#         except Exception as e:
#             print(f"[WARNING] Architecture detection failed: {e}")
#             print("[DEBUG] Defaulting to IR-50")
#             return 'ir_50'
    
#     def _load_adaface_model(self):
#         """Load AdaFace PyTorch model with AUTO architecture detection"""
#         try:
#             # Check for available models
#             model_candidates = [
#                 'models/adaface_ir50_ms1mv2.ckpt',
#                 'models/adaface_ir50_webface4m.ckpt',
#                 'models/adaface_ir101_webface12m.ckpt',
#                 'models/adaface_ir18_vgg2.ckpt'
#             ]
            
#             weight_path = None
#             for candidate in model_candidates:
#                 if os.path.exists(candidate):
#                     weight_path = candidate
#                     print(f"[INFO] Found model: {candidate}")
#                     break
            
#             if weight_path is None:
#                 raise FileNotFoundError("No AdaFace model found. Please download a model first.")
            
#             print(f"[RECOGNIZER] Loading AdaFace from: {weight_path}")
            
#             # AUTO-DETECT ARCHITECTURE
#             architecture = self._detect_architecture_from_checkpoint(weight_path)
#             print(f"[RECOGNIZER] Using architecture: {architecture.upper()}")
            
#             # Import AdaFace architecture
#             import sys
#             adaface_lib_path = 'adaface_lib'
#             if os.path.exists(adaface_lib_path):
#                 sys.path.insert(0, adaface_lib_path)
            
#             from net import build_model
            
#             # Build model with CORRECT architecture
#             self.model = build_model(architecture)
            
#             # Load weights
#             print("[RECOGNIZER] Loading checkpoint...")
#             checkpoint = torch.load(weight_path, map_location=self.device)
            
#             # Handle different checkpoint formats
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
            
#             # Remove prefixes
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 name = k.replace('model.', '').replace('module.', '')
#                 new_state_dict[name] = v
            
#             # Load with strict=False
#             missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
#             if missing_keys:
#                 print(f"[WARNING] Missing keys: {len(missing_keys)}")
#             if unexpected_keys:
#                 print(f"[WARNING] Unexpected keys: {len(unexpected_keys)}")
            
#             # Move to device and set eval mode
#             self.model.to(self.device)
#             self.model.eval()
            
#             self.model_loaded = True
#             print("[RECOGNIZER] ✓ AdaFace loaded successfully")
            
#         except Exception as e:
#             print(f"[ERROR] AdaFace loading failed: {e}")
#             import traceback
#             print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            
#             print("[RECOGNIZER] Switching to DeepFace fallback...")
#             self._load_deepface_fallback()
    
#     def _load_deepface_fallback(self):
#         """Fallback to DeepFace ArcFace"""
#         try:
#             from deepface import DeepFace
#             self.deepface = DeepFace
#             self.model_loaded = False
#             print("[RECOGNIZER] Using DeepFace ArcFace as fallback")
            
#             # Pre-load model
#             dummy = np.zeros((112, 112, 3), dtype=np.uint8)
#             _ = DeepFace.represent(dummy, model_name="ArcFace", enforce_detection=False)
#             print("[RECOGNIZER] ✓ ArcFace loaded successfully")
            
#         except Exception as e:
#             raise Exception(f"All recognition models failed: {e}")
    
#     def get_embedding(self, face_image):
#         """
#         Extract 512-dim embedding from face image
#         Args:
#             face_image: BGR face crop
#         Returns:
#             512-dim numpy array or None
#         """
#         try:
#             if self.model_loaded:
#                 # AdaFace PyTorch inference
#                 face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#                 face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
                
#                 with torch.no_grad():
#                     embedding = self.model(face_tensor)
                
#                 embedding = embedding.cpu().numpy()[0]
                
#                 # L2 normalize
#                 embedding = embedding / np.linalg.norm(embedding)
                
#                 return embedding
#             else:
#                 # DeepFace fallback
#                 face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#                 result = self.deepface.represent(
#                     face_rgb,
#                     model_name="ArcFace",
#                     enforce_detection=False,
#                     detector_backend='skip'
#                 )
                
#                 if len(result) > 0:
#                     embedding = np.array(result[0]["embedding"])
#                     embedding = embedding / np.linalg.norm(embedding)
#                     return embedding
                
#                 return None
                
#         except Exception as e:
#             print(f"[DEBUG] Embedding extraction error: {e}")
#             return None
    
#     def preprocess_face(self, face_image):
#         """Enhanced preprocessing"""
#         if len(face_image.shape) == 2:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
#         elif face_image.shape[2] == 4:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
#         else:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
#         h, w = face_image.shape[:2]
#         if h < 112 or w < 112:
#             scale = max(112/h, 112/w) * 1.1
#             new_h, new_w = int(h * scale), int(w * scale)
#             face_image = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
#         img_yuv = cv2.cvtColor(face_image, cv2.COLOR_RGB2YUV)
#         img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#         face_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
#         face_image = cv2.GaussianBlur(face_image, (3, 3), 0)
#         face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
#         return face_image
    
#     def generate_synthetic_poses(self, face_image):
#         """Generate 7 synthetic poses from single photo"""
#         augmented = []
        
#         face_image = self.preprocess_face(face_image)
#         h, w = face_image.shape[:2]
        
#         # 1. Original
#         augmented.append(face_image.copy())
        
#         # 2. Horizontal flip
#         augmented.append(cv2.flip(face_image, 1))
        
#         # 3-6. Rotations
#         for angle in [-10, -5, 5, 10]:
#             M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
#             rotated = cv2.warpAffine(face_image, M, (w, h), 
#                                     borderMode=cv2.BORDER_REPLICATE)
#             augmented.append(rotated)
        
#         # 7. Brightness
#         bright = cv2.convertScaleAbs(face_image, alpha=1.15, beta=0)
#         augmented.append(bright)
        
#         return augmented
    
#     def add_person(self, name, face_image, use_augmentation=True):
#         """Add person with synthetic augmentation"""
#         if use_augmentation:
#             samples = self.generate_synthetic_poses(face_image)
#             samples_desc = f"1 photo → {len(samples)} synthetic poses"
#         else:
#             samples = [self.preprocess_face(face_image)]
#             samples_desc = "1 sample (no augmentation)"
        
#         successful = 0
        
#         for sample in samples:
#             embedding = self.get_embedding(sample)
            
#             if embedding is not None:
#                 if name not in self.embeddings_db:
#                     self.embeddings_db[name] = []
                
#                 self.embeddings_db[name].append(embedding)
#                 successful += 1
        
#         if successful > 0:
#             self.save_database()
#             print(f"[RECOGNIZER] ✓ Added {name} ({samples_desc} → {successful} embeddings)")
#             return True
#         else:
#             print(f"[RECOGNIZER] ✗ Failed to add {name}")
#             return False
    
#     def cosine_similarity(self, emb1, emb2):
#         """Calculate cosine similarity"""
#         if emb1 is None or emb2 is None:
#             return 0.0
#         return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
#     def recognize(self, face_image, threshold=0.42):
#         """Recognize face using AdaFace"""
#         query_embedding = self.get_embedding(face_image)
        
#         if query_embedding is None:
#             return "Unknown", 0.0
        
#         best_match = None
#         best_score = threshold
        
#         for name, stored_embeddings in self.embeddings_db.items():
#             scores = []
            
#             for stored_emb in stored_embeddings:
#                 similarity = self.cosine_similarity(query_embedding, stored_emb)
#                 scores.append(similarity)
            
#             if scores:
#                 max_score = max(scores)
                
#                 if max_score > best_score:
#                     best_score = max_score
#                     best_match = name
        
#         if best_match:
#             return best_match, best_score
#         else:
#             return "Unknown", 0.0
    
#     def load_database(self):
#         """Load embeddings database"""
#         if os.path.exists(self.database_path):
#             with open(self.database_path, 'rb') as f:
#                 self.embeddings_db = pickle.load(f)
#             print(f"[RECOGNIZER] Loaded database: {len(self.embeddings_db)} identities")
#         else:
#             print("[RECOGNIZER] Starting fresh database")
    
#     def save_database(self):
#         """Save embeddings database"""
#         os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
#         with open(self.database_path, 'wb') as f:
#             pickle.dump(self.embeddings_db, f)
    
#     def get_all_registered_persons(self):
#         """Get list of registered persons"""
#         return list(self.embeddings_db.keys())


# # ============================================================================
# # MODEL DOWNLOAD HELPER
# # ============================================================================

# def download_adaface_models():
#     """
#     Download AdaFace models - VERIFIED WORKING LINKS (Oct 2025)
#     """
#     import urllib.request
#     import os
    
#     os.makedirs("models", exist_ok=True)
    
#     models = [
#         {
#             'name': 'AdaFace IR-50 MS1MV2 (RECOMMENDED)',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_ms1mv2.ckpt',
#             'filename': 'adaface_ir50_ms1mv2.ckpt',
#             'size': '~166MB',
#             'architecture': 'IR-50'
#         },
#         {
#             'name': 'AdaFace IR-50 WebFace4M',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.ckpt',
#             'filename': 'adaface_ir50_webface4m.ckpt',
#             'size': '~166MB',
#             'architecture': 'IR-50'
#         },
#         {
#             'name': 'AdaFace IR-18 VGG2 (Lightweight)',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir18_vgg2.ckpt',
#             'filename': 'adaface_ir18_vgg2.ckpt',
#             'size': '~45MB',
#             'architecture': 'IR-18'
#         }
#     ]
    
#     print("\n" + "="*70)
#     print("AdaFace MODEL DOWNLOAD")
#     print("="*70)
#     print("\nAvailable models:")
#     for idx, model in enumerate(models, 1):
#         status = "✓" if os.path.exists(f"models/{model['filename']}") else " "
#         print(f"{status} {idx}. {model['name']}")
#         print(f"     Architecture: {model['architecture']} | Size: {model['size']}")
    
#     choice = input("\nSelect model (1-3, default 1): ").strip() or "1"
    
#     try:
#         model_idx = int(choice) - 1
#         selected_model = models[model_idx]
#     except:
#         selected_model = models[0]
    
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
#             mb_downloaded = downloaded / (1024 * 1024)
#             mb_total = total_size / (1024 * 1024)
#             print(f"\r[DOWNLOAD] {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", 
#                   end='', flush=True)
        
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
#         print("Visit: https://github.com/mk-minchul/AdaFace/releases/tag/v1.0")
#         print(f"Download: {selected_model['filename']}")
#         print(f"Place in: models/ folder")
#         print("="*70)
#         return None


# if __name__ == "__main__":
#     print("AdaFace Recognition System Test")
#     print("="*70)
    
#     # Download models
#     download_adaface_models()
    
#     # Test recognition
#     recognizer = FaceRecognizer()
    
#     print("\n✓ System initialized successfully")
#     print(f"✓ Model loaded: {'AdaFace (PyTorch)' if recognizer.model_loaded else 'DeepFace (Fallback)'}")
#     print(f"✓ Device: {recognizer.device}")
#     print(f"✓ Database: {len(recognizer.embeddings_db)} identities")


# """
# AdaFace Recognition System - PyTorch Version
# FIXED: Handle tuple outputs from AdaFace model
# Works on Laptop (CUDA/CPU) and Jetson Nano
# """

# import cv2
# import numpy as np
# import pickle
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms


# class FaceRecognizer:
#     def __init__(self, database_path='database/embeddings_adaface.pkl'):
#         """
#         AdaFace Recognition System (PyTorch)
#         Works on: Laptop (CUDA/CPU) and Jetson Nano (CUDA)
#         """
#         print("[RECOGNIZER] Loading AdaFace Recognition System...")
#         print("[RECOGNIZER] Platform: PyTorch (CUDA/CPU compatible)")
        
#         self.database_path = database_path
#         self.embeddings_db = {}
        
#         # Auto-detect device
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#             print(f"[RECOGNIZER] Using GPU: {torch.cuda.get_device_name(0)}")
#         else:
#             self.device = torch.device('cpu')
#             print("[RECOGNIZER] Using CPU")
        
#         # Load AdaFace model
#         self._load_adaface_model()
        
#         # Transform pipeline
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((112, 112)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#         # Load database
#         self.load_database()
        
#         print(f"[RECOGNIZER] System ready with {len(self.embeddings_db)} identities")
    
#     def _detect_architecture_from_checkpoint(self, checkpoint_path):
#         """
#         Auto-detect model architecture from checkpoint
#         Returns: 'ir_50' or 'ir_101'
#         """
#         try:
#             checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
            
#             # Check a specific layer to determine architecture
#             # IR-50 has body.7.res_layer.1.weight with shape [256, 128, 3, 3]
#             # IR-101 has body.7.res_layer.1.weight with shape [128, 128, 3, 3]
            
#             for key in state_dict.keys():
#                 if 'body.7.res_layer.1.weight' in key or 'body.7.res_layer.1.weight' == key.replace('model.', '').replace('module.', ''):
#                     # Get the actual key (might have prefixes)
#                     actual_key = key
#                     shape = state_dict[actual_key].shape
                    
#                     print(f"[DEBUG] Detected layer shape: {shape}")
                    
#                     if shape[0] == 256:
#                         print("[DEBUG] Architecture detected: IR-50")
#                         return 'ir_50'
#                     elif shape[0] == 128:
#                         print("[DEBUG] Architecture detected: IR-101")
#                         return 'ir_101'
            
#             # Fallback: check file name
#             if 'ir50' in checkpoint_path.lower():
#                 print("[DEBUG] Architecture detected from filename: IR-50")
#                 return 'ir_50'
#             elif 'ir101' in checkpoint_path.lower():
#                 print("[DEBUG] Architecture detected from filename: IR-101")
#                 return 'ir_101'
            
#             # Default to IR-50
#             print("[DEBUG] Could not detect architecture, defaulting to IR-50")
#             return 'ir_50'
            
#         except Exception as e:
#             print(f"[WARNING] Architecture detection failed: {e}")
#             print("[DEBUG] Defaulting to IR-50")
#             return 'ir_50'
    
#     def _load_adaface_model(self):
#         """Load AdaFace PyTorch model with AUTO architecture detection"""
#         try:
#             # Check for available models
#             model_candidates = [
#                 'models/adaface_ir50_ms1mv2.ckpt',
#                 'models/adaface_ir50_webface4m.ckpt',
#                 'models/adaface_ir101_webface12m.ckpt',
#                 'models/adaface_ir18_vgg2.ckpt'
#             ]
            
#             weight_path = None
#             for candidate in model_candidates:
#                 if os.path.exists(candidate):
#                     weight_path = candidate
#                     print(f"[INFO] Found model: {candidate}")
#                     break
            
#             if weight_path is None:
#                 raise FileNotFoundError("No AdaFace model found. Please download a model first.")
            
#             print(f"[RECOGNIZER] Loading AdaFace from: {weight_path}")
            
#             # AUTO-DETECT ARCHITECTURE
#             architecture = self._detect_architecture_from_checkpoint(weight_path)
#             print(f"[RECOGNIZER] Using architecture: {architecture.upper()}")
            
#             # Import AdaFace architecture
#             import sys
#             adaface_lib_path = 'adaface_lib'
#             if os.path.exists(adaface_lib_path):
#                 sys.path.insert(0, adaface_lib_path)
            
#             from net import build_model
            
#             # Build model with CORRECT architecture
#             self.model = build_model(architecture)
            
#             # Load weights
#             print("[RECOGNIZER] Loading checkpoint...")
#             checkpoint = torch.load(weight_path, map_location=self.device)
            
#             # Handle different checkpoint formats
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
            
#             # Remove prefixes
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 name = k.replace('model.', '').replace('module.', '')
#                 new_state_dict[name] = v
            
#             # Load with strict=False
#             missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
#             if missing_keys:
#                 print(f"[WARNING] Missing keys: {len(missing_keys)}")
#             if unexpected_keys:
#                 print(f"[WARNING] Unexpected keys: {len(unexpected_keys)}")
            
#             # Move to device and set eval mode
#             self.model.to(self.device)
#             self.model.eval()
            
#             self.model_loaded = True
#             print("[RECOGNIZER] ✓ AdaFace loaded successfully")
            
#         except Exception as e:
#             print(f"[ERROR] AdaFace loading failed: {e}")
#             import traceback
#             print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            
#             print("[RECOGNIZER] Switching to DeepFace fallback...")
#             self._load_deepface_fallback()
    
#     def _load_deepface_fallback(self):
#         """Fallback to DeepFace ArcFace"""
#         try:
#             from deepface import DeepFace
#             self.deepface = DeepFace
#             self.model_loaded = False
#             print("[RECOGNIZER] Using DeepFace ArcFace as fallback")
            
#             # Pre-load model
#             dummy = np.zeros((112, 112, 3), dtype=np.uint8)
#             _ = DeepFace.represent(dummy, model_name="ArcFace", enforce_detection=False)
#             print("[RECOGNIZER] ✓ ArcFace loaded successfully")
            
#         except Exception as e:
#             raise Exception(f"All recognition models failed: {e}")
    
#     def get_embedding(self, face_image):
#         """
#         Extract 512-dim embedding from face image
#         Args:
#             face_image: BGR face crop
#         Returns:
#             512-dim numpy array or None
#         """
#         try:
#             if self.model_loaded:
#                 # AdaFace PyTorch inference
#                 face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#                 face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
                
#                 with torch.no_grad():
#                     output = self.model(face_tensor)
                    
#                     # CRITICAL FIX: Handle tuple outputs
#                     if isinstance(output, tuple):
#                         # AdaFace returns (embedding, norm) tuple
#                         embedding = output[0]
#                     else:
#                         embedding = output
                    
#                     # Convert to numpy
#                     embedding = embedding.cpu().numpy()[0]
                
#                 # L2 normalize
#                 embedding = embedding / np.linalg.norm(embedding)
                
#                 return embedding
#             else:
#                 # DeepFace fallback
#                 face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#                 result = self.deepface.represent(
#                     face_rgb,
#                     model_name="ArcFace",
#                     enforce_detection=False,
#                     detector_backend='skip'
#                 )
                
#                 if len(result) > 0:
#                     embedding = np.array(result[0]["embedding"])
#                     embedding = embedding / np.linalg.norm(embedding)
#                     return embedding
                
#                 return None
                
#         except Exception as e:
#             print(f"[DEBUG] Embedding extraction error: {e}")
#             import traceback
#             print(f"[DEBUG] Traceback: {traceback.format_exc()}")
#             return None
    
#     def preprocess_face(self, face_image):
#         """Enhanced preprocessing"""
#         if len(face_image.shape) == 2:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
#         elif face_image.shape[2] == 4:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
#         else:
#             face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
#         h, w = face_image.shape[:2]
#         if h < 112 or w < 112:
#             scale = max(112/h, 112/w) * 1.1
#             new_h, new_w = int(h * scale), int(w * scale)
#             face_image = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
#         img_yuv = cv2.cvtColor(face_image, cv2.COLOR_RGB2YUV)
#         img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#         face_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
#         face_image = cv2.GaussianBlur(face_image, (3, 3), 0)
#         face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
#         return face_image
    
#     def generate_synthetic_poses(self, face_image):
#         """Generate 7 synthetic poses from single photo"""
#         augmented = []
        
#         face_image = self.preprocess_face(face_image)
#         h, w = face_image.shape[:2]
        
#         # 1. Original
#         augmented.append(face_image.copy())
        
#         # 2. Horizontal flip
#         augmented.append(cv2.flip(face_image, 1))
        
#         # 3-6. Rotations
#         for angle in [-10, -5, 5, 10]:
#             M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
#             rotated = cv2.warpAffine(face_image, M, (w, h), 
#                                     borderMode=cv2.BORDER_REPLICATE)
#             augmented.append(rotated)
        
#         # 7. Brightness
#         bright = cv2.convertScaleAbs(face_image, alpha=1.15, beta=0)
#         augmented.append(bright)
        
#         return augmented
    
#     def add_person(self, name, face_image, use_augmentation=True):
#         """Add person with synthetic augmentation"""
#         if use_augmentation:
#             samples = self.generate_synthetic_poses(face_image)
#             samples_desc = f"1 photo → {len(samples)} synthetic poses"
#         else:
#             samples = [self.preprocess_face(face_image)]
#             samples_desc = "1 sample (no augmentation)"
        
#         successful = 0
        
#         for sample in samples:
#             embedding = self.get_embedding(sample)
            
#             if embedding is not None:
#                 if name not in self.embeddings_db:
#                     self.embeddings_db[name] = []
                
#                 self.embeddings_db[name].append(embedding)
#                 successful += 1
        
#         if successful > 0:
#             self.save_database()
#             print(f"[RECOGNIZER] ✓ Added {name} ({samples_desc} → {successful} embeddings)")
#             return True
#         else:
#             print(f"[RECOGNIZER] ✗ Failed to add {name}")
#             return False
    
#     def cosine_similarity(self, emb1, emb2):
#         """Calculate cosine similarity"""
#         if emb1 is None or emb2 is None:
#             return 0.0
#         return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
#     def recognize(self, face_image, threshold=0.42):
#         """Recognize face using AdaFace"""
#         query_embedding = self.get_embedding(face_image)
        
#         if query_embedding is None:
#             return "Unknown", 0.0
        
#         best_match = None
#         best_score = threshold
        
#         for name, stored_embeddings in self.embeddings_db.items():
#             scores = []
            
#             for stored_emb in stored_embeddings:
#                 similarity = self.cosine_similarity(query_embedding, stored_emb)
#                 scores.append(similarity)
            
#             if scores:
#                 max_score = max(scores)
                
#                 if max_score > best_score:
#                     best_score = max_score
#                     best_match = name
        
#         if best_match:
#             return best_match, best_score
#         else:
#             return "Unknown", 0.0
    
#     def load_database(self):
#         """Load embeddings database"""
#         if os.path.exists(self.database_path):
#             with open(self.database_path, 'rb') as f:
#                 self.embeddings_db = pickle.load(f)
#             print(f"[RECOGNIZER] Loaded database: {len(self.embeddings_db)} identities")
#         else:
#             print("[RECOGNIZER] Starting fresh database")
    
#     def save_database(self):
#         """Save embeddings database"""
#         os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
#         with open(self.database_path, 'wb') as f:
#             pickle.dump(self.embeddings_db, f)
    
#     def get_all_registered_persons(self):
#         """Get list of registered persons"""
#         return list(self.embeddings_db.keys())


# # ============================================================================
# # MODEL DOWNLOAD HELPER
# # ============================================================================

# def download_adaface_models():
#     """
#     Download AdaFace models - VERIFIED WORKING LINKS (Oct 2025)
#     """
#     import urllib.request
#     import os
    
#     os.makedirs("models", exist_ok=True)
    
#     models = [
#         {
#             'name': 'AdaFace IR-50 MS1MV2 (RECOMMENDED)',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_ms1mv2.ckpt',
#             'filename': 'adaface_ir50_ms1mv2.ckpt',
#             'size': '~166MB',
#             'architecture': 'IR-50'
#         },
#         {
#             'name': 'AdaFace IR-50 WebFace4M',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.ckpt',
#             'filename': 'adaface_ir50_webface4m.ckpt',
#             'size': '~166MB',
#             'architecture': 'IR-50'
#         },
#         {
#             'name': 'AdaFace IR-18 VGG2 (Lightweight)',
#             'url': 'https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir18_vgg2.ckpt',
#             'filename': 'adaface_ir18_vgg2.ckpt',
#             'size': '~45MB',
#             'architecture': 'IR-18'
#         }
#     ]
    
#     print("\n" + "="*70)
#     print("AdaFace MODEL DOWNLOAD")
#     print("="*70)
#     print("\nAvailable models:")
#     for idx, model in enumerate(models, 1):
#         status = "✓" if os.path.exists(f"models/{model['filename']}") else " "
#         print(f"{status} {idx}. {model['name']}")
#         print(f"     Architecture: {model['architecture']} | Size: {model['size']}")
    
#     choice = input("\nSelect model (1-3, default 1): ").strip() or "1"
    
#     try:
#         model_idx = int(choice) - 1
#         selected_model = models[model_idx]
#     except:
#         selected_model = models[0]
    
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
#             mb_downloaded = downloaded / (1024 * 1024)
#             mb_total = total_size / (1024 * 1024)
#             print(f"\r[DOWNLOAD] {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", 
#                   end='', flush=True)
        
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
#         print("Visit: https://github.com/mk-minchul/AdaFace/releases/tag/v1.0")
#         print(f"Download: {selected_model['filename']}")
#         print(f"Place in: models/ folder")
#         print("="*70)
#         return None


# if __name__ == "__main__":
#     print("AdaFace Recognition System Test")
#     print("="*70)
    
#     # Download models
#     download_adaface_models()
    
#     # Test recognition
#     recognizer = FaceRecognizer()
    
#     print("\n✓ System initialized successfully")
#     print(f"✓ Model loaded: {'AdaFace (PyTorch)' if recognizer.model_loaded else 'DeepFace (Fallback)'}")
#     print(f"✓ Device: {recognizer.device}")
#     print(f"✓ Database: {len(recognizer.embeddings_db)} identities")

# face_recognizer.py
# -----------------------------------------------------------------------------
# AdaFace IR-101 (WebFace12M) face embedding + recognition pipeline
# Robust loader (auto-architecture detect), strict normalization,
# augmentation (optional), adaptive thresholds per mode, and a simple
# persistent database using pickle.
# -----------------------------------------------------------------------------

import os
import cv2
import sys
import math
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms


# =============================================================================
# Utilities
# =============================================================================

def _ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Always return RGB uint8 image."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    if len(img_bgr.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n


# =============================================================================
# AdaFace model loader
# Expect a local adaface library folder (e.g., ./adaface_lib) containing
# the official architectures. This matches the structure many people use:
#   adaface_lib/
#       net.py  (has build_model(arch: str) -> torch.nn.Module)
# If yours is different, just adapt the import block below.
# =============================================================================

def _detect_architecture_from_ckpt(ckpt_path: str) -> str:
    """
    Infer the correct AdaFace backbone from the checkpoint tensors.
    Returns one of: 'ir_18', 'ir_50', 'ir_101'
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = None
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                # already a pure state dict
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Heuristic based on common layer channel sizes
        # Look for a middle layer that differs across variants
        for k, v in state_dict.items():
            name = k.replace('model.', '').replace('module.', '')
            if name.endswith('body.7.res_layer.1.weight'):
                shape0 = v.shape[0]
                if shape0 == 256:
                    return 'ir_50'
                if shape0 == 128:
                    return 'ir_101'
        # Fall back to filename hints
        low = ckpt_path.lower()
        if 'ir18' in low or 'ir_18' in low:
            return 'ir_18'
        if 'ir101' in low or 'ir_101' in low or 'webface12m' in low:
            return 'ir_101'
        if 'ir50' in low or 'ir_50' in low or 'ms1mv2' in low or 'webface4m' in low:
            return 'ir_50'
    except Exception as e:
        print(f"[AdaFace] Architecture detection warning: {e}")

    # Default
    return 'ir_101'


def _build_adaface_model(arch: str):
    """
    Import build_model from your AdaFace lib.
    Expected path: ./adaface_lib/net.py with function build_model(arch: str)
    """
    adaface_lib_path = 'adaface_lib'
    if os.path.isdir(adaface_lib_path) and adaface_lib_path not in sys.path:
        sys.path.insert(0, adaface_lib_path)

    try:
        from net import build_model  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import AdaFace model builder. "
            "Place the official AdaFace repo (net.py) in 'adaface_lib/'. "
            f"Original error: {e}"
        )
    return build_model(arch)


# =============================================================================
# FaceRecognizer
# =============================================================================

class FaceRecognizer:
    """
    AdaFace IR-101 / IR-50 / IR-18 recognizer with:
      - strict embedding normalization
      - adaptive thresholds per mode ('image', 'video', 'live')
      - optional augmentation on enrollment
      - pickle-based embedding database
    """

    def __init__(
        self,
        database_path: str = "database/embeddings_adaface.pkl",
        model_ckpt: str = "models/adaface_ir101_webface12m.ckpt",
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.database_path = database_path
        self.model_ckpt = model_ckpt
        self.verbose = verbose

        # Select device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.verbose:
            print("=" * 80)
            print("AdaFace Recognizer - Loader")
            print("=" * 80)
            print(f"[Device] {self.device}")

        # Build model with correct architecture
        arch = _detect_architecture_from_ckpt(self.model_ckpt)
        if self.verbose:
            print(f"[AdaFace] Detected architecture: {arch}")

        self.model: nn.Module = _build_adaface_model(arch)
        self._load_weights(self.model_ckpt)
        self.model.eval().to(self.device)

        # Preprocess to match AdaFace training
        self.transform = transforms.Compose([
            transforms.ToTensor(),                   # HWC uint8 -> CHW float [0,1]
            transforms.Resize((112, 112)),          # AdaFace input size
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]) # [-1,1] space
        ])

        # Load database
        self.db: Dict[str, List[np.ndarray]] = {}
        self._load_database()

        if self.verbose:
            print(f"[DB] Loaded identities: {len(self.db)}")
            print("=" * 80)

    # -------------------------------------------------------------------------
    # Model weights
    # -------------------------------------------------------------------------
    def _load_weights(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"AdaFace checkpoint not found: {ckpt_path}\n"
                "Download from: https://github.com/mk-minchul/AdaFace/releases"
            )

        if self.verbose:
            print(f"[AdaFace] Loading weights: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip common prefixes
        new_state = {}
        for k, v in state_dict.items():
            k2 = k.replace('model.', '').replace('module.', '')
            new_state[k2] = v

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        if self.verbose:
            if missing:
                print(f"[AdaFace] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[AdaFace] Unexpected keys: {len(unexpected)}")
            print("[AdaFace] Weights loaded ✓")

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    def _load_database(self):
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        if os.path.isfile(self.database_path):
            try:
                with open(self.database_path, "rb") as f:
                    self.db = pickle.load(f)

                # Ensure all vectors are L2-normalized
                for k, lst in self.db.items():
                    self.db[k] = [l2_normalize(v) for v in lst if v is not None]
                if self.verbose:
                    print(f"[DB] {self.database_path} loaded.")
            except Exception as e:
                print(f"[DB] Failed to load DB ({e}). Starting fresh.")
                self.db = {}
        else:
            self.db = {}
            if self.verbose:
                print(f"[DB] No existing DB. New database at {self.database_path}.")

    def save_database(self):
        with open(self.database_path, "wb") as f:
            pickle.dump(self.db, f)
        if self.verbose:
            print(f"[DB] Saved to {self.database_path} ✓")

    # -------------------------------------------------------------------------
    # Preprocess / Augmentation
    # -------------------------------------------------------------------------
    def _light_preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        A conservative enhancement used for VIDEO/LIVE frames
        to combat compression and slight motion blur.
        """
        if face_rgb is None or face_rgb.size == 0:
            return face_rgb
        # Gentle denoise (edge-preserving)
        face_rgb = cv2.bilateralFilter(face_rgb, 3, 75, 75)
        # Slight contrast & brightness tweak
        face_rgb = cv2.convertScaleAbs(face_rgb, alpha=1.10, beta=3)
        return face_rgb

    def _augment(self, face_rgb: np.ndarray) -> List[np.ndarray]:
        """
        Simple, quality-safe augmentations for enrollment to diversify embedding.
        """
        aug = []
        h, w = face_rgb.shape[:2]

        # Base
        aug.append(face_rgb.copy())

        # Horiz flip
        aug.append(cv2.flip(face_rgb, 1))

        # Small rotations
        for angle in (-8, -4, 4, 8):
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rot = cv2.warpAffine(face_rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            aug.append(rot)

        # Mild brightness change
        for alpha in (0.9, 1.1):
            tmp = cv2.convertScaleAbs(face_rgb, alpha=alpha, beta=0)
            aug.append(tmp)

        return aug

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    def _embed_tensor(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """Prepare input tensor (1,3,112,112) for AdaFace."""
        t = self.transform(rgb_uint8).unsqueeze(0)  # [1,3,112,112]
        return t.to(self.device)

    @torch.no_grad()
    def get_embedding(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-d normalized embedding from an RGB face crop.
        Returns None on failure.
        """
        if face_rgb is None or face_rgb.size == 0:
            return None

        # Safety: some callers may pass BGR by mistake — auto-fix if needed
        # (Heuristic: if the mean in channel 0 >> channel 2, likely BGR; but we
        # keep it simple: assume caller passed RGB since our pipeline enforces it.)
        try:
            tensor = self._embed_tensor(face_rgb)
            output = self.model(tensor)
            # AdaFace returns (embedding, norms) tuple
            if isinstance(output, tuple):
                emb = output[0]
            else:
                emb = output
            vec = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
            vec = l2_normalize(vec)
            return vec
        except Exception as e:
            print(f"[AdaFace] Embedding error: {e}")
            return None

    # -------------------------------------------------------------------------
    # Enrollment
    # -------------------------------------------------------------------------
    def add_person(
        self,
        name: str,
        face_bgr: np.ndarray,
        augment: bool = True,
        preprocess_for_video: bool = False
    ) -> bool:
        """
        Add a new person to the database.
        - face_bgr: any BGR face crop (we convert to RGB internally)
        - augment: if True, store several embeddings from safe augmentations
        - preprocess_for_video: if True, apply light denoise/contrast boost
          used in video/live paths to keep enrollment domain similar
        """
        if not name or face_bgr is None or face_bgr.size == 0:
            return False

        face_rgb = _ensure_rgb(face_bgr)
        if preprocess_for_video:
            face_rgb = self._light_preprocess(face_rgb)

        samples = self._augment(face_rgb) if augment else [face_rgb]

        success = 0
        for img in samples:
            emb = self.get_embedding(img)
            if emb is not None:
                if name not in self.db:
                    self.db[name] = []
                # Always store unit vectors
                self.db[name].append(l2_normalize(emb))
                success += 1

        if success > 0:
            self.save_database()
            if self.verbose:
                print(f"[Enroll] {name}: stored {success} embedding(s)")
            return True

        if self.verbose:
            print(f"[Enroll] {name}: FAILED (no valid embedding)")
        return False

    # -------------------------------------------------------------------------
    # Recognition
    # -------------------------------------------------------------------------
    def recognize(
        self,
        face_bgr_or_rgb: np.ndarray,
        mode: str = "image"
    ) -> Tuple[str, float]:
        """
        Recognize a face crop and return (best_name, confidence).
        mode:
          - 'image' : single image testing (highest threshold)
          - 'video' : compressed frames (lower threshold)
          - 'live'  : webcam/rtsp (middle threshold)
        """
        if face_bgr_or_rgb is None or face_bgr_or_rgb.size == 0:
            return "Unknown", 0.0

        # Always operate in RGB
        face_rgb = face_bgr_or_rgb
        if face_rgb.ndim == 3 and face_rgb.shape[2] in (3, 4):
            # If user passed BGR, convert to RGB
            # We can’t reliably detect colorspace from content, so we enforce:
            # assume BGR and convert; if already RGB, the cost is minimal.
            face_rgb = _ensure_rgb(face_rgb)

        # For video/live, apply gentle stabilization preprocessing
        if mode in ("video", "live"):
            face_rgb = self._light_preprocess(face_rgb)

        q = self.get_embedding(face_rgb)
        if q is None:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_sim = 0.0

        # Compare to all refs (use average of top-K or mean; here mean)
        for name, refs in self.db.items():
            if not refs:
                continue
            sims = [float(np.dot(q, r)) for r in refs]  # both are unit vectors
            mean_sim = float(np.mean(sims))
            if mean_sim > best_sim:
                best_sim = mean_sim
                best_name = name

        # Adaptive thresholds (tune if needed)
        if mode == "video":
            threshold = 0.35
        elif mode == "live":
            threshold = 0.40
        else:  # 'image'
            threshold = 0.50

        if best_sim < threshold:
            return "Unknown", best_sim

        return best_name, best_sim

    # -------------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------------
    def get_all_registered(self) -> List[str]:
        return sorted(list(self.db.keys()))

    def remove_person(self, name: str) -> bool:
        if name in self.db:
            del self.db[name]
            self.save_database()
            return True
        return False

    def clear_database(self):
        self.db = {}
        self.save_database()
        if self.verbose:
            print("[DB] Cleared.")


# =============================================================================
# Standalone test (optional)
# =============================================================================
if __name__ == "__main__":
    # Quick smoke test (requires a model file and a sample face crop)
    recognizer = FaceRecognizer(
        database_path="database/embeddings_adaface.pkl",
        model_ckpt="models/adaface_ir101_webface12m.ckpt",
        verbose=True
    )
    print("Registered:", recognizer.get_all_registered())
