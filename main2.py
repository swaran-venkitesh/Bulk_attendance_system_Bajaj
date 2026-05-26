# """
# DUAL CAMERA ATTENDANCE SYSTEM FOR JETSON NANO
# Optimized for 30-40 simultaneous faces
# Entry + Exit cameras with thread-safe processing
# """

# import cv2
# import numpy as np
# from datetime import datetime
# import time
# import threading
# from queue import Queue, Empty
# import os
# import glob
# from tkinter import Tk, filedialog

# from face_detector import FaceDetector
# from face_recognizer import FaceRecognizer
# from attendance_logger import AttendanceLogger


# def draw_label_with_background(frame, text, position, color, font_scale=0.8, thickness=2):
#     """Professional label with background"""
#     x, y = position
#     font = cv2.FONT_HERSHEY_SIMPLEX
    
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
#     padding_x, padding_y = 12, 8
#     frame_h, frame_w = frame.shape[:2]
    
#     # Adjust position if out of bounds
#     if y - text_height - padding_y * 2 < 0:
#         y = text_height + padding_y * 2 + 5
#     if x + text_width + padding_x * 2 > frame_w:
#         x = frame_w - text_width - padding_x * 2 - 5
#     if x < 5:
#         x = 5
    
#     # Background rectangle
#     bg_x1 = x - padding_x
#     bg_y1 = y - text_height - padding_y
#     bg_x2 = x + text_width + padding_x
#     bg_y2 = y + baseline + padding_y
    
#     # Semi-transparent background
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
#     cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
#     # Border
#     cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
    
#     # Text in BLACK
#     cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
#     return (bg_x1, bg_y1, bg_x2, bg_y2)


# class DualCameraAttendanceSystem:
#     def __init__(self):
#         """
#         Dual camera system for Jetson Nano
#         Handles entry + exit cameras simultaneously
#         """
#         print("=" * 70)
#         print("DUAL CAMERA ATTENDANCE SYSTEM - JETSON NANO")
#         print("=" * 70)
        
#         # Shared components
#         print("\n[1/3] Loading Face Detector (SCRFD)...")
#         self.detector = FaceDetector()
        
#         print("\n[2/3] Loading Face Recognizer (AdaFace)...")
#         self.recognizer = FaceRecognizer()
        
#         print("\n[3/3] Loading Attendance Logger...")
#         self.logger = AttendanceLogger()
        
#         # Thread-safe queues for each camera
#         self.entry_queue = Queue(maxsize=10)
#         self.exit_queue = Queue(maxsize=10)
        
#         # Thread control
#         self.running = False
#         self.threads = []
        
#         # Attendance tracking (thread-safe)
#         self.attendance_lock = threading.Lock()
#         self.attendance_log = []
#         self.last_seen = {}  # {name: timestamp} for cooldown
        
#         print("\n" + "=" * 70)
#         print("SYSTEM READY")
#         print("=" * 70)
    
#     def capture_camera(self, camera_id, queue, camera_type):
#         """
#         Camera capture thread
#         Continuously reads frames and puts them in queue
#         """
#         print(f"[{camera_type.upper()}] Starting camera {camera_id}...")
        
#         cap = cv2.VideoCapture(camera_id)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         cap.set(cv2.CAP_PROP_FPS, 30)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
#         if not cap.isOpened():
#             print(f"[{camera_type.upper()}] ERROR: Cannot open camera {camera_id}")
#             return
        
#         print(f"[{camera_type.upper()}] Camera {camera_id} started successfully")
        
#         frame_count = 0
        
#         while self.running:
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"[{camera_type.upper()}] ERROR: Failed to read frame")
#                 time.sleep(0.1)
#                 continue
            
#             frame_count += 1
            
#             # Skip frames to reduce processing load (process every 2nd frame)
#             if frame_count % 2 != 0:
#                 continue
            
#             # Put frame in queue (non-blocking)
#             try:
#                 queue.put((time.time(), frame, camera_type), block=False)
#             except:
#                 # Queue full, skip this frame
#                 pass
        
#         cap.release()
#         print(f"[{camera_type.upper()}] Camera {camera_id} stopped")
    
#     def process_camera(self, queue, camera_type):
#         """
#         Processing thread for one camera
#         Detects and recognizes faces from queue
#         """
#         print(f"[{camera_type.upper()}] Starting processor...")
        
#         while self.running:
#             try:
#                 # Get frame from queue (timeout 1 second)
#                 timestamp, frame, cam_type = queue.get(timeout=1.0)
                
#                 # Detect faces
#                 faces = self.detector.detect_faces(frame)
                
#                 if len(faces) == 0:
#                     continue
                
#                 # Extract face crops
#                 face_crops = self.detector.extract_face_crops(frame, faces)
                
#                 # Recognize each face
#                 for face_data in face_crops:
#                     face_img = face_data['image']
#                     quality = face_data.get('quality', 0.0)
                    
#                     # Skip low quality faces
#                     if quality < 0.3:
#                         continue
                    
#                     # Recognize
#                     name, confidence = self.recognizer.recognize(face_img)
                    
#                     # Log if recognized
#                     if name != "Unknown" and confidence > 0.42:
#                         self.log_attendance(name, confidence, cam_type, timestamp)
                
#             except Empty:
#                 # Queue empty, continue
#                 continue
#             except Exception as e:
#                 print(f"[{camera_type.upper()}] Processing error: {e}")
#                 continue
    
#     def log_attendance(self, name, confidence, camera_type, timestamp, cooldown=10):
#         """
#         Thread-safe attendance logging with cooldown
#         """
#         with self.attendance_lock:
#             current_time = time.time()
            
#             # Check cooldown
#             if name in self.last_seen:
#                 time_diff = current_time - self.last_seen[name]
#                 if time_diff < cooldown:
#                     return  # Skip (within cooldown)
            
#             # Update last seen
#             self.last_seen[name] = current_time
            
#             # Create log entry
#             log_entry = {
#                 'name': name,
#                 'confidence': confidence,
#                 'timestamp': current_time,
#                 'datetime': datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
#                 'type': camera_type
#             }
            
#             self.attendance_log.append(log_entry)
            
#             # Mark in Excel logger
#             self.logger.mark_present(name)
            
#             # Save to text file
#             os.makedirs('logs', exist_ok=True)
#             with open(f'logs/attendance_{camera_type}.txt', 'a') as f:
#                 f.write(f"{log_entry['datetime']} | {name} | {confidence:.2f} | {camera_type}\n")
            
#             print(f"[ATTENDANCE] {name} marked present at {camera_type} ({log_entry['datetime']})")
    
#     def display_camera(self, camera_id, queue, camera_type, window_position=(0, 0)):
#         """
#         Display thread for one camera with annotations
#         """
#         print(f"[{camera_type.upper()}] Starting display...")
        
#         window_name = f'Attendance System - {camera_type.upper()}'
#         cv2.namedWindow(window_name)
#         cv2.moveWindow(window_name, window_position[0], window_position[1])
        
#         fps_counter = 0
#         fps_start_time = time.time()
#         fps_display = 0
        
#         last_frame = None
        
#         while self.running:
#             try:
#                 # Try to get latest frame from queue
#                 while not queue.empty():
#                     try:
#                         last_frame = queue.get_nowait()
#                     except Empty:
#                         break
                
#                 if last_frame is None:
#                     time.sleep(0.01)
#                     continue
                
#                 timestamp, frame, cam_type = last_frame
#                 display_frame = frame.copy()
                
#                 # Detect and draw faces
#                 faces = self.detector.detect_faces(frame)
#                 face_crops = self.detector.extract_face_crops(frame, faces)
                
#                 # Process and draw each face
#                 for face_data in face_crops:
#                     face_img = face_data['image']
#                     bbox = face_data['bbox']
#                     quality = face_data.get('quality', 0.0)
#                     x1, y1, x2, y2 = bbox
                    
#                     # Recognize
#                     name, confidence = self.recognizer.recognize(face_img)
                    
#                     # Draw based on result
#                     if name != "Unknown" and confidence > 0.42:
#                         color = (0, 255, 0)  # Green
#                         label = name
#                         conf_text = f"{confidence:.2f}"
                        
#                         cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
#                         draw_label_with_background(display_frame, label, (x1, y1-15),
#                                                   color, font_scale=0.7, thickness=2)
#                         draw_label_with_background(display_frame, conf_text, (x1, y2+25),
#                                                   color, font_scale=0.6, thickness=1)
#                     else:
#                         color = (0, 165, 255)  # Orange
#                         label = "Unknown"
                        
#                         cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
#                         draw_label_with_background(display_frame, label, (x1, y1-15),
#                                                   color, font_scale=0.7, thickness=2)
                
#                 # Calculate FPS
#                 fps_counter += 1
#                 if time.time() - fps_start_time >= 1.0:
#                     fps_display = fps_counter
#                     fps_counter = 0
#                     fps_start_time = time.time()
                
#                 # Display stats
#                 stats_bg = display_frame[0:100, 0:400].copy()
#                 stats_bg[:] = (40, 40, 40)
#                 cv2.addWeighted(stats_bg, 0.7, display_frame[0:100, 0:400], 0.3, 0, display_frame[0:100, 0:400])
                
#                 cv2.putText(display_frame, f"Faces: {len(faces)}", (15, 35),
#                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
#                 cv2.putText(display_frame, f"Type: {camera_type}", (15, 75),
#                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
#                 # Display FPS
#                 cv2.putText(display_frame, f"FPS: {fps_display}", (15, display_frame.shape[0]-20),
#                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
#                 cv2.imshow(window_name, display_frame)
                
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     self.running = False
#                     break
                
#             except Exception as e:
#                 print(f"[{camera_type.upper()}] Display error: {e}")
#                 time.sleep(0.1)
        
#         cv2.destroyWindow(window_name)
#         print(f"[{camera_type.upper()}] Display stopped")
    
#     def run_dual_camera(self, entry_cam_id=0, exit_cam_id=1):
#         """
#         Run dual camera system with threading
#         """
#         # Start attendance session
#         registered_persons = self.recognizer.get_all_registered_persons()
#         self.logger.start_session(registered_persons)
        
#         print("\n" + "=" * 70)
#         print("STARTING DUAL CAMERA SYSTEM")
#         print("=" * 70)
#         print(f"Entry Camera: {entry_cam_id}")
#         print(f"Exit Camera: {exit_cam_id}")
#         print("Press 'q' in any window to quit")
#         print("=" * 70 + "\n")
        
#         self.running = True
        
#         # Create threads
#         threads = [
#             # Capture threads
#             threading.Thread(target=self.capture_camera, 
#                            args=(entry_cam_id, self.entry_queue, 'entry'), 
#                            daemon=True),
#             threading.Thread(target=self.capture_camera, 
#                            args=(exit_cam_id, self.exit_queue, 'exit'), 
#                            daemon=True),
            
#             # Processing threads
#             threading.Thread(target=self.process_camera, 
#                            args=(self.entry_queue, 'entry'), 
#                            daemon=True),
#             threading.Thread(target=self.process_camera, 
#                            args=(self.exit_queue, 'exit'), 
#                            daemon=True),
            
#             # Display threads
#             threading.Thread(target=self.display_camera, 
#                            args=(entry_cam_id, self.entry_queue, 'entry', (0, 0)), 
#                            daemon=True),
#             threading.Thread(target=self.display_camera, 
#                            args=(exit_cam_id, self.exit_queue, 'exit', (700, 0)), 
#                            daemon=True),
#         ]
        
#         # Start all threads
#         for thread in threads:
#             thread.start()
#             time.sleep(0.5)  # Stagger startup
        
#         # Wait for threads
#         try:
#             for thread in threads:
#                 thread.join()
#         except KeyboardInterrupt:
#             print("\n[SYSTEM] Interrupted by user")
#             self.running = False
        
#         # Export results
#         self._export_attendance()
    
#     def _export_attendance(self):
#         """Export attendance to Excel"""
#         print("\n" + "=" * 70)
#         print("EXPORTING ATTENDANCE RESULTS")
#         print("=" * 70)
        
#         excel_path = self.logger.export_to_excel()
#         summary = self.logger.get_summary()
        
#         print(f"\n[SUMMARY]")
#         print(f"  Total Registered: {summary['total']}")
#         print(f"  Present: {summary['present']}")
#         print(f"  Absent: {summary['absent']}")
#         print(f"\n[EXCEL] Saved to: {excel_path}")
        
#         # Save dual camera log
#         with open('logs/dual_camera_log.txt', 'w') as f:
#             f.write("DUAL CAMERA ATTENDANCE LOG\n")
#             f.write("=" * 70 + "\n\n")
            
#             for log in self.attendance_log:
#                 f.write(f"{log['datetime']} | {log['name']} | {log['confidence']:.2f} | {log['type']}\n")
        
#         print(f"[LOG] Saved to: logs/dual_camera_log.txt")
#         print("=" * 70 + "\n")


# class SingleCameraAttendanceSystem:
#     """Single camera system (for testing or single location)"""
    
#     def __init__(self, camera_type='entry'):
#         """Initialize single camera system"""
#         self.detector = FaceDetector()
#         self.recognizer = FaceRecognizer()
#         self.logger = AttendanceLogger()
#         self.camera_type = camera_type
#         self.attendance_log = []
        
#         print(f"[SYSTEM] Single Camera System initialized for {camera_type}")
    
#     def process_frame(self, frame):
#         """Process single frame"""
#         # Detect faces
#         faces = self.detector.detect_faces(frame)
#         face_crops = self.detector.extract_face_crops(frame, faces)
        
#         results = []
        
#         for face_data in face_crops:
#             face_img = face_data['image']
#             bbox = face_data['bbox']
#             quality = face_data.get('quality', 0.0)
#             x1, y1, x2, y2 = bbox
            
#             # Skip low quality
#             if quality < 0.3:
#                 continue
            
#             # Recognize
#             name, confidence = self.recognizer.recognize(face_img)
            
#             # Draw results
#             if name != "Unknown" and confidence > 0.42:
#                 color = (0, 255, 0)
#                 label = name
#                 conf_text = f"{confidence:.2f}"
                
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 draw_label_with_background(frame, label, (x1, y1-15),
#                                           color, font_scale=0.7, thickness=2)
#                 draw_label_with_background(frame, conf_text, (x1, y2+25),
#                                           color, font_scale=0.6, thickness=1)
                
#                 self.log_attendance(name, confidence)
#             else:
#                 color = (0, 165, 255)
#                 label = "Unknown"
                
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 draw_label_with_background(frame, label, (x1, y1-15),
#                                           color, font_scale=0.7, thickness=2)
            
#             results.append({
#                 'name': name,
#                 'confidence': confidence,
#                 'bbox': bbox
#             })
        
#         # Display stats
#         stats_bg = frame[0:100, 0:400].copy()
#         stats_bg[:] = (40, 40, 40)
#         cv2.addWeighted(stats_bg, 0.7, frame[0:100, 0:400], 0.3, 0, frame[0:100, 0:400])
        
#         cv2.putText(frame, f"Faces: {len(faces)}", (15, 35),
#                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
#         cv2.putText(frame, f"Type: {self.camera_type}", (15, 75),
#                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
#         return frame, results
    
#     def log_attendance(self, name, confidence, cooldown=10):
#         """Log attendance with cooldown"""
#         current_time = time.time()
        
#         for log in self.attendance_log:
#             if log['name'] == name:
#                 time_diff = current_time - log['timestamp']
#                 if time_diff < cooldown:
#                     return
        
#         log_entry = {
#             'name': name,
#             'confidence': confidence,
#             'timestamp': current_time,
#             'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'type': self.camera_type
#         }
        
#         self.attendance_log.append(log_entry)
#         self.logger.mark_present(name)
        
#         os.makedirs('logs', exist_ok=True)
#         with open(f'logs/attendance_{self.camera_type}.txt', 'a') as f:
#             f.write(f"{log_entry['datetime']} | {name} | {confidence:.2f} | {self.camera_type}\n")
        
#         print(f"[ATTENDANCE] {name} marked present at {log_entry['datetime']}")
    
#     def run_live(self, camera_id=0):
#         """Run single camera live"""
#         registered_persons = self.recognizer.get_all_registered_persons()
#         self.logger.start_session(registered_persons)
        
#         cap = cv2.VideoCapture(camera_id)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         cap.set(cv2.CAP_PROP_FPS, 30)
        
#         print(f"[SYSTEM] Starting live attendance on camera {camera_id}")
#         print("Press 'q' to quit")
        
#         fps_start_time = time.time()
#         fps_counter = 0
#         fps = 0
        
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("[ERROR] Failed to read frame")
#                     break
                
#                 annotated_frame, results = self.process_frame(frame)
                
#                 fps_counter += 1
#                 if time.time() - fps_start_time >= 1.0:
#                     fps = fps_counter
#                     fps_counter = 0
#                     fps_start_time = time.time()
                
#                 cv2.putText(annotated_frame, f"FPS: {fps}", (15, annotated_frame.shape[0]-20),
#                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
#                 cv2.imshow(f'Attendance System - {self.camera_type}', annotated_frame)
                
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
        
#         finally:
#             cap.release()
#             cv2.destroyAllWindows()
            
#             print("\n" + "=" * 70)
#             excel_path = self.logger.export_to_excel()
#             summary = self.logger.get_summary()
            
#             print("[ATTENDANCE SUMMARY]")
#             print(f"  Total Registered: {summary['total']}")
#             print(f"  Present: {summary['present']}")
#             print(f"  Absent: {summary['absent']}")
#             print(f"\n[EXCEL] Saved to: {excel_path}")
#             print("=" * 70)


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def browse_file(title="Select File", filetypes=None):
#     """Open file browser dialog"""
#     root = Tk()
#     root.withdraw()
#     root.attributes('-topmost', True)
    
#     if filetypes is None:
#         filetypes = [
#             ("Image files", "*.jpg *.jpeg *.png *.bmp"),
#             ("Video files", "*.mp4 *.avi *.mov *.mkv"),
#             ("All files", "*.*")
#         ]
    
#     file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
#     root.destroy()
    
#     return file_path


# def browse_folder(title="Select Folder"):
#     """Open folder browser dialog"""
#     root = Tk()
#     root.withdraw()
#     root.attributes('-topmost', True)
    
#     folder_path = filedialog.askdirectory(title=title)
#     root.destroy()
    
#     return folder_path


# def bulk_enroll_from_dataset(dataset_path='dataset'):
#     """
#     Bulk enrollment from dataset folder
#     Each person = one folder with images
#     """
#     print("=" * 70)
#     print("BULK ENROLLMENT WITH SYNTHETIC AUGMENTATION")
#     print("=" * 70)
    
#     if not os.path.exists(dataset_path):
#         print(f"[ERROR] Dataset folder '{dataset_path}' not found!")
#         return
    
#     recognizer = FaceRecognizer()
#     detector = FaceDetector()
    
#     person_folders = [f for f in os.listdir(dataset_path) 
#                      if os.path.isdir(os.path.join(dataset_path, f))]
    
#     if len(person_folders) == 0:
#         print(f"[ERROR] No person folders found in '{dataset_path}'!")
#         return
    
#     print(f"\n[INFO] Found {len(person_folders)} persons in dataset")
#     print(f"[MODE] Single photo per person → 7 synthetic poses")
#     print("-" * 70)
    
#     total_enrolled = 0
#     supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
#     for person_name in person_folders:
#         person_path = os.path.join(dataset_path, person_name)
        
#         # Get first image (golden image)
#         image_files = []
#         for ext in supported_formats:
#             pattern = os.path.join(person_path, f'*{ext}')
#             image_files.extend(glob.glob(pattern))
        
#         if len(image_files) == 0:
#             print(f"[SKIP] {person_name}: No images found")
#             continue
        
#         print(f"\n[PROCESSING] {person_name}")
        
#         # Use first image only
#         img_path = image_files[0]
        
#         try:
#             img = cv2.imread(img_path)
            
#             if img is None:
#                 print(f"  [ERROR] Could not read: {os.path.basename(img_path)}")
#                 continue
            
#             # Detect face
#             faces = detector.detect_faces(img)
            
#             if len(faces) == 0:
#                 print(f"  [SKIP] No face detected")
#                 continue
            
#             # Extract face crop
#             face_crops = detector.extract_face_crops(img, faces)
            
#             if len(face_crops) > 0:
#                 face_img = face_crops[0]['image']
                
#                 # Quality check
#                 h, w = face_img.shape[:2]
#                 if h < 60 or w < 60:
#                     print(f"  [SKIP] Face too small")
#                     continue
                
#                 # Add with augmentation
#                 if recognizer.add_person(person_name, face_img, use_augmentation=True):
#                     total_enrolled += 1
#                 else:
#                     print(f"  [ERROR] Failed to enroll")
                    
#         except Exception as e:
#             print(f"  [ERROR] {e}")
    
#     print("\n" + "=" * 70)
#     print(f"[COMPLETE] Total enrolled: {total_enrolled}/{len(person_folders)}")
#     print(f"[INFO] Database saved to: {recognizer.database_path}")
#     print("=" * 70)


# def test_image_recognition(test_image_path=None):
#     """Test recognition on single image"""
#     print("=" * 70)
#     print("IMAGE RECOGNITION TEST")
#     print("=" * 70)
    
#     if test_image_path is None:
#         print("\n[INFO] Opening file browser...")
#         test_image_path = browse_file(title="Select Test Image")
    
#     if not test_image_path or not os.path.exists(test_image_path):
#         print(f"[ERROR] No image selected")
#         return
    
#     print(f"\n[INFO] Loading: {os.path.basename(test_image_path)}")
    
#     detector = FaceDetector()
#     recognizer = FaceRecognizer()
#     logger = AttendanceLogger()
    
#     registered_persons = recognizer.get_all_registered_persons()
#     logger.start_session(registered_persons)
    
#     img = cv2.imread(test_image_path)
#     if img is None:
#         print(f"[ERROR] Could not read image!")
#         return
    
#     display_img = img.copy()
    
#     print(f"[INFO] Detecting faces...")
#     faces = detector.detect_faces(img)
    
#     if len(faces) == 0:
#         print(f"[RESULT] No faces detected!")
#         return
    
#     print(f"[INFO] Found {len(faces)} face(s)")
    
#     face_crops = detector.extract_face_crops(img, faces)
    
#     print(f"[INFO] Processing {len(face_crops)} faces for recognition...")
    
#     recognized = 0
#     unknown = 0
    
#     for face_data in face_crops:
#         face_img = face_data['image']
#         bbox = face_data['bbox']
#         x1, y1, x2, y2 = bbox
        
#         name, confidence = recognizer.recognize(face_img)
        
#         if name != "Unknown":
#             color = (0, 200, 0)
#             label = f"{name}"
#             conf_label = f"{confidence:.2f}"
#             recognized += 1
#             logger.mark_present(name)
            
#             cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 4)
#             draw_label_with_background(display_img, label, (x1, y1-15), 
#                                       color, font_scale=1.5, thickness=3)
#             draw_label_with_background(display_img, conf_label, (x1, y2+30), 
#                                       color, font_scale=1.0, thickness=2)
#         else:
#             color = (0, 165, 255)
#             label = "Unknown"
#             unknown += 1
            
#             cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 4)
#             draw_label_with_background(display_img, label, (x1, y1-15), 
#                                       color, font_scale=1.5, thickness=3)
    
#     excel_path = logger.export_to_excel()
#     summary = logger.get_summary()
    
#     print("\n" + "=" * 70)
#     print(f"[SUMMARY] Total: {len(faces)} | Recognized: {recognized} | Unknown: {unknown}")
#     print(f"[ATTENDANCE] Present: {summary['present']} | Absent: {summary['absent']}")
#     print(f"[EXCEL] Saved to: {excel_path}")
#     print("=" * 70)
    
#     # Show result
#     cv2.imshow('Recognition Result', display_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Save option
#     save = input("\nSave result? (y/n): ").strip().lower()
#     if save == 'y':
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         os.makedirs('logs', exist_ok=True)
#         save_path = f"logs/recognition_{timestamp}.jpg"
#         cv2.imwrite(save_path, display_img)
#         print(f"[SAVED] {save_path}")


# # ============================================================================
# # MAIN MENU
# # ============================================================================

# if __name__ == "__main__":
#     os.makedirs('logs', exist_ok=True)
#     os.makedirs('dataset', exist_ok=True)
#     os.makedirs('models', exist_ok=True)
#     os.makedirs('database', exist_ok=True)
    
#     while True:
#         print("\n" + "=" * 70)
#         print("JETSON NANO ATTENDANCE SYSTEM - MAIN MENU")
#         print("=" * 70)
#         print("1. Run DUAL Camera System (Entry + Exit)")
#         print("2. Run SINGLE Camera System")
#         print("3. Add New Person (Webcam Capture)")
#         print("4. Bulk Enrollment (Dataset Folder)")
#         print("5. Test Recognition on Image")
#         print("6. View Database")
#         print("7. Setup Models (Download)")
#         print("8. Exit")
#         print("=" * 70)
        
#         choice = input("Select option (1-8): ").strip()
        
#         if choice == '1':
#             # Dual camera system
#             entry_cam = int(input("Enter Entry camera ID (default 0): ").strip() or "0")
#             exit_cam = int(input("Enter Exit camera ID (default 1): ").strip() or "1")
            
#             system = DualCameraAttendanceSystem()
#             system.run_dual_camera(entry_cam_id=entry_cam, exit_cam_id=exit_cam)
            
#         elif choice == '2':
#             # Single camera system
#             camera_id = int(input("Enter camera ID (default 0): ").strip() or "0")
#             camera_type = input("Camera type (entry/exit, default entry): ").strip() or "entry"
            
#             system = SingleCameraAttendanceSystem(camera_type=camera_type)
#             system.run_live(camera_id=camera_id)
            
#         elif choice == '3':
#             # Add new person via webcam
#             recognizer = FaceRecognizer()
#             detector = FaceDetector()
            
#             name = input("Enter person's name: ").strip()
            
#             if not name:
#                 print("[ERROR] Name cannot be empty")
#                 continue
            
#             print("\n[INFO] Position face in center, press SPACE to capture")
#             print("[INFO] Press Q to cancel")
            
#             cap = cv2.VideoCapture(0)
            
#             captured = False
            
#             while not captured:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 faces = detector.detect_faces(frame)
                
#                 if len(faces) > 0:
#                     x1, y1, x2, y2, conf = faces[0]
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                     cv2.putText(frame, "Press SPACE to capture", (15, 40),
#                     cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
#                 else:
#                     cv2.putText(frame, "No face detected", (15, 40),
#                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                
#                 cv2.imshow('Capture Face - SPACE=capture, Q=quit', frame)
                
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord(' ') and len(faces) > 0:
#                     face_crops = detector.extract_face_crops(frame, faces)
#                     if len(face_crops) > 0:
#                         if recognizer.add_person(name, face_crops[0]['image'], use_augmentation=True):
#                             print(f"\n[SUCCESS] {name} enrolled successfully!")
#                             captured = True
#                         else:
#                             print(f"\n[ERROR] Failed to enroll {name}")
#                             break
#                 elif key == ord('q'):
#                     print("\n[CANCELLED]")
#                     break
            
#             cap.release()
#             cv2.destroyAllWindows()
            
#         elif choice == '4':
#             # Bulk enrollment
#             print("\n[INFO] Browse to select dataset folder...")
#             dataset_path = browse_folder(title="Select Dataset Folder")
            
#             if dataset_path:
#                 bulk_enroll_from_dataset(dataset_path)
#             else:
#                 print("[CANCELLED] No folder selected")
            
#         elif choice == '5':
#             # Test image recognition
#             test_image_recognition()
            
#         elif choice == '6':
#             # View database
#             recognizer = FaceRecognizer()
#             print("\n" + "=" * 70)
#             print("DATABASE SUMMARY")
#             print("=" * 70)
#             if len(recognizer.embeddings_db) == 0:
#                 print("[INFO] Database is empty")
#             else:
#                 for idx, (name, embeddings) in enumerate(recognizer.embeddings_db.items(), 1):
#                     print(f"  {idx}. {name}: {len(embeddings)} embeddings")
#                 print("-" * 70)
#                 print(f"  Total: {len(recognizer.embeddings_db)} persons")
#             print("=" * 70)
            
#         elif choice == '7':
#             # Setup models
#             print("\n" + "=" * 70)
#             print("MODEL SETUP")
#             print("=" * 70)
#             print("\n[INFO] Checking models...")
            
#             # Check SCRFD
#             if os.path.exists('models/scrfd_10g_bnkps.onnx'):
#                 print("✓ SCRFD model found")
#             else:
#                 print("✗ SCRFD model not found")
#                 print("\n[INFO] Downloading SCRFD model...")
#                 from face_detector import download_scrfd_model
#                 download_scrfd_model()
            
#             # Check AdaFace
#             if os.path.exists('models/adaface_ir101_webface12m.onnx'):
#                 print("✓ AdaFace model found")
#             else:
#                 print("✗ AdaFace model not found")
#                 print("\n[INFO] AdaFace setup:")
#                 from face_recognizer import download_adaface_model
#                 download_adaface_model()
            
#             # Check YOLOv11n fallback
#             if os.path.exists('models/yolo11n-face.pt'):
#                 print("✓ YOLOv11n fallback model found")
#             else:
#                 print("✗ YOLOv11n fallback not found")
#                 print("[INFO] Download from: https://github.com/ultralytics/ultralytics")
            
#             print("\n" + "=" * 70)
            
#         elif choice == '8':
#             print("\n[EXIT] System shutdown. Goodbye!")
#             break
            
#         else:
#             print("\n[ERROR] Invalid option")
# main2.py
# -----------------------------------------------------------------------------
# Robust Attendance & Recognition App (YOLOv11n-Face + AdaFace IR-101)
#
# Features:
#   - Live webcam recognition (real-time)
#   - Single image test
#   - Single video test with real-FPS playback + speed controls
#   - Bulk enrollment from dataset folder
#   - Attendance logging to Excel and text
#   - Clean overlays and fit-to-screen visualization (no GUI except Tk pickers)
#
# Keyboard (video/live windows):
#   Q / ESC : Quit
#   SPACE   : Pause/Resume (video)
#   F       : 2x faster (video)  [up to 4x]
#   R       : 0.5x slower (video) [down to 0.25x]
#   S       : Save screenshot
#
# Dependencies:
#   pip install ultralytics torch torchvision opencv-python pandas
#
# Folder/Files:
#   models/yolo11n-face.pt
#   models/adaface_ir101_webface12m.ckpt
#   database/embeddings_adaface.pkl (auto-created)
#   logs/* (auto-created)
# -----------------------------------------------------------------------------

import os
import cv2
import time
import glob
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict

from tkinter import Tk, filedialog

# Load torch-backed modules before pandas. On this Windows setup,
# importing pandas first can break the later torch DLL initialization.
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
import pandas as pd


# =============================================================================
# Small UI helpers (no heavy GUI)
# =============================================================================

def draw_label_with_bg(img, text, org, color=(60, 220, 60), font_scale=0.7, thickness=2):
    """Draws legible label with a solid (semi-opaque) background."""
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 10, 8
    x1, y1 = x - pad_x, y - th - pad_y
    x2, y2 = x + tw + pad_x, y + base + pad_y

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2); y2 = min(img.shape[0] - 1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def fit_to_screen(image, max_w=1600, max_h=900):
    """Preserve aspect ratio; never upscale; prevents cropped windows."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def browse_file(title, patterns):
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=patterns)
    root.destroy()
    return path


def browse_folder(title):
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


# =============================================================================
# Attendance Logger
# =============================================================================

class AttendanceLogger:
    def __init__(self, save_dir="logs"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.present = set()
        self.rows = []
        self.txt_paths = {}  # stream txt per run (keyed by label)

    def _append_txt(self, label: str, line: str):
        os.makedirs(self.save_dir, exist_ok=True)
        if label not in self.txt_paths:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = os.path.join(self.save_dir, f"{label}_{ts}.txt")
            self.txt_paths[label] = p
        with open(self.txt_paths[label], "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def start_session(self, registered_people: List[str]):
        self.present.clear()
        self.rows.clear()
        # optional: record roster
        self._append_txt("session", f"SESSION START {datetime.now().isoformat()} | Registered: {len(registered_people)}")

    def mark_present(self, name: str, confidence: float, source: str = "live"):
        if name in self.present:
            return
        self.present.add(name)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.rows.append({"Name": name, "Confidence": round(confidence, 3), "Timestamp": ts, "Source": source})
        self._append_txt("attendance", f"{ts} | {name} | {confidence:.3f} | {source}")
        print(f"[ATTENDANCE] {name} ({confidence:.2f})")

    def export_excel(self) -> str:
        if not self.rows:
            print("[Logger] Nothing to export.")
            return ""
        df = pd.DataFrame(self.rows)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, f"attendance_{ts}.xlsx")
        df.to_excel(path, index=False)
        print(f"[Logger] Saved Excel → {path}")
        return path

    def summary(self) -> Dict[str, int]:
        unique = len(self.present)
        return {"present_unique": unique, "rows": len(self.rows)}


# =============================================================================
# Core App
# =============================================================================

class AttendanceSystem:
    def __init__(self):
        print("=" * 90)
        print("ATTENDANCE & RECOGNITION SYSTEM  (YOLOv11n-Face + AdaFace IR-101)")
        print("=" * 90)

        print("[1/3] Loading detector...")
        self.detector = FaceDetector(
            model_path="models/yolo11n-face.pt",
            conf_threshold=0.5,
            iou_threshold=0.45,
            max_det=400,
            warmup=True,
            verbose=True,
        )

        print("[2/3] Loading recognizer (AdaFace)...")
        self.recognizer = FaceRecognizer(
            database_path="database/embeddings_adaface.pkl",
            model_ckpt="models/adaface_ir101_webface12m.ckpt",
            verbose=True
        )

        print("[3/3] Loading logger...")
        self.logger = AttendanceLogger(save_dir="logs")
        print("=" * 90)

    # -------------------------------------------------------------------------
    # Enrollment
    # -------------------------------------------------------------------------
    def bulk_enroll_from_dataset(self, dataset_dir: str = None, use_augmentation: bool = True):
        if dataset_dir is None:
            dataset_dir = browse_folder("Select dataset folder (each subfolder = person)")

        if not dataset_dir or not os.path.isdir(dataset_dir):
            print("[Enroll] Invalid dataset folder.")
            return

        people = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if not people:
            print("[Enroll] No person folders found.")
            return

        print(f"[Enroll] Found {len(people)} person folder(s). Starting...")
        added_total = 0

        for person in sorted(people):
            pdir = os.path.join(dataset_dir, person)
            # gather images
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG", "*.JPEG", "*.BMP"):
                imgs.extend(glob.glob(os.path.join(pdir, ext)))

            if not imgs:
                print(f"[Enroll] {person}: no images → skip")
                continue

            print(f"[Enroll] {person}: {len(imgs)} image(s)")
            added_this_person = 0

            # We enroll from crops extracted by YOLO to ensure consistent boxes
            for img_path in imgs:
                bgr = cv2.imread(img_path)
                if bgr is None:
                    print(f"  [!] Could not read: {img_path}")
                    continue

                faces = self.detector.detect_faces(bgr)
                if not faces:
                    print(f"  [!] No face in: {os.path.basename(img_path)}")
                    continue

                crops = self.detector.extract_face_crops(bgr, faces, padding=0.20, min_size=56, sort_by_quality=True)
                if not crops:
                    print(f"  [!] No valid crops: {os.path.basename(img_path)}")
                    continue

                # use best quality crop
                face = crops[0]["image"]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                ok = self.recognizer.add_person(
                    name=person,
                    face_bgr=face,                # recognizer auto-converts to RGB; passing BGR is fine
                    augment=use_augmentation,
                    preprocess_for_video=False    # keep enrollment clean; set True to match video domain
                )
                if ok:
                    added_this_person += 1

            print(f"[Enroll] {person}: added from {added_this_person} image(s)")
            if added_this_person > 0:
                added_total += 1

        print("=" * 60)
        print(f"[Enroll] Completed. People added: {added_total}/{len(people)}")
        print("[Enroll] Database:", self.recognizer.database_path)
        print("=" * 60)

    # -------------------------------------------------------------------------
    # Single image test
    # -------------------------------------------------------------------------
    def test_single_image(self, image_path=None):
        if image_path is None:
            image_path = browse_file("Select image", [("Images", "*.jpg;*.jpeg;*.png;*.bmp")])

        if not image_path or not os.path.isfile(image_path):
            print("[Image] Invalid path.")
            return

        bgr = cv2.imread(image_path)
        if bgr is None:
            print("[Image] Could not read image.")
            return

        vis = bgr.copy()
        faces = self.detector.detect_faces(bgr)
        if not faces:
            print("[Image] No faces found.")
            return

        crops = self.detector.extract_face_crops(bgr, faces, padding=0.20, min_size=56)
        recognized = 0

        for c in crops:
            x1, y1, x2, y2 = c["bbox"]
            face = c["image"]
            # IMAGE MODE (no video denoise)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            name, conf = self.recognizer.recognize(face_rgb, mode="image")

            color = (0, 205, 0) if name != "Unknown" else (0, 165, 255)
            if name != "Unknown":
                recognized += 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            draw_label_with_bg(vis, f"{name} ({conf:.2f})", (x1 + 6, y1 - 10), color=color)

        vis = fit_to_screen(vis)
        cv2.imshow("Image Recognition", vis)
        print(f"[Image] Faces: {len(crops)} | Recognized: {recognized}")
        print("[Image] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # Single video test (real-FPS + speed controls)
    # -------------------------------------------------------------------------
    def test_video(self, video_path=None):
        if video_path is None:
            video_path = browse_file("Select video", [("Video", "*.mp4;*.avi;*.mkv;*.mov")])

        if not video_path or not os.path.isfile(video_path):
            print("[Video] Invalid path.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[Video] Cannot open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # some containers don't report FPS
            fps = 30.0
        delay_ms = int(1000.0 / fps)

        play_speed = 1.0
        paused = False
        frame_idx = 0

        self.logger.start_session(self.recognizer.get_all_registered())
        print(f"[Video] {os.path.basename(video_path)} | FPS: {fps:.2f}")
        print("Controls: SPACE=Pause/Resume  F=Faster  R=Slower  S=Screenshot  Q=Quit")

        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("[Video] End of file.")
                    break
                frame_idx += 1

                faces = self.detector.detect_faces(frame)
                crops = self.detector.extract_face_crops(frame, faces, padding=0.20, min_size=56)

                for c in crops:
                    x1, y1, x2, y2 = c["bbox"]
                    face = c["image"].copy()

                    # VIDEO MODE FIXES: RGB + denoise + light contrast
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.bilateralFilter(face, 3, 75, 75)
                    face = cv2.convertScaleAbs(face, alpha=1.10, beta=3)

                    name, conf = self.recognizer.recognize(face, mode="video")
                    color = (0, 205, 0) if name != "Unknown" else (0, 165, 255)
                    if name != "Unknown":
                        self.logger.mark_present(name, conf, source="video")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label_with_bg(frame, f"{name} ({conf:.2f})", (x1 + 6, y1 - 10), color=color)

                # HUD
                hud = frame.copy()
                cv2.rectangle(hud, (0, 0), (420, 88), (40, 40, 40), -1)
                cv2.addWeighted(hud, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, f"FPS: {fps:.2f}  x{play_speed:.2f}", (14, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Faces: {len(crops)}  Frame: {frame_idx}", (14, 64),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                vis = fit_to_screen(frame)
                cv2.imshow("Video Recognition (SPACE/F/R/S/Q)", vis)

            # === Adaptive frame timing (real FPS simulation) ===
            frame_start = time.time()

            key = cv2.waitKey(1) & 0xFF  # minimal blocking for UI
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('f'):
                play_speed = min(4.0, play_speed * 2.0)
            elif key == ord('r'):
                play_speed = max(0.25, play_speed * 0.5)
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join("logs", f"video_snap_{ts}.jpg")
                os.makedirs("logs", exist_ok=True)
                if 'frame' in locals():
                    cv2.imwrite(path, frame)
                    print(f"[Video] Screenshot saved → {path}")

            # Calculate how long this frame took
            frame_elapsed = time.time() - frame_start
            target_delay = max(0, (1.0 / fps) / play_speed - frame_elapsed)

            # Sleep only if processing was faster than target frame duration
            if target_delay > 0:
                time.sleep(target_delay)

            # Optional: skip frames if lag accumulates
            # Example: skip every other frame when falling behind
            if frame_elapsed > (1.0 / fps) * 1.5:
                for _ in range(int(frame_elapsed / ((1.0 / fps) / play_speed))):
                    cap.grab()


        cap.release()
        cv2.destroyAllWindows()

        xlsx = self.logger.export_excel()
        print("[Video] Summary:", self.logger.summary())
        if xlsx:
            print("[Video] Attendance Excel:", xlsx)

    # -------------------------------------------------------------------------
    # Live webcam
    # -------------------------------------------------------------------------
    def run_live(self, cam_id: int = 0):
        #"rtsp://admin:123456@10.11.73.44:554/stream1"
        #""rtsp://admin:admin123@192.168.1.1:554/cam/realmonitor?channel=1&subtype=0""
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("[Live] Cannot open camera.")
            return

        # Try to configure a reasonable resolution/FPS (best effort)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.logger.start_session(self.recognizer.get_all_registered())
        print("[Live] Running. Press Q or ESC to stop.")

        t0 = time.time()
        frame_count = 0
        fps_display = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Live] Camera read failed.")
                break

            frame_count += 1
            if frame_count % 10 == 0:
                dt = time.time() - t0
                fps_display = 10.0 / dt if dt > 0 else 0.0
                t0 = time.time()

            faces = self.detector.detect_faces(frame)
            crops = self.detector.extract_face_crops(frame, faces, padding=0.20, min_size=56)

            for c in crops:
                x1, y1, x2, y2 = c["bbox"]
                face = c["image"].copy()

                # LIVE MODE FIXES: RGB + denoise + light contrast
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.bilateralFilter(face, 3, 75, 75)
                face = cv2.convertScaleAbs(face, alpha=1.10, beta=3)

                name, conf = self.recognizer.recognize(face, mode="live")

                color = (0, 205, 0) if name != "Unknown" else (0, 165, 255)
                if name != "Unknown":
                    self.logger.mark_present(name, conf, source="live")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label_with_bg(frame, f"{name} ({conf:.2f})", (x1 + 6, y1 - 10), color=color)

            # HUD
            hud = frame.copy()
            cv2.rectangle(hud, (0, 0), (360, 70), (40, 40, 40), -1)
            cv2.addWeighted(hud, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"FPS: {fps_display:0.1f}  Faces: {len(crops)}", (14, 44),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            vis = fit_to_screen(frame)
            cv2.imshow("Live Recognition (Q/Esc to quit)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

        cap.release()
        cv2.destroyAllWindows()

        xlsx = self.logger.export_excel()
        print("[Live] Summary:", self.logger.summary())
        if xlsx:
            print("[Live] Attendance Excel:", xlsx)

    # -------------------------------------------------------------------------
    # Menu / CLI
    # -------------------------------------------------------------------------
    def print_menu(self):
        print("\n" + "=" * 90)
        print("MAIN MENU")
        print("=" * 90)
        print("1) Run LIVE webcam recognition")
        print("2) Test SINGLE IMAGE")
        print("3) Test SINGLE VIDEO (real-FPS, speed controls)")
        print("4) BULK ENROLL from dataset folder")
        print("5) Show registered identities")
        print("6) Remove an identity")
        print("7) Clear database")
        print("8) Exit")
        print("=" * 90)

    def run(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("database", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        while True:
            self.print_menu()
            choice = input("Select option (1-8): ").strip()

            if choice == "1":
                cam = input("Camera index (default 0): ").strip()
                cam_id = int(cam) if cam else 0
                self.run_live(cam_id)

            elif choice == "2":
                self.test_single_image()

            elif choice == "3":
                self.test_video()

            elif choice == "4":
                path = browse_folder("Select dataset folder (subfolders = person names)")
                if path:
                    aug = input("Use augmentation during enrollment? (y/n, default y): ").strip().lower()
                    use_aug = (aug != "n")
                    self.bulk_enroll_from_dataset(path, use_augmentation=use_aug)
                else:
                    print("[Enroll] Cancelled.")

            elif choice == "5":
                people = self.recognizer.get_all_registered()
                if not people:
                    print("[DB] No identities in database.")
                else:
                    print("[DB] Registered identities:")
                    for i, p in enumerate(people, 1):
                        print(f"  {i:02d}. {p}")

            elif choice == "6":
                name = input("Enter exact identity name to remove: ").strip()
                if name:
                    ok = self.recognizer.remove_person(name)
                    print("[DB] Removed." if ok else "[DB] Not found.")

            elif choice == "7":
                sure = input("Type 'YES' to clear database: ").strip()
                if sure == "YES":
                    self.recognizer.clear_database()
                else:
                    print("[DB] Cancelled.")

            elif choice == "8":
                print("Goodbye!")
                break

            else:
                print("Invalid option. Try again.")


# =============================================================================
# Entry
# =============================================================================

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()
