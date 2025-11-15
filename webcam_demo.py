import os
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['__VK_LAYER_NV_optimus'] = 'NVIDIA_only'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# EGL Config
os.environ['EGL_PLATFORM'] = 'surfaceless'  # or device
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import yaml

class HandGestureMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=[512, 256, 128], dropout_rate=0.3):
        super(HandGestureMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class HandGesturePredictor:
    """Hand Gesture Predictor"""
    
    def __init__(self, model_left_path, model_right_path, input_size, num_classes, gesture_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gesture_names = gesture_names
        self.lmodel = self.load_model(model_left_path, input_size, num_classes)
        self.rmodel = self.load_model(model_right_path, input_size, num_classes)
        
    def load_model(self, model_path, input_size, num_classes):
        """Load trained model"""
        model = HandGestureMLP(input_size, num_classes)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def predict(self, raw_data, hand_type):
        """Predict hand gesture"""
        pos_data = []
        for data in raw_data:
            if len(pos_data) == 0:
                pos_data.extend([data.x, data.y, data.z])
            else:
                pos_data.extend([
                    data.x - pos_data[0], 
                    data.y - pos_data[1], 
                    data.z - pos_data[2]
                ])
        pos_data[0], pos_data[1], pos_data[2] = 0, 0, 0

        with torch.no_grad():
            # Move to device
            pos_data = torch.FloatTensor([pos_data]).to(self.device)
            
            # Predict
            if hand_type == "Right" and self.rmodel is not None:
                outputs = self.rmodel(pos_data)
            elif hand_type == "Left" and self.lmodel is not None:
                outputs = self.lmodel(pos_data)
            else:
                return -1, [0] * len(self.gesture_names)
                
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert to numpy
            predictions_np = predictions.cpu().item()
            probabilities_np = probabilities.cpu().numpy()[0]
            
        return predictions_np, probabilities_np

class HandTracker:
    def __init__(self, model_path):
        try:
            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.GPU
            )
            
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.detector = vision.HandLandmarker.create_from_options(options)
            self.is_initialized = True
        except Exception as e:
            print(f"Error initializing hand tracker: {e}")
            self.is_initialized = False
    
    def draw_landmarks(self, frame, detection_result, predicted_result, fps, gesture_names):
        """Draw keypoints and connection lines"""
        if not self.is_initialized:
            cv2.putText(frame, "Tracker not initialized", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        if predicted_result is not None:
            label, conf = predicted_result
        else:
            label, conf = -1, [0] * len(gesture_names)
        
        # Draw FPS information
        cv2.putText(frame, f"FPS: {int(fps)} (RTX 4070 GPU)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not hand_landmarks_list:
            cv2.putText(frame, "No hands detected", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Different colors for each hand
        colors = [(0, 255, 0), (255, 0, 0)]  # Green and Blue
        text_color = (0, 0, 255)
        
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            color = colors[idx % len(colors)]
            handedness = handedness_list[idx] if idx < len(handedness_list) else None
            
            # Draw keypoints
            h, w, _ = frame.shape
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, color, -1)
            
            # Draw connection lines
            self.draw_connections(frame, hand_landmarks, w, h, color)
            
            # Display hand information (Left/Right)
            if handedness:
                hand_type = handedness[0].category_name
                confidence = handedness[0].score
                
                # Find center position for label display
                x_coords = [lm.x * w for lm in hand_landmarks]
                y_coords = [lm.y * h for lm in hand_landmarks]
                center_x = int(sum(x_coords) / len(x_coords))
                center_y = int(sum(y_coords) / len(y_coords))
                
                cv2.putText(frame, f"{hand_type} ({confidence:.2f})", 
                           (center_x - 50, center_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                gesture_text = gesture_names[label] if 0 <= label < len(gesture_names) else "Unknown"
                confidence_value = conf[label] if 0 <= label < len(conf) else 0
                cv2.putText(frame, f"{gesture_text} ({confidence_value:.2f})", 
                           (center_x - 50, center_y - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
        
        cv2.putText(frame, f"Hands detected: {len(hand_landmarks_list)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def draw_connections(self, frame, landmarks, width, height, color):
        """Draw hand keypoint connections"""
        # Hand connection definitions (simplified)
        connections = [
            # Palm
            [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
            [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
            [0, 17], [17, 18], [18, 19], [19, 20],  # Little finger
            # Palm connections
            [5, 9], [9, 13], [13, 17]
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx].x * width), 
                              int(landmarks[start_idx].y * height))
                end_point = (int(landmarks[end_idx].x * width), 
                            int(landmarks[end_idx].y * height))
                cv2.line(frame, start_point, end_point, color, 2)
    
    def __del__(self):
        """Release resources"""
        if hasattr(self, 'detector') and self.is_initialized:
            self.detector.close()

class ConfigManager:
    """Configuration manager for gesture recognition"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        default_config = {
            'gestures': ['Rock', 'Scissors', 'Paper'],
            'models': {
                'left': 'trained_models/gesture_model_left.pth',
                'right': 'trained_models/gesture_model_right.pth'
            },
            'hand_landmark_model': 'hand_landmarker.task',
            'input_size': 63
        }
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
                # Merge with default config to ensure all required keys exist
                if config:
                    # Handle gestures from different possible config structures
                    if 'gestures' in config:
                        gestures = list(config['gestures'].keys()) if isinstance(config['gestures'], dict) else config['gestures']
                    elif 'datasets' in config:
                        gestures = list(config['datasets'].keys())
                    else:
                        gestures = default_config['gestures']
                    
                    merged_config = {
                        'gestures': gestures,
                        'models': config.get('models', default_config['models']),
                        'hand_landmark_model': config.get('hand_landmark_model', default_config['hand_landmark_model']),
                        'input_size': default_config['input_size']
                    }
                    return merged_config
                else:
                    return default_config
                    
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default configuration.")
            return default_config
        except Exception as e:
            print(f"Error loading config file: {e}. Using default configuration.")
            return default_config
    
    def get_gesture_names(self):
        """Get list of gesture names"""
        return self.config['gestures']
    
    def get_left_model_path(self):
        """Get left hand model path"""
        return self.config['models'].get('left', 'trained_models/gesture_model_left.pth')
    
    def get_right_model_path(self):
        """Get right hand model path"""
        return self.config['models'].get('right', 'trained_models/gesture_model_right.pth')
    
    def get_landmark_model_path(self):
        """Get hand landmark model path"""
        return self.config.get('hand_landmark_model', 'hand_landmarker.task')
    
    def get_input_size(self):
        """Get input size for the model"""
        return self.config.get('input_size', 63)
    
    def get_num_classes(self):
        """Get number of gesture classes"""
        return len(self.config['gestures'])

class HandGestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.is_running = False
        self.cap = None
        self.predictor = None
        self.tracker = None
        self.config_manager = None
        
        # Create GUI
        self.create_widgets()
        
        # Video frame queue
        self.video_queue = queue.Queue(maxsize=1)
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky='wens')
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky='we', pady=(0, 10))
        
        # Config file selection
        ttk.Label(control_frame, text="Configuration File:").grid(row=0, column=0, sticky=tk.W)
        self.config_var = tk.StringVar(value="config.yaml")
        ttk.Entry(control_frame, textvariable=self.config_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_config).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Reload Config", command=self.reload_config).grid(row=0, column=3, padx=5)
        
        # Gesture information display
        gesture_frame = ttk.LabelFrame(control_frame, text="Available Gestures", padding="5")
        gesture_frame.grid(row=1, column=0, columnspan=4, sticky='we', pady=5)
        
        self.gesture_label = ttk.Label(gesture_frame, text="No gestures loaded")
        self.gesture_label.grid(row=0, column=0, sticky=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Initialize System", command=self.initialize_system).pack(side=tk.LEFT, padx=5)
        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self.start_detection, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=1, column=0, sticky='wens', padx=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, background="black")
        self.video_label.grid(row=0, column=0, sticky='wens')
        
        # Status panel
        status_frame = ttk.LabelFrame(main_frame, text="Status Information", padding="10")
        status_frame.grid(row=1, column=1, sticky='wens')
        status_frame.columnconfigure(0, weight=1)
        
        self.status_text = tk.Text(status_frame, height=20, width=40, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        self.status_text.grid(row=0, column=0, sticky='wens')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Add initial status
        self.log_status("System ready. Please load configuration and initialize the system.")
    
    def browse_config(self):
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            self.config_var.set(filename)
            self.reload_config()
            
    def reload_config(self):
        try:
            self.config_manager = ConfigManager(self.config_var.get())
            gesture_names = self.config_manager.get_gesture_names()
            gesture_text = "Available gestures: " + ", ".join(gesture_names)
            self.gesture_label.config(text=gesture_text)
            self.log_status(f"Configuration loaded: {len(gesture_names)} gestures")
        except Exception as e:
            self.log_status(f"Error loading configuration: {str(e)}")
    
    def initialize_system(self):
        try:
            if self.config_manager is None:
                raise TimeoutError("config manager is none")
                
            self.log_status("Initializing system...")
            
            # Get configuration values
            gesture_names = self.config_manager.get_gesture_names()
            left_model_path = self.config_manager.get_left_model_path()
            right_model_path = self.config_manager.get_right_model_path()
            input_size = self.config_manager.get_input_size()
            num_classes = self.config_manager.get_num_classes()
            landmark_model_path = self.config_manager.get_landmark_model_path()
            
            # Initialize predictor
            self.predictor = HandGesturePredictor(
                left_model_path,
                right_model_path,
                input_size,
                num_classes,
                gesture_names
            )
            
            # Initialize tracker
            self.tracker = HandTracker(landmark_model_path)
            
            if not self.tracker.is_initialized:
                self.log_status("ERROR: Failed to initialize hand tracker")
                return
                
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log_status("ERROR: Cannot open camera")
                return
                
            self.log_status("System initialized successfully!")
            self.log_status(f"Left model: {left_model_path}")
            self.log_status(f"Right model: {right_model_path}")
            self.log_status(f"Gestures: {', '.join(gesture_names)}")
            self.start_button.config(state=tk.NORMAL)
            
        except RuntimeError as e:
            self.reload_config()
    
        except Exception as e:
            self.log_status(f"ERROR during initialization: {str(e)}")
        
    def start_detection(self):
        if not self.cap or not self.tracker or not self.tracker.is_initialized:
            self.log_status("ERROR: System not properly initialized")
            return
            
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start video processing in separate thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
        
        # Start video display update
        self.update_video_display()
        
        self.log_status("Hand detection started")
    
    def stop_detection(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_status("Hand detection stopped")
    
    def process_video(self):
        prev_time = time.time()
        gesture_names = self.config_manager.get_gesture_names() if self.config_manager else ["Unknown"]
        
        while self.is_running:
            if self.cap is None:
                raise Exception("no capture device")

            if self.tracker is None:
                raise Exception("no tracker")
                

            ret, frame = self.cap.read()
            if not ret:
                self.log_status("ERROR: Cannot read frame from camera")
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
            det_res = self.tracker.detector.detect_for_video(img, timestamp_ms)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
            prev_time = current_time

            predicted_results = []
            if det_res.hand_landmarks and self.predictor:
                for idx, hand_landmark in enumerate(det_res.hand_landmarks):
                    handedness = det_res.handedness[idx] if idx < len(det_res.handedness) else None
                    
                    if handedness:
                        hand_type = handedness[0].category_name
                        confidence = handedness[0].score
                        
                        pre_res = self.predictor.predict(hand_landmark, hand_type)
                        predicted_results.append((hand_type, pre_res))
                        
                        gesture_id = int(pre_res[0])
                        gesture_name = gesture_names[gesture_id] if 0 <= gesture_id < len(gesture_names) else "Unknown"
                        prob = pre_res[1][gesture_id] if 0 <= gesture_id < len(pre_res[1]) else 0
                        
                        status_msg = f"{hand_type}: {gesture_name} (Confidence: {prob:.2f})"
                        self.log_status(status_msg)

            # Draw landmarks and predictions
            display_frame = self.tracker.draw_landmarks(
                frame, det_res, 
                predicted_results[0][1] if predicted_results else None, 
                fps, gesture_names
            )
            
            # Convert frame for display
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (640, 480))
            
            # Update video queue
            try:
                self.video_queue.put_nowait(display_frame)
            except queue.Full:
                pass  # Skip frame if queue is full
    
    def update_video_display(self):
        try:
            frame = self.video_queue.get_nowait()
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # type: ignore
        except queue.Empty:
            pass
        
        if self.is_running:
            self.root.after(10, self.update_video_display)
    
    def log_status(self, message):
        self.status_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.tracker:
            del self.tracker
        self.root.destroy()

def main():
    root = tk.Tk()
    app = HandGestureGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()