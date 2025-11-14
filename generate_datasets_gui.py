import os
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['__VK_LAYER_NV_optimus'] = 'NVIDIA_only'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# EGL Config
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import csv
import time
from datetime import datetime
import numpy as np
from PIL import Image, ImageTk

class HandLandmarkRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Landmark Recorder")
        self.root.geometry("800x600")
        
        # State variables
        self.is_recording = False
        self.is_paused = False
        self.save_path = ""
        self.data_format = "npy"  # Default to numpy format
        
        # Initialize MediaPipe
        self.setup_mediapipe()
        
        # Create GUI
        self.setup_gui()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.current_photo = None  # Store reference to prevent garbage collection
        
        # Data storage
        self.recorded_data = []
        self.frame_count = 0
        
        # Start video stream
        self.update_video()
    
    def setup_mediapipe(self):
        """Initialize MediaPipe hand detector"""
        base_options = python.BaseOptions(
            model_asset_path='hand_landmarker.task',
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
    
    def setup_gui(self):
        """Setup GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Control area
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Save path selection
        path_frame = ttk.Frame(control_frame)
        path_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(path_frame, text="Save Path:").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar(value="No path selected")
        ttk.Label(path_frame, textvariable=self.path_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(path_frame, text="Select Path", command=self.select_save_path).grid(row=0, column=2)
        
        # File format selection
        format_frame = ttk.Frame(control_frame)
        format_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(format_frame, text="Save Format:").grid(row=0, column=0, sticky="w")
        self.format_var = tk.StringVar(value="npy")
        ttk.Radiobutton(format_frame, text="NumPy (.npy)", variable=self.format_var, value="npy").grid(row=0, column=1)
        ttk.Radiobutton(format_frame, text="JSON", variable=self.format_var, value="json").grid(row=0, column=2)
        ttk.Radiobutton(format_frame, text="CSV", variable=self.format_var, value="csv").grid(row=0, column=3)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Recording", command=self.start_recording)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Pause Recording", command=self.pause_recording, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=5)
        
        # Status display
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        
        self.status_var = tk.StringVar(value="Status: Waiting to start recording")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        
        self.frame_var = tk.StringVar(value="Frames: 0")
        ttk.Label(status_frame, textvariable=self.frame_var).grid(row=0, column=1, sticky="w", padx=20)
        
        # Video display area
        video_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding="10")
        video_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Data preview area
        data_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="10")
        data_frame.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        
        # Create text box and scrollbar
        self.text_frame = tk.Frame(data_frame)
        self.text_frame.grid(row=0, column=0, sticky="nsew")
        
        self.data_text = tk.Text(self.text_frame, width=50, height=20, wrap=tk.NONE)
        scrollbar_y = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.data_text.yview)
        scrollbar_x = ttk.Scrollbar(self.text_frame, orient=tk.HORIZONTAL, command=self.data_text.xview)
        
        self.data_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.data_text.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        # Configure weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        control_frame.columnconfigure(1, weight=1)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(0, weight=1)
    
    def select_save_path(self):
        """Select save path"""
        path = filedialog.asksaveasfilename(
            defaultextension="." + self.format_var.get(),
            filetypes=[
                ("NumPy files", "*.npy"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.save_path = path
            self.path_var.set(os.path.basename(path))
    
    def start_recording(self):
        """Start recording"""
        if not self.save_path:
            messagebox.showerror("Error", "Please select a save path first!")
            return
        
        self.is_recording = True
        self.is_paused = False
        self.recorded_data = []
        self.frame_count = 0
        
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Status: Recording...")
    
    def pause_recording(self):
        """Pause/resume recording"""
        if self.is_paused:
            self.is_paused = False
            self.pause_button.config(text="Pause Recording")
            self.status_var.set("Status: Recording...")
        else:
            self.is_paused = True
            self.pause_button.config(text="Resume Recording")
            self.status_var.set("Status: Paused")
    
    def stop_recording(self):
        """Stop recording and save data"""
        self.is_recording = False
        self.is_paused = False
        
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(text="Pause Recording")
        self.status_var.set("Status: Waiting to start recording")
        
        # Save data
        if self.recorded_data:
            self.save_data()
    
    def save_data(self):
        """Save recorded data in the selected format"""
        try:
            if self.format_var.get() == "npy":
                # Convert to numpy array format
                numpy_data = self.convert_to_numpy_format()
                np.save(self.save_path, numpy_data)
            elif self.format_var.get() == "json":
                with open(self.save_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': {
                            'total_frames': len(self.recorded_data),
                            'timestamp': datetime.now().isoformat(),
                            'data_format': 'hand_landmarks_relative_to_wrist'
                        },
                        'data': self.recorded_data
                    }, f, indent=2, ensure_ascii=False)
            else:  # CSV
                with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Write header
                    headers = ['frame', 'timestamp', 'hand_id', 'handedness']
                    for i in range(21):  # 21 hand landmarks
                        headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
                    writer.writerow(headers)
                    
                    # Write data
                    for frame_data in self.recorded_data:
                        for hand_data in frame_data['hands']:
                            row = [
                                frame_data['frame'],
                                frame_data['timestamp'],
                                hand_data['hand_id'],
                                hand_data['handedness']
                            ]
                            for landmark in hand_data['landmarks']:
                                row.extend([landmark['x'], landmark['y'], landmark['z']])
                            writer.writerow(row)
            
            messagebox.showinfo("Success", f"Data saved to: {self.save_path}\nTotal frames: {len(self.recorded_data)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {str(e)}")
    
    def convert_to_numpy_format(self):
        """Convert recorded data to numpy format"""
        # We'll create a structured numpy array
        # Each frame can have up to 2 hands, each with 21 landmarks (x, y, z)
        max_hands = 2
        num_landmarks = 21
        
        # Create a structured dtype for our data
        dtype = [
            ('frame', 'i4'),
            ('timestamp', 'f8'),
            ('num_hands', 'i4'),
            ('hand_data', [
                ('handedness', 'U10'),  # String for left/right
                ('landmarks', 'f4', (num_landmarks, 3))  # 21 landmarks, each with x,y,z
            ], max_hands)
        ]
        
        # Initialize numpy array
        numpy_data = np.zeros(len(self.recorded_data), dtype=dtype)
        
        # Fill the array
        for i, frame_data in enumerate(self.recorded_data):
            numpy_data[i]['frame'] = frame_data['frame']
            numpy_data[i]['timestamp'] = frame_data['timestamp']
            numpy_data[i]['num_hands'] = len(frame_data['hands'])
            
            for j, hand_data in enumerate(frame_data['hands']):
                if j < max_hands:
                    numpy_data[i]['hand_data'][j]['handedness'] = hand_data['handedness']
                    
                    # Fill landmarks
                    for k, landmark in enumerate(hand_data['landmarks']):
                        if k < num_landmarks:
                            numpy_data[i]['hand_data'][j]['landmarks'][k] = [
                                landmark['x'], landmark['y'], landmark['z']
                            ]
        
        return numpy_data
    
    def update_video(self):
        """Update video frame"""
        ret, frame = self.cap.read()
        if ret:
            # Flip horizontally for mirror view
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            # Draw hand landmarks
            self.draw_landmarks(frame, detection_result)
            
            # Record data
            if self.is_recording and not self.is_paused:
                self.record_frame_data(detection_result)
            
            # Update video display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (400, 300))
            photo = self.cv2_to_tkinter(img)
            self.video_label.configure(image=photo)
            # Store reference to prevent garbage collection
            self.current_photo = photo
        
        self.root.after(10, self.update_video)
    
    def draw_landmarks(self, image, detection_result):
        """Draw hand landmarks on image"""
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw custom landmarks
                self.draw_custom_landmarks(image, hand_landmarks)
    
    def draw_custom_landmarks(self, image, landmarks):
        """Custom drawing of hand landmarks and connections"""
        h, w, _ = image.shape
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections - hand landmark connections
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
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                start_point = (int(landmarks[connection[0]].x * w), 
                             int(landmarks[connection[0]].y * h))
                end_point = (int(landmarks[connection[1]].x * w), 
                           int(landmarks[connection[1]].y * h))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    def record_frame_data(self, detection_result):
        """Record current frame data with coordinates relative to wrist"""
        frame_data = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'hands': []
        }
        
        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Get wrist coordinates (landmark 0)
                wrist_x = hand_landmarks[0].x
                wrist_y = hand_landmarks[0].y
                wrist_z = hand_landmarks[0].z
                
                hand_data = {
                    'hand_id': i,
                    'handedness': detection_result.handedness[i][0].category_name if i < len(detection_result.handedness) else 'Unknown',
                    'wrist_position': {'x': wrist_x, 'y': wrist_y, 'z': wrist_z},
                    'landmarks': []
                }
                
                # Calculate all landmarks relative to wrist
                for j, landmark in enumerate(hand_landmarks):
                    # Subtract wrist coordinates to get relative position
                    relative_x = landmark.x - wrist_x
                    relative_y = landmark.y - wrist_y
                    relative_z = landmark.z - wrist_z
                    
                    hand_data['landmarks'].append({
                        'x': relative_x,
                        'y': relative_y,
                        'z': relative_z
                    })
                
                frame_data['hands'].append(hand_data)
        
        self.recorded_data.append(frame_data)
        self.frame_count += 1
        self.frame_var.set(f"Frames: {self.frame_count}")
        
        # Update data preview
        self.update_data_preview(frame_data)
    
    def update_data_preview(self, frame_data):
        """Update data preview area"""
        if frame_data['hands']:
            preview_text = f"Frame: {frame_data['frame']}\n"
            preview_text += f"Timestamp: {frame_data['timestamp']:.3f}\n"
            preview_text += f"Hands detected: {len(frame_data['hands'])}\n\n"
            
            for i, hand in enumerate(frame_data['hands']):
                preview_text += f"Hand {i+1} ({hand['handedness']}):\n"
                preview_text += f"Wrist position: ({hand['wrist_position']['x']:.3f}, {hand['wrist_position']['y']:.3f}, {hand['wrist_position']['z']:.3f})\n"
                preview_text += "Relative landmarks:\n"
                
                for j, landmark in enumerate(hand['landmarks']):
                    preview_text += f"  Point {j}: ({landmark['x']:.3f}, {landmark['y']:.3f}, {landmark['z']:.3f})\n"
                preview_text += "\n"
            
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(1.0, preview_text)
        else:
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(1.0, "No hands detected")
    
    def cv2_to_tkinter(self, image):
        """Convert OpenCV image to Tkinter format"""
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        return photo
    
    def __del__(self):
        """Destructor to release resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    app = HandLandmarkRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main()