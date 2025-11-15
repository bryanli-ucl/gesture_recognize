import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import threading
import os
import json

class GestureTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition Trainer")
        self.root.geometry("1200x800")
        
        # State variables
        self.training_in_progress = False
        self.training_data = {}
        self.dataset_files = {}
        self.training_history = {'loss': [], 'accuracy': []}
        
        # Create GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Setup GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Left panel - Dataset selection
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Right panel - Training and visualization
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Dataset Selection Section
        dataset_frame = ttk.LabelFrame(left_frame, text="Dataset Selection", padding="10")
        dataset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Dataset controls
        dataset_control_frame = ttk.Frame(dataset_frame)
        dataset_control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Button(dataset_control_frame, text="Add Dataset", 
                  command=self.add_dataset).grid(row=0, column=0, padx=5)
        ttk.Button(dataset_control_frame, text="Remove Selected", 
                  command=self.remove_dataset).grid(row=0, column=1, padx=5)
        ttk.Button(dataset_control_frame, text="Clear All", 
                  command=self.clear_datasets).grid(row=0, column=2, padx=5)
        
        # Dataset list
        list_frame = ttk.Frame(dataset_frame)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=5)
        
        # Create treeview for datasets
        columns = ('Gesture', 'File', 'Samples')
        self.dataset_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        
        # Define headings
        self.dataset_tree.heading('Gesture', text='Gesture Name')
        self.dataset_tree.heading('File', text='File Name')
        self.dataset_tree.heading('Samples', text='Samples')
        
        # Define columns
        self.dataset_tree.column('Gesture', width=150)
        self.dataset_tree.column('File', width=200)
        self.dataset_tree.column('Samples', width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscrollcommand=scrollbar.set)
        
        self.dataset_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Gesture naming
        naming_frame = ttk.Frame(dataset_frame)
        naming_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(naming_frame, text="Gesture Name:").grid(row=0, column=0, sticky="w")
        self.gesture_name_var = tk.StringVar()
        ttk.Entry(naming_frame, textvariable=self.gesture_name_var, width=20).grid(row=0, column=1, padx=5)
        ttk.Button(naming_frame, text="Update Name", 
                  command=self.update_gesture_name).grid(row=0, column=2, padx=5)
        
        # Training Controls Section
        control_frame = ttk.LabelFrame(left_frame, text="Training Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Training parameters
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(param_frame, text="Learning Rate:").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.lr_var = tk.StringVar(value="0.003")
        ttk.Entry(param_frame, textvariable=self.lr_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=4, sticky="w", padx=(20,0))
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(param_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=5, padx=5)
        
        # Model architecture
        arch_frame = ttk.Frame(control_frame)
        arch_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(arch_frame, text="Hidden Layers:").grid(row=0, column=0, sticky="w")
        self.hidden_layers_var = tk.StringVar(value="512,256,128")
        ttk.Entry(arch_frame, textvariable=self.hidden_layers_var, width=20).grid(row=0, column=1, padx=5)
        
        ttk.Label(arch_frame, text="Dropout:").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.dropout_var = tk.StringVar(value="0.3")
        ttk.Entry(arch_frame, textvariable=self.dropout_var, width=10).grid(row=0, column=3, padx=5)
        
        # Training buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Button(button_frame, text="Start Training", 
                  command=self.start_training).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Plot Results", 
                  command=self.plot_results).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Model", 
                  command=self.save_model).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Save/Load Config", 
                  command=self.save_config).grid(row=0, column=3, padx=5)
        
        # Training progress
        self.progress_var = tk.StringVar(value="Status: Ready - Add datasets to begin")
        ttk.Label(control_frame, textvariable=self.progress_var).grid(row=3, column=0, sticky="w", pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(control_frame, mode='determinate')
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Dataset info section
        info_frame = ttk.LabelFrame(left_frame, text="Dataset Information", padding="10")
        info_frame.grid(row=2, column=0, sticky="nsew")
        
        self.info_text = tk.Text(info_frame, height=10, width=50)
        self.info_text.grid(row=0, column=0, sticky="nsew")
        scrollbar_info = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar_info.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=scrollbar_info.set)
        
        # Visualization area
        viz_frame = ttk.LabelFrame(right_frame, text="Training Visualization", padding="10")
        viz_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results area
        results_frame = ttk.LabelFrame(right_frame, text="Training Results", padding="10")
        results_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        
        self.results_text = tk.Text(results_frame, height=8, width=80)
        self.results_text.grid(row=0, column=0, sticky="ew")
        scrollbar_results = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar_results.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar_results.set)
        
        # Configure weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        dataset_frame.columnconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        control_frame.columnconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def add_dataset(self):
        """Add a dataset file"""
        file_path = filedialog.askopenfilename(
            title="Select dataset file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_path:
            # Get filename without extension as default gesture name
            file_name = os.path.basename(file_path)
            gesture_name = os.path.splitext(file_name)[0]
            
            try:
                # Load data to check validity and get sample count
                data = np.load(file_path)
                sample_count = len(data)
                
                # Store dataset
                self.dataset_files[gesture_name] = file_path
                self.training_data[gesture_name] = data
                
                # Add to treeview
                self.dataset_tree.insert('', 'end', values=(gesture_name, file_name, sample_count))
                
                self.update_dataset_info()
                self.progress_var.set(f"Status: Added dataset '{gesture_name}' with {sample_count} samples")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def remove_dataset(self):
        """Remove selected dataset"""
        selected_item = self.dataset_tree.selection()
        if selected_item:
            item = self.dataset_tree.item(selected_item[0])
            gesture_name = item['values'][0]
            
            # Remove from storage
            if gesture_name in self.dataset_files:
                del self.dataset_files[gesture_name]
            if gesture_name in self.training_data:
                del self.training_data[gesture_name]
            
            # Remove from treeview
            self.dataset_tree.delete(selected_item[0])
            
            self.update_dataset_info()
            self.progress_var.set(f"Status: Removed dataset '{gesture_name}'")
    
    def clear_datasets(self):
        """Clear all datasets"""
        if messagebox.askyesno("Confirm", "Are you sure you want to remove all datasets?"):
            self.dataset_files.clear()
            self.training_data.clear()
            self.dataset_tree.delete(*self.dataset_tree.get_children())
            self.update_dataset_info()
            self.progress_var.set("Status: All datasets removed")
    
    def update_gesture_name(self):
        """Update the name of selected gesture"""
        selected_item = self.dataset_tree.selection()
        new_name = self.gesture_name_var.get().strip()
        
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a dataset to rename")
            return
        
        if not new_name:
            messagebox.showwarning("Warning", "Please enter a gesture name")
            return
        
        item = self.dataset_tree.item(selected_item[0])
        old_name = item['values'][0]
        file_name = item['values'][1]
        sample_count = item['values'][2]
        
        if old_name == new_name:
            return
        
        # Check if new name already exists
        if new_name in self.dataset_files:
            messagebox.showerror("Error", f"Gesture name '{new_name}' already exists")
            return
        
        # Update storage
        if old_name in self.dataset_files:
            self.dataset_files[new_name] = self.dataset_files.pop(old_name)
        if old_name in self.training_data:
            self.training_data[new_name] = self.training_data.pop(old_name)
        
        # Update treeview
        self.dataset_tree.item(selected_item[0], values=(new_name, file_name, sample_count))
        
        self.update_dataset_info()
        self.progress_var.set(f"Status: Renamed '{old_name}' to '{new_name}'")
    
    def update_dataset_info(self):
        """Update dataset information display"""
        total_samples = 0
        total_gestures = len(self.training_data)
        
        info_text = f"Dataset Summary:\n\n"
        info_text += f"Total Gestures: {total_gestures}\n"
        
        for gesture_name, data in self.training_data.items():
            samples = len(data)
            total_samples += samples
            info_text += f"\n{gesture_name}:\n"
            info_text += f"  Samples: {samples}\n"
            info_text += f"  Data type: {data.dtype}\n"
            info_text += f"  Shape: {data.shape}\n"
        
        info_text += f"\nTotal Samples: {total_samples}\n"
        
        if total_gestures > 0:
            feature_count = self.extract_hand_features(self.training_data[list(self.training_data.keys())[0][:1]]).shape[1]
            info_text += f"Feature Dimension: {feature_count}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def extract_hand_features(self, data_array):
        """Extract hand landmark features from structured array"""
        features = []
        
        for sample in data_array:
            if sample['num_hands'] > 0:
                # Use first detected hand
                hand_data = sample['hand_data'][0]
                landmarks = hand_data['landmarks'].flatten()
                features.append(landmarks)
            else:
                # If no hand detected, use zeros
                features.append(np.zeros(21 * 3, dtype=np.float32))
        
        return np.array(features)
    
    def start_training(self):
        """Start model training in a separate thread"""
        if len(self.training_data) < 2:
            messagebox.showerror("Error", "Need at least 2 gestures for training!")
            return
        
        if self.training_in_progress:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
        
        # Get training parameters
        try:
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.lr_var.get())
            batch_size = int(self.batch_size_var.get())
            dropout = float(self.dropout_var.get())
            
            # Parse hidden layers
            hidden_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',')]
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
            return
        
        self.training_in_progress = True
        self.progress_var.set("Training: Starting...")
        self.progress_bar['value'] = 0
        
        # Start training in separate thread
        training_thread = threading.Thread(
            target=self.train_model, 
            args=(epochs, learning_rate, batch_size, hidden_layers, dropout)
        )
        training_thread.daemon = True
        training_thread.start()
    
    def train_model(self, epochs, learning_rate, batch_size, hidden_layers, dropout):
        """Train the gesture recognition model"""
        try:
            # Update progress
            self.progress_var.set("Training: Extracting features...")
            
            # Extract features and create labels
            all_features = []
            all_labels = []
            gesture_names = list(self.training_data.keys())
            self.gesture_mapping = {i: name for i, name in enumerate(gesture_names)}
            
            for label_idx, (gesture_name, data) in enumerate(self.training_data.items()):
                features = self.extract_hand_features(data)
                labels = np.full(len(features), label_idx, dtype=np.int64)
                
                all_features.append(features)
                all_labels.append(labels)
            
            # Combine data
            all_data = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            # Display dataset statistics
            total_samples = len(all_data)
            feature_dim = all_data.shape[1]
            num_classes = len(gesture_names)
            
            self.root.after(0, lambda: self.progress_var.set(
                f"Training: {total_samples} samples, {feature_dim} features, {num_classes} classes"
            ))
            
            # Split data (train: 70%, validation: 15%, test: 15%)
            x_train, x_temp, y_train, y_temp = train_test_split(
                all_data, all_labels, test_size=0.3, stratify=all_labels, random_state=42
            )
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
            )
            
            # Convert to PyTorch tensors
            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)
            x_val = torch.FloatTensor(x_val)
            y_val = torch.LongTensor(y_val)
            x_test = torch.FloatTensor(x_test)
            y_test = torch.LongTensor(y_test)
            
            # Create data loaders
            train_dataset = TensorDataset(x_train, y_train)
            val_dataset = TensorDataset(x_val, y_val)
            test_dataset = TensorDataset(x_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Define MLP model
            class MLP(nn.Module):
                def __init__(self, input_size, output_size, hidden_layers, dropout=0.3):
                    super(MLP, self).__init__()
                    layers = []
                    current_size = input_size
                    
                    for hidden_size in hidden_layers:
                        layers.extend([
                            nn.Linear(current_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ])
                        current_size = hidden_size
                    
                    layers.append(nn.Linear(current_size, output_size))
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_size = x_train.shape[1]
            output_size = num_classes
            
            model = MLP(input_size, output_size, hidden_layers, dropout).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop
            self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            best_accuracy = 0.0
            
            self.root.after(0, lambda: self.progress_var.set(f"Training: Starting {epochs} epochs..."))
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                self.training_history['loss'].append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += criterion(output, target).item()
                        
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total
                
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                scheduler.step(avg_val_loss)
                
                # Update progress
                progress = (epoch + 1) / epochs * 100
                self.root.after(0, lambda p=progress: self.progress_bar.config(value=p))
                
                self.root.after(0, lambda: self.progress_var.set(
                    f"Training: Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                ))
                
                # Save best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    self.best_model = model.state_dict().copy()
                
                # Update plot every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.root.after(0, self.plot_results)
            
            # Final evaluation on test set
            model.load_state_dict(self.best_model)
            model.eval()
            test_correct = 0
            test_total = 0
            
            # Calculate confusion matrix
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
                    
                    # Update confusion matrix
                    for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
                        confusion_matrix[t, p] += 1
            
            test_accuracy = test_correct / test_total
            
            self.training_in_progress = False
            self.progress_bar['value'] = 100
            
            # Save the trained model
            self.trained_model = model
            self.model_accuracy = test_accuracy
            self.confusion_matrix = confusion_matrix
            
            # Update results display
            self.update_results_display(test_accuracy, best_accuracy, confusion_matrix)
            
            # Final plot update
            self.root.after(0, self.plot_results)
            
            self.root.after(0, lambda: self.progress_var.set(
                f"Training Completed! Test Accuracy: {test_accuracy:.4f}"
            ))
            
        except Exception as e:
            self.training_in_progress = False
            self.root.after(0, lambda: self.progress_var.set("Training: Failed!"))
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def update_results_display(self, test_accuracy, best_val_accuracy, confusion_matrix):
        """Update training results display"""
        results_text = "Training Results:\n\n"
        results_text += f"Test Accuracy: {test_accuracy:.4f}\n"
        results_text += f"Best Validation Accuracy: {best_val_accuracy:.4f}\n\n"
        
        results_text += "Gesture Mapping:\n"
        for idx, name in self.gesture_mapping.items():
            results_text += f"  {idx}: {name}\n"
        
        results_text += "\nConfusion Matrix:\n"
        results_text += "Actual \\ Predicted ->\n"
        
        # Header row
        results_text += "     "
        for i in range(len(self.gesture_mapping)):
            results_text += f"{i:4d} "
        results_text += "\n"
        
        # Matrix rows
        for i in range(len(self.gesture_mapping)):
            results_text += f"{i:3d}  "
            for j in range(len(self.gesture_mapping)):
                results_text += f"{confusion_matrix[i, j]:4d} "
            results_text += f"  {self.gesture_mapping[i]}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
    
    def plot_results(self):
        """Plot training results"""
        if not self.training_history['loss']:
            messagebox.showwarning("Warning", "No training data to plot!")
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        epochs = range(1, len(self.training_history['loss']) + 1)
        
        # Plot loss
        self.ax1.plot(epochs, self.training_history['loss'], 'b-', label='Training Loss', linewidth=2)
        self.ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        self.ax1.set_title('Training and Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        self.ax2.plot(epochs, self.training_history['val_accuracy'], 'g-', label='Validation Accuracy', linewidth=2)
        self.ax2.set_title('Validation Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 1)
        
        # Adjust layout and draw
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_model(self):
        """Save the trained model"""
        if not hasattr(self, 'trained_model'):
            messagebox.showerror("Error", "No trained model to save!")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pth",
                filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
            )
            
            if file_path:
                torch.save({
                    'model_state_dict': self.trained_model.state_dict(),
                    'accuracy': self.model_accuracy,
                    'training_history': self.training_history,
                    'gesture_mapping': self.gesture_mapping,
                    'confusion_matrix': self.confusion_matrix
                }, file_path)
                
                messagebox.showinfo("Success", f"Model saved successfully to: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving model: {str(e)}")
    
    def save_config(self):
        """Save or load dataset configuration"""
        try:
            config_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save dataset configuration"
            )
            
            if config_path:
                config = {
                    'datasets': self.dataset_files,
                    'training_params': {
                        'epochs': self.epochs_var.get(),
                        'learning_rate': self.lr_var.get(),
                        'batch_size': self.batch_size_var.get(),
                        'hidden_layers': self.hidden_layers_var.get(),
                        'dropout': self.dropout_var.get()
                    }
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                messagebox.showinfo("Success", f"Configuration saved to: {config_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving configuration: {str(e)}")

def main():
    root = tk.Tk()
    app = GestureTrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()