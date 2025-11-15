import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import time
import argparse
import matplotlib.pyplot as plt

class GestureTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        
        # State variables
        self.training_data = {}
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.trained_model = None
        self.gesture_mapping = None
        
        print("Gesture Recognition Trainer")
        print("=" * 40)
    
    def load_config(self):
        """Load configuration from YAML file with proper error handling"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError("Configuration file is empty or invalid")
            
            # Validate required sections
            required_sections = ['datasets', 'training_params', 'output_settings']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in configuration")
                
                if config[section] is None:
                    raise ValueError(f"Section '{section}' is empty in configuration")
            
            # Validate datasets section
            if not isinstance(config['datasets'], dict):
                raise ValueError("'datasets' section must be a dictionary")
            
            if len(config['datasets']) < 2:
                raise ValueError("Need at least 2 datasets for training")
            
            print(f"Configuration loaded from: {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")
    
    def validate_config(self):
        """Validate the configuration structure"""
        config = self.config
        
        # Check datasets
        if not config.get('datasets'):
            raise ValueError("No datasets configured")
        
        # Check training parameters
        training_params = config.get('training_params', {})
        required_params = ['epochs', 'learning_rate', 'batch_size', 'hidden_layers', 'dropout']
        for param in required_params:
            if param not in training_params:
                raise ValueError(f"Missing training parameter: {param}")
        
        # Check output settings
        output_settings = config.get('output_settings', {})
        if 'model_save_path' not in output_settings:
            output_settings['model_save_path'] = 'trained_models'
        if 'plot_save_path' not in output_settings:
            output_settings['plot_save_path'] = 'training_plots'
    
    def load_datasets(self):
        """Load all datasets from configuration with proper error handling"""
        self.training_data = {}
        
        # Use .get() to safely access the datasets dictionary
        datasets_config = self.config.get('datasets', {})
        
        if not datasets_config:
            raise ValueError("No datasets found in configuration")
        
        print("Loading datasets...")
        for gesture_name, file_path in datasets_config.items():  # This should now work
            if not file_path:
                print(f"Warning: Empty file path for gesture '{gesture_name}', skipping")
                continue
                
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            try:
                data = np.load(file_path)
                self.training_data[gesture_name] = data
                print(f"✓ Loaded {gesture_name}: {len(data)} samples")
            except Exception as e:
                raise ValueError(f"Error loading {gesture_name} from {file_path}: {str(e)}")
        
        if len(self.training_data) < 2:
            raise ValueError("Need at least 2 valid gestures for training!")
        
        return True
    
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
    
    def train(self):
        """Train the gesture recognition model"""
        # Validate configuration first
        self.validate_config()
        
        print("Loading datasets...")
        self.load_datasets()
        
        print("\nStarting Model Training...")
        print("=" * 40)
        
        # Extract training parameters with safe access
        params = self.config.get('training_params', {})
        epochs = params.get('epochs', 100)
        learning_rate = params.get('learning_rate', 0.003)
        batch_size = params.get('batch_size', 64)
        hidden_layers = params.get('hidden_layers', [512, 256, 128])
        dropout = params.get('dropout', 0.3)
        
        # Rest of the training code remains the same...
        # Extract features and create labels
        print("Extracting features...")
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
        
        print(f"Dataset Statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Gestures: {', '.join(gesture_names)}")
        
        # Split data (train: 70%, validation: 15%, test: 15%)
        x_train, x_temp, y_train, y_temp = train_test_split(
            all_data, all_labels, test_size=0.3, stratify=all_labels, random_state=42
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"Data split:")
        print(f"  Training: {len(x_train)} samples ({len(x_train)/total_samples*100:.1f}%)")
        print(f"  Validation: {len(x_val)} samples ({len(x_val)/total_samples*100:.1f}%)")
        print(f"  Test: {len(x_test)} samples ({len(x_test)/total_samples*100:.1f}%)")
        
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
        
        print(f"\nModel Architecture:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Output size: {output_size}")
        print(f"  Dropout: {dropout}")
        print(f"  Device: {device}")
        
        model = MLP(input_size, output_size, hidden_layers, dropout).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_accuracy = 0.0
        
        print(f"\nStarting training for {epochs} epochs...")
        start_time = time.time()
        
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
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.best_model = model.state_dict().copy()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
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
        
        # Save the trained model
        self.trained_model = model
        self.model_accuracy = test_accuracy
        self.confusion_matrix = confusion_matrix
        
        # Display results
        self.display_results(test_accuracy, best_accuracy, confusion_matrix)
        
        # Save model and plots
        self.save_model()
        self.plot_results()
        
        return True
    
    def display_results(self, test_accuracy, best_val_accuracy, confusion_matrix):
        """Display training results"""
        print("\n" + "=" * 40)
        print("TRAINING RESULTS")
        print("=" * 40)
        
        print(f"\nAccuracy Scores:")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
        
        if self.gesture_mapping is None:
            raise ValueError("gesture mapping is none")
        
        print(f"\nGesture Mapping:")
        for idx, name in self.gesture_mapping.items():
            print(f"  {idx}: {name}")
        
        print(f"\nConfusion Matrix:")
        print("Actual \\ Predicted ->", end="")
        for i in range(len(self.gesture_mapping)):
            print(f"{i:4d}", end=" ")
        print()
        
        for i in range(len(self.gesture_mapping)):
            print(f"{i:3d}  ", end="")
            for j in range(len(self.gesture_mapping)):
                print(f"{self.confusion_matrix[i, j]:4d}", end=" ")
            print(f"  {self.gesture_mapping[i]}")
    
    def plot_results(self):
        """Plot training results and save to file"""
        if not self.training_history['loss']:
            print("No training data to plot!")
            return
        
        # Create plots directory with safe access to config
        output_settings = self.config.get('output_settings', {})
        plot_dir = output_settings.get('plot_save_path', 'training_plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        epochs = range(1, len(self.training_history['loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, self.training_history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, self.training_history['val_accuracy'], 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plot_dir, f"training_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plot saved to: {plot_path}")
    
    def save_model(self):
        """Save the trained model"""
        if self.trained_model is None or not hasattr(self, 'trained_model'):
            print("No trained model to save!")
            return
        
        try:
            # Create model directory with safe access to config
            output_settings = self.config.get('output_settings', {})
            model_dir = output_settings.get('model_save_path', 'trained_models')
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"gesture_model_{timestamp}.pth")
            
            torch.save({
                'model_state_dict': self.trained_model.state_dict(),
                'accuracy': self.model_accuracy,
                'training_history': self.training_history,
                'gesture_mapping': self.gesture_mapping,
                'confusion_matrix': self.confusion_matrix,
                'config': self.config
            }, model_path)
            
            print(f"Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Gesture Recognition Trainer')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = GestureTrainer(args.config)
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())