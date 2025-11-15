# Hand Gesture Recognition System

A comprehensive system for recording, training, and real-time recognition of hand gestures using MediaPipe and PyTorch with GPU acceleration.

## 🚀 Features

- **Data Recording**: Capture hand landmark data from webcam with multiple export formats (NumPy, JSON, CSV)
- **Model Training**: Train MLP models for gesture recognition with comprehensive evaluation
- **Real-time Detection**: Live hand gesture recognition with GPU acceleration
- **Multi-format Support**: Flexible data storage options
- **Cross-hand Models**: Separate models for left and right hands
- **Visual Feedback**: Real-time visualization with landmarks and predictions

## 🛠 Requirements

### Hardware
- Webcam
- CUDA-compatible graphics drivers

### Software
- Python 3.8+
- CUDA 11.8+
- MediaPipe
- PyTorch with CUDA support
- OpenCV
- Tkinter

## 📦 Installation

1. **Clone the repository**
```bash
cd hand-gesture-recognition
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe opencv-python pillow numpy scikit-learn matplotlib pyyaml
```

3. **Download MediaPipe model (was already downloaded)**
```bash
# Download hand_landmarker.task from MediaPipe models
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## 🎯 Start a Demo

### 1.run webcam_demo.py

```bash
python webcam_demo.py
```

### 2.choose config file
choose `run_config.yaml`

### 3.click initalize system and start detection

## How to Use

### 1. Record Training Data
```bash
python generate_datasets_gui.py
```
- Click "Select Path" to choose save location
- Choose data format (NumPy recommended)
- Click "Start Recording" to begin capturing
- Perform gestures in front of camera
- Click "Stop Recording" when done

### 2. Train Models
Create a configuration file:
```yaml
datasets:
  Rock: ./data/rock.npy
  Paper: ./data/paper.npy
  Scissors: ./data/scissors.npy

training_params:
  epochs: 100
  learning_rate: 0.003
  batch_size: 64
  hidden_layers: [512, 256, 128]
  dropout: 0.3

output_settings:
  model_save_path: ./trained_models
  plot_save_path: ./training_plots
```

Train the model:
```bash
python gesture_trainer.py --config config.yaml
```

### 3. Real-time Recognition
```bash
python webcam_demo.py
```
- Load configuration file
- Click "Initialize System"
- Click "Start Detection" for real-time recognition

## ⚙️ Configuration

### Data Recording
- **Formats**: NumPy (.npy), JSON, CSV
- **Landmarks**: 21 points per hand with relative coordinates\

### Model Architecture
- **Type**: Multi-layer Perceptron (MLP)
- **Input**: 63 features (21 landmarks × 3 coordinates)
- **Hidden Layers**: Configurable (default: 512, 256, 128)
- **Activation**: ReLU with Batch Normalization
- **Regularization**: Dropout

### Training Parameters
- **Optimizer**: Adam with weight decay
- **Loss Function**: Cross Entropy
- **Scheduler**: ReduceLROnPlateau
- **Validation**: 15% split with stratification

## 🎮 Usage Examples

### Basic Gesture Set
```yaml
gestures: ["Rock", "Paper", "Scissors"]
```

### Advanced Gesture Set
```yaml
gestures: ["Thumbs Up", "Peace", "Fist", "Open Hand", "Pointing"]
```

### Custom Configuration
```yaml
datasets:
  Gesture1: ./data/gesture1.npy
  Gesture2: ./data/gesture2.npy
  Gesture3: ./data/gesture3.npy

training_params:
  epochs: 150
  learning_rate: 0.001
  batch_size: 32
  hidden_layers: [256, 128, 64]
  dropout: 0.2
```

## 📊 Performance

- **Inference Speed**: 30+ FPS on RTX 4070
- **Accuracy**: >95% with sufficient training data
- **Latency**: <50ms end-to-end processing
