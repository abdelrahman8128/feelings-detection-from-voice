# Voice Emotion Detection for Employee Wellness

A comprehensive deep learning system for detecting emotions from voice recordings, specifically designed for employee wellness monitoring. This system helps managers identify team members who may need support or are ready to take on new tasks based on their emotional state.

## 🎯 Project Overview

This project provides a complete pipeline for:
- **Data Collection**: Automatic download of emotion speech datasets
- **Audio Processing**: Advanced feature extraction from voice recordings
- **Model Training**: Multiple deep learning architectures with checkpointing
- **Real-time Inference**: Emotion prediction from voice samples
- **Mobile Deployment**: Export models for Flutter/mobile applications
- **Offline/Online Usage**: Support for both local and server-based deployment

## 🏗️ Architecture

### Supported Models
1. **CNN Model**: Convolutional Neural Network for mel spectrogram analysis
2. **RNN Model**: LSTM with attention mechanism for temporal patterns
3. **Transformer Model**: Self-attention based architecture
4. **Hybrid Model**: Combined CNN-RNN for maximum performance

### Emotion Classes
The system detects 8 primary emotions:
- **Neutral**: Calm, balanced emotional state
- **Happy**: Positive, enthusiastic mood
- **Sad**: Low energy, melancholic state
- **Angry**: High arousal negative emotion
- **Fearful**: Anxious, worried state
- **Surprised**: Unexpected reaction
- **Disgust**: Aversion or rejection
- **Calm**: Peaceful, relaxed state

## 📁 Project Structure

```
feelings-detection-from-voice/
├── data_downloader.py          # Dataset download and management
├── audio_processor.py          # Audio preprocessing and feature extraction
├── emotion_model.py           # Deep learning model architectures
├── train.py                   # Training pipeline with checkpointing
├── inference.py               # Model inference and export utilities
├── emotion_detection_notebook.ipynb  # Jupyter notebook for experimentation
├── kaggle_notebook.ipynb      # Ready-to-use Kaggle notebook
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd feelings-detection-from-voice

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Kaggle API (Required for dataset download)

1. Create a Kaggle account at [kaggle.com](https://kaggle.com)
2. Go to Account → API → Create New API Token
3. Place the downloaded `kaggle.json` file in `~/.kaggle/` directory
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 3. Quick Training

```python
# Run the complete training pipeline
python train.py
```

Or use the Jupyter notebook for interactive experimentation:
```bash
jupyter notebook emotion_detection_notebook.ipynb
```

## ☁️ Running on Kaggle

You can run this project entirely on [Kaggle](https://www.kaggle.com/) using free GPU acceleration.
A ready-to-use notebook is provided at [`kaggle_notebook.ipynb`](kaggle_notebook.ipynb).

### Step-by-step

1. **Create a Kaggle notebook**
   - Go to [kaggle.com/code](https://www.kaggle.com/code) and click **+ New Notebook**.

2. **Add the datasets**
   - In the right sidebar click **+ Add Data** and search for:
     - `uwrfkaggler/ravdess-emotional-speech-audio`
     - `ejlok1/toronto-emotional-speech-set-tess`
   - (Optional) Also add `ejlok1/cremad` for a larger training set.
   - The datasets will be mounted automatically at `/kaggle/input/<dataset-name>/`.

3. **Enable GPU**
   - Open **Settings → Accelerator** and select **GPU**.

4. **Enable Internet**
   - Open **Settings → Internet** and set it to **On** (needed for `pip install` and `git clone`).

5. **Upload or import the notebook**
   - Option A: Upload `kaggle_notebook.ipynb` from this repository via **File → Import Notebook**.
   - Option B: In a new notebook, clone the repo and run the code:
     ```python
     !git clone https://github.com/abdelrahman8128/feelings-detection-from-voice.git
     import os; os.chdir('feelings-detection-from-voice')
     ```

6. **Run all cells** – the notebook installs missing packages, detects the Kaggle
   environment automatically, and trains the model using the pre-mounted datasets.

### Key differences from local setup

| | Local | Kaggle |
|---|---|---|
| **Datasets** | Downloaded via Kaggle API to `data/` | Pre-mounted at `/kaggle/input/` (no download needed) |
| **Output** | Saved in `checkpoints/` and `logs/` | Saved under `/kaggle/working/` |
| **GPU** | Requires local NVIDIA GPU + CUDA | Free GPU provided by Kaggle |
| **API key** | Required (`~/.kaggle/kaggle.json`) | Not required |

## 📊 Datasets

The system automatically downloads and processes these datasets:

### Primary Datasets
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **TESS**: Toronto Emotional Speech Set
- **CREMA-D**: Crowdsourced Emotional Multimodal Actors Dataset

### Data Statistics
- **Total Samples**: ~3,000+ audio files
- **Duration**: 3-second clips (automatically padded/trimmed)
- **Format**: WAV, MP3, FLAC supported
- **Sample Rate**: 22,050 Hz (automatically resampled)

## 🔧 Usage Examples

### Basic Training

```python
from train import EmotionTrainer

# Initialize trainer
trainer = EmotionTrainer(
    model_type='cnn',  # Options: 'cnn', 'rnn', 'transformer', 'hybrid'
    data_dir='data',
    checkpoint_dir='checkpoints'
)

# Prepare data (downloads datasets automatically)
trainer.prepare_data(
    datasets=['ravdess', 'tess'],
    batch_size=16
)

# Create and train model
trainer.create_model()
trainer.train(epochs=30)

# Evaluate
results = trainer.evaluate()
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

### Real-time Inference

```python
from inference import EmotionPredictor

# Load trained model
predictor = EmotionPredictor('checkpoints/cnn_best.pth')

# Predict from audio file
result = predictor.predict_from_file('audio_sample.wav')
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.3f}")

# Predict from raw audio data
import numpy as np
audio_data = np.random.randn(22050 * 3)  # 3 seconds
result = predictor.predict_from_audio_data(audio_data, 22050)
```

### Model Export for Mobile

```python
from inference import ModelExporter

# Export model for deployment
exporter = ModelExporter('checkpoints/cnn_best.pth')

# Create complete deployment package
exporter.create_deployment_package('deployment/')

# Export to ONNX for cross-platform use
exporter.export_to_onnx('model.onnx')

# Export to TorchScript for mobile
exporter.export_to_torchscript('model.pt')
```

## 🎛️ Configuration Options

### Model Parameters

```python
# CNN Model
model_params = {
    'n_mels': 128,           # Mel frequency bands
    'n_classes': 8,          # Number of emotions
    'dropout_rate': 0.3      # Regularization
}

# RNN Model
model_params = {
    'input_size': 13,        # MFCC coefficients
    'hidden_size': 128,      # LSTM hidden units
    'num_layers': 2,         # LSTM layers
    'bidirectional': True    # Bidirectional LSTM
}

# Training Configuration
training_config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4
}
```

### Audio Processing

```python
# Audio Processor Configuration
processor = AudioProcessor(
    sample_rate=22050,       # Target sample rate
    duration=3.0,            # Fixed duration (seconds)
    n_mfcc=13,              # MFCC coefficients
    n_mels=128,             # Mel frequency bands
    hop_length=512,         # STFT hop length
    n_fft=2048              # FFT window size
)
```

## 📈 Performance Metrics

### Expected Performance
- **CNN Model**: ~85-90% accuracy
- **RNN Model**: ~80-85% accuracy
- **Transformer Model**: ~88-92% accuracy
- **Hybrid Model**: ~90-95% accuracy

### Training Time (approximate)
- **CPU**: 2-4 hours for 30 epochs
- **GPU**: 30-60 minutes for 30 epochs

## 🔄 Resume Training

The system supports automatic checkpointing and resume capability:

```python
# Resume from checkpoint
trainer.train(
    epochs=50,
    resume_from='checkpoints/cnn_epoch_20.pth'
)
```

Checkpoints are automatically saved every 5 epochs and include:
- Model weights
- Optimizer state
- Training history
- Feature scalers
- Configuration

## 📱 Flutter Integration

### For Offline Use
1. Export model to ONNX or TorchScript format
2. Use `onnxruntime` or `pytorch_mobile` in Flutter
3. Include the exported model in your app assets

### For Online Use
1. Deploy the model on a server (Flask/FastAPI)
2. Create REST API endpoints for prediction
3. Send audio data from Flutter app to server

### Example Server Deployment

```python
from flask import Flask, request, jsonify
from inference import EmotionPredictor

app = Flask(__name__)
predictor = EmotionPredictor('model.pth')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    audio_file = request.files['audio']
    # Process audio and return prediction
    result = predictor.predict_from_file(audio_file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🛠️ Advanced Features

### Custom Dataset Integration
```python
# Add your own dataset
downloader = EmotionDataDownloader()
file_paths, labels = your_custom_data_loader()
```

### Feature Visualization
```python
# Visualize extracted features
processor = AudioProcessor()
features = processor.process_audio_file('sample.wav')

import matplotlib.pyplot as plt
plt.imshow(features['mel_spectrogram'])
plt.title('Mel Spectrogram')
plt.show()
```

### Batch Processing
```python
# Process multiple files
predictor = EmotionPredictor('model.pth')
results = predictor.predict_batch(['file1.wav', 'file2.wav'])

# Generate summary statistics
summary = predictor.get_emotion_summary(results)
print(f"Dominant emotion: {summary['dominant_emotion']}")
```

## 🔍 Monitoring and Logging

### TensorBoard Integration
```bash
# Start TensorBoard to monitor training
tensorboard --logdir=logs
```

### Training Visualization
- Loss curves (training/validation)
- Accuracy metrics
- Learning rate scheduling
- Confusion matrices
- Classification reports

## 🚨 Troubleshooting

### Common Issues

1. **Kaggle API Error**
   - Ensure `kaggle.json` is in `~/.kaggle/`
   - Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`

2. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

3. **Audio Loading Errors**
   - Install additional audio codecs: `pip install soundfile`
   - Check audio file format compatibility

4. **Low Accuracy**
   - Increase training epochs
   - Try different model architectures
   - Add data augmentation
   - Adjust learning rate

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB for datasets and models
- **GPU**: Optional but recommended for training

### Dependencies
See `requirements.txt` for complete list:
- PyTorch 2.0+
- librosa
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- kaggle

## 🤝 Employee Wellness Use Cases

### Manager Dashboard Integration
- **Real-time Monitoring**: Analyze team calls/meetings
- **Stress Detection**: Identify employees under pressure
- **Workload Distribution**: Assign tasks based on emotional state
- **Support Identification**: Proactively offer help to struggling team members

### Privacy Considerations
- **Consent**: Always obtain explicit consent before monitoring
- **Data Security**: Encrypt and secure all voice data
- **Anonymization**: Remove personal identifiers from analysis
- **Transparency**: Clearly communicate monitoring purposes

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **RAVDESS**: Livingstone & Russo (2018)
- **TESS**: University of Toronto
- **CREMA-D**: Cao et al. (2014)
- **PyTorch Team**: For the deep learning framework
- **Librosa Team**: For audio processing utilities

## 📞 Support

For questions, issues, or contributions:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include error logs and system information
4. Provide sample code to reproduce the problem

---

**Note**: This system is designed for research and development purposes. For production deployment in employee monitoring, ensure compliance with local privacy laws and obtain appropriate consent from all participants.
# feelings-detection-from-voice
