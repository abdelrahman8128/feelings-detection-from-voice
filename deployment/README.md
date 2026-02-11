# Voice Emotion Detection Model

## Model Information
- **Type**: CNN
- **Emotion Classes**: angry, calm, disgust, fear, fearful, happy, neutral, pleasant_surprised, sad, surprised
- **Parameters**: 430,602

## Usage

### Python
```python
from inference import EmotionPredictor

# Load model
predictor = EmotionPredictor('model.pth')

# Predict from file
result = predictor.predict_from_file('audio.wav')
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Input Requirements
{
  "audio_format": "WAV, MP3, FLAC, or raw numpy array",
  "sample_rate": 22050,
  "duration": "3.0 seconds (will be padded/trimmed)",
  "channels": "mono (single channel)",
  "preprocessing": "automatic normalization and feature extraction",
  "input_type": "mel spectrogram",
  "input_shape": "(1, 128, time_frames)"
}

## Files
- `model.pth`: PyTorch model checkpoint
- `model.onnx`: ONNX format (if available)
- `model.pt`: TorchScript format (if available)
- `metadata.json`: Model metadata
- `label_encoder.pkl`: Label encoder for emotion classes

## Integration
This model can be integrated into:
- Flutter mobile apps (using ONNX or TorchScript)
- Web applications (using ONNX.js)
- Server deployments (using PyTorch)
- Edge devices (using optimized formats)
