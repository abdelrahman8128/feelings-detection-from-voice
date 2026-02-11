"""
Inference module for voice emotion detection.
Handles model loading, real-time prediction, and batch inference.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from sklearn.preprocessing import LabelEncoder

from audio_processor import AudioProcessor
from emotion_model import EmotionModelFactory


class EmotionPredictor:
    """Real-time emotion prediction from voice."""
    
    def __init__(self, 
                 model_path: str,
                 scalers_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize EmotionPredictor.
        
        Args:
            model_path: Path to trained model checkpoint
            scalers_path: Path to feature scalers (optional)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load model checkpoint
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model information
        self.model_type = checkpoint['model_type']
        self.model_params = checkpoint['model_params']
        self.label_encoder = checkpoint['label_encoder']
        processor_config = checkpoint['processor_config']
        
        # Override model params to match the saved model's number of classes
        self.model_params['n_classes'] = len(self.label_encoder.classes_)
        
        # Initialize audio processor
        self.processor = AudioProcessor(
            sample_rate=processor_config['sample_rate'],
            duration=processor_config['duration'],
            n_mfcc=processor_config['n_mfcc'],
            n_mels=processor_config['n_mels'],
            hop_length=processor_config['hop_length'],
            n_fft=processor_config['n_fft']
        )
        
        # Load scalers if available
        if scalers_path and Path(scalers_path).exists():
            self.processor.load_scalers(scalers_path)
        elif 'scalers' in checkpoint:
            self.processor.scalers = checkpoint['scalers']
            self.processor.is_fitted = True
        
        # Create and load model with the exact same parameters from training
        self.model = EmotionModelFactory.create_model(self.model_type, **self.model_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get emotion classes
        self.emotion_classes = self.label_encoder.classes_
        
        print(f"Model loaded successfully!")
        print(f"Model type: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Emotion classes: {list(self.emotion_classes)}")
    
    def predict(self, audio_path: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict emotion from audio file (simplified interface).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (predicted_emotion, confidence, sorted_probabilities)
        """
        result = self.predict_from_file(audio_path)
        
        if 'error' in result:
            raise RuntimeError(result['error'])
        
        # Sort probabilities by confidence
        sorted_probs = sorted(result['emotion_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        return result['predicted_emotion'], result['confidence'], sorted_probs
    
    def predict_from_file(self, audio_path: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Process audio file
            features = self.processor.process_audio_file(audio_path, extract_stats=True)
            
            if self.processor.is_fitted:
                features = self.processor.transform_features(features)
            
            # Prepare input based on model type
            if self.model_type == 'cnn':
                mel_spec = features['mel_spectrogram']
                input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
                
            elif self.model_type == 'rnn':
                mfcc = features['mfcc']
                input_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).to(self.device)
                
            elif self.model_type == 'transformer':
                mel_spec = features['mel_spectrogram']
                input_tensor = torch.FloatTensor(mel_spec.T).unsqueeze(0).to(self.device)
                
            elif self.model_type == 'hybrid':
                mel_spec = torch.FloatTensor(features['mel_spectrogram']).unsqueeze(0).unsqueeze(0).to(self.device)
                mfcc = torch.FloatTensor(features['mfcc'].T).unsqueeze(0).to(self.device)
                input_tensor = (mel_spec, mfcc)
            
            # Make prediction
            with torch.no_grad():
                if self.model_type == 'hybrid':
                    outputs = self.model(input_tensor[0], input_tensor[1])
                else:
                    outputs = self.model(input_tensor)
                
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_emotion = self.emotion_classes[predicted.item()]
                confidence_score = confidence.item()
                
                # Get probabilities for all emotions
                emotion_probs = {}
                for i, emotion in enumerate(self.emotion_classes):
                    emotion_probs[emotion] = probabilities[0][i].item()
            
            return {
                'predicted_emotion': predicted_emotion,
                'confidence': confidence_score,
                'emotion_probabilities': emotion_probs,
                'file_path': audio_path
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'file_path': audio_path
            }
    
    def predict_from_audio_data(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict emotion from raw audio data.
        
        Args:
            audio_data: Raw audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Resample if necessary
            if sample_rate != self.processor.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.processor.sample_rate)
            
            # Preprocess audio
            audio_data = self.processor.preprocess_audio(audio_data)
            
            # Extract features
            features = self.processor.extract_all_features(audio_data)
            
            # Compute statistical features if needed
            if self.processor.is_fitted:
                stat_features = {}
                for feature_name, feature_data in features.items():
                    stat_features[f"{feature_name}_stats"] = self.processor.compute_statistics(feature_data)
                features.update(stat_features)
                features = self.processor.transform_features(features)
            
            # Prepare input based on model type
            if self.model_type == 'cnn':
                mel_spec = features['mel_spectrogram']
                input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
                
            elif self.model_type == 'rnn':
                mfcc = features['mfcc']
                input_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).to(self.device)
                
            elif self.model_type == 'transformer':
                mel_spec = features['mel_spectrogram']
                input_tensor = torch.FloatTensor(mel_spec.T).unsqueeze(0).to(self.device)
                
            elif self.model_type == 'hybrid':
                mel_spec = torch.FloatTensor(features['mel_spectrogram']).unsqueeze(0).unsqueeze(0).to(self.device)
                mfcc = torch.FloatTensor(features['mfcc'].T).unsqueeze(0).to(self.device)
                input_tensor = (mel_spec, mfcc)
            
            # Make prediction
            with torch.no_grad():
                if self.model_type == 'hybrid':
                    outputs = self.model(input_tensor[0], input_tensor[1])
                else:
                    outputs = self.model(input_tensor)
                
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_emotion = self.emotion_classes[predicted.item()]
                confidence_score = confidence.item()
                
                # Get probabilities for all emotions
                emotion_probs = {}
                for i, emotion in enumerate(self.emotion_classes):
                    emotion_probs[emotion] = probabilities[0][i].item()
            
            return {
                'predicted_emotion': predicted_emotion,
                'confidence': confidence_score,
                'emotion_probabilities': emotion_probs
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predict emotions for a batch of audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for audio_path in audio_paths:
            result = self.predict_from_file(audio_path)
            results.append(result)
        return results
    
    def get_emotion_summary(self, predictions: List[Dict]) -> Dict[str, Union[int, float, Dict[str, int]]]:
        """
        Generate summary statistics from multiple predictions.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Summary statistics
        """
        valid_predictions = [p for p in predictions if 'predicted_emotion' in p]
        
        if not valid_predictions:
            return {'error': 'No valid predictions found'}
        
        # Count emotions
        emotion_counts = {}
        total_confidence = 0.0
        
        for pred in valid_predictions:
            emotion = pred['predicted_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += pred['confidence']
        
        # Calculate percentages
        total_predictions = len(valid_predictions)
        emotion_percentages = {emotion: (count / total_predictions) * 100 
                             for emotion, count in emotion_counts.items()}
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            'total_predictions': total_predictions,
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'dominant_emotion': dominant_emotion[0],
            'dominant_emotion_percentage': emotion_percentages[dominant_emotion[0]],
            'average_confidence': total_confidence / total_predictions
        }


class ModelExporter:
    """Export trained models for deployment."""
    
    def __init__(self, model_path: str):
        """
        Initialize ModelExporter.
        
        Args:
            model_path: Path to trained model checkpoint
        """
        self.model_path = model_path
        self.checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract model information
        self.model_type = self.checkpoint['model_type']
        self.model_params = self.checkpoint['model_params']
        self.label_encoder = self.checkpoint['label_encoder']
        
        # Override model params to match the saved model's number of classes
        self.model_params['n_classes'] = len(self.label_encoder.classes_)
        
        # Create model
        self.model = EmotionModelFactory.create_model(self.model_type, **self.model_params)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
    
    def export_to_onnx(self, output_path: str, input_shape: Optional[Tuple] = None) -> bool:
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input shape for the model
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            import onnx
            import onnxruntime
            
            # Determine input shape based on model type
            if input_shape is None:
                if self.model_type == 'cnn':
                    input_shape = (1, 1, 128, 130)  # (batch, channels, height, width)
                elif self.model_type == 'rnn':
                    input_shape = (1, 130, 13)  # (batch, sequence, features)
                elif self.model_type == 'transformer':
                    input_shape = (1, 130, 128)  # (batch, sequence, features)
                else:
                    raise ValueError(f"ONNX export not supported for {self.model_type} model")
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify the exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            print(f"Model successfully exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            return False
    
    def export_to_torchscript(self, output_path: str) -> bool:
        """
        Export model to TorchScript for mobile deployment.
        
        Args:
            output_path: Path to save TorchScript model
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Convert to TorchScript
            if self.model_type == 'hybrid':
                print("TorchScript export not supported for hybrid models")
                return False
            
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)
            
            print(f"Model successfully exported to TorchScript: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting to TorchScript: {e}")
            return False
    
    def create_deployment_package(self, output_dir: str) -> bool:
        """
        Create a complete deployment package with model and metadata.
        
        Args:
            output_dir: Directory to save deployment package
            
        Returns:
            True if package created successfully, False otherwise
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Save model checkpoint
            model_path = output_dir / "model.pth"
            torch.save(self.checkpoint, model_path)
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'model_params': self.model_params,
                'emotion_classes': self.label_encoder.classes_.tolist(),
                'processor_config': self.checkpoint['processor_config'],
                'input_requirements': self._get_input_requirements()
            }
            
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save label encoder
            encoder_path = output_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Export to different formats
            if self.model_type != 'hybrid':
                onnx_path = output_dir / "model.onnx"
                self.export_to_onnx(str(onnx_path))
                
                torchscript_path = output_dir / "model.pt"
                self.export_to_torchscript(str(torchscript_path))
            
            # Create README
            readme_content = self._generate_readme()
            readme_path = output_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            print(f"Deployment package created successfully: {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error creating deployment package: {e}")
            return False
    
    def _get_input_requirements(self) -> Dict:
        """Get input requirements for the model."""
        config = self.checkpoint['processor_config']
        
        requirements = {
            'audio_format': 'WAV, MP3, FLAC, or raw numpy array',
            'sample_rate': config['sample_rate'],
            'duration': f"{config['duration']} seconds (will be padded/trimmed)",
            'channels': 'mono (single channel)',
            'preprocessing': 'automatic normalization and feature extraction'
        }
        
        if self.model_type == 'cnn':
            requirements['input_type'] = 'mel spectrogram'
            requirements['input_shape'] = f"(1, {config['n_mels']}, time_frames)"
        elif self.model_type == 'rnn':
            requirements['input_type'] = 'MFCC features'
            requirements['input_shape'] = f"(time_frames, {config['n_mfcc']})"
        elif self.model_type == 'transformer':
            requirements['input_type'] = 'mel spectrogram'
            requirements['input_shape'] = f"(time_frames, {config['n_mels']})"
        elif self.model_type == 'hybrid':
            requirements['input_type'] = 'mel spectrogram + MFCC features'
            requirements['input_shape'] = f"mel: (1, {config['n_mels']}, time_frames), mfcc: (time_frames, {config['n_mfcc']})"
        
        return requirements
    
    def _generate_readme(self) -> str:
        """Generate README for deployment package."""
        return f"""# Voice Emotion Detection Model

## Model Information
- **Type**: {self.model_type.upper()}
- **Emotion Classes**: {', '.join(self.label_encoder.classes_)}
- **Parameters**: {sum(p.numel() for p in self.model.parameters()):,}

## Usage

### Python
```python
from inference import EmotionPredictor

# Load model
predictor = EmotionPredictor('model.pth')

# Predict from file
result = predictor.predict_from_file('audio.wav')
print(f"Emotion: {{result['predicted_emotion']}}")
print(f"Confidence: {{result['confidence']:.2f}}")
```

### Input Requirements
{json.dumps(self._get_input_requirements(), indent=2)}

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
"""


def main():
    """Main function for testing inference."""
    print("Voice Emotion Detection - Inference")
    print("=" * 50)
    
    # Example usage
    model_path = "checkpoints/cnn_best.pth"
    
    if Path(model_path).exists():
        # Initialize predictor
        predictor = EmotionPredictor(model_path)
        
        # Test with dummy audio
        dummy_audio = np.random.randn(22050 * 3)  # 3 seconds of audio
        result = predictor.predict_from_audio_data(dummy_audio, 22050)
        
        print("Prediction result:")
        print(json.dumps(result, indent=2))
        
        # Export model
        exporter = ModelExporter(model_path)
        exporter.create_deployment_package("deployment")
        
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")


if __name__ == "__main__":
    main()
