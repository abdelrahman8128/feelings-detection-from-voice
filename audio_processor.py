"""
Audio preprocessing and feature extraction module for voice emotion detection.
Handles audio loading, preprocessing, and feature extraction using librosa and PyTorch.
"""

import os
import librosa
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    """Handles audio preprocessing and feature extraction for emotion detection."""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 duration: float = 3.0,
                 n_mfcc: int = 13,
                 n_mels: int = 128,
                 hop_length: int = 512,
                 n_fft: int = 2048):
        """
        Initialize AudioProcessor with configuration parameters.
        
        Args:
            sample_rate: Target sample rate for audio
            duration: Fixed duration for audio clips (seconds)
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel frequency bands
            hop_length: Hop length for STFT
            n_fft: FFT window size
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_length = int(sample_rate * duration)
        
        # Feature scalers
        self.scalers = {}
        self.is_fitted = False
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Try librosa first
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            try:
                # Fallback to soundfile
                audio, sr = sf.read(file_path)
                if sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                return audio, self.sample_rate
            except Exception as e2:
                raise Exception(f"Could not load audio file {file_path}: {e}, {e2}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio by normalizing and fixing length.
        
        Args:
            audio: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Fix length
        if len(audio) > self.max_length:
            # Trim audio
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            # Pad audio
            pad_length = self.max_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        
        return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return mfcc
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features from audio."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return chroma
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral contrast features from audio."""
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return contrast
    
    def extract_tonnetz(self, audio: np.ndarray) -> np.ndarray:
        """Extract tonnetz (tonal centroid) features from audio."""
        tonnetz = librosa.feature.tonnetz(
            y=audio,
            sr=self.sample_rate
        )
        return tonnetz
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate from audio."""
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=self.hop_length
        )
        return zcr
    
    def extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff from audio."""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return rolloff
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid from audio."""
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return centroid
    
    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract RMS energy from audio."""
        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length
        )
        return rms
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all audio features.
        
        Args:
            audio: Preprocessed audio data
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Time-frequency features
        features['mfcc'] = self.extract_mfcc(audio)
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        features['chroma'] = self.extract_chroma(audio)
        features['spectral_contrast'] = self.extract_spectral_contrast(audio)
        features['tonnetz'] = self.extract_tonnetz(audio)
        
        # Spectral features
        features['zcr'] = self.extract_zero_crossing_rate(audio)
        features['spectral_rolloff'] = self.extract_spectral_rolloff(audio)
        features['spectral_centroid'] = self.extract_spectral_centroid(audio)
        features['rms_energy'] = self.extract_rms_energy(audio)
        
        return features
    
    def compute_statistics(self, feature: np.ndarray) -> np.ndarray:
        """
        Compute statistical measures (mean, std, min, max) across time axis.
        
        Args:
            feature: Feature array with shape (n_features, n_time_frames)
            
        Returns:
            Statistical features
        """
        stats = []
        stats.extend(np.mean(feature, axis=1))  # Mean
        stats.extend(np.std(feature, axis=1))   # Standard deviation
        stats.extend(np.min(feature, axis=1))   # Minimum
        stats.extend(np.max(feature, axis=1))   # Maximum
        
        return np.array(stats)
    
    def process_audio_file(self, file_path: str, extract_stats: bool = True) -> Dict[str, Any]:
        """
        Process a single audio file and extract features.
        
        Args:
            file_path: Path to audio file
            extract_stats: Whether to compute statistical features
            
        Returns:
            Dictionary containing processed features
        """
        # Load and preprocess audio
        audio, _ = self.load_audio(file_path)
        audio = self.preprocess_audio(audio)
        
        # Extract features
        features = self.extract_all_features(audio)
        
        if extract_stats:
            # Compute statistical features
            stat_features = {}
            for feature_name, feature_data in features.items():
                stat_features[f"{feature_name}_stats"] = self.compute_statistics(feature_data)
            features.update(stat_features)
        
        # Add raw audio for deep learning models
        features['raw_audio'] = audio
        
        return features
    
    def process_batch(self, file_paths: List[str], extract_stats: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of audio files.
        
        Args:
            file_paths: List of audio file paths
            extract_stats: Whether to compute statistical features
            
        Returns:
            List of feature dictionaries
        """
        processed_features = []
        
        for file_path in file_paths:
            try:
                features = self.process_audio_file(file_path, extract_stats)
                features['file_path'] = file_path
                processed_features.append(features)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return processed_features
    
    def fit_scalers(self, features_list: List[Dict[str, Any]]) -> None:
        """
        Fit scalers on the training data.
        
        Args:
            features_list: List of feature dictionaries from training data
        """
        # Collect all statistical features
        feature_names = set()
        for features in features_list:
            for key in features.keys():
                if key.endswith('_stats'):
                    feature_names.add(key)
        
        # Fit scalers for each feature type
        for feature_name in feature_names:
            feature_data = []
            for features in features_list:
                if feature_name in features:
                    feature_data.append(features[feature_name])
            
            if feature_data:
                feature_array = np.array(feature_data)
                scaler = StandardScaler()
                scaler.fit(feature_array)
                self.scalers[feature_name] = scaler
        
        self.is_fitted = True
    
    def transform_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform features using fitted scalers.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Scaled feature dictionary
        """
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        scaled_features = features.copy()
        
        for feature_name, scaler in self.scalers.items():
            if feature_name in features:
                scaled_features[feature_name] = scaler.transform(
                    features[feature_name].reshape(1, -1)
                ).flatten()
        
        return scaled_features
    
    def save_scalers(self, filepath: str) -> None:
        """Save fitted scalers to file."""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        scaler_data = {
            'scalers': self.scalers,
            'config': {
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'n_mfcc': self.n_mfcc,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'n_fft': self.n_fft
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
    
    def load_scalers(self, filepath: str) -> None:
        """Load scalers from file."""
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scalers = scaler_data['scalers']
        config = scaler_data['config']
        
        # Update configuration
        self.sample_rate = config['sample_rate']
        self.duration = config['duration']
        self.n_mfcc = config['n_mfcc']
        self.n_mels = config['n_mels']
        self.hop_length = config['hop_length']
        self.n_fft = config['n_fft']
        self.max_length = int(self.sample_rate * self.duration)
        
        self.is_fitted = True
    
    def create_spectrogram_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """
        Create mel spectrogram tensor for deep learning models.
        
        Args:
            audio: Preprocessed audio data
            
        Returns:
            Mel spectrogram tensor
        """
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        
        # Create mel spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Apply transform
        mel_spec = mel_transform(audio_tensor)
        
        # Convert to log scale
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        return mel_spec_db.squeeze(0)  # Remove batch dimension


def main():
    """Main function for testing the audio processor."""
    processor = AudioProcessor()
    
    print("Voice Emotion Detection - Audio Processor")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Sample Rate: {processor.sample_rate} Hz")
    print(f"  Duration: {processor.duration} seconds")
    print(f"  MFCC coefficients: {processor.n_mfcc}")
    print(f"  Mel bands: {processor.n_mels}")
    print(f"  Hop length: {processor.hop_length}")
    print(f"  FFT size: {processor.n_fft}")
    
    # Test with dummy data
    dummy_audio = np.random.randn(processor.max_length)
    print(f"\nTesting with dummy audio of length: {len(dummy_audio)}")
    
    # Extract features
    features = processor.extract_all_features(dummy_audio)
    
    print(f"\nExtracted features:")
    for feature_name, feature_data in features.items():
        print(f"  {feature_name}: {feature_data.shape}")
    
    # Test statistical features
    stats = processor.compute_statistics(features['mfcc'])
    print(f"\nMFCC statistics shape: {stats.shape}")


if __name__ == "__main__":
    main()
