"""
Deep learning model architectures for voice emotion detection.
Includes CNN, RNN, and hybrid models optimized for emotion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Tuple, Optional
import numpy as np


class EmotionCNN(nn.Module):
    """Convolutional Neural Network for emotion detection from mel spectrograms."""
    
    def __init__(self, 
                 n_mels: int = 128,
                 n_classes: int = 8,
                 dropout_rate: float = 0.3):
        """
        Initialize EmotionCNN.
        
        Args:
            n_mels: Number of mel frequency bands
            n_classes: Number of emotion classes
            dropout_rate: Dropout rate for regularization
        """
        super(EmotionCNN, self).__init__()
        
        self.n_mels = n_mels
        self.n_classes = n_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input shape: (batch_size, 1, n_mels, time_frames)
        
        # Convolutional layers with batch norm and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class EmotionRNN(nn.Module):
    """Recurrent Neural Network for emotion detection from sequential features."""
    
    def __init__(self,
                 input_size: int = 13,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 n_classes: int = 8,
                 dropout_rate: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize EmotionRNN.
        
        Args:
            input_size: Size of input features (e.g., MFCC coefficients)
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            n_classes: Number of emotion classes
            dropout_rate: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(EmotionRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Linear(lstm_output_size, 1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input shape: (batch_size, sequence_length, input_size)
        
        # LSTM layers
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(attended_output)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class EmotionTransformer(nn.Module):
    """Transformer model for emotion detection from audio features."""
    
    def __init__(self,
                 input_size: int = 128,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 n_classes: int = 8,
                 dropout_rate: float = 0.1):
        """
        Initialize EmotionTransformer.
        
        Args:
            input_size: Size of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            n_classes: Number of emotion classes
            dropout_rate: Dropout rate
        """
        super(EmotionTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input shape: (batch_size, sequence_length, input_size)
        
        # Project input to model dimension
        x = self.input_projection(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class HybridEmotionModel(nn.Module):
    """Hybrid model combining CNN and RNN for emotion detection."""
    
    def __init__(self,
                 n_mels: int = 128,
                 mfcc_size: int = 13,
                 n_classes: int = 8,
                 dropout_rate: float = 0.3):
        """
        Initialize HybridEmotionModel.
        
        Args:
            n_mels: Number of mel frequency bands
            mfcc_size: Size of MFCC features
            n_classes: Number of emotion classes
            dropout_rate: Dropout rate
        """
        super(HybridEmotionModel, self).__init__()
        
        # CNN branch for mel spectrograms
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # RNN branch for MFCC features
        self.rnn_branch = nn.LSTM(
            input_size=mfcc_size,
            hidden_size=64,
            num_layers=2,
            dropout=dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        
        # Fusion and classification layers
        self.fusion = nn.Linear(128 + 128, 256)  # CNN features + RNN features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, mel_spec, mfcc):
        """
        Forward pass through the hybrid network.
        
        Args:
            mel_spec: Mel spectrogram input (batch_size, 1, n_mels, time_frames)
            mfcc: MFCC input (batch_size, sequence_length, mfcc_size)
        """
        # CNN branch
        cnn_features = self.cnn_branch(mel_spec)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # RNN branch
        rnn_out, (hidden, _) = self.rnn_branch(mfcc)
        # Use the last hidden state from both directions
        rnn_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Fusion
        fused_features = torch.cat([cnn_features, rnn_features], dim=1)
        fused_features = F.relu(self.fusion(fused_features))
        
        # Classification
        output = self.classifier(fused_features)
        
        return output


class EmotionModelFactory:
    """Factory class for creating emotion detection models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> nn.Module:
        """
        Create emotion detection model.
        
        Args:
            model_type: Type of model ('cnn', 'rnn', 'transformer', 'hybrid')
            **kwargs: Model-specific parameters
            
        Returns:
            Initialized model
        """
        if model_type.lower() == 'cnn':
            return EmotionCNN(**kwargs)
        elif model_type.lower() == 'rnn':
            return EmotionRNN(**kwargs)
        elif model_type.lower() == 'transformer':
            return EmotionTransformer(**kwargs)
        elif model_type.lower() == 'hybrid':
            return HybridEmotionModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict:
        """Get information about model architecture."""
        info = {
            'cnn': {
                'description': 'Convolutional Neural Network for mel spectrograms',
                'input_type': 'mel_spectrogram',
                'advantages': ['Good for spatial patterns', 'Translation invariant', 'Efficient'],
                'best_for': 'Spectral pattern recognition'
            },
            'rnn': {
                'description': 'Recurrent Neural Network with attention for sequential features',
                'input_type': 'mfcc_sequence',
                'advantages': ['Good for temporal patterns', 'Attention mechanism', 'Variable length'],
                'best_for': 'Temporal pattern recognition'
            },
            'transformer': {
                'description': 'Transformer model with self-attention',
                'input_type': 'feature_sequence',
                'advantages': ['Self-attention', 'Parallel processing', 'Long-range dependencies'],
                'best_for': 'Complex temporal relationships'
            },
            'hybrid': {
                'description': 'Hybrid CNN-RNN model combining spatial and temporal features',
                'input_type': 'mel_spectrogram + mfcc',
                'advantages': ['Best of both worlds', 'Multi-modal', 'Robust'],
                'best_for': 'Maximum performance'
            }
        }
        return info.get(model_type.lower(), {})


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main function for testing the emotion models."""
    print("Voice Emotion Detection - Model Architectures")
    print("=" * 50)
    
    # Test different model architectures
    models = {
        'CNN': EmotionCNN(n_mels=128, n_classes=8),
        'RNN': EmotionRNN(input_size=13, n_classes=8),
        'Transformer': EmotionTransformer(input_size=128, n_classes=8),
        'Hybrid': HybridEmotionModel(n_mels=128, mfcc_size=13, n_classes=8)
    }
    
    for model_name, model in models.items():
        param_count = count_parameters(model)
        print(f"\n{model_name} Model:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Architecture: {model.__class__.__name__}")
    
    # Test forward pass
    batch_size = 4
    n_mels = 128
    time_frames = 130
    mfcc_frames = 130
    mfcc_size = 13
    
    # Create dummy inputs
    mel_input = torch.randn(batch_size, 1, n_mels, time_frames)
    mfcc_input = torch.randn(batch_size, mfcc_frames, mfcc_size)
    
    print(f"\nTesting forward pass with batch size {batch_size}:")
    
    # Test CNN
    cnn_output = models['CNN'](mel_input)
    print(f"  CNN output shape: {cnn_output.shape}")
    
    # Test RNN
    rnn_output = models['RNN'](mfcc_input)
    print(f"  RNN output shape: {rnn_output.shape}")
    
    # Test Transformer
    transformer_input = torch.randn(batch_size, time_frames, 128)
    transformer_output = models['Transformer'](transformer_input)
    print(f"  Transformer output shape: {transformer_output.shape}")
    
    # Test Hybrid
    hybrid_output = models['Hybrid'](mel_input, mfcc_input)
    print(f"  Hybrid output shape: {hybrid_output.shape}")


if __name__ == "__main__":
    main()
