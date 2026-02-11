"""
Training module for voice emotion detection models.
Handles model training with checkpointing, resume capability, and comprehensive logging.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from data_downloader import EmotionDataDownloader
from audio_processor import AudioProcessor
from emotion_model import EmotionModelFactory, count_parameters


class EmotionDataset(Dataset):
    """Dataset class for emotion detection."""
    
    def __init__(self, 
                 file_paths: List[str], 
                 labels: List[str],
                 processor: AudioProcessor,
                 model_type: str = 'cnn',
                 transform_features: bool = True):
        """
        Initialize EmotionDataset.
        
        Args:
            file_paths: List of audio file paths
            labels: List of emotion labels
            processor: AudioProcessor instance
            model_type: Type of model ('cnn', 'rnn', 'transformer', 'hybrid')
            transform_features: Whether to apply feature scaling
        """
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.model_type = model_type.lower()
        self.transform_features = transform_features
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Dataset initialized with {len(file_paths)} samples")
        print(f"Emotion classes: {list(self.label_encoder.classes_)}")
        print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        try:
            # Process audio file
            features = self.processor.process_audio_file(
                self.file_paths[idx], 
                extract_stats=True
            )
            
            if self.transform_features and self.processor.is_fitted:
                features = self.processor.transform_features(features)
            
            # Prepare inputs based on model type
            if self.model_type == 'cnn':
                # Use mel spectrogram for CNN
                mel_spec = features['mel_spectrogram']
                # Add channel dimension and convert to tensor
                input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
                
            elif self.model_type == 'rnn':
                # Use MFCC for RNN
                mfcc = features['mfcc']
                # Transpose to (time_frames, n_mfcc)
                input_tensor = torch.FloatTensor(mfcc.T)
                
            elif self.model_type == 'transformer':
                # Use mel spectrogram for Transformer
                mel_spec = features['mel_spectrogram']
                # Transpose to (time_frames, n_mels)
                input_tensor = torch.FloatTensor(mel_spec.T)
                
            elif self.model_type == 'hybrid':
                # Use both mel spectrogram and MFCC
                mel_spec = torch.FloatTensor(features['mel_spectrogram']).unsqueeze(0)
                mfcc = torch.FloatTensor(features['mfcc'].T)
                input_tensor = (mel_spec, mfcc)
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            label = torch.LongTensor([self.encoded_labels[idx]])
            
            return input_tensor, label
            
        except Exception as e:
            print(f"Error processing {self.file_paths[idx]}: {e}")
            # Return dummy data in case of error
            if self.model_type == 'cnn':
                return torch.zeros(1, 128, 130), torch.LongTensor([0])
            elif self.model_type == 'rnn':
                return torch.zeros(130, 13), torch.LongTensor([0])
            elif self.model_type == 'transformer':
                return torch.zeros(130, 128), torch.LongTensor([0])
            else:  # hybrid
                return (torch.zeros(1, 128, 130), torch.zeros(130, 13)), torch.LongTensor([0])


class EmotionTrainer:
    """Trainer class for emotion detection models."""
    
    def __init__(self,
                 model_type: str = 'cnn',
                 model_params: Optional[Dict] = None,
                 data_dir: str = 'data',
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        """
        Initialize EmotionTrainer.
        
        Args:
            model_type: Type of model to train
            model_params: Model-specific parameters
            data_dir: Directory containing datasets
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.downloader = EmotionDataDownloader(data_dir)
        self.processor = AudioProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.label_encoder = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': []
        }
        
        print(f"EmotionTrainer initialized")
        print(f"Model type: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Log directory: {self.log_dir}")
    
    def prepare_data(self, 
                     datasets: Optional[List[str]] = None,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     batch_size: int = 32,
                     num_workers: int = 4) -> None:
        """
        Prepare datasets and data loaders.
        
        Args:
            datasets: List of dataset names to use
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
        """
        print("Preparing data...")
        
        # Download datasets if not available
        if datasets is None:
            datasets = ['ravdess', 'tess']  # Use smaller datasets for faster training
        
        for dataset in datasets:
            if not self.downloader.is_dataset_available(dataset):
                print(f"Downloading {dataset} dataset...")
                self.downloader.download_dataset(dataset)
        
        # Get file paths and labels
        file_paths, labels = self.downloader.get_file_paths_and_labels(datasets)
        
        if len(file_paths) == 0:
            raise ValueError("No audio files found. Please check dataset availability.")
        
        print(f"Total samples: {len(file_paths)}")
        
        # Create dataset
        dataset = EmotionDataset(
            file_paths=file_paths,
            labels=labels,
            processor=self.processor,
            model_type=self.model_type,
            transform_features=False  # Will fit scalers first
        )
        
        self.label_encoder = dataset.label_encoder
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Fit scalers on training data
        print("Fitting feature scalers...")
        train_features = []
        for i in tqdm(range(min(len(train_dataset), 1000)), desc="Processing training samples"):
            idx = train_dataset.indices[i]
            features = self.processor.process_audio_file(file_paths[idx], extract_stats=True)
            train_features.append(features)
        
        self.processor.fit_scalers(train_features)
        
        # Update datasets to use feature scaling
        dataset.transform_features = True
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print("Data preparation completed!")
    
    def create_model(self) -> None:
        """Create and initialize the model."""
        print(f"Creating {self.model_type} model...")
        
        # Set default parameters based on model type
        if self.model_type == 'cnn':
            default_params = {
                'n_mels': 128,
                'n_classes': len(self.label_encoder.classes_),
                'dropout_rate': 0.3
            }
        elif self.model_type == 'rnn':
            default_params = {
                'input_size': 13,
                'hidden_size': 128,
                'num_layers': 2,
                'n_classes': len(self.label_encoder.classes_),
                'dropout_rate': 0.3,
                'bidirectional': True
            }
        elif self.model_type == 'transformer':
            default_params = {
                'input_size': 128,
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'n_classes': len(self.label_encoder.classes_),
                'dropout_rate': 0.1
            }
        elif self.model_type == 'hybrid':
            default_params = {
                'n_mels': 128,
                'mfcc_size': 13,
                'n_classes': len(self.label_encoder.classes_),
                'dropout_rate': 0.3
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Update with user-provided parameters
        default_params.update(self.model_params)
        
        # Create model
        self.model = EmotionModelFactory.create_model(self.model_type, **default_params)
        self.model.to(self.device)
        
        # Print model info
        param_count = count_parameters(self.model)
        print(f"Model created with {param_count:,} parameters")
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch: int, writer: SummaryWriter) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if self.model_type == 'hybrid':
                mel_spec, mfcc = inputs
                mel_spec = mel_spec.to(self.device)
                mfcc = mfcc.to(self.device)
                inputs = (mel_spec, mfcc)
            else:
                inputs = inputs.to(self.device)
            
            targets = targets.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.model_type == 'hybrid':
                outputs = self.model(inputs[0], inputs[1])
            else:
                outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch: int, writer: SummaryWriter) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for inputs, targets in pbar:
                if self.model_type == 'hybrid':
                    mel_spec, mfcc = inputs
                    mel_spec = mel_spec.to(self.device)
                    mfcc = mfcc.to(self.device)
                    inputs = (mel_spec, mfcc)
                else:
                    inputs = inputs.to(self.device)
                
                targets = targets.squeeze().to(self.device)
                
                # Forward pass
                if self.model_type == 'hybrid':
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, targets)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'label_encoder': self.label_encoder,
            'processor_config': {
                'sample_rate': self.processor.sample_rate,
                'duration': self.processor.duration,
                'n_mfcc': self.processor.n_mfcc,
                'n_mels': self.processor.n_mels,
                'hop_length': self.processor.hop_length,
                'n_fft': self.processor.n_fft
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'{self.model_type}_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_type}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
        
        # Save scalers
        scaler_path = self.checkpoint_dir / f'{self.model_type}_scalers.pkl'
        self.processor.save_scalers(scaler_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the epoch number."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.history = checkpoint['history']
        
        # Load other components
        self.label_encoder = checkpoint['label_encoder']
        
        # Load scalers
        scaler_path = Path(checkpoint_path).parent / f"{self.model_type}_scalers.pkl"
        if scaler_path.exists():
            self.processor.load_scalers(scaler_path)
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}")
        
        return epoch
    
    def train(self, 
              epochs: int = 50,
              resume_from: Optional[str] = None,
              save_every: int = 5) -> None:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Initialize tensorboard writer
        writer = SummaryWriter(self.log_dir / f'{self.model_type}_{int(time.time())}')
        
        start_epoch = 0
        best_val_acc = 0.0
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            if self.history['val_acc']:
                best_val_acc = max(self.history['val_acc'])
        
        # Training loop
        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch, writer)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch, writer)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log metrics
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Accuracy', train_acc, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_acc, epoch)
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['epochs'].append(epoch)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        writer.close()
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        if data_loader is None:
            data_loader = self.test_loader
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc='Evaluating')
            
            for inputs, targets in pbar:
                if self.model_type == 'hybrid':
                    mel_spec, mfcc = inputs
                    mel_spec = mel_spec.to(self.device)
                    mfcc = mfcc.to(self.device)
                    inputs = (mel_spec, mfcc)
                else:
                    inputs = inputs.to(self.device)
                
                targets = targets.squeeze().to(self.device)
                
                # Forward pass
                if self.model_type == 'hybrid':
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_loss = total_loss / len(data_loader)
        
        # Generate classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }
        
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {avg_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        return results


def main():
    """Main function for training the emotion detection model."""
    print("Voice Emotion Detection - Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'model_type': 'cnn',  # Options: 'cnn', 'rnn', 'transformer', 'hybrid'
        'epochs': 30,
        'batch_size': 16,
        'datasets': ['ravdess', 'tess'],
        'resume_from': None  # Set to checkpoint path to resume training
    }
    
    # Initialize trainer
    trainer = EmotionTrainer(
        model_type=config['model_type'],
        model_params={},
        data_dir='data',
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    # Prepare data
    trainer.prepare_data(
        datasets=config['datasets'],
        batch_size=config['batch_size']
    )
    
    # Create model
    trainer.create_model()
    
    # Train model
    trainer.train(
        epochs=config['epochs'],
        resume_from=config['resume_from']
    )
    
    # Evaluate model
    results = trainer.evaluate()
    
    # Save final results
    results_path = trainer.checkpoint_dir / f"{config['model_type']}_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'class_names': results['class_names'].tolist()
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
