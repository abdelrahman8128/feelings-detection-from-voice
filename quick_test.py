#!/usr/bin/env python3
"""
Quick test script to verify the voice emotion detection system is working.
"""

import sys
import numpy as np
from pathlib import Path

# Import our modules
from data_downloader import EmotionDataDownloader
from audio_processor import AudioProcessor
from emotion_model import EmotionModelFactory
from train import EmotionTrainer

def test_data_availability():
    """Test if datasets are available and accessible."""
    print("🔍 Testing Data Availability...")
    
    downloader = EmotionDataDownloader()
    
    # Check dataset info
    info = downloader.get_dataset_info()
    print(f"   Available datasets: {len(info)}")
    
    # Get file paths and labels
    file_paths, labels = downloader.get_file_paths_and_labels(['ravdess', 'tess'])
    print(f"   Total audio files: {len(file_paths)}")
    print(f"   Unique emotions: {len(set(labels))}")
    print(f"   Emotions found: {sorted(set(labels))}")
    
    return len(file_paths) > 0

def test_audio_processing():
    """Test audio processing capabilities."""
    print("\n🎵 Testing Audio Processing...")
    
    processor = AudioProcessor()
    
    # Test with dummy audio
    dummy_audio = np.random.randn(processor.max_length)
    features = processor.extract_all_features(dummy_audio)
    
    print(f"   Sample rate: {processor.sample_rate} Hz")
    print(f"   Duration: {processor.duration} seconds")
    print(f"   Extracted {len(features)} feature types:")
    
    for name, data in features.items():
        print(f"     - {name}: {data.shape}")
    
    return True

def test_model_architectures():
    """Test model creation."""
    print("\n🧠 Testing Model Architectures...")
    
    models = ['cnn', 'rnn', 'transformer', 'hybrid']
    
    for model_type in models:
        try:
            if model_type == 'cnn':
                model = EmotionModelFactory.create_model(model_type, n_mels=128, n_classes=8)
            elif model_type == 'rnn':
                model = EmotionModelFactory.create_model(model_type, input_size=13, n_classes=8)
            elif model_type == 'transformer':
                model = EmotionModelFactory.create_model(model_type, input_size=128, n_classes=8)
            elif model_type == 'hybrid':
                model = EmotionModelFactory.create_model(model_type, n_mels=128, mfcc_size=13, n_classes=8)
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   ✅ {model_type.upper()}: {param_count:,} parameters")
            
        except Exception as e:
            print(f"   ❌ {model_type.upper()}: {e}")
            return False
    
    return True

def test_training_setup():
    """Test training pipeline setup."""
    print("\n🏋️ Testing Training Setup...")
    
    try:
        trainer = EmotionTrainer(
            model_type='cnn',
            data_dir='data',
            checkpoint_dir='checkpoints',
            log_dir='logs'
        )
        
        print(f"   ✅ Trainer initialized")
        print(f"   Device: {trainer.device}")
        print(f"   Checkpoint dir: {trainer.checkpoint_dir}")
        print(f"   Log dir: {trainer.log_dir}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training setup failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Voice Emotion Detection System - Quick Test")
    print("=" * 60)
    
    tests = [
        ("Data Availability", test_data_availability),
        ("Audio Processing", test_audio_processing),
        ("Model Architectures", test_model_architectures),
        ("Training Setup", test_training_setup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status}")
        except Exception as e:
            print(f"\n❌ FAIL - {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready for training.")
        print("\n🚀 Next steps:")
        print("   1. Run 'python train.py' for full training")
        print("   2. Or use 'jupyter notebook emotion_detection_notebook.ipynb' for interactive training")
        print("   3. Monitor training with 'tensorboard --logdir=logs'")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
