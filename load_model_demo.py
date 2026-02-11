#!/usr/bin/env python3
"""
Demo script to load the trained emotion detection model correctly.
This script shows how to load the model using the exact configuration from the checkpoint.
"""

import torch
from pathlib import Path
from inference import EmotionPredictor

def main():
    model_path = "checkpoints/cnn_best.pth"
    
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return
    
    print("Loading trained emotion detection model...")
    print("=" * 50)
    
    try:
        # Load the model - it will use the configuration saved in the checkpoint
        predictor = EmotionPredictor(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {predictor.model_type}")
        print(f"Device: {predictor.device}")
        print(f"Number of emotion classes: {len(predictor.emotion_classes)}")
        print(f"Emotion classes: {list(predictor.emotion_classes)}")
        
        # Test with a sample audio file if available
        sample_files = list(Path("data").glob("**/03-01-01-01-01-01-*.wav"))
        if sample_files:
            sample_file = sample_files[0]
            print(f"\nTesting with sample file: {sample_file}")
            
            # Predict emotion
            emotion, confidence, probabilities = predictor.predict(str(sample_file))
            print(f"Predicted emotion: {emotion}")
            print(f"Confidence: {confidence:.2%}")
            print("Top 3 predictions:")
            for i, (emo, prob) in enumerate(probabilities[:3]):
                print(f"  {i+1}. {emo}: {prob:.2%}")
        else:
            print("\nNo sample audio files found for testing.")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're using the correct dataset configuration")
        print("2. The saved model was trained with both RAVDESS and TESS datasets (10 classes)")
        print("3. Your current configuration should match this")

if __name__ == "__main__":
    main()
