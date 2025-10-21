#!/usr/bin/env python3
"""
Training scripts and utilities for Chinese-LiPS Lipreading
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from lipreading_model import (
    TrainingConfig, ChineseTokenizer, ChineseLiPSDataset, ProcessedLiPSDataset,
    LipreadingModel, ChineseLiPSTrainer, collate_fn,
    build_tokenizer, calculate_cer, evaluate_model
)
import torch
from torch.utils.data import DataLoader
import logging


# ========================================================================================
# TRAIN SCRIPT
# ========================================================================================

def train(args):
    """Main training function"""
    
    # Load config
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()
    
    # Override config with command line arguments
    if args.data_root:
        config.data_root = args.data_root
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Save config
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    config_path = os.path.join(config.checkpoint_dir, "config.yaml")
    config.save(config_path)
    print(f"Saved config to {config_path}")
    
    # Build or load tokenizer
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = ChineseTokenizer.load(tokenizer_path)
    else:
        print("Building tokenizer from training data...")
        tokenizer = build_tokenizer(config.data_root, config)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ProcessedLiPSDataset(
        config.data_root,
        config.train_dir,
        "meta_train.csv",
        config,
        tokenizer
    )
    val_dataset = ProcessedLiPSDataset(
        config.data_root,
        config.val_dir,
        "meta_valid.csv",
        config,
        tokenizer
    )

    print(len(train_dataset))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    print("Creating model...")
    model = LipreadingModel(tokenizer.vocab_size, config)
    
    # Create trainer
    trainer = ChineseLiPSTrainer(model, config, tokenizer)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)


# ========================================================================================
# TEST SCRIPT
# ========================================================================================

def test(args):
    """Test model on test set"""
    
    # Load config
    checkpoint_dir = args.checkpoint_dir or "./checkpoints"
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = TrainingConfig.load(config_path)
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    tokenizer = ChineseTokenizer.load(tokenizer_path)
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = ChineseLiPSDataset(
        args.data_root or config.data_root, 
        config.test_dir, 
        config, 
        tokenizer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = LipreadingModel(tokenizer.vocab_size, config)
    
    # Create trainer
    trainer = ChineseLiPSTrainer(model, config, tokenizer)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    
    # Evaluate
    avg_cer = evaluate_model(trainer, test_loader)
    print(f"\nTest Set Results:")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average Accuracy: {(1 - avg_cer) * 100:.2f}%")


# ========================================================================================
# INFERENCE SCRIPT
# ========================================================================================

def inference(args):
    """Run inference on a single video"""
    
    # Load config
    checkpoint_dir = args.checkpoint_dir or "./checkpoints"
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = TrainingConfig.load(config_path)
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    tokenizer = ChineseTokenizer.load(tokenizer_path)
    
    # Create model
    model = LipreadingModel(tokenizer.vocab_size, config)
    
    # Create trainer
    trainer = ChineseLiPSTrainer(model, config, tokenizer)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    
    # Load video
    print(f"Loading video: {args.video_path}")
    
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(args.video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / config.fps))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (config.frame_width, config.frame_height))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        frame_idx += 1
        
        if len(frames) >= config.max_frames:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames loaded from video")
    
    # Pad to max_frames
    if len(frames) < config.max_frames:
        pad_size = config.max_frames - len(frames)
        frames.extend([np.zeros_like(frames[0])] * pad_size)
    
    # Convert to tensor
    frames = np.stack(frames, axis=0)
    video = torch.from_numpy(frames).unsqueeze(0)  # Add channel dimension
    
    # Predict
    print("Running inference...")
    predicted_text = trainer.predict(video)
    
    print(f"\nPredicted text: {predicted_text}")
    
    # If ground truth is provided
    if args.ground_truth:
        cer = calculate_cer(predicted_text, args.ground_truth)
        print(f"Ground truth: {args.ground_truth}")
        print(f"CER: {cer:.4f}")


# ========================================================================================
# DATASET STATISTICS
# ========================================================================================

def dataset_stats(args):
    """Print dataset statistics"""
    
    data_root = Path(args.data_root)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        
        if not split_dir.exists():
            print(f"{split} directory not found")
            continue
        
        print(f"\n{split.upper()} SPLIT:")
        print("=" * 50)
        
        num_samples = 0
        num_speakers = 0
        total_duration = 0
        text_lengths = []
        
        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            num_speakers += 1
            face_dir = speaker_dir / "FACE"
            wav_dir = speaker_dir / "WAV"
            
            if not face_dir.exists() or not wav_dir.exists():
                continue
            
            for video_file in face_dir.glob("*_FACE.mp4"):
                base_name = video_file.stem.replace("_FACE", "")
                json_file = wav_dir / f"{base_name}.json"
                
                if json_file.exists():
                    num_samples += 1
                    
                    # Get video duration
                    import cv2
                    cap = cv2.VideoCapture(str(video_file))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    total_duration += duration
                    cap.release()
                    
                    # Get text length
                    import json
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        text = data.get('text', data.get('transcript', ''))
                        text_lengths.append(len(text))
        
        print(f"Number of speakers: {num_speakers}")
        print(f"Number of samples: {num_samples}")
        print(f"Total duration: {total_duration / 3600:.2f} hours")
        print(f"Average duration: {total_duration / num_samples:.2f} seconds")
        
        if text_lengths:
            import numpy as np
            print(f"Text length stats:")
            print(f"  Min: {min(text_lengths)}")
            print(f"  Max: {max(text_lengths)}")
            print(f"  Mean: {np.mean(text_lengths):.2f}")
            print(f"  Median: {np.median(text_lengths):.2f}")


# ========================================================================================
# MAIN ENTRY POINT
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(description="Chinese-LiPS Lipreading Training")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, help='Config file path')
    train_parser.add_argument('--data_root', type=str, help='Dataset root directory')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    train_parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Test command
        # Test command
    test_parser = subparsers.add_parser('test', help='Test model')
    test_parser.add_argument('--data_root', type=str, help='Dataset root directory')
    test_parser.add_argument('--batch_size', type=int, help='Batch size')
    test_parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    test_parser.add_argument('--checkpoint', type=str, help='Checkpoint file path')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on a video')
    inference_parser.add_argument('--video_path', type=str, required=True, help='Path to video file')
    inference_parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    inference_parser.add_argument('--checkpoint', type=str, help='Checkpoint file path')
    inference_parser.add_argument('--ground_truth', type=str, help='Ground truth text for comparison')
    
    # Dataset stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--data_root', type=str, required=True, help='Dataset root directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'inference':
        inference(args)
    elif args.command == 'stats':
        dataset_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()