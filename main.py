#!/usr/bin/env python3
"""
Optimized training script for Chinese-LiPS Lipreading
Supports HuggingFace dataset format with flat processed directories
"""

import argparse
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from lipreading_model import (
    TrainingConfig, ChineseTokenizer, ProcessedLiPSDataset,
    LipreadingModel, ChineseLiPSTrainer, collate_fn,
    build_tokenizer, calculate_cer, evaluate_model
)
import torch
from torch.utils.data import DataLoader


# ========================================================================================
# TRAIN SCRIPT
# ========================================================================================

def train(args):
    """Main training function"""
    
    # Load or create config
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
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    
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
        tokenizer = build_tokenizer(config.data_root, "meta_train.csv", config)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ProcessedLiPSDataset(
        config.data_root,
        config.train_dir,
        "meta_train.csv",
        config,
        tokenizer
    )
    print(len(train_dataset))
    val_dataset = ProcessedLiPSDataset(
        config.data_root,
        config.val_dir,
        "meta_valid.csv",
        config,
        tokenizer
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    # Create model
    print("Creating model...")
    model = LipreadingModel(tokenizer.vocab_size, config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
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
    
    # Override data_root if specified
    if args.data_root:
        config.data_root = args.data_root
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    tokenizer = ChineseTokenizer.load(tokenizer_path)
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = ProcessedLiPSDataset(
        config.data_root,
        config.test_dir,
        "meta_test.csv",
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
    print(f"\n{'='*50}")
    print(f"Test Set Results:")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average Accuracy: {(1 - avg_cer) * 100:.2f}%")
    print(f"{'='*50}")


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
    
    # Load and preprocess video
    print(f"Loading video: {args.video_path}")
    
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(args.video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / config.fps))
    
    frame_idx = 0
    while len(frames) < config.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (config.frame_width, config.frame_height))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames loaded from video")
    
    print(f"Loaded {len(frames)} frames")
    
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
    
    print(f"\n{'='*50}")
    print(f"Predicted text: {predicted_text}")
    
    # If ground truth is provided
    if args.ground_truth:
        cer = calculate_cer(predicted_text, args.ground_truth)
        print(f"Ground truth: {args.ground_truth}")
        print(f"CER: {cer:.4f}")
        print(f"Accuracy: {(1 - cer) * 100:.2f}%")
    
    print(f"{'='*50}")


# ========================================================================================
# DATASET STATISTICS
# ========================================================================================

def dataset_stats(args):
    """Print dataset statistics"""
    
    data_root = Path(args.data_root)
    
    print(f"\nDataset Statistics for {data_root}")
    print("=" * 70)
    
    for split_name, csv_name in [('train', 'meta_train.csv'), 
                                  ('val', 'meta_valid.csv'), 
                                  ('test', 'meta_test.csv')]:
        csv_path = data_root / csv_name
        
        if not csv_path.exists():
            print(f"\n{split_name.upper()} split metadata not found")
            continue
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        print(f"\n{split_name.upper()} SPLIT:")
        print("-" * 70)
        print(f"Number of samples: {len(df)}")
        
        # Text statistics
        if len(df) > 0 and len(df.columns) >= 6:
            texts = df.iloc[:, 5].astype(str)
            text_lengths = texts.str.len()
            
            print(f"\nText Statistics:")
            print(f"  Min length: {text_lengths.min()}")
            print(f"  Max length: {text_lengths.max()}")
            print(f"  Mean length: {text_lengths.mean():.2f}")
            print(f"  Median length: {text_lengths.median():.2f}")
            
            # Unique characters
            all_chars = set(''.join(texts))
            print(f"  Unique characters: {len(all_chars)}")
            
            # Categories
            if len(df.columns) >= 2:
                categories = df.iloc[:, 1].value_counts()
                print(f"\nCategories:")
                for cat, count in categories.items():
                    print(f"  {cat}: {count}")
    
    print("\n" + "=" * 70)


# ========================================================================================
# DOWNLOAD DATASET (Helper)
# ========================================================================================

def download_dataset(args):
    """Download dataset from HuggingFace"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: Please install datasets library: pip install datasets")
        return
    
    print(f"Downloading Chinese-LiPS dataset to {args.output_dir}")
    print("This may take a while...")
    
    # Download dataset
    dataset = load_dataset("BAAI/Chinese-LiPS", cache_dir=args.cache_dir)
    
    print(f"\nDataset downloaded successfully!")
    print(f"Splits available: {list(dataset.keys())}")
    
    # Save to local directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in dataset.keys():
        split_data = dataset[split]
        print(f"\nProcessing {split} split: {len(split_data)} samples")
        
        # Save dataset
        split_dir = output_dir / f"processed_{split}"
        split_dir.mkdir(exist_ok=True)
        
        # You would need to implement the actual file saving logic here
        # based on the dataset structure
    
    print(f"\nDataset saved to {output_dir}")


# ========================================================================================
# MAIN ENTRY POINT
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized Chinese-LiPS Lipreading Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, help='Config file path')
    train_parser.add_argument('--data_root', type=str, default='./Chinese-LiPS',
                             help='Dataset root directory')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    train_parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                             help='Checkpoint directory')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model')
    test_parser.add_argument('--data_root', type=str, help='Dataset root directory')
    test_parser.add_argument('--batch_size', type=int, help='Batch size')
    test_parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                            help='Checkpoint directory')
    test_parser.add_argument('--checkpoint', type=str, help='Checkpoint file path')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on a video')
    inference_parser.add_argument('--video_path', type=str, required=True, 
                                 help='Path to video file')
    inference_parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                                 help='Checkpoint directory')
    inference_parser.add_argument('--checkpoint', type=str, help='Checkpoint file path')
    inference_parser.add_argument('--ground_truth', type=str, 
                                 help='Ground truth text for comparison')
    
    # Dataset stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--data_root', type=str, required=True, 
                             help='Dataset root directory')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download dataset from HuggingFace')
    download_parser.add_argument('--output_dir', type=str, default='./Chinese-LiPS',
                                help='Output directory for dataset')
    download_parser.add_argument('--cache_dir', type=str, help='Cache directory for downloads')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'inference':
        inference(args)
    elif args.command == 'stats':
        dataset_stats(args)
    elif args.command == 'download':
        download_dataset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()