import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional, List, Dict
import cv2
import json
import os
from pathlib import Path
from dataclasses import dataclass
import yaml
from tqdm import tqdm
import logging
import pandas as pd

# ========================================================================================
# CONFIGURATION
# ========================================================================================

@dataclass
class TrainingConfig:
    """Training configuration for Chinese-LiPS dataset"""
    
    # Dataset paths
    data_root: str = "./Chinese-LiPS"
    train_dir: str = "processed_train"
    val_dir: str = "processed_val"
    test_dir: str = "processed_test"
    
    # Model architecture
    hidden_dim: int = 256  # Reduced for 12GB VRAM
    temporal_model: str = "lstm"  # 'lstm' or 'transformer'
    num_lstm_layers: int = 2  # Reduced for memory
    num_transformer_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.3
    
    # Video processing
    max_frames: int = 75
    frame_height: int = 96
    frame_width: int = 96
    fps: int = 25  # Target FPS for resampling
    
    # Training hyperparameters
    batch_size: int = 4  # Small batch for 12GB VRAM
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    gradient_clip: float = 5.0
    mixed_precision: bool = True  # Enable AMP for memory efficiency
    
    # Optimization
    warmup_epochs: int = 5
    scheduler_type: str = "cosine"  # 'cosine' or 'step'
    step_size: int = 10
    gamma: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5
    keep_last_n: int = 3
    
    # Logging
    log_dir: str = "./logs"
    log_every: int = 50
    
    # Chinese character tokenization
    use_char_level: bool = True
    max_text_length: int = 200
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def save(self, path: str):
        """Save config to YAML file"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, allow_unicode=True)
    
    @classmethod
    def load(cls, path: str):
        """Load config from YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# ========================================================================================
# CHINESE TEXT PROCESSING
# ========================================================================================

class ChineseTokenizer:
    """Tokenizer for Chinese characters"""
    
    def __init__(self):
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_size = 4
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from text corpus"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Sort for consistency
        chars = sorted(list(chars))
        
        for char in chars:
            if char not in self.char2idx:
                idx = self.vocab_size
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                self.vocab_size += 1
        
        logging.info(f"Built vocabulary with {self.vocab_size} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token indices"""
        tokens = [self.char2idx['<sos>']] if add_special_tokens else []
        tokens.extend([self.char2idx.get(c, self.char2idx['<unk>']) for c in text])
        if add_special_tokens:
            tokens.append(self.char2idx['<eos>'])
        return tokens
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token indices to text"""
        chars = []
        for idx in indices:
            if skip_special_tokens and idx <= 2:  # Skip <pad>, <sos>, <eos>
                continue
            if idx == 2:  # Stop at <eos>
                break
            chars.append(self.idx2char.get(idx, '<unk>'))
        return ''.join(chars)
    
    def save(self, path: str):
        """Save tokenizer"""
        data = {
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer"""
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer.char2idx = data['char2idx']
        tokenizer.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer


# ========================================================================================
# DATASET FOR CHINESE-LIPS
# ========================================================================================

class ChineseLiPSDataset(Dataset):
    """Dataset for Chinese-LiPS lipreading"""
    
    def __init__(self, data_root: str, split: str, config: TrainingConfig, 
                 tokenizer: ChineseTokenizer, transform=None):
        """
        Args:
            data_root: Root directory of Chinese-LiPS dataset
            split: 'train', 'val', or 'test'
            config: Training configuration
            tokenizer: Chinese tokenizer
            transform: Optional video transforms
        """
        self.data_root = Path(data_root)
        self.split = split
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load data samples
        self.samples = self._load_samples()
        logging.info(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the split directory"""
        samples = []
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Iterate through speaker directories
        for speaker_dir in sorted(split_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            
            face_dir = speaker_dir / "FACE"
            wav_dir = speaker_dir / "WAV"
            
            if not face_dir.exists() or not wav_dir.exists():
                continue
            
            # Get all video files
            for video_file in sorted(face_dir.glob("*_FACE.mp4")):
                # Get corresponding JSON annotation
                base_name = video_file.stem.replace("_FACE", "")
                json_file = wav_dir / f"{base_name}.json"
                
                if json_file.exists():
                    samples.append({
                        'video_path': str(video_file),
                        'json_path': str(json_file),
                        'speaker_id': speaker_dir.name
                    })
        
        return samples
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate to achieve target FPS
        frame_interval = max(1, int(fps / self.config.fps))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_idx % frame_interval == 0:
                # Convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize to target size
                frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
            
            frame_idx += 1
            
            # Stop if we have enough frames
            if len(frames) >= self.config.max_frames:
                break
        
        cap.release()
        
        if len(frames) == 0:
            logging.warning(f"No frames loaded from {video_path}")
            frames = [np.zeros((self.config.frame_height, self.config.frame_width), dtype=np.float32)]
        
        # Convert to numpy array
        frames = np.stack(frames, axis=0)  # (T, H, W)
        
        # Pad or truncate to max_frames
        T = frames.shape[0]
        if T < self.config.max_frames:
            pad_size = self.config.max_frames - T
            frames = np.concatenate([frames, np.zeros((pad_size, *frames.shape[1:]))], axis=0)
        else:
            frames = frames[:self.config.max_frames]
        
        # Convert to tensor and add channel dimension: (T, H, W) -> (1, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, min(T, self.config.max_frames)
    
    def _load_annotation(self, json_path: str) -> str:
        """Load text annotation from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text (assuming it's in 'text' or 'transcript' field)
        text = data.get('text', data.get('transcript', ''))
        
        return text
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        # Load video
        video, video_length = self._load_video(sample['video_path'])
        
        # Load text
        text = self._load_annotation(sample['json_path'])
        
        # Encode text
        text_tokens = self.tokenizer.encode(text)
        
        # Truncate if necessary
        if len(text_tokens) > self.config.max_text_length:
            text_tokens = text_tokens[:self.config.max_text_length-1] + [self.tokenizer.char2idx['<eos>']]
        
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        text_length = len(text_tokens)
        
        return video, text_tensor, video_length, text_length

class ProcessedLiPSDataset(Dataset):
    """Dataset for the flat 'processed' Chinese-LiPS data."""
    
    def __init__(self, data_root: str, data_dir: str, meta_csv_path: str, config: TrainingConfig, 
                 tokenizer: ChineseTokenizer, transform=None):
        """
        Args:
            data_dir: Directory with the flat .mp4 and .wav files (e.g., './processed_train')
            meta_csv_path: Path to the metadata CSV (e.g., './meta_train.csv')
            config: Training configuration
            tokenizer: Chinese tokenizer
            transform: Optional video transforms
        """
        self.data_root = Path(data_root)
        self.data_dir = Path(data_dir)
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load metadata
        try:
            self.meta_df = pd.read_csv(str(self.data_root / meta_csv_path))
            logging.info(f"Loaded metadata from {meta_csv_path}")
        except Exception as e:
            raise ValueError(f"Could not load metadata CSV: {e}")

        # Find all video files
        self.video_files = sorted((self.data_root/self.data_dir).glob("*.mp4"))
        if not self.video_files:
            raise ValueError(f"No .mp4 files found in {data_dir}")
        
        logging.info(f"Found {len(self.video_files)} video samples in {data_dir}")

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames (same as original class)"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_interval = max(1, int(fps / self.config.fps))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            frame_idx += 1
            if len(frames) >= self.config.max_frames:
                break
        
        cap.release()
        
        if len(frames) == 0:
            logging.warning(f"No frames loaded from {video_path}")
            frames = [np.zeros((self.config.frame_height, self.config.frame_width), dtype=np.float32)]
        
        frames = np.stack(frames, axis=0)
        T = frames.shape[0]
        if T < self.config.max_frames:
            pad_size = self.config.max_frames - T
            frames = np.concatenate([frames, np.zeros((pad_size, *frames.shape[1:]))], axis=0)
        else:
            frames = frames[:self.config.max_frames]
        
        frames = torch.from_numpy(frames).float().unsqueeze(0)
        return frames, min(T, self.config.max_frames)

    def _load_annotation(self, video_path: str) -> str:
        """Load text annotation from the metadata CSV using the filename."""
        # Get the base filename without extension
        base_name = Path(video_path).stem
        
        # Find the row in the DataFrame matching this filename
        # **IMPORTANT**: Adjust 'filename_column' and 'text_column' to match your CSV!
        try:
            # Try common column names
            filename_col = None
            text_col = None
            for col in self.meta_df.columns:
                if 'file' in col.lower() or 'id' in col.lower() or 'path' in col.lower():
                    filename_col = col
                if 'text' in col.lower() or 'transcript' in col.lower() or 'content' in col.lower():
                    text_col = col
            
            if filename_col is None or text_col is None:
                raise ValueError("Could not find filename or text columns in CSV.")

            row = self.meta_df[self.meta_df[filename_col] == base_name]
            if row.empty:
                logging.warning(f"No annotation found for {base_name}")
                return ""
            
            text = row[text_col].iloc[0]
            return str(text)
        except Exception as e:
            logging.error(f"Error loading annotation for {base_name}: {e}")
            return ""

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        video_path = self.video_files[idx]
        
        # Load video
        video, video_length = self._load_video(str(video_path))
        
        # Load text
        text = self._load_annotation(str(video_path))
        
        # Encode text
        text_tokens = self.tokenizer.encode(text)
        if len(text_tokens) > self.config.max_text_length:
            text_tokens = text_tokens[:self.config.max_text_length-1] + [self.tokenizer.char2idx['<eos>']]
        
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        text_length = len(text_tokens)
        
        return video, text_tensor, video_length, text_length


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    videos, texts, video_lengths, text_lengths = zip(*batch)
    
    # Stack videos (all same size due to padding)
    videos = torch.stack(videos, dim=0)  # (B, 1, T, H, W)
    
    # Pad text sequences
    max_text_len = max(text_lengths)
    padded_texts = []
    for text in texts:
        if len(text) < max_text_len:
            padding = torch.zeros(max_text_len - len(text), dtype=torch.long)
            text = torch.cat([text, padding], dim=0)
        padded_texts.append(text)
    
    texts = torch.stack(padded_texts, dim=0)  # (B, L)
    video_lengths = torch.tensor(video_lengths, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    
    return videos, texts, video_lengths, text_lengths


# ========================================================================================
# MODEL COMPONENTS (Optimized for 12GB VRAM)
# ========================================================================================

class SpatioTemporalConv(nn.Module):
    """3D Convolution with optional residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv3d(x)))


class SpatialEncoder(nn.Module):
    """Lightweight 3D CNN for spatial feature extraction"""
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        
        # Frontend: reduce spatial dimensions quickly
        self.frontend = nn.Sequential(
            SpatioTemporalConv(in_channels, 32, kernel_size=(5, 7, 7), 
                              stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            SpatioTemporalConv(32, 64, kernel_size=(3, 3, 3), 
                              stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        
        # Middle: process features
        self.middle = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            SpatioTemporalConv(64, 128, stride=(1, 1, 1)),
            SpatioTemporalConv(128, 256, stride=(1, 1, 1)),
        )
        
        # Backend: global pooling
        self.backend = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Projection
        self.fc = nn.Linear(256, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            features: (B, T, hidden_dim)
        """
        x = self.frontend(x)
        x = self.middle(x)
        x = self.backend(x)
        
        # Reshape: (B, 256, T, 1, 1) -> (B, T, 256)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        # Project
        x = self.fc(x)
        
        return x


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM for temporal modeling"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        output, _ = self.lstm(x)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return output


class AttentionDecoder(nn.Module):
    """Attention-based decoder"""
    
    def __init__(self, hidden_dim: int, vocab_size: int, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention
        self.attention_W = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_v = nn.Linear(hidden_dim, 1)
        
        # Output
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, encoder_outputs: torch.Tensor, targets: Optional[torch.Tensor] = None,
                max_length: int = 200, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        B = encoder_outputs.size(0)
        
        if targets is not None:
            return self._forward_train(encoder_outputs, targets, teacher_forcing_ratio)
        else:
            return self._forward_inference(encoder_outputs, max_length)
    
    def _compute_attention(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, 1, hidden_dim) or (B, hidden_dim)
            encoder_outputs: (B, T, hidden_dim)
        Returns:
            context: (B, 1, hidden_dim)
        """
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        
        B, T, H = encoder_outputs.shape
        
        # Compute attention scores
        query = self.attention_W(hidden)  # (B, 1, H)
        keys = self.attention_U(encoder_outputs)  # (B, T, H)
        
        # Expand query for all time steps
        query = query.expand(-1, T, -1)  # (B, T, H)
        
        # Compute scores
        scores = torch.tanh(query + keys)  # (B, T, H)
        scores = self.attention_v(scores).squeeze(-1)  # (B, T)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=1)  # (B, T)
        
        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B, 1, H)
        
        return context
    
    def _forward_train(self, encoder_outputs: torch.Tensor, targets: torch.Tensor,
                      teacher_forcing_ratio: float) -> torch.Tensor:
        B, L = targets.shape
        
        embedded = self.dropout(self.embedding(targets))  # (B, L, H)
        
        outputs = []
        hidden = None
        
        use_teacher_forcing = np.random.random() < teacher_forcing_ratio
        
        for t in range(L - 1):  # Exclude last token
            # Get query for attention
            if hidden is None:
                query = torch.zeros(B, self.hidden_dim, device=encoder_outputs.device)
            else:
                query = hidden[0][-1]  # (B, H)
            
            # Compute attention and context
            context = self._compute_attention(query, encoder_outputs)  # (B, 1, H)
            
            # Get input embedding
            if use_teacher_forcing:
                input_emb = embedded[:, t:t+1, :]  # (B, 1, H)
            else:
                if t == 0:
                    input_emb = embedded[:, 0:1, :]
                else:
                    prev_output = outputs[-1]  # (B, 1, vocab_size)
                    prev_token = prev_output.argmax(dim=-1)  # (B, 1)
                    input_emb = self.dropout(self.embedding(prev_token))  # (B, 1, H)
            
            # Concatenate input with context
            lstm_input = torch.cat([input_emb, context], dim=2)  # (B, 1, H*2)
            
            # LSTM step
            output, hidden = self.lstm(lstm_input, hidden)
            
            # Project to vocabulary
            output = self.fc(output)  # (B, 1, vocab_size)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)  # (B, L-1, vocab_size)
        return outputs
    
    def _forward_inference(self, encoder_outputs: torch.Tensor, max_length: int) -> torch.Tensor:
        B = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Start with <sos> token (index 1)
        input_token = torch.ones(B, 1, dtype=torch.long, device=device)
        
        outputs = []
        hidden = None
        
        for t in range(max_length):
            # Embed input
            embedded = self.embedding(input_token)  # (B, 1, H)
            
            # Compute attention
            if hidden is None:
                query = torch.zeros(B, self.hidden_dim, device=device)
            else:
                query = hidden[0][-1]
            
            context = self._compute_attention(query, encoder_outputs)
            
            # LSTM step
            lstm_input = torch.cat([embedded, context], dim=2)
            output, hidden = self.lstm(lstm_input, hidden)
            
            # Project to vocabulary
            output = self.fc(output)  # (B, 1, vocab_size)
            outputs.append(output)
            
            # Next input is predicted token
            input_token = output.argmax(dim=-1)
            
            # Check for <eos> token
            if (input_token == 2).all():
                break
        
        outputs = torch.cat(outputs, dim=1)
        return outputs


class LipreadingModel(nn.Module):
    """Complete lipreading model for Chinese-LiPS"""
    
    def __init__(self, vocab_size: int, config: TrainingConfig):
        super().__init__()
        
        self.spatial_encoder = SpatialEncoder(in_channels=1, hidden_dim=config.hidden_dim)
        
        if config.temporal_model == 'lstm':
            self.temporal_encoder = BiLSTMEncoder(
                config.hidden_dim, config.hidden_dim, 
                config.num_lstm_layers, config.dropout
            )
        else:
            raise ValueError(f"Temporal model {config.temporal_model} not supported in optimized version")
        
        self.decoder = AttentionDecoder(config.hidden_dim, vocab_size, dropout=config.dropout)
        
    def forward(self, videos: torch.Tensor, targets: Optional[torch.Tensor] = None,
                video_lengths: Optional[torch.Tensor] = None, 
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        # Extract spatial features
        spatial_features = self.spatial_encoder(videos)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(spatial_features, video_lengths)
        
        # Decode
        outputs = self.decoder(temporal_features, targets, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return outputs


# ========================================================================================
# TRAINING LOOP
# ========================================================================================

class ChineseLiPSTrainer:
    """Trainer for Chinese-LiPS dataset with mixed precision support"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, tokenizer: ChineseTokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs - self.config.warmup_epochs
            )
        elif self.config.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Gradually reduce teacher forcing
        teacher_forcing_ratio = max(0.5, 1.0 - self.epoch * 0.05)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (videos, texts, video_lengths, text_lengths) in enumerate(pbar):
            videos = videos.to(self.device)
            texts = texts.to(self.device)
            video_lengths = video_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(videos, texts, video_lengths, teacher_forcing_ratio)
                    
                    # Compute loss
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    targets = texts[:, 1:].reshape(-1)  # Shift targets
                    
                    loss = self.criterion(outputs, targets)
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(videos, texts, video_lengths, teacher_forcing_ratio=1.0)
                
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = texts[:, 1:].reshape(-1)
                
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_epoch_{self.epoch}_loss_{val_loss:.4f}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model with val_loss={val_loss:.4f}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoints = sorted(
            [f for f in os.listdir(self.config.checkpoint_dir) if f.startswith("checkpoint_epoch_")],
            key=lambda x: os.path.getctime(os.path.join(self.config.checkpoint_dir, x))
        )
        
        while len(checkpoints) > self.config.keep_last_n:
            old_checkpoint = checkpoints.pop(0)
            os.remove(os.path.join(self.config.checkpoint_dir, old_checkpoint))
            logging.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        logging.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        logging.info("Starting training...")
        logging.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logging.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            logging.info(f"Epoch {epoch}: val_loss={val_loss:.4f}")
            
            # Update learning rate
            if self.scheduler and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(val_loss)
        
        logging.info("Training completed!")
    
    @torch.no_grad()
    def predict(self, video: torch.Tensor, max_length: int = 200) -> str:
        """Predict text from video"""
        self.model.eval()
        
        if video.dim() == 4:  # (C, T, H, W)
            video = video.unsqueeze(0)  # Add batch dimension
        
        video = video.to(self.device)
        
        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(video, targets=None)
        else:
            outputs = self.model(video, targets=None)
        
        # Get predictions
        predicted_ids = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()
        
        # Decode
        text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
        
        return text
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for videos, texts, video_lengths, text_lengths in tqdm(val_loader, desc="Validation"):
            videos = videos.to(self.device)
            texts = texts.to(self.device)
            video_lengths = video_lengths.to(self.device)
            
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(videos, texts, video_lengths, teacher_forcing_ratio=1.0)
                    
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    targets = texts[:, 1:].reshape(-1)
                    
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(videos, texts, video_lengths, teacher_forcing_ratio=1.0)
                
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = texts[:, 1:].reshape(-1)
                
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches


# ========================================================================================
# MAIN TRAINING SCRIPT
# ========================================================================================

def build_tokenizer(data_root: str, config: TrainingConfig) -> ChineseTokenizer:
    """Build tokenizer from all training data"""
    tokenizer = ChineseTokenizer()
    
    # Load all training texts
    train_dir = Path(data_root) / config.train_dir
    texts = []
    
    logging.info("Building vocabulary from training data...")
    
    for speaker_dir in train_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
        
        wav_dir = speaker_dir / "WAV"
        if not wav_dir.exists():
            continue
        
        for json_file in wav_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('text', data.get('transcript', ''))
                if text:
                    texts.append(text)
    
    tokenizer.build_vocab(texts)
    
    # Save tokenizer
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer.json")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer.save(tokenizer_path)
    logging.info(f"Saved tokenizer to {tokenizer_path}")
    
    return tokenizer


def main():
    """Main training function"""
    
    # Load or create config
    config = TrainingConfig()
    
    # Save config
    config_path = os.path.join(config.checkpoint_dir, "config.yaml")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    config.save(config_path)
    logging.info(f"Saved config to {config_path}")
    
    # Build tokenizer
    tokenizer = build_tokenizer(config.data_root, config)
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = ChineseLiPSDataset(
        config.data_root, 'train', config, tokenizer
    )
    val_dataset = ChineseLiPSDataset(
        config.data_root, 'val', config, tokenizer
    )
    
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
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    logging.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    logging.info("Creating model...")
    model = LipreadingModel(tokenizer.vocab_size, config)
    
    # Create trainer
    trainer = ChineseLiPSTrainer(model, config, tokenizer)
    
    # Train
    trainer.train(train_loader, val_loader)


# ========================================================================================
# INFERENCE SCRIPT
# ========================================================================================

def inference_example():
    """Example inference script"""
    
    # Load config
    checkpoint_dir = "./checkpoints"
    config = TrainingConfig.load(os.path.join(checkpoint_dir, "config.yaml"))
    
    # Load tokenizer
    tokenizer = ChineseTokenizer.load(os.path.join(checkpoint_dir, "tokenizer.json"))
    
    # Create model
    model = LipreadingModel(tokenizer.vocab_size, config)
    
    # Create trainer (for inference utilities)
    trainer = ChineseLiPSTrainer(model, config, tokenizer)
    
    # Load best checkpoint
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    trainer.load_checkpoint(best_model_path)
    
    # Load a test video (example)
    test_dataset = ChineseLiPSDataset(
        config.data_root, 'test', config, tokenizer
    )
    
    video, text_tokens, video_length, text_length = test_dataset[0]
    
    # Predict
    predicted_text = trainer.predict(video)
    ground_truth = tokenizer.decode(text_tokens.tolist(), skip_special_tokens=True)
    
    print(f"Ground truth: {ground_truth}")
    print(f"Predicted: {predicted_text}")


# ========================================================================================
# EVALUATION METRICS
# ========================================================================================

def calculate_cer(pred: str, target: str) -> float:
    """Calculate Character Error Rate"""
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    
    # Simple edit distance implementation
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n] / n


def evaluate_model(trainer: ChineseLiPSTrainer, test_loader: DataLoader):
    """Evaluate model on test set"""
    logging.info("Evaluating model on test set...")
    
    total_cer = 0
    num_samples = 0
    
    for videos, texts, video_lengths, text_lengths in tqdm(test_loader, desc="Testing"):
        for i in range(videos.size(0)):
            video = videos[i]
            ground_truth = trainer.tokenizer.decode(texts[i].tolist(), skip_special_tokens=True)
            
            predicted_text = trainer.predict(video)
            
            cer = calculate_cer(predicted_text, ground_truth)
            total_cer += cer
            num_samples += 1
            
            if num_samples <= 5:  # Print first 5 examples
                logging.info(f"\nExample {num_samples}:")
                logging.info(f"Ground truth: {ground_truth}")
                logging.info(f"Predicted: {predicted_text}")
                logging.info(f"CER: {cer:.4f}")
    
    avg_cer = total_cer / num_samples
    logging.info(f"\nAverage CER: {avg_cer:.4f}")
    
    return avg_cer