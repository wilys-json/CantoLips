import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Tuple, Optional, List, Dict
import cv2
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

    # Tensorboard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    log_images: bool = True  # Log sample video frames
    log_text_samples: int = 5  # Number of prediction samples to log
    
    # Dataset paths
    data_root: str = "./Chinese-LiPS"
    train_dir: str = "processed_train"
    val_dir: str = "processed_val"
    test_dir: str = "processed_test"
    
    # Model architecture
    hidden_dim: int = 256
    temporal_model: str = "lstm"
    num_lstm_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.3
    
    # Video processing
    max_frames: int = 75
    frame_height: int = 96
    frame_width: int = 96
    fps: int = 25
    
    # Training hyperparameters
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    gradient_clip: float = 5.0
    mixed_precision: bool = True
    accumulation_steps: int = 4  # Gradient accumulation
    
    # Optimization
    warmup_epochs: int = 5
    scheduler_type: str = "cosine"
    label_smoothing: float = 0.1  # Label smoothing for better generalization
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5
    keep_last_n: int = 3
    
    # Logging
    log_dir: str = "./logs"
    log_every: int = 50
    
    # Text processing
    max_text_length: int = 200
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, allow_unicode=True)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# ========================================================================================
# CHINESE TEXT PROCESSING
# ========================================================================================

class ChineseTokenizer:
    """Optimized tokenizer for Chinese characters"""
    
    def __init__(self):
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_size = 4
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from text corpus"""
        chars = set()
        for text in texts:
            # Remove whitespace and punctuation if needed
            chars.update(text)
        
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
            if skip_special_tokens and idx <= 2:
                continue
            if idx == 2:  # Stop at <eos>
                break
            chars.append(self.idx2char.get(idx, '<unk>'))
        return ''.join(chars)
    
    def save(self, path: str):
        import json
        data = {
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        import json
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer.char2idx = data['char2idx']
        tokenizer.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer


# ========================================================================================
# OPTIMIZED DATASET
# ========================================================================================

class ProcessedLiPSDataset(Dataset):
    """Optimized dataset for processed Chinese-LiPS data"""
    
    def __init__(self, data_root: str, data_dir: str, meta_csv_path: str, 
                 config: TrainingConfig, tokenizer: ChineseTokenizer):
        """
        Args:
            data_root: Root directory (e.g., './Chinese-LiPS')
            data_dir: Subdirectory name (e.g., 'processed_train')
            meta_csv_path: CSV filename (e.g., 'meta_train.csv')
            config: Training configuration
            tokenizer: Chinese tokenizer
        """
        self.data_root = Path(data_root)
        self.config = config
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        
        # Load metadata
        meta_path = self.data_root / meta_csv_path
        if not meta_path.exists():
            raise ValueError(f"Metadata CSV not found: {meta_path}")
        
        self.meta_df = pd.read_csv(meta_path)
        logging.info(f"Loaded {len(self.meta_df)} samples from {meta_csv_path}")
        
        # Parse metadata columns
        self._parse_metadata()
        
    def _parse_metadata(self):
        """Parse metadata to extract video paths and texts"""
        self.samples = []
        
        for idx, row in self.meta_df.iterrows():
            # Assume columns: file_id, category, wav_path, ppt_path, face_path, text
            # Adjust based on actual CSV structure
            if len(row) >= 6:
                file_id = row.iloc[0]
                face_path = '/'.join(Path(row.iloc[4]).parts[1:])  # FACE video path
                text = row.iloc[5]  # Transcript text
                
                # Construct full path
                video_path = self.data_root / self.data_dir / face_path
                if video_path.exists():
                    self.samples.append({
                        'video_path': str(video_path),
                        'text': str(text),
                        'file_id': str(file_id)
                    })
        
        logging.info(f"Found {len(self.samples)} valid video samples")
    
    def _load_video(self, video_path: str) -> Tuple[torch.Tensor, int]:
        """Optimized video loading with caching"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.warning(f"Failed to open video: {video_path}")
            # Return dummy data
            frames = torch.zeros(1, self.config.max_frames, 
                               self.config.frame_height, 
                               self.config.frame_width)
            return frames, 1
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.config.fps))
        
        frame_idx = 0
        while len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Convert to grayscale and resize in one go
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, 
                                 (self.config.frame_width, self.config.frame_height),
                                 interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((self.config.frame_height, self.config.frame_width), 
                              dtype=np.float32)]
        
        # Convert to numpy array and normalize
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
        actual_length = len(frames)
        
        # Pad if needed
        if len(frames) < self.config.max_frames:
            pad_size = self.config.max_frames - len(frames)
            frames = np.concatenate([
                frames, 
                np.zeros((pad_size, self.config.frame_height, self.config.frame_width), 
                        dtype=np.float32)
            ], axis=0)
        
        # Convert to tensor: (T, H, W) -> (1, T, H, W)
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, actual_length
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        # Load video
        video, video_length = self._load_video(sample['video_path'])
        
        # Encode text
        text = sample['text']
        text_tokens = self.tokenizer.encode(text)
        
        # Truncate if necessary
        if len(text_tokens) > self.config.max_text_length:
            text_tokens = text_tokens[:self.config.max_text_length-1] + \
                         [self.tokenizer.char2idx['<eos>']]
        
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        text_length = len(text_tokens)
        
        return video, text_tensor, video_length, text_length


def collate_fn(batch):
    """Optimized collate function"""
    videos, texts, video_lengths, text_lengths = zip(*batch)
    
    # Stack videos
    videos = torch.stack(videos, dim=0)
    
    # Pad text sequences
    max_text_len = max(text_lengths)
    padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
    
    video_lengths = torch.tensor(video_lengths, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    
    return videos, padded_texts, video_lengths, text_lengths


# ========================================================================================
# OPTIMIZED MODEL
# ========================================================================================

class SpatioTemporalConv(nn.Module):
    """Efficient 3D convolution block"""
    
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
    """Optimized 3D CNN backbone"""
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        
        # Efficient feature extraction
        self.frontend = nn.Sequential(
            SpatioTemporalConv(in_channels, 32, kernel_size=(5, 7, 7), 
                              stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        
        self.middle = nn.Sequential(
            SpatioTemporalConv(32, 64, stride=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            SpatioTemporalConv(64, 128, stride=(1, 1, 1)),
        )
        
        self.backend = nn.Sequential(
            SpatioTemporalConv(128, 256, stride=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
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
        x = self.fc(x)
        
        return x


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder"""
    
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
    """Attention-based decoder with label smoothing support"""
    
    def __init__(self, hidden_dim: int, vocab_size: int, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        
        # Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, 
                                               dropout=dropout, batch_first=True)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, encoder_outputs: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        if targets is not None:
            return self._forward_train(encoder_outputs, targets, teacher_forcing_ratio)
        else:
            return self._forward_inference(encoder_outputs)
    
    def _forward_train(self, encoder_outputs: torch.Tensor, 
                       targets: torch.Tensor,
                       teacher_forcing_ratio: float) -> torch.Tensor:
        B, L = targets.shape
        
        # Embed targets
        embedded = self.dropout(self.embedding(targets))
        
        outputs = []
        hidden = None
        
        use_teacher_forcing = np.random.random() < teacher_forcing_ratio
        
        for t in range(L - 1):
            # Get current input
            if use_teacher_forcing or t == 0:
                current_input = embedded[:, t:t+1, :]
            else:
                prev_output = outputs[-1]
                prev_token = prev_output.argmax(dim=-1)
                current_input = self.dropout(self.embedding(prev_token))
            
            # LSTM step
            lstm_out, hidden = self.lstm(current_input, hidden)
            
            # Attention
            attn_out, _ = self.attention(lstm_out, encoder_outputs, encoder_outputs)
            
            # Combine and project
            combined = lstm_out + attn_out
            output = self.fc(combined)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def _forward_inference(self, encoder_outputs: torch.Tensor, 
                          max_length: int = 200) -> torch.Tensor:
        B = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        input_token = torch.ones(B, 1, dtype=torch.long, device=device)
        
        outputs = []
        hidden = None
        
        for t in range(max_length):
            embedded = self.embedding(input_token)
            
            lstm_out, hidden = self.lstm(embedded, hidden)
            attn_out, _ = self.attention(lstm_out, encoder_outputs, encoder_outputs)
            
            combined = lstm_out + attn_out
            output = self.fc(combined)
            outputs.append(output)
            
            input_token = output.argmax(dim=-1)
            
            if (input_token == 2).all():  # <eos>
                break
        
        return torch.cat(outputs, dim=1)


class LipreadingModel(nn.Module):
    """Complete optimized lipreading model"""
    
    def __init__(self, vocab_size: int, config: TrainingConfig):
        super().__init__()
        
        self.spatial_encoder = SpatialEncoder(in_channels=1, hidden_dim=config.hidden_dim)
        self.temporal_encoder = BiLSTMEncoder(
            config.hidden_dim, config.hidden_dim, 
            config.num_lstm_layers, config.dropout
        )
        self.decoder = AttentionDecoder(config.hidden_dim, vocab_size, config.dropout)
        
    def forward(self, videos: torch.Tensor, targets: Optional[torch.Tensor] = None,
                video_lengths: Optional[torch.Tensor] = None, 
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        spatial_features = self.spatial_encoder(videos)
        temporal_features = self.temporal_encoder(spatial_features, video_lengths)
        outputs = self.decoder(temporal_features, targets, teacher_forcing_ratio)
        return outputs


# ========================================================================================
# OPTIMIZED TRAINER
# ========================================================================================

class ChineseLiPSTrainer:
    """Optimized trainer with gradient accumulation and mixed precision"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, tokenizer: ChineseTokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Label smoothing loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, 
            label_smoothing=config.label_smoothing
        )
        
        # Mixed precision scaler - FIXED
        if config.mixed_precision:
            if hasattr(torch.amp, 'GradScaler'):
                # PyTorch 2.0+
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                # PyTorch 1.x (fallback)
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs - config.warmup_epochs
        )
        
        self._setup_logging()
        self._setup_tensorboard()
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer"""
        if self.config.use_tensorboard:
            from datetime import datetime
            log_dir = os.path.join(
                self.config.tensorboard_dir,
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.writer = SummaryWriter(log_dir)
            logging.info(f"TensorBoard logging to {log_dir}")
            
            # Log model graph - FIXED VERSION
            try:
                # Set model to eval mode for graph logging
                self.model.eval()
                
                # Create dummy inputs matching actual model input
                dummy_video = torch.randn(
                    1, 1, self.config.max_frames,
                    self.config.frame_height, self.config.frame_width
                ).to(self.device)
                
                dummy_text = torch.randint(
                    1, self.tokenizer.vocab_size, 
                    (1, 10)
                ).to(self.device)
                
                dummy_video_lengths = torch.tensor([self.config.max_frames]).to(self.device)
                
                # Try to log graph with proper inputs
                with torch.no_grad():
                    self.writer.add_graph(
                        self.model, 
                        (dummy_video, dummy_text, dummy_video_lengths)
                    )
                
                # Set model back to train mode
                self.model.train()
                
            except Exception as e:
                logging.warning(f"Could not log model graph: {e}")
                logging.info("This is not critical - training will continue normally")
        else:
            self.writer = None
    
    def _setup_logging(self):
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
        """Optimized training epoch with TensorBoard logging"""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        teacher_forcing_ratio = max(0.5, 1.0 - self.epoch * 0.05)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (videos, texts, video_lengths, text_lengths) in enumerate(pbar):
            videos = videos.to(self.device)
            texts = texts.to(self.device)
            video_lengths = video_lengths.to(self.device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(videos, texts, video_lengths, teacher_forcing_ratio)
                
                # Compute loss
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets = texts[:, 1:].reshape(-1)
                loss = self.criterion(outputs_flat, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                self.config.gradient_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # ============ TENSORBOARD LOGGING ============
                if self.writer and self.global_step % self.config.log_every == 0:
                    # Log loss
                    self.writer.add_scalar('Train/Loss', 
                                        loss.item() * self.config.accumulation_steps, 
                                        self.global_step)
                    
                    # Log learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                    
                    # Log teacher forcing ratio
                    self.writer.add_scalar('Train/TeacherForcingRatio', 
                                        teacher_forcing_ratio, self.global_step)
                    
                    # Log gradient norm
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    self.writer.add_scalar('Train/GradientNorm', total_norm, self.global_step)
                # =============================================
            
            total_loss += loss.item() * self.config.accumulation_steps
            pbar.set_postfix({'loss': loss.item() * self.config.accumulation_steps})
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validation loop with TensorBoard logging"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_ground_truths = []
        sample_videos = []
        
        for batch_idx, (videos, texts, video_lengths, text_lengths) in enumerate(
            tqdm(val_loader, desc="Validation")
        ):
            videos = videos.to(self.device)
            texts = texts.to(self.device)
            video_lengths = video_lengths.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(videos, texts, video_lengths, teacher_forcing_ratio=1.0)
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets = texts[:, 1:].reshape(-1)
                loss = self.criterion(outputs_flat, targets)
            
            total_loss += loss.item()
            
            # Collect samples for TensorBoard logging
            if self.writer and batch_idx == 0:
                # Get predictions for first few samples
                predicted_ids = outputs.argmax(dim=-1)
                for i in range(min(self.config.log_text_samples, videos.size(0))):
                    pred_text = self.tokenizer.decode(
                        predicted_ids[i].cpu().tolist(), 
                        skip_special_tokens=True
                    )
                    gt_text = self.tokenizer.decode(
                        texts[i].cpu().tolist(), 
                        skip_special_tokens=True
                    )
                    all_predictions.append(pred_text)
                    all_ground_truths.append(gt_text)
                    
                    # Save sample video frames
                    if self.config.log_images:
                        sample_videos.append(videos[i].cpu())
        
        avg_loss = total_loss / len(val_loader)
        
        # ============ TENSORBOARD LOGGING ============
        if self.writer:
            # Log validation loss
            self.writer.add_scalar('Val/Loss', avg_loss, self.epoch)
            
            # Log text predictions
            if all_predictions:
                text_table = "| Ground Truth | Prediction |\n|---|---|\n"
                for gt, pred in zip(all_ground_truths, all_predictions):
                    text_table += f"| {gt} | {pred} |\n"
                self.writer.add_text('Val/Predictions', text_table, self.epoch)
            
            # Log sample video frames
            if self.config.log_images and sample_videos:
                for i, video in enumerate(sample_videos):
                    # Take middle frame: (1, T, H, W) -> (H, W)
                    frame = video[0, video.size(1)//2]  
                    # Normalize to [0, 1] and add channel dim
                    frame = frame.unsqueeze(0)  # (1, H, W)
                    self.writer.add_image(f'Val/Sample_{i}', frame, self.epoch)
            
            # Calculate and log CER for samples
            if all_predictions:
                from lipreading_model import calculate_cer
                cer_scores = [
                    calculate_cer(pred, gt) 
                    for pred, gt in zip(all_predictions, all_ground_truths)
                ]
                avg_cer = sum(cer_scores) / len(cer_scores)
                self.writer.add_scalar('Val/CER', avg_cer, self.epoch)
                self.writer.add_scalar('Val/Accuracy', 1 - avg_cer, self.epoch)
        # =============================================
        
        return avg_loss
    
    def close_tensorboard(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()
            logging.info("TensorBoard writer closed")
    
    def save_checkpoint(self, val_loss: float):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_epoch_{self.epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model with val_loss={val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logging.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        logging.info("Starting training...")
        logging.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Log hyperparameters to TensorBoard
        if self.writer:
            hparams = {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'hidden_dim': self.config.hidden_dim,
                'num_lstm_layers': self.config.num_lstm_layers,
                'dropout': self.config.dropout,
                'max_frames': self.config.max_frames,
            }
            self.writer.add_hparams(hparams, {})
        
        try:
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
    
        finally:
            # Always close TensorBoard writer
            self.close_tensorboard()
    
    @torch.no_grad()
    def predict(self, video: torch.Tensor) -> str:
        """Predict text from video"""
        self.model.eval()
        
        if video.dim() == 4:
            video = video.unsqueeze(0)
        
        video = video.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            outputs = self.model(video, targets=None)
        
        predicted_ids = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()
        text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
        
        return text


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def build_tokenizer(data_root: str, meta_csv: str, config: TrainingConfig) -> ChineseTokenizer:
    """Build tokenizer from metadata"""
    tokenizer = ChineseTokenizer()
    
    meta_path = Path(data_root) / meta_csv
    df = pd.read_csv(meta_path)
    
    texts = df.iloc[:, 5].astype(str).tolist()  # Assuming text is in 6th column
    tokenizer.build_vocab(texts)
    
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer.json")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer.save(tokenizer_path)
    logging.info(f"Saved tokenizer to {tokenizer_path}")
    
    return tokenizer


def calculate_cer(pred: str, target: str) -> float:
    """Calculate Character Error Rate"""
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    
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
    logging.info("Evaluating model...")
    
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
            
            if num_samples <= 5:
                logging.info(f"\nExample {num_samples}:")
                logging.info(f"Ground truth: {ground_truth}")
                logging.info(f"Predicted: {predicted_text}")
                logging.info(f"CER: {cer:.4f}")
    
    avg_cer = total_cer / num_samples
    logging.info(f"\nAverage CER: {avg_cer:.4f}")
    
    return avg_cer