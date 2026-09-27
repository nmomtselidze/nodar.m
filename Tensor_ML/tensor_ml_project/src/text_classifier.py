"""
Production-Ready Text Classification Model using Tensor Engineering
Demonstrates real-world ML implementation for CloudGeometry AI Developer role
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
import re


class TextDataset(Dataset):
    """
    Custom dataset for text classification with tensor-based preprocessing.
    """
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Optional[dict] = None, max_length: int = 128):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            vocab: Optional vocabulary dictionary
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab
            
        self.vocab_size = len(self.vocab)
    
    def _build_vocab(self, texts: List[str], min_freq: int = 2) -> dict:
        """Build vocabulary from text corpus."""
        word_freq = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_freq.update(words)
        
        # Filter by minimum frequency
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1
        
        return vocab
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization using regex."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def _text_to_indices(self, text: str) -> List[int]:
        """Convert text to indices using vocabulary."""
        words = self._tokenize(text)
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return indices
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text_indices = self._text_to_indices(self.texts[idx])
        label = self.labels[idx]
        
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier using advanced tensor operations.
    Demonstrates production ML engineering with tensor optimizations.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 128, 
        num_heads: int = 4, 
        num_layers: int = 2,
        num_classes: int = 2,
        max_length: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize transformer classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = self._create_positional_encoding(max_length, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def _create_positional_encoding(self, max_length: int, embed_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits


class TensorTrainer:
    """
    Training pipeline with tensor engineering optimizations.
    Demonstrates production ML training practices.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Initialize trainer with optimizer and loss function.
        
        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def create_attention_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """Create padding mask for transformer."""
        return (batch == 0).to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(self.device), labels.to(self.device)
            mask = self.create_attention_mask(texts)
            
            self.optimizer.zero_grad()
            outputs = self.model(texts, mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                mask = self.create_attention_mask(texts)
                
                outputs = self.model(texts, mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 3
    ) -> dict:
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training on {self.device}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }


def create_sample_data() -> Tuple[List[str], List[int]]:
    """Create sample text data for demonstration."""
    texts = [
        "This is a great product I love it",
        "Amazing service and fast delivery",
        "Excellent quality and good price",
        "Terrible experience do not buy",
        "Worst product ever made",
        "Poor quality and bad service",
        "Outstanding performance and reliability",
        "Fantastic features and design",
        "Disappointing quality and support",
        "Horrible experience waste of money",
        "Highly recommend this product",
        "Best purchase I ever made",
        "Avoid this product at all costs",
        "Complete waste of time and money",
        "Superb quality and customer service",
        "Impressive features and performance",
    ]
    
    labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  # 1=positive, 0=negative
    
    return texts, labels


if __name__ == "__main__":
    print("=== Text Classification with Tensor Engineering ===\n")
    
    # Create sample data
    texts, labels = create_sample_data()
    print(f"Created {len(texts)} sample texts with {len(set(labels))} classes\n")
    
    # Split data
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, max_length=32)
    val_dataset = TextDataset(val_texts, val_labels, vocab=train_dataset.vocab, max_length=32)
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Create model
    model = TransformerClassifier(
        vocab_size=train_dataset.vocab_size,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        num_classes=2,
        max_length=32
    )
    
    # Train model
    trainer = TensorTrainer(model, learning_rate=0.001, device='cpu')
    history = trainer.train(train_loader, val_loader, epochs=20, patience=5)
    
    print("\n=== Training Complete ===")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.2f}%")
