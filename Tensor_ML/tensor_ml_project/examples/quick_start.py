"""
Quick Start Guide for Tensor ML Project
This script demonstrates the basic workflow of the project
"""

import sys
import os

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import project modules
try:
    from tensor_ops import TensorEngineer
    from text_classifier import TransformerClassifier, TextDataset, TensorTrainer, create_sample_data
    from data_pipeline import DataPreprocessor, TensorDataPipeline
    from model_evaluator import ModelEvaluator
    from visualization import TensorVisualizer, TensorAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this script from the examples directory")
    print("Current directory:", os.getcwd())
    print("Expected src directory:", src_dir)
    sys.exit(1)


def main():
    """Main quick start workflow"""
    print("=" * 60)
    print("Tensor ML Project - Quick Start Guide")
    print("=" * 60)
    print()
    
    # Step 1: Initialize tensor engineer
    print("Step 1: Initialize Tensor Engineer")
    print("-" * 40)
    engineer = TensorEngineer(device='cpu')
    print("✓ Tensor engineer initialized")
    print()
    
    # Step 2: Create sample data
    print("Step 2: Create Sample Data")
    print("-" * 40)
    texts, labels = create_sample_data()
    print(f"✓ Created {len(texts)} text samples")
    print(f"✓ Classes: {set(labels)}")
    print(f"✓ Sample: '{texts[0]}' -> Label: {labels[0]}")
    print()
    
    # Step 3: Prepare data pipeline
    print("Step 3: Prepare Data Pipeline")
    print("-" * 40)
    preprocessor = DataPreprocessor(device='cpu')
    pipeline = TensorDataPipeline(preprocessor)
    
    # Split data
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, max_length=32)
    val_dataset = TextDataset(val_texts, val_labels, vocab=train_dataset.vocab, max_length=32)
    
    print(f"✓ Vocabulary size: {train_dataset.vocab_size}")
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print()
    
    # Step 4: Create data loaders
    print("Step 4: Create Data Loaders")
    print("-" * 40)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print()
    
    # Step 5: Build model
    print("Step 5: Build Transformer Model")
    print("-" * 40)
    model = TransformerClassifier(
        vocab_size=train_dataset.vocab_size,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        num_classes=2,
        max_length=32
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {param_count:,} parameters")
    print(f"✓ Architecture: 64-dim embeddings, 2 attention heads, 2 layers")
    print()
    
    # Step 6: Train model
    print("Step 6: Train Model")
    print("-" * 40)
    trainer = TensorTrainer(model, learning_rate=0.001, device='cpu')
    print("Training started...")
    history = trainer.train(train_loader, val_loader, epochs=10, patience=3, verbose=True)
    print("✓ Training completed")
    print()
    
    # Step 7: Evaluate model
    print("Step 7: Evaluate Model")
    print("-" * 40)
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
    evaluator = ModelEvaluator(model, device='cpu')
    
    predictions, true_labels, probabilities = evaluator.predict(val_loader)
    metrics = evaluator.compute_metrics(true_labels, predictions, probabilities)
    
    print("✓ Model evaluation completed")
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Step 8: Analyze results
    print("Step 8: Analyze Results")
    print("-" * 40)
    visualizer = TensorVisualizer()
    analyzer = TensorAnalyzer()
    
    # Analyze model embeddings
    sample_text = train_dataset[0][0].unsqueeze(0)
    with torch.no_grad():
        embeddings = model.embedding(sample_text)
    
    embedding_props = analyzer.analyze_tensor_properties(embeddings)
    print("✓ Embedding analysis completed")
    print(f"  Shape: {embedding_props['shape']}")
    print(f"  Mean: {embedding_props['mean']:.4f}")
    print(f"  Std: {embedding_props['std']:.4f}")
    print(f"  Memory: {embedding_props['memory_usage_mb']:.4f} MB")
    print()
    
    # Summary
    print("=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print()
    print("You have successfully:")
    print("  ✓ Set up tensor operations")
    print("  ✓ Processed text data")
    print("  ✓ Built a transformer model")
    print("  ✓ Trained and evaluated the model")
    print("  ✓ Analyzed model performance")
    print()
    print("Next Steps:")
    print("  • Try with your own data")
    print("  • Experiment with model hyperparameters")
    print("  • Explore the full notebook (notebooks/tensor_ml_demo.ipynb)")
    print("  • Check the README.md for detailed documentation")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during quick start: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
