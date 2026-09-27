user@user:~$ python -m venv venv
user@user:~$ source venv/bin/activate
(venv) user@user:~$ cd /media/user/FE75-F816/AI_Developer/tensor_ml_project
(venv) user@user:/media/user/FE75-F816/AI_Developer/tensor_ml_project$ uv run python examples/basic_usage.py
-----------------------------------
==================================================
Tensor ML Project - Basic Usage Test
==================================================

=== Testing Tensor Operations ===
TensorEngineer initialized on device: cpu
✓ Attention mask created: torch.Size([2, 10, 10])
✓ Positional encoding created: torch.Size([1, 10, 64])
✓ Tensor normalization works: torch.Size([2, 5, 10])
✓ Tensor operations test passed!

=== Testing Data Pipeline ===
✓ Vocabulary built: 3 words
✓ Text converted to tensor: torch.Size([10])
✓ Numerical data normalized: torch.Size([10, 5])
✓ Data pipeline test passed!

=== Testing Text Classifier ===
✓ Sample data created: 16 samples
✓ Dataset created: 16 samples
✓ Vocabulary size: 16
✓ Model created: 13282 parameters
✓ Forward pass successful: torch.Size([1, 2])
✓ Text classifier test passed!

=== Testing Model Evaluator ===
/media/user/FE75-F816/AI_Developer/tensor_ml_project/examples/../src/text_classifier.py:129: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
✓ Evaluator created
✓ Predictions generated: (16,)
✓ Metrics computed: ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
✓ Model evaluator test passed!

=== Testing Visualization ===
✓ Tensor analysis completed: ['shape', 'size', 'device', 'dtype', 'mean', 'std', 'min', 'max', 'median', 'nnz', 'sparsity', 'memory_usage_mb']
✓ Tensor comparison completed: ['shape_equal', 'mean_diff', 'std_diff', 'mse', 'mae', 'cosine_similarity']
✓ Anomaly detection completed: 2 anomalies
✓ Visualization test passed!

==================================================
Test Summary
==================================================
Tensor Operations    ✓ PASSED
Data Pipeline        ✓ PASSED
Text Classifier      ✓ PASSED
Model Evaluator      ✓ PASSED
Visualization        ✓ PASSED

Total: 5/5 tests passed
🎉 All tests passed! The project is ready to use.

