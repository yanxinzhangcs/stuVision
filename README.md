Flash-STU for Vision

Flash-STU is a distributed training framework that merges advanced sequence processing modules with vision transformer (ViT) design principles. Originally developed for NLP tasks, the repository has been extended to support computer vision and vision-language tasks through a modified architecture (FlashSTU_ViT). The framework leverages PyTorch and Hugging Face’s Transformers library for efficient multi-modal model training, including support for mixed precision and distributed training across multiple GPUs.

Overview

Flash-STU integrates two main components:
	•	FlashSTU Modules (STU/Attention): Innovative modules originally designed for long-range sequence modeling in NLP.
	•	Vision Transformer (ViT) Design: A patch-based image embedding approach adapted to work with FlashSTU modules. An extra classification token and position embeddings are added before passing the data through a Transformer encoder that alternates between STU and Attention layers.

This design allows for:
	•	Efficient handling of long sequences for NLP.
	•	Adaptation to computer vision tasks by treating image patches as tokens.
	•	Potential extension to multi-modal tasks (vision-language), where both text and image inputs are fused in a single framework.

Features
	•	Distributed Training: Built-in support for distributed data parallel training using NCCL backend.
	•	Mixed Precision Support: Runs in bfloat16 or other specified data types using PyTorch’s AMP and custom type handling.
	•	Modular Design: Easy to swap between STU and conventional attention layers.
	•	Configurable via Python: Hyperparameters and training settings are defined in a Python configuration file (flash_stu/config.py), making it simple to adjust settings without JSON files.
	•	Flexible Applications: Primarily designed for NLP tasks but with extensions for image classification (ViT-style) and potential vision-language fusion.