# Sentence Transformer & Multi-Task Learning

## Overview
This repository demonstrates:
1. **Sentence Transformer** implementation and pretraining.  
2. **Multi-Task Learning** expansion for joint Intent Classification and Named Entity Recognition (NER).  
3. **Training Considerations** and transfer learning strategies.  
4. Complete **training loops** with best practices (AMP, schedulers, early stopping).

## Project Structure
```
.
├── tokenizer.py                  # BPE tokenizer implementation
├── bpe_merged.json               # Trained BPE merge map
├── sent_transformer.py           # SentenceTransformer backbone module
├── Pretrain_SentenceTransformer.ipynb   # Notebook for backbone pretraining
├── Tokenizer-walkthrough.ipynb   # Notebook showing BPE training & usage
├── Train_MultiTask_Model.ipynb   # Notebook for multi-task training (intent + NER)
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and instructions
```

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Notebooks

### 1. Tokenizer Training & Usage
- **Tokenizer-walkthrough.ipynb**  
  - Trains a BPE tokenizer on Wikipedia text.  
  - Explains regex-based splitting, pair statistics, merge loops, and encode/decode functions.

### 2. Sentence Transformer Pretraining
- **Pretrain_SentenceTransformer.ipynb**  
  - Implements a minimalist transformer: embeddings, fixed sinusoidal PE, multi-layer Transformer encoder, projection head.  
  - Pretrains via triplet loss on STS-B and QQP datasets.  
  - Demonstrates AMP, gradient clipping, cosine-annealing scheduler, and early stopping.

### 3. Multi-Task Learning Expansion
- **Train_MultiTask_Model.ipynb**  
  - Loads MASSIVE dataset (utterance, intent, NER tags).  
  - BPE tokenization and label alignment across subword tokens.  
  - Weighted sampling for class imbalance.  
  - Architecture: pretrained transformer backbone + intent head +  NER head.  
  - Training best practices: discriminative learning rates, AMP, scheduler, early stopping.

## Key Decisions & Task Mapping

### Task 1: Sentence Transformer Implementation
- **Choice**: Lightweight Transformer (4 layers, 8 heads, 64‑dim hidden).  
- **Pooling**: CLS token + projection head → fixed-length embedding.  
- **Loss**: TripletMarginWithDistanceLoss with cosine distance.

### Task 2: Multi-Task Learning Expansion
- **Intent**: Single classification head on CLS embedding.  
- **NER**: Sequence tagging head via BiLSTM + MLP.  
- **Architecture Change**: Combined two heads sharing the same encoder output.

### Task 3: Training Considerations
- **Frozen Entire Network**: Useful for feature extraction; only heads trained.  
- **Frozen Backbone**: Fine-tune heads quickly when backbone already strong.  
- **Frozen One Head**: Freeze easier task head to let other task-specific gradients dominate.  
- **Transfer Learning**:  
  - Start from pre-trained transformer (e.g., BERT / custom triplet‑trained).  
  - Freeze lower layers; fine-tune top layers + heads with discriminative LRs.

### Task 4: Training Loop Implementation
- **Data Handling**: Collate variable-length BPE token sequences and aligned labels.  
- **Optimization**:  
  - CrossEntropyLoss for both tasks (with class weighting for rare NER tags).  
  - Sum of intent + NER loss.  
- **Metrics & Logging**: Track separate intent accuracy and NER token‑level loss/accuracy.

## Running the Pipeline

1. **Tokenizer**  
   ```bash
   jupyter notebook Tokenizer-walkthrough.ipynb
   ```
2. **Backbone Pretraining**  
   ```bash
   jupyter notebook Pretrain_SentenceTransformer.ipynb
   ```
3. **Multi-Task Training**  
   ```bash
   jupyter notebook Train_MultiTask_Model.ipynb
   ```

## Requirements
See [`requirements.txt`](requirements.txt) for exact package versions.
