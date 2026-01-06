# HSD-SMR: Hierarchical State Decomposition for Sequential Multimodal Recommendation

This repository contains the implementation for the paper "Behavioral Deviation as Deliberative Signal: Hierarchical State Decomposition for Sequential Multimodal Recommendation".

## Overview

HSD-SMR leverages **behavioral deviation signals**—when users deviate from established patterns by uploading images or writing longer reviews—as predictive features for sequential recommendation. Our key empirical finding is that such deviations correlate with **reduced 5-star ratings** while 1-star ratings remain stable, suggesting deliberative processing rather than sentiment extremity.

The model applies **hierarchical debiasing** to sequence encoder outputs:
1. **Systematic correction**: Group-stratified temporal statistics capturing platform-wide trends
2. **Individual correction**: User-specific deviation signals with learned reliability gates

## Requirements
```
python>=3.8
torch>=2.0
numpy
pandas
tqdm
sentence-transformers
clip (openai)
```

Install dependencies:
```bash
pip install torch numpy pandas tqdm sentence-transformers
pip install git+https://github.com/openai/CLIP.git
```

## Data

We use [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset. The experiments are conducted on three categories:
- Toys & Games
- Pet Supplies  
- Sports & Outdoors

## Preprocessing

The preprocessing pipeline transforms raw Amazon reviews into training-ready format:

### Overview

1. **Data Loading & Filtering**
   - Load reviews from HuggingFace datasets
   - Filter events before 2014 (retain ~95% data with cleaner temporal dynamics)
   - Apply 10-core filtering (users and items with ≥10 interactions)

2. **Temporal Processing**
   - Assign monthly time bins (2014-01 to 2023-03, with 2023-Q2+ merged)
   - Compute unix timestamps for recency encoding

3. **User/Item History Construction**
   - Build chronologically ordered interaction sequences per user
   - Construct item interaction histories for context features

4. **Train/Val/Test Split**
   - Leave-last-two-out: last interaction → test, second-to-last → validation

5. **Feature Engineering**
   - **User history features**: previous review count, mean rating, image upload rate, mean text length, time since previous review
   - **Item history features**: previous review count, mean/std rating, image rate, mean text length
   - **Deviation indicators**: unusual image upload, unusual text/title length flags
   - **Missing pattern encoding**: 3-bit pattern for title/text/image presence

6. **Embedding Generation**
   - Text embeddings: Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim)
   - Image embeddings: CLIP (`ViT-B/32`, 512-dim)

7. **Population Statistics (for HSD)**
   - Compute global O^t statistics per time bin: mean/std rating, image rate, mean/std text length
   - Compute group-specific statistics with shrinkage toward global (λ=50)
   - Generate lagged versions for temporal validity

8. **User Grouping**
   - Partition users into 3 groups based on historical image upload rate (train data only):
     - Group 0 (never): image_rate = 0
     - Group 1 (rare): 0 < image_rate ≤ 0.2
     - Group 2 (frequent): image_rate > 0.2

### Output Structure
```
preprocessed_sequential/
├── data/
│   ├── chunk_0000.parquet          # Event data chunks
│   ├── chunk_0000_title_emb.npy    # Title embeddings
│   ├── chunk_0000_text_emb.npy     # Text embeddings
│   └── chunk_0000_image_emb.npy    # Image embeddings
├── mappings.pkl                     # user_id/item_id to index
├── user_histories.pkl               # {user_idx: [global_idx, ...]}
├── split_indices.pkl                # train/val/test global indices
├── train_global_stats.pkl           # Global statistics from training set
├── time_bin_config.json             # Time bin mapping
├── user_groups_train_only.pkl       # User group assignments
├── global_time_stats_lag.pkl        # Lagged global O^t
├── group_time_stats_lag.pkl         # Lagged group O^t with shrinkage
├── user_seen_items.pkl              # For negative sampling
└── chunk_info.json                  # Chunk metadata
```

## Model Architecture

### Components

- **EventEncoder**: Encodes each historical event combining item embedding, gated multimodal fusion (title/text/image), numeric features, and temporal recency
- **CausalTransformerEncoder**: Processes event sequences with left-to-right attention masks
- **TimeContextEncoder**: Encodes population statistics and their temporal changes (Δ)
- **HSDUpdater**: Applies hierarchical debiasing with bounded updates

### HSD Update Mechanism
```
z_u^t = z̃_u^t + η₂·h̄_{g(u)}^t + γ_u^t·η₃·h_u^t
```

where:
- `z̃_u^t`: Base state from Transformer encoder
- `h̄_{g(u)}^t`: Systematic correction from group-level trends  
- `h_u^t`: Individual correction from deviation signals
- `γ_u^t`: Learned reliability gate
- Bounded update constraint: `||Δ|| ≤ α·||z̃||`

### Deviation Vector (16 dimensions)

| Dim | Feature | Description |
|-----|---------|-------------|
| 0 | Image deviation | Current - historical image rate |
| 1 | Text length deviation | Standardized departure from user mean |
| 2 | Title length deviation | Standardized departure from user mean |
| 3 | Rating deviation | Departure from user mean rating |
| 4 | History reliability | min(history_length / 20, 1) |
| 5 | Abs image deviation | |δ₀| |
| 6-8 | Unusual flags | Binary indicators for rare uploaders |
| 9-10 | Rating deltas | User/item rating changes |
| 11-12 | Engagement | log(num_images), log(helpful_votes) |
| 13 | First-time upload | Stricter threshold indicator |
| 14-15 | Time gaps | Log-transformed user/item time gaps |

## Training
```bash
python hsd_smr.py
```

### Training Features

- **Mixed negative sampling**: 60% uniform + 40% popularity-based (α=0.75)
- **In-batch negatives**: Additional negatives from batch
- **Auxiliary tasks**: Rating prediction, modality prediction, content embedding prediction (with warmup)
- **Learning rate**: Linear warmup (3 epochs) + cosine annealing
- **Early stopping**: Patience of 10 epochs on validation Recall@20

## Evaluation

We use **full-sort evaluation** (ranking all items) with the following metrics:
- Recall@K (K=10, 20, 50)
- NDCG@K (K=10, 20, 50)
- MRR

Previously interacted items are excluded from ranking.

## Project Structure
```
├── hsd_smr.py              # Model definition and training
├── preprocessed_sequential/  # Preprocessed data (not included)
├── results_hsd/          # Training outputs
│   ├── best.pt           # Best checkpoint
│   └── test_results.json # Final metrics
└── README.md
```
