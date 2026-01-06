import os
import json
import math
import time
import random
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    DATA_DIR: str = "./preprocessed_sequential"
    RESULTS_DIR: str = "./results_hsd"
    BEST_CKPT_PATH: str = "./results_hsd/best.pt"

    MAX_USER_SEQ_LEN: int = 100
    NUM_NEGATIVES_TRAIN: int = 100

    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    TITLE_EMB_DIM: int = 384
    TEXT_EMB_DIM: int = 384
    IMAGE_EMB_DIM: int = 512

    HIDDEN_DIM: int = 320
    DROPOUT: float = 0.1
    NUM_HEADS: int = 4
    NUM_LAYERS: int = 3

    O_T_DIM: int = 5
    DEVIATION_DIM: int = 16
    N_GROUPS: int = 3
    NUM_BINS: int = 112
    PAD_IDX: int = 0
    
    NUM_MISSING_PATTERNS: int = 8
    MISSING_PATTERN_DIM: int = 16
    
    BIN_EMB_DIM: int = 16

    MAX_UPDATE_RATIO: float = 0.10
    MAX_UPDATE_RATIO_FINAL: float = 0.18
    UPDATE_RATIO_WARMUP_EPOCHS: int = 15
    MIX_LOGIT_INIT: float = 0.0
    
    USE_ADAPTIVE_MIX: bool = True
    
    BPR_TEMPERATURE: float = 0.5
    USE_HARD_NEG_FOCUS: bool = True
    HARD_NEG_TEMPERATURE: float = 0.5
    
    USE_MIXED_NEG_SAMPLING: bool = True
    UNIFORM_NEG_RATIO: float = 0.6
    POPULARITY_ALPHA: float = 0.75
    
    USE_INBATCH_NEGATIVES: bool = True
    
    USE_ITEM_BIAS: bool = True

    EPOCHS: int = 50
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    AMP: bool = True
    SEED: int = 42

    USE_SCHEDULER: bool = True
    SCHEDULER_T_MAX: int = 50
    SCHEDULER_ETA_MIN: float = 1e-6
    WARMUP_EPOCHS: int = 3
    
    USE_EMA: bool = True
    EMA_DECAY: float = 0.999

    USE_SCORE_CALIBRATION: bool = True
    CALIBRATION_N_POP_BINS: int = 10
    CALIBRATION_W_GLOBAL: float = 0.3
    CALIBRATION_W_GROUP: float = 0.3
    CALIBRATION_W_POP: float = 0.4

    W_RANK: float = 1.0
    W_RATING: float = 0.1
    W_MODALITY: float = 0.05
    W_CONTENT: float = 0.01
    DETACH_AUX: bool = True

    AUX_WARMUP_EPOCHS: int = 8
    AUX_RAMPUP_EPOCHS: int = 5

    EARLY_STOP_PATIENCE: int = 10

    VAL_EVERY: int = 1
    VAL_MAX_BATCHES: int = 50
    SAMPLED_EVAL_NEGATIVES: int = 200
    K_VALUES: Tuple[int, ...] = (10, 20, 50)
    FULL_SORT_CHUNK_SIZE: int = 8192
    
  
    FULL_SORT_VAL_EVERY: int = 1
    FULL_SORT_VAL_BATCHES: int = 30
    
    MAX_SEEN_LEN: int = 5000
    SEEN_BLOCK_SIZE: int = 128

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# SharedResources
# =============================================================================

class SharedResources:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        print("[Shared] Loading mappings...")
        with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        self.num_items = mappings["num_items"]
        self.pad_idx = mappings.get("pad_idx", 0)

        print("[Shared] Loading time config...")
        with open(os.path.join(data_dir, "time_bin_config.json"), "r") as f:
            time_config = json.load(f)
        self.num_bins = time_config["num_bins"]
        self.n_groups = time_config["n_groups"]

        print("[Shared] Loading histories...")
        with open(os.path.join(data_dir, "user_histories.pkl"), "rb") as f:
            self.user_histories = pickle.load(f)

        print("[Shared] Loading splits...")
        with open(os.path.join(data_dir, "split_indices.pkl"), "rb") as f:
            self.splits = pickle.load(f)

        print("[Shared] Loading train stats...")
        with open(os.path.join(data_dir, "train_global_stats.pkl"), "rb") as f:
            self.train_stats = pickle.load(f)

        print("[Shared] Building global index mapper...")
        self._build_index_mapper()

        print("[Shared] Loading time stats with delta...")
        self._load_time_stats_with_delta()

        seen_path = os.path.join(data_dir, "user_seen_items.pkl")
        if os.path.exists(seen_path):
            print("[Shared] Loading user_seen_items.pkl...")
            with open(seen_path, "rb") as f:
                user_seen_dict = pickle.load(f)
            MAX_SEEN_FOR_NEG = 5000
            self.user_seen_np: Dict[int, np.ndarray] = {}
            for u, items in user_seen_dict.items():
                arr = np.asarray(items, dtype=np.int64)
                if len(arr) > MAX_SEEN_FOR_NEG:
                    arr = arr[-MAX_SEEN_FOR_NEG:]
                self.user_seen_np[int(u)] = arr
        else:
            self.user_seen_np = {}

        item_seq_path = os.path.join(data_dir, "user_item_seq.pkl")
        if os.path.exists(item_seq_path):
            print("[Shared] Loading user_item_seq.pkl...")
            with open(item_seq_path, "rb") as f:
                self.user_item_seq = pickle.load(f)
            self._use_fast_seen = True
        else:
            self.user_item_seq = {}
            self._use_fast_seen = False

        print("[Shared] Building user_for_target mapping...")
        self.user_for_target = {}
        for u, seq in self.user_histories.items():
            for pos, g in enumerate(seq):
                self.user_for_target[int(g)] = (int(u), int(pos))

      

      
        print("[Shared] Building item popularity counts...")
        self._build_item_counts()

        self._run_sanity_checks()

        print(f"[Shared] Ready: {self.num_items} items, {self.num_bins} bins, {self.n_groups} groups")

    def _build_index_mapper(self):
        with open(os.path.join(self.data_dir, "chunk_info.json"), "r") as f:
            chunk_info = json.load(f)

        self.chunk_paths = {m["chunk_id"]: os.path.join(self.data_dir, m["file"]) for m in chunk_info}
        self.max_global_idx = max(m["global_idx_max"] for m in chunk_info) + 1

        self.g2cid = np.full(self.max_global_idx, -1, dtype=np.int32)
        self.g2row = np.full(self.max_global_idx, -1, dtype=np.int32)

        for meta in tqdm(chunk_info, desc="Indexing chunks", leave=False):
            cid = meta["chunk_id"]
            path = os.path.join(self.data_dir, meta["file"])
            df = pd.read_parquet(path, columns=["global_idx"])
            idxs = df["global_idx"].to_numpy(np.int64)
            self.g2cid[idxs] = cid
            self.g2row[idxs] = np.arange(len(idxs), dtype=np.int32)
            del df

    def _load_time_stats_with_delta(self):
        """Load time statistics with delta computation."""
        with open(os.path.join(self.data_dir, "global_time_stats_lag.pkl"), "rb") as f:
            global_lag = pickle.load(f)
        with open(os.path.join(self.data_dir, "group_time_stats_lag.pkl"), "rb") as f:
            group_lag = pickle.load(f)
        with open(os.path.join(self.data_dir, "user_groups_train_only.pkl"), "rb") as f:
            self.user_groups = pickle.load(f)

        self.global_ot = np.zeros((self.num_bins, 5), dtype=np.float32)
        for b in range(self.num_bins):
            ot = global_lag.get(b, {})
            self.global_ot[b, 0] = ot.get("rating_mean", 4.0) / 5.0
            self.global_ot[b, 1] = ot.get("rating_std", 1.0) / 2.0
            self.global_ot[b, 2] = ot.get("image_rate", 0.08)
            self.global_ot[b, 3] = ot.get("text_len_mean", 50.0) / 100.0
            self.global_ot[b, 4] = ot.get("text_len_std", 30.0) / 50.0

      
        self.global_ot_delta = np.zeros((self.num_bins, 5), dtype=np.float32)
        for b in range(1, self.num_bins):
            self.global_ot_delta[b] = self.global_ot[b] - self.global_ot[b - 1]

        self.group_ot = np.zeros((self.n_groups, self.num_bins, 5), dtype=np.float32)
        for g in range(self.n_groups):
            for b in range(self.num_bins):
                ot = group_lag.get((g, b), {})
                self.group_ot[g, b, 0] = ot.get("rating_mean", 4.0) / 5.0
                self.group_ot[g, b, 1] = ot.get("rating_std", 1.0) / 2.0
                self.group_ot[g, b, 2] = ot.get("image_rate", 0.08)
                self.group_ot[g, b, 3] = ot.get("text_len_mean", 50.0) / 100.0
                self.group_ot[g, b, 4] = ot.get("text_len_std", 30.0) / 50.0

      
        self.group_ot_delta = np.zeros((self.n_groups, self.num_bins, 5), dtype=np.float32)
        for g in range(self.n_groups):
            for b in range(1, self.num_bins):
                self.group_ot_delta[g, b] = self.group_ot[g, b] - self.group_ot[g, b - 1]

    def _build_item_counts(self):
        """Build item popularity counts for mixed negative sampling.

        Notes
        -----
        - Robust: does not depend on user_item_seq.pkl (which may be missing).
        - We count *training targets*; in sequential recommendation, each interaction usually
          appears as a target exactly once in train, so this approximates interaction frequency.
        - Excludes pad_idx and out-of-range ids.
        """
        train_indices = self.splits.get("train", [])
        item_counts = np.zeros(self.num_items, dtype=np.float32)

        if not train_indices:
            # Fallback: uniform probabilities (except pad)
            pop = np.ones(self.num_items, dtype=np.float32)
            pop[self.pad_idx] = 0.0
            pop = pop / (pop.sum() + 1e-8)
            self.item_counts = item_counts
            self.item_pop_probs = pop
            print("  Item popularity: empty train split (fallback uniform).")
            return

        g = np.asarray(train_indices, dtype=np.int64)
        g = g[(g >= 0) & (g < self.max_global_idx)]
        if g.size == 0:
            pop = np.ones(self.num_items, dtype=np.float32)
            pop[self.pad_idx] = 0.0
            pop = pop / (pop.sum() + 1e-8)
            self.item_counts = item_counts
            self.item_pop_probs = pop
            print("  Item popularity: no valid train indices after filtering (fallback uniform).")
            return

        cids = self.g2cid[g]
        rows = self.g2row[g]
        valid = (cids != -1)
        cids = cids[valid]
        rows = rows[valid]

        if cids.size == 0:
            pop = np.ones(self.num_items, dtype=np.float32)
            pop[self.pad_idx] = 0.0
            pop = pop / (pop.sum() + 1e-8)
            self.item_counts = item_counts
            self.item_pop_probs = pop
            print("  Item popularity: all train indices mapped to invalid chunks (fallback uniform).")
            return

        uniq_cids = np.unique(cids)
        for cid in tqdm(uniq_cids, desc="Counting item popularity", leave=False):
            cid_int = int(cid)
            path = self.chunk_paths.get(cid_int)
            if path is None or (not os.path.exists(path)):
                continue

            try:
                df = pd.read_parquet(path, columns=["item_idx"])
            except Exception as e:
                print(f"  WARNING: failed reading {path} for item counts: {e}")
                continue

            items = df["item_idx"].to_numpy(dtype=np.int64, copy=False)
            r = rows[cids == cid]
            r = r[(r >= 0) & (r < items.shape[0])]
            if r.size == 0:
                continue

            sel = items[r]
            sel = sel[(sel >= 0) & (sel < self.num_items)]
            if sel.size == 0:
                continue

            sel = sel[sel != self.pad_idx]
            if sel.size == 0:
                continue

            item_counts += np.bincount(sel, minlength=self.num_items).astype(np.float32, copy=False)

        # Smooth counts and compute sampling probabilities.
        # Keep alpha=0.75 to soften popularity bias (per your prior design choice).
        alpha = 0.75
        smoothed = np.power(item_counts + 1.0, alpha).astype(np.float32, copy=False)
        smoothed[self.pad_idx] = 0.0
        pop_probs = smoothed / (smoothed.sum() + 1e-8)

        self.item_counts = item_counts
        self.item_pop_probs = pop_probs

        print(
            f"  Item popularity: max_count={item_counts.max():.0f}, "
            f"items_with_count>0={int(np.sum(item_counts > 0))}, "
            f"alpha={alpha}"
        )

    def _run_sanity_checks(self):
        """Run data integrity checks."""
        print("\n[Sanity Check] Running data integrity checks...")
        
        all_splits = []
        for split_name in ["train", "val", "test"]:
            all_splits.extend(self.splits.get(split_name, []))
        
        total = len(all_splits)
        if total == 0:
            print("  WARNING: No split indices found!")
            return
        
        sample_size = min(10000, total)
        sample_indices = random.sample(all_splits, sample_size)
        bad_mapping = sum(1 for g in sample_indices if int(g) >= self.max_global_idx or self.g2cid[int(g)] == -1)
        bad_rate = bad_mapping / sample_size * 100
        print(f"  g2cid=-1 rate (sample {sample_size}): {bad_rate:.2f}%")
        if bad_rate > 1.0:
            print(f"  WARNING: High g2cid=-1 rate ({bad_rate:.2f}%) - data may be corrupted!")
        
        mapped_targets = sum(1 for g in all_splits if int(g) in self.user_for_target)
        coverage = mapped_targets / total * 100
        print(f"  user_for_target coverage: {coverage:.2f}%")
        
        valid_targets = 0
        for g in all_splits:
            g_int = int(g)
            if g_int in self.user_for_target:
                _, pos = self.user_for_target[g_int]
                if pos > 0:
                    valid_targets += 1
        valid_rate = valid_targets / total * 100
        print(f"  Valid targets (pos_in_user > 0): {valid_rate:.2f}%")
        
        print("[Sanity Check] Complete\n")

    def get_loc(self, global_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = global_idxs.astype(np.int64)
        return self.g2cid[g], self.g2row[g]

    def get_user_group(self, user_idx: int) -> int:
        return self.user_groups.get(user_idx, 0)

    def get_global_ot(self, bin_id: int) -> np.ndarray:
        return self.global_ot[min(bin_id, self.num_bins - 1)]

    def get_global_ot_delta(self, bin_id: int) -> np.ndarray:
        """Get global O^t delta."""
        return self.global_ot_delta[min(bin_id, self.num_bins - 1)]

    def get_group_ot(self, group_id: int, bin_id: int) -> np.ndarray:
        return self.group_ot[min(group_id, self.n_groups - 1), min(bin_id, self.num_bins - 1)]

    def get_group_ot_delta(self, group_id: int, bin_id: int) -> np.ndarray:
        """Get group O^t delta."""
        return self.group_ot_delta[min(group_id, self.n_groups - 1), min(bin_id, self.num_bins - 1)]

    def get_user_seen_np(self, user_idx: int) -> np.ndarray:
        return self.user_seen_np.get(user_idx, np.array([], dtype=np.int64))

    def get_user_seen_items_fast(self, user_idx: int, pos_in_user: int) -> np.ndarray:
        if self._use_fast_seen and user_idx in self.user_item_seq and pos_in_user > 0:
            seq = self.user_item_seq[user_idx]
            return np.asarray(seq[:pos_in_user], dtype=np.int64)
        return np.array([], dtype=np.int64)
    
    def is_valid_global_idx(self, g: int) -> bool:
        """Check if global_idx is valid."""
        return 0 <= g < self.max_global_idx and self.g2cid[g] != -1


# =============================================================================
# ChunkCache & EmbeddingStore
# =============================================================================

class ChunkCache:
  
    NEEDED_COLS = [
        "global_idx", "user_idx", "item_idx", "rating", "unix_time", "bin_id",
        "has_title", "has_text", "has_image",
        "len_title_tokens", "len_text_tokens",
        # User history features
        "u_num_prev_reviews", "u_prev_mean_rating", "u_prev_image_rate",
        "u_prev_mean_title_len", "u_prev_mean_text_len",
        "u_is_first_review", "u_time_since_prev",
      
        "u_is_image_unusual", "u_is_text_long_unusual", "u_is_title_long_unusual",
        "u_rating_delta",
        # Item history features  
        "i_num_prev_reviews", "i_prev_mean_rating", "i_prev_rating_std",
        "i_prev_image_rate", "i_prev_mean_text_len", "i_is_first_review",
      
        "i_rating_delta", "i_time_since_prev",
      
        "helpful_vote", "verified_purchase", "num_images", "day_of_week",
      
        "missing_pattern_tti",
    ]

    def __init__(self, shared: SharedResources, max_chunks: int = 32):
        self.shared = shared
        self.max_chunks = max_chunks
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._lru: List[int] = []

    def _ensure(self, cid: int):
        if cid in self._cache:
            self._lru.remove(cid)
            self._lru.append(cid)
            return

        path = self.shared.chunk_paths.get(cid)
        if path is None:
            return

        df = pd.read_parquet(path, columns=self.NEEDED_COLS)
        self._cache[cid] = {c: df[c].to_numpy() for c in self.NEEDED_COLS}
        del df

        self._lru.append(cid)
        if len(self._lru) > self.max_chunks:
            old = self._lru.pop(0)
            self._cache.pop(old, None)

    def get(self, cids: np.ndarray, rows: np.ndarray, cols: List[str]) -> Dict[str, np.ndarray]:
        n = len(rows)
        INT64_COLS = {"global_idx", "user_idx", "item_idx", "bin_id"}
        out = {}
        for c in cols:
            if c == "unix_time":
                out[c] = np.zeros(n, dtype=np.float64)
            elif c in INT64_COLS:
                out[c] = np.zeros(n, dtype=np.int64)
            else:
                out[c] = np.zeros(n, dtype=np.float32)

        for cid in np.unique(cids):
            if cid < 0:
                continue
            cid_int = int(cid)
            mask = cids == cid_int
            idxs = np.where(mask)[0]
            self._ensure(cid_int)

            if cid_int not in self._cache:
                continue

            table = self._cache[cid_int]
            r = rows[idxs].astype(np.int64)
            for c in cols:
                if c in table:
                    out[c][idxs] = table[c][r].astype(out[c].dtype)

        return out


class EmbeddingStore:
    def __init__(self, data_dir: str, max_cache: int = 64):
        self.data_dir = data_dir
        self._cache: Dict[Tuple[int, str], np.ndarray] = {}
        self._lru: List[Tuple[int, str]] = []
        self.max_cache = max_cache

    def get(self, cids: np.ndarray, rows: np.ndarray, kind: str, dim: int) -> np.ndarray:
        n = len(rows)
        out = np.zeros((n, dim), dtype=np.float32)

        for cid in np.unique(cids):
            if cid < 0:
                continue
            cid_int = int(cid)
            mask = cids == cid_int
            idxs = np.where(mask)[0]

            key = (cid_int, kind)
            if key not in self._cache:
                path = os.path.join(self.data_dir, "data", f"chunk_{cid_int:04d}_{kind}_emb.npy")
                if os.path.exists(path):
                    self._cache[key] = np.load(path, mmap_mode="r")
                    self._lru.append(key)
                    if len(self._lru) > self.max_cache:
                        old_key = self._lru.pop(0)
                        self._cache.pop(old_key, None)
                else:
                    continue
            else:
                if key in self._lru:
                    self._lru.remove(key)
                    self._lru.append(key)

            arr = self._cache[key]
            r = rows[idxs].astype(np.int64)
            out[idxs] = np.asarray(arr[r], dtype=np.float32)

        return out


# =============================================================================
# Dataset
# =============================================================================

class HSDDataset(Dataset):
    def __init__(
        self,
        shared: SharedResources,
        split: str,
        max_user_len: int,
        num_negatives: int,
        return_user_seen: bool = False,
        cfg: Optional[Config] = None,
    ):
        self.shared = shared
        self.split = split
        self.max_user_len = max_user_len
        self.num_negatives = num_negatives
        self.return_user_seen = return_user_seen
        self.cfg = cfg or Config()
        self.chunk_cache = ChunkCache(shared, max_chunks=32)
        self.emb_store = EmbeddingStore(shared.data_dir)

        raw_targets = shared.splits[split]
        valid_targets = []
        invalid_count = 0
        for t in raw_targets:
            t_int = int(t)
            if t_int not in shared.user_for_target:
                invalid_count += 1
                continue
            user_idx, pos_in_user = shared.user_for_target[t_int]
            if pos_in_user <= 0:
                invalid_count += 1
                continue
          
            if not shared.is_valid_global_idx(t_int):
                invalid_count += 1
                continue
            valid_targets.append(t)

        if invalid_count > 0:
            print(f"  [{split}] Filtered {invalid_count} invalid targets (total: {len(raw_targets)} -> {len(valid_targets)})")

        self.target_indices = np.array(valid_targets, dtype=np.int64)
        self.user_for_target = shared.user_for_target
      
        if len(self.target_indices) > 0:
            sample_size = min(1000, len(self.target_indices))
            sample_targets = self.target_indices[:sample_size]
            g_arr = np.array(sample_targets, dtype=np.int64)
            cids, rows = self.shared.get_loc(g_arr)
            sample_vals = self.chunk_cache.get(cids, rows, ["item_idx"])
            bad_count = (sample_vals["item_idx"] == self.shared.pad_idx).sum()
            if bad_count > 0:
                raise RuntimeError(
                    f"[{split}] Found {bad_count}/{sample_size} samples with target_item==pad_idx. "
                    "Check preprocessing: item indices may start at 0 and collide with padding."
                )
        self.global_text_mean = shared.train_stats.get("mean_text_len", 50.0)
        self.global_image_rate = shared.train_stats.get("image_rate", 0.08)

    def __len__(self) -> int:
        return len(self.target_indices)

    def _get_values(self, global_idxs: List[int], cols: List[str]) -> Dict[str, np.ndarray]:
        if not global_idxs:
            return {c: np.zeros(0, dtype=np.float32) for c in cols}
        g = np.array(global_idxs, dtype=np.int64)
        cids, rows = self.shared.get_loc(g)
        return self.chunk_cache.get(cids, rows, cols)

    def _get_embeddings(self, global_idxs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not global_idxs:
            return (np.zeros((0, 384), np.float32), np.zeros((0, 384), np.float32), np.zeros((0, 512), np.float32))
        g = np.array(global_idxs, dtype=np.int64)
        cids, rows = self.shared.get_loc(g)
        return (
            self.emb_store.get(cids, rows, "title", 384),
            self.emb_store.get(cids, rows, "text", 384),
            self.emb_store.get(cids, rows, "image", 512),
        )

    def _build_numeric_features(self, vals: Dict[str, np.ndarray], target_unix: float) -> Tuple[np.ndarray, np.ndarray]:
        L = len(next(iter(vals.values()))) if vals else 0
        x = np.zeros((L, 18), dtype=np.float32)

        ts = self.shared.train_stats

        def get(name, default):
            return vals.get(name, np.full(L, default, dtype=np.float32))

        x[:, 0] = get("rating", 4.0) / 5.0
        x[:, 1] = get("has_title", 1.0)
        x[:, 2] = get("has_text", 1.0)
        x[:, 3] = get("has_image", 0.0)
        x[:, 4] = get("len_title_tokens", 5.0) / 50.0
        x[:, 5] = get("len_text_tokens", 50.0) / 200.0
        x[:, 6] = get("u_num_prev_reviews", 0.0) / 50.0
        x[:, 7] = get("u_prev_mean_rating", ts["mean_rating"]) / 5.0
        x[:, 8] = get("u_prev_image_rate", ts["image_rate"])
        x[:, 9] = get("u_prev_mean_text_len", ts["mean_text_len"]) / 100.0
        x[:, 10] = get("u_is_first_review", 0.0)
        x[:, 11] = get("u_time_since_prev", 0.0) / 5.0
        x[:, 12] = get("i_num_prev_reviews", 0.0) / 50.0
        x[:, 13] = get("i_prev_mean_rating", ts["mean_rating"]) / 5.0
        x[:, 14] = get("i_prev_rating_std", 1.0) / 2.0
        x[:, 15] = get("i_prev_image_rate", ts["image_rate"])
        x[:, 16] = get("i_prev_mean_text_len", ts["mean_text_len"]) / 100.0
        x[:, 17] = get("i_is_first_review", 0.0)

        np.nan_to_num(x, copy=False, nan=0.0)

        hist_unix = get("unix_time", target_unix)
        age_seconds = np.maximum(target_unix - hist_unix, 0.0)
        age_days = age_seconds / 86400.0
        time_age = np.log1p(age_days).astype(np.float32)

        return x, time_age

    def _build_deviation_from_history(self, hist_vals: Dict[str, np.ndarray], 
                                       target_vals: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Build deviation features from history."""
        dev = np.zeros(16, dtype=np.float32)

        L = len(hist_vals.get("has_image", []))
        if L < 2:
            # Still use preprocessed features even with short history
            if target_vals is not None:
                dev[6] = float(target_vals.get("u_is_image_unusual", 0.0))
                dev[7] = float(target_vals.get("u_is_text_long_unusual", 0.0))
                dev[8] = float(target_vals.get("u_is_title_long_unusual", 0.0))
                dev[9] = np.clip(float(target_vals.get("u_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
                dev[10] = np.clip(float(target_vals.get("i_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
                dev[11] = np.log1p(float(target_vals.get("num_images", 0.0))) / 2.0
                dev[12] = np.log1p(float(target_vals.get("helpful_vote", 0.0))) / 5.0
            return dev

        last_has_image = float(hist_vals["has_image"][-1])
        last_text_len = float(hist_vals["len_text_tokens"][-1])
        last_title_len = float(hist_vals["len_title_tokens"][-1])
        last_rating = float(hist_vals["rating"][-1])

        prev_image_rate = float(hist_vals["has_image"][:-1].mean())
        prev_text_mean = float(hist_vals["len_text_tokens"][:-1].mean())
        prev_title_mean = float(hist_vals["len_title_tokens"][:-1].mean())
        prev_rating_mean = float(hist_vals["rating"][:-1].mean())

        # Original 6 dimensions
        dev[0] = last_has_image - prev_image_rate
        text_std = max(prev_text_mean * 0.5, 15.0)
        dev[1] = np.clip((last_text_len - prev_text_mean) / text_std, -3.0, 3.0)
        title_std = max(prev_title_mean * 0.5, 2.0)
        dev[2] = np.clip((last_title_len - prev_title_mean) / title_std, -3.0, 3.0)
        dev[3] = (last_rating - prev_rating_mean) / 2.0
        dev[4] = min(L / 20.0, 1.0)  # History reliability (length)
        dev[5] = abs(dev[0])  # Absolute image deviation
        
      
        if target_vals is not None:
            # Preprocessed "unusual" flags - direct deviation signals
            dev[6] = float(target_vals.get("u_is_image_unusual", 0.0))
            dev[7] = float(target_vals.get("u_is_text_long_unusual", 0.0))
            dev[8] = float(target_vals.get("u_is_title_long_unusual", 0.0))
            
            # Rating deltas (standardized)
            dev[9] = np.clip(float(target_vals.get("u_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
            dev[10] = np.clip(float(target_vals.get("i_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
            
            # Behavioral features
            dev[11] = np.log1p(float(target_vals.get("num_images", 0.0))) / 2.0  # log1p(num_images)
            dev[12] = np.log1p(float(target_vals.get("helpful_vote", 0.0))) / 5.0  # log1p(helpful_vote)
            
            # First-time image upload signal
            dev[13] = 1.0 if (prev_image_rate < 0.05 and last_has_image > 0.5) else 0.0
            
            # Time-based features
            u_time_since = float(target_vals.get("u_time_since_prev", 0.0))
            i_time_since = float(target_vals.get("i_time_since_prev", 0.0))
            dev[14] = np.clip(np.log1p(u_time_since) / 10.0, 0, 1.0)  # User time gap
            dev[15] = np.clip(np.log1p(i_time_since) / 10.0, 0, 1.0)  # Item time gap
        else:
            # Fallback: first-time image from history
            dev[13] = 1.0 if (prev_image_rate < 0.05 and last_has_image > 0.5) else 0.0
            dev[14] = np.log1p(L) / 5.0  # Use history length as proxy
            dev[15] = 0.0

        return dev

    def _build_item_features(self, target_vals: Dict[str, float]) -> np.ndarray:
        ts = self.shared.train_stats
        feat = np.zeros(6, dtype=np.float32)
        feat[0] = float(target_vals.get("i_num_prev_reviews", 0.0)) / 100.0
        feat[1] = float(target_vals.get("i_prev_mean_rating", ts["mean_rating"])) / 5.0
        feat[2] = float(target_vals.get("i_prev_rating_std", 1.0)) / 2.0
        feat[3] = float(target_vals.get("i_prev_image_rate", ts["image_rate"]))
        feat[4] = float(target_vals.get("i_prev_mean_text_len", ts["mean_text_len"])) / 100.0
        feat[5] = float(target_vals.get("i_is_first_review", 0.0))
        return feat

    def _sample_negatives(self, user_idx: int, target_item: int) -> Tuple[np.ndarray, np.ndarray]:
        """Mixed negative sampling (uniform + popularity), with de-duplication.

        Returns
        -------
        negatives: (num_neg,) int64
        neg_valid: (num_neg,) bool
        """
        cfg = self.cfg
        num_items = self.shared.num_items
        pad_idx = self.shared.pad_idx
        num_neg = self.num_negatives

        seen_array = self.shared.get_user_seen_np(user_idx)

        # Exclude: user's seen items, target item, pad
        used = set(seen_array.tolist()) if len(seen_array) > 0 else set()
        used.add(int(target_item))
        used.add(int(pad_idx))

        sample_min = 1 if pad_idx == 0 else 0
        negatives = np.full(num_neg, pad_idx, dtype=np.int64)
        neg_valid = np.zeros(num_neg, dtype=np.bool_)

        if num_neg <= 0:
            return negatives, neg_valid

        # Split budget
        if cfg.USE_MIXED_NEG_SAMPLING and hasattr(self.shared, "item_pop_probs"):
            n_uniform = int(round(num_neg * cfg.UNIFORM_NEG_RATIO))
            n_uniform = max(0, min(num_neg, n_uniform))
            n_popularity = num_neg - n_uniform
        else:
            n_uniform = num_neg
            n_popularity = 0

        collected = 0

        # 1) Uniform sampling
        max_rounds = 4
        for round_i in range(max_rounds):
            need = n_uniform - collected
            if need <= 0:
                break

            n_candidates = need * (30 if round_i == 0 else 60)
            candidates = np.random.randint(sample_min, num_items, size=n_candidates)

            # Basic range filter
            valid_candidates = candidates[(candidates >= sample_min) & (candidates < num_items)]
            if valid_candidates.size == 0:
                continue

            # Unique, then fill while excluding used
            uniq = np.unique(valid_candidates)
            for c in uniq.tolist():
                if c not in used:
                    negatives[collected] = c
                    neg_valid[collected] = True
                    used.add(c)
                    collected += 1
                    if collected >= n_uniform:
                        break

        # 2) Popularity-based sampling
        if n_popularity > 0 and hasattr(self.shared, "item_pop_probs"):
            pop_probs = self.shared.item_pop_probs.copy()

            if used:
                idx = np.fromiter(used, dtype=np.int64, count=len(used))
                idx = idx[(idx >= 0) & (idx < num_items)]
                pop_probs[idx] = 0.0

            pop_sum = float(pop_probs.sum())
            if pop_sum > 1e-8:
                pop_probs = pop_probs / pop_sum

                # Oversample then pick unseen
                n_candidates = n_popularity * 10
                pop_candidates = np.random.choice(num_items, size=n_candidates, replace=True, p=pop_probs)

                target_total = n_uniform + n_popularity
                for c in pop_candidates.tolist():
                    if c not in used:
                        negatives[collected] = c
                        neg_valid[collected] = True
                        used.add(c)
                        collected += 1
                        if collected >= target_total:
                            break

        return negatives, neg_valid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        target_g = int(self.target_indices[idx])
        user_idx, pos_in_user = self.user_for_target[target_g]
        user_seq = self.shared.user_histories.get(user_idx, [])

        start = max(0, pos_in_user - self.max_user_len)
        user_hist_g = [
            int(g) for g in user_seq[start:pos_in_user]
            if self.shared.is_valid_global_idx(int(g))
        ]
        # ============================================================

        target_vals = self._get_values([target_g], ChunkCache.NEEDED_COLS)
        target_vals_dict = {k: float(v[0]) if len(v) > 0 else 0.0 for k, v in target_vals.items()}

        target_item = int(target_vals_dict.get("item_idx", 0))
        target_bin = int(target_vals_dict.get("bin_id", 0))
        target_rating = float(target_vals_dict.get("rating", 4.0))
        target_unix = float(target_vals_dict.get("unix_time", 0.0))
        if target_item == self.shared.pad_idx:
            raise RuntimeError(
                f"Found target_item==pad_idx ({self.shared.pad_idx}) at global_idx={target_g}. "
                "This usually means item indices start at 0 and collide with padding, "
                "or the data is corrupted. Check your preprocessing!"
            )
        # ======================================================

        user_group = self.shared.get_user_group(user_idx)
        global_ot = self.shared.get_global_ot(target_bin)
        group_ot = self.shared.get_group_ot(user_group, target_bin)
      
        global_ot_delta = self.shared.get_global_ot_delta(target_bin)
        group_ot_delta = self.shared.get_group_ot_delta(user_group, target_bin)

        if user_hist_g:
            hist_vals = self._get_values(user_hist_g, ChunkCache.NEEDED_COLS)
            user_items = hist_vals["item_idx"].astype(np.int64)
            user_num, time_age = self._build_numeric_features(hist_vals, target_unix)
            title_emb, text_emb, image_emb = self._get_embeddings(user_hist_g)
            deviation = self._build_deviation_from_history(hist_vals, target_vals_dict)
          
            user_hist_missing_pattern = hist_vals.get("missing_pattern_tti", np.zeros(len(user_hist_g), dtype=np.int64)).astype(np.int64)
            user_hist_bin_ids = hist_vals.get("bin_id", np.zeros(len(user_hist_g), dtype=np.int64)).astype(np.int64)
        else:
            user_items = np.zeros(0, dtype=np.int64)
            user_num = np.zeros((0, 18), dtype=np.float32)
            time_age = np.zeros(0, dtype=np.float32)
            title_emb = np.zeros((0, 384), dtype=np.float32)
            text_emb = np.zeros((0, 384), dtype=np.float32)
            image_emb = np.zeros((0, 512), dtype=np.float32)
            deviation = np.zeros(16, dtype=np.float32)
            user_hist_missing_pattern = np.zeros(0, dtype=np.int64)
            user_hist_bin_ids = np.zeros(0, dtype=np.int64)

        item_features = self._build_item_features(target_vals_dict)

        t_title, t_text, t_image = self._get_embeddings([target_g])
        target_title = t_title[0] if len(t_title) > 0 else np.zeros(384, dtype=np.float32)
        target_text = t_text[0] if len(t_text) > 0 else np.zeros(384, dtype=np.float32)
        target_image = t_image[0] if len(t_image) > 0 else np.zeros(512, dtype=np.float32)

        if self.split == "train" and self.num_negatives > 0:
            negatives, neg_valid = self._sample_negatives(user_idx, target_item)
        else:
            negatives = np.zeros(0, dtype=np.int64)
            neg_valid = np.zeros(0, dtype=np.bool_)

        user_seen = np.zeros(0, dtype=np.int64)
        if self.return_user_seen:
            user_seen = self.shared.get_user_seen_items_fast(user_idx, pos_in_user)
            if len(user_seen) == 0 and not self.shared._use_fast_seen:
                past_seq = list(map(int, user_seq[:pos_in_user]))
                if past_seq:
                    seen_vals = self._get_values(past_seq, ["item_idx"])
                    user_seen = seen_vals["item_idx"].astype(np.int64)

        return {
            "pad_idx": self.shared.pad_idx,
            "user_idx": user_idx,
            "user_group": user_group,
            "target_item": target_item,
            "target_bin": target_bin,
            "target_rating": np.float32(target_rating),
            "target_has_title": np.float32(target_vals_dict.get("has_title", 1.0)),
            "target_has_text": np.float32(target_vals_dict.get("has_text", 1.0)),
            "target_has_image": np.float32(target_vals_dict.get("has_image", 0.0)),
            "target_title_emb": target_title,
            "target_text_emb": target_text,
            "target_image_emb": target_image,
            "global_ot": global_ot.astype(np.float32),
            "group_ot": group_ot.astype(np.float32),
          
            "global_ot_delta": global_ot_delta.astype(np.float32),
            "group_ot_delta": group_ot_delta.astype(np.float32),
            "deviation": deviation,
            "item_features": item_features,
            "user_hist_len": len(user_hist_g),
            "user_hist_items": user_items,
            "user_hist_num": user_num,
            "user_hist_time_age": time_age,
            "user_hist_title_emb": title_emb,
            "user_hist_text_emb": text_emb,
            "user_hist_image_emb": image_emb,
            "user_hist_missing_pattern": user_hist_missing_pattern,
            "user_hist_bin_ids": user_hist_bin_ids,
            "target_missing_pattern": np.int64(target_vals_dict.get("missing_pattern_tti", 0)),
            "negative_items": negatives,
            "neg_valid": neg_valid,
            "user_seen_items": user_seen,
        }


# =============================================================================
# Collate Function
# =============================================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = [b for b in batch if b is not None]
    B = len(batch)
    if B == 0:
        raise ValueError("Empty batch")

    pad_idx = batch[0].get("pad_idx", 0)

    user_idx = torch.tensor([b["user_idx"] for b in batch], dtype=torch.long)
    user_group = torch.tensor([b["user_group"] for b in batch], dtype=torch.long)
    target_item = torch.tensor([b["target_item"] for b in batch], dtype=torch.long)
    target_bin = torch.tensor([b["target_bin"] for b in batch], dtype=torch.long)
    target_rating = torch.tensor([b["target_rating"] for b in batch], dtype=torch.float32)
    target_has_title = torch.tensor([b["target_has_title"] for b in batch], dtype=torch.float32)
    target_has_text = torch.tensor([b["target_has_text"] for b in batch], dtype=torch.float32)
    target_has_image = torch.tensor([b["target_has_image"] for b in batch], dtype=torch.float32)

    target_title_emb = torch.from_numpy(np.stack([b["target_title_emb"] for b in batch]))
    target_text_emb = torch.from_numpy(np.stack([b["target_text_emb"] for b in batch]))
    target_image_emb = torch.from_numpy(np.stack([b["target_image_emb"] for b in batch]))

    global_ot = torch.from_numpy(np.stack([b["global_ot"] for b in batch]))
    group_ot = torch.from_numpy(np.stack([b["group_ot"] for b in batch]))
  
    global_ot_delta = torch.from_numpy(np.stack([b["global_ot_delta"] for b in batch]))
    group_ot_delta = torch.from_numpy(np.stack([b["group_ot_delta"] for b in batch]))
    
    deviation = torch.from_numpy(np.stack([b["deviation"] for b in batch]))
    item_features = torch.from_numpy(np.stack([b["item_features"] for b in batch]))

    user_lens = [b["user_hist_len"] for b in batch]
    L = max(1, max(user_lens) if user_lens else 1)

    user_hist_len = torch.tensor(user_lens, dtype=torch.long)
    user_hist_items = torch.full((B, L), pad_idx, dtype=torch.long)
    user_hist_num = torch.zeros(B, L, 18, dtype=torch.float32)
    user_hist_time_age = torch.zeros(B, L, dtype=torch.float32)
    user_hist_mask = torch.zeros(B, L, dtype=torch.bool)
    user_hist_title_emb = torch.zeros(B, L, 384, dtype=torch.float32)
    user_hist_text_emb = torch.zeros(B, L, 384, dtype=torch.float32)
    user_hist_image_emb = torch.zeros(B, L, 512, dtype=torch.float32)
  
    user_hist_missing_pattern = torch.zeros(B, L, dtype=torch.long)
    user_hist_bin_ids = torch.zeros(B, L, dtype=torch.long)
    target_missing_pattern = torch.tensor([b.get("target_missing_pattern", 0) for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        l = b["user_hist_len"]
        if l > 0:
            user_hist_items[i, :l] = torch.from_numpy(b["user_hist_items"][:l])
            user_hist_num[i, :l] = torch.from_numpy(b["user_hist_num"][:l])
            user_hist_time_age[i, :l] = torch.from_numpy(b["user_hist_time_age"][:l])
            user_hist_mask[i, :l] = True
            user_hist_title_emb[i, :l] = torch.from_numpy(b["user_hist_title_emb"][:l])
            user_hist_text_emb[i, :l] = torch.from_numpy(b["user_hist_text_emb"][:l])
            user_hist_image_emb[i, :l] = torch.from_numpy(b["user_hist_image_emb"][:l])
          
            if "user_hist_missing_pattern" in b and len(b["user_hist_missing_pattern"]) > 0:
                user_hist_missing_pattern[i, :l] = torch.from_numpy(b["user_hist_missing_pattern"][:l])
            if "user_hist_bin_ids" in b and len(b["user_hist_bin_ids"]) > 0:
                user_hist_bin_ids[i, :l] = torch.from_numpy(b["user_hist_bin_ids"][:l])

    for i in range(B):
        if not user_hist_mask[i].any():
            user_hist_mask[i, 0] = True

    max_negs = max((b["negative_items"].shape[0] for b in batch), default=0)
    if max_negs > 0:
        neg_items = torch.full((B, max_negs), pad_idx, dtype=torch.long)
        neg_valid_mask = torch.zeros(B, max_negs, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["negative_items"].shape[0]
            if n > 0:
                neg_items[i, :n] = torch.from_numpy(b["negative_items"])
                if "neg_valid" in b and len(b["neg_valid"]) > 0:
                    neg_valid_mask[i, :n] = torch.from_numpy(b["neg_valid"][:n])
                else:
                    neg_valid_mask[i, :n] = True
    else:
        neg_items = torch.empty(B, 0, dtype=torch.long)
        neg_valid_mask = torch.empty(B, 0, dtype=torch.bool)

    max_seen_len = 5000
    max_seen = min(max_seen_len, max((b["user_seen_items"].shape[0] for b in batch), default=0))
    user_seen = torch.full((B, max(1, max_seen)), pad_idx, dtype=torch.long)
    user_seen_mask = torch.zeros(B, max(1, max_seen), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = min(b["user_seen_items"].shape[0], max_seen_len)
        if n > 0:
            user_seen[i, :n] = torch.from_numpy(b["user_seen_items"][:n])
            user_seen_mask[i, :n] = True

    return {
        "user_idx": user_idx,
        "user_group": user_group,
        "target_item": target_item,
        "target_bin": target_bin,
        "target_rating": target_rating,
        "target_has_title": target_has_title,
        "target_has_text": target_has_text,
        "target_has_image": target_has_image,
        "target_title_emb": target_title_emb,
        "target_text_emb": target_text_emb,
        "target_image_emb": target_image_emb,
        "global_ot": global_ot,
        "group_ot": group_ot,
      
        "global_ot_delta": global_ot_delta,
        "group_ot_delta": group_ot_delta,
        "deviation": deviation,
        "item_features": item_features,
        "user_hist_len": user_hist_len,
        "user_hist_items": user_hist_items,
        "user_hist_num": user_hist_num,
        "user_hist_time_age": user_hist_time_age,
        "user_hist_mask": user_hist_mask,
        "user_hist_title_emb": user_hist_title_emb,
        "user_hist_text_emb": user_hist_text_emb,
        "user_hist_image_emb": user_hist_image_emb,
        "user_hist_missing_pattern": user_hist_missing_pattern,
        "user_hist_bin_ids": user_hist_bin_ids,
        "target_missing_pattern": target_missing_pattern,
        "negative_items": neg_items,
        "neg_valid_mask": neg_valid_mask,
        "user_seen_items": user_seen,
        "user_seen_mask": user_seen_mask,
    }


# =============================================================================
# Model Components
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class GatedModalityFusion(nn.Module):
    def __init__(self, title_dim: int, text_dim: int, image_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.title_proj = nn.Sequential(
            nn.Linear(title_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.gate = nn.Linear(hidden_dim * 3, 3)

    def forward(self, title_emb, text_emb, image_emb, has_title, has_text, has_image):
        t_title = self.title_proj(title_emb)
        t_text = self.text_proj(text_emb)
        t_image = self.image_proj(image_emb)

        logits = self.gate(torch.cat([t_title, t_text, t_image], dim=-1))

        if has_title.dim() < logits.dim():
            has_title = has_title.unsqueeze(-1)
            has_text = has_text.unsqueeze(-1)
            has_image = has_image.unsqueeze(-1)

        avail = torch.stack([has_title > 0.5, has_text > 0.5, has_image > 0.5], dim=-1).squeeze(-2)
        logits = logits.masked_fill(~avail, -1e4)
        weights = F.softmax(logits, dim=-1)
        weights = weights.masked_fill(~avail.any(dim=-1, keepdim=True), 0.0)

        return weights[..., 0:1] * t_title + weights[..., 1:2] * t_text + weights[..., 2:3] * t_image


class EventEncoder(nn.Module):
    """Event encoder with missing pattern and bin embeddings."""
    def __init__(self, hidden_dim: int, num_feat: int, item_emb: nn.Embedding, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.item_emb = item_emb
        self.num_mlp = nn.Sequential(
            nn.Linear(num_feat, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(cfg.DROPOUT)
        )
        self.modality_fusion = GatedModalityFusion(
            cfg.TITLE_EMB_DIM, cfg.TEXT_EMB_DIM, cfg.IMAGE_EMB_DIM, hidden_dim, cfg.DROPOUT
        )
        self.recency_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        
      
        self.missing_pattern_emb = nn.Embedding(cfg.NUM_MISSING_PATTERNS, cfg.MISSING_PATTERN_DIM)
        self.pattern_proj = nn.Linear(cfg.MISSING_PATTERN_DIM, hidden_dim)
        
      
        self.bin_emb = nn.Embedding(cfg.NUM_BINS + 1, hidden_dim // 4)  # +1 for padding
        self.bin_proj = nn.Linear(hidden_dim // 4, hidden_dim)
        
        self.out_ln = nn.LayerNorm(hidden_dim)

    def forward(self, item_ids, num_feat, time_age, title_emb, text_emb, image_emb,
                missing_pattern=None, bin_ids=None):
        item_term = self.item_emb(item_ids)
        num_term = self.num_mlp(num_feat)
        rec_term = self.recency_mlp(time_age.unsqueeze(-1))
        mod_term = self.modality_fusion(
            title_emb, text_emb, image_emb, num_feat[..., 1], num_feat[..., 2], num_feat[..., 3]
        )
        
        result = item_term + num_term + rec_term + mod_term
        
      
        if missing_pattern is not None:
            pattern_ids = missing_pattern.clamp(0, self.cfg.NUM_MISSING_PATTERNS - 1)
            pattern_term = self.pattern_proj(self.missing_pattern_emb(pattern_ids))
            result = result + pattern_term
        
      
        if bin_ids is not None:
            bin_ids_clamped = bin_ids.clamp(0, self.cfg.NUM_BINS)
            bin_term = self.bin_proj(self.bin_emb(bin_ids_clamped))
            result = result + bin_term
        
        return self.out_ln(result)


class CausalTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, num_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.pos = PositionalEncoding(hidden_dim, max_len, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def _get_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        key = (L, device)
        if key not in self._mask_cache:
            mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float("-inf"))
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for i in range(mask.size(0)):
            if not mask[i].any():
                mask[i, 0] = True

        x = self.pos(x)
        B, L, _ = x.shape
        causal_mask = self._get_causal_mask(L, x.device)
        y = self.enc(x, mask=causal_mask, src_key_padding_mask=~mask)
        lengths = mask.sum(dim=1).clamp(min=1) - 1
        return y[torch.arange(B, device=x.device), lengths]


class TimeContextEncoderV7(nn.Module):
    """Time context encoder with delta information."""
    def __init__(self, n_groups: int, num_bins: int, o_t_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.group_emb = nn.Embedding(n_groups, hidden_dim)
        self.bin_emb = nn.Embedding(num_bins, hidden_dim // 2)

        self.global_ot_encoder = nn.Sequential(
            nn.Linear(o_t_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.group_ot_encoder = nn.Sequential(
            nn.Linear(o_t_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

      
        self.global_delta_encoder = nn.Sequential(
            nn.Linear(o_t_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        self.group_delta_encoder = nn.Sequential(
            nn.Linear(o_t_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        # - g_emb + global_delta_emb + group_delta_emb: [B, H]
        # - global_ot_emb: [B, H]
        # - group_ot_emb: [B, H]
        # - b_emb: [B, H//2]
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        # ============================================================

    def forward(
        self,
        group_ids: torch.Tensor,
        bin_ids: torch.Tensor,
        global_ot: torch.Tensor,
        group_ot: torch.Tensor,
        global_ot_delta: torch.Tensor,
        group_ot_delta: torch.Tensor
    ) -> torch.Tensor:
      
        group_ids = group_ids.clamp(0, self.group_emb.num_embeddings - 1)
        bin_ids = bin_ids.clamp(0, self.bin_emb.num_embeddings - 1)
        
        g_emb = self.group_emb(group_ids)  # [B, H]
        b_emb = self.bin_emb(bin_ids)  # [B, H//2]
        global_ot_emb = self.global_ot_encoder(global_ot)  # [B, H]
        group_ot_emb = self.group_ot_encoder(group_ot)  # [B, H]

      
        global_delta_emb = self.global_delta_encoder(global_ot_delta)  # [B, H]
        group_delta_emb = self.group_delta_encoder(group_ot_delta)  # [B, H]

        combined = torch.cat([
            g_emb + global_delta_emb + group_delta_emb,
            global_ot_emb,
            group_ot_emb,
            b_emb
        ], dim=-1)

        avg_change = self.fuse(combined)
        return avg_change


class HSDUpdater(nn.Module):
    """DiD updater with adaptive per-sample mix_weight."""
    def __init__(self, hidden_dim: int, deviation_dim: int, dropout: float, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim

        self.deviation_encoder = nn.Sequential(
            nn.Linear(deviation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + deviation_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.hetero_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

      
        if cfg.USE_ADAPTIVE_MIX:
            self.mix_net = nn.Sequential(
                nn.Linear(deviation_dim + 1, hidden_dim // 4),  # deviation + reliability proxy
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
            # Learnable bias initialized to match MIX_LOGIT_INIT
            self.mix_bias = nn.Parameter(torch.tensor(cfg.MIX_LOGIT_INIT))
        else:
            # Fallback to global scalar
            self.mix_logit = nn.Parameter(torch.tensor(cfg.MIX_LOGIT_INIT))

    def forward(
        self,
        z_prev: torch.Tensor,
        avg_change: torch.Tensor,
        deviation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Bounded update:
        1. raw_update = (1-w) * avg_change + w * gate * hetero_change
        2. bounded_update = scale * raw_update, where ||bounded_update|| <= MAX_UPDATE_RATIO * ||z_prev||
        3. z_new = z_prev + bounded_update
        """
        dev_emb = self.deviation_encoder(deviation)

        gate_input = torch.cat([z_prev, deviation], dim=-1)
        gate = self.gate_net(gate_input)

        hetero_input = torch.cat([z_prev, dev_emb], dim=-1)
        hetero_change = self.hetero_net(hetero_input)

      
        if self.cfg.USE_ADAPTIVE_MIX:
            # Use deviation[4] as reliability proxy (history length feature)
            reliability = deviation[..., 4:5]  # [B, 1]
            mix_input = torch.cat([deviation, reliability], dim=-1)  # [B, deviation_dim + 1]
            mix_logit = self.mix_net(mix_input).squeeze(-1) + self.mix_bias  # [B]
            w = torch.sigmoid(mix_logit).unsqueeze(-1)  # [B, 1]
        else:
            w = torch.sigmoid(self.mix_logit)

        # Construct raw update vector
        raw_update = (1 - w) * avg_change + w * gate * hetero_change

        # Compute norms
        z_prev_norm = z_prev.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        update_norm = raw_update.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Compute allowed max update norm
        max_update_norm = self.cfg.MAX_UPDATE_RATIO * z_prev_norm

        # Compute scale factor
        scale = (max_update_norm / update_norm).clamp(max=1.0)

        # Apply scaling
        bounded_update = scale * raw_update

        # Final update
        z_new = z_prev + bounded_update

        # Compute actual update ratio for monitoring
        actual_ratio = (bounded_update.norm(dim=-1) / z_prev_norm.squeeze(-1)).mean()
        
        # For monitoring: get scalar mix_weight
        if self.cfg.USE_ADAPTIVE_MIX:
            mix_weight_scalar = w.squeeze(-1).mean()
        else:
            mix_weight_scalar = w

        return {
            "z_new": z_new,
            "hetero_change": hetero_change,
            "gate": gate,
            "mix_weight": mix_weight_scalar,
            "actual_ratio": actual_ratio,
            "avg_change_norm": avg_change.norm(dim=-1).mean(),
            "hetero_change_norm": hetero_change.norm(dim=-1).mean(),
        }


class ItemContextEncoder(nn.Module):
    def __init__(self, hidden_dim: int, item_feat_dim: int, o_t_dim: int, dropout: float):
        super().__init__()
        self.feat_encoder = nn.Sequential(
            nn.Linear(item_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ot_encoder = nn.Sequential(
            nn.Linear(o_t_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, item_features: torch.Tensor, global_ot: torch.Tensor) -> torch.Tensor:
        feat_emb = self.feat_encoder(item_features)
        ot_emb = self.ot_encoder(global_ot)
        return self.fuse(torch.cat([feat_emb, ot_emb], dim=-1))


class MLPHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Full Model
# =============================================================================

class HSDModel(nn.Module):
    """DiD model with item bias and extended event encoding."""
    def __init__(self, num_items: int, pad_idx: int, cfg: Config):
        super().__init__()
        self.num_items = num_items
        self.pad_idx = pad_idx
        self.cfg = cfg
        H = cfg.HIDDEN_DIM

        self.item_emb = nn.Embedding(num_items, H, padding_idx=pad_idx)
      
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=pad_idx)
        nn.init.zeros_(self.item_bias.weight)
        
        self.event_encoder = EventEncoder(H, 18, self.item_emb, cfg)
        self.user_seq = CausalTransformerEncoder(H, cfg.NUM_HEADS, cfg.NUM_LAYERS, cfg.DROPOUT, cfg.MAX_USER_SEQ_LEN)

        self.time_context = TimeContextEncoderV7(cfg.N_GROUPS, cfg.NUM_BINS, cfg.O_T_DIM, H, cfg.DROPOUT)
        self.hsd_updater = HSDUpdater(H, cfg.DEVIATION_DIM, cfg.DROPOUT, cfg)

        self.item_context = ItemContextEncoder(H, 6, cfg.O_T_DIM, cfg.DROPOUT)

        self.rating_head = MLPHead(H * 2, 1, cfg.DROPOUT)
        self.modality_head = MLPHead(H * 2, 1, cfg.DROPOUT)
        self.title_head = MLPHead(H * 2, cfg.TITLE_EMB_DIM, cfg.DROPOUT)
        self.text_head = MLPHead(H * 2, cfg.TEXT_EMB_DIM, cfg.DROPOUT)
        self.image_head = MLPHead(H * 2, cfg.IMAGE_EMB_DIM, cfg.DROPOUT)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def encode_user(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
      
        event_emb = self.event_encoder(
            batch["user_hist_items"], batch["user_hist_num"], batch["user_hist_time_age"],
            batch["user_hist_title_emb"], batch["user_hist_text_emb"], batch["user_hist_image_emb"],
            missing_pattern=batch.get("user_hist_missing_pattern"),
            bin_ids=batch.get("user_hist_bin_ids")
        )
        z_prev = self.user_seq(event_emb, batch["user_hist_mask"])

      
        avg_change = self.time_context(
            batch["user_group"], batch["target_bin"],
            batch["global_ot"], batch["group_ot"],
            batch["global_ot_delta"], batch["group_ot_delta"]
        )

        hsd_out = self.hsd_updater(z_prev, avg_change, batch["deviation"])
        z_new = hsd_out["z_new"]

        return z_new, hsd_out

    def forward(self, batch: Dict[str, torch.Tensor], compute_aux: bool = True) -> Dict[str, torch.Tensor]:
        user_state, hsd_out = self.encode_user(batch)

        # ===== Positive =====
        pos_items = batch["target_item"]
        pos_item_emb = self.item_emb(pos_items)
        pos_score = (user_state * pos_item_emb).sum(dim=-1)
        pos_score = pos_score + self.item_bias(pos_items).squeeze(-1)

        # ===== Sampled negatives =====
        neg_items = batch["negative_items"]
        if neg_items.numel() > 0 and neg_items.size(1) > 0:
            neg_item_emb = self.item_emb(neg_items)
            neg_scores = torch.einsum("bh,bnh->bn", user_state, neg_item_emb)
            neg_scores = neg_scores + self.item_bias(neg_items).squeeze(-1)

            neg_mask = batch.get("neg_valid_mask", None)
            if neg_mask is None:
                neg_mask = torch.ones_like(neg_scores, dtype=torch.bool)
            else:
                neg_mask = neg_mask.to(dtype=torch.bool)
        else:
            neg_scores = torch.zeros(user_state.size(0), 0, device=user_state.device)
            neg_mask = torch.zeros(user_state.size(0), 0, device=user_state.device, dtype=torch.bool)

        # ===== In-batch negatives =====
        if self.cfg.USE_INBATCH_NEGATIVES and self.training:
            inb_scores, inb_mask = self._build_inbatch_negatives(user_state, pos_items)
            if inb_scores.numel() > 0:
                neg_scores = torch.cat([neg_scores, inb_scores], dim=1)
                neg_mask = torch.cat([neg_mask, inb_mask], dim=1)

        out = {
            "user_state": user_state,
            "pos_score": pos_score,
            "neg_scores": neg_scores,
            "neg_mask": neg_mask,  # IMPORTANT
            "hetero_change": hsd_out["hetero_change"],
            "gate": hsd_out["gate"],
            "mix_weight": hsd_out["mix_weight"],
            "actual_ratio": hsd_out["actual_ratio"],
            "avg_change_norm": hsd_out["avg_change_norm"],
            "hetero_change_norm": hsd_out["hetero_change_norm"],
        }

        if compute_aux:
            user_state_aux = user_state.detach() if self.cfg.DETACH_AUX else user_state
            item_ctx = self.item_context(batch["item_features"], batch["global_ot"])
            ui_state = torch.cat([user_state_aux, item_ctx], dim=-1)

            out["rating_pred"] = self.rating_head(ui_state).squeeze(-1)
            out["image_pred"] = self.modality_head(ui_state).squeeze(-1)
            out["pred_title_emb"] = self.title_head(ui_state)
            out["pred_text_emb"] = self.text_head(ui_state)
            out["pred_image_emb"] = self.image_head(ui_state)

        return out

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_emb(item_ids)
    
    def get_item_bias(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item bias for scoring."""
        return self.item_bias(item_ids).squeeze(-1)

    def _build_inbatch_negatives(
        self,
        user_state: torch.Tensor,
        pos_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build in-batch negatives from other samples positive items.

        Returns
        -------
        inb_scores: (B, B) scores for user_i against item_j
        inb_mask  : (B, B) True where valid negative, False for same item / pad
        """
        B = pos_items.size(0)
        if B <= 1:
            scores = torch.zeros(B, 0, device=user_state.device, dtype=user_state.dtype)
            mask = torch.zeros(B, 0, device=user_state.device, dtype=torch.bool)
            return scores, mask

        item_emb = self.item_emb(pos_items)  # (B, H)
        scores = user_state @ item_emb.t()   # (B, B)
        scores = scores + self.get_item_bias(pos_items).view(1, B)  # broadcast bias

        # valid negatives: not the same item
        mask = pos_items.unsqueeze(1) != pos_items.unsqueeze(0)

        # exclude padding item if ever present
        mask = mask & (pos_items.unsqueeze(0) != self.pad_idx)

        return scores, mask


# =============================================================================
# Loss
# =============================================================================

class HSDLoss(nn.Module):
    """Loss function with temperature BPR and hard-negative focus."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def bpr_loss(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
        neg_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Numerically-stable temperature BPR with optional hard-negative focus.

        We define x = (neg - pos) / tau.
        Pairwise logistic loss per negative is softplus(x) = log(1 + exp(x)).

        - Standard: average softplus(x) over valid negatives.
        - Hard-neg focus: weight negatives by softmax(neg / hard_tau) (masked),
          then take weighted sum of softplus(x). This emphasizes hard negatives
          while remaining stable.

        Important: invalid negatives must NEVER be turned into +1e9 inside exp().
        """
        if neg.numel() == 0:
            return torch.tensor(0.0, device=pos.device)

        tau = float(self.cfg.BPR_TEMPERATURE)

        if neg_mask is not None:
            neg_mask = neg_mask.to(dtype=torch.bool, device=neg.device)

        # x: [B, N]
        x = (neg - pos.unsqueeze(1)) / tau
        pair_loss = F.softplus(x)  # always >= 0, stable

        # Standard BPR: mean over valid negatives
        if not self.cfg.USE_HARD_NEG_FOCUS:
            if neg_mask is None:
                return pair_loss.mean()

            pair_loss = pair_loss.masked_fill(~neg_mask, 0.0)
            n_valid = neg_mask.sum()
            if n_valid == 0:
                return torch.tensor(0.0, device=pos.device)
            return pair_loss.sum() / n_valid

        # Hard negative focus (stable)
        hard_tau = float(self.cfg.HARD_NEG_TEMPERATURE)

        if neg_mask is None:
            # weights from neg scores
            w = F.softmax(neg / hard_tau, dim=1)  # [B, N]
            loss_vec = (w * pair_loss).sum(dim=1)  # [B]
            return loss_vec.mean()

        # Masked hard-neg: handle rows with 0 valid negatives
        valid_counts = neg_mask.sum(dim=1)  # [B]
        rows = valid_counts > 0
        if rows.sum() == 0:
            return torch.tensor(0.0, device=pos.device)

        neg_r = neg[rows]
        mask_r = neg_mask[rows]
        pair_r = pair_loss[rows]

        # Make invalid negatives extremely small for softmax (NOT -inf to avoid NaN softmax edge cases)
        neg_for_w = neg_r.masked_fill(~mask_r, -1e9)
        w = F.softmax(neg_for_w / hard_tau, dim=1)  # invalid -> ~0 weight

        pair_r = pair_r.masked_fill(~mask_r, 0.0)
        loss_vec = (w * pair_r).sum(dim=1)  # [rows]
        return loss_vec.mean()

    @staticmethod
    def cosine_loss_masked(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask > 0.5
        if m.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        p = F.normalize(pred[m], dim=-1)
        t = F.normalize(true[m], dim=-1)
        return (1.0 - (p * t).sum(dim=-1)).mean()

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        aux_weight_scale: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logs: Dict[str, float] = {}

        # Prefer model-produced mask (includes in-batch negatives), fallback to batch mask
        neg_mask = out.get("neg_mask", batch.get("neg_valid_mask", None))

        loss_rank = self.bpr_loss(out["pos_score"], out["neg_scores"], neg_mask)
        total = self.cfg.W_RANK * loss_rank
        logs["loss_rank"] = float(loss_rank.detach().item())

        if neg_mask is not None and neg_mask.numel() > 0:
            logs["neg_valid_rate"] = float(neg_mask.float().mean().detach().item())

        effective_w_rating = self.cfg.W_RATING * aux_weight_scale
        effective_w_modality = self.cfg.W_MODALITY * aux_weight_scale
        effective_w_content = self.cfg.W_CONTENT * aux_weight_scale

        if effective_w_rating > 0 and "rating_pred" in out:
            loss_rating = self.mse(out["rating_pred"], batch["target_rating"] / 5.0)
            total = total + effective_w_rating * loss_rating
            logs["loss_rating"] = float(loss_rating.detach().item())

        if effective_w_modality > 0 and "image_pred" in out:
            loss_mod = self.bce(out["image_pred"], batch["target_has_image"])
            total = total + effective_w_modality * loss_mod
            logs["loss_modality"] = float(loss_mod.detach().item())

        if effective_w_content > 0 and "pred_title_emb" in out:
            loss_title = self.cosine_loss_masked(out["pred_title_emb"], batch["target_title_emb"], batch["target_has_title"])
            loss_text = self.cosine_loss_masked(out["pred_text_emb"], batch["target_text_emb"], batch["target_has_text"])
            loss_image = self.cosine_loss_masked(out["pred_image_emb"], batch["target_image_emb"], batch["target_has_image"])
            loss_content = loss_title + loss_text + loss_image
            total = total + effective_w_content * loss_content
            logs["loss_content"] = float(loss_content.detach().item())

        logs["loss_total"] = float(total.detach().item())
        logs["aux_scale"] = float(aux_weight_scale)

        # monitoring
        logs["mix_weight"] = float(out["mix_weight"].detach().item())
        logs["actual_ratio"] = float(out["actual_ratio"].detach().item())
        logs["gate_mean"] = float(out["gate"].mean().detach().item())
        logs["avg_change_norm"] = float(out["avg_change_norm"].detach().item())
        logs["hetero_norm"] = float(out["hetero_change_norm"].detach().item())

        return total, logs



# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply shadow weights for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after evaluation"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# =============================================================================
# Score Calibrator
# =============================================================================

class ScoreCalibrator:
    """
    Empirical Bayes score calibration using validation set statistics.
    Corrects systematic biases across user groups and item popularity strata.
    """
    
    def __init__(
        self,
        num_items: int,
        n_groups: int = 3,
        n_pop_bins: int = 10,
        w_global: float = 0.3,
        w_group: float = 0.3,
        w_pop: float = 0.4,
    ):
        self.num_items = num_items
        self.n_groups = n_groups
        self.n_pop_bins = n_pop_bins
        self.w_global = w_global
        self.w_group = w_group
        self.w_pop = w_pop
        
        self.global_bias = 0.0
        self.global_mean_pos = 0.0
        self.group_biases: Dict[int, float] = {}
        self.popularity_biases: Dict[int, float] = {}
        self.item_to_pop_bin: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def _prepare_pop_bins(self, item_counts: np.ndarray):
        sorted_indices = np.argsort(item_counts)
        items_per_bin = len(sorted_indices) // self.n_pop_bins
        self.item_to_pop_bin = np.zeros(self.num_items, dtype=np.int32)
        for b in range(self.n_pop_bins):
            start = b * items_per_bin
            end = (b + 1) * items_per_bin if b < self.n_pop_bins - 1 else len(sorted_indices)
            self.item_to_pop_bin[sorted_indices[start:end]] = b
    
    @torch.no_grad()
    def fit(self, val_loader: DataLoader, model: nn.Module, item_counts: np.ndarray, device: torch.device):
        self._prepare_pop_bins(item_counts)
        
        records = []
        model.eval()
        
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            user_state, _ = model.encode_user(batch)
            target_items = batch["target_item"]
            user_groups = batch["user_group"]
            
            pos_emb = model.get_item_embeddings(target_items)
            pos_scores = (user_state * pos_emb).sum(dim=-1)
            if model.cfg.USE_ITEM_BIAS:
                pos_scores = pos_scores + model.get_item_bias(target_items)
            
            B = target_items.size(0)
            for i in range(B):
                item_id = target_items[i].item()
                records.append({
                    'item_id': item_id,
                    'pos_score': pos_scores[i].item(),
                    'user_group': user_groups[i].item(),
                    'pop_bin': self.item_to_pop_bin[item_id] if item_id < self.num_items else 0,
                })
        
        if len(records) == 0:
            self.is_fitted = True
            return
        
        pos_scores_list = [r['pos_score'] for r in records]
        self.global_mean_pos = np.mean(pos_scores_list)
        self.global_bias = -self.global_mean_pos * 0.1
        
        for g in range(self.n_groups):
            group_records = [r for r in records if r['user_group'] == g]
            if len(group_records) > 10:
                group_mean = np.mean([r['pos_score'] for r in group_records])
                self.group_biases[g] = (self.global_mean_pos - group_mean) * 0.5
            else:
                self.group_biases[g] = 0.0
        
        for b in range(self.n_pop_bins):
            bin_records = [r for r in records if r['pop_bin'] == b]
            if len(bin_records) > 5:
                bin_mean = np.mean([r['pos_score'] for r in bin_records])
                self.popularity_biases[b] = (self.global_mean_pos - bin_mean) * 0.8
            else:
                self.popularity_biases[b] = 0.0
        
        self.is_fitted = True
    
    def calibrate_chunk(
        self, 
        chunk_scores: torch.Tensor, 
        item_start: int,
        item_end: int,
        user_groups: torch.Tensor
    ) -> torch.Tensor:
        if not self.is_fitted:
            return chunk_scores
        
        device = chunk_scores.device
        dtype = chunk_scores.dtype
        calibrated = chunk_scores.clone()
        
        calibrated = calibrated + self.w_global * self.global_bias
        
        for g in range(self.n_groups):
            g_mask = (user_groups == g)
            if g_mask.any():
                calibrated[g_mask] = calibrated[g_mask] + self.w_group * self.group_biases.get(g, 0.0)
        
        pop_biases = torch.tensor(
            [self.popularity_biases.get(self.item_to_pop_bin[i], 0.0) for i in range(item_start, item_end)],
            device=device, dtype=dtype
        )
        calibrated = calibrated + self.w_pop * pop_biases.unsqueeze(0)
        
        return calibrated

# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(self, cfg: Config, model: HSDModel, train_loader: DataLoader,
                 val_loader: DataLoader, test_loader: DataLoader, item_counts: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = HSDLoss(cfg)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

        if cfg.USE_SCHEDULER:
            self.scheduler = CosineAnnealingLR(self.opt, T_max=cfg.SCHEDULER_T_MAX, eta_min=cfg.SCHEDULER_ETA_MIN)
        else:
            self.scheduler = None
        
        self.base_lr = cfg.LR
        self.warmup_epochs = cfg.WARMUP_EPOCHS
        
        self.initial_update_ratio = cfg.MAX_UPDATE_RATIO
        self.final_update_ratio = cfg.MAX_UPDATE_RATIO_FINAL
        self.update_ratio_warmup = cfg.UPDATE_RATIO_WARMUP_EPOCHS

        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.AMP and self.device.type == "cuda"))
        self.best_val = -1e9
        self.patience_counter = 0
        
        if cfg.USE_EMA:
            self.ema = EMA(self.model, decay=cfg.EMA_DECAY)
        else:
            self.ema = None
        
        self.calibrator: Optional[ScoreCalibrator] = None
        self.item_counts = item_counts
        if cfg.USE_SCORE_CALIBRATION and item_counts is not None:
            self.calibrator = ScoreCalibrator(
                num_items=model.num_items,
                n_groups=cfg.N_GROUPS,
                n_pop_bins=cfg.CALIBRATION_N_POP_BINS,
                w_global=cfg.CALIBRATION_W_GLOBAL,
                w_group=cfg.CALIBRATION_W_GROUP,
                w_pop=cfg.CALIBRATION_W_POP,
            )

    def _to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _get_aux_weight_scale(self, epoch: int) -> float:
        warmup = self.cfg.AUX_WARMUP_EPOCHS
        rampup = self.cfg.AUX_RAMPUP_EPOCHS

        if epoch <= warmup:
            return 0.0
        elif epoch <= warmup + rampup:
            return (epoch - warmup) / rampup
        else:
            return 1.0
    
    def _adjust_lr_warmup(self, epoch: int):
        """Adjust learning rate for warmup epochs."""
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            current_lr = self.base_lr * warmup_factor
            for param_group in self.opt.param_groups:
                param_group['lr'] = current_lr
    
    def _adjust_update_ratio(self, epoch: int):
        """Dynamically adjust MAX_UPDATE_RATIO."""
        if epoch <= self.update_ratio_warmup:
            new_ratio = self.initial_update_ratio
        else:
            progress = min(1.0, (epoch - self.update_ratio_warmup) / 
                          (self.cfg.EPOCHS - self.update_ratio_warmup))
            new_ratio = self.initial_update_ratio + progress * (
                self.final_update_ratio - self.initial_update_ratio
            )
        
        self.model.hsd_updater.cfg.MAX_UPDATE_RATIO = new_ratio
        return new_ratio

    def _fit_calibrator(self):
        if self.calibrator is None or self.item_counts is None:
            return
        if self.calibrator.is_fitted:
            return
        
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.calibrator.fit(self.val_loader, self.model, self.item_counts, self.device)
        
        if self.ema is not None:
            self.ema.restore()

    @torch.no_grad()
    def evaluate_sampled(self, loader: DataLoader, num_neg: int, max_batches: int) -> Dict[str, float]:
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        all_ranks = []
        all_neg_valid_rates = []

        for bi, batch in enumerate(tqdm(loader, desc="Eval (sampled)", leave=False)):
            if bi >= max_batches:
                break

            b = self._to_device(batch)
            user_state, _ = self.model.encode_user(b)

            pos_items = b["target_item"]
            pos_vec = self.model.get_item_embeddings(pos_items)
            pos_scores = (user_state * pos_vec).sum(dim=-1)
            pos_scores = pos_scores + self.model.get_item_bias(pos_items)

            pos_scores = torch.where(torch.isnan(pos_scores), torch.full_like(pos_scores, float("-inf")), pos_scores)

            B = pos_items.size(0)
            sample_min = 1 if self.model.pad_idx == 0 else 0
            sample_size = num_neg * 5

            neg_candidates = torch.randint(sample_min, self.model.num_items, (B, sample_size), device=self.device)

            valid = torch.ones(B, sample_size, dtype=torch.bool, device=self.device)
            valid &= (neg_candidates != pos_items.unsqueeze(1))
            valid &= (neg_candidates != self.model.pad_idx)

            seen_items = b["user_seen_items"]
            seen_mask = b["user_seen_mask"]

            if seen_mask.any():
                S = seen_items.size(1)
                SEEN_BLOCK = self.cfg.SEEN_BLOCK_SIZE
                is_seen_any = torch.zeros(B, sample_size, dtype=torch.bool, device=self.device)

                for blk_start in range(0, S, SEEN_BLOCK):
                    blk_end = min(blk_start + SEEN_BLOCK, S)
                    seen_block = seen_items[:, blk_start:blk_end]
                    mask_block = seen_mask[:, blk_start:blk_end]

                    if not mask_block.any():
                        continue

                    is_match = (neg_candidates.unsqueeze(2) == seen_block.unsqueeze(1)) & mask_block.unsqueeze(1)
                    is_seen_any |= is_match.any(dim=2)

                valid &= ~is_seen_any

            sort_key = torch.where(
                valid,
                torch.arange(sample_size, device=self.device).unsqueeze(0).expand(B, -1),
                torch.full((B, sample_size), sample_size + 1, device=self.device),
            )
            sorted_indices = sort_key.argsort(dim=1)[:, :num_neg]
            neg_items = neg_candidates.gather(1, sorted_indices)
            neg_valid = valid.gather(1, sorted_indices)

            neg_vec = self.model.get_item_embeddings(neg_items)
            neg_scores = torch.einsum("bh,bnh->bn", user_state, neg_vec)
            neg_scores = neg_scores + self.model.get_item_bias(neg_items)

            neg_scores = torch.where(torch.isnan(neg_scores), torch.full_like(neg_scores, float("-inf")), neg_scores)

            neg_scores = neg_scores.masked_fill(~neg_valid, float("-inf"))

            scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            rank = (scores[:, 1:] > scores[:, :1]).sum(dim=1) + 1
            all_ranks.extend(rank.detach().cpu().tolist())

            all_neg_valid_rates.append(neg_valid.float().mean().item())

        if self.ema is not None:
            self.ema.restore()

        metrics = self._metrics_from_ranks(all_ranks)
        metrics["neg_valid_rate_eval"] = float(np.mean(all_neg_valid_rates)) if all_neg_valid_rates else 1.0
        return metrics

    @torch.no_grad()
    def evaluate_full_sort(self, loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        all_ranks = []

        num_items = self.model.num_items
        pad_idx = self.model.pad_idx
        chunk_size = self.cfg.FULL_SORT_CHUNK_SIZE
        
        use_calibration = (self.calibrator is not None and self.calibrator.is_fitted)

        for bi, batch in enumerate(tqdm(loader, desc="Eval (full-sort)", leave=False)):
            if max_batches is not None and bi >= max_batches:
                break

            b = self._to_device(batch)
            user_state, _ = self.model.encode_user(b)

            pos_items = b["target_item"]
            pos_vec = self.model.get_item_embeddings(pos_items)
            pos_scores = (user_state * pos_vec).sum(dim=-1)
            pos_scores = pos_scores + self.model.get_item_bias(pos_items)

            pos_scores = torch.where(torch.isnan(pos_scores), torch.full_like(pos_scores, float("-inf")), pos_scores)

            B = pos_items.size(0)
            seen_items = b["user_seen_items"]
            seen_mask = b["user_seen_mask"]
            user_groups = b["user_group"]

            higher_count = torch.zeros(B, dtype=torch.long, device=self.device)

            for start in range(0, num_items, chunk_size):
                end = min(start + chunk_size, num_items)
                item_ids = torch.arange(start, end, device=self.device)
                item_vecs = self.model.get_item_embeddings(item_ids)
                chunk_scores = user_state @ item_vecs.t()
                chunk_bias = self.model.get_item_bias(item_ids)
                chunk_scores = chunk_scores + chunk_bias.unsqueeze(0)

                if use_calibration:
                    chunk_scores = self.calibrator.calibrate_chunk(chunk_scores, start, end, user_groups)

                chunk_scores = torch.where(
                    torch.isnan(chunk_scores),
                    torch.full_like(chunk_scores, float("-inf")),
                    chunk_scores
                )

                exclude_mask = torch.zeros(B, end - start, dtype=torch.bool, device=self.device)

                if start <= pad_idx < end:
                    exclude_mask[:, pad_idx - start] = True

                pos_in_chunk = (pos_items >= start) & (pos_items < end)
                if pos_in_chunk.any():
                    batch_indices = torch.where(pos_in_chunk)[0]
                    chunk_indices = pos_items[pos_in_chunk] - start
                    exclude_mask[batch_indices, chunk_indices] = True

                seen_in_range = (seen_items >= start) & (seen_items < end) & seen_mask
                if seen_in_range.any():
                    batch_indices, s_indices = torch.where(seen_in_range)
                    chunk_indices = seen_items[batch_indices, s_indices] - start
                    exclude_mask[batch_indices, chunk_indices] = True

                chunk_scores[exclude_mask] = float("-inf")
                higher_in_chunk = (chunk_scores > pos_scores.unsqueeze(1)).sum(dim=1)
                higher_count += higher_in_chunk

            ranks = (higher_count + 1).cpu().tolist()
            all_ranks.extend(ranks)

        if self.ema is not None:
            self.ema.restore()

        return self._metrics_from_ranks(all_ranks)

    def _metrics_from_ranks(self, ranks: List[int]) -> Dict[str, float]:
        r = np.array(ranks, dtype=np.int64)
        out = {}
        for K in self.cfg.K_VALUES:
            hits = (r <= K).astype(np.float64)
            out[f"Recall@{K}"] = float(hits.mean())
            out[f"NDCG@{K}"] = float(np.where(r <= K, 1.0 / np.log2(r.astype(np.float64) + 1.0), 0.0).mean())
        out["MRR"] = float((1.0 / r.astype(np.float64)).mean())
        return out

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self._adjust_lr_warmup(epoch)
        
        current_update_ratio = self._adjust_update_ratio(epoch)

        self.model.train()
        logs_accum: Dict[str, float] = {}
        n_batches = 0

        aux_scale = self._get_aux_weight_scale(epoch)
        compute_aux = aux_scale > 0

        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            b = self._to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                out = self.model(b, compute_aux=compute_aux)
                loss, logs = self.loss_fn(out, b, aux_weight_scale=aux_scale)

            if not torch.isfinite(loss).item():
                print(f"\n[WARN] Non-finite loss detected at epoch={epoch}, step={step}. "
                      f"loss={loss.detach().item()}. Skipping this batch to prevent NaN propagation.")
                self.opt.zero_grad(set_to_none=True)
                continue

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.cfg.GRAD_CLIP > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)

            self.scaler.step(self.opt)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update()

            for k, v in logs.items():
                logs_accum[k] = logs_accum.get(k, 0.0) + float(v)
            n_batches += 1

            postfix = {
                "loss": f"{logs.get('loss_total', 0.0):.4f}",
                "ratio": f"{logs.get('actual_ratio', 0.0):.3f}",
                "mix": f"{logs.get('mix_weight', 0.0):.3f}",
            }
            pbar.set_postfix(postfix)

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: v / max(1, n_batches) for k, v in logs_accum.items()}

    def train(self):
        os.makedirs(self.cfg.RESULTS_DIR, exist_ok=True)

        print(f"\n{'=' * 70}")
        print("Training")
        print(f"{'=' * 70}")
        print(f"MAX_UPDATE_RATIO: {self.cfg.MAX_UPDATE_RATIO}")
        print(f"MIX_LOGIT_INIT: {self.cfg.MIX_LOGIT_INIT} -> w~{torch.sigmoid(torch.tensor(self.cfg.MIX_LOGIT_INIT)).item():.3f}")
        print(f"BPR_TEMPERATURE: {self.cfg.BPR_TEMPERATURE}")
        print(f"USE_HARD_NEG_FOCUS: {self.cfg.USE_HARD_NEG_FOCUS}")
        print(f"UNIFORM_NEG_RATIO: {self.cfg.UNIFORM_NEG_RATIO}")
        print(f"USE_EMA: {self.cfg.USE_EMA} (decay={self.cfg.EMA_DECAY})")
        print(f"DEVIATION_DIM: {self.cfg.DEVIATION_DIM}")
        print(f"NUM_LAYERS: {self.cfg.NUM_LAYERS}, MAX_USER_SEQ_LEN: {self.cfg.MAX_USER_SEQ_LEN}")

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            train_logs = self.train_epoch(epoch)
            dt = time.time() - t0

            lr = self.opt.param_groups[0]['lr']
            aux_scale = self._get_aux_weight_scale(epoch)

            current_max_ratio = self.model.hsd_updater.cfg.MAX_UPDATE_RATIO
            print(f"\n[Epoch {epoch}] time={dt:.1f}s lr={lr:.2e} aux_scale={aux_scale:.2f} max_ratio={current_max_ratio:.3f}")
            print(f"  mix_w={train_logs.get('mix_weight', 0):.4f} "
                  f"ratio={train_logs.get('actual_ratio', 0):.4f} "
                  f"gate={train_logs.get('gate_mean', 0):.4f}")
            print(f"  avg_norm={train_logs.get('avg_change_norm', 0):.2f} "
                  f"hetero_norm={train_logs.get('hetero_norm', 0):.2f}")
            print(f"  Losses: " + " ".join([f"{k}={v:.4f}" for k, v in train_logs.items() if k.startswith("loss")]))

            if epoch % self.cfg.VAL_EVERY == 0:
                if epoch % self.cfg.FULL_SORT_VAL_EVERY == 0:
                    val_full = self.evaluate_full_sort(self.val_loader, max_batches=self.cfg.FULL_SORT_VAL_BATCHES)
                    print(f"  Val (full-sort, {self.cfg.FULL_SORT_VAL_BATCHES} batches): " + 
                          " ".join([f"{k}={v:.4f}" for k, v in val_full.items()]))
                    
                    score = val_full.get("Recall@20", 0.0)
                else:
                    val_metrics = self.evaluate_sampled(self.val_loader, self.cfg.SAMPLED_EVAL_NEGATIVES, self.cfg.VAL_MAX_BATCHES)
                    neg_valid_rate = val_metrics.pop("neg_valid_rate_eval", 1.0)
                    print(f"  Val (sampled): " + " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]) + 
                          f" neg_valid={neg_valid_rate:.3f}")
                    score = val_metrics.get("Recall@20", 0.0)

                if score > self.best_val:
                    self.best_val = score
                    self.patience_counter = 0

                    if self.ema is not None:
                        self.ema.apply_shadow()

                    torch.save({
                        "model": self.model.state_dict(),
                        "epoch": epoch,
                        "best_val": self.best_val,
                        "cfg": vars(self.cfg),
                    }, self.cfg.BEST_CKPT_PATH)

                    if self.ema is not None:
                        self.ema.restore()

                    print(f"  Saved best (Recall@20={score:.4f})")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement ({self.patience_counter}/{self.cfg.EARLY_STOP_PATIENCE})")

                    if self.patience_counter >= self.cfg.EARLY_STOP_PATIENCE:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

        self._final_test()

    def _final_test(self):
        print("\n" + "=" * 60)
        print("Final Test (Full-Sort)")
        print("=" * 60)

        if os.path.exists(self.cfg.BEST_CKPT_PATH):
            ckpt = torch.load(self.cfg.BEST_CKPT_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded best (epoch={ckpt['epoch']}, val={ckpt['best_val']:.4f})")

        self._fit_calibrator()

        print("\n[Sampled Evaluation]")
        test_sampled = self.evaluate_sampled(self.test_loader, self.cfg.SAMPLED_EVAL_NEGATIVES, 10000)
        neg_valid_rate = test_sampled.pop("neg_valid_rate_eval", 1.0)
        print(f"Test (sampled): " + " ".join([f"{k}={v:.4f}" for k, v in test_sampled.items()]) + 
              f" neg_valid={neg_valid_rate:.3f}")

        print("\n[Full-Sort Evaluation]")
        test_full = self.evaluate_full_sort(self.test_loader, max_batches=None)
        print(f"Test (full-sort): " + " ".join([f"{k}={v:.4f}" for k, v in test_full.items()]))

        with open(os.path.join(self.cfg.RESULTS_DIR, "test_results.json"), "w") as f:
            json.dump({
                "test_sampled": test_sampled,
                "test_full_sort": test_full,
                "best_val": self.best_val,
                "neg_valid_rate_eval": neg_valid_rate,
                "config": {
                    "MAX_UPDATE_RATIO_INIT": self.initial_update_ratio,
                    "MAX_UPDATE_RATIO_FINAL": self.final_update_ratio,
                    "MIX_LOGIT_INIT": self.cfg.MIX_LOGIT_INIT,
                    "HIDDEN_DIM": self.cfg.HIDDEN_DIM,
                    "NUM_LAYERS": self.cfg.NUM_LAYERS,
                    "MAX_USER_SEQ_LEN": self.cfg.MAX_USER_SEQ_LEN,
                    "DEVIATION_DIM": self.cfg.DEVIATION_DIM,
                    "BPR_TEMPERATURE": self.cfg.BPR_TEMPERATURE,
                    "USE_HARD_NEG_FOCUS": self.cfg.USE_HARD_NEG_FOCUS,
                    "UNIFORM_NEG_RATIO": self.cfg.UNIFORM_NEG_RATIO,
                    "USE_EMA": self.cfg.USE_EMA,
                    "EMA_DECAY": self.cfg.EMA_DECAY,
                }
            }, f, indent=2)

        print(f"\nResults saved to {self.cfg.RESULTS_DIR}/test_results.json")


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()
    set_seed(cfg.SEED)

    print("=" * 70)
    print("HSD Sequential Recommendation")
    print("=" * 70)

    print("\n[1/4] Loading shared resources...")
    shared = SharedResources(cfg.DATA_DIR)

    cfg.N_GROUPS = shared.n_groups
    cfg.NUM_BINS = shared.num_bins
    cfg.PAD_IDX = shared.pad_idx

    print(f"\nConfig from data:")
    print(f"  num_items: {shared.num_items}")
    print(f"  num_bins: {cfg.NUM_BINS}")
    print(f"  n_groups: {cfg.N_GROUPS}")
    print(f"  pad_idx: {cfg.PAD_IDX}")

    print("\n[2/4] Creating datasets...")
    train_ds = HSDDataset(shared, "train", cfg.MAX_USER_SEQ_LEN, cfg.NUM_NEGATIVES_TRAIN, return_user_seen=False, cfg=cfg)
    val_ds = HSDDataset(shared, "val", cfg.MAX_USER_SEQ_LEN, 0, return_user_seen=True, cfg=cfg)
    test_ds = HSDDataset(shared, "test", cfg.MAX_USER_SEQ_LEN, 0, return_user_seen=True, cfg=cfg)

    print(f"  Train: {len(train_ds):,}")
    print(f"  Val: {len(val_ds):,}")
    print(f"  Test: {len(test_ds):,}")

    loader_kwargs = {
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": cfg.PIN_MEMORY,
        "collate_fn": collate_fn,
        "drop_last": False,
    }
    if cfg.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, **loader_kwargs)

    print("\n[3/4] Creating model...")
    model = HSDModel(shared.num_items, cfg.PAD_IDX, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print("\n[4/4] Training...")
    item_counts = shared.item_counts if hasattr(shared, 'item_counts') else None
    trainer = Trainer(cfg, model, train_loader, val_loader, test_loader, item_counts=item_counts)
    trainer.train()

    print("\nComplete!")


if __name__ == "__main__":
    main()
