import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.preprocessing import convert_to_bin

def check_and_convert_dataset(dataset_name: str) -> None:
    bin_dir = Path(f'bins/{dataset_name}_bin')
    if not bin_dir.exists() or not (bin_dir / 'all_items.npy').exists() or not (bin_dir / 'neg_pool.npy').exists():
        print(f"Binary data not found for {dataset_name}. Running conversion...")
        convert_to_bin(dataset_name)
        print("Conversion complete.")

def load_metadata(dataset_name: str) -> Tuple[int, int]:
    bin_dir = Path(f'bins/{dataset_name}_bin')
    meta_path = bin_dir / 'meta.txt'
    with open(meta_path, 'r') as f:
        usernum, itemnum = map(int, f.read().strip().split(','))
    return usernum, itemnum

def build_index(dataset_name: str) -> Tuple[List[List[int]], List[List[int]]]:
    ui_mat = np.loadtxt(f'data/{dataset_name}.txt', dtype=np.int32)
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]
    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])
    return u2i_index, i2u_index

def data_partition(fname: str) -> Tuple[Dict, Dict, Dict, int, int]:
    from collections import defaultdict
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    with open(f'data/{fname}.txt', 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    
    return user_train, user_valid, user_test, usernum, itemnum

class SASRecDataset(Dataset):
    def __init__(self, dataset_name: str, maxlen: int, mode: str = 'train', num_negatives: int = 1, neg_sampling_mode: str = 'random') -> None:
        self.dataset_name: str = dataset_name
        self.maxlen: int = maxlen
        self.mode: str = mode
        self.num_negatives: int = num_negatives
        self.neg_sampling_mode: str = neg_sampling_mode
        
        bin_dir = Path(f'bins/{dataset_name}_bin')
        self.all_items = np.load(bin_dir / 'all_items.npy', mmap_mode='r')
        self.user_index = np.load(bin_dir / 'user_index.npy', mmap_mode='r')
        self.neg_pool = np.load(bin_dir / 'neg_pool.npy', mmap_mode='r')
        self.usernum, self.itemnum = load_metadata(dataset_name)
        
        self.valid_users = np.where(self.user_index[:, 1] > 0)[0]
        if mode == 'train':
            self.valid_users = self.valid_users[self.user_index[self.valid_users, 1] >= 4]
        
        self.neg_pool_size = self.neg_pool.shape[1]
        self.neg_counters = np.zeros(len(self.user_index), dtype=np.int32)
    
    def __len__(self) -> int:
        return len(self.valid_users)
    
    def _get_user_sequence(self, uid: int) -> np.ndarray:
        offset, length = self.user_index[uid]
        return np.array(self.all_items[offset:offset + length], dtype=np.int32)
    
    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        uid = self.valid_users[idx]
        full_seq = self._get_user_sequence(uid)
        
        if self.mode == 'train':
            seq_items = full_seq[:-2] if len(full_seq) >= 4 else full_seq
        elif self.mode == 'valid':
            seq_items = full_seq[:-1]
        else:
            seq_items = full_seq

        seq_len = len(seq_items)
        if seq_len <= 1:
             return (
                 uid,
                 np.zeros(self.maxlen, dtype=np.int32),
                 np.zeros(self.maxlen, dtype=np.int32),
                 np.zeros((self.maxlen, self.num_negatives), dtype=np.int32),
             )
        
        input_seq = seq_items[:-1]
        target_pos = seq_items[1:]
        
        eff_len = min(len(input_seq), self.maxlen)
        
        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        
        seq[-eff_len:] = input_seq[-eff_len:]
        pos[-eff_len:] = target_pos[-eff_len:]
        
        num_samples = eff_len * self.num_negatives
        start_idx = self.neg_counters[uid]
        end_idx = start_idx + num_samples
        
        if end_idx <= self.neg_pool_size:
            neg_flat = self.neg_pool[uid, start_idx:end_idx].astype(np.int32, copy=False)
            self.neg_counters[uid] = end_idx
        else:
            wrap_needed = end_idx - self.neg_pool_size
            neg_flat = np.empty(num_samples, dtype=np.int32)
            remaining = self.neg_pool_size - start_idx
            neg_flat[:remaining] = self.neg_pool[uid, start_idx:]
            neg_flat[remaining:] = self.neg_pool[uid, :wrap_needed]
            self.neg_counters[uid] = wrap_needed
        
        neg_data = neg_flat.reshape(eff_len, self.num_negatives)
        
        neg = np.zeros((self.maxlen, self.num_negatives), dtype=np.int32)
        neg[-eff_len:, :] = neg_data
        
        return uid, seq, pos, neg

def get_dataloader(dataset_name: str, maxlen: int, batch_size: int, mode: str = 'train', num_workers: int = 4, num_negatives: int = 1, neg_sampling_mode: str = 'random') -> DataLoader:
    dataset = SASRecDataset(dataset_name, maxlen, mode, num_negatives, neg_sampling_mode)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=(mode == 'train'),
        persistent_workers=(True if num_workers > 0 else False)
    )

def _precompute_eval_negatives(dataset: Tuple, mode: str, device: str, num_negs: int = 100) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    train, valid, test, usernum, itemnum = dataset
    
    if usernum > 10000:
        eval_users = random.sample(range(1, usernum + 1), 10000)
    else:
        eval_users = list(range(1, usernum + 1))
    
    valid_mask = []
    for u in eval_users:
        if mode == 'test':
            valid_mask.append(len(train[u]) >= 1 and len(test[u]) >= 1)
        else:
            valid_mask.append(len(train[u]) >= 1 and len(valid[u]) >= 1)
    
    eval_users = [u for u, mask in zip(eval_users, valid_mask) if mask]
    num_users = len(eval_users)
    
    all_items = torch.arange(1, itemnum + 1, dtype=torch.int32)
    neg_items = torch.zeros((num_users, num_negs), dtype=torch.int32)
    
    for idx, u in enumerate(eval_users):
        rated = set(train[u])
        rated.add(0)
        valid_items_mask = torch.ones(itemnum, dtype=torch.bool)
        for item in rated:
            if 1 <= item <= itemnum:
                valid_items_mask[item - 1] = False
        valid_items = all_items[valid_items_mask]
        if len(valid_items) < num_negs:
            sampled = valid_items[torch.randint(len(valid_items), (num_negs,))]
        else:
            sampled = valid_items[torch.randperm(len(valid_items))[:num_negs]]
        neg_items[idx] = sampled
    
    return torch.tensor(eval_users, dtype=torch.int32), neg_items.to(device), None, None

def _batch_evaluate_logic(model, dataset, args: Any, mode: str = 'test', precomputed_negs: Optional[Tuple] = None) -> Tuple[float, float]:
    train, valid, test, _, _ = dataset
    
    if precomputed_negs is None:
        precomputed_negs = _precompute_eval_negatives(dataset, mode, args.device)
    
    eval_users, neg_items, _, _ = precomputed_negs
    num_users = len(eval_users)
    eval_batch_size = 100
    
    all_ranks = []
    
    for batch_start in range(0, num_users, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, num_users)
        batch_size = batch_end - batch_start
        
        batch_u_ids = eval_users[batch_start:batch_end].cpu().numpy()
        batch_seqs = np.zeros((batch_size, args.maxlen), dtype=np.int32)
        batch_targets = np.zeros(batch_size, dtype=np.int32)
        
        for i, u in enumerate(batch_u_ids):
            idx = args.maxlen - 1
            if mode == 'test':
                batch_seqs[i, idx] = valid[u][0]
                idx -= 1
                source_seq = train[u]
                batch_targets[i] = test[u][0]
            else:
                source_seq = train[u]
                batch_targets[i] = valid[u][0]
            
            for item in reversed(source_seq):
                batch_seqs[i, idx] = item
                idx -= 1
                if idx == -1:
                    break
        
        batch_negs = neg_items[batch_start:batch_end]
        batch_items = torch.cat([torch.tensor(batch_targets, device=args.device).unsqueeze(1), batch_negs], dim=1)
        
        predictions = model.predict(batch_u_ids, batch_seqs, batch_items.cpu().numpy())
        
        target_scores = predictions[:, 0]
        ranks = (predictions > target_scores.unsqueeze(1)).sum(dim=1)
        all_ranks.append(ranks)
    
    all_ranks = torch.cat(all_ranks, dim=0).float()
    hits = (all_ranks < 10).float()
    ndcgs = hits / torch.log2(all_ranks.float() + 2.0)
    
    return ndcgs.mean().item(), hits.mean().item()

_cached_test_negs = {}
_cached_valid_negs = {}

def evaluate(model, dataset: Tuple, args: Any) -> Tuple[float, float]:
    cache_key = args.dataset
    if cache_key not in _cached_test_negs:
        _cached_test_negs[cache_key] = _precompute_eval_negatives(dataset, 'test', args.device)
    return _batch_evaluate_logic(model, dataset, args, mode='test', precomputed_negs=_cached_test_negs[cache_key])

def evaluate_valid(model, dataset: Tuple, args: Any) -> Tuple[float, float]:
    cache_key = args.dataset
    if cache_key not in _cached_valid_negs:
        _cached_valid_negs[cache_key] = _precompute_eval_negatives(dataset, 'valid', args.device)
    return _batch_evaluate_logic(model, dataset, args, mode='valid', precomputed_negs=_cached_valid_negs[cache_key])