import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from torch.utils.data import Dataset, DataLoader

def check_and_convert_dataset(dataset_name: str) -> None:
    bin_dir = Path(f'bins/{dataset_name}_bin')
    if not bin_dir.exists() or not (bin_dir / 'all_items.npy').exists():
        print(f"Binary data not found for {dataset_name}. Running conversion...")
        from preprocess import convert_to_bin
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
    def __init__(self, dataset_name: str, maxlen: int, mode: str = 'train', num_negatives: int = 1):
        self.dataset_name = dataset_name
        self.maxlen = maxlen
        self.mode = mode
        self.num_negatives = max(1, int(num_negatives))
        
        bin_dir = Path(f'bins/{dataset_name}_bin')
        self.all_items = np.load(bin_dir / 'all_items.npy', mmap_mode='r')
        self.user_index = np.load(bin_dir / 'user_index.npy', mmap_mode='r')
        self.usernum, self.itemnum = load_metadata(dataset_name)
        
        self.valid_users = np.where(self.user_index[:, 1] > 0)[0]
        if mode == 'train':
            self.valid_users = self.valid_users[self.user_index[self.valid_users, 1] >= 4]
    
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
        
        ts = set(seq_items)
        num_samples = eff_len * self.num_negatives
        
        samples = np.random.randint(1, self.itemnum + 1, size=int(num_samples * 1.2), dtype=np.int32)
        
        valid_samples = [s for s in samples if s not in ts]
        
        while len(valid_samples) < num_samples:
            t = np.random.randint(1, self.itemnum + 1)
            while t in ts:
                t = np.random.randint(1, self.itemnum + 1)
            valid_samples.append(t)
            
        neg_data = np.array(valid_samples[:num_samples], dtype=np.int32).reshape(eff_len, self.num_negatives)
        
        neg = np.zeros((self.maxlen, self.num_negatives), dtype=np.int32)
        neg[-eff_len:, :] = neg_data
        
        return uid, seq, pos, neg

def get_dataloader(dataset_name, maxlen, batch_size, mode='train', num_workers=4, num_negatives: int = 1):
    dataset = SASRecDataset(dataset_name, maxlen, mode, num_negatives=num_negatives)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(mode == 'train'),
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(True if num_workers > 0 else False)
    )

def _batch_evaluate_logic(model, dataset, args, mode='test'):
    [train, valid, test, usernum, itemnum] = dataset
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = list(range(1, usernum + 1))
    eval_batch_size = 100 
    
    for start_idx in range(0, len(users), eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, len(users))
        batch_users = users[start_idx:end_idx]
        batch_u_ids = []
        batch_seqs = []
        batch_item_indices = []
        for i, u in enumerate(batch_users):
            if len(train[u]) < 1 or (mode == 'test' and len(test[u]) < 1) or (mode == 'valid' and len(valid[u]) < 1):
                continue
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            if mode == 'test':
                seq[idx] = valid[u][0]
                idx -= 1
                source_seq = train[u]
            else:
                source_seq = train[u]
            for item in reversed(source_seq):
                seq[idx] = item
                idx -= 1
                if idx == -1: break
            rated = set(train[u])
            rated.add(0)
            if mode == 'test':
                target_item = test[u][0]
            else:
                target_item = valid[u][0]
            item_idx = [target_item]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            batch_u_ids.append(u)
            batch_seqs.append(seq)
            batch_item_indices.append(item_idx)
        if len(batch_u_ids) == 0:
            continue
        np_u_ids = np.array(batch_u_ids)
        np_seqs = np.array(batch_seqs)
        np_items = np.array(batch_item_indices)
        predictions = model.predict(np_u_ids, np_seqs, np_items)
        target_scores = predictions[:, 0]
        ranks = (predictions > target_scores.unsqueeze(1)).sum(dim=1)
        ranks = ranks.cpu().numpy()
        valid_user += len(ranks)
        hits = (ranks < 10).astype(np.float32)
        ndcgs = hits * (1.0 / np.log2(ranks + 2.0))
        HT += hits.sum()
        NDCG += ndcgs.sum()
    return NDCG / valid_user, HT / valid_user

def evaluate(model, dataset: Tuple, args) -> Tuple[float, float]:
    return _batch_evaluate_logic(model, dataset, args, mode='test')

def evaluate_valid(model, dataset: Tuple, args) -> Tuple[float, float]:
    return _batch_evaluate_logic(model, dataset, args, mode='valid')