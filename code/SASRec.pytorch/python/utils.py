import sys
import copy
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


class SASRecDataset(Dataset):
    def __init__(
        self, 
        dataset_name: str, 
        maxlen: int, 
        mode: str = 'train'
    ):
        self.dataset_name = dataset_name
        self.maxlen = maxlen
        self.mode = mode
        
        bin_dir = Path(f'bins/{dataset_name}_bin')
        
        # CRITICAL: Load with mmap_mode='r' for disk streaming
        self.all_items = np.load(bin_dir / 'all_items.npy', mmap_mode='r')
        self.user_index = np.load(bin_dir / 'user_index.npy', mmap_mode='r')
        
        self.usernum, self.itemnum = load_metadata(dataset_name)
        
        # Get valid users (those with at least 1 interaction)
        self.valid_users = np.where(self.user_index[:, 1] > 0)[0]
        
        # For train mode, filter users with at least 4 interactions (need train/val/test split)
        if mode == 'train':
            self.valid_users = self.valid_users[self.user_index[self.valid_users, 1] >= 4]
        
        # Shuffle users for training
        if mode == 'train':
            np.random.shuffle(self.valid_users)
    
    def __len__(self) -> int:
        return len(self.valid_users)
    
    def _get_user_sequence(self, uid: int) -> np.ndarray:
        offset, length = self.user_index[uid]
        return np.array(self.all_items[offset:offset + length], dtype=np.int32)
    
    def _random_neq(self, l: int, r: int, s: set) -> int:
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t
    
    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        uid = self.valid_users[idx]
        full_seq = self._get_user_sequence(uid)
        
        # Split sequence based on mode
        if self.mode == 'train':
            if len(full_seq) >= 4:
                seq_items = full_seq[:-2]  # Exclude last 2 for val/test
            else:
                seq_items = full_seq
        elif self.mode == 'valid':
            seq_items = full_seq[:-1]  # Exclude last 1 for test
        else:  # test
            seq_items = full_seq
        
        # Skip users with insufficient training data
        if len(seq_items) <= 1:
            # Return empty sequences (will be filtered by model)
            return (uid, 
                    np.zeros(self.maxlen, dtype=np.int32),
                    np.zeros(self.maxlen, dtype=np.int32),
                    np.zeros(self.maxlen, dtype=np.int32))
        
        # Initialize output arrays
        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        neg = np.zeros(self.maxlen, dtype=np.int32)
        
        # Build training sequences (input: [:-1], target: [1:])
        nxt = seq_items[-1]
        idx = self.maxlen - 1
        ts = set(seq_items)
        
        # Fill sequences from right to left
        for i in reversed(seq_items[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = self._random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        return uid, seq, pos, neg


def get_dataloader(
    dataset_name: str, 
    maxlen: int,
    batch_size: int,
    mode: str = 'train',
    num_workers: int = 4
) -> DataLoader:
    dataset = SASRecDataset(dataset_name, maxlen, mode)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


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
    
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset: List, args) -> Tuple[float, float]:
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        
        rank = predictions.argsort().argsort()[0].item()
        
        valid_user += 1
        
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    
    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset: List, args) -> Tuple[float, float]:
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        
        rank = predictions.argsort().argsort()[0].item()
        
        valid_user += 1
        
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    
    return NDCG / valid_user, HT / valid_user

