import random
from pathlib import Path
from typing import Tuple, Dict, List, Any
from multiprocessing import Process, Queue

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.preprocessing import convert_to_bin


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1
        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(User, usernum, itemnum, batch_size, maxlen,
                          self.result_queue, np.random.randint(2e9)),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def check_and_convert_dataset(dataset_name: str) -> None:
    bin_dir = Path(f'bins/{dataset_name}_bin')
    if not bin_dir.exists() or not (bin_dir / 'all_items.npy').exists() or not (bin_dir / 'user_index.npy').exists():
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
    bin_dir = Path(f'bins/{fname}_bin')
    data_path = bin_dir / 'data.txt'

    # Fallback to raw file nếu chưa có bin (chạy convert trước)
    if not data_path.exists():
        data_path = Path(f'data/{fname}.txt')

    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train, user_valid, user_test = {}, {}, {}

    with open(data_path, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u, i = int(u), int(i)
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
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]

    return user_train, user_valid, user_test, usernum, itemnum

class SASRecDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        maxlen: int,
        mode: str = 'train',
        num_negatives: int = 1,
        neg_sampling_mode: str = 'random',
        mem_maxlen: int = 0,
        use_duorec: bool = False,
    ) -> None:
        self.dataset_name: str = dataset_name
        self.maxlen: int = maxlen
        self.mem_maxlen: int = mem_maxlen  # extended sequence length for memory (0=disabled)
        self.mode: str = mode
        self.num_negatives: int = num_negatives
        self.neg_sampling_mode: str = neg_sampling_mode
        # DuoRec semantic-augmentation only makes sense in training mode.
        self.use_duorec: bool = bool(use_duorec) and mode == 'train'

        bin_dir = Path(f'bins/{dataset_name}_bin')
        self.all_items = np.load(bin_dir / 'all_items.npy')
        self.user_index = np.load(bin_dir / 'user_index.npy')
        self.usernum, self.itemnum = load_metadata(dataset_name)
        
        self.valid_users = np.where(self.user_index[:, 1] > 0)[0]
        if mode == 'train':
            self.valid_users = self.valid_users[self.user_index[self.valid_users, 1] >= 4]

        # DuoRec "same target" index: map the user's last training target to
        # the list of uids whose training sequence ends in the same item.
        # Used by `__getitem__` to draw a semantic augmentation on the fly.
        # Built once per worker at init (O(|train users|)).
        self.same_target_index: Dict[int, List[int]] = {}
        if self.use_duorec:
            for uid_arr in self.valid_users:
                uid_int = int(uid_arr)
                full_seq = self._get_user_sequence(uid_int)
                seq_items = full_seq[:-2] if len(full_seq) >= 4 else full_seq
                if len(seq_items) < 2:
                    continue
                last_target = int(seq_items[-1])
                self.same_target_index.setdefault(last_target, []).append(uid_int)
    
    def __len__(self) -> int:
        return len(self.valid_users)
    
    def _get_user_sequence(self, uid: int) -> np.ndarray:
        offset, length = self.user_index[uid]
        return np.asarray(self.all_items[offset:offset + length], dtype=np.int32)

    def __getitem__(self, idx: int) -> Tuple:
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
            # [ĐÃ SỬA]: Xóa np.zeros của neg đi, chỉ trả về uid, seq, pos
            result = (
                 uid,
                 np.zeros(self.maxlen, dtype=np.int32),
                 np.zeros(self.maxlen, dtype=np.int32)
             )
            if self.mem_maxlen > 0:
                result = result + (np.zeros(self.mem_maxlen, dtype=np.int32),)
            return result
        
        input_seq = seq_items[:-1]
        target_pos = seq_items[1:]
        
        eff_len = min(len(input_seq), self.maxlen)
        
        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        
        seq[-eff_len:] = input_seq[-eff_len:]
        pos[-eff_len:] = target_pos[-eff_len:]
        
        result = (uid, seq, pos)

        # DuoRec supervised augmentation: pick another user whose training
        # sequence ends in the same target item. Input view drops the final
        # target token to mirror the encoder's input preparation for `seq`.
        if self.use_duorec:
            uid_int = int(uid)
            last_target = int(seq_items[-1])
            bucket = self.same_target_index.get(last_target)
            if bucket:
                sem_uid = random.choice(bucket)
                tries = 0
                while len(bucket) > 1 and sem_uid == uid_int and tries < 5:
                    sem_uid = random.choice(bucket)
                    tries += 1
                sem_full = self._get_user_sequence(sem_uid)
                sem_items = sem_full[:-2] if len(sem_full) >= 4 else sem_full
                sem_input = sem_items[:-1] if len(sem_items) > 1 else sem_items
                sem_eff = min(len(sem_input), self.maxlen)
                sem_seq = np.zeros(self.maxlen, dtype=np.int32)
                if sem_eff > 0:
                    sem_seq[-sem_eff:] = sem_input[-sem_eff:]
            else:
                # No other user with the same target -> self-augmentation
                # (reduces to pure dropout-view contrastive for this row).
                sem_seq = seq.copy()
            result = result + (sem_seq,)

        if self.mem_maxlen > 0:
            mem_eff_len = min(len(input_seq), self.mem_maxlen)
            mem_seq = np.zeros(self.mem_maxlen, dtype=np.int32)
            mem_seq[-mem_eff_len:] = input_seq[-mem_eff_len:]
            result = result + (mem_seq,)
        
        return result

def get_dataloader(dataset_name: str, maxlen: int, batch_size: int, mode: str = 'train', num_workers: int = 4, num_negatives: int = 1, neg_sampling_mode: str = 'random', mem_maxlen: int = 0, use_duorec: bool = False) -> DataLoader:
    dataset = SASRecDataset(dataset_name, maxlen, mode, num_negatives, neg_sampling_mode, mem_maxlen=mem_maxlen, use_duorec=use_duorec)
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

def _batch_evaluate_logic(model, dataset, args: Any, mode: str = 'test') -> Tuple[float, float]:
    amp_enabled = getattr(args, 'use_amp', False)
    amp_dt = torch.bfloat16 if getattr(args, 'amp_dtype', 'bf16') == 'bf16' else torch.float16
    train, valid, test, usernum, itemnum = dataset
    
    # Random sample 10.000 users để evaluate (nếu dataset quá lớn)
    if usernum > 10000:
        eval_users = random.sample(range(1, usernum + 1), 10000)
    else:
        eval_users = list(range(1, usernum + 1))
    
    if mode == 'test':
        eval_users = [u for u in eval_users if len(train[u]) >= 1 and len(test.get(u, [])) >= 1]
    else:
        eval_users = [u for u in eval_users if len(train[u]) >= 1 and len(valid.get(u, [])) >= 1]
    
    num_users = len(eval_users)
    eval_batch_size = 100
    
    mem_maxlen = getattr(args, 'mem_maxlen', 0)
    use_mem = mem_maxlen > 0 and getattr(args, '_mem_active', False)
    
    all_ranks = []
    
    for batch_start in range(0, num_users, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, num_users)
        batch_size = batch_end - batch_start
        
        batch_u_ids = eval_users[batch_start:batch_end]
        batch_seqs = np.zeros((batch_size, args.maxlen), dtype=np.int32)
        batch_targets = np.zeros(batch_size, dtype=np.int32)
        batch_mem_seqs = np.zeros((batch_size, mem_maxlen), dtype=np.int32) if use_mem else None
        
        mask = torch.zeros((batch_size, itemnum), dtype=torch.bool, device=args.device)
        
        for i, u in enumerate(batch_u_ids):
            idx = args.maxlen - 1
            if mode == 'test':
                batch_seqs[i, idx] = valid[u][0]
                idx -= 1
                source_seq = train[u]
                batch_targets[i] = test[u][0]
                seen_items = train[u] + valid[u]  # Ở test, mask cả train và valid
            else:
                source_seq = train[u]
                batch_targets[i] = valid[u][0]
                seen_items = train[u]             # Ở valid, chỉ mask train
            
            for item in reversed(source_seq):
                batch_seqs[i, idx] = item
                idx -= 1
                if idx == -1:
                    break
            
            if use_mem:
                if mode == 'test':
                    mem_items = list(train[u]) + [valid[u][0]]
                else:
                    mem_items = list(train[u])
                mem_eff = min(len(mem_items), mem_maxlen)
                batch_mem_seqs[i, -mem_eff:] = mem_items[-mem_eff:]
            
            seen_idx = [item - 1 for item in seen_items if 1 <= item <= itemnum]
            if seen_idx:
                mask[i, seen_idx] = True

        all_items = torch.arange(1, itemnum + 1, device=args.device).unsqueeze(0).expand(batch_size, -1)
        
        with torch.amp.autocast(device_type='cuda', dtype=amp_dt, enabled=amp_enabled):
            if use_mem:
                predictions = model.predict(batch_u_ids, batch_seqs, all_items, mem_seqs=batch_mem_seqs)
            else:
                predictions = model.predict(batch_u_ids, batch_seqs, all_items)
        
        predictions.masked_fill_(mask, -float('inf'))
        
        target_idx = torch.tensor(batch_targets - 1, device=args.device)
        target_scores = predictions[torch.arange(batch_size, device=args.device), target_idx]
        
        higher_ranks = (predictions > target_scores.unsqueeze(1)).sum(dim=1)
        equal_items = (predictions == target_scores.unsqueeze(1)).sum(dim=1) - 1
        random_ties = (torch.rand_like(equal_items.float(), device=args.device) * (equal_items + 1)).long()
        ranks = higher_ranks + random_ties

        all_ranks.append(ranks)
    
    all_ranks = torch.cat(all_ranks, dim=0).float()

    hits_5 = (all_ranks < 5).float()
    ndcgs_5 = hits_5 / torch.log2(all_ranks + 2.0)

    hits_10 = (all_ranks < 10).float()
    ndcgs_10 = hits_10 / torch.log2(all_ranks + 2.0)
    
    return ndcgs_5.mean().item(), hits_5.mean().item(), ndcgs_10.mean().item(), hits_10.mean().item()


def evaluate(model, dataset: Tuple, args: Any) -> Tuple[float, float]:
    return _batch_evaluate_logic(model, dataset, args, mode='test')


def evaluate_valid(model, dataset: Tuple, args: Any) -> Tuple[float, float]:
    return _batch_evaluate_logic(model, dataset, args, mode='valid')


def export_bsarec_txt(dataset_name: str, out_path: Path) -> None:
    """Convert CONGA's preprocessed bin data to BSARec's user-sequence text format.

    BSARec/WEARec format (one line per user):
        user_id item1 item2 item3 ...  (space-separated, chronological order)

    CONGA's bin format (data.txt) has one interaction per line:
        user_id item_id

    Args:
        dataset_name: CONGA dataset name (e.g. "ml-1m", "Beauty").
        out_path: Destination .txt file path.
    """
    from collections import defaultdict as _dd

    bin_dir = Path(f'bins/{dataset_name}_bin')
    data_path = bin_dir / 'data.txt'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_path}. "
            "Run check_and_convert_dataset(dataset_name) first."
        )

    sequences: dict = _dd(list)
    with open(data_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                uid, iid = int(parts[0]), int(parts[1])
                sequences[uid].append(iid)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for uid in sorted(sequences.keys()):
            items_str = ' '.join(str(i) for i in sequences[uid])
            f.write(f'{uid} {items_str}\n')