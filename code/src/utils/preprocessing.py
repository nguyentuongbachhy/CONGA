import polars as pl
import numpy as np
import os

def convert_to_bin(dataset_name: str, neg_pool_size: int = 10000) -> None:
    data_path = f'data/{dataset_name}.txt'
    bin_dir = f'bins/{dataset_name}_bin'
    os.makedirs(bin_dir, exist_ok=True)
    
    q = (
        pl.scan_csv(
            data_path,
            separator=' ',
            has_header=False,
            new_columns=['uid', 'iid'],
            schema={'uid': pl.Int32, 'iid': pl.Int32}
        ).sort('uid')
    )
    
    df = q.collect()
    all_items = df['iid'].to_numpy()
    user_stats = df.group_by('uid', maintain_order=True).len()
    
    unique_uids = user_stats['uid'].to_numpy()
    lengths = user_stats['len'].to_numpy()
    
    offsets = np.zeros(len(lengths), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]
    
    max_user = unique_uids.max()
    user_index = np.zeros((max_user + 1, 2), dtype=np.int64)
    
    user_index[unique_uids, 0] = offsets
    user_index[unique_uids, 1] = lengths
    
    itemnum = int(df['iid'].max())
    usernum = int(max_user)
    
    print(f"Generating negative samples pool (size={neg_pool_size} per user)...")
    neg_pool = np.zeros((max_user + 1, neg_pool_size), dtype=np.int32)
    for uid in unique_uids:
        offset, length = user_index[uid]
        user_items = set(all_items[offset:offset + length])
        valid_items = np.array([i for i in range(1, itemnum + 1) if i not in user_items], dtype=np.int32)
        if len(valid_items) >= neg_pool_size:
            neg_pool[uid] = np.random.choice(valid_items, size=neg_pool_size, replace=False)
        else:
            neg_pool[uid] = np.random.choice(valid_items, size=neg_pool_size, replace=True)
    
    np.save(f'{bin_dir}/all_items.npy', all_items)
    np.save(f'{bin_dir}/user_index.npy', user_index)
    np.save(f'{bin_dir}/neg_pool.npy', neg_pool)
    
    with open(f'{bin_dir}/meta.txt', 'w') as f:
        f.write(f'{usernum},{itemnum}')
    