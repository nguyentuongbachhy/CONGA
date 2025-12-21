import polars as pl
import numpy as np
import os

def convert_to_bin(dataset_name: str):
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
    
    itemnum = df['iid'].max()
    usernum = max_user
    
    np.save(f'{bin_dir}/all_items.npy', all_items)
    np.save(f'{bin_dir}/user_index.npy', user_index)
    
    with open(f'{bin_dir}/meta.txt', 'w') as f:
        f.write(f'{usernum},{itemnum}')
    