import polars as pl
import numpy as np
import os


def apply_kcore_filter(df: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    while True:
        user_counts = df.group_by("uid").len()
        item_counts = df.group_by("iid").len()

        valid_users = user_counts.filter(pl.col("len") >= k)["uid"]
        valid_items = item_counts.filter(pl.col("len") >= k)["iid"]

        filtered = df.filter(pl.col("uid").is_in(valid_users) & pl.col("iid").is_in(valid_items))

        if len(filtered) == len(df):
            break
        df = filtered

    return df


def convert_to_bin(dataset_name: str, k: int = 5) -> None:
    data_path = f'data/{dataset_name}.txt'
    bin_dir = f'bins/{dataset_name}_bin'
    os.makedirs(bin_dir, exist_ok=True)

    df = (
        pl.scan_csv(
            data_path,
            separator=' ',
            has_header=False,
            new_columns=['uid', 'iid'],
            schema={'uid': pl.Int32, 'iid': pl.Int32}
        )
        .with_row_index("order")
        .collect()
    )

    print(f"Before {k}-core: {df['uid'].n_unique()} users, {df['iid'].n_unique()} items, {len(df)} interactions")
    df = apply_kcore_filter(df, k)
    print(f"After  {k}-core: {df['uid'].n_unique()} users, {df['iid'].n_unique()} items, {len(df)} interactions")

    # Re-index uid và iid về [1, N] liên tục sau khi filter
    uid_map = {old: new for new, old in enumerate(sorted(df['uid'].unique().to_list()), start=1)}
    iid_map = {old: new for new, old in enumerate(sorted(df['iid'].unique().to_list()), start=1)}

    df = df.with_columns([
        pl.col("uid").replace(uid_map).alias("uid"),
        pl.col("iid").replace(iid_map).alias("iid"),
    ]).sort(["uid", "order"])

    all_items = df['iid'].to_numpy()
    user_stats = df.group_by('uid', maintain_order=True).len()

    unique_uids = user_stats['uid'].to_numpy()
    lengths = user_stats['len'].to_numpy()

    offsets = np.zeros(len(lengths), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]

    max_user = int(unique_uids.max())
    user_index = np.zeros((max_user + 1, 2), dtype=np.int64)
    user_index[unique_uids, 0] = offsets
    user_index[unique_uids, 1] = lengths

    usernum = max_user
    itemnum = int(df['iid'].max())

    np.save(f'{bin_dir}/all_items.npy', all_items)
    np.save(f'{bin_dir}/user_index.npy', user_index)

    with open(f'{bin_dir}/meta.txt', 'w') as f:
        f.write(f'{usernum},{itemnum}')

    # Save filtered+reindexed interactions for data_partition
    df.select(['uid', 'iid']).write_csv(f'{bin_dir}/data.txt', separator=' ', include_header=False)