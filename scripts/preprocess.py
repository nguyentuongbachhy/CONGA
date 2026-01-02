#!/usr/bin/env python
"""
Data preprocessing script.

Downloads and preprocesses datasets for sequential recommendation.

Usage:
    python scripts/preprocess.py --dataset ml-1m
    python scripts/preprocess.py --dataset beauty
    python scripts/preprocess.py --dataset sports
"""

import os
import sys
import gzip
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_amazon_json(path: str) -> List[Tuple[str, str, float]]:
    """Parse Amazon review JSON file."""
    data = []
    
    def parse_line(line):
        try:
            return json.loads(line)
        except:
            return None
    
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                item = parse_line(line)
                if item and 'reviewerID' in item and 'asin' in item:
                    timestamp = item.get('unixReviewTime', 0)
                    data.append((item['reviewerID'], item['asin'], timestamp))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = parse_line(line)
                if item and 'reviewerID' in item and 'asin' in item:
                    timestamp = item.get('unixReviewTime', 0)
                    data.append((item['reviewerID'], item['asin'], timestamp))
    
    return data


def load_movielens(path: str) -> List[Tuple[str, str, float]]:
    """Load MovieLens ratings file."""
    data = []
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id, item_id, rating, timestamp = parts[:4]
                data.append((user_id, item_id, float(timestamp)))
    
    return data


def filter_data(
    data: List[Tuple[str, str, float]],
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> List[Tuple[str, str, float]]:
    """Filter users and items with too few interactions."""
    # Count interactions
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    
    for user, item, _ in data:
        user_counts[user] += 1
        item_counts[item] += 1
    
    # Filter
    filtered = [
        (user, item, ts)
        for user, item, ts in data
        if user_counts[user] >= min_user_interactions
        and item_counts[item] >= min_item_interactions
    ]
    
    return filtered


def create_mappings(
    data: List[Tuple[str, str, float]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create user and item ID mappings."""
    users = sorted(set(user for user, _, _ in data))
    items = sorted(set(item for _, item, _ in data))
    
    user_map = {u: i + 1 for i, u in enumerate(users)}  # Start from 1
    item_map = {i: j + 1 for j, i in enumerate(items)}  # Start from 1
    
    return user_map, item_map


def build_sequences(
    data: List[Tuple[str, str, float]],
    user_map: Dict[str, int],
    item_map: Dict[str, int],
) -> Dict[int, List[int]]:
    """Build user sequences sorted by timestamp."""
    user_sequences = defaultdict(list)
    
    # Sort by timestamp
    sorted_data = sorted(data, key=lambda x: x[2])
    
    for user, item, ts in sorted_data:
        user_id = user_map[user]
        item_id = item_map[item]
        user_sequences[user_id].append(item_id)
    
    return dict(user_sequences)


def save_dataset(
    sequences: Dict[int, List[int]],
    output_path: str,
):
    """Save dataset in required format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for user_id, items in sorted(sequences.items()):
            for item_id in items:
                f.write(f"{user_id} {item_id}\n")
    
    print(f"Saved to {output_path}")


def print_stats(sequences: Dict[int, List[int]], name: str):
    """Print dataset statistics."""
    num_users = len(sequences)
    num_items = len(set(item for items in sequences.values() for item in items))
    num_interactions = sum(len(items) for items in sequences.values())
    avg_seq_len = num_interactions / num_users
    
    print(f"\n{name} Statistics:")
    print(f"  Users: {num_users:,}")
    print(f"  Items: {num_items:,}")
    print(f"  Interactions: {num_interactions:,}")
    print(f"  Avg sequence length: {avg_seq_len:.2f}")
    print(f"  Density: {num_interactions / (num_users * num_items) * 100:.4f}%")


def preprocess_movielens_1m(raw_path: str, output_path: str):
    """Preprocess MovieLens-1M dataset."""
    ratings_file = os.path.join(raw_path, "ratings.dat")
    
    if not os.path.exists(ratings_file):
        print(f"Error: {ratings_file} not found")
        print("Please download from https://grouplens.org/datasets/movielens/1m/")
        return
    
    print("Loading MovieLens-1M...")
    data = load_movielens(ratings_file)
    print(f"Raw interactions: {len(data):,}")
    
    # Filter
    data = filter_data(data, min_user_interactions=5, min_item_interactions=5)
    print(f"After filtering: {len(data):,}")
    
    # Create mappings and sequences
    user_map, item_map = create_mappings(data)
    sequences = build_sequences(data, user_map, item_map)
    
    print_stats(sequences, "MovieLens-1M")
    save_dataset(sequences, output_path)


def preprocess_amazon(raw_path: str, output_path: str, category: str = "Beauty"):
    """Preprocess Amazon review dataset."""
    # Look for JSON file
    json_file = None
    for filename in os.listdir(raw_path):
        if category.lower() in filename.lower() and filename.endswith(('.json', '.json.gz')):
            json_file = os.path.join(raw_path, filename)
            break
    
    if not json_file:
        print(f"Error: Amazon {category} dataset not found in {raw_path}")
        print("Please download from https://jmcauley.ucsd.edu/data/amazon/")
        return
    
    print(f"Loading Amazon {category}...")
    data = parse_amazon_json(json_file)
    print(f"Raw interactions: {len(data):,}")
    
    # Filter
    data = filter_data(data, min_user_interactions=5, min_item_interactions=5)
    print(f"After filtering: {len(data):,}")
    
    # Create mappings and sequences
    user_map, item_map = create_mappings(data)
    sequences = build_sequences(data, user_map, item_map)
    
    print_stats(sequences, f"Amazon {category}")
    save_dataset(sequences, output_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ml-1m", "beauty", "sports", "toys", "yelp"],
                        help="Dataset to preprocess")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Directory containing raw data")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory")
    
    args = parser.parse_args()
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}.txt")
    
    if args.dataset == "ml-1m":
        raw_path = os.path.join(args.raw_dir, "ml-1m")
        preprocess_movielens_1m(raw_path, output_path)
    elif args.dataset == "beauty":
        preprocess_amazon(args.raw_dir, output_path, "Beauty")
    elif args.dataset == "sports":
        preprocess_amazon(args.raw_dir, output_path, "Sports")
    elif args.dataset == "toys":
        preprocess_amazon(args.raw_dir, output_path, "Toys")
    else:
        print(f"Dataset {args.dataset} not yet supported")


if __name__ == "__main__":
    main()
