"""
Pattern mining script using fast graph-based approach
"""

import argparse
import time
import pickle
from pathlib import Path
from pattern_mining.graph_pattern_miner import GraphPatternMiner
from utils import data_partition


def mine_patterns_from_dataset(dataset_name: str, min_support: float = 0.01, 
                               max_pattern_length: int = 5, min_pattern_length: int = 2,
                               top_k: int = 1000, output_dir: str = 'pattern_data',
                               window: int = 5):
    """
    Mine patterns from dataset using fast graph-based approach
    
    Args:
        dataset_name: Name of the dataset
        min_support: Minimum support threshold (ratio)
        max_pattern_length: Maximum pattern length
        min_pattern_length: Minimum pattern length
        top_k: Number of top patterns to keep
        output_dir: Output directory for pattern database
        window: Co-occurrence window size
    """
    print(f"\n{'='*60}")
    print(f"PATTERN MINING: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    user_train, user_valid, user_test, usernum, itemnum = data_partition(dataset_name)
    
    # Extract sequences
    sequences = []
    for user in user_train:
        if len(user_train[user]) >= min_pattern_length:
            sequences.append(user_train[user])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Users: {usernum:,}")
    print(f"  Items: {itemnum:,}")
    print(f"  Sequences: {len(sequences):,}")
    
    # Create miner
    miner = GraphPatternMiner(
        min_support=min_support,
        window=window,
        max_pattern_length=max_pattern_length,
        min_pattern_length=min_pattern_length,
        top_k=top_k
    )
    
    # Mine patterns
    start_time = time.time()
    patterns = miner.mine_patterns(sequences)
    elapsed_time = time.time() - start_time
    
    # Get statistics
    stats = miner.get_statistics()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"MINING RESULTS")
    print(f"{'='*60}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Speed: {len(sequences)/elapsed_time:.0f} sequences/sec")
    print(f"  Patterns found: {len(patterns):,}")
    
    if stats:
        print(f"\n  Graph Statistics:")
        print(f"    Nodes: {stats['num_nodes']:,}")
        print(f"    Edges: {stats['num_edges']:,}")
        print(f"    Avg degree: {stats['avg_degree']:.2f}")
        print(f"    Max degree: {stats['max_degree']:,}")
        
        if 'pattern_length_distribution' in stats:
            print(f"\n  Pattern Length Distribution:")
            for length in sorted(stats['pattern_length_distribution'].keys()):
                count = stats['pattern_length_distribution'][length]
                print(f"    Length {length}: {count:,} patterns")
    
    # Show top patterns
    print(f"\n  Top 10 Patterns:")
    for i, (pattern, support) in enumerate(patterns[:10], 1):
        print(f"    {i}. {pattern} (support: {support})")
    
    # Save patterns
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    db_file = output_path / f"{dataset_name}_patterns.pkl"
    
    with open(db_file, 'wb') as f:
        pickle.dump({
            'patterns': patterns,
            'stats': stats,
            'config': {
                'min_support': min_support,
                'window': window,
                'max_pattern_length': max_pattern_length,
                'min_pattern_length': min_pattern_length,
                'top_k': top_k
            }
        }, f)
    
    print(f"\n  ✓ Patterns saved to: {db_file}")
    print(f"{'='*60}\n")
    
    return patterns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast Graph-based Pattern Mining')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name')
    parser.add_argument('--min_support', type=float, default=0.01, 
                        help='Minimum support threshold (default: 0.01)')
    parser.add_argument('--max_pattern_length', type=int, default=5, 
                        help='Maximum pattern length (default: 5)')
    parser.add_argument('--min_pattern_length', type=int, default=2, 
                        help='Minimum pattern length (default: 2)')
    parser.add_argument('--top_k', type=int, default=1000, 
                        help='Number of top patterns to keep (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='pattern_data', 
                        help='Output directory (default: pattern_data)')
    parser.add_argument('--window', type=int, default=5, 
                        help='Co-occurrence window size (default: 5)')
    
    args = parser.parse_args()
    
    mine_patterns_from_dataset(
        dataset_name=args.dataset,
        min_support=args.min_support,
        max_pattern_length=args.max_pattern_length,
        min_pattern_length=args.min_pattern_length,
        top_k=args.top_k,
        output_dir=args.output_dir,
        window=args.window
    )
