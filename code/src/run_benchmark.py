"""
Wrapper script to run benchmark with config from config.txt file
"""

import subprocess
import sys
from pathlib import Path


def parse_config(config_file: str):
    """Parse config.txt file into command line arguments"""
    args = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ',' in line:
                key, value = line.split(',', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert boolean strings
                if value.lower() == 'true':
                    args[key] = True
                elif value.lower() == 'false':
                    args[key] = False
                else:
                    args[key] = value
    
    return args


def build_command(config_args, pattern_file, graph_emb_file):
    """Build command line for benchmark_configs.py"""
    cmd = [sys.executable, 'benchmark_configs.py']
    
    # Filter out train_dir (benchmark creates its own directories)
    filtered_args = {k: v for k, v in config_args.items() if k != 'train_dir'}
    
    # Add all config arguments
    for key, value in filtered_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    
    # Add pattern and graph files
    if pattern_file:
        cmd.append('--pattern_file')
        cmd.append(pattern_file)
    
    if graph_emb_file:
        cmd.append('--graph_emb_file')
        cmd.append(graph_emb_file)
    
    # Add fixed seed for reproducibility
    cmd.append('--seed')
    cmd.append('42')
    
    return cmd


def main():
    # Parse config file
    config_file = 'config.txt'
    if not Path(config_file).exists():
        print(f"Error: {config_file} not found!")
        sys.exit(1)
    
    print(f"Loading config from {config_file}...")
    config_args = parse_config(config_file)
    
    # Get dataset name
    dataset = config_args.get('dataset', 'ml-1m')
    
    # Set pattern and graph files
    pattern_file = f'pattern_data/{dataset}_patterns.pkl'
    graph_emb_file = f'pretrained_embeddings/{dataset}_graph_embeddings.pt'
    
    # Check if files exist
    if not Path(pattern_file).exists():
        print(f"Warning: Pattern file not found: {pattern_file}")
        print(f"   Run: python mine_patterns.py --dataset {dataset}")
        pattern_file = None
    else:
        print(f"✓ Pattern file found: {pattern_file}")
    
    if not Path(graph_emb_file).exists():
        print(f"Warning: Graph embeddings not found: {graph_emb_file}")
        print(f"   Run: python train_graph_pretrain.py --dataset {dataset}")
        graph_emb_file = None
    else:
        print(f"✓ Graph embeddings found: {graph_emb_file}")
    
    # Build and run command
    cmd = build_command(config_args, pattern_file, graph_emb_file)
    
    print(f"\nStarting benchmark with 3 configurations...")
    print(f"   Seed: 42 (fixed for fair comparison)")
    print(f"   Loss: gbce")
    print(f"   Results will be saved to: benchmark/\n")
    
    # Run benchmark
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
