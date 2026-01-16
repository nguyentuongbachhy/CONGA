"""
Wrapper script to run hyperparameter tuning with config from config.txt
"""

import subprocess
import sys
from pathlib import Path


def parse_config(config_file: str):
    """Parse config.txt file"""
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
                
                if value.lower() == 'true':
                    args[key] = True
                elif value.lower() == 'false':
                    args[key] = False
                else:
                    args[key] = value
    
    return args


def main():
    # Parse config
    config_file = 'config.txt'
    if not Path(config_file).exists():
        print(f"Error: {config_file} not found!")
        sys.exit(1)
    
    print(f"Loading config from {config_file}...")
    config_args = parse_config(config_file)
    
    dataset = config_args.get('dataset', 'ml-1m')
    pattern_file = f'pattern_data/{dataset}_patterns.pkl'
    graph_emb_file = f'pretrained_embeddings/{dataset}_graph_embeddings.pt'
    
    # Check files
    if not Path(pattern_file).exists():
        print(f"Error: Pattern file not found: {pattern_file}")
        print(f"   Run: python mine_patterns.py --dataset {dataset}")
        sys.exit(1)
    
    if not Path(graph_emb_file).exists():
        print(f"Error: Graph embeddings not found: {graph_emb_file}")
        print(f"   Run: python train_graph_pretrain.py --dataset {dataset}")
        sys.exit(1)
    
    print(f"✓ Pattern file: {pattern_file}")
    print(f"✓ Graph embeddings: {graph_emb_file}")
    
    # Build command
    cmd = [sys.executable, 'tune_hyperparams.py']
    
    # Filter and add args (exclude train_dir)
    for key, value in config_args.items():
        if key == 'train_dir':
            continue
        
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    
    # Add pattern and graph files
    cmd.extend(['--pattern_file', pattern_file])
    cmd.extend(['--graph_emb_file', graph_emb_file])
    
    # Override num_epochs to 100 for fast tuning
    if '--num_epochs' in cmd:
        idx = cmd.index('--num_epochs')
        cmd[idx + 1] = '100'
    else:
        cmd.extend(['--num_epochs', '100'])
    
    # Add seed
    cmd.extend(['--seed', '42'])
    
    print(f"\nStarting hyperparameter tuning...")
    print(f"   Stage 1: Tune alpha (4 configs × 100 epochs)")
    print(f"   Stage 2: Tune reg_weight (4 configs × 100 epochs)")
    print(f"   Total: 8 configs × 100 epochs = 800 epochs")
    print(f"   Seed: 42 (fixed)\n")
    
    # Run tuning
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
