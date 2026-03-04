import os
import sys
import itertools
import json
import argparse
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloader
from src.trainer import Trainer
from src.utils import setup_seed, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Grid Search")
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_name', type=str, default='weibo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='./results/grid_search')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Define Grid Search Space
    param_grid = {
        'lr': [1e-5, 2e-5, 5e-5],
        'batchsize': [16, 32],
        'weight_decay': [1e-4, 1e-5],
        # Add more parameters as needed
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations: {len(combinations)}")
    
    results = []
    
    for i, params in enumerate(combinations):
        print(f"\nRunning combination {i+1}/{len(combinations)}: {params}")
        
        # Base Config
        config = {
            'root_path': args.root_path,
            'data_name': args.data_name,
            'model_name': 'MM', # Default or parameterized
            'bert_path': '/data01/qy/models/bert-base-chinese',
            'img_encoder_path': '/data01/qy/models/chinese-clip-vit-base-patch16',
            'emb_dim': 768,
            'co_attention_dim': 300,
            'epoch': 10, # Reduced for grid search
            'early_stop': 3,
            'seed': 3759,
            'use_cuda': True,
            'max_len': 170,
            'model': { # Default model params
                'mlp': {'dims': [384], 'dropout': 0.2},
                'llm_judgment_predictor_weight': -1,
                'rationale_usefulness_evaluator_weight': -1,
                'kd_loss_weight': 1,
                'evidential_error_weight': -1,
            }
        }
        
        # Update with grid params
        config.update(params)
        
        # Setup Output for this run
        run_name = f"run_{i}"
        run_dir = os.path.join(args.output_dir, run_name)
        config['save_log_dir'] = os.path.join(run_dir, 'logs')
        config['save_param_dir'] = os.path.join(run_dir, 'checkpoints')
        config['tensorboard_dir'] = os.path.join(run_dir, 'tensorboard')
        
        for d in [config['save_log_dir'], config['save_param_dir'], config['tensorboard_dir']]:
            os.makedirs(d, exist_ok=True)
            
        logger = get_logger(config['save_log_dir'], f'grid_search_{i}')
        setup_seed(config['seed'])
        
        # Create Dataloaders (Re-create to ensure clean state or reuse if possible)
        # Re-creating is safer for shuffling/seeding
        train_loader = create_dataloader(config, mode='train')
        val_loader = create_dataloader(config, mode='val')
        
        trainer = Trainer(config, logger)
        best_metric = trainer.fit(train_loader, val_loader)
        
        results.append({
            'params': params,
            'best_metric': best_metric
        })
        
        # Save intermediate results
        with open(os.path.join(args.output_dir, 'grid_search_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
    print("Grid Search Completed.")

if __name__ == '__main__':
    main()
