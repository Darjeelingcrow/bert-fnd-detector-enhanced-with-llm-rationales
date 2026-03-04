import os
import sys
import argparse
import pathlib
import json
import torch
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pdb
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloader
from src.trainer import Trainer
from src.utils import setup_seed, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Fakenews Detection Training")
    # Config file
    parser.add_argument('--config', type=str, default=None, help='Path to config file (JSON or YAML)')
    
    # Data Params
    parser.add_argument('--root_path', type=str, help='Root path of dataset')
    parser.add_argument('--data_name', type=str, default='weibo', help='Name of dataset (e.g., weibo, gossip)')
    parser.add_argument('--image_root', type=str, default='/data01/qy/fakenews/Dataset/weibo', help='Root path for images')
    
    # K-Fold Params
    parser.add_argument('--k_folds', type=int, default=0, help='Number of folds for K-Fold Cross Validation. 0 or 1 to disable.')

    # Model Params
    parser.add_argument('--model_name', type=str, default='MM', help='Model architecture name')
    parser.add_argument('--text_encoder', type=str, default='bert')
    parser.add_argument('--text_encoder_path', type=str, default='')
    parser.add_argument('--text_max_len', type=int, default=170)
    parser.add_argument('--rational_encoder_path', type=str, default='')
    parser.add_argument('--rational_max_len', type=int, default=170)
    parser.add_argument('--img_encoder', type=str, default='clip')
    parser.add_argument('--img_encoder_path', type=str, default='/data01/qy/models/chinese-clip-vit-base-patch16', help='Path to Image Encoder (CLIP)')
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--co_attention_dim', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=2)
    
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--w_dim', type=int, default=64)
    # Training Params
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--early_stop', type=int, default=6)
    parser.add_argument('--seed', type=int, default=3759)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_model', type=bool, default=True)
    # Loss Weights
    parser.add_argument('--llm_judgment_predictor_weight', type=float, default=-1)
    parser.add_argument('--rationale_usefulness_evaluator_weight', type=float, default=-1)
    parser.add_argument('--evidential_error_weight', type=float, default=-1)
    parser.add_argument('--kd_loss_weight', type=float, default=1)
    
    parser.add_argument('--align_weight', type=float, default=0.1)
    parser.add_argument('--self_re_weight', type=float, default=1)
    parser.add_argument('--cross_re_weight', type=float, default=1)
    
    # Output Params
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--info', type=str, default=None, help='info')
    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration dict from a JSON or YAML file."""
    p = pathlib.Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = p.suffix.lower()
    with open(p, 'r', encoding='utf-8') as f:
        if suffix in ['.json']:
            return json.load(f)
        elif suffix in ['.yml', '.yaml']:
            try:
                import yaml  # Optional dependency
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config. Install with: pip install pyyaml")
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file type: {suffix}. Use .json or .yaml/.yml")

def run_kfold(args, logger, base_config):
    logger.info(f"Starting {args.k_folds}-Fold Cross Validation")
    
    # Load original train data
    train_path = os.path.join(args.root_path, 'train.jsonl')

    data = []
    labels = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            data.append(item)
            # Use raw label for stratification
            # Normalize labels using the same logic as dataset.py
            raw_label = item['label']
            # label_map from dataset.py: {"real": 0, "fake": 1, 0: 1, 1: 0}
            if raw_label == "real": label = 0
            elif raw_label == "fake": label = 1
            elif raw_label == 0: label = 1
            elif raw_label == 1: label = 0
            else: label = 0
            labels.append(label)
            
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # Base output dir for kfold (use the timestamp dir created in main)
    exp_dir = base_config['save_log_dir'].replace('/logs', '') 
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        fold_num = fold + 1
        logger.info(f"========== Processing Fold {fold_num}/{args.k_folds} ==========")
        
        fold_dir = os.path.join(exp_dir, f'fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Prepare data files
        fold_train_data = [data[i] for i in train_idx]
        fold_val_data = [data[i] for i in val_idx]
        
        with open(os.path.join(fold_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
            for item in fold_train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        with open(os.path.join(fold_dir, 'val.jsonl'), 'w', encoding='utf-8') as f:
            for item in fold_val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Symlink test.jsonl (on Windows symlink may require admin privileges)
        src_test = os.path.join(args.root_path, 'test.jsonl')
        dst_test = os.path.join(fold_dir, 'test.jsonl')
        if os.path.exists(src_test):
            # Ensure fold directory exists
            os.makedirs(fold_dir, exist_ok=True)
            # If destination exists, attempt to remove it first
            if os.path.exists(dst_test):
                try:
                    os.remove(dst_test)
                except Exception as ex_remove:
                    logger.warning(f"Could not remove existing test file {dst_test}: {ex_remove}. Will attempt to overwrite.")
            # On Windows, creating symlinks often requires privileges; prefer copy on Windows
            try:
                if os.name == 'nt':
                    # Copy and overwrite atomically: copy to temp then replace
                    tmp_dst = dst_test + '.tmp'
                    shutil.copy2(src_test, tmp_dst)
                    try:
                        os.replace(tmp_dst, dst_test)
                    except Exception:
                        # fallback to copy2 directly
                        shutil.copy2(src_test, dst_test)
                    logger.info(f"Copied test file to {dst_test}")
                else:
                    os.symlink(os.path.abspath(src_test), dst_test)
                    logger.info(f"Symlinked test file to {dst_test}")
            except Exception as e:
                # Best-effort fallback: try a simple copy (may fail if permission denied)
                try:
                    shutil.copy2(src_test, dst_test)
                    logger.info(f"Fallback copied test file to {dst_test}")
                except Exception as e2:
                    logger.warning(f"Failed to symlink or copy test file: {e}; {e2}")
            
        # Config for this fold
        fold_config = base_config.copy()
        fold_config['root_path'] = fold_dir
        
        # Update save dirs
        fold_config['save_log_dir'] = os.path.join(fold_dir, 'logs')
        fold_config['save_param_dir'] = os.path.join(fold_dir, 'checkpoints')
        fold_config['tensorboard_dir'] = os.path.join(fold_dir, 'tensorboard')
        fold_config['test_save_path'] = os.path.join(fold_dir, 'test.json')
        
        for d in [fold_config['save_log_dir'], fold_config['save_param_dir'], fold_config['tensorboard_dir']]:
            os.makedirs(d, exist_ok=True)
            
        # Dataloaders
        train_loader = create_dataloader(fold_config, mode='train')
        val_loader = create_dataloader(fold_config, mode='val')
        test_path = os.path.join(args.root_path, 'test.jsonl')
        if os.path.exists(test_path):
            test_loader = create_dataloader(fold_config, mode='test')
        else:
            test_loader = None
        # Trainer
        fold_logger = get_logger(fold_config['save_log_dir'], f'train_fold_{fold_num}')
        
        trainer = Trainer(fold_config, fold_logger)
        metric = trainer.fit(train_loader, val_loader, test_loader)
        fold_results.append(metric)
        
    # Aggregate results
    avg_metrics = {}
    if fold_results:
        keys = fold_results[0].keys()
        for k in keys:
            values = [r[k] for r in fold_results if isinstance(r[k], (int, float))]
            if values:
                avg_metrics[k] = sum(values) / len(values)
    
    logger.info(f"K-Fold Average Results: {avg_metrics}")
    with open(os.path.join(exp_dir, 'kfold_results.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)

def main():
    args = parse_args()
    # If a config file is provided, load it and use values to override defaults
    file_cfg = None
    if args.config:
        file_cfg = load_config_from_file(args.config)
        # Merge: file config -> argparse namespace
        # Priority: command-line overrides file content when provided
        for k, v in file_cfg.items():
            if hasattr(args, k):
                # Only set if CLI arg is default (i.e., user didn't provide a different value)
                # We cannot easily distinguish here; prefer file values then let CLI overwrite below
                setattr(args, k, v)
        # Now re-parse CLI to overwrite any file-provided values
        # Note: For simplicity, we keep the current args; users can override key params via CLI
    
    # Setup Environment
    # Parse GPU list (can be comma-separated like "0,1,2" or single like "0")
    gpu_list = [g.strip() for g in args.gpu.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_list)
    use_multi_gpu = len(gpu_list) > 1
    
    setup_seed(args.seed)
    
    # Create Output Directories
    if args.info is None:
        timestamp = os.path.join(args.output_dir, f"{args.model_name}_{args.data_name}")
    else:
        timestamp = os.path.join(args.output_dir, f"{args.model_name}_{args.data_name}_{args.info}")

    log_dir = os.path.join(timestamp, 'logs')
    param_dir = os.path.join(timestamp, 'checkpoints')
    tensorboard_dir = os.path.join(timestamp, 'tensorboard')
    test_save_path = os.path.join(timestamp, 'test.json')

    for d in [log_dir, param_dir, tensorboard_dir]:
        os.makedirs(d, exist_ok=True)
        
    logger = get_logger(log_dir, 'train')
    logger.info(f"Arguments: {args}")
    
    # Config Dictionary
    config = vars(args)
    # If file had nested dicts like model section, preserve them
    if file_cfg and isinstance(file_cfg, dict):
        # Merge nested dictionaries (shallow merge; file values win unless replaced above)
        for k, v in file_cfg.items():
            if isinstance(v, dict):
                existing = config.get(k, {})
                merged = {**existing, **v}
                config[k] = merged
    config['use_cuda'] = True
    config['use_multi_gpu'] = use_multi_gpu
    config['gpu_list'] = gpu_list
    config['num_gpus'] = len(gpu_list)
    config['save_log_dir'] = log_dir
    config['save_param_dir'] = param_dir
    config['tensorboard_dir'] = tensorboard_dir
    # config['max_len'] = 170 # Hardcoded or add to args
    config['test_save_path'] = test_save_path
    # pdb.set_trace()
    # Model specific config structure (to match original model expectations)
    # if config['model_name'] == "distangle":
    #     config['model'] = {
    #         'mlp': {'dims': [384], 'dropout': 0.2},
    #         'align_weight': args.align_weight,
    #         'self_re_weight': args.self_re_weight,
    #         'cross_re_weight': args.cross_re_weight,
    #         'z_dim': args.z_dim,
    #         'w_dim': args.w_dim
    #     }
    # elif config['model_name'] == 'distangle_1':
    #     config['model'] = {
    #         'num_heads': 8,
    #         'embed_dim': 128,
    #         'conv1d_kernel_size_m': 3,
    #         'conv1d_kernel_size_r': 5,
    #         'layer': {
    #             'encoder': 2,
    #             'cross_query': 2,
    #             'mm': 2,
    #             'mr': 2,
    #             'shared': 2,
    #             'm_self': 2,
    #             'r_self': 2,
    #             'r_p': 2,
    #         },
    #         'drop_out':{
    #             'output': 0.5,
    #             'encoder': 0.3,
    #             'cross_query': 0.3,
    #             'mm': 0.3,
    #             'mr': 0.3,
    #             'shared': 0.3,
    #             'm_self': 0.3,
    #             'r_self': 0.3,
    #             'r_p': 0.3,
    #             'attn':{
    #                 'relu': 0.0,
    #                 'res': 0.0,
    #                 'embed': 0.2,
    #             }
    #         },
    #         'hyper_parameters':{
    #             # 'cls': {'mm': 0.5, 'r1': 0.5, 'r2': 0.5, 'shared': 0.5},
    #             'cls': 0.5,
    #             'recon_origin': 0.1,
    #             'recon_specific': 0.1,
    #             'ort': 0.1,
    #             'contrastive': 0.1
    #         }
    #     }
    # else:
    #     config['model'] = {
    #         'mlp': {'dims': [384], 'dropout': 0.2},
    #         'llm_judgment_predictor_weight': args.llm_judgment_predictor_weight,
    #         'rationale_usefulness_evaluator_weight': args.rationale_usefulness_evaluator_weight,
    #         'kd_loss_weight': args.kd_loss_weight,
    #         'evidential_error_weight': args.evidential_error_weight,
    #     }


    
    # Save Config
    with open(os.path.join(timestamp, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.k_folds > 1:
        run_kfold(args, logger, config)
        return

    # Create Dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloader(config, mode='train')
    val_loader = create_dataloader(config, mode='val')
    test_path = os.path.join(args.root_path, 'test.jsonl')
    
    if os.path.exists(test_path):
        test_loader = create_dataloader(config, mode='test')
    else:
        test_loader = None
    
    # Initialize Trainer
    trainer = Trainer(config, logger)
    
    # Start Training
    best_metric = trainer.fit(train_loader, val_loader, test_loader)
    
    logger.info(f"Best Metric: {best_metric}")

if __name__ == '__main__':
    main()
