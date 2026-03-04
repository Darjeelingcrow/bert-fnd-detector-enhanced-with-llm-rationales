import os
import random
import numpy as np
import torch
import logging
from datetime import datetime

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(log_dir, name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def data2gpu(batch, use_cuda):
    if use_cuda:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda()
    return batch

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Recorder():
    def __init__(self, early_stop):
        self.early_stop = early_stop
        self.best_metric = -1
        self.bad_count = 0
        self.best_model_path = None

    def add(self, metric):
        if metric > self.best_metric:
            self.best_metric = metric
            self.bad_count = 0
            return True # Improved
        else:
            self.bad_count += 1
            return False # Not improved

    def stop(self):
        return self.bad_count >= self.early_stop
