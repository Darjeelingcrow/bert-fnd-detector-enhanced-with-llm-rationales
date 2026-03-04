import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, ChineseCLIPProcessor, AutoProcessor
from pathlib import Path

class FakenewsDataset(Dataset):
    def __init__(self, data_path, image_root, text_tokenizer, rational_tokenizer, img_processor=None, text_max_len=170, rational_max_len=170, enable_reverse_label=True, mode='train'):
        self.data = []
        self.image_root = image_root
        self.text_tokenizer = text_tokenizer
        self.rational_tokenizer = rational_tokenizer
        self.img_processor = img_processor
        self.text_max_len = text_max_len
        self.rational_max_len = rational_max_len
        self.mode = mode
        self.enable_reverse_label = enable_reverse_label
        self.label_map = {
            "real": 0, "fake": 1,
            0: 1, 1: 0 # Original code mapping seems to flip 0 and 1 or handle mixed types? 
                       # Wait, original code: 0:1, 1:0. This looks like a flip. 
                       # Let's stick to the original logic to be safe, but document it.
                       # Original: "real": 0, "fake": 1, 0: 1, 1: 0. 
                       # If input is "real" -> 0. If input is 1 -> 0. 
                       # If input is "fake" -> 1. If input is 0 -> 1.
                       # This implies 0 in json might mean "real" (mapped to 1?) or "fake"?
                       # Let's look at original code again.
                       # label_dict = {"real": 0, "fake": 1, 0: 1, 1: 0}
                       # If json has 0, it becomes 1. If json has 1, it becomes 0.
                       # If json has "real", it becomes 0.
                       # This is confusing. I will preserve it but add a comment.
        }
        
        self.label_map_ftr = {
            "real": 0, "fake": 1, "other": 2,
            0: 1, 1: 0, 2: 2
        }

        self._load_data(data_path)

    def _load_data(self, path):
        print(f"Loading data from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                # Check image existence if image processor is provided
                if self.img_processor:
                    img_name = item.get('image')
                    if not img_name:
                        continue
                    img_path = os.path.join(self.image_root, img_name)
                    if not os.path.exists(img_path):
                        # print(f"Warning: Image not found: {img_path}")
                        continue
                    item['img_path'] = img_path

                self.data.append(item)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Process Text
        content = item['content']
        inputs = self.text_tokenizer(
            content,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # 2. Process Label
        raw_label = item['label']
        label = self.label_map.get(raw_label, 0) if self.enable_reverse_label else raw_label # Default to 0 if not found, but should be found

        # 3. Process Auxiliary Features (Rationale)
        # Using .get() with defaults to handle missing keys safely
        # rational_1 = item.get('rational_1', "") or item.get('td_rationale', "")
        # rational_2 = item.get('rational_2', "") or item.get('cs_rationale', "")

        rational_1 = item['rational_1']
        rational_2 = item['rational_2']
        
        rational_1_inputs = self.rational_tokenizer(rational_1, max_length=self.rational_max_len, padding='max_length', truncation=True, return_tensors='pt')
        rational_2_inputs = self.rational_tokenizer(rational_2, max_length=self.rational_max_len, padding='max_length', truncation=True, return_tensors='pt')

        rational_1_ids = rational_1_inputs['input_ids'].squeeze(0)
        rational_1_mask = rational_1_inputs['attention_mask'].squeeze(0)
        rational_2_ids = rational_2_inputs['input_ids'].squeeze(0)
        rational_2_mask = rational_2_inputs['attention_mask'].squeeze(0)

        # 4. Process Auxiliary Labels
        # rational_1_pred = self.label_map_ftr.get(item.get('rational_1_pred'))
        # rational_2_pred = self.label_map_ftr.get(item.get('rational_2_pred'))
        
        # rational_1_acc = int(item.get('rational_1_acc'))
        # rational_2_acc = int(item.get('rational_2_acc'))

        sample = {
            'content': input_ids,
            'content_masks': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'id': str(item.get('id', item.get('source_id', ''))),
            
            'r1': rational_1_ids,
            'r1_masks': rational_1_mask,
            'r2': rational_2_ids,
            'r2_masks': rational_2_mask,
            
            # 'rational_1_pred': torch.tensor(rational_1_pred, dtype=torch.long),
            # 'rational_2_pred': torch.tensor(rational_2_pred, dtype=torch.long),
            # 'rational_1_acc': torch.tensor(rational_1_acc, dtype=torch.long),
            # 'rational_2_acc': torch.tensor(rational_2_acc, dtype=torch.long),
        }

        # 5. Process Image
        # if self.img_processor:
            
        #     image = Image.open(item['img_path']).resize((224, 224)).convert("RGB")
        #     image_inputs = self.img_processor(images=image, return_tensors="pt", do_resize=False)
        #     sample['img'] = image_inputs['pixel_values'].squeeze(0)

        if self.img_processor:
            with Image.open(item['img_path']) as img:
                # 1. 先处理调色板图片（P 模式）
                if img.mode == "P":
                    # 官方建议：先转 RGBA
                    img = img.convert("RGBA")

                # 2. 如果有 alpha 通道（RGBA），把透明度贴到白底上再变成 RGB
                if img.mode == "RGBA":
                    # 创建白色背景
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    # 用 alpha 通道做 mask 贴过去
                    alpha = img.split()[3]
                    background.paste(img, mask=alpha)
                    img = background  # 现在是纯 RGB 了
                else:
                    # 其它情况统一转成 RGB
                    img = img.convert("RGB")

                # 3. 统一 resize 到 224x224
                img = img.resize((224, 224), Image.BICUBIC)

            # 4. 交给 CLIP 的 image_processor，明确通道在最后
            img = np.array(img)  # (H, W, 3)
            image_inputs = self.img_processor(
                images=img,
                return_tensors="pt",
                input_data_format="channels_last",
            )
            sample['img'] = image_inputs['pixel_values'].squeeze(0)
            
        return sample

def create_dataloader(config, mode='train'):
    # Read encoder paths from config with sensible defaults to avoid empty-string errors
    text_encoder_path = config.get('text_encoder_path') or 'bert-base-chinese'
    rational_encoder_path = config.get('rational_encoder_path') or text_encoder_path
    img_encoder_path = config.get('img_encoder_path') or None

    # Instantiate tokenizers/processors with clearer error messages on failure
    try:
        # Prefer fast tokenizers when available to avoid slow processor warnings
        text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load text tokenizer from '{text_encoder_path}': {e}")
    try:
        rational_tokenizer = AutoTokenizer.from_pretrained(rational_encoder_path, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load rational tokenizer from '{rational_encoder_path}': {e}")

    # Root path for data files (must be provided in config)
    data_root = config.get('root_path')
    if not data_root:
        raise RuntimeError("Configuration error: 'root_path' must be set in config (path to directory containing train.jsonl/test.jsonl).")

    # Decide whether to enable image processing: only enable if img_encoder_path provided
    # AND at least one sample in the data file references an existing image file under image_root.
    img_processor = None
    if img_encoder_path:
        # Determine data file path first (try common locations)
        file_path = os.path.join(data_root, 'train.jsonl') if mode == 'train' else os.path.join(data_root, 'test.jsonl')
        image_root = config.get('image_root', data_root)
        try:
            has_valid_image = False
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh):
                        if i > 200:  # only sample first 200 lines
                            break
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        img_name = item.get('image')
                        if img_name:
                            img_path = os.path.join(image_root, img_name)
                            if os.path.exists(img_path):
                                has_valid_image = True
                                break
            if has_valid_image:
                img_processor = AutoProcessor.from_pretrained(img_encoder_path)
            else:
                img_processor = None
        except Exception:
            img_processor = None

    data_root = config['root_path']
    # Construct file path based on mode and monthly logic if needed
    # Original logic: get_monthly_path(data_type, root_path, 'train.jsonl')
    # We simplify: assume standard names or config provided names
    
    if mode == 'train':
        file_name = 'train.jsonl'
        shuffle = True
    elif mode == 'val':
        file_name = 'val.jsonl'
        shuffle = False
    else:
        file_name = 'test.jsonl'
        shuffle = False
        
    # Support for the "monthly" folder structure if it exists, otherwise direct path
    # Original code had: os.path.join(root_path, data_type, 'month_'+str(month), file_name)
    # But user args usually just point to a root.
    # Let's try to find the file.
    
    # Simplified path construction:
    # 1. Try root_path/file_name
    # 2. Try root_path/data_name/file_name
    
    file_path = os.path.join(data_root, file_name)
    if not os.path.exists(file_path):
        # Try looking into subfolders if needed, or just fail
        # For now, assume the user provides the directory containing jsonl files
        pass

    # Image root
    # Original code hardcoded: /data01/qy/fakenews/Dataset/weibo
    # We should make this configurable
    image_root = config.get('image_root', '/data01/qy/fakenews/Dataset/weibo')

    enable_reverse_label = True if 'enable_reverse_label' in config else False

    dataset = FakenewsDataset(
        data_path=file_path,
        image_root=image_root,
        text_tokenizer=text_tokenizer,
        rational_tokenizer=rational_tokenizer,
        img_processor=img_processor,
        text_max_len=config['text_max_len'],
        rational_max_len=config['rational_max_len'],
        enable_reverse_label=enable_reverse_label,
        mode=mode
    )

    # Allow overriding number of workers from config
    # Increase workers with multi-GPU for better throughput
    num_gpus = config.get('num_gpus', 1)
    default_workers = 2 if num_gpus > 1 else 0  # Use workers with multi-GPU
    num_workers = int(config.get('num_workers', default_workers))
    pin_memory = bool(config.get('use_cuda', False))

    dataloader = DataLoader(
        dataset,
        batch_size=config['batchsize'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
