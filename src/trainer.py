import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # New AMP Import

from src.utils import data2gpu, Averager, Recorder
from src.loss import *
from src.models.multimodal import *
from transformers import get_cosine_schedule_with_warmup

class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if config['use_cuda'] and torch.cuda.is_available() else "cpu")
        
        self._init_model()
        self._init_optimizer()
        self._init_criterion()
        
        # AMP Scaler
        self.scaler = GradScaler('cuda', enabled=config['use_cuda'])
        
        self.recorder = Recorder(config['early_stop'])
        self.writer = SummaryWriter(log_dir=config['tensorboard_dir'])

    def _init_model(self):
        model_name = self.config['model_name']
        self.model_name = model_name
        data_name = self.config['data_name']
        
        self.logger.info(f"Initializing model: {model_name} for data: {data_name}")
        
        if model_name == 'uncertainty':
            self.model = MModel_uncertainty(self.config)
        elif model_name == 'evidential':
            self.model = MModel_evidential_learning(self.config)
        elif model_name == 'distangle':
            self.model = MModel_distangle(self.config)
        elif model_name == 'distangle_1':
            self.model = MModel_distangle_1(self.config)
        else:
            self.model = MModel_CE(self.config)
            
        self.model.to(self.device)
        
        if self.config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"Using DataParallel with {num_gpus} GPUs")
            self.model = nn.DataParallel(self.model)
        elif self.config.get('use_multi_gpu', False):
            self.logger.warning(f"multi-GPU requested but only {torch.cuda.device_count()} GPU(s) available. Proceeding with single GPU.")

    def _init_optimizer(self):
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        encoder_lr = self.config.get('encoder_lr', 2e-5)
        base_lr = self.config.get('lr', 2e-4)
        weight_decay = self.config.get('weight_decay', 5e-5)

        encoder_params = []
        other_params = []
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('bert_content') or name.startswith('bert_FTR') or ('encoder.layer.' in name and 'bert' in name) or ('text_encoder' in name):
                encoder_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if len(encoder_params) > 0:
            param_groups.append({'params': encoder_params, 'lr': encoder_lr})
        if len(other_params) > 0:
            param_groups.append({'params': other_params, 'lr': base_lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    def _init_scheduler(self, num_training_steps):
        warmup_ratio = self.config.get('warmup_ratio', 0.1)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        self.logger.info(f"Initializing Cosine Schedule with Warmup: Total Steps={num_training_steps}, Warmup Steps={num_warmup_steps}")
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def _init_criterion(self):
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=self.config.get('label_smoothing', 0.0))
        self.criterion_bce = nn.BCELoss()
        if self.model_name == "distangle_1":
            self.criterion_mse = nn.MSELoss()
            self.criterion_cosine = nn.CosineEmbeddingLoss()
            self.criterion_contrast = SupervisedContrastiveLoss(temperature=self.config.get('contrastive_temperature', 0.07))
        self.logger.info("Loss functions initialized.")

    def compute_loss(self, res, batch_data):
        label = batch_data['label']
        loss_classify = self.criterion_cls(res['classify_pred'], label)
        loss_info = None

        if self.model_name == 'distangle':
            loss_align = F.mse_loss(res['z_m'], res['z_r'])
            loss_recon_self = F.mse_loss(res['m_rec_self'], res['m_target']) + F.mse_loss(res['r_rec_self'], res['r_target'])
            loss_recon_cross = F.mse_loss(res['r_from_m'], res['m_target']) + F.mse_loss(res['m_from_r'], res['r_target'])
            loss = loss_classify + self.config['model']['align_weight'] * loss_align + \
                                   self.config['model']['self_re_weight'] * loss_recon_self + \
                                   self.config['model']['cross_re_weight'] * loss_recon_cross
        elif self.model_name == 'distangle_1':
            loss_cls_mm = self.criterion_cls(res['logits_m_p'], label.long())
            loss_cls_r1 = self.criterion_cls(res['logits_r1_p'], label.long())
            loss_cls_r2 = self.criterion_cls(res['logits_r2_p'], label.long())
            loss_cls_shared = self.criterion_cls(res['logits_s'], label.long())
            loss_cls_aux = loss_cls_mm + loss_cls_r1 + loss_cls_r2 + loss_cls_shared
            
            loss_recon_mm = self.criterion_mse(res['m_recon'], res['proj_x_m'])
            loss_recon_r1 = self.criterion_mse(res['r1_recon'], res['proj_x_r1'])
            loss_recon_r2 = self.criterion_mse(res['r2_recon'], res['proj_x_r2'])
            loss_recon_origin = loss_recon_mm + loss_recon_r1 + loss_recon_r2
            
            # Recon Specific:
            # m_recon_p: (BS, Seq, Dim)
            # m_p: (Seq, BS, Dim)
            # Action: Transpose m_recon_p to match m_p
            loss_recon_s_mm = self.criterion_mse(res['m_recon_p'].transpose(0, 1), res['m_p'])
            loss_recon_s_r1 = self.criterion_mse(res['r1_recon_p'].transpose(0, 1), res['r1_p'])
            loss_recon_s_r2 = self.criterion_mse(res['r2_recon_p'].transpose(0, 1), res['r2_p'])
            loss_recon_specific = loss_recon_s_mm + loss_recon_s_r1 + loss_recon_s_r2
            
            dim = self.config['model']['embed_dim']
            loss_ort_mm = self.criterion_cosine(res['m_p'].reshape(-1, dim), res['m_s'].reshape(-1, dim), torch.tensor([-1]).to(self.device))
            loss_ort_r1 = self.criterion_cosine(res['r1_p'].reshape(-1, dim), res['r1_s'].reshape(-1, dim), torch.tensor([-1]).to(self.device))
            loss_ort_r2 = self.criterion_cosine(res['r2_p'].reshape(-1, dim), res['r2_s'].reshape(-1, dim), torch.tensor([-1]).to(self.device))
            loss_ort = loss_ort_mm + loss_ort_r1 + loss_ort_r2

            feats = torch.cat([res['m_s_sim'], res['r1_s_sim'], res['r2_s_sim']], dim=0)
            labels_all = torch.cat([label, label, label], dim=0)
            loss_contrast = self.criterion_contrast(feats, labels_all)

            loss = loss_classify + self.config['model']['hyper_parameters']['cls'] * loss_cls_aux +\
                                   self.config['model']['hyper_parameters']['recon_origin'] * loss_recon_origin +\
                                   self.config['model']['hyper_parameters']['recon_specific'] * loss_recon_specific +\
                                   self.config['model']['hyper_parameters']['ort'] * loss_ort +\
                                   self.config['model']['hyper_parameters']['contrastive'] * loss_contrast
            
            loss_info = {
                'loss': loss.item(),
                'cls': loss_classify.item(),
                'cls_aux_loss': loss_cls_aux.item(),
                'recon_origin': loss_recon_origin.item(),
                'recon_specific': loss_recon_specific.item(),
                'ort': loss_ort.item(),
                'contrastive': loss_contrast.item()
            }
        else:
            loss = loss_classify
            loss_info = {'loss': loss.item(), 'cls': loss_classify.item()}

        return loss, loss_classify, loss_info

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        avg_loss = Averager()
        avg_cls_loss = Averager()
        grad_acc_steps = self.config.get('gradient_accumulation_steps', 1)

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            batch = data2gpu(batch, self.config['use_cuda'])
            
            with autocast('cuda', enabled=self.config['use_cuda']):
                model_inputs = {**self.config, **batch}
                out = self.model(**model_inputs)
                if isinstance(out, tuple) and len(out) == 2:
                    res, debug_info = out
                else:
                    res = out
                    debug_info = None

                loss, cls_loss, loss_info = self.compute_loss(res, batch)
                loss = loss / grad_acc_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            avg_loss.add(loss.item())
            avg_cls_loss.add(cls_loss.item())
            
            if loss_info is None:
                pbar.set_postfix({'loss': avg_loss.item(), 'cls_loss': avg_cls_loss.item()})
            else:
                pbar.set_postfix(loss_info)
            
            global_step = epoch * len(train_loader) + step
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Cls_Loss', cls_loss.item(), global_step)

        return avg_loss.item()

    def evaluate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        misclassified_indices = []
        base_index = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
                batch = data2gpu(batch, self.config['use_cuda'])
                with autocast('cuda', enabled=self.config['use_cuda']):
                    model_inputs = {**self.config, **batch}
                    res, debug_info = self.model(**model_inputs)
                
                preds = torch.argmax(res['classify_pred'], dim=1).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

                if 'idx' in batch:
                    batch_ids = batch['idx']
                elif 'id' in batch:
                    batch_ids = batch['id']
                elif 'sample_id' in batch:
                    batch_ids = batch['sample_id']
                else:
                    batch_ids = list(range(base_index, base_index + len(labels)))
                
                for i, (p, y, sid) in enumerate(zip(preds, labels, batch_ids)):
                    if p != y:
                        misclassified_indices.append(sid)
                base_index += len(labels)
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        p_class, r_class, f_class, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1], average=None, zero_division=0)
        
        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fake_precision': p_class[0].item(),
            'fake_recall': r_class[0].item(),
            'fake_f1': f_class[0].item(),
            'real_precision': p_class[1].item(),
            'real_recall': r_class[1].item(),
            'real_f1': f_class[1].item()
        }
        info = {'misclassified_indices': misclassified_indices}
        return metrics, info

    def fit(self, train_loader, val_loader, test_loader=None):
        self.logger.info("Start training...")
        grad_acc_steps = self.config.get('gradient_accumulation_steps', 1)
        steps_per_epoch = int(math.ceil(len(train_loader) / float(grad_acc_steps)))
        num_training_steps = steps_per_epoch * self.config['epoch']
        self._init_scheduler(num_training_steps)

        for epoch in range(self.config['epoch']):
            train_loss = self.train_epoch(train_loader, epoch)
            self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
            val_metrics, val_info = self.evaluate(val_loader)
            self.logger.info(f"Epoch {epoch} | Val Metrics: {val_metrics}")
            
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'Val/{k}', v, epoch)
            
            is_best = self.recorder.add(val_metrics['f1'])
            if is_best:
                self.save_checkpoint('best_model.pth')
                self.logger.info("New best model saved!")
            
            if self.recorder.stop():
                self.logger.info("Early stopping triggered.")
                break
            
            #if test_loader is not None:
            #   epoch_test_metrics, epoch_test_info = self.evaluate(test_loader)
            #   self.logger.info(f"Epoch {epoch} | test Metrics: {epoch_test_metrics}")

        self.logger.info("Training finished.")
        
        if test_loader:
            self.load_checkpoint('best_model.pth')
            test_metrics, test_info = self.evaluate(test_loader)
            self.logger.info(f"Test Metrics: {test_metrics}")
            
            with open(self.config['test_save_path'], 'w') as f:
                json.dump({'metrics': test_metrics, 'info': test_info}, f, indent=4)
            return test_metrics
        
        return self.recorder.best_metric

    def save_checkpoint(self, filename):
        if not self.config['save_model']:
            return
        save_path = os.path.join(self.config['save_param_dir'], filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'recorder': {'best_metric': getattr(self.recorder, 'best_metric', None), 'history': getattr(self.recorder, 'history', None)},
            'config': self.config
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, filename):
        save_path = os.path.join(self.config['save_param_dir'], filename)
        ckpt = torch.load(save_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)

        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer') and ckpt['optimizer_state_dict'] is not None:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                self.logger.warning('Failed to load optimizer state dict — continuing without it.')

        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler') and ckpt['scheduler_state_dict'] is not None:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                self.logger.warning('Failed to load scheduler state dict — continuing without it.')