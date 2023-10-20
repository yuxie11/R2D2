#coding:utf-8
import os
import sys
import time
import yaml
import json
import random
import torch
import datetime
import logging
import argparse
import traceback
import numpy as np            
from tqdm import tqdm
from sklearn import metrics
from logging import getLogger
import pytorch_lightning as pl
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.plugins import DDPPlugin

from data.matching_datasets import *
from solver.lr_scheduler import OffsetCyclicLR
from models.r2d2_matching import create_matching_model


def seed_everything(seed=2021):
    pl.seed_everything(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = float(cfg['base_lr'])
        weight_decay = cfg['weight_decay']
        if "bias" in key:
            lr = float(cfg['base_lr']) * 2
            weight_decay = 0
        params += [{
            "name": key, 
            "params": [value], 
            "lr": lr, 
            "weight_decay": weight_decay, 
            "freeze": False
            }]
    optimizer = getattr(torch.optim, 'AdamW')(params)
    return optimizer


class Lite(LightningLite):
    def eval(self, model, dataloader):
        model.eval()
        total_pred = []   
        total_targets = []
        with torch.no_grad():
            for j, batch in enumerate(tqdm(dataloader)):
                img_inputs, text_inputs, targets = batch
                img_inputs = img_inputs.pop('pixel_values')
                img_inputs = img_inputs.to(self.device)
                pred, loss = model(img_inputs, text_inputs, targets)
                total_targets.extend(targets.cpu().tolist())
                total_pred.append(pred)
            total_pred = torch.cat(total_pred, dim=0)
            self.barrier()
            total_targets = torch.Tensor(total_targets)
            total_targets = self.all_gather(total_targets, sync_grads=False).reshape(-1, *total_targets.shape[1:])
            total_pred = self.all_gather(total_pred, sync_grads=False).reshape(-1, *total_pred.shape[1:])
        
        if self.is_global_zero:
            label = total_targets.cpu().numpy()
            pred  = total_pred.cpu().numpy()[:, 1]
            auc = metrics.roc_auc_score(label, pred)
            return auc
        return 0.0

    def run(self, model, cfg, start_step=0):
        log_steps = 10
        num_epochs = cfg['num_epochs']
        global_step = start_step

        optimizer = make_optimizer(cfg, model)
        lr_scheduler = OffsetCyclicLR(optimizer,
                               base_lr=float(cfg['base_lr']) * 0.05,
                               max_lr=float(cfg['base_lr']),
                               step_size_up=cfg['warmup_steps'],
                               cycle_momentum=False,
                               offset=start_step)
        model, optimizer = self.setup(model, optimizer)
        logger = getLogger(cfg['model_name'])

        valid_dataset = ITM360Dataset(cfg['ann_path'], cfg['image_path'], cfg['data'], 'val')
        valid_dataloader = DataLoader(
            valid_dataset,
            shuffle=False,
            collate_fn=lambda x: itm_data_collator(x),
            batch_size=cfg['valid_batch_size'],
            num_workers=cfg['num_workers'],
            drop_last=False,
        )
        valid_dataloader = self.setup_dataloaders(valid_dataloader)
        print("val data num: ", len(valid_dataloader))

        train_dataset = ITM360Dataset(cfg['ann_path'], cfg['image_path'], cfg['data'], 'train')
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=lambda x: itm_data_collator(x),
            batch_size=cfg['train_batch_size'],
            num_workers=cfg['num_workers'],
            drop_last=True,
        )
        train_dataloader = self.setup_dataloaders(train_dataloader)
        print("train data num: ", len(train_dataloader))

        for epoch in range(0, num_epochs):
            # eval
            valid_auc = self.eval(model, valid_dataloader)
            self.barrier('eval_end')
            if self.is_global_zero:
                log_stats = {'valid_auc': valid_auc, 'epoch': epoch}
                logger.info(log_stats)

                if not os.path.exists(cfg['output_dir']):
                    os.system("mkdir -p %s"%cfg['output_dir'])
                with open(os.path.join(cfg['output_dir'], "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            if cfg['evaluate']:
                break

            model.train()
            metric = {'loss': 0.0, 'log_step': 0.0}
            optimizer.zero_grad()
            for j, batch in enumerate(tqdm(train_dataloader)):
                img_inputs, text_inputs, targets = batch
                img_inputs = img_inputs.pop('pixel_values')
                img_inputs = img_inputs.to(self.device)
                pred, loss = model(img_inputs, text_inputs, targets)
                self.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                metric['loss'] += loss.detach()
                metric['log_step'] += 1
                if (global_step + 1) % log_steps == 0 and (metric['log_step'] >= log_steps - 1):
                    log_dict = {'loss': metric["loss"] / metric['log_step'],
                                'global_step': global_step,
                                'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                }
                    if self.is_global_zero:
                        logger.info(log_dict)
                        metric = {'loss': 0.0, 'log_step': 0.0}
                self.barrier('step_end')

            if self.is_global_zero:
                save_obj = {
                    'model': model.module.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.exists(cfg['output_dir']):
                    os.system("mkdir -p %s"%cfg['output_dir'])
                modelname = "epoch%s.pth"%epoch
                savefile = os.path.join(cfg['output_dir'], modelname)
                torch.save(save_obj, savefile)
            self.barrier('epoch_end')

        self.barrier()
                                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--config', required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='checkpoints/output')  
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['output_dir'] = args.output_dir
    cfg['evaluate'] = args.evaluate

    seed_everything(args.seed)
    ckpt = cfg['pretrained_model'] if not args.checkpoint else args.checkpoint
    model = create_matching_model(pretrained=ckpt, model_type=cfg['vit_type'])
    print('resume from: ', ckpt, flush=True)

    start_step = 0
    logger = getLogger(cfg['model_name'])
    logger.setLevel(logging.DEBUG)
    ddp_plugin = DDPPlugin(find_unused_parameters=True)
    lite = Lite(
        strategy=ddp_plugin,
        gpus=cfg['gpus'],
        accelerator='gpu',
        precision=16,
        num_nodes=int(os.environ.get('workers', '1'))
    )
    try:
        lite.run(model, cfg, start_step)
    except:
        logger.error(traceback.format_exc())

