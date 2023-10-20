import argparse
import os
import traceback
import ruamel.yaml as yaml
import numpy as np
import random
import math
import json
from shutil import copyfile
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.r2d2_retrieval import create_r2d2_retrieval_model
import utils
from utils import cosine_lr_schedule, warmup_lr_schedule
from data import create_dataset, create_sampler, create_loader

def train(model, data_loader, optimizer, epoch, device, config):
    model.train()
    for i, (image, caption, idx) in enumerate(tqdm(data_loader)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        current_step = epoch * len(data_loader) + i
        if epoch == 0 or current_step < config['warmup_steps']:
            warmup_lr_schedule(optimizer, current_step, config['warmup_steps'], config['warmup_lr'], config['base_lr'])

        loss_gcpr, loss_fgr = model(image, caption, idx=idx)
        loss = loss_gcpr + loss_fgr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    log_dict = {'loss': loss.item(),
                'loss_gcpr': loss_gcpr.item(),
                'loss_fgr': loss_fgr.item(),
                'lr': optimizer.param_groups[0]["lr"],
                'step': current_step}
    return log_dict

@torch.no_grad()
def compute_text_features(model, texts, device):
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    text_atts = []
    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i:min(num_text, i + text_bs)]
        text_input = model.tokenize_text(text).to(device)
        text_feat, text_embed = model.encode_text(text_input)
        text_feats.append(text_feat)
        text_embeds.append(text_embed)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    return text_embeds, text_feats, text_atts

@torch.no_grad()
def compute_image_features(model, data_loader, device):
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_input = image.to(device)
        image_feat, image_embed = model.encode_image(image_input)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    return image_embeds, image_feats

@torch.no_grad()
def compute_cross_score(model, sims_matrix, image_feats, text_feats, text_atts, device, is_i2t):
    top_k = config['top_k']
    cross_score_matrix = torch.full((sims_matrix.size(0), sims_matrix.size(1)), -100.0).to(device)
    start,end = get_dist_data_para(sims_matrix.size(0))

    for i, sims in enumerate(tqdm(sims_matrix[start:end])):
        topk_sim, topk_idx = sims.topk(k=top_k, dim=0)
        if is_i2t:
            image_select_feats = image_feats[start + i].repeat(top_k, 1, 1).to(device)
            image_select_att = torch.ones(image_select_feats.size()[:-1], dtype=torch.long).to(device)
            text_select_feats = text_feats[topk_idx]
            text_select_atts = text_atts[topk_idx]
        else:
            image_select_feats = image_feats[topk_idx].to(device)
            image_select_att = torch.ones(image_select_feats.size()[:-1], dtype=torch.long).to(device)
            text_select_feats = text_feats[start + i].repeat(top_k, 1, 1)
            text_select_atts = text_atts[start + i].repeat(top_k, 1)
        text_output, img_output = model.cross_output(text_select_feats, text_select_atts, image_select_feats, image_select_att)
        score = torch.softmax(model.itm_head(text_output.last_hidden_state[:, 0, :]),dim=-1)[:, 1]
        score = (score + torch.softmax(model.itm_head_i(img_output.last_hidden_state[:, 0, :]),dim=-1)[:, 1]) / 2
        cross_score_matrix[start + i, topk_idx] = score + topk_sim
    return cross_score_matrix

def get_dist_data_para(total_num):
    pre_gpu_num = math.ceil(total_num/utils.get_world_size())
    start = utils.get_rank() * pre_gpu_num
    end = min(total_num, start + pre_gpu_num)

    return start, end

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()
    texts = data_loader.dataset.text
    text_embeds, text_feats, text_atts = compute_text_features(model, texts, device)
    image_embeds, image_feats = compute_image_features(model, data_loader, device)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = compute_cross_score(model, sims_matrix, image_feats, text_feats, text_atts, device, is_i2t=True)
    score_matrix_t2i = compute_cross_score(model, sims_matrix.t(), image_feats, text_feats, text_atts, device, is_i2t=False)

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    #compute image to text recalls
    image_num = scores_i2t.shape[0]
    rank1 = 0
    rank5 = 0
    rank10 = 0
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        for k in range(10): 
            if inds[k] in img2txt[index]:
                rank10 = rank10+1
                if k<5:
                    rank5 = rank5+1
                    if k<1:
                        rank1 = rank1+1
                break
    tr1 = 100.0 * rank1 / image_num
    tr5 = 100.0 * rank5 / image_num
    tr10 = 100.0 * rank10 / image_num

    #compute text to image recalls
    text_num = scores_t2i.shape[0]
    rank1 = 0
    rank5 = 0
    rank10 = 0
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        for k in range(10): 
            if inds[k] == txt2img[index]:
                rank10 = rank10+1
                if k<5:
                    rank5 = rank5+1
                    if k<1:
                        rank1 = rank1+1
                break
    ir1 = 100.0 * rank1 / text_num
    ir5 = 100.0 * rank5 / text_num
    ir10 = 100.0 * rank10 / text_num

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': tr_mean,
        'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': ir_mean,
        'r_mean': r_mean
    }
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    rand_seed = args.seed + utils.get_rank()
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = True

    # create model
    print("start creat model")
    if len(args.checkpoint) > 0:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = config['pretrained_model']
    model = create_r2d2_retrieval_model(pretrained=checkpoint_path, vit_type=config['vit_type'])
    model = model.to(device)
    model_without_dist = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_dist = model.module
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['base_lr'], weight_decay=config['weight_decay'])

    #creating dataset
    print("start creat dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s' % config['data'], config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],
                                                          samplers,
                                                          batch_size=[config['train_batch_size']] +
                                                          [config['test_batch_size']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    best_result = 0
    best_epoch = 0
    for epoch in range(0, config['num_epoch']):
        if args.evaluate:
            score_test_i2t, score_test_t2i = evaluation(model_without_dist, test_loader, device, config)
            if utils.is_main_process():
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                result_log = {
                    **{f'val_{k}': v
                       for k, v in test_result.items()},
                }
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(result_log) + "\n")
            break
        else:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['num_epoch'], config['base_lr'], config['min_lr'])
            train_log = train(model, train_loader, optimizer, epoch, device, config)
            score_val_i2t, score_val_t2i, = evaluation(model_without_dist, test_loader, device, config)
        
            if utils.is_main_process():
                test_result = itm_eval(score_val_i2t, score_val_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                print(f'test: {test_result}')
    
                if test_result['r_mean'] > best_result:
                    save_obj = {
                        'model': model_without_dist.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best_result = test_result['r_mean']
                    best_epoch = epoch
                result_log = {
                    **{f'train_{k}': v
                       for k, v in train_log.items()},
                    **{f'test_{k}': v
                       for k, v in test_result.items()},
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(result_log) + "\n")

        dist.barrier()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--config', default='./configs/retrieval_flickr_large.yaml')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')  
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    copyfile(args.config, os.path.join(args.output_dir, 'config.yaml'))

    try:
        main(args, config)
    except:
        error = traceback.format_exc()
        print(error)
        raise Exception(error)