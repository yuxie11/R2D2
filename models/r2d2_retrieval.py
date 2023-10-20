import os
import transformers
from transformers import CLIPModel, CLIPConfig
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.r2d2 import R2D2, load_checkpoint

class R2D2_Retrieval(R2D2):
    def __init__(
        self,
        vit_type='large',
        image_size=224,
        embed_dim=768,
    ):
        super().__init__(vit_type, image_size=224, embed_dim=768)
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

    def forward(self, image, caption, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)   
            image_embeds = self.visual_encoder(image).last_hidden_state
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        
        text = self.tokenize_text(caption).to(image.device)
        text_output, text_feat = self.encode_text(text)        
        
        # Global Contrastive Pre-Ranking
        with torch.no_grad():
            idx = idx.view(-1, 1)
            idx_all = gather_with_no_grad(idx).t()
            pos_idx = torch.eq(idx, idx_all).float()
            cur_sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        loss_gcpr = self.compute_gcpr_loss(image_feat, text_feat, cur_sim_targets)
        
        # Fine-Grained Ranking
        text_hidden_state = text_output.clone()
        output_pos, img_output_pos = self.cross_output(text_hidden_state, text.attention_mask, image_embeds, image_atts)
        
        weights_i2t, weights_t2i = self.compute_it_similarity(idx, image_feat, text_feat)
                
        batch_size = image.size(0)
        # select negative image   
        image_embeds_neg = []
        total_image_embeds = gather_with_grad(image_embeds)
        for i in range(batch_size):
            neg_idx = torch.multinomial(weights_t2i[i], 1).item()
            image_embeds_neg.append(total_image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        # select negative image text
        text_atts_neg = []
        text_hidden_states_neg = []
        total_text_att_mask = gather_with_grad(text.attention_mask)
        total_text_hidden_state = gather_with_grad(text_hidden_state)
        for i in range(batch_size):
            neg_idx = torch.multinomial(weights_i2t[i], 1).item()
            text_atts_neg.append(total_text_att_mask[neg_idx])
            text_hidden_states_neg.append(total_text_hidden_state[neg_idx])
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        text_hidden_states_neg = torch.stack(text_hidden_states_neg, dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)
        text_hidden_states_all = torch.cat([text_hidden_state, text_hidden_states_neg], dim=0)
        
        output_neg, img_output_neg = self.cross_output(text_hidden_states_all, text_atts_all, image_embeds_all, image_atts_all)
        
        itm_labels = torch.cat(
            [torch.ones(1 * batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)], dim=0).to(image.device)
        loss_weight = torch.tensor([1.0, 2], device=image.device)
        loss_fgr  = self.compute_fgr_loss(output_pos, output_neg, img_output_pos, img_output_neg, loss_weight, itm_labels)

        return loss_gcpr, loss_fgr 

    def compute_it_similarity(self, idx, image_feat, text_feat):
        idxs = gather_with_no_grad(idx)
        with torch.no_grad():
            pos_idx = torch.eq(idx, idxs.T)
            total_image_feat = gather_with_no_grad(image_feat)
            total_text_feat = gather_with_no_grad(text_feat)
            sim_i2t = image_feat @ total_text_feat.T / self.temp
            sim_t2i = text_feat @ total_image_feat.T / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(pos_idx, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(pos_idx, 0)
        return weights_i2t, weights_t2i 

    def compute_gcpr_loss(self, image_feat, text_feat, cur_sim_targets):
        text_feats = gather_with_grad(text_feat)
        image_feats = gather_with_grad(image_feat)

        sim_i2t = image_feat @ text_feats.T / self.temp
        sim_t2i = text_feat @ image_feats.T / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * cur_sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * cur_sim_targets, dim=1).mean()

        loss_gcpr = (loss_i2t + loss_t2i) / 2
        return loss_gcpr
    
    def compute_fgr_loss(self, output_pos, output_neg, img_output_pos, img_output_neg, loss_weight, itm_labels):
        vl_embeddings = torch.cat([
            output_pos.last_hidden_state[:, 0],
            output_neg.last_hidden_state[:, 0],
        ], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        vl_embeddings_i = torch.cat([
            img_output_pos.last_hidden_state[:, 0],
            img_output_neg.last_hidden_state[:, 0],
        ], dim=0)
        vl_output_i = self.itm_head_i(vl_embeddings_i)
        
        loss_fgr_t = F.cross_entropy(vl_output, itm_labels, weight=loss_weight)
        loss_fgr_i = F.cross_entropy(vl_output_i, itm_labels, weight=loss_weight)
        loss_fgr  = (loss_fgr_t + loss_fgr_i) / 2
        return loss_fgr

    def cross_output(self, text_embeds, text_atts, image_embeds, image_atts):
        output = self.text_joint_layer(
            encoder_embeds=text_embeds,
            attention_mask=text_atts,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        img_output = self.img_joint_layer(
            encoder_embeds=self.img_joint_proj(image_embeds),
            attention_mask=image_atts,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=text_atts,
            return_dict=True,
        )
        return output, img_output

@torch.no_grad()
def gather_with_no_grad(tensor, sync_grad=False):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    if sync_grad:
        tensors_gather[torch.distributed.get_rank()] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output

class Grad_GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

def gather_with_grad(tensors):
    # get all the gathered tensors
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensors
    tensor_all = Grad_GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
                   
def create_r2d2_retrieval_model(pretrained='', **kwargs):
    model = R2D2_Retrieval(**kwargs)
    if pretrained:
        model, err = load_checkpoint(model, pretrained)
    return model
