#coding=utf-8
import copy
import torch
import transformers
from torch import nn
import torch.nn.functional as F
from models.bert import BertConfig, BertModel
from models.r2d2 import R2D2


class R2D2_Matching(R2D2):
    def __init__(self, model_type, embed_dim=768, image_size=224):
        super().__init__(vit_type=model_type, image_size=image_size, embed_dim=embed_dim)
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

    def forward(self, image, text, target):
        with torch.no_grad():
            image_embeds = self.visual_encoder(image).last_hidden_state
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder(text['input_ids'], attention_mask=text['attention_mask'], return_dict=True)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        text_hidden_state = text_output.last_hidden_state.clone()

        output_pos = self.text_joint_layer(
            encoder_embeds=text_hidden_state,
            attention_mask=text['attention_mask'],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        img_output_pos = self.img_joint_layer(
            encoder_embeds=self.img_joint_proj(image_embeds),
            attention_mask=image_atts,
            encoder_hidden_states=text_hidden_state,
            encoder_attention_mask=text['attention_mask'],
            return_dict=True,
        )

        vl_output = F.softmax(self.itm_head(output_pos.last_hidden_state[:, 0]), dim=-1)
        vl_output_i = F.softmax(self.itm_head_i(img_output_pos.last_hidden_state[:, 0]), dim=-1)
        pred = (vl_output + vl_output_i) / 2.0
        loss = F.cross_entropy(pred, target)
        return pred, loss


def create_matching_model(pretrained='', **kwargs):
    model = R2D2_Matching(**kwargs)
    print("init model done...")
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        for key in model.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != model.state_dict()[key].shape:
                    del state_dict[key]

        err = model.load_state_dict(state_dict, strict=False)
    return model
