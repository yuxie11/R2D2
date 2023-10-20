import os
import transformers
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.bert import BertConfig, BertModel
class R2D2(nn.Module):
    def __init__(
        self,
        vit_type='large',
        image_size=224,
        embed_dim=768,
    ):
        """
        Args:
            image_size (int): input image size
            embed_dim (int): output embedding size
        """
        super().__init__()
        if vit_type=='large':
            print('large')
            vision_width = 1024
            clip_config=transformers.CLIPConfig.from_pretrained('./checkpoints/r2d2_config/vision_config.json')
            num_patches = (image_size // 14)**2
        elif vit_type=='base':
            print('base')
            vision_width = 768
            clip_config=transformers.CLIPConfig.from_pretrained('./checkpoints/r2d2_config/vision_config_base.json')
            num_patches = (image_size // 16)**2
        clip = transformers.CLIPModel(config=clip_config).eval()
        self.visual_encoder = clip.vision_model

        if self.visual_encoder.embeddings.num_patches != num_patches:
            self.visual_encoder.embeddings.num_patches = num_patches
            self.visual_encoder.embeddings.position_embedding = nn.Embedding(num_patches + 1,
                                                                             self.visual_encoder.embeddings.embed_dim)
            self.visual_encoder.embeddings.position_ids = torch.arange(num_patches + 1).unsqueeze(0)

        self.tokenizer = BertTokenizer.from_pretrained('./checkpoints/hfl_roberta')
        text_config = BertConfig.from_pretrained('./checkpoints/hfl_roberta')
        self.text_encoder = transformers.BertModel(config=text_config)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # create joint layers
        bert_config='./checkpoints/r2d2_config/bert_config.json'
        encoder_config = BertConfig.from_json_file(bert_config)
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = 6
        self.text_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        encoder_config.encoder_width = text_width        
        encoder_config.vocab_size = 21128
        self.img_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        self.img_joint_proj = nn.Linear(vision_width, text_width)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.itm_head = nn.Linear(text_width, 2)
        self.itm_head_i = nn.Linear(text_width, 2)


    def tokenize_text(self, text):
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        return tokenized_text

    def encode_text(self, text):
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask).last_hidden_state
        text_embed = F.normalize(self.text_proj(text_output[:, 0, :]), dim=-1)
        return text_output, text_embed

    def encode_image(self, image):
        image_output = self.visual_encoder(image).last_hidden_state
        image_embed = F.normalize(self.vision_proj(image_output[:, 0, :]), dim=-1)
        return image_output, image_embed  

def r2d2(pretrained='', **kwargs):
    model = R2D2(**kwargs)
    if pretrained:
        model, _ = load_checkpoint(model, pretrained)
    return model
    
def load_checkpoint(model, url_or_filename):
    if os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    load_msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, load_msg
