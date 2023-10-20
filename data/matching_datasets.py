#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import json
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data.randaugment import RandomAugment

ImageFile.LOAD_TRUNCATED_IMAGES = True
prefix_map = {
        'icm': 'long',
        'iqm': 'short',
        }


class ITM360Dataset(Dataset):
    def __init__(self, ann_path, images_root, dataset, datatype):
        self.datatype = datatype
        self.tokenizer = BertTokenizer.from_pretrained('./checkpoints/hfl_roberta')
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]

        self.all_pair = []
        json_path = os.path.join(ann_path, datatype)
        json_filename = os.path.join(json_path, prefix_map[dataset]+"_itm_"+datatype+".json")
        with open(json_filename, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                anno = json.loads(line)
                img_name = os.path.join(images_root, anno['image_path'].split('/')[-1])
                self.all_pair.append([img_name, anno['text'], anno['label']])

        if datatype == 'train':
            self.transforms = transforms.Compose([                        
                    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(2,5, isPIL=True, augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                                      'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])        
        elif datatype in ['val', 'test']:
            self.transforms = transforms.Compose([
                    transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])  
        else:
            raise Exception('invalid datatype')
       
    def __getitem__(self, index):
        img_path, text, label = self.all_pair[index]

        outputs = self.tokenizer(text, max_length=64, truncation=True, padding='max_length')

        img = Image.open(img_path)
        img = img.convert('RGB')
        pixel_value = self.transforms(img).numpy()

        outputs['pixel_values'] = pixel_value
        outputs['label'] = label
        return outputs

    def __len__(self):
        return len(self.all_pair)


def itm_data_collator(examples):
    img_inputs = {}
    pixel_values = [e.pop('pixel_values') for e in examples]
    labels = [e.pop('label') for e in examples]
    input_ids = [e.pop('input_ids') for e in examples]
    token_type_ids = [e.pop('token_type_ids') for e in examples]
    attention_mask = [e.pop('attention_mask') for e in examples]
    text_inputs = {
            'input_ids': torch.tensor(input_ids),
            'token_type_ids': torch.tensor(token_type_ids),
            'attention_mask': torch.tensor(attention_mask),
            }
    pixel_values = torch.tensor(np.array(pixel_values), dtype=torch.float32).squeeze(1)
    img_inputs['pixel_values'] = pixel_values
    return img_inputs, text_inputs, torch.tensor(np.array(labels), dtype=torch.int64)
