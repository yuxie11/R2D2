import copy
import os
import json

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class icr_retrieval_train(Dataset):

    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):
        filename = 'train/long_itr_train.json'

        self.annotation = []
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0

        with open(os.path.join(ann_root, filename), 'r', encoding='utf8') as f:
            for line in f:
                json_d = json.loads(line)
                text = json_d['text']
                image_path = json_d['image_path'].split('/')[-1]
                image_id = image_path[:-4]
                if image_id not in self.img_ids.keys():
                    self.img_ids[image_id] = n
                    n += 1
                self.annotation.append({'image': image_path, 'caption': text, 'image_id': image_id})

    def __len__(self):
        print(len(self.annotation))
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + ann['caption']
        return image, caption, self.img_ids[ann['image_id']]


class icr_retrieval_eval(Dataset):

    def __init__(self, transform, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val': 'val/long_itr_val.json', 'test': 'val/long_itr_val.json'}
        self.split = split

        self.annotation = []
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0

        self.img_ids = dict()
        img_id = 0
        with open(os.path.join(ann_root, filenames[split]), 'r', encoding='utf8') as f:
            for line in f:
                json_d = json.loads(line)
                text = json_d['text']
                image_path = json_d['image_path'].split('/')[-1]
                img_fn = image_path[:-4]
                tmp_id = copy.deepcopy(img_id)
                if img_fn not in self.img_ids.keys():
                    self.img_ids[img_fn] = img_id
                    self.image.append(image_path)
                    self.annotation.append({'image': image_path, 'image_id': img_id})
                    img_id += 1
                else:
                    tmp_id = self.img_ids[img_fn]
                if tmp_id in self.img2txt.keys():
                    self.img2txt[tmp_id].append(txt_id)
                else:
                    self.img2txt[tmp_id] = [txt_id]
                self.txt2img[txt_id] = tmp_id
                self.text.append(text)
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, self.annotation[index]['image_id']
