import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.prd2 import prd2


def preprocess(image, image_size):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    return transform(image)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = 'checkpoints/prd2/prd2_pretrain_250m.pth'
    model = prd2(pretrained = ckpt)
    model = model.to(device)
    model.eval()
    image = preprocess(Image.open("image/test.jpg").convert('RGB'), image_size=224).unsqueeze(0).to(device)    
    Chinese_text = ['一群脚踩雪地靴，穿着冬季的远足服装的人们，正站在一个建筑的前面，建筑看起来像是用冰块搭建而成的。', 
                    '一个穿着格子花呢夹克衫的小男孩正在南瓜地里抓一个大南瓜',
                    '一个厨师正忙碌地照顾着几个在炉具上燃烧的锅', 
                    '这个穿蓝色短裤的男孩正在床上蹦蹦跳跳',
                    '两名戴水肺的潜水员正在水下潜水，并遇到了一只友好的海豚',
                    '三个穿着蓝色衬衫的孩子在秋千上荡秋千。']
    text = model.tokenize_text(Chinese_text).to(device)
    
    with torch.no_grad():
        image_embedding = model.encode_image(image)
        text_embedding = model.encode_text(text)       
        '''
            searching images by a text
        '''
        # sims_matrix = text_embedding @ image_embedding.t() / model.temp
        ''' 
            searching texts by an image
        '''
        sims_matrix = image_embedding @ text_embedding.t() / model.temp
        results = sims_matrix.softmax(dim=-1).cpu().numpy()
    print("Label probabilities:", results)

if __name__ == '__main__':
    main()