from PIL import Image
import torch
import torch.nn.functional as F

from models.r2d2 import r2d2

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def preprocess(image, image_size):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    return transform(image)

@torch.no_grad()
def evaluation(model, texts, image, image_size, device): 
    print('Computing features for evaluation...')
    
    text_input = model.tokenize_text(texts).to(device)
    image = preprocess(Image.open(image).convert('RGB'), image_size).unsqueeze(0).to(device) 

    text_output, text_embeds = model.encode_text(text_input)
    image_feats, image_embeds = model.encode_image(image)

    sims_matrix = torch.einsum("ij,ij->i", [text_embeds, image_embeds])/model.temp
    
    encoder_output = image_feats.repeat(len(texts), 1, 1).to(device)
    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

    output = model.text_joint_layer(
            encoder_embeds=text_output,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )

    img_output = model.img_joint_layer(encoder_embeds=model.img_joint_proj(encoder_output),
                                       attention_mask=encoder_att,
                                       encoder_hidden_states=text_output,
                                       encoder_attention_mask=text_input.attention_mask)

    score = torch.softmax(model.itm_head(output.last_hidden_state[:, 0, :]),dim=-1)[:, 1]
    score = (score + torch.softmax(model.itm_head_i(img_output.last_hidden_state[:, 0, :]),dim=-1)[:, 1]) / 2
    sims_matrix = score

    return sims_matrix.cpu().numpy()

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_path = 'checkpoints/r2d2/r2d2_pretrain_250m.pth'
    model = r2d2(pretrained=checkpoint_path)
    model = model.to(device)
    model.eval()
    
    texts = ['一群脚踩雪地靴，穿着冬季的远足服装的人们，正站在一个建筑的前面，建筑看起来像是用冰块搭建而成的。', 
            '一个穿着格子花呢夹克衫的小男孩正在南瓜地里抓一个大南瓜',
            '一个厨师正忙碌地照顾着几个在炉具上燃烧的锅', 
            '这个穿蓝色短裤的男孩正在床上蹦蹦跳跳',
            '两名戴水肺的潜水员正在水下潜水，并遇到了一只友好的海豚',
            '三个穿着蓝色衬衫的孩子在秋千上荡秋千。']
    image = "image/test.jpg"
    image_size = 224
    score_test_it = evaluation(model, texts, image, image_size, device)
    print(score_test_it)

if __name__ == '__main__':
    main()
