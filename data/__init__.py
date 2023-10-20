import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode

from data.flickr30k_cna_dataset import flickr30k_cna_train, flickr30k_cna_retrieval_eval
from data.icr_dataset import icr_retrieval_train, icr_retrieval_eval
from data.iqr_dataset import iqr_retrieval_train, iqr_retrieval_eval
from data.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
   
    if dataset=='retrieval_flickrcna':
        print('flickrcna`')
        train_dataset = flickr30k_cna_train(transform_train, config['image_path'], config['ann_path'])
        val_dataset = flickr30k_cna_retrieval_eval(transform_test, config['image_path'], config['ann_path'], 'val') 
        test_dataset = flickr30k_cna_retrieval_eval(transform_test, config['image_path'], config['ann_path'], 'test')          
        return train_dataset, val_dataset, test_dataset   
    
    elif dataset=='retrieval_icr':          
        train_dataset = icr_retrieval_train(transform_train, config['image_path'], config['ann_path'])
        val_dataset = icr_retrieval_eval(transform_test, config['image_path'], config['ann_path'], 'val') 
        test_dataset = icr_retrieval_eval(transform_test, config['image_path'], config['ann_path'], 'test')          
        return train_dataset, val_dataset, test_dataset  

    elif dataset=='retrieval_iqr':          
        train_dataset = iqr_retrieval_train(transform_train, config['image_path'], config['ann_path'])
        val_dataset = iqr_retrieval_eval(transform_test, config['image_path'], config['ann_path'], 'val') 
        test_dataset = iqr_retrieval_eval(transform_test, config['image_path'], config['ann_path'], 'test')          
        return train_dataset, val_dataset, test_dataset    
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

