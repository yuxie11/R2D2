# Zero and R2D2: A Large-scale Chinese Cross-modal Dataset and A Vision-Language Framework



**CCMB: A Large-scale Chinese Cross-modal Benchmark (ACM MM 2023)**

This repo is the official implementation of CCMB and R2D2</a>. It includes datasets, code, and models as following:

<!-- &#x2705;<a href="http://zero.so.com">Zero benchmark</a> is available. The detailed introduction and download URL are in <font size=4>**http://zero.so.com**</font>. The 250M data is in -->

CCMB is available. It include pre-train dataset (Zero) and 5 downstream datasets. The detailed introduction and download URL are in <font size=4>**http://zero.so.com**</font>. The 250M data is in https://pan.baidu.com/s/1gnNbjOdCQdqZ4bRNN1S-Vw?pwd=iau8

&#x2705;Pre-trained checkpoints.

&#x2705;Inference demo.

&#x2705;Fine-tuning code and checkpoints for Image-Text Retrieval and Image-Text Matching tasks.

<!-- &#x274C;Pre-training code (coming soon). -->

<img src="image/framework.png">

## Requirements
<pre/>pip install -r requirements.txt</pre> 



## Pre-trained checkpoints
Pre-trained image-text pairs | R2D2<sub><font size=1.5>ViT-B</font></sub> | R2D2<sub><font size=1.5>ViT-L</font></sub> | PRD2<sub><font size=1.5>ViT-L</font></sub>
--- | :---:| :---: | :---:
250M | - | <a href="https://drive.google.com/file/d/18Fd3vGvj0Dz8rPlxROxugjZaF8Z4jf7g/view?usp=sharing">Download</a> | <a href="https://drive.google.com/file/d/15zDdam7_-YT0suA3Wc226vvxcyBxWZ_O/view?usp=sharing">Download
23M | - | <a href="https://drive.google.com/file/d/1vvvMv3mTRFGAUojbSJoZiTuqYPJqIquh/view?usp=sharing">Download</a> | -
<!-- 2.3M | - | <a href="https://drive.google.com/file/d/1SKH-d1Vd-1wn3qUt6YKnep7VsTXfbTK0/view?usp=sharing">Download</a> | - -->

## Fine-tuned checkpoints
Dataset | R2D2<sub><font size=1.5>ViT-B</font></sub>(23M) | R2D2<sub><font size=1.5>ViT-L</font></sub>(23M)
--- | :---: | :---:
Flickr-CNA | <a href="https://drive.google.com/file/d/1qgbDIqSUBqGz6rGCGKtW14wzPTIcnLLg/view?usp=sharing">Download</a> | -
IQR | <a href="https://drive.google.com/file/d/1lQ6rqMXukzul6XQJ8uZe_BQh-tuL1KNm/view?usp=sharing">Download</a> | -
ICR | <a href="https://drive.google.com/file/d/15Zsr8n49AEjOi2MkOfp1ZtUAKGss_Xbz/view?usp=sharing">Download</a> | -
IQM | <a href="https://drive.google.com/file/d/1JxLL6mlhDz_pjoUuyeeRVTHw0q8gW5et/view?usp=sharing">Download</a> | -
ICM | <a href="https://drive.google.com/file/d/1FI9RzJT-0j30ftcfkx0zDF2v3T7iXZtG/view?usp=sharing">Download</a> | -

## Inference demo
- To evaluate the pretrained R2D2 model on image-text pairs, run:
    <pre>python r2d2_inference_demo.py</pre> 
- To evaluate the pretrained PRD2 model on image-text pairs, run:
    <pre>python prd2_inference_demo.py</pre> 

## Downstream Tasks
1. Download datasets and pretrained models.
    for ICR, IQR, ICM, IQM tasks, after downloading you should see the following folder structure:
    ```
    ├── IQR_IQM_ICR_ICM_images
    │   
    ├── IQR
    │   ├── train
    │   └── val
    ├── ICR
    │   ├── train
    │   └── val
    ├── IQM
    │   ├── train
    │   └── val
    │── ICM
    │   ├── train
    │   └── val
    for Flickr30k-CNA, after downloading you should see the following folder structure:
    ```
    ├── Flickr30k-images
    │   
    ├── train
    │   
    ├── val
    │  
    └── test
    ```
  2. In config/retrieval_*.yaml, set the paths for the dataset and pretrain model paths.
  3. Run fine-tuning for the Image-Text Retrieval task.
      ```
      sh train_r2d2_retrieval.sh
      ```
  4. Run fine-tuning for the Image-Text Matching task.
      ```
      sh train_r2d2_matching.sh
      ```
    
### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@article{xie2022zero,
  title={Zero and R2D2: A Large-scale Chinese Cross-modal Benchmark and A Vision-Language Framework},
  author={Xie, Chunyu and Cai, Heng and Song, Jianfei and Li, Jincheng and Kong, Fanjing and Wu, Xiaoyu and Morimitsu, Henrique and Yao, Lin and Wang, Dexin and Leng, Dawei and Ji, Xiangyang and Deng, Yafeng },
  journal={arXiv preprint arXiv:2205.03860},
  year={2022}
}</pre>
