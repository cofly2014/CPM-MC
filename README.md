##  Consistency Prototype Module and Motion Compensation for Few-Shot Action  Recognition (CLIP-CPM^2C)

## Abstract 

Recently, few-shot action recognition has significantly progressed by learning the feature discriminability and designing suitable comparison methods. 
Still, there are the following restrictions. (a) Previous works are mainly based on visual mono-modal. Although some multi-modal works use labels as 
supplementary to construct prototypes of support videos, they can not use this information for query videos. The labels are not used efficiently. 
(b) Most of the works ignore the motion feature of video, although the motion features are essential for distinguishing. We proposed a Consistency 
Prototype and Motion Compensation Network(CLIP-CPM^2C) to address these issues. Firstly, we use the CLIP for multi-modal few-shot action recognition 
with the text-image comparison for domain adaption. Secondly, in order to make the amount of information between the prototype and the query more similar, 
we propose a novel method to compensate for the text(prompt) information of query videos when text(prompt) does not exist, which depends on a Consistency Loss. 
Thirdly, we use the differential features of the adjacent frames in two directions as the motion features, which explicitly embeds the network with motion dynamics. 
We also apply the Consistency Loss to the motion features. Extensive experiments on standard benchmark datasets demonstrate that the proposed method can compete 
with state-of-the-art results. 

## Frame work of the paper
![framework](./framework.jpg)



[Paper] (https://arxiv.org/abs/2401.08345)


        
This paper has been accepted by Neurocomputing (https://doi.org/10.1016/j.neucom.2024.128649)

This repo contains code for the method introduced in the paper:
(https://github.com/cofly2014/CPM-MC)


## Where to download the dataset

##TODO

## How to configure the multiple level Segment Transformer
##TODO


## Splits
We used https://github.com/ffmpbgrnn/CMN for Kinetics and SSv2, which are provided by the authors of the authors of [CMN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.pdf) (Zhu and Yang, ECCV 2018). We also used the split from [OTAM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf) (Cao et al. CVPR 2020) for SSv2, and splits from [ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf) (Zhang et al. ECCV 2020) for HMDB and UCF.  These are all the in the splits folder.


## Citation
If you use this code/method or find it helpful, please cite:

    @article{GUO2025128649,
    title = {Consistency Prototype Module and Motion Compensation for few-shot action recognition (CLIP-CPM2C)},
    journal = {Neurocomputing},
    volume = {611},
    pages = {128649},
    year = {2025},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2024.128649},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231224014206},
    author = {Fei Guo and YiKang Wang and Han Qi and Li Zhu and Jing Sun},
    }
 

## Acknowledgements

We based our code on [CLIP-FSAR  CLIP-guided Prototype Modulating for Few-shot Action Recognition IJCV' 2023](https://github.com/alibaba-mmai-research/CLIP-FSAR) (logging, training, evaluation etc.).
We use [torch_videovision](https://github.com/hassony2/torch_videovision) for video transforms. 
We took inspiration from the Motion differ.) 