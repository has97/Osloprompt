# OSLOPROMPT: Bridging Low-Supervision Challenges and Open-Set Domain Generalization in CLIP (CVPR 2025)

Official repository of OSLOPROMPT, one of the first works in Low-shot Open Domain Generalization using pre-trained vision-language model (VLM) [CLIP](https://arxiv.org/abs/2503.16106) to focus on the completely unlablled real-world open samples in low-shot scenarios.

[![paper](https://img.shields.io/badge/Conference-Paper-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/C_OSLoPrompt_Bridging_Low-Supervision_Challenges_and_Open-Set_Domain_Generalization_in_CLIP_CVPR_2025_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2503.16106)

## Abstract
<img src="https://github.com/has97/Osloprompt/blob/main/main_latest-1.png" width="1000">

We introduce Low-Shot Open-Set Domain Generalization (LSOSDG), a novel paradigm unifying low-shot learning with open-set domain generalization (ODG). While prompt-based methods using models like CLIP have advanced DG, they falter in low-data regimes (e.g., 1-shot) and lack precision in detecting open-set samples with fine-grained semantics related to training classes. 
To address these challenges, we propose OSLoPrompt, an advanced prompt-learning framework for CLIP with two core innovations. First, to manage limited supervision across source domains and improve DG, we introduce a domain-agnostic prompt-learning mechanism that integrates adaptable domain-specific cues and visually guided semantic attributes through a novel cross-attention module, besides being supported by learnable domain- and class-generic visual prompts to enhance cross-modal adaptability.  Second, to improve outlier rejection during inference, we classify unfamiliar samples as unknown and train specialized prompts with systematically synthesized pseudo-open samples that maintain fine-grained relationships to known classes, generated through a targeted query strategy with off-the-shelf foundation models. This strategy enhances feature learning, enabling our model to detect open samples with varied granularity more effectively. 
Extensive evaluations across five benchmarks demonstrate that OSLoPrompt establishes a new state-of-the-art in LSOSDG, significantly outperforming existing methods
## Architecture

![Alt text](https://github.com/has97/Osloprompt/blob/main/main_latest-1.png "Title")

## How to install

Please follow the [ODG-CLIP](https://github.com/mainaksingha01/ODG-CLIP) Repo for installation.

To the run model for a specific dataset
 
 ```
$ python pacs.py \
        --source_domains "$source_str" \
        --target_domain "$target" \
        --shots $shots \
        --config $config \
        --output_dir "./experiments_mini" \
        --data_root "$data_root"
```
where source_str is source domains (as a single string seperated by commas) given in the sorted order and $target represents the target.  

- Change the source domains and target domain orders accordingly . For examples, source_domains = 'art_painting,cartoon,photo', target= 'sketch'.
- Results will be saved in output_dir provided.
- Training models of each epoch will also be saved in `output_dir` folder.


## Bibtex

Please cite the paper if you use our work . Thanks.

```
@InProceedings{C_2025_CVPR,
    author    = {C, Mohamad Hassan N and Gupta, Divyam and Singha, Mainak and Rongali, Sai Bhargav and Jha, Ankit and Khan, Muhammad Haris and Banerjee, Biplab},
    title     = {OSLoPrompt: Bridging Low-Supervision Challenges and Open-Set Domain Generalization in CLIP},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {10110-10120}
}
```

