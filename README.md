# End-to-End Neural Discourse Deixis Resolution in Dialogue

## Overview

This repository stores code for our paper _[End-to-End Neural Discourse Deixis Resolution in Dialogue](https://aclanthology.org/2022.emnlp-main.778/)_. If you use our
code, please consider citing our paper.

```
@inproceedings{li-ng-2022-end,
    title = "End-to-End Neural Discourse Deixis Resolution in Dialogue",
    author = "Li, Shengjie  and
      Ng, Vincent",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.778",
    pages = "11322--11334",
    abstract = "We adapt Lee et al.{'}s (2018) span-based entity coreference model to the task of end-to-end discourse deixis resolution in dialogue, specifically by proposing extensions to their model that exploit task-specific characteristics. The resulting model, dd-utt, achieves state-of-the-art results on the four datasets in the CODI-CRAC 2021 shared task.",
}
```

Some code in this repository was borrowed
from [Xu and Choi's implementation of coref-hoi](https://github.com/lxucs/coref-hoi). Please check out their repository
too.

## Usage

### Setup

1. Install dependencies   
   ```pip install -r requirements.txt```
2. Set `data_dir` in [experiments.conf](experiments.conf)
3. Prepare data using the jupyter notebook [here](https://github.com/samlee946/utd-codi-crac2022/tree/main/data)


### Training

```python run.py config_name gpu_id random_seed```

### Batch Evaluation

```python run.py batch gpu_id random_seed```  
