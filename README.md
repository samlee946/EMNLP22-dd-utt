# End-to-End Neural Discourse Deixis Resolution in Dialogue

## Overview

This repository stores code for our paper _End-to-End Neural Discourse Deixis Resolution in Dialogue_. If you use our
code, please consider citing our paper.

```
Bibtex will appear here once it's ready.
```

Some code in this repository was borrowed
from [Xu and Choi's implementation of coref-hoi](https://github.com/lxucs/coref-hoi). Please check out their repository
too.

## Usage

### Setup

1. Install dependencies   
   ```pip install -r requirements.txt```
2. Set `data_dir` in [experiments.conf](experiments.conf)
3. Prepare data (I haven't cleaned up the scripts for data preprocessing yet. If you are interested, send me an email
   at [sxl180006@hlt.utdallas.edu](mailto:sxl180006@hlt.utdallas.edu)
   )

### Training

```python run.py config_name gpu_id random_seed```

### Batch Evaluation

```python run.py batch gpu_id random_seed```  
