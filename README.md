# FlashRNA - An Efficient Model for Regulatory Genomics

An efficient genomic sequence-to-function model that significantly improves computational and memory efficiency while maintaining high predictive performance.

## Overview

FlashRNA is a sequence-to-function model that significantly improves computational efficiencies of existing transformer-based models in regulatory genomics while maintaining competitive performance. FlashRNA leverages FlashAttention to address high computational costs associated with self-attention layers. Combined with additional improvements in model architecture training setup, FlashRNA significantly reduces training and inference speed. Notably, FlashRNA can be trained from scratch without depending on another pre-trained model. 

FlashRNA is trained on a large set of functional genomics tracks, including RNA-seq, DNase-seq, and ATAC-seq. Here, we train FlashRNA with the open-sourced that was processed and shared by the authors of [Borzoi](https://github.com/calico/borzoi/blob/main/README.md), after some additional processing. However, the model can be trained from scratch on any similar genomic track dataset.

This repository contains the FlashRNA model code, instructions for downloading model weights, and example usage code. We plan on open-sourcing code for model training and evaluation shortly, along with more details on our approach.

## Setup

### Installation

```bash
conda env create -f environment.yml
conda activate flashrna
pip install .
```

## Usage

We have provided the model itself for use. Open-source training and inference setup is coming soon.

Important: FlashRNA uses [FlashAttention](https://github.com/Dao-AILab/flash-attention) and currently only runs on GPUs that support FlashAttention. Support for GPUs incompatible with FlashAttention is coming soon. 

### Example Usage

Please check out [`examples/basic_setup.ipynb`](examples/basic_setup.ipynb)

More comprehensive usage examples, along with useful helper functions for inference, will be 
added soon!

### Pre-trained Models

Pre-trained FlashRNA models are available as Wandb Artifacts:

**Single FlashRNA models (4 replicates):**
- `deep-genomics-open-source/FlashRNA/single-model-rep-1:v0` ([link](https://wandb.ai/deep-genomics-open-source/FlashRNA/artifacts/model/single-model-rep-1/v0))
- `deep-genomics-open-source/FlashRNA/single-model-rep-2:v0` ([link](https://wandb.ai/deep-genomics-open-source/FlashRNA/artifacts/model/single-model-rep-2/v0))
- `deep-genomics-open-source/FlashRNA/single-model-rep-3:v0` ([link](https://wandb.ai/deep-genomics-open-source/FlashRNA/artifacts/model/single-model-rep-3/v0))
- `deep-genomics-open-source/FlashRNA/single-model-rep-4:v0` ([link](https://wandb.ai/deep-genomics-open-source/FlashRNA/artifacts/model/single-model-rep-4/v0))

**Distilled model:**
- `deep-genomics-open-source/FlashRNA/distilled-model:v0` ([link](https://wandb.ai/deep-genomics-open-source/FlashRNA/artifacts/model/distilled-model/v0))

These models can be loaded using the following methods:
```python
from flash_rna.models import FlashRNA

# Requires Wandb login
model = FlashRNA.from_ckpt(wandb_artifact="deep-genomics-open-source/FlashRNA/single-model-rep-1:v0")

# Alternatively, manually download the checkpoint files from URLs
model = FlashRNA.from_ckpt(ckpt_path="path/to/downloaded/model.ckpt")
```

## Contact

Please contact andrew.jung (at) deepgenomics.com, andrewjung (at) psi.toronto.edu, or open a GitHub issue for questions. 

## Preprint and Citation

We will be releasing a preprint shortly. In the meanwhile, please cite this repo if you use it in your work.

```
@misc{flashrna2025,
  author       = {Andrew J. Jung, and contributors},
  title        = {FlashRNA - An Efficient Model for Regulatory Genomics},
  year         = {2025},
  howpublished = {\url{https://github.com/deepgenomics/flashrna_internal}},
  note         = {Efficient genomic sequence-to-function model with FlashAttention and additional optimizations in model architecture and training setup.},
  abstract     = {FlashRNA significantly improves computational and memory efficiency through FlashAttention, advancements in model architecture, and optimized training setup. This efficiency comes at no cost to performance or dependency on pretrained weights. Trained from scratch, FlashRNA achieves comparable or slightly improved predictive performance compared to similar-sized models.},
}
```

## Acknowledgements

We thank the authors of [Borzoi](https://github.com/calico/borzoi) for providing a comprehensive open-sourced repo to reproduce their dataset, model, training, and evaluatons. We also like to thank the authors [Flashzoi](https://github.com/johahi/borzoi-pytorch/tree/main) for sharing PyTorch version of Borzoi, implementing efficient relative shift operation, and their model Flashzoi. 

If you use FlashRNA in your research, please also consider citing them.

This work was also pursued as part of the first author's academic research at the University of Toronto. We would like to acknowledge Deep Genomics and the Vector Institute for compute resources.

(C) Deep Genomics Inc. (2025)
FlashRNA is licensed under the Apache License, Version 2.0 (the "License"); you may not use FlashRNA except in compliance with the License.