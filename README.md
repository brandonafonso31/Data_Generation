# Synthetic Data Generation
This is the official code for our paper "Comparative Analysis of Methods for Generating Heterogeneous Multi-dimensional Realistic Data"

## Models
We compare 5 different models:
- TVAE [official repo](https://github.com/sdv-dev/CTGAN) / [paper](https://arxiv.org/pdf/1907.00503.pdf)
- GAN / [paper](https://arxiv.org/pdf/1406.2661)
- CTGAN [official repo](https://github.com/sdv-dev/CTGAN) / [paper](https://arxiv.org/pdf/1907.00503.pdf)
- TabDDPM [official repo](https://github.com/yandex-research/tab-ddpm) / [paper](https://arxiv.org/pdf/2209.15421.pdf)
- Diffusion Model [implementation](https://github.com/tanelp/tiny-diffusion) / [paper](https://arxiv.org/abs/2006.11239.pdf)

## Original Data
Data can be found [here](https://www.dropbox.com/scl/fo/vz49vv8tsbg5fquy690dp/ALUDJ2F49mSXhqzsddP_xF0?rlkey=sxs7lf2xlbgd8ndx3ctpqgztc&st=icwfdi5g&dl=0)

## How to use it

1. Use the [Train notebook](Train.ipynb) to train the models. The trained models are automaticaly saved.
2. Use the [Sample notebook](Sample.ipynb) to generate new data. The trained models are automaticaly loaded. Set the variable "num_samples" accordingly.
3. Use the [Resample notebook](Resample.ipynb) to generate new data within a predefined variability percentage. For this, you need to sample first at least 3 times more data than the original data.
4. Use the [Stats notebook](Stats.ipynb) to display the graphs and the statistics.

## Metrics

- PRDC [official repo](https://github.com/clovaai/generative-evaluation-prdc) / [paper](https://arxiv.org/pdf/2002.09797)
- MMD [implementation](https://github.com/jindongwang/transferlearning/tree/master/code/distance)
- Wasserstein [library pot](https://pythonot.github.io/)
