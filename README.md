# Synthetic Data Generation
This is the official code for our paper "Comparative Analysis of Methods for Generating Heterogeneous Multi-dimensional Realistic Data"

## Models
We compare 5 different models:
- TVAE [official repo](https://github.com/sdv-dev/CTGAN) / [paper](https://arxiv.org/pdf/1907.00503.pdf)
- GAN / [paper](https://arxiv.org/pdf/1406.2661)
- CTGAN [official repo](https://github.com/sdv-dev/CTGAN) / [paper](https://arxiv.org/pdf/1907.00503.pdf)
- TabDDPM [official repo](https://github.com/yandex-research/tab-ddpm) / [paper](https://arxiv.org/pdf/2209.15421.pdf)
- Diffusion Model [code from](https://github.com/tanelp/tiny-diffusion) / [paper](https://arxiv.org/abs/2006.11239.pdf)

## Original Data
Data can be found [here](https://www.dropbox.com/scl/fo/vz49vv8tsbg5fquy690dp/ALUDJ2F49mSXhqzsddP_xF0?rlkey=sxs7lf2xlbgd8ndx3ctpqgztc&st=icwfdi5g&dl=0)

## How to use it
Execute the [notebook](Generation.ipynb) file.

1. Load the original data 
2. Create the datasets
3. Train the model you want to train
4. Sample data
5. Resample if you need more data
6. Display the results
