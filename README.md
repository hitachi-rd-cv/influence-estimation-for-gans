# Influence Estimation for Generative Adversarial Networks

This code reproduces the experiments in the following paper:

> Naoyuki Terashita, Hiroki Ohashi, Yuichi Nonaka, and Takashi Kanemaru
>
> [Influence Estimation for Generative Adversarial Networks](https://openreview.net/forum?id=opHLcXxYTC_)
>
> International Conference on Learning Representations (ICLR), 2021.


## Requirements

Experiments were conducted on Ubuntu 18.04 with Python 3.6.9 and CUDA 11.2. Other dependencies are summarized in `requirements.txt`.

## Running Experiments
### Preparation: extracting 2D-Normal dataset
- To make sure the generated 2D-Normal datasets correctly match those of the paper's setting, enter the root of this repository and run,  
    ~~~bash
    tar -zxvf data/processed_iclr_2d_valid.tar.gz -C .
    tar -zxvf data/processed_iclr_2d_cleansing.tar.gz -C .
    ~~~

### Experiment 1: Estimation Accuracy
1. Run the following command to reproduce the case of 2D-Normal & FCGAN & Influence on ALL
    ~~~bash
    LUIGI_CONFIG_PARSER=toml LUIGI_CONFIG_PATH=conf/2d_valid.toml python3 main.py TotalizeValid --local-scheduler
    ~~~
2. Run the following command to reproduce the case of MNIST & DCGAN & Influence on IS / FID
    ~~~bash
    LUIGI_CONFIG_PARSER=toml LUIGI_CONFIG_PATH=conf/mnist_valid.toml python3 main.py TotalizeValid --local-scheduler
    ~~~
3. Run [`plot_valid.ipynb`](plot_valid.ipynb) to reproduce Figure 1.

### Experiment 2: Data Cleansing
1. Run the following command to reproduce the case of 2D-Normal & FCGAN & Influence on ALL
    ~~~bash
    LUIGI_CONFIG_PARSER=toml LUIGI_CONFIG_PATH=conf/2d_cleansing.toml python3 main.py TotalizeCleansingWrtEval --local-scheduler
    ~~~
2. Run the following command to reproduce the case of MNIST & DCGAN & Influence on IS / FID
    ~~~bash
    LUIGI_CONFIG_PARSER=toml LUIGI_CONFIG_PATH=conf/mnist_cleansing.toml python3 main.py TotalizeCleansingWrtEval --local-scheduler
    ~~~
3. Run,
    - [`plot_cleansing.ipynb`](plot_cleansing.ipynb) to reproduce Figure 2.
    - [`plot_quality_2d.ipynb`](plot_quality_2d.ipynb) to reproduce Figure 3, Table 9 and 11.
    - [`plot_quality_mnist.ipynb`](plot_quality_mnist.ipynb) to reproduce Figure 4, Table 10 and 12.

## Citation

Please consider citing our paper if it helps your research:

```
@inproceedings{
terashita2021influence,
title={Influence Estimation for Generative Adversarial Networks},
author={Naoyuki Terashita and Hiroki Ohashi and Yuichi Nonaka and Takashi Kanemaru},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=opHLcXxYTC_}
}
```
---

If you have questions, please contact Naoyuki Terashita [naoyuki.terashita.sk@hitachi.com](mailto:naoyuki.terashita.sk@hitachi.com)
