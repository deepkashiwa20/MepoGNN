# MepoGNN: Metapopulation Epidemic Forecasting with Graph Neural Networks


#### [ECMLPKDD22] Q. Cao, R. Jiang#, C. Yang, Z. Fan, X. Song, R. Shibasaki, "MepoGNN: Metapopulation Epidemic Forecasting with Graph Neural Networks", Proc. of the 26th European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECMLPKDD), 2022.

#### Code and data are now available.
```bibtex
@inproceedings{cao2022mepognn,
  title={Mepognn: Metapopulation epidemic forecasting with graph neural networks},
  author={Cao, Qi and Jiang, Renhe and Yang, Chuang and Fan, Zipei and Song, Xuan and Shibasaki, Ryosuke},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={453--468},
  year={2022},
  organization={Springer}
}
```
#### Long Version of The Conference Paper
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=MepoGNN&color=red&logo=arxiv)](https://arxiv.org/abs/2306.14857)

## Introduction
We propose a novel hybrid model called MepoGNN for multi-step (day) multi-region (prefecture) infection number prediction by incorporating spatio-temporal Graph Neural Networks (GNNs) and graph learning mechanisms into Metapopulation SIR model.
 
 In this study, we use historical daily infection data and human mobility data to implement epidemic forecasting for the total 47 prefectures of Japan.

## Data Description
#### jp20200401_20210921.npy 
contains a dictionary of three numpy array: 'node' for node features; 'SIR' for S, I, R data; 'od' for OD flow data.
#### commute_jp.npy 
contains commuter survey data. 

#### Input and Output
* Input node features: historical daily confirmed cases, daily movement change, the ratio of daily confirmed cases in active cases and day of week. 
* Input for adaptive graph learning: commuter survey data
* Input for dynamic graph learning: OD flow data
* Output: predicted daily confirmed cases


## Installation Dependencies
Working environment and major dependencies:
* Ubuntu 18.04.5 LTS
* Python 3 (3.8; Anaconda Distribution)
* NumPy (1.19.5)
* Pytorch (1.9.0)

## Run Model

Download this project into your device, then run the following:

``
cd /model
``


Choose graph learning type (Adaptive or Dynamic) and run the main program to train, validate and test on GPU 0:

``
python Main.py -GPU cuda:0 -graph Adaptive
``

``
python Main.py -GPU cuda:0 -graph Dynamic
``
