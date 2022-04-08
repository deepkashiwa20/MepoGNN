# MepoGNN: Metapopulation Epidemic Forecasting with Graph Neural Networks

## Introduction
In this study, we use historical daily infection data and human mobility data to implement epidemic forecasting for the total 47 prefectures of Japan. 
We propose a novel hybrid model called MepoGNN for multi-step (day) multi-region (prefecture) infection number prediction by incorporating spatio-temporal Graph Neural Networks (GNNs) and graph learning mechanisms into Metapopulation SIR model.

## Data Description
#### jp20200401_20210921.npy 
contains a dictionary of three numpy array: 'node' for node features; 'SIR' for S, I, R data; 'od' for OD flow data.
#### commute_jp.npy 
contains commuter flow data. 

### Input and output
* Input node features: historical daily confirmed cases, daily movement change, the ratio of daily confirmed cases in active cases and day of week. 
* Input for adaptive graph learning: commuter survey data
* Input for dynamic graph learning: OD flow data
* Output target: future daily confirmed cases


## Installation Dependencies
Working environment and major dependencies:
* Ubuntu 18.04.5 LTS
* Python 3 (3.8; Anaconda Distribution)
* NumPy (1.19.5)
* Pytorch (1.9.0)

## Run Model

Download this project into your device， then run the following:

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
