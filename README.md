# Deep Learning Project

## Authors:

* Mykhailo Kruts
* Yevhen Zinenko
* Oleksandr Shmelov

## Description

Dataset used for this project:
- FER2013+

The FER+ annotations provide a set of new labels for the standard Emotion FER dataset. In FER+, each image has been labeled by 10 crowd-sourced taggers, which provide better quality ground truth for still image emotion than the original FER labels. Having 10 taggers for each image enables researchers to estimate an emotion probability distribution per face. This allows constructing algorithms that produce statistical distributions or multi-label outputs instead of the conventional single-label output,

![image](https://raw.githubusercontent.com/Microsoft/FERPlus/master/FER+vsFER.png)

## Pre-requisites

* Python v3.10

## Set-up

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate FER2013+

Assuming you are in the current directory of this project 

```
python fer_generator/generate_training_data.py -d ./data -fer fer_generator/fer2013.csv -ferplus fer_generator/fer2013new.csv
```

## Run Tensorboard
```
tensorboard --log_dir=./runs
```

## Methodology

#### Augmentations

#### Model architecture

#### Optimization


## Findings 
## Future recommendations
