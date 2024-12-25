# Deep Learning Project

## Authors:

* Mykhailo Kruts
* Yevhen Zinenko
* Oleksandr Shmelov

## Description

Dataset used for this project:
- FER2013+

The main difference between FER2013 and FER2013+ lies in the labeling quality and data refinement

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