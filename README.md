# Deep Learning Project

## Authors:

* Mykhailo Kruts
* Yevhen Zinenko
* Oleksandr Shmelov

## Description

<<<<<<< HEAD
=======
Dataset used for this project:
- FER2013+

The FER+ annotations provide a set of new labels for the standard Emotion FER dataset. In FER+, each image has been labeled by 10 crowd-sourced taggers, which provide better quality ground truth for still image emotion than the original FER labels. Having 10 taggers for each image enables researchers to estimate an emotion probability distribution per face. This allows constructing algorithms that produce statistical distributions or multi-label outputs instead of the conventional single-label output,

![image](https://raw.githubusercontent.com/Microsoft/FERPlus/master/FER+vsFER.png)
>>>>>>> 28f7d3fbf23b88a2b6c7315f635a8869b99060e0

## Pre-requisites

* Python v3.10
* Git LFS

## Set-up

Set-up the environment

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 🚨 Important Notice
Please pay attention to this crucial information!  

Git utilizes a separate library `LFS` to handle loading and tracking large file > 100MB.  
File `fer2013.csv` which contains original images of FER dataset has a size ~200MB.

Command to check if Git LFS exists:
```
git lfs --version
```

If LFS is not present run these:
```
git lfs install 
git lfs pull
```

## Generate Dataset

Dataset used for this project:
- FER2013+

The main difference between FER2013 and FER2013+ lies in the labeling quality and data refinement

Before proceeding check again if Git **LFS** is present.
Assuming you are in the current directory of this project, run this command:

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
