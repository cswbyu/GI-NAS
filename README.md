# GI-NAS: Boosting Gradient Inversion Attacks through Adaptive Neural Architecture Search

This repository provides the official PyTorch implementation of our work *"GI-NAS: Boosting Gradient Inversion Attacks through Adaptive Neural Architecture Search"*, which has been accepted for publication at **IEEE Transactions on Information Forensics & Security 2025**.

If you have any concerns when using this repository, feel free to contact me (Email: wenbo.research@gmail.com).

## Pipeline

![pipeline](./assets/pipeline.png)

## Visualization

![visualization](./assets/visualization.png)

## Setup

Install the conda environment and activate it as follows:

```
conda env create -f environment.yml
conda activate gi_nas
```

## Attack & Evaluation

Change the configurations in `runcifar10.py` and `runimagenet.py`, and then run:

```
python runcifar10.py
python runimagenet.py
```