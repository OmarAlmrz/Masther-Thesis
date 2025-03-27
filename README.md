
# Enhancing Smartwatch Application for Swimming through Transfer Learning and User Customization

## Overview
This repository contains the implementation of a smartwatch application using transfer learning to enhance swimming activity recognition. It includes pre-trained models, fine-tuning scripts, datasets, and source code for training and evaluation. Additionally, it provides documentation for setup, usage, and further model optimization.

## Project Structure
The repository is structured as follows:

## Installation
Steps to install dependencies and set up the project:
```sh
# Clone the repository
git clone https://github.com/OmarAlmrz/Masther-Thesis.git
cd Masther-Thesis

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
Master-Thesis/
│   README.md           
│   .gitignore           
│   model.txt            # Architecture of the model      
│   requirements.txt     # Dependencies 
│
├── code/                           # Source code directory
│   ├── freezing.py                 # Re-train using freezing
│   ├── fine_tuning.py              # Re-train using fine tuning
│   ├── progressive_learning.py     # Re-train using progressive learning
|   ├── progressive_learning_ml.py  # Re-train using progressive learning + ml classifiers
│   ├── labeling/                   # Scripts to add labels to data
│   ├── loading_models/             # Scripts to use re-trained models
│
├── data/                # Datasets 
│
├── results/             # Folder to store metrics of the training           
│
└── save_path/           # Folder to store trained models
```

## Labeling
For the labeling, we employed an open-source tool called Label Studio.
https://labelstud.io/

## Data
Raw and processed data can be found in https://tubcloud.tu-berlin.de/apps/files/files?dir=/Shared/smartwatch-projects/project6_omar
