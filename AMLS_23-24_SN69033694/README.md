# mednmist_ucl

This repository contains the code for the UCL Applied Machine Learning Systems final report.

## Files
- `main.py`: Main moduel, loads functions in A/taskA.py and B/taskB.py and the .h5 files with the pre-trained parameters for each model.

- `A/taskA.py`: Functions needed to load and evaluate model for task A

- `A/best_model.h5`: Trained parameters for TaskA

- `A/taskA.ipynb`: This file contains the data exploration, training of different models, cross validation, early stop exercise, and matplotlib graphs for TaskA.

- `B/taskB.py`: Functions needed to load and evaluate model for task B

- `B/optimal_model2.h5`: Trained parameters for TaskB

- `B/taskB.ipynb`: This file contains the data exploration, training of different models, cross validation, early stop exercise, and matplotlib graphs for TaskB

## Usage
Here are the available flags and their descriptions:

- `-h, --help`: Display the help message and exit.
- `-t, --type`: Followed by either 'A' or 'B'
- `-f, --file`: Load npz file with the dataset. The dataset must have a 'test_images' and 'test_labels' keys
- `-m, --mednmist`: Download files from medmnist, extract the Test dataset and evaluate

To run the code use '-t' with arguments 'A' or 'B', then either '-f' followed by a path to a .npz file or '-m' to download from MedNMIST websitee

## Examples:
### Run task A with dataset stored in an npz file locally 
python main.py -t A -f Datasets/pneumoniamnist.npz

### Run task B with data from MedNMIST
python main.py -t B -m 


## Contact
ramiro.herrera16@ucl.ac.uk


