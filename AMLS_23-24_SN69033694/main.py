import sys
import argparse
from A.taskA import load_dataset_npz as taskA_load_dataset
from B.taskB import load_dataset_npz as taskB_load_dataset
from A.taskA import load_model_and_evaluate as taskA_load_model_and_evaluate
from B.taskB import load_model_and_evaluate as taskB_load_model_and_evaluate
# from A.taskA import load_model_and_evaluate as taskA_load_model_and_evaluate


# Parse arguments
parser = argparse.ArgumentParser(description='Loads model and evaluates it')
parser.add_argument('-t', '--type', help='type of model to load')
parser.add_argument('-f', '--file', help='Path to testing file')
parser.add_argument('-m', '--mednmist', action='store_true', help='load the files from mednmist dataset and evaluate')


args = parser.parse_args()

if (args.type in ['a', 'A']) and (args.file is not None):
    print('Loading model for task A')
    taskA_load_dataset(args.file)

if (args.type in ['a', 'A']) and (args.mednmist):
    print('Loading test dataset from mednmist website')
    taskA_load_model_and_evaluate()


if (args.type in ['b', 'B']) and (args.file is not None):
    print('Loading model for task A')
    taskB_load_dataset(args.file)

if (args.type in ['b', 'B']) and (args.mednmist):
    print('Loading test dataset from mednmist website')
    taskB_load_model_and_evaluate()



    