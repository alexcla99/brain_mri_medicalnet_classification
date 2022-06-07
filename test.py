"""
Author: alexcla99
Version: 1.0.0
Title: Test implementation of a fine-tuned CNN for brain MRIs classification
"""

from utils import info, load_params
from dataset import AnoxiaDataset

from torch.utils.data import DataLoader
from distutils.util import strtobool
import numpy as np
import torch
import sys, traceback, os

RESULTS_DIR = "results" # The folder where results are stored

if __name__ == "__main__":
    """Main program for the test task."""
    if len(sys.argv) != 3 or not os.path.isdir(os.path.join(RESULTS_DIR, "r%s" % str(sys.argv[1].split("_")[-1]))):
        print("Usage: python3 test.py resnet_<layers:int> <debug:bool>")
        print("Example: python3 test.py resnet_101 False")
        print("Note: the layers specified should match the subdirectory 'results/r<layers>'")
        print("      such as 'results/r101' if you selected the 'resnet_101' model")
    else:
        subdir = "r%s" % str(sys.argv[1].split("_")[-1])
        debug = strtobool(str(sys.argv[2]))
        try:
            # Loading the test dataset
            info("Loading the test dataset")
            test_dataset = AnoxiaDataset("test")
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=True
            )
            info("Using %d test samples" % test_dataset.__len__())
            # Loading the trained model
            info("Loading the trained model")
            model = torch.load(
                os.path.join(RESULTS_DIR, subdir, "best_model.pth"),
                map_location=torch.device('cpu')
            )
            model.eval()
            info("Using %s" % sys.argv[1])
            if debug == True:
                info(str(model), state=2)
            # Defining structures to store results
            targets = list()
            predictions = list()
            # Testing the model
            info("Testing the model")
            for batch_id, batch_data in enumerate(test_loader):
                # Loading data
                volume, target = batch_data
                # Making prediction
                prediction = model(volume)
                # Appending results
                targets.append(target)
                predictions.append(prediction)
            # Saving results
            if debug == true:
                info("Targets:\n%s" % str(targets), state=2)
                info("Predictions:\n%s" % str(predictions), state=2)
            np.save(targets, os.path.join(RESULTS_DIR, subdir, "targets.npy"))
            np.save(predictions, os.path.join(RESULTS_DIR, subdir, "predictions.npy"))
            # Finishing testing
            info("Testing done")
        except:
            info(traceback.format_exc(), state=1)
