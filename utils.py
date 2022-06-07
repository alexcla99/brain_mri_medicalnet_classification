from model import MedicalNetClassifier
from torch.utils.data import DataLoader
from torch import nn, optim

import torch
import numpy as np
import os, json

RESULTS_DIR = "results" # The directory where are stored results
TREESHOLD = .5          # Results binarization treeshold

def info(s:str, state:int=0) -> None:
    """Display an info / error / debug message."""
    assert state in [0, 1, 2]
    if state == 0:
        header = "[INFO]"
    elif state == 1:
        header = "[ERROR]"
    elif state == 2:
        header = "[DEBUG]"
    print("%s %s" % (header, s))

def load_params(path:str) -> dict:
    """Load the model parameters."""
    with open(path) as f:
        params = json.load(f)
    f.close()
    return params

def save_metrics(metrics:str, subdir:str) -> None:
    """Save the metrics buffer in the results folder."""
    with open(os.path.join(RESULTS_DIR, subdir, "metrics.csv"), "w+") as f:
        f.write(metrics)
    f.close()

def compute_f_score(predictions:torch.Tensor, targets:torch.Tensor) -> float:
    """Compute the f-score of the model for an epoch."""
    tps = 0.
    fps = 0.
    fns = 0.
    for p, t in zip(predictions, targets):
        if (p and t) == 1.:
            tps += 1.
        elif p == 1. and t == 0.:
            fps += 1.
        elif p == 0. and t == 1.:
            fns += 1.
    precision = tps / (tps + fps) if (tps + fps) != 0. else 0.
    recall = tps / (tps + fns) if (tps + fns) != 0. else 0.
    f_score = 2. * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.
    return f_score

def compute_acc(predictions:torch.Tensor, targets:torch.Tensor) -> float:
    """Compute the accuracy of the model for an epoch."""
    targets = torch.squeeze(targets)
    predictions = torch.squeeze(predictions)
    assert(len(targets) == len(predictions))
    predictions = (predictions >= TREESHOLD).float()
    #return compute_f_score(predictions, targets)
    return torch.sum(predictions == targets) / len(targets)

def validation(model:MedicalNetClassifier, debug:bool, val_loader:DataLoader,
    loss_function:nn.BCELoss, batch_size:int) -> (float, float):
    """Validate the model and return its computed metrics."""
    model.eval()
    epoch_predictions = list()
    epoch_targets = list()
    # running_loss = 0.
    with torch.no_grad():
        for data in val_loader:
            volume = data[0]
            target = data[1].float()
            if debug == False:
                volume = volume.cuda()
                target = target.cuda()
            output = model(volume)
            if output.shape[0] > 1:
                output = torch.squeeze(output)
            else:
                output = output.resize_(1)
            epoch_predictions.append(output)
            epoch_targets.append(target)
            # loss = loss_function(output, target)
            # running_loss += loss.item() * volume.size(0)
    epoch_predictions = torch.cat(epoch_predictions)
    epoch_targets = torch.cat(epoch_targets)
    acc = compute_acc(epoch_predictions, epoch_targets)
    loss = loss_function(epoch_predictions, epoch_targets)
    # loss = running_loss / val_loader.__len__()
    return acc, loss

class SaveBestModel:
    """Class to save the best model while training."""

    def __init__(self, best_val_loss:float) -> None:
        """Instanciate the saver."""
        self.best_val_loss = best_val_loss

    def __call__(self, curr_val_loss:float, model_dict:dict, subdir:str):
        """Save the best model's weights."""
        if curr_val_loss < self.best_val_loss:
            self.best_val_loss = curr_val_loss
            torch.save(model_dict, os.path.join(RESULTS_DIR, subdir, "best_model.pth"))
            info("Best model saved")