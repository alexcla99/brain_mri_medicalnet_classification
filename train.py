"""
Author: alexcla99
Version: 1.0.0
Title: Train implementation of a fine-tuned CNN for brain MRIs classification
"""

from dataset import AnoxiaDataset
from utils import SaveBestModel, info, load_params, save_metrics, compute_acc, validation
from model import MedicalNetClassifier, generate_model

from torch.utils.data import DataLoader
from torch import nn, optim
from distutils.util import strtobool
import numpy as np
import torch
import sys, traceback, os

LAYERS = [10, 18, 34, 50, 101, 152, 200] # Supported layers
MODELS_DIR = "models"                    # The directory where are stored models
LAST_LOSS = 100.                         # Random value to be replaced during the first epoch

if __name__ == "__main__":
    """Main program for the train task."""
    if len(sys.argv) != 3 or int(sys.argv[1].split("_")[-1]) not in LAYERS:
        print("Usage: python3 train.py resnet_<layers:int> <debug:bool>")
        print("Available networks:" + "".join(["\n* resnet_" + str(e) for e in LAYERS]))
        print("Example: python3 train.py resnet_50 False")
    else:
        subdir = "r%s" % str(sys.argv[1].split("_")[-1])
        debug = strtobool(str(sys.argv[2]))
        try:
            # Loading parameters
            info("Loading parameters")
            params = load_params("settings.json")
            model_params = params["model"]
            other_params = params["debug"] if debug == True else params["default"]
            # Loading train / val datasets
            info("Loading train / val datasets")
            assert sum(model_params["data_rep"]) == 1.
            train_dataset = AnoxiaDataset("train")
            train_loader = DataLoader(
                train_dataset,
                batch_size=other_params["batch_size"],
                shuffle=True,
                num_workers=other_params["num_workers"],
                pin_memory=debug
            )
            info("Using %d training samples" % train_dataset.__len__())
            val_dataset = AnoxiaDataset("val")
            val_loader = DataLoader(
                val_dataset,
                batch_size=other_params["batch_size"],
                shuffle=True,
                num_workers=other_params["num_workers"],
                pin_memory=debug
            )
            info("Using %d validation samples" % val_dataset.__len__())
            # Loading model checkpoints
            info("Loading model checkpoints")
            base_checkpoint = torch.load(
                os.path.join(MODELS_DIR, "%s.pth" % sys.argv[1]),
                map_location=torch.device('cpu') if debug == True else None
            )["state_dict"]
            checkpoint = dict()
            for k in base_checkpoint.keys():
                checkpoint[".".join(k.split(".")[1:])] = base_checkpoint[k]
            # Loading the pretrained model
            info("Loading the pretrained model")
            model = generate_model(
                int(sys.argv[1].split("_")[-1]),
                other_params["batch_size"],
                model_params,
                checkpoint,
                debug
            )
            info("Using %s" % sys.argv[1])
            if debug == True:
                info(str(model), state=2)
            # Setting the optimizer
            info("Setting the optimizer")
            optimizer = optim.SGD(
                model.parameters(),
                lr=model_params["lr"],
                momentum=model_params["momentum"],
                weight_decay=model_params["weight_decay"]
            ) # TODO Adam?
            # Defining checkpoints
            info("Defining checkpoints")
            # lr_reducer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            lr_reducer = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)
            save_best_model = SaveBestModel(LAST_LOSS)
            # Training the model
            info("Training the model")
            info("Using %s epoch(s) in total with %s batch(es) per epoch" % (
                other_params["epochs"],
                other_params["batch_size"]
            ))
            model.train()
            # Defining the loss function
            loss_classify = nn.BCELoss()
            if debug == False:
                loss_classify = loss_classify.cuda()
            last_loss = LAST_LOSS
            # Defining variables for the early stopping
            patience = other_params["patience"]
            trigger_times = 0
            # Defining a buffer to save metrics
            b_metrics = "train_acc, train_loss, val_acc, val_loss"
            # Looping over epochs
            for epoch in range(other_params["epochs"]):
                running_loss = 0.
                # Defining sructures to store predictions and targets 
                epoch_predictions = list()
                epoch_targets = list()
                info("Epoch %s/%s" % (epoch + 1, other_params["epochs"]))
                # Looping over batches
                for batch_id, batch_data in enumerate(train_loader):
                    # Loading batch data
                    volumes, targets = batch_data
                    if debug == False:
                        volumes = volumes.cuda()
                    optimizer.zero_grad()
                    # Making predictions
                    predictions = model(volumes)
                    # Gathering corresponding targets
                    new_targets = torch.Tensor(np.expand_dims(targets, axis=-1))
                    if debug == False:
                        new_targets = new_targets.cuda()
                    epoch_predictions.append(torch.squeeze(predictions))
                    epoch_targets.append(torch.squeeze(new_targets))
                    # Computing loss for this batch
                    loss = loss_classify(predictions, new_targets)
                    # Backwarding loss to the model and the lr reducer
                    loss.backward()
                    optimizer.step()
                    lr_reducer.step()#loss)
                    running_loss += loss.item() * volumes.size(0)
                # Computing accuracy and loss for this epoch
                epoch_predictions = torch.cat(epoch_predictions)
                epoch_targets = torch.cat(epoch_targets)
                train_acc = compute_acc(epoch_predictions, epoch_targets)
                # train_loss = loss_classify(epoch_predictions, epoch_targets)
                train_loss = running_loss / train_dataset.__len__()
                info("Train acc = %.8f, loss = %.8f" % (train_acc, train_loss))
                # Computing validation accuracy and loss for this epoch
                val_acc, val_loss = validation(
                    model,
                    debug,
                    val_loader,
                    loss_classify,
                    other_params["batch_size"]
                )
                # Saving metrics in the buffer
                b_metrics += ("\n%.8f, %.8f, %.8f, %.8f" % (
                    train_acc,
                    train_loss,
                    val_acc,
                    val_loss
                ))
                # Computing early stopping
                if val_loss > last_loss:
                    trigger_times += 1
                    if trigger_times >= patience:
                        info("Early stopping")
                        quit()
                else:
                    trigger_times = 0
                last_loss = val_loss
                # Saving the best model's weights
                save_best_model(val_loss, model.state_dict(), subdir)
            # Saving train / val metrics
            save_metrics(b_metrics, subdir)
            info("Metrics saved")
            # Finishing training
            info("Training done")
        except:
            info(traceback.format_exc(), state=1)