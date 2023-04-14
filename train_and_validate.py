import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

def eval_metrics(ground_truth, predictions):
    f1_scores = f1_score(ground_truth, predictions, average=None)
    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "f1": f1_scores,
        "average f1": np.mean(f1_scores),
        "confusion matrix": confusion_matrix(ground_truth, predictions),
    }


def train_cnn(model, loader, optimizer, device, silent)
    model.train()
    ground_truth = []
    predictions = []
    losses = []
    report_interval = 100

    for i, data_batch in enumerate(loader):
        images = data_batch["images"].to(device, non_blocking=True)
        tag_idx = data_batch["tag_idx"].to(device, non_blocking=True)

        # TODO: 
        # 1. Perform the forward pass to calculate the model's output. Save it to the variable "logits".
        # 2. Calculate the loss using the output and the ground truth tags. Save it to the variable "loss".
        # 3. Perform the backward pass to calculate the gradient.
        # 4. Use the optimizer to update model parameters.
        # Caveat: You may need to call optimizer.zero_grad(). Figure out what it does!
        # START HERE
        
        logits = model(images) #perform forward pass
        loss = F.cross_entropy(logits, tag_idx) #calculate loss
        loss.backward() #backward pass, calculate gradient
        optimizer.step() #update model params
        optimizer.zero_grad() #call zero_grad

        # END

        losses.append(loss.item())
        ground_truth.extend(tag_idx.tolist())
        predictions.extend(logits.argmax(dim=-1).tolist())

        if not silent and i > 0 and i % report_interval == 0:
            print(
                "\t[%06d/%06d] Loss: %f"
                % (i, len(loader), np.mean(losses[-report_interval:]))
            )

    return np.mean(losses), eval_metrics(ground_truth, predictions)

def validate_cnn(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, Dict[str, Any]]:
    """
    -Validate the CNN model
    Return values:
        1. the average validation loss
        2. validation metrics such as accuracy and F1 score
    """
    model.eval()
    ground_truth = []
    predictions = []
    losses = []

    with torch.no_grad():

        for data_batch in loader:
           images = data_batch["images"].to(device, non_blocking=True)
            tag_idx = data_batch["tag_idx"].to(device, non_blocking=True)

            # TODO: Similar to what you did in train_ffnn, but only step 1 and 2.
            # START HERE
            logits = model(images) #perform forward pass
            loss = F.cross_entropy(logits, tag_idx) #calculate loss;
            # END

            losses.append(loss.item())
            ground_truth.extend(tag_idx.tolist())
            predictions.extend(logits.argmax(dim=-1).tolist())

    return np.mean(losses), eval_metrics(ground_truth, predictions)


def train_val_loop_cnn(hyperparams: Dict[str, Any], model_type= String) -> None:
    """
    Train and validate the CNN model for a number of epochs.
    """

    # Create the model

    if model_type = "Zhang":
        model = zhang-model-CNN(train.images, train.tags)
    else: 
        model = wang-model-CNN(train.images, train.tags)

    device = get_device()
    model.to(device)
    print(model)
    # Create the optimizer
    optimizer = optim.RMSprop(
        model.parameters(), hyperparams["learning_rate"], weight_decay=hyperparams["l2"]
    )

    # Train and validate
    for i in range(hyperparams["num_epochs"]):
        print("Epoch #%d" % i)

        print("Training..")
        loss_train, metrics_train = train_cnn(
            model, loader_train, optimizer, device, silent=True
        )
        print("Training loss: ", loss_train)
        print("Training metrics:")
        for k, v in metrics_train.items():
            print("\t", k, ": ", v)

        print("Validating..")
        loss_val, metrics_val = validate_cnn(model, loader_val, device)
        print("Validation loss: ", loss_val)
        print("Validation metrics:")
        for k, v in metrics_val.items():
            print("\t", k, ": ", v)

    print("Done!")
