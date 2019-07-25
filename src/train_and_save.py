from collections import OrderedDict
import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

def train(model, features, adj, idx_train, labels, epoch, optimizer):
    # t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))

def test(model, features, adj, idx_test, labels):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))
    return OrderedDict({"test_loss": loss_test, "test_acc" : acc_test})


def run():
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # replacing argparse with defaults
    epochs = 200
    lr = 0.01
    weight_decay = 5e-4
    hidden = 16
    dropout = 0.5

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(model, features, adj, idx_train, labels, epoch, optimizer)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    metrics = test(model, features, adj, idx_test, labels)
    params = model.state_dict()
    for k in params:
        params[k] = params[k].reshape(-1).data.tolist()
    # params = OrderedDict({k: v.reshape(-1).data.tolist() for k,v in params})
    final_results = {**params, **metrics}

    # Cache to file
    # if the file exists, simply append:
    records_filename = "records.csv"
    if os.path.exists(records_filename):
        with open(records_filename, "a") as target:
            fieldnames = list(final_results.keys())
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writerow(final_results)
    else:
        with open(records_filename, "w") as target:
            fieldnames = list(final_results.keys())
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(final_results)
run()
#
# def main():
#     run()

# if __name__ == "__main__":
#     main()
