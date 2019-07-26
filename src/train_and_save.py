from collections import OrderedDict
import os
import csv

import time
import torch.nn.functional as F

import torch.optim as optim

from utils import load_data, accuracy, timing
from models import GCN
# import pandas as pd


@timing
def train(model, features, adj, idx_train, labels, optimizer):
    """
        Trains model for one epoch
    """
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

@timing
def test(model, features, adj, idx_test, labels):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test]).data.tolist()
    acc_test = accuracy(output[idx_test], labels[idx_test]).data.tolist()
    return OrderedDict({"test_loss": loss_test, "test_acc" : acc_test})

@timing
def write_results_to_file(records_filename, final_results):
    if os.path.exists(records_filename):
        with open(records_filename, "a") as target:
            fieldnames = list(final_results.keys())
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writerow(final_results)
    else:
        print(f"{records_filename} has been created")
        with open(records_filename, "w") as target:
            fieldnames = list(final_results.keys())
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(final_results)

def report_progress(i, start_time, done_training_time, done_testing_time, end_time):
        # Print Timing
        elapsed_time_seconds_3digits = round(end_time - start_time, 3)
        training_time_seconds_3digits = round(done_training_time - start_time,
                                              3)
        testing_time_seconds_3digits = round(done_testing_time -
                                             done_training_time, 3)
        print(f'Trial {i+1} completed in {elapsed_time_seconds_3digits}s')
        print(f"Training = {training_time_seconds_3digits}s")
        print(f"Testing = {testing_time_seconds_3digits}s")
        print(f"Writing Time = {round(end_time - done_testing_time, 3)} s")


def run(n_trials=100):
    # Load data
    adj, features, labels, idx_train, idx_test = load_data()

    epochs = 200
    lr = 0.01
    weight_decay = 5e-4
    hidden = 16
    dropout = 0.5
    for i in range(n_trials):
        start_time = time.time()
        # Model and optimizer
        model = GCN(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=labels.max().item() + 1,
                    dropout=dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, weight_decay=weight_decay)
        # Train model
        for _ in range(epochs):
            train(model, features, adj, idx_train, labels, optimizer)
        done_training_time = time.time()

        # Testing
        metrics = test(model, features, adj, idx_test, labels)
        params = model.state_dict()
        for k in params:
            params[k] = params[k].reshape(-1).data.tolist()
        final_results = {**params, **metrics}
        done_testing_time = time.time()

        # Cache to file
        records_filename = "records.csv"
        write_results_to_file(records_filename, final_results)
        end_time = time.time()
        report_progress(i, start_time, done_training_time,
                          done_testing_time, end_time)


def main():
    run()

if __name__ == "__main__":
    main()
