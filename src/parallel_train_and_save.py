from collections import OrderedDict
import os
import csv
import time
import threading # for parallelism
import queue # for parallelism

import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, timing
from models import GCN

from train_and_save import train, test, write_results_to_file

class TestAndWriteThread(threading.Thread):
    def __init__(self, queue, features, adj, idx_test, labels,
                 records_filename="records.csv"):
        threading.Thread.__init__(self)
        self.queue = queue
        self.features = features
        self.adj = adj
        self.idx_test = idx_test
        self.labels = labels
        self.records_filename = records_filename

    def test_model(self):
        model = self.queue.get()
        if model is None:
            return None
        # Testing
        metrics = test(model, self.features,
                       self.adj, self.idx_test, self.labels)
        params = model.state_dict()
        for k in params:
            params[k] = params[k].reshape(-1).data.tolist()
        final_results = {**params, **metrics}
        return final_results

    def write_results(self, final_results):
        write_results_to_file(self.records_filename, final_results)
        self.queue.task_done()

    def run(self):
        while True:
            final_results = self.test_model()
            if final_results is None:
                break
            self.write_results(final_results)


class TrainingThread(threading.Thread):
    def __init__(self, out_queue, features, adj, idx_train, labels, max_trials):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5
        self.features = features
        self.adj = adj
        self.idx_train = idx_train
        self.labels = labels
        self.MAX_EXPERIMENTS = 5

    def train(self):
        # Model and optimizer
        model = GCN(nfeat=self.features.shape[1],
                    nhid=self.hidden,
                    nclass=self.labels.max().item() + 1,
                    dropout=self.dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        # Train model
        for _ in range(self.epochs):
            train(model, self.features, self.adj,
                  self.idx_train, self.labels, optimizer)
        return model

    def run(self):
        for i in range(self.MAX_EXPERIMENTS):
            model = self.train()
            self.out_queue.put(model)
            print(f'Experiment {i+1} completed')


def main():
    start_time = time.time()
    # Load data
    adj, features, labels, idx_train, idx_test = load_data()
    buffer = queue.Queue()
    print("Spawning TrainingThreads")
    # spawn threads to begin training
    n_workers = 3
    n_trials_per_worker = 5
    training_threads = []
    for i in range(n_workers):
        t = TrainingThread(buffer, features, adj, idx_train, labels,
                           n_trials_per_worker)
        t.setDaemon(True)
        t.start()
        training_threads.append(t)

    print("Spawning TestAndWriteThread")
    # create a single writer thread
    test_and_write = TestAndWriteThread(buffer, features, adj, idx_test, labels)
    test_and_write.setDaemon(True)
    test_and_write.start()

    # print("Waiting until buffer is complete")
    # buffer.join()

    # print("Stopping training workers")
    # for t in training_threads:
    #     t.join()

    print("Joining on test_and_write")
    test_and_write.join()
    end_time = time.time()
    print("Done.")
    tasks_completed = n_workers * n_trials_per_worker
    elapsed_time = end_time - start_time
    print(f'{tasks_completed} in {round(elapsed_time, 3)} seconds.')
    print(f'Task speed: {round(tasks_completed / elapsed_time, 3)} tasks/second.')

if __name__ == "__main__":
    main()
